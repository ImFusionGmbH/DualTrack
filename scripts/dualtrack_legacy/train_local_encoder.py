import argparse
from functools import partial
import logging
import os
from pathlib import Path
import pathlib
import sys
from typing import Callable, List, Literal, Optional
import torch
from src.models import model_registry
from src.utils.utils import UnstructuredArgsAction, load_model_weights
from src.logger import get_logger, get_default_log_dir
from src.optimizer import setup_optimizer
from src.engine.loops import (
    export_features as _export_features,
    run_full_test_loop,
    run_training,
)
import src.models.local_encoder  # need this import to register the local encoder models
from src.datasets.loader_factory.local_encoder_pretraining import (
    LoaderArgs,
    get_loaders as _get_loaders,
)


TRACKED_METRIC = "ddf/5pt-avg_global_displacement_error"
MODEL_REGISTRY = {}


def get_parser():
    parser = argparse.ArgumentParser(description="Train a local encoder model")
    parser.add_argument("--log_dir", default=get_default_log_dir())
    parser.add_argument("--logger", default="wandb")
    parser.add_argument("--dataset", default="tus-rec")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_dataloader_workers", default=4, type=int)
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--augmentations", action="store_true")
    parser.add_argument("--sequence_length_train", default=None, type=int)
    parser.add_argument("--run_validation_every_n_epochs", default=1, type=int)
    parser.add_argument(
        "--cached_features_file",
        help="Some models support caching intermediates \
        then training on those. If using one of these models, provide the filepath \
        here where the cached features can be found. If they dont exist, use `export_features` to create them.",
    )
    class DebugAction(argparse._StoreTrueAction):
        def __call__(self, parser, namespace, *args, **kwargs): 
            super().__call__(parser, namespace, *args, **kwargs)
            if namespace.debug: 
                namespace.num_dataloader_workers = 0 
                namespace.log_dir += '_debug'

    parser.add_argument("--debug", action=DebugAction, help='Set debug mode')

    group = parser.add_argument_group("model")
    group.add_argument("--model", default="dualtrack_loc_enc_stg1")
    group.add_argument(
        "--backbone_weights",
        default=None,
        help="Path to the weights of the previous stage",
    )
    group.add_argument(
        '--model_kw', action=UnstructuredArgsAction, help='Additional key=value settings to pass to model constructor.'
    )

    # === Subparsers ===
    subparsers = parser.add_subparsers(dest="subcommand")
    subparser = subparsers.add_parser("test", help="test model")
    subparser.add_argument("--train_dir")
    subparser.add_argument("--model_weights")
    subparser.add_argument("--output_dir")
    subparser.add_argument("--dataset", required=True)

    subparsers.add_parser("export_features", help='export features of the model')

    return parser


def train(args):
    logger = get_logger(args.logger, args.log_dir, args, disable_checkpoint=args.debug)
    state = logger.get_checkpoint("last.pt")

    torch.manual_seed(args.seed)
    model = get_model(args)
    if state:
        logging.info(f"Loading model weights from checkpoint")
        model.load_state_dict(state["model"])

    model.to(args.device)
    logging.info(
        f"Model: {model.__class__}, {sum(p.numel() for p in model.parameters())/1e6} Million params"
    )

    train_loader, val_loader = get_loaders(args)

    optimizer, scheduler = setup_optimizer(
        model,
        lr=args.lr,
        scheduler_name="cosine",
        weight_decay=args.weight_decay,
        total_epochs=args.epochs,
        num_steps_per_epoch=len(train_loader),
        warmup_steps=0,
        state=state,
    )
    scaler = torch.GradScaler(enabled=args.use_amp)
    if state:
        scaler.load_state_dict(state["scaler"])

    best_metric = state["best_score"] if state else float("inf")
    start_epoch = state["epoch"] if state else 0

    run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        logger,
        args.epochs,
        scaler,
        get_prediction,
        get_loss,
        best_metric,
        start_epoch,
        use_amp=args.use_amp,
        validation_mode="full",
        validate_every_n_epochs=args.run_validation_every_n_epochs,
        device=args.device
    )


def test(args):
    model_weights = args.model_weights or os.path.join(
        args.train_dir, "checkpoint", "best.pt"
    )
    output_dir = args.output_dir or os.path.join(args.train_dir, "test", args.dataset)

    model = get_model(args).to(args.device)
    print(load_model_weights(model, model_weights))
    train_loader, val_loader = get_loaders(args)

    run_full_test_loop(
        model,
        val_loader,
        get_prediction,
        pathlib.Path(output_dir),
        args.device,
        args.use_amp,
    )


def export_features(args):
    output_file = args.cached_features_file

    model = get_model(args, features_only=True).to(args.device)
    #assert hasattr(model, "output_mode")
    #model.output_mode = "features"
    #model.input_mode = "images"

    loader_args = LoaderArgs(
        dataset=args.dataset,
        sequence_length_train=None,
        batch_size=1,
        num_dataloader_workers=args.num_dataloader_workers,
        resize_to=None,
        random_crop=False,
        random_horizontal_flip=False,
        random_reverse_sweep=False,
        validation_mode="full",
    )
    train_loader, val_loader = _get_loaders(loader_args)

    batch = next(iter(train_loader))
    features_preview = model(batch["images"].to(args.device))
    print(f"Features preview {features_preview.shape}")

    _export_features(
        model,
        train_loader,
        output_file,
        args.device,
    )
    _export_features(model, val_loader, output_file, args.device)


def get_loaders(args):
    is_test = args.subcommand == "test" or args.subcommand == "export_features"
    use_augmentations = args.augmentations and not is_test

    if args.cached_features_file is None:
        loader_args = LoaderArgs(
            dataset=args.dataset,
            sequence_length_train=args.sequence_length_train,
            batch_size=args.batch_size,
            num_dataloader_workers=args.num_dataloader_workers,
            resize_to=None,
            random_crop=use_augmentations,
            random_horizontal_flip=use_augmentations,
            random_reverse_sweep=use_augmentations,
            validation_mode="full",
            drop_keys=['images_downsampled-224']
        )
    else:
        loader_args = LoaderArgs(
            dataset=args.dataset,
            sequence_length_train=args.sequence_length_train,
            batch_size=args.batch_size,
            num_dataloader_workers=args.num_dataloader_workers,
            resize_to=None,
            random_reverse_sweep=use_augmentations,
            validation_mode="full",
            drop_keys=["images", "images_downsampled-224"],
            cached_features_map={"image_features": args.cached_features_file},
        )

    return _get_loaders(loader_args, debug=args.debug)


def get_model(args, **kwargs):
    model_kw = dict(
        name=args.model, 
        **args.model_kw
    )
    if args.backbone_weights: 
        model_kw['backbone_weights'] = args.backbone_weights
    model_kw.update(kwargs)

    return model_registry.get_model(**model_kw)


def get_loss(batch, model, device, pred=None):
    if pred is None:
        outputs = get_prediction(batch, model, device)
    else:
        outputs = pred
    targets = batch["targets"].to(device)
    loss = torch.nn.functional.mse_loss(outputs, targets, reduction="none").mean()
    return loss


def get_prediction(batch, model, device):
    if "images" in batch:
        images = batch["images"].to(device)
        B, C, N, H, W = images.shape
        # targets = batch["targets"].to(args.device)
        outputs = model(images)
    else:
        outputs = model(batch["image_features"].to(device))
    return outputs


def main():
    args = get_parser().parse_args()
    if args.subcommand == "test":
        test(args)
    elif args.subcommand == "export_features":
        export_features(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
