from argparse import ArgumentParser
from contextlib import contextmanager
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
from src.datasets.sweeps_dataset import SweepsDatasetWithAdditionalCachedData
from src.engine.loops import run_full_evaluation_loop, run_full_test_loop, run_training, Callback
import torch
from torch import nn
from src.logger import Logger, get_default_log_dir
from src.optimizer import setup_optimizer
from src.models import model_registry
from src import transform as T
from src.models.fusion_model.fusion_model import dualtrack_fusion_model
from src.datasets.loader_factory.fusion_model_training import get_loaders
from src.models import get_model
import argparse


def get_parser():
    # fmt: off
    parser = argparse.ArgumentParser(description="Train a local encoder model")
    parser.set_defaults(name="pretrain_global_encoder")

    parser.add_argument("--log_dir", default=get_default_log_dir())
    parser.add_argument("--logger", default="wandb")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--epochs", default=800, type=int)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--sequence_length_train", default=None, type=int)
    parser.add_argument("--val_every", default=10, type=int)

    class DebugAction(argparse._StoreTrueAction):
        def __call__(self, parser, namespace, *args, **kwargs):
            super().__call__(parser, namespace, *args, **kwargs)
            if namespace.debug:
                namespace.num_workers = 0
                namespace.log_dir += "_debug"

    parser.add_argument("--debug", action=DebugAction, help="Set debug mode")

    group = parser.add_argument_group("model")
    group.add_argument("--model", default='dualtrack_custom_build')
    group.add_argument("--local_encoder_name", default='dualtrack_loc_enc_stg3', 
        choices=[model for model in model_registry.list_models() if 'loc_enc' in model])
    group.add_argument("--local_encoder_ckpt", )
    group.add_argument("--global_encoder_name", default='global_encoder_cnn', 
        choices=[model for model in model_registry.list_models() if 'global_encoder' in model])
    group.add_argument("--global_encoder_ckpt")

    group = parser.add_argument_group("data")
    group.add_argument("--dataset", default="tus-rec")
    group.add_argument("--batch_size", default=1, type=int)
    group.add_argument("--use_augmentations", action="store_true")
    group.add_argument("--mean", type=float, nargs="+", default=[0])
    group.add_argument("--std", type=float, nargs="+", default=[1])
    group.add_argument("--in_channels", type=int, default=1)
    group.add_argument("--num_workers", default=8, type=int)
    group.add_argument(
        "--load_preprocessed_from_disk",
        action="store_true",
        help="If you already ran the script to preprocess and save the \
            global encoder inputs, set this flag to speed up data loading.",
    )
    group.add_argument("--loc_encoder_intermediates_cache", help='If you already pre-computed local encoder \
        features, specify their path here.')

    # === Subparsers ===
    subparsers = parser.add_subparsers(dest="subcommand")
    subparser = subparsers.add_parser("test", help="test model")
    # subparser.add_argument("--train_dir")
    subparser.add_argument("--model_weights")
    subparser.add_argument("--output_dir", required=True)
    subparser.add_argument("--dataset", required=True)

    return parser
    # fmt: on


def train(args):

    print(json.dumps(vars(args), indent=4))

    logger = Logger.get_logger(
        args.logger, args.log_dir, args, disable_checkpoint=args.debug
    )
    state = logger.get_checkpoint()

    loader_kw = dict(
        dataset=args.dataset,
        global_encoder_preprocessing_kw=dict(
            in_channels=args.in_channels,
            mean=args.mean,
            std=args.std,
        ),
        use_augmentations=args.use_augmentations,
        load_preprocessed_images_from_disk=args.load_preprocessed_from_disk,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        features_paths=(
            {"local_encoder_intermediates": args.loc_encoder_intermediates_cache}
            if args.loc_encoder_intermediates_cache is not None
            else {}
        ),
    )

    train_loader, val_loader = get_loaders(**loader_kw)

    torch.manual_seed(args.seed)  # <- make model weights reproducible
    
    if args.model == 'dualtrack_custom_build': 
        model = dualtrack_fusion_model(
            local_encoder_cfg=dict(
                name=args.local_encoder_name,
                checkpoint=args.local_encoder_ckpt,
                features_only=True,
            ),
            global_encoder_cfg=dict(
                name=args.global_encoder_name,
                checkpoint=args.global_encoder_ckpt,
                features_only=True,
            ),
        ).to(args.device)
    else: 
        model = get_model(args.model).to(args.device)

    if state:
        model.load_state_dict(state["model"])

    def validate_cache():
        _, sanity_check_loader = get_loaders(
            include_local_encoder_images=True, **loader_kw
        )
        batch = next(iter(sanity_check_loader))
        model.validate(batch)

    if args.loc_encoder_intermediates_cache:
        validate_cache()

    optimizer, scheduler = setup_optimizer(
        model,
        scheduler_name="warmup_cosine",
        num_steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        weight_decay=args.weight_decay,
        lr=args.lr,
        state=state,
    )
    scaler = torch.GradScaler(device=args.device, enabled=args.use_amp)
    if state:
        scaler.load_state_dict(state["scaler"])

    best_score = state["best_score"] if state else float("inf")
    start_epoch = state["epoch"] if state else 0

    run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        logger,
        scaler=scaler,
        epochs=args.epochs,
        pred_fn=pred_fn,
        device=args.device,
        loss_fn=loss_fn,
        validate_every_n_epochs=args.val_every,
        validation_mode="full",
        use_amp=args.use_amp,
        best_score=best_score,
        start_epoch=start_epoch,
    )


def test(args):

    os.makedirs(args.output_dir, exist_ok=True)
    print(OmegaConf.create(vars(args)))
    OmegaConf.save(
        OmegaConf.create(vars(args)), os.path.join(args.output_dir, 'config.yaml')
    )

    train_loader, val_loader = get_loaders(
        dataset=args.dataset,
        global_encoder_preprocessing_kw=dict(
            in_channels=args.in_channels,
            mean=args.mean,
            std=args.std,
            resize_to=(224, 224),
        ),
        use_augmentations=args.use_augmentations,
        load_preprocessed_images_from_disk=args.load_preprocessed_from_disk,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        features_paths=(
            {"local_encoder_intermediates": args.loc_encoder_intermediates_cache}
            if args.loc_encoder_intermediates_cache is not None
            else {}
        ),
    )

    if args.model == 'dualtrack_custom_build': 
        model = dualtrack_fusion_model(
            local_encoder_cfg=dict(
                name=args.local_encoder_name,
                checkpoint=args.local_encoder_ckpt,
                features_only=True,
            ),
            global_encoder_cfg=dict(
                name=args.global_encoder_name,
                checkpoint=args.global_encoder_ckpt,
                features_only=True,
            ),
        ).to(args.device)
    else: 
        model = get_model(args.model, checkpoint=args.model_weights).to(args.device)

    model.to(args.device)
    model.eval()

    metrics = run_full_test_loop(
        model,
        val_loader,
        pred_fn,
        output_dir=Path(args.output_dir),
        device=args.device,
        use_amp=args.use_amp,
    )
    print(metrics)

    # if args.log_tests_wandb:
    #     import wandb


#
#     wandb.init(project="trackerless-ultrasound", config=args, job_type="test")
#     wandb.log({f"{k}/test": v for k, v in metrics.items()})


def pred_fn(batch, model, device):
    batch = model.forward_dict(batch)
    pred = batch["prediction"]

    # pred = model.predict(batch, device)
    return pred


def loss_fn(batch, model, device, pred=None):
    pred = pred if pred is not None else pred_fn(batch, model, device)

    targets = batch["targets"].to(device)
    padding_lengths = batch["padding_size"]

    mse_loss = nn.MSELoss(reduction="none")

    def _get_loss(pred, targets):
        B, N, D = (
            pred.shape
            if isinstance(pred, torch.Tensor)
            else list(pred.values())[0].shape
        )

        mask = torch.ones(B, N, D, dtype=torch.bool, device=device)
        for i, padding_length in enumerate(padding_lengths):
            if padding_length > 0:
                mask[i, -padding_length:, :] = 0

        loss = mse_loss(pred, targets)
        masked_loss = torch.where(mask, loss, torch.nan)
        mse_loss_val = masked_loss.nanmean()
        return mse_loss_val

    if isinstance(pred, dict):
        loss = torch.tensor(0.0, device=device)
        if "local" in pred:
            loss += _get_loss(pred["local"], targets)
        if "global" in pred:
            loss += _get_loss(pred["global"], batch["targets_global"].to(device))
        if "absolute" in pred:
            loss += _get_loss(pred["absolute"], batch["targets_absolute"].to(device))
        return loss
    else:
        return _get_loss(pred, targets)


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.subcommand == "test":
        test(args)
    else:
        train(args)
    # train(get_parser().parse_args())
