from argparse import ArgumentParser
import argparse
from dataclasses import dataclass
from src.engine.loops import run_training
import torch
from torch import nn
from src.logger import Logger, get_default_log_dir
from src.utils.cli_helper import add_config_path_args, parse_config
from src.optimizer import setup_optimizer
from src.models.model_registry import get_model
from src.datasets.loader_factory.global_encoder_pretraining import (
    get_loaders,
    LoaderConfig,
)
import src.models.global_encoder


def get_parser():
    parser = argparse.ArgumentParser(description="Train a local encoder model")
    parser.set_defaults(name="pretrain_global_encoder")

    parser.add_argument("--log_dir", default=get_default_log_dir())
    parser.add_argument("--logger", default="wandb")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dataset", default="tus-rec")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=800, type=int)
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
    group.add_argument("--model", default="global_encoder_cnn")
    group.add_argument(
        "--backbone_weights",
        help="If using a backbone that requires loaded weights, specify the path here.",
    )

    group = parser.add_argument_group("data")
    group.add_argument("--mean", type=float, nargs="+", default=[0])
    group.add_argument("--std", type=float, nargs="+", default=[1])
    group.add_argument("--in_channels", type=int, default=1)
    group.add_argument("--use_augmentations", action="store_true")
    group.add_argument("--num_workers", default=8, type=int)
    group.add_argument(
        "--load_preprocessed_from_disk",
        action="store_true",
        help="If you already ran the script to preprocess and save the \
            global encoder inputs, set this flag to speed up data loading.",
    )

    # === Subparsers ===
    subparsers = parser.add_subparsers(dest="subcommand")
    subparser = subparsers.add_parser("test", help="test model")
    subparser.add_argument("--train_dir")
    subparser.add_argument("--model_weights")
    subparser.add_argument("--output_dir")
    subparser.add_argument("--dataset", required=True)

    return parser


def train(args):

    logger = Logger.get_logger(
        "wandb", args.log_dir, args, disable_checkpoint=args.debug
    )
    state = logger.get_checkpoint()

    train_loader, val_loader = get_loaders(
        LoaderConfig(
            dataset=args.dataset,
            num_workers=args.num_workers,
            n_samples=64,
            in_channels=args.in_channels,
            use_augmentations=args.use_augmentations,
            mean=args.mean,
            std=args.std,
            load_preprocessed_from_disk=args.load_preprocessed_from_disk,
        ),
        debug=args.debug,
    )

    model = get_model(
        name=args.model, seed=args.seed, backbone_weights=args.backbone_weights
    ).to(args.device)
    if state:
        model.load_state_dict(state["model"])

    optimizer, scheduler = setup_optimizer(
        model,
        scheduler_name="warmup_cosine",
        num_steps_per_epoch=len(train_loader),
        warmup_epochs=10,
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
        loss_fn=Criterion(),
        validate_every_n_epochs=args.val_every,
        validation_mode="full",
        best_score=best_score,
        start_epoch=start_epoch,
        device=args.device,
    )


def pred_fn(batch, model, device):
    pred = model.predict(batch)
    return pred


@dataclass
class Criterion:

    base_loss: str = "mse"

    def __call__(self, batch, model, device, pred=None):
        pred = pred if pred is not None else pred_fn(batch, model, device)

        targets = batch["targets"].to(device)
        padding_lengths = batch["padding_size"]

        if self.base_loss == "mse":
            crit = nn.MSELoss(reduction="none")
        elif self.base_loss == "mae":
            crit = nn.L1Loss(reduction="none")
        else:
            raise NotImplementedError(self.base_loss)

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

            loss = crit(pred, targets)
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
                loss += _get_loss(
                    pred["absolute"], batch["targets_absolute"].to(device)
                )
            return loss
        else:
            return _get_loss(pred, targets)


# def get_parser():
#     parser = argparse.ArgumentParser(description="Pretrain the global encoder")
#     add_config_path_args(parser)
#     return parser


def main(args):
    train(args)


if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())
