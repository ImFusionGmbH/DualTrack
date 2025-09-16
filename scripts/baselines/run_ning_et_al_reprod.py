# method from paper
# "SPATIAL POSITION ESTIMATION METHOD FOR 3D ULTRASOUND
# RECONSTRUCTION BASED ON HYBRID TRANSFOMERS"
# Ning et. al


import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
import tensordict
import torch
from src.batch_collator import BatchCollator
from src.datasets import SweepsDatasetWithAdditionalCachedData
from src.engine.loops import (
    run_full_evaluation_loop,
    run_full_test_loop,
    run_training,
    Callback,
)
from src.models.bert import BertEncoder
from src.models.model_registry import register_model, get_model


from src.logger import Logger, get_default_log_dir
from src.optimizer import setup_optimizer
from src import transform as T
from torch import nn


def get_parser():
    parser = argparse.ArgumentParser()

    parser.set_defaults(name="ning-et-al-baseline")
    os.environ["WANDB_TAGS"] = "baseline"
    parser.add_argument("--log_dir", default=get_default_log_dir())
    parser.add_argument("--logger", default="wandb")
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dataset", default="tus-rec")
    parser.add_argument("--test_dataset", default="tus-rec-val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument(
        "--val_mode", type=str, choices=("loss", "full"), default="full"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--clip_grad_norm", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--warmup_epochs", type=int, default=10)

    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--n_features", type=int, default=512)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--subsequence_length_train", type=int)
    parser.add_argument("--features_path", )

    subparsers = parser.add_subparsers(dest="command")
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--train_dir")
    test_parser.add_argument("--log_wandb", action="store_true")
    test_parser.add_argument("--mode", choices=("basic", "full"), default="full")

    return parser


def train(cfg):
    print(json.dumps(vars(cfg), indent=4))
    logger = Logger.get_logger(
        cfg.logger, cfg.log_dir, cfg, disable_checkpoint=cfg.debug
    )

    logger = Logger.get_logger(
        cfg.logger, cfg.log_dir, cfg, disable_checkpoint=cfg.debug
    )
    state = logger.get_checkpoint()
    if not state and cfg.init_checkpoint:
        state = torch.load(cfg.init_checkpoint, map_location="cpu")

    train_loader, val_loader, test_loader = get_loaders(cfg)

    torch.manual_seed(cfg.seed)  # <- make model weights reproducible
    model = get_model_from_args(cfg).to(cfg.device)
    if state:
        model.load_state_dict(state["model"])

    optimizer, scheduler = setup_optimizer(
        model,
        scheduler_name="warmup_cosine",
        num_steps_per_epoch=len(train_loader),
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.epochs,
        weight_decay=cfg.wd,
        lr=cfg.lr,
        state=state,
    )
    scaler = torch.GradScaler(device=cfg.device, enabled=cfg.use_amp)
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
        epochs=cfg.epochs,
        pred_fn=pred_fn,
        device=cfg.device,
        loss_fn=loss_fn,
        validate_every_n_epochs=cfg.val_every,
        validation_mode=cfg.val_mode,
        use_amp=cfg.use_amp,
        best_score=best_score,
        start_epoch=start_epoch,
        clip_grad_norm=cfg.clip_grad_norm,
    )

    state = logger.get_checkpoint("best.pt")
    if state:
        model.load_state_dict(state["model"])

    run_full_evaluation_loop(
        model,
        test_loader,
        pred_logic=pred_fn,
        loss_fn=loss_fn,
        device=cfg.device,
        use_amp=cfg.use_amp,
        suffix="/test",
        logger=logger,
    )


def test(cfg):
    output_dir = os.path.join(cfg.train_dir, 'test', cfg.test_dataset)
    train_cfg = OmegaConf.load(os.path.join(cfg.train_dir, "config.yaml"))
    state = torch.load(os.path.join(cfg.train_dir, "checkpoint", "best.pt"))
    train_cfg.test_dataset = cfg.test_dataset
    train_loader, val_loader, test_loader = get_loaders(train_cfg)
    model = get_model_from_args(train_cfg).to(cfg.device)
    if state:
        model.load_state_dict(state["model"])

    if cfg.mode == "basic":
        metrics = run_full_evaluation_loop(
            model,
            test_loader,
            pred_logic=pred_fn,
            loss_fn=loss_fn,
            device=cfg.device,
            use_amp=cfg.use_amp,
        )
        print(metrics)
        if train_cfg.log_wandb:
            import wandb

            wandb.init(project="trackerless-ultrasound", job_type="test")
            wandb.log({f"{k}/test": v for k, v in metrics.items()})
    else:
        run_full_test_loop(
            model,
            test_loader,
            pred_logic=pred_fn,
            output_dir=Path(output_dir), 
            device=cfg.device,
            use_amp=cfg.use_amp,
        )


def pred_fn(batch, model, device):
    pred = model.predict(batch, device)
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


def get_model_from_args(cfg):
    using_cached_features = cfg.features_path is not None

    from src.models.local_encoder import vidrn18_small_window_trck_reg_causal

    if not using_cached_features:
        backbone = vidrn18_small_window_trck_reg_causal(features_only=True)
    else:
        backbone = None

    return NingEtAlModel(
        in_features=cfg.n_features, hidden_size=cfg.hidden_size, backbone=backbone
    )


class NingEtAlModel(nn.Module):
    def __init__(self, in_features, hidden_size=512, backbone=None):
        super().__init__()
        from transformers.models.bert.modeling_bert import BertConfig, BertEncoder

        self.backbone = backbone
        self.pos_emb = nn.Embedding(1024, hidden_size)
        self.proj = nn.Linear(in_features, hidden_size, bias=False)
        self.transformer = BertEncoder(
            BertConfig(
                attn_implementation="eager",
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 2,
                num_attention_heads=8,
            )
        )
        self.head = nn.Linear(hidden_size, 6)

    def forward(self, features=None, images=None):
        if features is None:
            assert images is not None
            assert self.backbone is not None
            features = self.backbone(images)

        B, N, D = features.shape
        x = self.proj(features)

        position_ids = (
            torch.arange(N, device=features.device).unsqueeze(0).repeat_interleave(B, 0)
        )
        position_embedding = self.pos_emb(position_ids)
        x = x + position_embedding
        x = self.transformer(x).last_hidden_state
        return self.head(x)[:, 1:, :]

    def predict(self, batch, device):
        if "features" in batch:
            return self(features=batch["features"].to(device))
        else:
            return self(images=batch["images"].to(device))


def get_loaders(cfg):

    using_cached_features = cfg.features_path is not None

    def _features_to_tensor(item):
        if not using_cached_features:
            return item
        item["features"] = torch.tensor(item["features"])
        return item

    def get_transform():
        transform = T.Compose(
            [
                T.SelectIndices(),
                T.FramesArrayToTensor(),
                (
                    T.CropAndUpdateTransforms(
                        (256, 256),
                        "center",
                    )
                    if not using_cached_features
                    else lambda x: x
                ),
                T.Add6DOFTargets(),
                _features_to_tensor,
            ]
        )
        return transform

    train_transform = get_transform()
    val_transform = get_transform()

    features_path_mapping = (
        dict(features=cfg.features_path) if using_cached_features else {}
    )

    train_dataset = SweepsDatasetWithAdditionalCachedData(
        cfg.dataset,
        split="train",
        transform=train_transform,
        subsequence_length=cfg.subsequence_length_train,
        subsequence_samples_per_scan="one",
        limit_scans=2 if cfg.debug else None,
        mode="h5_dynamic_load",
        original_image_shape=(480, 640),
        drop_keys=["images"] if using_cached_features else [],
        features_paths=features_path_mapping,
    )
    val_dataset = SweepsDatasetWithAdditionalCachedData(
        cfg.dataset,
        split="val" if not cfg.debug else "train",
        transform=val_transform,
        subsequence_length=(
            cfg.subsequence_length_train if cfg.val_mode == "loss" else None
        ),
        subsequence_samples_per_scan="one",
        limit_scans=2 if cfg.debug else None,
        mode="h5_dynamic_load",
        original_image_shape=(480, 640),
        drop_keys=["images"] if using_cached_features else [],
        features_paths=features_path_mapping,
    )
    test_dataset = SweepsDatasetWithAdditionalCachedData(
        cfg.test_dataset,
        split="val",
        transform=val_transform,
        subsequence_length=(
            cfg.subsequence_length_train if cfg.val_mode == "loss" else None
        ),
        subsequence_samples_per_scan="one",
        limit_scans=2 if cfg.debug else None,
        mode="h5_dynamic_load",
        original_image_shape=(480, 640),
        drop_keys=["images"] if using_cached_features else [],
        features_paths=features_path_mapping,
    )

    collate_fn = BatchCollator(
        pad_keys=[
            "targets",
            "targets_global",
            "images",
            "sample_indices",
            "targets_absolute",
            "features",
        ]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size if cfg.val_mode == "loss" else 1,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size if cfg.val_mode == "loss" else 1,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.command == "test":
        test(args)
    else:
        train(args)
