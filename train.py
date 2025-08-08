from argparse import ArgumentParser
from contextlib import contextmanager
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
from src.datasets.sweeps_dataset import SweepsDatasetWithAdditionalCachedData
from src.engine.loops import (
    run_full_evaluation_loop,
    run_full_test_loop,
    run_training,
    Callback,
)
import torch
from torch import nn
from src.logger import Logger, get_default_log_dir
from src.optimizer import setup_optimizer
from src.models import model_registry
from src import transform as T
from src.models.fusion_model.fusion_model import dualtrack_fusion_model
from src.datasets import get_dataloaders
from src.models import get_model
import argparse


def train(cfg):

    print("=====================")
    print(OmegaConf.to_yaml(cfg))
    print("=====================")

    logger = Logger.get_logger(
        cfg.logger, cfg.log_dir, cfg, disable_checkpoint=cfg.debug, **cfg.logger_kw
    )
    state = logger.get_checkpoint() if cfg.resume else None
    train_loader, val_loader = get_dataloaders(**cfg.data)

    torch.manual_seed(cfg.seed)  # <- make model weights reproducible
    model = get_model(**cfg.model).to(cfg.device)

    if state:
        model.load_state_dict(state["model"])

    optimizer, scheduler = setup_optimizer(
        model,
        scheduler_name="warmup_cosine",
        num_steps_per_epoch=len(train_loader),
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=cfg.train.epochs,
        weight_decay=cfg.train.weight_decay,
        lr=cfg.train.lr,
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
        epochs=cfg.train.epochs,
        pred_fn=None, #pred_fn,
        device=cfg.device,
        loss_fn=None, #loss_fn,
        validate_every_n_epochs=cfg.train.val_every,
        validation_mode="full",
        use_amp=cfg.use_amp,
        best_score=best_score,
        start_epoch=start_epoch,
        evaluator_kw=cfg.evaluator_kw,
        log_image_indices=cfg.get("log_image_indices", []),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a local encoder model")
    parser.set_defaults(name="pretrain_global_encoder")

    parser.add_argument("--log_dir", default=get_default_log_dir())
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--config", "-c", help="Path to yaml configuration file")
    parser.add_argument("--overrides", "-ov", nargs="+", help="Overrides to config")
    args = parser.parse_args()

    if (
        not args.no_resume
        and os.path.exists(args.log_dir)
        and "config.yaml" in os.listdir(args.log_dir)
    ):
        args.config = os.path.join(args.log_dir, "config.yaml")

    cfg = OmegaConf.create({"log_dir": args.log_dir})
    if args.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    train(cfg)
