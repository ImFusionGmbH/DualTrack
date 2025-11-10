import logging
import random

import numpy as np
from omegaconf import OmegaConf
from src.engine.loops import (
    run_training,
)
import torch
from torch import nn
from src.logger import Logger, get_default_log_dir
from src.optimizer import setup_optimizer
from src.datasets import get_dataloaders
from src.models import get_model
from src.utils.distributed import init_distributed
import argparse

from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist


def train(cfg):

    print("== Training Configuration ==")
    print(OmegaConf.to_yaml(cfg))
    print("=============================")

    if cfg.get("do_distributed_training"):
        init_distributed()
        rank = dist.get_rank()
    else: 
        rank = 0 

    logger = Logger.get_logger(
        cfg.logger, cfg.log_dir, cfg, disable_checkpoint=cfg.debug, **cfg.logger_kw
    )
    state = logger.get_checkpoint() if not cfg.get("no_resume", False) else None
    train_loader, val_loader = get_dataloaders(**cfg.data)

    logging.info(f"Setting random seeds to {cfg.seed + rank}")
    random.seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    torch.manual_seed(cfg.seed + rank)
    torch.cuda.manual_seed_all(cfg.seed + rank)

    model = get_model(**cfg.model).to(cfg.device)

    if state:
        model.load_state_dict(state["model"])

    if cfg.get("do_distributed_training"):
        if not cfg.get("train_impl_v2", False):
            raise NotImplementedError("Distributed training is only supported for train_impl_v2")

        if cfg.get("apply_sync_batchnorm"):
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            def check_syncbn(model):
                n_bn  = sum(1 for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
                n_sbn = sum(1 for m in model.modules() if isinstance(m, torch.nn.SyncBatchNorm))
                print(f"BatchNorm*D: {n_bn}, SyncBatchNorm: {n_sbn}")
            check_syncbn(model)

        model = DistributedDataParallel(model, find_unused_parameters=True)
        logging.info("Using DistributedDataParallel")

    num_steps_per_epoch = cfg.get('effective_epoch_length', len(train_loader))
    optimizer, scheduler = setup_optimizer(
        model,
        scheduler_name=cfg.train.get("scheduler", "warmup_cosine"),
        num_steps_per_epoch=num_steps_per_epoch,
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=cfg.train.epochs,
        weight_decay=cfg.train.weight_decay,
        lr=cfg.train.lr,
        state=state,
        use_sam=cfg.train.get("use_sam", False),
        sched_kwargs=cfg.train.get("sched_kwargs", {})
    )
    scaler = torch.GradScaler(device=cfg.device, enabled=cfg.use_amp)
    if state:
        scaler.load_state_dict(state["scaler"])

    best_score = state["best_score"] if state else float("inf")
    start_epoch = state["epoch"] if state else 0

    if cfg.get('train_impl_v2', False):

        # def prepare_batch(batch, device): 
        #     if cfg.get('model_input_key'): 
        #         input = batch[cfg.model_input_key].to(device)
        #     elif cfg.get('model_input_keymap'): 
        #         input = {}
        #         for model_input_key, batch_key in cfg.model_input_keymap.items(): 
        #             input[model_input_key] = batch[batch_key].to(device)
        #     else: 
        #         input = batch['images'].to(device)
# 
        #     extra = {}
        #     if "extra_model_input_keys" in cfg: 
        #         for extra_key in cfg.extra_model_input_keys: 
        #             extra[extra_key] = batch[extra_key].to(device)
# 
        #     target = batch['targets'].to(device)
        #     return input, target, extra

        from src.engine.loops_v2 import run_training as run_training_v2
        from src.engine.loops_v2 import Task, DefaultTrackingEstimationTask

        class GlobalEncoderTask(DefaultTrackingEstimationTask): 
            def forward(self, model: nn.Module, batch: dict, device: torch.device):
                images= batch['images'].to(device)
                sample_indices = batch['sample_indices'].to(device)
                prediction = model(images, sample_indices)
                return prediction

        class FusionModelTask(DefaultTrackingEstimationTask): 
            def forward(self, model: nn.Module, batch: dict, device: torch.device):
                global_encoder_images = batch['global_encoder_images'].to(device)
                local_encoder_inputs = batch['local_encoder_images'].to(device)
                prediction = model(global_encoder_images, local_encoder_inputs)
                return prediction

        task_dict = {
            "global_encoder": GlobalEncoderTask(),
            "fusion": FusionModelTask(),
            "default": DefaultTrackingEstimationTask(),
        }
        task = task_dict.get(cfg.get('task_name', 'default'), DefaultTrackingEstimationTask())

        run_training_v2(
            model=model, 
            task=task,
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            logger=logger, 
            scaler=scaler,
            epochs=cfg.train.epochs,
            device=cfg.device,
            validate_every_n_epochs=cfg.train.val_every,
            validation_mode="full",
            use_amp=cfg.use_amp,
            best_score=best_score,
            start_epoch=start_epoch,
            evaluator_kw=cfg.evaluator_kw,
            log_image_indices=cfg.get("log_image_indices", []),
            config_dict=OmegaConf.to_object(cfg),
            tracked_metric=cfg.get('tracked_metric', "ddf/5pt-avg_global_displacement_error")
        )
    
    else: 
        # TODO this loop implementation should be phased because it does not support DDP. 
        # However, train_impl_v2 is currently buggy for some models, so we keep this as a fallback.
        run_training(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            logger,
            scaler=scaler,
            epochs=cfg.train.epochs,
            pred_fn=None,  # predict_fn implemented by the model will be used
            device=cfg.device,
            loss_fn=None,  # loss implemented by the model will be used
            validate_every_n_epochs=cfg.train.val_every,
            validation_mode="full",
            use_amp=cfg.use_amp,
            best_score=best_score,
            start_epoch=start_epoch,
            evaluator_kw=cfg.evaluator_kw,
            log_image_indices=cfg.get("log_image_indices", []),
            config_dict=OmegaConf.to_object(cfg),
        )


def load_cfg_from_torch_ckpt(path): 
    state = torch.load(path, weights_only=False, map_location='cpu')
    return OmegaConf.create(state['config'])

OmegaConf.register_new_resolver('load_cfg_from_torch_ckpt', load_cfg_from_torch_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", default=get_default_log_dir())
    parser.add_argument("--config", "-c", help="Path to yaml configuration file")
    parser.add_argument(
        "overrides", nargs=argparse.REMAINDER, help="Overrides to config"
    )
    args = parser.parse_args()

    cfg = OmegaConf.create({"log_dir": args.log_dir})
    if args.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    train(cfg)
