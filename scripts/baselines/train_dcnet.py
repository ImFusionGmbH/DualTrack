import argparse
import json
import logging
import os
from copy import deepcopy

import einops
import numpy as np
import pandas as pd
import torch
import torch.amp
import torchvision
from torchvision.transforms import v2 as T
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from argparse import ArgumentParser
from src.datasets import DATASET_INFO, SweepsDataset
from src.evaluator import TrackingEstimationEvaluator
from src.logger import Logger, TensorBoardLogger, get_default_log_dir, get_logger
from src.models.guo_et_al_resnext import resnet50
from src.optimizer import setup_optimizer
from src.pose import (
    get_global_and_relative_gt_trackings,
    get_global_and_relative_pred_trackings_from_vectors,
    invert_pose_matrix,
    matrix_to_pose_vector,
)
from src.transform import tus_rec_256_crop, tus_rec_224_crop
from src.utils.utils import load_model_weights

torch.set_float32_matmul_precision("medium")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default=get_default_log_dir())
    parser.add_argument("-c", "--config")
    subparsers = parser.add_subparsers(dest='command')
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--train_dir")
    test_parser.add_argument('--output_dir', required=False)
    test_parser.add_argument('--model_weights', required=False)
    test_parser.add_argument('--train_cfg', required=False)
    test_parser.add_argument('--test_dataset', default='tus-rec-val')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.command == 'test': 
        return test(args)

    cfg = OmegaConf.load(args.config)

    if cfg.use_bfloat:
        torch.set_float32_matmul_precision("medium")

    logger = get_logger(cfg.logger, args.log_dir, cfg)
    state = logger.get_checkpoint()

    # DATASET
    transform = Transform(
        cfg.window_size,
        cfg.resize_to,
        DATASET_INFO[cfg.dataset].pixel_mean,
        DATASET_INFO[cfg.dataset].pixel_std,
        tus_rec_crop=cfg.tus_rec_crop,
    )
    train_dataset = SweepsDataset(
        metadata_csv_path=DATASET_INFO[cfg.dataset].data_csv_path,
        split="train",
        transform=transform.train_transform,
        subsequence_length=cfg.window_size,
    )
    val_dataset = SweepsDataset(
        metadata_csv_path=DATASET_INFO[cfg.dataset].data_csv_path,
        split="val",
        transform=transform.val_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dataloader_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )

    model = GuoEtAlModel()
    if state:
        model.load_state_dict(state["model"])
    model.to(cfg.device)

    optimizer, scheduler = setup_optimizer(
        model,
        cfg.optimizer,
        cfg.scheduler,
        cfg.lr,
        cfg.weight_decay,
        cfg.epochs * len(train_loader),
        state,
    )
    scaler = torch.GradScaler(enabled=cfg.use_amp)
    if state:
        scaler.load_state_dict(state["scaler"])

    get_amp_context = lambda: torch.autocast(
        cfg.device, torch.bfloat16 if cfg.use_bfloat else None, enabled=cfg.use_amp
    )

    start_epoch = state["epoch"] if state else 0
    best_score = state['best_score'] if state else float('inf')

    for epoch in range(start_epoch, cfg.epochs):
        logger.save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_score": best_score
            }
        )

        logging.info(f"Starting epoch {epoch}")

        # TRAIN LOOP
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            if os.environ.get("DEBUG") and batch_idx > 10:
                break

            img, label = batch
            img = img.to(cfg.device)
            label = label.to(cfg.device)

            with get_amp_context():
                output = model(img)
                loss = torch.nn.MSELoss()(output, label)

            if cfg.use_corr_loss:
                loss += get_correlation_loss(label, output)

            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
        logger.log({"loss/train": total_loss / len(train_loader)}, epoch)

        if (epoch + 1) % cfg.check_val_every_n_epoch != 0:
            continue

        # TEST LOOP
        model.eval()
        evaluator = TrackingEstimationEvaluator()

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(val_loader, leave=False, desc="Validation")
            ):

                (
                    img,
                    calibration_matrix,
                    gt_tracking,
                    sweep_id,
                    original_image_shape,
                ) = batch
                img = img.to(cfg.device)
                img = einops.rearrange(img[0], "b n c h w -> n b c h w")

                outputs = []
                for img_i in DataLoader(img, cfg.inference_batch_size):
                    outputs.append(model(img_i))
                outputs = torch.cat(outputs).cpu().numpy()

                gt_tracking_glob, gt_tracking_loc = (
                    get_global_and_relative_gt_trackings(
                        gt_tracking[0].cpu().numpy()
                    )
                )
                pred_tracking_glob, pred_tracking_loc = (
                    get_global_and_relative_pred_trackings_from_vectors(outputs[1:])
                )
                metrics, figures = evaluator(
                    sweep_id[0],
                    gt_tracking_glob,
                    pred_tracking_glob,
                    gt_tracking_loc,
                    pred_tracking_loc,
                    calibration_matrix[0].cpu().numpy(),
                    image_shape_hw=original_image_shape[0],
                    include_images=batch_idx == 0,
                )
                if batch_idx == 0:
                    logger.log(
                        {
                            f"{name}/{name}_val": figure
                            for name, figure in figures.items()
                        }
                    )
                    plt.close("all")

            metrics = evaluator.aggregate()
            logger.log(
                {f"{k}/val": v for k, v in metrics.items()},
                epoch,
            )
            if metrics['ddf/5pt-avg_local_displacement_error'] < best_score:
                logger.save_checkpoint(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score
                    }, 
                    'best.pt'
                )
                best_score = metrics['ddf/5pt-avg_local_displacement_error']

    logging.info("Training completed!")


def test(args): 
    if (d := args.train_dir): 
        args.output_dir = args.output_dir or os.path.join(d, 'test', args.test_dataset)
        args.model_weights = args.model_weights or os.path.join(d, 'checkpoint', 'best.pt')
        args.train_cfg = args.train_cfg or os.path.join(d, 'config_resolved.yaml')

    cfg = OmegaConf.load(args.train_cfg)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(cfg.logger, output_dir, cfg)

    # DATASET
    transform = Transform(
        cfg.window_size,
        cfg.resize_to,
        DATASET_INFO[cfg.dataset].pixel_mean,
        DATASET_INFO[cfg.dataset].pixel_std,
        tus_rec_crop=cfg.tus_rec_crop,
    )
    val_dataset = SweepsDataset(
        metadata_csv_path=DATASET_INFO[args.test_dataset].data_csv_path,
        split="val",
        transform=transform.val_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )

    model = GuoEtAlModel()
    load_model_weights(
        model,
        args.model_weights, 
    )
    model.to(cfg.device)

    # TEST LOOP
    model.eval()
    evaluator = TrackingEstimationEvaluator()

    metrics_by_case = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(val_loader, leave=False, desc="Validation")
        ):

            case_i_metrics = {}
            (
                img,
                calibration_matrix,
                gt_tracking,
                sweep_id,
                original_image_shape,
            ) = batch
            img = img.to(cfg.device)
            img = einops.rearrange(img[0], "b n c h w -> n b c h w")

            outputs = []
            for img_i in DataLoader(img, cfg.inference_batch_size):
                outputs.append(model(img_i))
            outputs = torch.cat(outputs).cpu().numpy()

            gt_tracking_glob, gt_tracking_loc = (
                get_global_and_relative_gt_trackings(
                    gt_tracking[0].cpu().numpy()
                )
            )
            pred_tracking_glob, pred_tracking_loc = (
                get_global_and_relative_pred_trackings_from_vectors(outputs[1:])
            )
            metrics, figures = evaluator(
                sweep_id[0],
                gt_tracking_glob,
                pred_tracking_glob,
                gt_tracking_loc,
                pred_tracking_loc,
                calibration_matrix[0].cpu().numpy(),
                image_shape_hw=original_image_shape[0],
                include_images=batch_idx == 0,
            )

            case_i_metrics['sweep_id'] = sweep_id[0]
            case_i_metrics.update(metrics)
            metrics_by_case.append(case_i_metrics)

            if batch_idx == 0:
                logger.log(
                    {
                        f"{name}/{name}_val": figure
                        for name, figure in figures.items()
                    }
                )
                plt.close("all")

        metrics_by_case = pd.DataFrame(metrics_by_case)
        metrics_by_case.to_csv(
            os.path.join(output_dir, 'full_metrics.csv')
        )
        logger.log(
            {f"{k}/test": v for k, v in evaluator.aggregate().items()},
        )



class Transform:
    def __init__(
        self, window_size=5, resize_to=None, mean=None, std=None, tus_rec_crop=False
    ):

        self.window_size = window_size
        self.resize_to = resize_to
        self.mean = mean
        self.std = std
        self.tus_rec_crop = tus_rec_crop

    def img_transform(self, images):
        images = torch.tensor(images) / 255
        if self.mean:
            images = images - self.mean
        if self.std:
            images = images / self.std
        if self.tus_rec_crop:
            images = tus_rec_224_crop(images)
        if self.resize_to:
            images = T.functional.resize(images, self.resize_to)
        return images

    def train_transform(self, item):
        out = {}
        N, H, W = item["images"].shape

        # sample a random start position
        start = item["start_idx"]
        stop = item["stop_idx"]
        images = item["images"][start:stop]
        images = self.img_transform(images)
        images = images[None, ...]

        tracking = item["tracking"][start:stop]
        relative_tracking = np.zeros((self.window_size - 1, 6))
        for i in range(self.window_size - 1):
            relative_tracking[i] = matrix_to_pose_vector(
                invert_pose_matrix(tracking[i]) @ tracking[i + 1]
            )
        relative_tracking = relative_tracking.mean(0)
        relative_tracking = torch.tensor(relative_tracking, dtype=torch.float32)

        return images, relative_tracking

    def val_transform(self, item):
        images = item["images"][:]
        images = self.img_transform(images)

        # we are using the full sequence but the model only understands subsequences.
        # view it as a sliding window:
        pad_size = self.window_size // 2
        images = torch.nn.functional.pad(images, (0, 0, 0, 0, pad_size, pad_size))
        images = images.unfold(0, self.window_size, 1)
        images = einops.rearrange(images, "n h w c -> 1 n c h w")

        calibration_matrix = item["calibration"][:]
        gt_tracking = item["tracking"][:]
        scan_id = item["sweep_id"]
        original_image_shape = np.array(item["original_image_shape"])

        return images, calibration_matrix, gt_tracking, scan_id, original_image_shape


class GuoEtAlModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet50(sample_size=2, sample_duration=16, cardinality=32)
        model.conv1 = torch.nn.Conv3d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        model.fc = torch.nn.Linear(model.fc.in_features, 6)
        self.model = model

    def forward(self, x):
        outputs, features = self.model(x)
        return outputs


def get_correlation_loss(labels, outputs):
    x = outputs.flatten()
    y = labels.flatten()
    xy = x * y
    mean_xy = torch.mean(xy)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = mean_xy - mean_x * mean_y

    var_x = torch.sum((x - mean_x) ** 2 / x.shape[0])
    var_y = torch.sum((y - mean_y) ** 2 / y.shape[0])

    corr_xy = cov_xy / (torch.sqrt(var_x * var_y))

    loss = 1 - corr_xy

    return loss


if __name__ == "__main__":
    main()