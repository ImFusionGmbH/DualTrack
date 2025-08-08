from argparse import ArgumentParser
import json
import logging
import os
from pathlib import Path

import einops
import tensordict
from src.engine.loops import (
    run_full_test_loop,
    run_training,
)
import torch
from torch import nn
from src.logger import Logger, get_default_log_dir
from src.optimizer import setup_optimizer
from src.batch_collator import BatchCollator
from src.datasets import SweepsDataset
from src import transform as T


def get_parser():
    # fmt: off
    parser = ArgumentParser(
        description="Train a temporal attention model on top of pre-computed features",
    )
    parser.set_defaults(name='monet-baseline')
    
    parser.add_argument("--log_dir", default=get_default_log_dir())
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument("--compile_model", action='store_true')
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--val_mode", type=str, choices=('loss', 'full'), default='full')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--use_full_scan_for_val", action='store_true')

    group = parser.add_argument_group("model")

    group = parser.add_argument_group("data")
    parser.add_argument("--dataset", default="tus-rec")
    parser.add_argument("--num_workers", type=int, default=4)
    group.add_argument("--subsequence_length_train", type=lambda s: None if s == "null" else int(s), default=64)
    group.add_argument("--batch_size", default=4, type=int)
    group.add_argument("--use_augmentations", action='store_true', help='Use augmentations')
    group.add_argument("--crop_size", nargs='+', default=(256, 256))

    subparsers = parser.add_subparsers(dest='command')
    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--log_tests_wandb', action='store_true')

    return parser
    # fmt: on


def main():
    parser = get_parser()
    cfg = parser.parse_args()

    if cfg.command == "test":
        return test(cfg)

    print(json.dumps(vars(cfg), indent=4))

    logger = Logger.get_logger("wandb", cfg.log_dir, cfg, disable_checkpoint=cfg.debug)
    state = logger.get_checkpoint()
    if not state and cfg.init_checkpoint:
        state = torch.load(cfg.init_checkpoint, map_location="cpu")

    train_loader, val_loader = get_loaders(cfg)
    logging.info(f"Created data loaders")

    batch = next(iter(train_loader))
    logging.info(f"Loaded sample batch:")
    logging.info(tensordict.TensorDict(batch))

    model = get_model_from_args(cfg).cuda()
    if cfg.compile_model:
        model = torch.compile(model)
    if state:
        print(model.load_state_dict(state["model"]))
    logging.info(
        f"Created model with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters"
    )

    optimizer, scheduler = setup_optimizer(
        model,
        scheduler_name="warmup_cosine",
        num_steps_per_epoch=len(train_loader),
        warmup_epochs=10,
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
        loss_fn=loss_fn,
        validate_every_n_epochs=cfg.val_every,
        validation_mode=cfg.val_mode,
        best_score=best_score,
        start_epoch=start_epoch,
    )


def test(cfg):

    output_dir = os.path.join(cfg.log_dir, "test", cfg.dataset)
    state = torch.load(os.path.join(cfg.log_dir, "checkpoint", "best.pt"))

    _, val_loader = get_loaders(cfg)
    logging.info(f"Created data loaders")

    model = get_model_from_args(cfg).cuda()
    if cfg.compile_model:
        model = torch.compile(model)
    if state:
        print(model.load_state_dict(state["model"]))
    logging.info(
        f"Created model with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters"
    )

    run_full_test_loop(
        model,
        val_loader,
        pred_fn,
        Path(output_dir),
        device=cfg.device,
        use_amp=cfg.use_amp,
    )

    # metrics = run_full_evaluation_loop(
    #     model,
    #     val_loader,
    #     pred_fn,
    #     loss_fn,
    # )
    # print(metrics)


#
# if cfg.log_tests_wandb:
#     import wandb
#     wandb.init(project='trackerless-ultrasound', job_type='test')
#     wandb.log(
#         {f"{k}/test": v for k, v in metrics.items()}
#     )


def pred_fn(batch, model, device):
    pred = model.predict(batch, device)
    return pred


def loss_fn(batch, model, device, pred=None):
    pred = pred if pred is not None else pred_fn(batch, model, device)

    targets = batch["targets"].to(device)

    mse_loss = nn.MSELoss()
    return mse_loss(pred, targets) + get_correlation_loss(pred, targets)


def get_model_from_args(cfg):
    return ResnetLSTM()


class ResnetLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        from src.models.video_resnet import causal_video_rn18_2_frames

        self.resnet = causal_video_rn18_2_frames()

        self.lstm = nn.LSTM(512, 256, num_layers=4, batch_first=True)
        self.fc = nn.Linear(256, 6)

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = einops.rearrange(x, "b n c h w -> b c n h w")
        features = self.resnet(x)
        features = features.mean((-2, -1))  # spatial pooling
        features = einops.rearrange(features, "b c n -> b n c")
        features = self.lstm(features)[0]
        outputs = self.fc(features)[:, 1:, :]
        return outputs

    def predict(self, batch, device):
        return self(batch["images"].to(device))


def get_loaders(cfg):

    def get_transform(use_augmentations):
        transform = T.Compose(
            [
                T.SelectIndices(),
                (
                    T.RandomHorizontalFlipImageAndTracking()
                    if use_augmentations
                    else T.Identity()
                ),
                T.RandomPlaySweepBackwards() if use_augmentations else T.Identity(),
                T.FramesArrayToTensor(),
                T.ApplyToDictFields(["images"], T.CenterCrop(cfg.crop_size)),
                T.Add6DOFTargets(),
            ]
        )
        return transform

    train_transform = get_transform(cfg.use_augmentations)
    val_transform = get_transform(False)

    train_dataset = SweepsDataset(
        cfg.dataset,
        split="train",
        transform=train_transform,
        subsequence_length=cfg.subsequence_length_train,
        subsequence_samples_per_scan="one",
        limit_scans=2 if cfg.debug else None,
        mode="h5_dynamic_load",
        original_image_shape=(480, 640),
    )
    val_dataset = SweepsDataset(
        cfg.dataset,
        split="val" if not cfg.debug else "train",
        transform=val_transform,
        subsequence_length=(
            cfg.subsequence_length_train if not cfg.use_full_scan_for_val else None
        ),
        subsequence_samples_per_scan="one",
        limit_scans=2 if cfg.debug else None,
        mode="h5_dynamic_load",
        original_image_shape=(480, 640),
    )
    train_loader = (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=BatchCollator(
                pad_keys=[
                    "targets",
                    "targets_global",
                    "images",
                    "sample_indices",
                    "targets_absolute",
                ]
            ),
            num_workers=cfg.num_workers,
        )
        if len(train_dataset) > 0
        else None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=BatchCollator(
            pad_keys=[
                "targets",
                "targets_global",
                "images",
                "sample_indices",
                "targets_absolute",
            ]
        ),
        num_workers=cfg.num_workers,
    )

    return train_loader, val_loader


# copied from https://github.com/DIAL-RPI/FreehandUSRecon/blob/master/train_network.py
def get_correlation_loss(labels, outputs):
    # print('labels shape {}, outputs shape {}'.format(labels.shape, outputs.shape))
    x = outputs.flatten()
    y = labels.flatten()
    # print('x shape {}, y shape {}'.format(x.shape, y.shape))
    # print('x shape\n{}\ny shape\n{}'.format(x, y))
    xy = x * y
    mean_xy = torch.mean(xy)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = mean_xy - mean_x * mean_y
    # print('xy shape {}'.format(xy.shape))
    # print('xy {}'.format(xy))
    # print('mean_xy {}'.format(mean_xy))
    # print('cov_xy {}'.format(cov_xy))

    var_x = torch.sum((x - mean_x) ** 2 / x.shape[0])
    var_y = torch.sum((y - mean_y) ** 2 / y.shape[0])
    # print('var_x {}'.format(var_x))

    corr_xy = cov_xy / (torch.sqrt(var_x * var_y))
    # print('correlation_xy {}'.format(corr_xy))

    loss = 1 - corr_xy
    # time.sleep(30)
    # x = output
    # y = target
    #
    # vx = x - torch.mean(x)
    # vy = y - torch.mean(y)
    #
    # loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    # print('correlation loss {}'.format(loss))
    # time.sleep(30)
    return loss


if __name__ == "__main__":
    main()
