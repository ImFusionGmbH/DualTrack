import argparse
import json
import logging
import os
import random
import sys

import h5py
import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from tqdm import tqdm

import src.logger
from src.datasets import DATASET_INFO
from src.evaluator import TrackingEstimationEvaluator
from src.optimizer import setup_optimizer
from src.utils.pose import (
    get_global_and_relative_gt_trackings,
    get_global_and_relative_pred_trackings_from_vectors,
    get_relative_pose,
    matrix_to_pose_vector,
)
from src.transform import IMAGE_AUGMENTATIONS_FACTORIES

sys.path.append(os.getcwd())


def get_args():
    parser = argparse.ArgumentParser(description='Run the 2-frame CNN baseline')
    parser.add_argument(
        "--dataset",
        choices=DATASET_INFO.keys(),
        default=next(iter(DATASET_INFO.keys())),
    )
    parser.add_argument("--num_dataloader_workers", default=8, type=int)
    parser.add_argument("--log_dir", default=src.logger.get_default_log_dir())
    parser.add_argument(
        "--logger", choices=("wandb", "tensorboard", "console"), default="wandb"
    )

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_bfloat", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--scheduler",
        default="cosine",
        choices=["cosine", "none"],
        help="Learning rate scheduler.",
    )
    parser.add_argument("--clip_grad", type=float, default=3)
    parser.add_argument(
        "--optimizer", choices=("adam", "sgd", "adagrad"), default="adam"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument(
        "--model_weights", help="Path to the model weights path to load"
    )
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--optical_flow", action="store_true", help='Set this flag to use optical flow, \
         but note that we did not observe any benefit of it when using larger CNNs')
    parser.add_argument(
        "--resize_to",
        type=int,
        nargs="+",
        default=None,
        help="set to 0 for no resize",
    )
    parser.add_argument("--validate_every", type=int, default=1)
    parser.add_argument("--log_images", action="store_true")
    parser.add_argument("--save_model_every", type=int, default=None)
    parser.add_argument("--model", default="resnet10")
    parser.add_argument(
        "--augmentations", type=str, choices=("none", "v0"), default="none"
    )
    parser.add_argument("--dropout", default=0.25, type=float)
    parser.add_argument("--flip_h_prob", type=float, default=0)
    parser.add_argument("--reverse_sweep_prob", type=float, default=0)
    parser.add_argument("--skip_frame_prob", type=float, default=0)

    parser.add_argument(
        "--epoch_mode",
        choices=["default", "tus_rec"],
        default="default",
        help="If `default`, goes through every single pair of images for each sweep per epoch. \
             If `tus-rec`, does one randomly sampled pair of images per sweep per epoch.",
    )

    subparsers = parser.add_subparsers(dest='command')
    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--train_dir')
    test_parser.add_argument("--test_dataset", default='tus-rec-val')

    export_features_subparser = subparsers.add_parser('export_features')
    export_features_subparser.add_argument('--train_dir')
    export_features_subparser.add_argument('--output')

    args = parser.parse_args()

    if args.val_datasets == []:
        args.val_datasets.append(args.dataset)

    print(json.dumps(vars(args), indent=4))

    return args


def train(args):
    print(json.dumps(vars((args)), indent=4))

    torch.random.manual_seed(args.seed)

    logger = src.logger.get_logger(args.logger, args.log_dir, args)
    logging.info(f"Running in {args.log_dir}.")
    state = logger.get_checkpoint()

    data_info = DATASET_INFO[args.dataset]
    data_csv_path = data_info["data_csv_path"]

    # create the dataset
    train_dataset = FramesDataset(
        data_csv_path,  # args.data_csv_path,
        "train",
        args.optical_flow,
        args.resize_to,
        mean=data_info["pixel_mean"],
        std=data_info["pixel_std"],
        augmentations=args.augmentations,
        frames_per_sweep_per_epoch="all" if args.epoch_mode == "default" else "one",
        flip_h_prob=args.flip_h_prob,
        reverse_sweep_prob=args.reverse_sweep_prob,
        skip_frame_prob=args.skip_frame_prob,
    )
    logging.info(f"Number of training examples: {len(train_dataset)}")

    if os.environ.get("DEBUG_TRUNCATE"):
        train_dataset = [train_dataset[i] for i in range(10)]

    val_dataset = SweepsDataset(DATASET_INFO[args.dataset].data_csv_path, "val")

    # create the dataloader
    train_loader = DataLoader(
        train_dataset,  # type:ignore
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_dataloader_workers,
    )

    model = create_model(args).to(args.device)
    if state: 
        model.load_state_dict(state['model'])
    logging.info(f"model {model.__class__}")

    optimizer, scheduler = setup_optimizer(
        model,
        args.optimizer,
        args.scheduler,
        args.lr,
        args.weight_decay,
        args.epochs,
        state=state
    )
 
    criterion = nn.MSELoss()

    best_loss = state['best_loss'] if state else float("inf")
    start_epoch = 0
    if state and 'epoch' in state: 
        start_epoch = state['epoch']
    elif args.start_epoch: 
        start_epoch = args.start_epoch    

    for epoch in range(start_epoch, args.epochs):

        def get_state(): 
            return dict(
                model=model.state_dict(), 
                optimizer=optimizer.state_dict(), 
                scheduler=scheduler.state_dict(), 
                best_loss=best_loss, 
                epoch=epoch
            )

        logging.info(f"Starting epoch {epoch}")

        # Train one epoch
        total_loss = 0
        model.train()

        for inp, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inp = inp.to(args.device)
            target = target.to(args.device)

            out = model(inp)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()

            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        logger.log({"loss/train": total_loss / len(train_loader)}, epoch)

        if (epoch + 1) % args.validate_every == 0:

            def validate(val_dataset, best_loss):
                val_metrics = run_validation_epoch(
                    args,
                    model,
                    val_dataset,
                    epoch,
                    criterion,
                    logger,
                    image_preprocessing_fn=train_dataset.apply_image_preprocessing,
                    log_suffix=f"val",
                )

                if val_metrics["loss"] < best_loss:
                    logging.info(f"Epoch {epoch} - best_loss achieved")
                    best_loss = val_metrics["loss"]
                    torch.save(
                        get_state(),
                        os.path.join(args.log_dir, "checkpoint", f"best.pt"),
                    )

                return best_loss

            best_loss = validate(val_dataset, best_loss)

        if (c := args.save_model_every) and epoch % c == 0:
            torch.save(
                get_state(),
                os.path.join(args.log_dir, "checkpoint", f"epoch_{epoch}.pt"),
            )

        torch.save(
            get_state(), os.path.join(args.log_dir, "checkpoint", "last.pt")
        )


def test(args): 
    train_dir = args.train_dir 
    output_dir = os.path.join(train_dir, 'test', args.test_dataset)
    os.makedirs(output_dir, exist_ok=True)
    state = torch.load(
        os.path.join(train_dir, 'checkpoint', 'best.pt')
    )

    data_info = DATASET_INFO[args.dataset]
    data_csv_path = data_info["data_csv_path"]
    train_dataset = FramesDataset(
        data_csv_path,  # args.data_csv_path,
        "train",
        args.optical_flow,
        args.resize_to,
        mean=data_info["pixel_mean"],
        std=data_info["pixel_std"],
        augmentations=args.augmentations,
        frames_per_sweep_per_epoch="all" if args.epoch_mode == "default" else "one",
        flip_h_prob=args.flip_h_prob,
        reverse_sweep_prob=args.reverse_sweep_prob,
        skip_frame_prob=args.skip_frame_prob,
    )

    test_dataset = SweepsDataset(DATASET_INFO[args.test_dataset].data_csv_path, "val")

    model = create_model(args)
    model.load_state_dict(state['model'])
    model.to(args.device)

    test_metrics_table = run_test_epoch(
        args,
        model,
        test_dataset,
        image_preprocessing_fn=train_dataset.apply_image_preprocessing,
    )
    test_metrics_table.to_csv(os.path.join(output_dir, 'full_metrics.csv'))


def export_features(args): 
    ...


@torch.no_grad()
def run_validation_epoch(
    conf,
    model,
    dataset,
    epoch,
    criterion,
    logger: src.logger.Logger,
    image_preprocessing_fn,
    log_suffix="val",
):
    model.eval()

    evaluator = TrackingEstimationEvaluator()

    for iter, example in enumerate(
        tqdm(dataset, desc=f"Validation - {log_suffix}", leave=False)
    ):
        sweep_id, img, gt_tracking, calibration_matrix, *_ = example
        N, H, W = img.shape
        outputs = compute_predictions_for_sweep(
            model, img, conf.batch_size, conf.device, image_preprocessing_fn
        ).cpu()

        # compute loss function based on inputs and outputs
        targets = []
        for i in range(N - 1):
            targets.append(
                torch.tensor(
                    matrix_to_pose_vector(
                        get_relative_pose(gt_tracking[i], gt_tracking[i + 1])
                    )
                )
            )
        targets = torch.stack(targets)
        validation_loss = criterion(outputs, targets)
        evaluator.add_metric("loss", validation_loss.item())
        # validation_metrics["loss"].append(validation_loss.item())
        outputs = outputs.numpy()

        # ======= Compute tracking metrics =============
        gt_tracking_glob, gt_tracking_loc = get_global_and_relative_gt_trackings(
            gt_tracking
        )
        pred_tracking_glob, pred_tracking_loc = (
            get_global_and_relative_pred_trackings_from_vectors(outputs)
        )

        metrics, figures = evaluator(
            sweep_id,
            gt_tracking_glob,
            pred_tracking_glob,
            gt_tracking_loc,
            pred_tracking_loc,
            calibration_matrix,
            (H, W),
            include_images=conf.log_images and (iter == 0),
            include_full_ddf=False,
        )
        if figures:
            logger.log(
                {f"{key}/{log_suffix}_0": fig for key, fig in figures.items()}, epoch
            )

    agg_val_metrics = evaluator.aggregate()
    log_metrics = {}
    for key, value in agg_val_metrics.items():
        log_metrics[f"{key}/{log_suffix}"] = value
    logger.log(log_metrics, epoch)

    return agg_val_metrics


@torch.no_grad()
def run_test_epoch(
    conf,
    model,
    dataset,
    image_preprocessing_fn,
):
    model.eval()

    evaluator = TrackingEstimationEvaluator()

    test_metrics = []

    for iter, example in enumerate(
        tqdm(dataset, desc=f"test", leave=False)
    ):
        sweep_id, img, gt_tracking, calibration_matrix, *_ = example
        N, H, W = img.shape
        outputs = compute_predictions_for_sweep(
            model, img, conf.batch_size, conf.device, image_preprocessing_fn
        ).cpu()

        # compute loss function based on inputs and outputs
        targets = []
        for i in range(N - 1):
            targets.append(
                torch.tensor(
                    matrix_to_pose_vector(
                        get_relative_pose(gt_tracking[i], gt_tracking[i + 1])
                    )
                )
            )
        targets = torch.stack(targets)

        outputs = outputs.numpy()

        # ======= Compute tracking metrics =============
        gt_tracking_glob, gt_tracking_loc = get_global_and_relative_gt_trackings(
            gt_tracking
        )
        pred_tracking_glob, pred_tracking_loc = (
            get_global_and_relative_pred_trackings_from_vectors(outputs)
        )

        metrics, figures = evaluator(
            sweep_id,
            gt_tracking_glob,
            pred_tracking_glob,
            gt_tracking_loc,
            pred_tracking_loc,
            calibration_matrix,
            (H, W),
            include_images=conf.log_images and (iter == 0),
            include_full_ddf=False,
        )
        
        test_metrics.append(
            dict(
                sweep_id=sweep_id, 
                **metrics
            )
        )

    return pd.DataFrame(test_metrics)


def compute_predictions_for_sweep(
    model, sweep, batch_size, device, img_preprocessing_fn
):
    """
    Args:
        sweep (np.array) - N, H, W image sweep
    """
    N, *_ = sweep.shape
    inputs = []

    # get model input from sweep

    class HackDataset:

        def __getitem__(self, i):
            im1 = sweep[i]
            im2 = sweep[i + 1]

            # inp = np.stack([im1, im2], axis=-1)
            inp = img_preprocessing_fn(im1, im2)
            return inp

        def __len__(self):
            return N - 1

    # def get_input():
    #     for i in range(N - 1):
    #         im1 = sweep[i]
    #         im2 = sweep[i + 1]
    #
    #         # inp = np.stack([im1, im2], axis=-1)
    #         inp, _ = Transform(conf)(im1, im2)
    #
    #         yield
    #         # inputs.append(inp)

    outputs = []
    for input_batch in DataLoader(
        HackDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    ):  # type:ignore

        outputs.append(model(input_batch.to(device)))

    return torch.concat(outputs, dim=0)


class FramesDataset(Dataset):

    def __init__(
        self,
        metadata_csv_path,
        split,
        optical_flow=False,
        resize_to=None,
        flip_h_prob=0,
        mode="train",
        mean=None,
        std=None,
        augmentations="none",
        frames_per_sweep_per_epoch="all",
        reverse_sweep_prob=0,
        skip_frame_prob=0,
    ):

        self.metadata = pd.read_csv(metadata_csv_path)
        self.metadata = self.metadata.loc[self.metadata["split"] == split]
        self.optical_flow = optical_flow
        self.resize_to = resize_to
        self.flip_h_prob = flip_h_prob
        self.reverse_sweep_prob = reverse_sweep_prob
        self.skip_frame_prob = skip_frame_prob
        self.mode = mode
        self.mean = mean
        self.std = std
        self.augmentations = augmentations

        self.filepaths = self.metadata["processed_sweep_path"].to_list()
        self.sweep_ids = self.metadata["sweep_id"].to_list()
        #self.h5_handles = []
        #for path in self.filepaths:
        #    self.h5_handles.append(h5py.File(path))

        self._indices = []
        for i, path in enumerate(self.filepaths):
            with h5py.File(path) as handle:
                N, *_ = handle["images"].shape
                if frames_per_sweep_per_epoch == "all":
                    self._indices.extend(
                        [
                            (i, j) for j in range(N - 1)
                        ]  # N-1 data items total - one for each pair of frames
                    )
                else:
                    self._indices.append((i, None))

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i, j = self._indices[idx]

        with h5py.File(self.filepaths[i]) as handle:
    
            if j is None:
                N = handle["images"].shape[0]
                j = random.randint(0, N - 2)

            if (
                self.skip_frame_prob
                and torch.rand((1,)).item() < self.skip_frame_prob
                and (j + 2) < N
            ):
                offset = 2
            else:
                offset = 1

            sweep_id = self.sweep_ids[i]
            img1 = handle["images"][j]
            img2 = handle["images"][j + offset]
            t1 = handle["tracking"][j]
            t2 = handle["tracking"][j + offset]

        if (
            self.reverse_sweep_prob
            and torch.rand((1,)).item() < self.reverse_sweep_prob
        ):
            img1, img2 = reversed([img1, img2])
            t1, t2 = reversed([t1, t2])

        if self.flip_h_prob and self.mode == "train":
            flip_h = torch.rand((1,)).item() < self.flip_h_prob
        else:
            flip_h = False

        inp = self.apply_image_preprocessing(img1, img2)

        if flip_h:
            inp = torch.flip(inp, dims=(1,))
            calibration = np.eye(4)
            calibration[0, 0] = -1
            t1 = t1 @ calibration
            t2 = t2 @ calibration

        if t1 is not None:
            target = matrix_to_pose_vector(get_relative_pose(t1, t2))
            target = torch.tensor(target, dtype=torch.float32)
        else:
            target = None

        return inp, target

    def apply_image_preprocessing(self, img1, img2):

        inp = np.stack([img1, img2], axis=-1)
        inp = np.stack([img1, img2], axis=-1)
        inp = T.ToImage()(inp)
        inp = T.ToDtype(torch.float32, scale=True)(inp)

        if self.augmentations == "v0":
            inp = inp.unsqueeze(1)
            inp = IMAGE_AUGMENTATIONS_FACTORIES["pixel_level_augmentations_v0"]()(inp)
            inp = inp[:, 0, ...]

        if self.mean is not None:
            inp = inp - self.mean
        if self.std is not None:
            inp = inp / self.std

        if self.optical_flow:
            # compute optical flow between images
            import cv2

            flow = np.zeros_like(img1)
            flow = cv2.calcOpticalFlowFarneback(
                img1, img2, flow, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow = flow.transpose(2, 0, 1)
            flow = torch.tensor(flow, dtype=torch.float32)
            inp = torch.concat([inp, flow], dim=0)

        if p := self.resize_to:
            inp = T.Resize(p)(inp)

        return inp


class SweepsDataset(Dataset):

    def __init__(self, metadata_csv_path, split, transform=None):
        self.metadata = pd.read_csv(metadata_csv_path)
        self.metadata = self.metadata.loc[self.metadata["split"] == split]
        self.filepaths = self.metadata["processed_sweep_path"].to_list()
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        sweep_id = self.metadata.iloc[idx]["sweep_id"][:]

        with h5py.File(self.filepaths[idx]) as f:
            img = f["images"][:]
            tracking = f["tracking"][:]
            calibration = f["pixel_to_image"][:]
            spacing = f["spacing"][:]
            dimensions = f["dimensions"][:]

        return sweep_id, img, tracking, calibration, spacing, dimensions


def create_model(args):
    in_channels = 4 if args.optical_flow else 2

    if args.model == "prevostnet":
        return Prevost2018ConvNet(in_channels=in_channels, custom_init=False)

    elif args.model == "prevostnet_bn":
        return Prevost2018ConvNet(
            in_channels=in_channels, custom_init=False, norm="batch"
        )

    elif args.model == "resnet10":
        return timm.models.create_model(
            "resnet10t", in_chans=in_channels, num_classes=6, drop_rate=args.dropout
        )

    elif args.model == "efficientnet_b1":
        return timm.models.create_model(
            "efficientnet_b1", in_chans=2, num_classes=6, drop_rate=args.dropout
        )

    elif args.model == "resnet10t_attn":
        from src.models.resnet_plus import resnet10t_attn

        return resnet10t_attn(in_chans=2, num_classes=6, drop_rate=args.dropout)

    else:
        raise ValueError()
    # return hydra.utils.instantiate(conf.model)


class Prevost2018ConvNet(nn.Module):

    def __init__(
        self, in_channels=2, dropout=0.25, custom_init=True, input_size=128, norm=None
    ):
        super().__init__()

        if norm == "batch":
            norm_layer = torch.nn.BatchNorm2d
        else:

            def norm_layer(d):
                return torch.nn.Identity()

        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=5, stride=2, padding=2
        )
        self.norm1 = norm_layer(64)
        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2
        )
        self.norm2 = norm_layer(64)
        self.pool1 = torch.nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
        )
        self.norm3 = norm_layer(64)
        self.conv4 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
        )
        self.norm4 = norm_layer(64)
        self.pool2 = torch.nn.MaxPool2d(stride=2, kernel_size=2)

        self.dropout = torch.nn.Dropout(p=dropout)

        self.act = torch.nn.ReLU()

        fmap_size = input_size // 2**6

        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(fmap_size * fmap_size * 64, 512)
        self.fc2 = torch.nn.Linear(512, 6)

        if custom_init:
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        for p in m.parameters():
            torch.nn.init.normal_(p.data, 0, 0.01)

    def forward(self, x):
        x = self.norm1(self.act(self.conv1(x)))
        x = self.norm2(self.act(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.norm1(self.act(self.conv3(x)))
        x = self.norm2(self.act(self.conv4(x)))
        x = self.dropout(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def ConvNetTanh(backbone):
    return torch.nn.Sequential(backbone, nn.Tanh())


if __name__ == "__main__":
    args = get_args()
    if args.command == 'test':
        test(args)
    else: 
        train(args)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', '-c', default='conf/baseline.yaml')
    # args = parser.parse_args()
    #
    # conf = get_config_from_cli(["conf/baseline.yaml"])
    # main(conf)

# from timm.models import resnet