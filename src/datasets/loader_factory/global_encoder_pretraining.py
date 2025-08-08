from dataclasses import dataclass, field
from typing import Optional, List
from src import transform as T
from src.datasets import SweepsDataset
import torch
from src.batch_collator import BatchCollator


@dataclass
class LoaderConfig:
    """Configuration for data"""

    dataset: str = "tus-rec"
    num_workers: int = 4
    sample_mode: Optional[str] = None  # Deprecated setting.
    n_samples: int = (
        64  # Length of subsample sequence generated from the full ultrasound scan.
    )
    in_channels: int = 1
    subsequence_length_train: Optional[int] = None
    batch_size: int = 1
    use_augmentations: bool = False
    mean: List[float] = field(default_factory=lambda: [0.0])
    std: List[float] = field(default_factory=lambda: [1.0])
    imagenet_stats: bool = False  # Set this flag to use imagenet mean and std
    resize_to: List[int] = field(default_factory=lambda: [224, 224])
    load_preprocessed_from_disk: bool = (
        False  # Whether to load the already preprocessed downsampled images for the disk, which saves dataloading time if it is a bottleneck.
    )

    def __post_init__(self):
        if self.imagenet_stats:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.in_channels = 3
        if self.in_channels > 1 and len(self.mean) == 1:
            self.mean = self.mean * self.in_channels
            self.std = self.std * self.in_channels


def get_loaders(cfg: LoaderConfig, debug=False):

    def _replace_full_res_image_with_preprocessed(item):
        item = item.copy()
        item["images"] = item.pop("images_downsampled-224")
        return item

    def get_transform(use_augmentations):
        transform = T.Compose(
            [
                (
                    _replace_full_res_image_with_preprocessed
                    if cfg.load_preprocessed_from_disk
                    else T.Identity()
                ),
                T.RandomSparseSampleTemporal(cfg.n_samples),
                (
                    T.RandomHorizontalFlipImageAndTracking()
                    if use_augmentations
                    else T.Identity()
                ),
                T.RandomPlaySweepBackwards() if use_augmentations else T.Identity(),
                # (
                #     T.RegularSparseSampleTemporal(cfg.sample_every, random_offset=True)
                #     if cfg.sample_mode == "regular"
                #     else T.RandomSparseSampleTemporal(cfg.n_samples)
                # ),
                T.FramesArrayToTensor(),
                (
                    T.ApplyToDictFields(["images"], T.Resize(cfg.resize_to))
                    if not cfg.load_preprocessed_from_disk
                    else T.Identity()
                ),
                T.RepeatChannels(["images"], cfg.in_channels),
                T.ApplyToDictFields(["images"], T.Normalize(cfg.mean, cfg.std)),
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
        limit_scans=2 if debug else None,
        mode="h5_dynamic_load",
        drop_keys=["images"] if cfg.load_preprocessed_from_disk else [],
        original_image_shape=(480, 640),
    )
    val_dataset = SweepsDataset(
        cfg.dataset,
        split="val" if not debug else "train",
        transform=val_transform,
        subsequence_length=cfg.subsequence_length_train,
        subsequence_samples_per_scan="one",
        limit_scans=2 if debug else None,
        mode="h5_dynamic_load",
        drop_keys=["images"] if cfg.load_preprocessed_from_disk else [],
        original_image_shape=(480, 640),
    )
    train_loader = torch.utils.data.DataLoader(
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


def get_loaders_simple(
    dataset="tus-rec",
    num_workers=4,
    in_channels=1,
    use_augmentations=False,
    mean=[0], 
    std=[1], 
    load_preprocessed_from_disk=False,
    debug=False,
    batch_size=4
):
    return get_loaders(
        LoaderConfig(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            n_samples=64,
            in_channels=in_channels,
            use_augmentations=use_augmentations,
            mean=mean,
            std=std,
            load_preprocessed_from_disk=load_preprocessed_from_disk,
        ),
        debug=debug,
    )
