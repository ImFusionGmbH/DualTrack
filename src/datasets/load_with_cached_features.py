from dataclasses import dataclass, field
from typing import Any
import torch
from contextlib import contextmanager
import h5py
from src.datasets import SweepsDataset as _SweepsDataset
from src import transform as T
from src.batch_collator import BatchCollator


from src.data_factory import (
    TrackingEstimationDataFactory,
    TrackingEstimationDataFactoryConfig,
)


class SweepsDataset(_SweepsDataset):
    def __init__(
        self,
        *args,
        features_path="data/pre-computed-features/lyric-dragon/data.h5",
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.features_path = features_path

    @contextmanager
    def load(self, path):
        try:
            f = h5py.File(path)
            yield f
        except:
            f = None
        finally:
            if f:
                f.close()

    def _load_raw_data(self, scan_idx, sweep_id):
        data = super()._load_raw_data(scan_idx, sweep_id)

        with self.load(self.features_path) as f:
            data["pooled_cnn_features"] = f[sweep_id][:]
        data["_extra_sequence_keys"].append("pooled_cnn_features")
        return data


@dataclass
class LoaderConfig:
    dataset: str = "tus-rec"
    include_context_images: bool = True
    include_local_images: bool = False
    resize_to: tuple[int, int] = (224, 224)
    in_channels: int = 1
    mean: list[float] = field(default_factory=lambda: [0])
    std: list[float] = field(default_factory=lambda: [1])
    imagenet_mean: bool = False
    use_augmentations: bool = False
    subsequence_length_train: int | None = None
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = False
    debug: bool = False
    features_path: str = (
        "data/pre-computed-features/lyric-dragon/data.h5"
    )
    val_mode: str = "full"

    def __post_init__(self):
        if self.imagenet_mean:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]


def get_loaders(cfg: LoaderConfig):

    def _drop_images(item):
        item.pop("images")
        return item

    def _features_to_tensor(item):
        item["pooled_cnn_features"] = torch.tensor(item["pooled_cnn_features"])
        return item

    def _include_pre_features_images(item):
        item["images_for_features"] = item["images"].copy()
        item["_extra_sequence_keys"].append("images_for_features")
        return item

    def _preprocess_images_for_features(item):
        item = T.FramesArrayToTensor(key="images_for_features")(item)
        item["images_for_features"] = T.CenterCrop((256, 256))(
            item["images_for_features"]
        )
        return item

    def get_transform(use_augmentations):
        transform = T.Compose(
            [
                (
                    _include_pre_features_images
                    if cfg.include_local_images
                    else T.Identity()
                ),
                _drop_images if not cfg.include_context_images else T.Identity(),
                T.SelectIndices(),
                (
                    T.RandomPlaySweepBackwards(
                        image_keys=[
                            "images",
                            "pooled_cnn_features",
                            "images_for_features",
                        ]
                    )
                    if use_augmentations
                    else T.Identity()
                ),
                T.FramesArrayToTensor(),
                T.ApplyToDictFields(["images"], T.Resize(cfg.resize_to)),
                T.RepeatChannels(["images"], cfg.in_channels),
                T.ApplyToDictFields(["images"], T.Normalize(cfg.mean, cfg.std)),
                (
                    _preprocess_images_for_features
                    if cfg.include_local_images
                    else T.Identity()
                ),
                T.Add6DOFTargets(),
                _features_to_tensor,
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
        features_path=cfg.features_path,
    )
    val_dataset = SweepsDataset(
        cfg.dataset,
        split="val" if not cfg.debug else "train",
        transform=val_transform,
        subsequence_length=(
            cfg.subsequence_length_train if cfg.val_mode == "loss" else None
        ),
        subsequence_samples_per_scan="one",
        limit_scans=2 if cfg.debug else None,
        mode="h5_dynamic_load",
        features_path=cfg.features_path,
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
                "pooled_cnn_features",
                "images_for_features",
            ]
        ),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size if cfg.val_mode == "loss" else 1,
        collate_fn=BatchCollator(
            pad_keys=[
                "targets",
                "targets_global",
                "images",
                "sample_indices",
                "targets_absolute",
                "pooled_cnn_features",
                "images_for_features",
            ]
        ),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return train_loader, val_loader
