import argparse
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from numpy import resize
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.batch_collator import BatchCollator
from src.configuration import BaseConfig
from src.datasets import SweepsDataset, SweepsDatasetWithAdditionalCachedData
from src import transform as T
import torch

# factory classes


@dataclass
class DataFactoryConfig(BaseConfig):
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    use_distributed: bool = False

    # limit number of samples to load

    full_scan_val: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.full_scan_val:
            self.val_batch_size = 1
        else:
            self.val_batch_size = self.batch_size


class BaseDataFactory:
    def __init__(self, config: DataFactoryConfig):
        self.config = config

    def get_train_dataset(self) -> Dataset:
        raise NotImplementedError

    def get_val_dataset(self) -> Dataset:
        raise NotImplementedError

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = self.get_train_dataset()
        val_dataset = self.get_val_dataset()

        if self.config.use_distributed:
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset)
        else:
            train_sampler = None
            val_sampler = None

        if len(train_dataset) > 0:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True if train_sampler is None else None,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                sampler=train_sampler,
            )
        else:
            train_loader = None
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False if val_sampler is None else None,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            sampler=val_sampler,
        )
        return train_loader, val_loader


@dataclass
class TrackingEstimationDataFactoryConfig(DataFactoryConfig):
    random_horizontal_flip: bool = False
    random_reverse_sweep: bool = False
    crop_size: int | None = None
    random_crop: bool = False
    smooth_targets: bool = False
    resize_to: Tuple[int, int] | None = None
    tus_rec_crop: bool = (
        False  # this is a slightly off-center crop that seemed reasonable for the fan-beam ultrasound shape of the ultrasound images
    )
    subsequence_length: int | None = 2
    subsequence_samples_per_scan: Literal["one", "all"] = "all"
    dataset: str = "tus-rec"
    limit_scans: int | None = None
    limit_samples: int | None = None
    train_as_val: bool = False
    usfm_mode: bool = (
        False  # if set, uses the preprocessing for the USFM moundation model
    )

    def __post_init__(self):
        super().__post_init__()
        if self.full_scan_val:
            self.subsequence_length_val = None
            self.subsequence_samples_per_scan_val = "one"
        else:
            self.subsequence_length_val = self.subsequence_length
            self.subsequence_samples_per_scan_val = self.subsequence_samples_per_scan

    @classmethod
    def dev_cfg(cls):
        return cls(limit_scans=1, crop_size=256, subsequence_length=None, batch_size=1)


class TrackingEstimationDataFactory(BaseDataFactory):

    def __init__(
        self,
        config: TrackingEstimationDataFactoryConfig,
        dataset_class: type[SweepsDataset] = SweepsDataset,
        **kwargs
    ):
        super().__init__(config)
        self.config = config
        self.dataset_class = dataset_class
        self.kwargs = kwargs

    def get_train_transform(self):
        return T.Compose(
            [
                T.SelectIndices(),
                (
                    T.RandomHorizontalFlipImageAndTracking()
                    if self.config.random_horizontal_flip
                    else T.Identity()
                ),
                (
                    T.RandomPlaySweepBackwards()
                    if self.config.random_reverse_sweep
                    else T.Identity()
                ),
                (
                    T.FramesArrayToTensor(
                        resize_to=self.config.resize_to,
                        tus_rec_crop=self.config.tus_rec_crop,
                    )
                    if not self.config.usfm_mode
                    else T.ImagePreprocessingUSFM(
                        ["images"], resize_to=self.config.resize_to
                    )
                ),
                (
                    T.CropAndUpdateTransforms(
                        (self.config.crop_size, self.config.crop_size),
                        "random" if self.config.random_crop else "center",
                    )
                    if self.config.crop_size
                    else T.Identity()
                ),
                T.Add6DOFTargets(smooth_targets=self.config.smooth_targets),
            ]
        )

    def get_val_transform(self):
        return T.Compose(
            [
                T.SelectIndices(),
                (
                    T.FramesArrayToTensor(
                        resize_to=self.config.resize_to,
                        tus_rec_crop=self.config.tus_rec_crop,
                    )
                    if not self.config.usfm_mode
                    else T.ImagePreprocessingUSFM(
                        ["images"], resize_to=self.config.resize_to
                    )
                ),
                (
                    T.CropAndUpdateTransforms(
                        (self.config.crop_size, self.config.crop_size),
                        "center",
                    )
                    if self.config.crop_size
                    else T.Identity()
                ),
                T.Add6DOFTargets(smooth_targets=self.config.smooth_targets),
            ]
        )

    def get_train_dataset(self) -> Dataset:
        dataset = self.dataset_class(
            name=self.config.dataset,
            split="train",
            transform=self.get_train_transform(),
            subsequence_length=self.config.subsequence_length,
            subsequence_samples_per_scan=self.config.subsequence_samples_per_scan,
            limit_scans=self.config.limit_scans,
            limit_samples=self.config.limit_samples,
            **self.kwargs
        )
        return dataset

    def get_val_dataset(self) -> Dataset:
        dataset = self.dataset_class(
            name=self.config.dataset,
            split="val" if not self.config.train_as_val else "train",
            transform=self.get_val_transform(),
            subsequence_length=self.config.subsequence_length_val,
            subsequence_samples_per_scan=self.config.subsequence_samples_per_scan_val,
            limit_scans=self.config.limit_scans,
            limit_samples=self.config.limit_samples,
            **self.kwargs
        )
        return dataset


def get_loaders_basic(
    dataset: str = "tus-rec",
    random_horizontal_flip: bool = False,
    random_reverse_sweep: bool = False,
    crop_size: int | None = None,
    random_crop: bool = False,
    resize_to: Tuple[int, int] | None = None,
    in_channels: int = 1,
    mean: list[float] = [0],
    std: list[float] = [1],
    subsequence_length_train: Optional[int] = None,
    full_scan_val: bool = True,
    debug: bool = False,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_key="images",
):

    def _remap_image_key(item):
        if image_key != "images":
            item["images"] = item.pop(image_key)
        return item

    def get_transform(augmentations):
        return T.Compose(
            [
                _remap_image_key,
                T.SelectIndices(),
                (
                    T.RandomHorizontalFlipImageAndTracking()
                    if random_horizontal_flip and augmentations
                    else T.Identity()
                ),
                (
                    T.RandomPlaySweepBackwards()
                    if random_reverse_sweep and augmentations
                    else T.Identity()
                ),
                (T.FramesArrayToTensor()),
                (
                    T.CropAndUpdateTransforms(
                        (crop_size, crop_size),
                        "random" if (random_crop and augmentations) else "center",
                    )
                    if crop_size
                    else T.Identity()
                ),
                (
                    T.ApplyToDictFields(["images"], T.Resize(resize_to))
                    if resize_to
                    else T.Identity()
                ),
                (T.RepeatChannels(["images"], in_channels)),
                T.ApplyToDictFields(["images"], T.Normalize(mean, std)),
                T.Add6DOFTargets(),
            ]
        )

    train_transform = get_transform(True)
    val_transform = get_transform(False)

    train_dataset = SweepsDataset(
        dataset,
        split="train",
        transform=train_transform,
        subsequence_length=subsequence_length_train,
        subsequence_samples_per_scan="one",
        limit_scans=2 if debug else None,
        mode="h5_dynamic_load",
        original_image_shape=(480, 640),
    )
    val_dataset = SweepsDataset(
        dataset,
        split="val" if not debug else "train",
        transform=val_transform,
        subsequence_length=(subsequence_length_train if not full_scan_val else None),
        subsequence_samples_per_scan="one",
        limit_scans=2 if debug else None,
        mode="h5_dynamic_load",
        original_image_shape=(480, 640),
    )

    if len(train_dataset) == 0:
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
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
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size if not full_scan_val else 1,
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
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# factory functions
def get_loaders_for_hybrid_models(
    dataset: str = "tus-rec",
    use_augmentations: bool = False,
    fm_resize_to: tuple[int, int] = (512, 512),
    crop_size_cnn: tuple[int, int] | None = (256, 256),
    subsequence_length_train: int | None = None,
    full_scan_val: bool = True,
    debug: bool = False,
    batch_size: int = 1,
    foundation_model="usfm",
    fm_crop_size: tuple[int, int] | None = None,
    dataset_cls: type[SweepsDataset] = SweepsDataset,
):
    """
    Dataloader factory.

    Builds a dataloader that creates multiple preprocessed images, one full resolution
    but possibly cropped image for a cnn model and one full sized one
    for foundation model with appropriate preprocessing.
    """

    def copy_and_add_fm_images(item):
        item["images_fm"] = item["images"].copy()
        return item

    if foundation_model == "usfm":
        fm_image_processing = T.ImagePreprocessingUSFM(
            ["images_fm"], resize_to=fm_resize_to, crop_size=fm_crop_size
        )
    elif "medsam" in foundation_model:
        fm_image_processing = T.ImagePreprocessingMedSAM(
            ["images_fm"], resize_to=fm_resize_to, crop_size=fm_crop_size
        )
    else:
        raise NotImplementedError(foundation_model)

    train_transform = T.Compose(
        [
            T.SelectIndices(),
            (
                T.RandomHorizontalFlipImageAndTracking(
                    image_keys=["images", "images_fm"]
                )
                if use_augmentations
                else T.Identity()
            ),
            (
                T.RandomPlaySweepBackwards(image_keys=["images", "images_fm"])
                if use_augmentations
                else T.Identity()
            ),
            copy_and_add_fm_images,
            fm_image_processing,
            T.FramesArrayToTensor(),
            (
                T.CropAndUpdateTransforms(crop_size_cnn)
                if crop_size_cnn is not None
                else T.Identity()
            ),
            T.Add6DOFTargets(),
        ]
    )

    val_transform = T.Compose(
        [
            T.SelectIndices(),
            copy_and_add_fm_images,
            fm_image_processing,
            T.FramesArrayToTensor(),
            (
                T.CropAndUpdateTransforms(crop_size_cnn)
                if crop_size_cnn is not None
                else T.Identity()
            ),
            T.Add6DOFTargets(),
        ]
    )

    train_dataset = dataset_cls(
        dataset,
        split="train",
        transform=train_transform,
        subsequence_length=subsequence_length_train,
        subsequence_samples_per_scan="one",
        limit_scans=2 if debug else None,
    )
    val_dataset = dataset_cls(
        dataset,
        split="train" if debug else "val",
        transform=val_transform,
        subsequence_length=(subsequence_length_train if not full_scan_val else None),
        subsequence_samples_per_scan="one",
        limit_scans=2 if debug else None,
    )

    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    else:
        train_loader = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=1 if full_scan_val else batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_loaders_with_cached_image_features(
    dataset: str = "tus-rec",
    use_augmentations: bool = False,
    subsequence_length_train: int | None = None,
    full_scan_val: bool = True,
    debug: bool = False,
    batch_size: int = 1,
    features_paths_mapping={},
    cache=False,
    **loader_kw
):

    def _features_to_tensor(item):
        for name in features_paths_mapping.keys():
            item[name] = torch.tensor(item[name])
        return item

    def get_transform(use_augmentations):
        transform = T.Compose(
            [
                T.SelectIndices(),
                (
                    T.RandomPlaySweepBackwards(
                        image_keys=features_paths_mapping.keys()
                    )
                    if use_augmentations
                    else T.Identity()
                ),
                T.FramesArrayToTensor(),
                T.Add6DOFTargets(),
                _features_to_tensor,
            ]
        )
        return transform

    train_transform = get_transform(use_augmentations)
    val_transform = get_transform(False)

    train_dataset = SweepsDatasetWithAdditionalCachedData(
        dataset,
        split="train",
        transform=train_transform,
        subsequence_length=subsequence_length_train,
        subsequence_samples_per_scan="one",
        limit_scans=2 if debug else None,
        mode="h5_dynamic_load",
        original_image_shape=(480, 640),
        drop_keys=["images", "images_downsampled-224"],
        features_paths=features_paths_mapping,
        cache=cache
    )
    val_dataset = SweepsDatasetWithAdditionalCachedData(
        dataset,
        split="val" if not debug else "train",
        transform=val_transform,
        subsequence_length=(subsequence_length_train if not full_scan_val else None),
        subsequence_samples_per_scan="one",
        limit_scans=2 if debug else None,
        mode="h5_dynamic_load",
        original_image_shape=(480, 640),
        drop_keys=["images", "images_downsampled-224"],
        features_paths=features_paths_mapping,
        cache=cache
    )

    pad_keys = [
        "targets",
        "targets_global",
        "sample_indices",
        "targets_absolute",
    ] + list(features_paths_mapping.keys())

    if len(train_dataset) > 1:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=BatchCollator(pad_keys=pad_keys),
            **loader_kw
        )
    else:
        train_loader = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1 if full_scan_val else batch_size,
        collate_fn=BatchCollator(pad_keys=pad_keys),
        **loader_kw
    )
    return train_loader, val_loader
