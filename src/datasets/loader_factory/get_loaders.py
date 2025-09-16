from dataclasses import dataclass, field
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.datasets.sweeps_dataset_v2 import SweepsDataset
from torch import distributed as dist

from src.utils.samplers import EpochSampler, InfiniteSampler
from .get_transform import get_transforms


def get_loaders(
    *,
    dataset: str = "tus-rec",
    val_dataset_name: str | dict[str, str] | None = None,
    sequence_length_train: int | None = None,
    batch_size: int = 1,
    num_dataloader_workers: int = 4,
    validation_mode: str = "full",
    drop_keys: list[str] | None = None,
    subsequence_samples_per_scan: str = "one",
    persistent_workers: bool = False,
    pin_memory: bool = False,
    features_cache_map={},
    shuffle_train=True,
    sampling="epoch",
    effective_epoch_length=None,
    sequence_keys=["images", "tracking"],
    transform_kwargs={},
    dataset_kwargs={},
    no_distributed_val_sampler=False,
):
    """
    Refactored version of get_loaders that takes keyword arguments instead of an args object.

    Args:
        dataset: Dataset name
        val_dataset: Validation dataset name (if different from train dataset name)
        sequence_length_train: Training sequence length
        batch_size: Batch size for training loader
        num_dataloader_workers: Number of workers for data loading
        resize_to: Resize images to this size (height, width)
        tus_rec_crop: Whether to apply TUS-REC specific cropping
        crop_size: Size to crop images to (height, width)
        random_crop: Whether to use random cropping during training
        random_horizontal_flip: Whether to use random horizontal flipping
        random_reverse_sweep: Whether to randomly reverse sweep direction
        validation_mode: Validation mode ("full" or "loss")
        cached_features_map: Mapping of feature names to cache files
        drop_keys: Keys to drop from dataset items
        subsequence_samples_per_scan: How to sample subsequences ("one" or other)
        debug: Whether in debug mode
        **kwargs: Additional arguments

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from src import transform as T

    # Handle default values for mutable arguments
    if features_cache_map is None:
        features_cache_map = {}
    if drop_keys is None:
        drop_keys = []

    train_transform, val_transform = get_transforms(
        **transform_kwargs,
        features_cache_map=features_cache_map,
        sequence_keys=sequence_keys,
    )

    dataset_cls = SweepsDataset

    # DATASET
    train_dataset = dataset_cls(
        name=dataset,
        subsequence_length=sequence_length_train,
        split="train",
        transform=train_transform,
        drop_keys=drop_keys,
        original_image_shape=(480, 640),
        subsequence_samples_per_scan=subsequence_samples_per_scan,
        features_cache_map=features_cache_map,
        sequence_keys=sequence_keys,
        **dataset_kwargs,
    )

    val_dataset_name = val_dataset_name or dataset

    def _get_val_dataset(name):
        return dataset_cls(
            name=name,
            split="val",
            transform=val_transform,
            subsequence_length=(
                None if validation_mode == "full" else sequence_length_train
            ),
            drop_keys=drop_keys,
            original_image_shape=(480, 640),
            subsequence_samples_per_scan=subsequence_samples_per_scan,
            features_cache_map=features_cache_map,
            sequence_keys=sequence_keys,
            **dataset_kwargs,
        )

    if isinstance(val_dataset_name, str):
        val_dataset = _get_val_dataset(val_dataset_name)
    else:
        val_dataset = {
            key: _get_val_dataset(name) for key, name in val_dataset_name.items()
        }

    shuffle_train = shuffle_train
    sampler_train = None
    sampler_val = None

    if no_distributed_val_sampler: 
        _get_sampler_val = lambda ds: None

    elif sampling == "epoch" and dist.is_initialized():
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, shuffle=shuffle_train
        )
        shuffle_train = None

        def _get_sampler_val(dataset):
            return torch.utils.data.DistributedSampler(dataset, shuffle=False)

    elif sampling == "artificial_epoch" and dist.is_initialized():
        start = 0 if not dist.is_initialized() else dist.get_rank()
        world_size = 1 if not dist.is_initialized() else dist.get_world_size()
        step = 1 if not dist.is_initialized() else dist.get_world_size()

        effective_batch_size = batch_size * world_size

        sampler_train = EpochSampler(
            sample_count=len(train_dataset),
            size=effective_epoch_length * effective_batch_size,
            shuffle=shuffle_train,
            seed=0,
            start=start,
            step=step,
        )
        shuffle_train = None

        def _get_sampler_val(dataset):
            return EpochSampler(
                sample_count=len(dataset),
                size=len(dataset),
                shuffle=False,
                seed=0,
                start=start,
                step=step,
            )

    else:
        _get_sampler_val = lambda ds: None

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_dataloader_workers,
            pin_memory=pin_memory,
            shuffle=shuffle_train,
            sampler=sampler_train,
            persistent_workers=persistent_workers,
        )
        if len(train_dataset) > 0
        else None
    )

    def _get_val_loader(ds):
        return DataLoader(
            ds,
            batch_size=1,
            num_workers=num_dataloader_workers,
            pin_memory=pin_memory,
            sampler=_get_sampler_val(ds),
            persistent_workers=persistent_workers,
        )

    if isinstance(val_dataset, dict):
        val_loader = {key: _get_val_loader(ds) for key, ds in val_dataset.items()}
    else:
        val_loader = _get_val_loader(val_dataset)

    return train_loader, val_loader
