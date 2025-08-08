from dataclasses import dataclass, field

from arrow import get
import torch
from src.datasets import DATASET_INFO, SweepsDataset
from torch.utils.data import DataLoader

from src.datasets.sweeps_dataset import SweepsDatasetWithAdditionalCachedData


@dataclass
class LoaderArgs:
    dataset: str = "tus-rec"
    sequence_length_train: int | None = None
    batch_size: int = 1
    num_dataloader_workers: int = 4
    resize_to: tuple[int, int] | None = None
    tus_rec_crop: bool = False
    crop_size: tuple[int, int] | None = (256, 256)
    random_crop: bool = False
    random_horizontal_flip: bool = False
    random_reverse_sweep: bool = False
    validation_mode: str = "full"
    cached_features_map: dict = field(default_factory=dict)
    drop_keys: list[str] = field(default_factory=list)


def get_loaders(args: LoaderArgs = LoaderArgs(), debug=False):

    from src import transform as T

    def cached_features_transform(item):
        for key in args.cached_features_map.keys():
            item[key] = torch.tensor(item[key])
        return item

    def get_transform(train=True):
        return T.Compose(
            [
                T.SelectIndices(),
                (
                    T.RandomHorizontalFlipImageAndTracking()
                    if args.random_horizontal_flip and train
                    else T.Identity()
                ),
                (
                    T.RandomPlaySweepBackwards()
                    if args.random_reverse_sweep and train
                    else T.Identity()
                ),
                T.FramesArrayToTensor(
                    resize_to=args.resize_to,
                    tus_rec_crop=args.tus_rec_crop,
                ),
                (
                    T.CropAndUpdateTransforms(
                        args.crop_size,
                        "random" if args.random_crop and train else "center",
                    )
                    if args.crop_size
                    else T.Identity()
                ),
                T.Add6DOFTargets(),
                cached_features_transform,
            ]
        )

    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)

    # DATASET
    train_dataset = SweepsDatasetWithAdditionalCachedData(
        metadata_csv_path=DATASET_INFO[args.dataset].data_csv_path,
        subsequence_length=args.sequence_length_train,
        split="train",
        transform=train_transform,
        drop_keys=args.drop_keys,
        features_paths=args.cached_features_map,
        original_image_shape=(480, 640),
    )
    val_dataset = SweepsDatasetWithAdditionalCachedData(
        metadata_csv_path=DATASET_INFO[args.dataset].data_csv_path,
        split="val",
        transform=val_transform,
        subsequence_length=(
            None if args.validation_mode == "full" else args.sequence_length_train
        ),
        drop_keys=args.drop_keys,
        features_paths=args.cached_features_map,
        original_image_shape=(480, 640),
    )
    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers,
            pin_memory=True,
            shuffle=True,
        )
        if len(train_dataset) > 0
        else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_dataloader_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_loaders_simple(
    dataset="tus-rec",
    augmentations=True,
    cached_features_file=None,
    sequence_length_train=None,
    batch_size=1, 
    num_dataloader_workers=4, 
    mode="train",
    debug=False
):
    is_test = mode != "train"
    use_augmentations = augmentations and not is_test

    if cached_features_file is None:
        loader_args = LoaderArgs(
            dataset=dataset,
            sequence_length_train=sequence_length_train,
            batch_size=batch_size,
            num_dataloader_workers=num_dataloader_workers,
            resize_to=None,
            random_crop=use_augmentations,
            random_horizontal_flip=use_augmentations,
            random_reverse_sweep=use_augmentations,
            validation_mode="full",
            drop_keys=["images_downsampled-224"],
        )
    else:
        loader_args = LoaderArgs(
            dataset=dataset,
            sequence_length_train=sequence_length_train,
            batch_size=batch_size,
            num_dataloader_workers=num_dataloader_workers,
            resize_to=None,
            random_reverse_sweep=use_augmentations,
            validation_mode="full",
            drop_keys=["images", "images_downsampled-224"],
            cached_features_map={"image_features": cached_features_file},
        )

    return get_loaders(loader_args, debug=debug)
