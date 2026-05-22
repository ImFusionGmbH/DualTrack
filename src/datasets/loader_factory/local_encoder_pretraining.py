from dataclasses import dataclass, field
import torch
from src.datasets import DATASET_INFO, SweepsDataset
from torch.utils.data import DataLoader
import numpy as np

from src.datasets.sweeps_dataset import SweepsDatasetWithAdditionalCachedData
from torch import distributed as dist


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
    subsequence_samples_per_scan: str = "one"


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
        subsequence_samples_per_scan=args.subsequence_samples_per_scan,
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
        subsequence_samples_per_scan=args.subsequence_samples_per_scan,
    )

    shuffle_train = True 
    sampler_train = None
    sampler_val = None
    if dist.is_initialized(): 
        shuffle_train = False
        sampler_train = torch.utils.data.DistributedSampler(train_dataset)
        sampler_val = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers,
            pin_memory=True,
            shuffle=shuffle_train,
            sampler=sampler_train,
        )
        if len(train_dataset) > 0
        else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_dataloader_workers,
        pin_memory=True,
        sampler=sampler_val,
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
    debug=False, 
    **kwargs
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
            **kwargs
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
            **kwargs
        )

    return get_loaders(loader_args, debug=debug)


def get_loaders_optimized(
    dataset: str = "tus-rec",
    val_dataset_name: str | None = None,
    sequence_length_train: int | None = None,
    batch_size: int = 1,
    num_dataloader_workers: int = 4,
    resize_to: tuple[int, int] | None = None,
    tus_rec_crop: bool = False,
    crop_size: tuple[int, int] | None = (256, 256),
    random_crop: bool = False,
    random_horizontal_flip: bool = False,
    random_reverse_sweep: bool = False,
    validation_mode: str = "full",
    cached_features_map: dict | None = None,
    drop_keys: list[str] | None = None,
    subsequence_samples_per_scan: str = "one",
    debug: bool = False,
    # Performance optimization parameters
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    use_optimized_transforms: bool = True,
    enable_selective_loading: bool = True,
    **kwargs
):
    """
    Performance-optimized version of get_loaders with:
    - Better DataLoader settings for throughput
    - Optimized transform pipeline
    - Optional selective data loading
    - Improved memory usage patterns
    """
    from src import transform as T

    # Handle default values for mutable arguments
    if cached_features_map is None:
        cached_features_map = {}
    if drop_keys is None:
        drop_keys = []

    def cached_features_transform(item):
        for key in cached_features_map.keys():
            item[key] = torch.tensor(item[key])
        return item

    class OptimizedFramesArrayToTensor:
        """Optimized tensor conversion using OpenCV for faster resizing."""
        def __init__(self, resize_to=None, tus_rec_crop=False):
            self.resize_to = resize_to
            self.tus_rec_crop = tus_rec_crop
        
        def __call__(self, item):
            if "images" in item:
                item["images"] = self.transform(item["images"])
            return item
        
        def transform(self, img):
            # More efficient conversion
            if isinstance(img, np.ndarray):
                img = img.astype(np.float32) / 255.0
                
                # Apply crop before resize for efficiency
                if self.tus_rec_crop:
                    img = img[..., 50:306, 192:448]
                
                # Use OpenCV for faster resizing
                if self.resize_to:
                    resized = []
                    for frame in img:
                        resized_frame = cv2.resize(
                            frame, self.resize_to[::-1], 
                            interpolation=cv2.INTER_LINEAR
                        )
                        resized.append(resized_frame)
                    img = np.stack(resized)
                
                # Convert to tensor at the end
                img = torch.from_numpy(img)[:, None, :, :]
            else:
                # Fallback for tensor inputs
                img = torch.tensor(img)[:, None, :, :] / 255.0
                if self.tus_rec_crop:
                    from src.transform import tus_rec_256_crop
                    img = tus_rec_256_crop(img)
                if self.resize_to:
                    img = T.Resize(self.resize_to)(img)
            
            return img

    def get_transform(train=True):
        if use_optimized_transforms:
            return T.Compose([
                T.SelectIndices(),
                (
                    T.RandomHorizontalFlipImageAndTracking()
                    if random_horizontal_flip and train
                    else T.Identity()
                ),
                (
                    T.RandomPlaySweepBackwards()
                    if random_reverse_sweep and train
                    else T.Identity()
                ),
                OptimizedFramesArrayToTensor(
                    resize_to=resize_to,
                    tus_rec_crop=tus_rec_crop,
                ),
                (
                    T.CropAndUpdateTransforms(
                        crop_size,
                        "random" if random_crop and train else "center",
                    )
                    if crop_size
                    else T.Identity()
                ),
                T.Add6DOFTargets(),
                cached_features_transform,
            ])
        else:
            # Use original transforms
            return T.Compose([
                T.SelectIndices(),
                (
                    T.RandomHorizontalFlipImageAndTracking()
                    if random_horizontal_flip and train
                    else T.Identity()
                ),
                (
                    T.RandomPlaySweepBackwards()
                    if random_reverse_sweep and train
                    else T.Identity()
                ),
                T.FramesArrayToTensor(
                    resize_to=resize_to,
                    tus_rec_crop=tus_rec_crop,
                ),
                (
                    T.CropAndUpdateTransforms(
                        crop_size,
                        "random" if random_crop and train else "center",
                    )
                    if crop_size
                    else T.Identity()
                ),
                T.Add6DOFTargets(),
                cached_features_transform,
            ])

    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)

    # Choose dataset class based on optimization setting
    if enable_selective_loading:
        try:
            from src.datasets.optimized_sweeps_dataset import OptimizedSweepsDatasetWithAdditionalCachedData
            DatasetClass = OptimizedSweepsDatasetWithAdditionalCachedData
        except ImportError:
            print("Warning: Optimized dataset not available, falling back to standard dataset")
            DatasetClass = SweepsDatasetWithAdditionalCachedData
    else:
        DatasetClass = SweepsDatasetWithAdditionalCachedData

    # DATASET
    train_dataset = DatasetClass(
        metadata_csv_path=DATASET_INFO[dataset].data_csv_path,
        subsequence_length=sequence_length_train,
        split="train",
        transform=train_transform,
        drop_keys=drop_keys,
        features_paths=cached_features_map,
        original_image_shape=(480, 640),
        subsequence_samples_per_scan=subsequence_samples_per_scan,
    )
    
    val_dataset_name = val_dataset_name or dataset
    val_dataset = DatasetClass(
        metadata_csv_path=DATASET_INFO[val_dataset_name].data_csv_path,
        split="val",
        transform=val_transform,
        subsequence_length=(
            None if validation_mode == "full" else sequence_length_train
        ),
        drop_keys=drop_keys,
        features_paths=cached_features_map,
        original_image_shape=(480, 640),
        subsequence_samples_per_scan=subsequence_samples_per_scan,
    )

    # Distributed training setup
    shuffle_train = True 
    sampler_train = None
    sampler_val = None
    if dist.is_initialized(): 
        shuffle_train = False
        sampler_train = torch.utils.data.DistributedSampler(train_dataset)
        sampler_val = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)

    # Optimized DataLoader settings
    dataloader_kwargs = {
        'num_workers': num_dataloader_workers,
        'pin_memory': True,
        'persistent_workers': persistent_workers and num_dataloader_workers > 0,
        'prefetch_factor': prefetch_factor if num_dataloader_workers > 0 else None,
    }

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=sampler_train,
            **dataloader_kwargs
        )
        if len(train_dataset) > 0
        else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=sampler_val,
        **dataloader_kwargs
    )
    
    return train_loader, val_loader


def get_loaders_kw(
    dataset: str = "tus-rec",
    val_dataset_name: str | None = None,
    sequence_length_train: int | None = None,
    batch_size: int = 1,
    num_dataloader_workers: int = 4,
    resize_to: tuple[int, int] | None = None,
    tus_rec_crop: bool = False,
    crop_size: tuple[int, int] | None = (256, 256),
    random_crop: bool = False,
    random_horizontal_flip: bool = False,
    random_reverse_sweep: bool = False,
    validation_mode: str = "full",
    cached_features_map: dict | None = None,
    drop_keys: list[str] | None = None,
    subsequence_samples_per_scan: str = "one",
    debug: bool = False,
    dataset_cls_name = 'default',
    **kwargs
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
    if cached_features_map is None:
        cached_features_map = {}
    if drop_keys is None:
        drop_keys = []

    def cached_features_transform(item):
        for key in cached_features_map.keys():
            item[key] = torch.tensor(item[key])
        return item

    def get_transform(train=True):
        return T.Compose(
            [
                T.SelectIndices(),
                (
                    T.RandomHorizontalFlipImageAndTracking()
                    if random_horizontal_flip and train
                    else T.Identity()
                ),
                (
                    T.RandomPlaySweepBackwards()
                    if random_reverse_sweep and train
                    else T.Identity()
                ),
                T.FramesArrayToTensor(
                    resize_to=resize_to,
                    tus_rec_crop=tus_rec_crop,
                ),
                (
                    T.CropAndUpdateTransforms(
                        crop_size,
                        "random" if random_crop and train else "center",
                    )
                    if crop_size
                    else T.Identity()
                ),
                T.Add6DOFTargets(),
                cached_features_transform,
            ]
        )

    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)

    dataset_cls = SweepsDatasetWithAdditionalCachedData
    if dataset_cls_name == 'optimized':
        from src.datasets.optimized_sweeps_dataset import OptimizedSweepsDatasetWithAdditionalCachedData
        dataset_cls = OptimizedSweepsDatasetWithAdditionalCachedData

    # DATASET
    train_dataset = dataset_cls(
        metadata_csv_path=DATASET_INFO[dataset].data_csv_path,
        subsequence_length=sequence_length_train,
        split="train",
        transform=train_transform,
        drop_keys=drop_keys,
        features_paths=cached_features_map,
        original_image_shape=(480, 640),
        subsequence_samples_per_scan=subsequence_samples_per_scan,
        **kwargs,
    )
    val_dataset_name = val_dataset_name or dataset
    val_dataset = dataset_cls(
        metadata_csv_path=DATASET_INFO[val_dataset_name].data_csv_path,
        split="val",
        transform=val_transform,
        subsequence_length=(
            None if validation_mode == "full" else sequence_length_train
        ),
        drop_keys=drop_keys,
        features_paths=cached_features_map,
        original_image_shape=(480, 640),
        subsequence_samples_per_scan=subsequence_samples_per_scan,
        **kwargs,
    )

    shuffle_train = True 
    sampler_train = None
    sampler_val = None
    if dist.is_initialized(): 
        shuffle_train = False
        sampler_train = torch.utils.data.DistributedSampler(train_dataset)
        sampler_val = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_dataloader_workers,
            pin_memory=True,
            shuffle=shuffle_train,
            sampler=sampler_train,
        )
        if len(train_dataset) > 0
        else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=num_dataloader_workers,
        pin_memory=True,
        sampler=sampler_val,
    )
    return train_loader, val_loader
