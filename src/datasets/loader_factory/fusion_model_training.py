from functools import partial
from src import transform as T
import torch
from src.datasets import SweepsDatasetWithAdditionalCachedData
from src.batch_collator import BatchCollator
from torch import distributed as dist


def get_loaders(
    dataset="tus-rec",
    global_encoder_preprocessing_kw=dict(
        resize_to=(224, 224),
        in_channels=1,
        mean=[0],
        std=[1],
    ),
    local_encoder_preprocessing_kw=dict(
        center_crop_size=(256, 256), 
        resize_to=None,
    ),
    use_augmentations=False,
    features_paths={},
    subsequence_length_train=None,
    batch_size=1,
    load_preprocessed_images_from_disk=True,
    debug=False,
    drop_images=False,
    val_mode="full",
    num_workers=8,
    pin_memory=False,
    include_local_encoder_images=None,
    mode="train",
    **kwargs
):

    if mode == "test":
        batch_size = 1
        use_augmentations = False
        val_mode = "full"

    # if not specified, we include the local encoder images
    # only if the features paths (which are assumed to contain the locations
    # of cached local encoder outputs) is not provided.
    if include_local_encoder_images is None:
        include_local_encoder_images = not features_paths

    def transform(item, train=True):

        # dereference the h5 files
        item = T.SelectIndices(sequence_keys=["images", "images_downsampled-224"])(item)

        # make the correct keys for global and local encoder images (or features)
        if load_preprocessed_images_from_disk:
            item["global_encoder_images"] = item["images_downsampled-224"]
        else:
            item["global_encoder_images"] = item["images"].copy()

        if not features_paths:
            item["local_encoder_images"] = item["images"].copy()

        item.pop("images")

        if train and use_augmentations:
            item = T.RandomPlaySweepBackwards(
                image_keys=[
                    "local_encoder_images",
                    "global_encoder_images",
                    *features_paths.keys(),
                ]
            )(item)

        # convert images to tensors
        item = T.FramesArrayToTensor(key="global_encoder_images")(item)
        if "local_encoder_images" in item:
            item = T.FramesArrayToTensor(key="local_encoder_images")(item)

        # apply global encoder image processing
        if not load_preprocessed_images_from_disk:
            item["global_encoder_images"] = T.Resize(
                global_encoder_preprocessing_kw["resize_to"]
            )(item["global_encoder_images"])
        item["global_encoder_images"] = T.RepeatChannels(
            global_encoder_preprocessing_kw["in_channels"]
        ).transform(item["global_encoder_images"])
        item["global_encoder_images"] = T.Normalize(
            global_encoder_preprocessing_kw["mean"],
            global_encoder_preprocessing_kw["std"],
        )(item["global_encoder_images"])

        # apply local encoder image processing
        if "local_encoder_images" in item:
            crop_size = local_encoder_preprocessing_kw.get('center_crop_size', (256, 256))
            if crop_size:
                item["local_encoder_images"] = T.CenterCrop(crop_size)(
                    item["local_encoder_images"]
                )

            crop_params = local_encoder_preprocessing_kw.get('crop_params', None)
            if crop_params: 
                item['local_encoder_images'] = T.functional.crop(
                    item['local_encoder_images'], **crop_params
                )

            if local_encoder_preprocessing_kw.get('resize_to', None): 
                item["local_encoder_images"] = T.Resize(local_encoder_preprocessing_kw['resize_to'])(
                    item["local_encoder_images"]
                )
            
        # convert any additional features loaded from paths to tensor
        for key in features_paths.keys():
            item[key] = torch.tensor(item[key])

        item = T.Add6DOFTargets()(item)

        return item

    train_transform = partial(transform, train=True)
    val_transform = partial(transform, train=False)

    train_dataset = SweepsDatasetWithAdditionalCachedData(
        dataset,
        split="train",
        transform=train_transform,
        subsequence_length=subsequence_length_train,
        subsequence_samples_per_scan="one",
        mode="h5_dynamic_load",
        original_image_shape=(480, 640),
        drop_keys=["images"] if drop_images else [],
        features_paths=features_paths,
        **kwargs
    )
    val_dataset = SweepsDatasetWithAdditionalCachedData(
        dataset,
        split="val" if not debug else "train",
        transform=val_transform,
        subsequence_length=(subsequence_length_train if val_mode == "loss" else None),
        subsequence_samples_per_scan="one",
        mode="h5_dynamic_load",
        original_image_shape=(480, 640),
        drop_keys=["images"] if drop_images else [],
        features_paths=features_paths,
        **kwargs
    )

    collate_fn = BatchCollator(
        pad_keys=[
            "targets",
            "targets_global",
            "local_encoder_images",
            "sample_indices",
            "targets_absolute",
            "global_encoder_images",
            *features_paths.keys(),
        ]
    )

    shuffle_train = True 
    sampler_train = None
    sampler_val = None
    if dist.is_initialized(): 
        shuffle_train = False
        sampler_train = torch.utils.data.DistributedSampler(train_dataset)
        sampler_val = torch.utils.data.DistributedSampler(val_dataset)

    if len(train_dataset) == 0:
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=sampler_train,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size if val_mode == "loss" else 1,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler_val,
    )
    return train_loader, val_loader
