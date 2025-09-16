from src import transform as T
import torch
from functools import partial


def fusion_model_transform(
    *,
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
    random_reverse_sweep=False,
    random_subsample_sequence=False,
    random_horizontal_flip=False,
    random_crop=False,
    features_cache_map={},
    load_preprocessed_images_from_disk=True,
    sequence_keys=["images", "tracking"],
    include_raw_images=False
):

    def transform(item, train=True):
        if include_raw_images: 
            item['raw_images'] = item['images'].copy()

        # make the correct keys for global and local encoder images (or features)
        if load_preprocessed_images_from_disk:
            item["global_encoder_images"] = item["images_downsampled-224"]
        else:
            item["global_encoder_images"] = item["images"].copy()

        if "images" in item:
            item["local_encoder_images"] = item["images"].copy()

        item.pop("images", None)

        if train and random_reverse_sweep:
            item = T.RandomPlaySweepBackwards(
                keys=[
                    "tracking",
                    "local_encoder_images",
                    "global_encoder_images",
                    *features_cache_map.keys(),
                ]
            )(item)

        if train and random_subsample_sequence:
            item = T.RandomSubsampleSequence(
                keys=[
                    "tracking",
                    "local_encoder_images",
                    "global_encoder_images",
                    *features_cache_map.keys(),
                ]
            )(item)

        if train and random_horizontal_flip:
            item = T.RandomHorizontalFlipImageAndTracking(
                image_keys=["local_encoder_images", "global_encoder_images"]
            )(item)

        # apply global encoder image processing
        if "global_encoder_images" in item:
            item = T.FramesArrayToTensor(key="global_encoder_images")(item)
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
            item = T.FramesArrayToTensor(key="local_encoder_images")(item)

            if local_encoder_preprocessing_kw.get('crop_params'):
                crop_size = local_encoder_preprocessing_kw.get(
                "center_crop_size", (256, 256)
                )
                if crop_size:
                    item["local_encoder_images"] = T.CenterCrop(crop_size)(
                        item["local_encoder_images"]
                    )
                crop_params = local_encoder_preprocessing_kw.get("crop_params", None)
                if crop_params:
                    item["local_encoder_images"] = T.functional.crop(
                        item["local_encoder_images"], **crop_params
                    )
                if local_encoder_preprocessing_kw.get("resize_to", None):
                    item["local_encoder_images"] = T.Resize(
                        local_encoder_preprocessing_kw["resize_to"]
                    )(item["local_encoder_images"])

            else: 
                item = T.CropAndUpdateTransforms(
                    shape=(256, 256),
                    crop_type="random" if (train and random_crop) else "center",
                    image_key="local_encoder_images",
                    implementation_version=1,
                )(item)

        # convert any additional features loaded from paths to tensor
        for key in features_cache_map.keys():
            item[key] = torch.tensor(item[key])

        item = T.Add6DOFTargets()(item)

        return item

    return partial(transform, train=True), partial(transform, train=False)


def local_encoder_transform(
    *,
    resize_to: tuple[int, int] | None = None,
    tus_rec_crop: bool = False,
    crop_size: tuple[int, int] | None = (256, 256),
    random_crop: bool = False,
    random_horizontal_flip: bool = False,
    random_reverse_sweep: bool = False,
    features_cache_map={},
    sequence_keys=["images", "tracking"],
):

    def cached_features_transform(item):
        for key in features_cache_map.keys():
            item[key] = torch.tensor(item[key]).float()
        return item

    def get_transform(train=True):
        return T.Compose(
            [
                (
                    T.RandomHorizontalFlipImageAndTracking()
                    if random_horizontal_flip and train
                    else T.Identity()
                ),
                (
                    T.RandomPlaySweepBackwards(keys=sequence_keys)
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

    return get_transform(True), get_transform(False)


def get_transforms(transform_version="local_encoder", **kwargs):
    return {
        "local_encoder": local_encoder_transform,
        "fusion": fusion_model_transform,
    }[transform_version](**kwargs)
