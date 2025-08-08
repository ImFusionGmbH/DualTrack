import inspect
import random
from copy import copy, deepcopy
from dataclasses import dataclass
from inspect import isclass, isfunction
from typing import Literal, Sequence
import warnings

import numpy as np
import torch
from git import Optional
from torchvision.transforms.v2 import *

from .utils.pose import invert_pose_matrix, matrix_to_pose_vector


def applies_to_dict(key):
    """
        This decorator allows a function or method that operates on a single argument to also work seamlessly with dictionaries.
    If the function is called with a dictionary instead of a regular argument, the function will process the value corresponding to a specified key in the dictionary.
    This is particularly useful in data-processing pipelines where each step may handle individual objects or parts of a dictionary representing a larger data structure.
    """

    def decorator(fn):
        def wrapper_fn(arg):
            if not isinstance(arg, dict):
                return fn(arg)
            else:
                out = arg
                out[key] = fn(out[key])
                return out

        def wrapper_method(self, arg):
            if not isinstance(arg, dict):
                return fn(self, arg)
            else:
                out = arg
                out[key] = fn(self, out[key])
                return out

        if "self" in inspect.signature(fn).parameters:
            return wrapper_method
        else:
            return wrapper_fn

    return decorator


@dataclass
class Add6DOFTargets:
    """
    Add relative transformations as 6 degrees of freedom format as targets for the given data item.
    """

    targets_mode: tuple[str, str] = ("local", 'global')  # unused
    smooth_targets: Optional[int] = None

    def __call__(self, item):
        if "tracking" not in item: 
            warnings.warn(f"Tracking not found in data item. This will be a no-op.")
            return item

        gt_tracking_world = item["tracking"]

        targets_absolute = matrix_to_pose_vector(gt_tracking_world)
        if 'absolute' in self.targets_mode:
            item['targets_absolute'] = torch.tensor(targets_absolute, dtype=torch.float32)

        # set to global relative to first frame
        gt_tracking = (
            invert_pose_matrix(gt_tracking_world[0])[None, ...] @ gt_tracking_world
        )

        # targets
        # targets_global = gt_tracking
        # item['targets_global'] = matrix_to_pose_vector(targets_global)
        # item['targets_global'] = torch.tensor(targets, dtype=)

        gt_tracking_rel = np.zeros((len(gt_tracking_world) - 1, 4, 4))
        # gt_tracking_rel[0] = np.eye(4)
        for i in range(len(gt_tracking_world) - 1):
            gt_tracking_rel[i] = invert_pose_matrix(gt_tracking[i]) @ gt_tracking[i + 1]
        targets_local = gt_tracking_rel

        if self.smooth_targets:
            # we will want to compute a moving average of the targets
            from scipy.ndimage import uniform_filter1d

            targets_local = uniform_filter1d(targets_local, self.smooth_targets, axis=0)

        targets_local = matrix_to_pose_vector(targets_local)
        targets_local = torch.tensor(targets_local, dtype=torch.float32)

        if 'local' in self.targets_mode:
            item["targets"] = targets_local

        targets_global = gt_tracking[1:]  # skip first pose as it is always identity
        targets_global = matrix_to_pose_vector(targets_global)
        targets_global = torch.tensor(targets_global, dtype=torch.float32)
        
        if 'global' in self.targets_mode:
            item["targets_global"] = targets_global

        return item


@dataclass
class FramesArrayToTensor:
    mean: Optional[float] = None
    std: Optional[float] = None
    resize_to: Optional[tuple[int, int]] = None
    tus_rec_crop: bool = False
    key: str = "images"

    def __call__(self, item):
        if self.key in item: 
            item[self.key] = self.transform(item[self.key])
        return item

    def transform(self, img):
        img = torch.tensor(img)[:, None, :, :] / 255.0
        if self.mean and self.std:
            img -= self.mean
            img /= self.std

        if self.resize_to:
            img = Resize(self.resize_to)(img)
        if self.tus_rec_crop:
            img = tus_rec_256_crop(img)
        return img


class ApplyToDictFields:
    def __init__(self, keys, transform=None):
        self.keys = keys
        self._transform = transform

    def transform(self, img): 
        return self._transform(img)

    def __call__(self, item):
        for key in self.keys:
            if key in item: 
                item[key] = self.transform(item[key])
        return item


class ImagePreprocessingUSFM(ApplyToDictFields):
    def __init__(
        self,
        key=["images"],
        resize_to: Optional[tuple[int, int]] = (512, 512),
        crop_size: tuple[int, int] | None = None,
    ):
        super().__init__(key)
        self.resize_to = resize_to
        self.crop_size = crop_size

    def transform(self, img):
        images_fm = img
        images_fm = torch.tensor(images_fm) / 255.0

        if self.crop_size is not None:
            images_fm = functional.center_crop(images_fm, self.crop_size)

        if self.resize_to is not None:
            images_fm = Resize(self.resize_to)(images_fm)

        images_fm = images_fm[:, None, :, :]
        images_fm = images_fm.repeat([1, 3, 1, 1])
        images_fm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(
            images_fm
        )

        return images_fm


class ImagePreprocessingMedSAM(ApplyToDictFields):
    def __init__(
        self,
        key=["images"],
        resize_to=(1024, 1024),
        crop_size: tuple[int, int] | None = None,
    ):
        super().__init__(key)
        self.resize_to = resize_to
        self.crop_size = crop_size

    def transform(self, img):
        images_fm = img
        images_fm = torch.tensor(images_fm) / 255.0

        if self.crop_size is not None:
            images_fm = functional.center_crop(images_fm, self.crop_size)

        if self.resize_to is not None:
            images_fm = Resize(self.resize_to)(images_fm)

        images_fm = images_fm[:, None, :, :]
        images_fm = images_fm.repeat([1, 3, 1, 1])
        return images_fm


class SelectIndices:

    def __init__(self, sequence_keys=["images", "tracking"]): 
        self.sequence_keys = sequence_keys

    def __call__(self, item):
        i0 = item["start_idx"]
        i1 = item["stop_idx"]
        sequence_keys = self.sequence_keys
        sequence_keys.extend(item.get("_extra_sequence_keys", []))

        for key in sequence_keys:
            if key in item:
                item[key] = item[key][i0:i1]

        h5_keys = ['calibration', 'spacing', 'dimensions']
        h5_keys.extend(item.get('_extra_h5_keys', []))

        # item["calibration"] = item["calibration"][:]
        # item["spacing"] = item["spacing"][:]
        # item["dimensions"] = item["dimensions"][:]

        for key in h5_keys: 
            if key in item: 
                item[key] = item[key][:]

        # item.pop("img")
        return item


@dataclass
class SelectIndicesTwoFrame:
    prob_skip_one_frame: float = 0

    def __call__(self, item):

        if (
            self.prob_skip_one_frame
            and torch.rand((1,)).item() < self.prob_skip_one_frame
        ):
            if item["stop_idx"] < len(item["images"]):
                item["stop_idx"] += 1  # skip a frame

        i0 = item["start_idx"]
        i1 = item["stop_idx"]
        item["images"] = item["images"][[i0, i1 - 1]]
        item["tracking"] = item["tracking"][[i0, i1 - 1]]
        item["calibration"] = item["calibration"][:]
        item["spacing"] = item["spacing"][:]
        item["dimensions"] = item["dimensions"][:]

        item.pop("img")
        return item


class TrackedUltrasoundSweepTransform:
    def __init__(
        self,
        max_sequence_length=None,
        targets_mode="local",
        resize_to=None,
        smooth_targets=None,
        mean=None,
        std=None,
    ):
        self.max_sequence_length = max_sequence_length
        self.targets_mode = targets_mode
        self.resize_to = resize_to
        self.smooth_targets = smooth_targets
        self.pixel_level_augmentations = Identity()
        self.mean = mean
        self.std = std

    def with_pixel_level_augmentations(
        self,
        on: bool = True,
        random_gamma_prob: float = 0.5,
        random_gamma_range: tuple[float, float] = (0.5, 4),
        random_contrast_prob: float = 0.5,
        random_contrast_range: tuple[float, float] = (0.6, 3),
    ):
        if on:
            self.pixel_level_augmentations = Compose(
                [
                    RandomApply(
                        [RandomContrast(random_contrast_range)], p=random_contrast_prob
                    ),
                    RandomApply([RandomGamma(random_gamma_range)], p=random_gamma_prob),
                ]
            )
        return self

    def __call__(self, item):
        out = {}

        N, H, W = item["img"].shape
        if "start_idx" in item:
            start = item["start_idx"]
            stop = item["stop_idx"]
        elif self.max_sequence_length is not None:
            random_offset = torch.randint(0, N - self.max_sequence_length, (1,)).item()
            start = random_offset
            stop = random_offset + self.max_sequence_length
        else:
            start = 0
            stop = N

        out["start"] = start
        out["stop"] = stop
        out["sweep_id"] = item["sweep_id"]
        out["original_img_size"] = np.array(item["img"].shape[-2:])

        # augmentations and preprocessing for the image
        out["images"] = torch.tensor(item["img"][start:stop])[:, None, :, :] / 255.0
        out["images"] = self.pixel_level_augmentations(out["images"])
        if self.mean and self.std:
            out["images"] -= self.mean
            out["images"] /= self.std

        if self.resize_to:
            out["images"] = Resize(self.resize_to)(out["images"])

        gt_tracking_world = item["tracking"][start:stop]
        gt_tracking = (
            invert_pose_matrix(gt_tracking_world[0])[None, ...] @ gt_tracking_world
        )
        out["gt_tracking"] = gt_tracking

        # targets
        if self.targets_mode == "global":
            targets = gt_tracking
        else:
            gt_tracking_rel = np.zeros((len(gt_tracking_world), 4, 4))
            gt_tracking_rel[0] = np.eye(4)
            for i in range(len(gt_tracking_world) - 1):
                gt_tracking_rel[i + 1] = (
                    invert_pose_matrix(gt_tracking[i]) @ gt_tracking[i + 1]
                )
            targets = gt_tracking_rel

            if self.smooth_targets:
                # we will want to compute a moving average of the targets
                from scipy.ndimage import uniform_filter1d

                targets = uniform_filter1d(targets, self.smooth_targets, axis=0)

        targets = matrix_to_pose_vector(targets)
        targets = torch.tensor(targets, dtype=torch.float32)

        out["targets"] = targets
        out["calibration_matrix"] = item["calibration"][:]
        out["spacing"] = item["spacing"][:]
        out["dimensions"] = item["dimensions"][:]
        return out


class RandomPlaySweepBackwards:
    def __init__(self, prob=0.5, image_keys=['images', 'features']):
        self.prob = prob
        self.image_keys = image_keys

    def is_valid_image_key(self, key):
        return key in self.image_keys

    def __call__(self, item):
        if torch.rand((1,)).item() > self.prob:
            return item
            
        for key in [key for key in item.keys() if self.is_valid_image_key(key)]:
            if key in item: 
                item[key] = np.flip(item[key], 0).copy()

        item["tracking"] = np.flip(item["tracking"], 0).copy()
        return item


class RandomHorizontalFlipImageAndTracking:
    def __init__(self, p=0.5, image_keys=['images', 'features']):
        self.p = p
        self.image_keys = image_keys

    def is_valid_image_key(self, key):
        return key in self.image_keys

    def __call__(self, item):
        if torch.rand((1,)).item() > self.p:
            return item

        for key in item.keys():
            if self.is_valid_image_key(key):
                item[key] = np.flip(item[key], 2).copy()  # horizontal flip

        flip_matrix = np.eye(4)
        flip_matrix[0, 0] = -1

        item["tracking"] = item["tracking"] @ flip_matrix[None, ...]
        return item


class RandomGamma:
    def __init__(self, gamma_range=(0.5, 4)):
        self.gamma_range = gamma_range

    @applies_to_dict("images")
    def __call__(self, img):
        gamma = np.random.uniform(*self.gamma_range)
        return functional.adjust_gamma(img, gamma)


class RandomContrast:
    def __init__(self, contrast_range=(0.6, 3)):
        self.contrast_range = contrast_range

    @applies_to_dict("images")
    def __call__(self, img):
        contrast = np.random.uniform(*self.contrast_range)
        return functional.adjust_contrast(img, contrast)


def compute_center_crop_params(img, shape):
    H, W = img.shape[-2:]
    new_H, new_W = shape

    top = int((H - new_H) / 2)
    left = int((W - new_W) / 2)

    assert top > 0
    assert left > 0

    return top, left, new_H, new_W


def compute_random_crop_params(img, shape):
    return RandomCrop.get_params(img, shape)


def generate_cropped_image_and_img2world_tracking(
    img,
    top,
    left,
    height,
    width,
    px2img_matrix,
    img2world_tracking_sequence,
):
    img_cropped = functional.crop(img, top, left, height, width)
    spacing_ver = px2img_matrix[1, 1]
    spacing_hor = px2img_matrix[0, 0]

    # get pixel 2 image cropped matrix
    pixel2image_cropped = np.eye(4)
    pixel2image_cropped[0, 0] = spacing_hor
    pixel2image_cropped[1, 1] = spacing_ver
    pixel2image_cropped[0, 3] = -width * spacing_hor / 2
    pixel2image_cropped[1, 3] = -height * spacing_ver / 2

    image2pixel_cropped = np.linalg.inv(pixel2image_cropped)

    # get image cropped to image matrix in pixel coords
    img_cropped_to_img_matrix_px = np.eye(4)
    img_cropped_to_img_matrix_px[0, 3] = left
    img_cropped_to_img_matrix_px[1, 3] = top

    #
    img_cropped_to_image = (
        px2img_matrix @ img_cropped_to_img_matrix_px @ image2pixel_cropped
    )
    tracking_cropped_img_2_world = (
        img2world_tracking_sequence @ img_cropped_to_image[None, ...]
    )

    return img_cropped, tracking_cropped_img_2_world, img_cropped_to_image


def thermal_noise(img: np.ndarray, snr_db=50):
    """
    Adds simulated thermal noise to an image

    Args:
        img (np.ndarray) - the image to add noise to. It should be in 0-255 uint8 storage.
        snr_db (float) - the signal to noise ratio measured in decibels.
    """
    noise = np.random.normal(size=img.shape) / 10 ** (snr_db / 10)
    x = np.abs(noise) + 1e-7
    x = np.log10(x) * 10
    x = np.clip(x, -50, 0)

    x = x + 50
    x = x / 50 * 255

    img = img.astype("float") + x
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _run_thermal_noise_demo():
    from src.datasets import DATASET_INFO, SweepsDataset
    import matplotlib.pyplot as plt

    ds = SweepsDataset(DATASET_INFO["forearms2018-no-imu"].data_csv_path, "train")

    item = ds[0]
    plt.imshow(item["img"][0], aspect="auto")
    plt.colorbar()
    plt.savefig("test.png")

    plt.figure()
    plt.imshow(thermal_noise(item["img"][0]), aspect="auto")
    plt.colorbar()
    plt.savefig("test_noise.png")


@dataclass
class GenerateMultiCrop:
    shape: tuple[int, int] = (256, 256)
    num_crops: int = 10

    def __call__(self, item):
        item = item.copy()
        img = item["images"]  # uncropped images
        px2img = item["calibration"]
        item["uncropped_img_tracking"] = item["tracking"]
        tracking_img_2_world = item["tracking"]

        cropped_image_sequences = []
        cropped_tracking_sequences = []
        img_cropped_to_image_matrices = []

        for _ in range(self.num_crops):
            top, left, height, width = compute_random_crop_params(img, self.shape)

            img_cropped, tracking_cropped_img_2_world, img_cropped_to_image = (
                generate_cropped_image_and_img2world_tracking(
                    img, top, left, height, width, px2img, tracking_img_2_world
                )
            )
            cropped_image_sequences.append(img_cropped)
            cropped_tracking_sequences.append(tracking_cropped_img_2_world)
            img_cropped_to_image_matrices.append(img_cropped_to_image)

        cropped_image_sequences = torch.stack(cropped_image_sequences)
        cropped_tracking_sequences = torch.stack(cropped_tracking_sequences)
        img_cropped_to_image_matrices = torch.stack(img_cropped_to_image_matrices)

        item["images"] = cropped_image_sequences
        item["tracking"] = cropped_tracking_sequences
        item["image_cropped_to_image"] = img_cropped_to_image_matrices

        return item


@dataclass
class CropAndUpdateTransforms:
    shape: tuple[int, int]
    crop_type: Literal["center", "random"] = "center"

    def __call__(self, item):
        if 'images' not in item:
            return item

        item = item.copy()
        img = item["images"]

        px2img = item["calibration"]
        item["uncropped_img_tracking"] = item["tracking"]
        tracking_img_2_world = item["tracking"]

        H, W = img.shape[-2:]

        if self.crop_type == "random":
            top, left, height, width = compute_random_crop_params(img, self.shape)
        elif self.crop_type == "center":
            top, left, height, width = compute_center_crop_params(img, self.shape)
        else:
            raise NotImplementedError()

        img_cropped = functional.crop(img, top, left, height, width)

        spacing_ver = px2img[1, 1]
        spacing_hor = px2img[0, 0]

        # get pixel 2 image cropped matrix
        pixel2image_cropped = np.eye(4)
        pixel2image_cropped[0, 0] = spacing_hor
        pixel2image_cropped[1, 1] = spacing_ver
        pixel2image_cropped[0, 3] = -width * spacing_hor / 2
        pixel2image_cropped[1, 3] = -height * spacing_ver / 2

        image2pixel_cropped = np.linalg.inv(pixel2image_cropped)

        # get image cropped to image matrix in pixel coords
        img_cropped_to_img_matrix_px = np.eye(4)
        img_cropped_to_img_matrix_px[0, 3] = left
        img_cropped_to_img_matrix_px[1, 3] = top

        #
        img_cropped_to_image = (
            px2img @ img_cropped_to_img_matrix_px @ image2pixel_cropped
        )
        tracking_cropped_img_2_world = (
            tracking_img_2_world @ img_cropped_to_image[None, ...]
        )

        item["img_cropped_to_image"] = img_cropped_to_image
        item["tracking"] = tracking_cropped_img_2_world
        item["images"] = img_cropped
        item["calibration"] = pixel2image_cropped

        return item


def get_pixel_level_augmentations_v0(
    random_gamma_prob: float = 0.5,
    random_gamma_range: tuple[float, float] = (0.5, 4),
    random_contrast_prob: float = 0.5,
    random_contrast_range: tuple[float, float] = (0.6, 3),
):
    return Compose(
        [
            RandomApply(
                [RandomContrast(random_contrast_range)], p=random_contrast_prob
            ),
            RandomApply([RandomGamma(random_gamma_range)], p=random_gamma_prob),
        ]
    )


class RegularSparseSampleTemporal:
    def __init__(self, sample_every: int = 8, random_offset: bool = False):
        self.sample_every = sample_every
        self.random_offset = random_offset

    def __call__(self, item):
        step = self.sample_every
        start = random.randint(0, self.sample_every - 1) if self.random_offset else 0
        original_len = len(item['tracking'])

        sequence_keys = ["images", "tracking", *item.get("_extra_sequence_keys")]
        for key in sequence_keys:
            item[key] = item[key][start::step]

        item['sample_indices'] = torch.arange(original_len)[start::step]

        return item


class RandomSparseSampleTemporal: 
    def __init__(self, n_samples=32, apply_indexing=True): 
        self.n_samples = n_samples 
        self.apply_indexing = apply_indexing

    def __call__(self, item): 
        i0 = item["start_idx"]
        i1 = item["stop_idx"]

        samples = torch.sort(torch.randperm(i1 - i0)[:self.n_samples]).values
        samples = torch.arange(i0, i1)[samples]

        item['sample_indices'] = samples
        if not self.apply_indexing:
            return item

        N = len(item['tracking'])

        sequence_keys = ["images", "tracking", *item.get("_extra_sequence_keys")]
        for key in sequence_keys:
            if key in item:
                item[key] = item[key][samples]
        
        h5_keys = ['calibration', 'spacing', 'dimensions']
        h5_keys.extend(item.get('_extra_h5_keys', []))

        # item["calibration"] = item["calibration"][:]
        # item["spacing"] = item["spacing"][:]
        # item["dimensions"] = item["dimensions"][:]

        for key in h5_keys: 
            if key in item: 
                item[key] = item[key][:]

        return item 


class RepeatChannels(ApplyToDictFields):
    def __init__(self, keys=['images'], channels=1):
        super().__init__(keys)
        self.channels = channels

    def transform(self, img):
        return img.repeat([1, self.channels, 1, 1])


class CropAndDownsampleForContextImages: 
    def __call__(self, img): 
        return Compose([CenterCrop((380, 540)), Resize((224, 224))])(img)


@applies_to_dict("images")
def tus_rec_256_crop(image):
    image = image[..., 50 : 50 + 256, 640 // 2 - 256 // 2 : 640 // 2 + 256 // 2]
    return image


@applies_to_dict("images")
def tus_rec_224_crop(image):
    image = image[..., 50 : 50 + 224, 640 // 2 - 224 // 2 : 640 // 2 + 224 // 2]
    return image


IMAGE_AUGMENTATIONS_FACTORIES = {
    "pixel_level_augmentations_v0": get_pixel_level_augmentations_v0
}
