from collections import defaultdict
from typing import Literal, Optional
import h5py
import pandas as pd
import torch
from dataclasses import dataclass
from copy import deepcopy
from torchvision.transforms import functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np

# from src.submission import predictor
from . import dense_displacement_field as ddf


class BaseImagePreprocessing:
    registry = {}

    def __init_subclass__(cls, *args, **kwargs):
        name = kwargs.pop("name")
        cls.registry[name] = cls
        super().__init_subclass__(*args, **kwargs)

    def __call__(self, image_array, device):
        """Preprocess the image array and return a dictionary. The contents of the dictionary will
        be passed as kwargs to the model."""
        raise NotImplementedError()


class ImageProcessing(BaseImagePreprocessing, name='default'):
    def __init__(
        self,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        center_crop_size: Optional[tuple[int, int]] = None,
    ): 
        self.mean = mean
        self.std = std
        self.center_crop_size = center_crop_size

    def __call__(self, image_array, device):
        # convert to tensor
        img = torch.tensor(image_array)[:, None, :, :] / 255.0

        # maybe normalize
        if self.mean and self.std:
            img -= self.mean
            img /= self.std

        # maybe center crop
        if self.center_crop_size is not None:
            top, left, height, width = compute_center_crop_params(
                img, self.center_crop_size
            )
            img = F.crop(img, top, left, height, width)

        # we need to expand along batch dimension for model input
        img = img[None, ...]
        return [img.to(device)], {}


class LocalAndContextImagesPreprocessing(BaseImagePreprocessing, name="glob_loc"):
    def __call__(self, image_array, device):
        from torchvision import transforms as T
        tensor = torch.tensor(image_array) / 255
        images_for_features = tensor.clone()
        images = tensor.clone()
        images_for_features = T.CenterCrop((256, 256))(images)
        images = T.Resize((224, 224))(images)
        images_for_features = images_for_features[None, :, None, :, :]
        images = images[None, :, None, :, :]

        return [], {"images": images.to(device), "images_for_features": images_for_features.to(device)}


class Predictor:
    def __init__(
        self,
        device,
        model_cfg=None,
        model_path=None,
        expected_raw_image_size_hw: tuple[int, int] = (480, 640),
        image_processing=ImageProcessing()
    ):
        self.model_path = model_path
        self.model_cfg = model_cfg
        self.device = device
        self.expected_raw_image_size_hw = expected_raw_image_size_hw
        self.image_preprocessing = image_processing
        
        self._model_outputs = None
        self._model_inputs = None
        self.pred_tracking_matrices_glob = None
        self.pred_tracking_matrices_loc = None
        self.gt_tracking_matrices_glob = None
        self.gt_tracking_matrices_loc = None

    def setup_model(self):
        if self.model_path:
            self.model = torch.jit.load(self.model_path).eval().to(self.device)
        else: 
            assert self.model_cfg is not None 
            from src.models import get_model
            self.model = get_model(**self.model_cfg)

    def preprocess_model_inputs(self, array):
        B, H, W = array.shape
        assert (H, W) == self.expected_raw_image_size_hw, "Shape mismatch"
        self._model_inputs = self.image_preprocessing(array, self.device)

    def run_model(self):
        assert self.model is not None
        assert self._model_inputs is not None
        with torch.no_grad():
            args, kwargs = self._model_inputs
            self._model_outputs = self.model(*args, **kwargs)

    def postprocess_model_outputs(self):
        """Do necessary postprocessing steps and update the predictor class in-place.

        In this case, we generate the local and global transform sequences.
        """
        model_outputs = self._model_outputs
        model_outputs = model_outputs[0].cpu().numpy()

        N = len(model_outputs)

        pred_tracking_glob = np.zeros((N + 1, 4, 4), dtype=np.float32)
        pred_tracking_glob[0] = np.eye(4)
        pred_tracking_loc = np.zeros((N + 1, 4, 4), dtype=np.float32)
        pred_tracking_loc[0] = np.eye(4)

        for i in range(N):
            pred_rel_pose_matrix = pose_vector_to_matrix(model_outputs[i])
            pred_tracking_loc[i + 1] = pred_rel_pose_matrix
            pred_tracking_glob[i + 1] = pred_tracking_glob[i] @ pred_rel_pose_matrix

        self.pred_tracking_matrices_glob = pred_tracking_glob[1:]
        self.pred_tracking_matrices_loc = pred_tracking_loc[1:]

    def run_inference(self, images_array):
        if self.model is None:
            self.setup_model()

        self.preprocess_model_inputs(images_array)
        self.run_model()
        self.postprocess_model_outputs()

    def set_ground_truth(self, tracking_sequence):
        """
        Set the ground truth tracking (can be used to conveniently compute the errors)
        """

        N = len(tracking_sequence)

        gt_tracking_glob = (
            invert_pose_matrix(tracking_sequence[0])[None, ...] @ tracking_sequence
        )

        # Generate relative gt trackings
        gt_tracking_local = np.zeros((N, 4, 4), dtype=np.float32)
        gt_tracking_local[0] = np.eye(4)

        for i in range(N - 1):
            gt_tracking_local[i + 1] = get_relative_pose(
                gt_tracking_glob[i], gt_tracking_glob[i + 1]
            )

        self.gt_tracking_matrices_glob = gt_tracking_glob[1:]
        self.gt_tracking_matrices_loc = gt_tracking_local[1:]

    def _get_ddf(self, points, transform):
        orig = points[0:3, :]

    def get_ddfs(self, mode="all-pts", device="cpu", landmark_pts=None):
        outputs = {}

        # all points
        H, W = self.expected_raw_image_size_hw
        points_px = ddf.make_image_points(H, W, mode=mode)
        points_px = torch.tensor(points_px).float().to(device)
        px2img = torch.tensor(PIXEL2IMG_MATRIX).float().to(device)

        landmark_pts = (
            landmark_pts
            if landmark_pts is not None
            else ddf.make_example_landmark_points(len(self._model_inputs), H, W)
        ).to(device)

        outputs["pred_glob"] = ddf.cal_global_allpts(
            torch.tensor(self.pred_tracking_matrices_glob).float().to(device),
            px2img,
            points_px,
        )
        if self.gt_tracking_matrices_glob is not None:
            outputs["gt_glob"] = ddf.cal_global_allpts(
                torch.tensor(self.gt_tracking_matrices_glob).float().to(device),
                px2img,
                points_px,
            )
        outputs["pred_loc"] = ddf.cal_local_allpts(
            torch.tensor(self.pred_tracking_matrices_loc).float().to(device),
            px2img,
            points_px,
        )
        if self.gt_tracking_matrices_loc is not None:
            outputs["gt_loc"] = ddf.cal_local_allpts(
                torch.tensor(self.gt_tracking_matrices_loc).float().to(device),
                px2img,
                points_px,
            )

        # landmark

        outputs["pred_glob_landmark"] = ddf.cal_global_landmark(
            torch.tensor(self.pred_tracking_matrices_glob).float().to(device),
            landmark_pts,
            px2img,
        )
        if self.gt_tracking_matrices_glob is not None:
            outputs["gt_glob_landmark"] = ddf.cal_global_landmark(
                torch.tensor(self.gt_tracking_matrices_glob).float().to(device),
                landmark_pts,
                px2img,
            )
        outputs["pred_loc_landmark"] = ddf.cal_local_landmark(
            torch.tensor(self.pred_tracking_matrices_loc).float().to(device),
            landmark_pts,
            px2img,
        )
        if self.gt_tracking_matrices_loc is not None:
            outputs["gt_loc_landmark"] = ddf.cal_local_landmark(
                torch.tensor(self.gt_tracking_matrices_loc).float().to(device),
                landmark_pts,
                px2img,
            )

        return outputs


def pose_vector_to_matrix(params):
    """
    Convert a 6-parameter vector (translation and Euler angles) to a 4x4 pose matrix.

    Parameters:
    params (list or np.array): A list or array with 6 elements: [tx, ty, tz, roll, pitch, yaw]

    Returns:
    np.array: A 4x4 transformation matrix
    """
    assert len(params) == 6, "Input must be a 6-element list or array"

    # Extract translation components
    tx, ty, tz = params[:3]

    # Extract Euler angles
    roll, pitch, yaw = params[3:]

    # Create the rotation matrix using scipy's Rotation
    rotation_matrix = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()

    # Create the 4x4 pose matrix
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = [tx, ty, tz]

    return pose_matrix


PIXEL2IMG_MATRIX = np.array(
    [
        [0.22938919, 0.0, 0.0, -73.28984642],
        [0.0, 0.22097969, 0.0, -52.92463589],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def compute_center_crop_params(img, shape):
    H, W = img.shape[-2:]
    new_H, new_W = shape

    top = int((H - new_H) / 2)
    left = int((W - new_W) / 2)

    assert top > 0
    assert left > 0

    return top, left, new_H, new_W


def invert_pose_matrix(pose):
    # Extract rotation matrix and translation vector from the pose matrix
    R = pose[..., :3, :3]
    t = pose[..., :3, 3][..., None]

    # Compute the transpose of the rotation matrix (which is its inverse)

    R_inv = R.swapaxes(-1, -2)

    # Compute the inverse translation
    t_inv = -R_inv @ t

    # Construct the inverse pose matrix
    pose_inv = deepcopy(pose) * 0
    pose_inv[..., :3, :3] = R_inv
    pose_inv[..., :3, 3] = t_inv[..., 0]
    pose_inv[..., 3, 3] = 1  # Homogeneous coordinate part

    return pose_inv


def get_relative_pose(t_start, t_end):
    return invert_pose_matrix(t_start) @ t_end


def get_predictor(config):     
    preprocessing_name = config.preprocessing.pop('name', 'default')
    preprocessing_cfg = config.pop('preprocessing')

    preprocessing = BaseImagePreprocessing.registry[preprocessing_name](**preprocessing_cfg)

    return Predictor(
        **config, 
        image_processing=preprocessing
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to inference config")
    parser.add_argument("--h5_file", help="h5 file to test on")
    parser.add_argument(
        "--dataset_csv",
        help="If specified runs the predictor on the validation set of the given dataset",
    )
    args = parser.parse_args()

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    predictor = get_predictor(cfg)

    all_metrics = defaultdict(list)

    if args.dataset_csv:
        table = pd.read_csv(args.dataset_csv)
        table = table.loc[table.split == "val"]
        h5_files = table.processed_sweep_path.to_list()
    else:
        h5_files = [args.h5_file]

    for file in h5_files:

        with h5py.File(file, "r") as f:
            print(f.keys())
            print(f["images"].shape)
            # print(f["pixel_to_image"][:])
            images_arr = f["images"][:]
            gt_tracking = f["tracking"][:]

        
        predictor.run_inference(images_arr)
        predictor.set_ground_truth(gt_tracking)
        ddfs = predictor.get_ddfs(mode="5pt-landmark")

        global_err = (((ddfs["pred_glob"] - ddfs["gt_glob"]) ** 2).sum(1) ** 0.5).mean(
            -1
        )
        local_err = (((ddfs["pred_loc"] - ddfs["gt_loc"]) ** 2).sum(1) ** 0.5).mean(-1)

        print("Global avg error:", global_err.mean().item())
        print("Local avg error:", local_err.mean().item())

        all_metrics["global_err"].append(global_err.mean().item())
        all_metrics["local_err"].append(local_err.mean().item())

        global_err = (
            ((ddfs["pred_glob_landmark"] - ddfs["gt_glob_landmark"]) ** 2).sum(1) ** 0.5
        ).mean(-1)
        local_err = (
            ((ddfs["pred_loc_landmark"] - ddfs["gt_loc_landmark"]) ** 2).sum(1) ** 0.5
        ).mean(-1)

        print("Global avg error landmark:", global_err.mean().item())
        print("Local avg error landmark:", local_err.mean().item())

    for k, v in all_metrics.items():
        print(f"Average {k}: {sum(v)/len(v)}")

    # print(predictor.pred_tracking_matrices_glob.shape)
    # print(predictor.pred_tracking_matrices_loc.shape)
    # print(predictor.gt_tracking_matrices_glob.shape)
    # print(predictor.gt_tracking_matrices_loc.shape)

    # predictor = Predictor(
    #     'experiments/tests/lyric-dragon/debug/traced_model.jit.pt',
    #     "cuda", center_crop_size=(256, 256)
    # )
    # predictor.setup_model()
#
# import torch
# obj = torch.load('experiments/tests/lyric-dragon/debug/debug_objects.pt')
#
# assert torch.allclose(predictor.model(obj['images']), obj['expected_outputs'])
#
# with h5py.File('data/tus-rec-processed/tus-rec_000_LH_Per_S_DtP.h5') as f:
#     images_array = f['images'][:]
#
# predictor.preprocess_model_inputs(images_array)
# print(predictor._model_inputs.shape)
# print(obj['images'].shape)
# print(torch.allclose(obj['images'], predictor._model_inputs))
