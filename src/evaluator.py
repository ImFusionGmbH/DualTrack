from collections import defaultdict
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
from src.utils.pose import (
    compute_mean_average_errors,
    get_ddf_metrics,
    get_drift_metrics,
    hausdorff_distance,
    plot_pose_differences,
    get_global_and_relative_gt_trackings,
    get_global_and_relative_pred_trackings_from_vectors,
)
from torch import distributed as dist


class TrackingEstimationEvaluator:
    def __init__(
        self,
        include_images=True,
        include_full_ddf=False,
        image_shape_hw=(224, 224),
        include_drift_metrics=True,
        include_mae_metrics=True,
    ):
        self.metrics = defaultdict(list)
        self.scan_ids = []
        self.images = {}
        self.include_images = include_images
        self.include_full_ddf = include_full_ddf
        self.image_shape_hw = image_shape_hw
        self.include_drift_metrics = include_drift_metrics
        self.include_mae_metrics = include_mae_metrics

        self._current_update_cache = {}
        self._is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        if self._is_distributed: 
            self._is_main_rank = dist.get_rank() == 0
        else: 
            self._is_main_rank = True

    def __call__(
        self,
        scan_id: Optional[str],
        gt_tracking_glob,
        pred_tracking_glob,
        gt_tracking_loc,
        pred_tracking_loc,
        calibration_matrix=None,
        image_shape_hw=None,
        include_images=None,
        include_full_ddf=None,
    ):
        """Computes the metrics for a single scan and collects them to the aggregator. Returns the single scan
        metrics as a dictionary.
        """
        include_images = (
            include_images if include_images is not None else self.include_images
        )
        include_full_ddf = (
            include_full_ddf if include_full_ddf is not None else self.include_full_ddf
        )
        metrics = {}
        figures = {}

        if self.include_mae_metrics:
            error_metrics = compute_mean_average_errors(
                gt_tracking_glob, pred_tracking_glob
            )
            metrics.update(
                {f"mae/{key}": value for key, value in error_metrics.items()}
            )

        if self.include_drift_metrics:
            drift_metrics = get_drift_metrics(gt_tracking_glob, pred_tracking_glob)
            drift_metrics["hausdorff"] = hausdorff_distance(
                gt_tracking_glob, pred_tracking_glob
            )
            metrics.update(
                {f"drift/{key}": value for key, value in drift_metrics.items()}
            )

        image_shape_hw = image_shape_hw if image_shape_hw is not None else self.image_shape_hw
        if calibration_matrix is not None and image_shape_hw is not None:
            ddf_metrics = get_ddf_metrics(
                pred_tracking_glob,
                pred_tracking_loc,
                gt_tracking_glob,
                gt_tracking_loc,
                calibration_matrix,
                image_shape_hw,
                mode="5pt-landmark",
            )
            metrics.update(
                {f"ddf/5pt-{key}": value for key, value in ddf_metrics.items()}
            )

            if include_full_ddf:
                ddf_metrics = get_ddf_metrics(
                    pred_tracking_glob,
                    pred_tracking_loc,
                    gt_tracking_glob,
                    gt_tracking_loc,
                    calibration_matrix,
                    image_shape_hw,
                    mode="all-pts",
                )
                metrics.update(
                    {f"ddf/all-pts-{key}": value for key, value in ddf_metrics.items()}
                )

        if include_images:
            fig = plot_pose_differences(pred_tracking_glob, gt_tracking_glob)
            figures[f"errors-global_example"] = fig
            fig = plot_pose_differences(pred_tracking_loc, gt_tracking_loc)
            figures[f"errors-local_example"] = fig

        self.scan_ids.append(scan_id)
        for key, value in metrics.items():
            self.metrics[key].append(value)

        return metrics, figures

    def set_current_pred_tracking_from_relative_pose_vector(
        self, pred_tracking, infer_global_tracking=True
    ):
        """Sets the predictions for the current sample from predicted transformations
        in relative pose vector format.

        Args:
            pred_tracking (np.ndarray): N-1 x 6 dimensional matrix of relative pose vectors
        """

        pred_tracking_glob, pred_tracking_loc = (
            get_global_and_relative_pred_trackings_from_vectors(pred_tracking)
        )

        if infer_global_tracking:
            self._current_update_cache["pred_tracking_glob"] = pred_tracking_glob

        self._current_update_cache["pred_tracking_loc"] = pred_tracking_loc
        return self

    def set_current_pred_tracking_from_global_pose_vectors(self, pred_tracking):
        """Sets the predictions for the current sample from predicted transformations in
        relative pose vector format.

        Args:
            pred_tracking (np.ndarray): (N - 1) x 6 dimensional matrix of global pose vectors
        """

        pred_tracking_glob, pred_tracking_loc = (
            get_global_and_relative_pred_trackings_from_vectors(
                pred_tracking, outputs_mode="global"
            )
        )

        self._current_update_cache["pred_tracking_glob"] = pred_tracking_glob
        return self

    def set_current_gt_tracking_from_world(
        self, gt_tracking_world, calibration_matrix=None
    ):
        """Sets the targets for the current sample from transformation matrices in
        world coordinates.

        Args:
            gt_tracking_world: N x 4 x 4 dimensional matrix of absolute position matrices. Position matrices
                are assumed to be in image coordinate system to world coordinate system.
            calibration_matrix: 4 x 4 calibration matrix mapping the pixel coordinate system (x y z 1)
                to the image coordinate system.
        """

        gt_tracking_glob, gt_tracking_loc = get_global_and_relative_gt_trackings(
            gt_tracking_world
        )
        self._current_update_cache["gt_tracking_glob"] = gt_tracking_glob
        self._current_update_cache["gt_tracking_loc"] = gt_tracking_loc
        self._current_update_cache["calibration_matrix"] = calibration_matrix
        return self

    def complete_update(self, *args, **kwargs):
        out = self(
            self._current_update_cache.get("scan_id", None),
            self._current_update_cache["gt_tracking_glob"],
            self._current_update_cache["pred_tracking_glob"],
            self._current_update_cache["gt_tracking_loc"],
            self._current_update_cache["pred_tracking_loc"],
            self._current_update_cache["calibration_matrix"],
            self.image_shape_hw,
            *args,
            **kwargs,
        )
        self._current_update_cache = {}
        return out

    def add_metric(self, key, value):
        self.metrics[key].append(value)

    def aggregate(self):
        metrics = {}

        for key, value in self.metrics.items():
            # value is a list. if we are in a distributed environment, we'll have to concatenate 
            # across processes before reducing. 
            if self._is_distributed: 
                # Convert to numpy array for convenience
                mean = global_mean(value)
            else: 
                mean = sum(value) / len(value)

            metrics[key] = mean  # average across scans

        if self._is_main_rank: 
            for key, value in self.images.items():
                metrics[key] = value

        return metrics
    


def global_mean(local_floats):
    """
    Compute global mean of a list of floats across all processes in DDP.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert local list to tensor
    local_tensor = torch.tensor(local_floats, dtype=torch.float64, device=device)
    
    # Local sum and count
    local_sum = torch.sum(local_tensor)
    local_count = torch.tensor([local_tensor.numel()], dtype=torch.float64, device=device)

    # Global sum and count
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    # Compute global mean
    global_mean = local_sum / local_count
    return global_mean.item()