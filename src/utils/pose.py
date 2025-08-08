"""
Functions for handling tracking sequences. 

This module includes many utility functions for handling tracking sequences, including 
format conversion and metric calculation functions for pose estimation.

Tracking sequences consist of a sequence of poses. We use two formats for storing these: 
    - 4x4 matrices 
    - 6DOF vector x, y, z, theta_x, theta_y, theta_z (Euler angles)

We use a few naming conventions for tracking sequences: 
    - "World" the tracking sequence consists of absolute positions, in an arbitrary world coordinate system (T_0, T_1, ..., T_N)
    - "Global" the tracking sequence consists of absolute positions relative to the position of the first timestep (T_0^-1 @ T_0, T_0^-1 @ T_1, ..., T_0^-1 @ T_N)
    - "Local" the tracking sequence consists of the relative positions between consecutive timesteps (T_0^-1 @ T_1, T_1^-1 @ T_2, ..., T_{N-1}^-1 @ T_N)

*Note: Even though the "Local" tracking format should be 1 timestep shorter than "world" and "global" format, sometimes it is padded at the begginning with the identity 
    transform to match their length.
"""

from copy import deepcopy
from typing import Literal

import numpy as np
import scipy
import scipy.spatial
import torch
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R


def get_ddf_metrics(
    pred_tracking_glob,
    pred_tracking_loc,
    gt_tracking_glob,
    gt_tracking_loc,
    calibration_matrix,
    im_shape,
    mode: Literal["all-pts", "5pt-landmark"] = "all-pts",
):
    """
    Computes all dense displacement field metrics.

    Args:
        pred_tracking_glob: predicted tracking in global format (N x 4 x 4) torch tensor
        pred_tracking_loc: predicted tracking in local format (N x 4 x 4) torch tensor
        gt_tracking_glob: ground truth tracking in global format (N x 4 x 4) torch tensor
        gt_tracking_loc: ground truth tracking in local format (N x 4 x 4) torch tensor
        calibration_matrix: the calibration matrix which maps us from the pixel coordinate system to the
            image coordinate system, where the transforms are predicted.
        im_shape: tuple of image shape (height, width).
        mode: mode for dense displacement field calculation, either "all-pts" or "5pt-landmark".
            "all-pts" means the dense displacement field error is based on every pixel in the image.
            "5pt-landmark" means the dense displacement field error is based on the 4 corner points and the center.
            we mostly use "5pt-landmark" because it is much faster to compute, although slightly less accurate.
    """

    H, W = im_shape
    metrics = {}

    # 1. Compute DDF
    # permutation = (
    #     get_xy_permutation_matrix()
    # )  # needed because imfusion points are internally stored in (y, x, 0, 1) not (x, y, 0, 1) format
    #

    from .. import dense_displacement_field

    points_list = dense_displacement_field.make_image_points(H, W, mode=mode)

    pred_tracking_glob = torch.tensor(pred_tracking_glob).float()
    gt_tracking_glob = torch.tensor(gt_tracking_glob).float()
    pred_tracking_loc = torch.tensor(pred_tracking_loc).float()
    gt_tracking_local = torch.tensor(gt_tracking_loc).float()
    calibration_matrix = torch.tensor(calibration_matrix).float()
    # permutation = torch.tensor(permutation).float()
    points_list = torch.tensor(points_list).float()

    pred_global_ddf = dense_displacement_field.cal_global_allpts(
        pred_tracking_glob[1:], calibration_matrix, points_list
    )
    gt_global_ddf = dense_displacement_field.cal_global_allpts(
        gt_tracking_glob[1:], calibration_matrix, points_list
    )
    pred_local_ddf = dense_displacement_field.cal_local_allpts(
        pred_tracking_loc[1:], calibration_matrix, points_list
    )
    gt_local_ddf = dense_displacement_field.cal_local_allpts(
        gt_tracking_local[1:], calibration_matrix, points_list
    )
    # 2. Compute error
    global_err = (((pred_global_ddf - gt_global_ddf) ** 2).sum(1) ** 0.5).mean(-1)
    local_err = (((pred_local_ddf - gt_local_ddf) ** 2).sum(1) ** 0.5).mean(-1)

    metrics["avg_global_displacement_error"] = global_err.mean().item()
    metrics["max_global_dislacement_error"] = global_err.max().item()
    metrics["avg_local_displacement_error"] = local_err.mean().item()
    metrics["max_local_displacement_error"] = local_err.max().item()

    relative_global_err = ((((pred_global_ddf - gt_global_ddf) ** 2).sum(1) ** 0.5) / (((gt_global_ddf) ** 2).sum(1) ** 0.5)).mean(-1)
    metrics["relative_global_err_pct"] = relative_global_err.mean().item()

    return metrics


def compute_mean_average_errors(gt_tracking, pred_tracking):
    """
    Computes mean average error for each degree of freedom across the predicted and target tracking.

    Args:
        pred_tracking: predicted tracking in global format (N x 4 x 4) torch tensor
        gt_tracking: ground truth tracking in global format (N x 4 x 4) torch tensor
    """

    N = len(gt_tracking)
    gt_as_vector = np.zeros((N, 6))
    pred_as_vector = np.zeros((N, 6))

    for i in range(N):
        gt_as_vector[i] = matrix_to_pose_vector(gt_tracking[i])
        pred_as_vector[i] = matrix_to_pose_vector(pred_tracking[i])

    names = ["x", "y", "z", "pitch", "roll", "yaw"]
    outputs = {}

    for i in range(6):
        mae = np.abs(gt_as_vector[:, i] - pred_as_vector[:, i]).mean()
        outputs[names[i]] = mae

    return outputs


def get_drift(gt_tracking, pred_tracking):
    translation_pred = matrix_to_pose_vector(pred_tracking)[:, :3]
    translation_gt = matrix_to_pose_vector(gt_tracking)[:, :3]

    return np.sqrt(((translation_pred - translation_gt) ** 2).sum(-1))


def dist_from_origin(tracking):
    tracking = invert_pose_matrix(tracking[0])[None, ...] @ tracking
    dist = matrix_to_pose_vector(tracking)[:, :3]
    dist = np.sqrt((dist**2).sum(-1))
    return dist


def get_drift_rate(gt_tracking, pred_tracking):
    return (
        (get_drift(gt_tracking, pred_tracking)) / (dist_from_origin(gt_tracking)) * 100
    )


def get_drift_metrics(gt_tracking, pred_tracking):
    drift_ = get_drift(gt_tracking, pred_tracking)
    drift_rate = get_drift_rate(gt_tracking, pred_tracking)

    metrics = {}
    metrics["final_drift_rate"] = drift_rate[-1]
    metrics["avg_drift_rate"] = drift_rate[~np.isnan(drift_rate)].mean()
    metrics["max_drift"] = np.max(drift_)
    metrics["sum_of_drift"] = np.sum(drift_)

    return metrics


def hausdorff_distance(gt_tracking, pred_tracking):
    u = matrix_to_pose_vector(gt_tracking)[:, :3]
    v = matrix_to_pose_vector(pred_tracking)[:, :3]
    return max(
        scipy.spatial.distance.directed_hausdorff(u, v)[0],
        scipy.spatial.distance.directed_hausdorff(v, u)[0],
    )


def plot_pose_differences(pred, gt, ax=None):
    pred_tracking = np.stack([matrix_to_pose_vector(matrix) for matrix in pred])
    gt_tracking = np.stack([matrix_to_pose_vector(matrix) for matrix in gt])

    errors = np.abs(pred_tracking - gt_tracking).mean(0)

    if ax is None: 
        fig, ax = plt.subplots(2, 3)
    else: 
        fig = plt.gcf()

    for i in range(6):
        ax_ = ax.flatten()[i]
        tags = ["x", "y", "z", "pitch", "roll", "yaw"]

        ax_.plot(pred_tracking[:, i], label="pred", alpha=0.8)
        ax_.plot(gt_tracking[:, i], label="gt", alpha=0.8)
        ax_.set_title(f"mae={errors[i]:.2f}")

        if i <= 2:
            ax_.set_ylabel(f"{tags[i]} (mm)")
        else:
            ax_.set_ylabel(f"{tags[i]} (°)")
        ax_.set_xlabel(f"timestep")

        if i == 5:
            ax_.legend()

    fig.tight_layout()
    return fig


def get_xy_permutation_matrix():
    permutation = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return permutation


def get_relative_pose(t_start, t_end):
    return invert_pose_matrix(t_start) @ t_end


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


def matrix_to_pose_vector(pose_matrix):
    translation = pose_matrix[..., :3, 3]
    rotation_matrix = pose_matrix[..., :3, :3]
    rotation = R.from_matrix(rotation_matrix)

    # Get Euler angles (ZYX order)
    euler_angles = rotation.as_euler("xyz", degrees=True)  # Returns angles in degrees

    # print("Translation:", translation)
    # print("Yaw (ψ), Pitch (θ), Roll (φ):", euler_angles)

    result = np.concatenate((translation, euler_angles), axis=-1)
    return result


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


def euler_angles_to_rotation_matrix_torch(rot_angles, degrees=True):
    """
    Differentiably convert the provided rotation angles to a rotation matrix.

    Based on ImFusion::Pose::eulerToMat, which is equivalent to
    scipy.spatial.transform.Rotation.from_euler('xyz', rot_angles_radians, degrees)
    """

    *dims, D = rot_angles.shape
    assert D == 3

    if degrees:
        rot_angles_radians = torch.deg2rad(rot_angles)
    else:
        rot_angles_radians = rot_angles

    x, y, z = rot_angles_radians.unbind(-1)

    cx = x.cos()
    sx = x.sin()
    cy = y.cos()
    sy = y.sin()
    cz = z.cos()
    sz = z.sin()

    return torch.stack(
        [
            cy * cz,
            cz * sx * sy - cx * sz,
            sx * sz + cx * cz * sy,
            cy * sz,
            sx * sy * sz + cx * cz,
            cx * sy * sz - cz * sx,
            -sy,
            cy * sx,
            cx * cy,
        ],
        dim=-1,
    ).reshape(*dims, 3, 3)


def pose_vector_to_rotation_matrix_torch(pose_vector, degrees=True):
    *dims, D = pose_vector.shape
    if not D == 6:
        raise ValueError("Incorrect shape of pose vector")

    out = torch.zeros([*dims, 4, 4], device=pose_vector.device)

    translation = pose_vector[..., :3]
    euler_angles = pose_vector[..., 3:]
    rotation_matrix = euler_angles_to_rotation_matrix_torch(euler_angles, degrees)
    out[..., :3, :3] = rotation_matrix
    out[..., :3, 3] = translation
    out[..., 3, 3] = 1

    return out


def get_global_and_relative_gt_trackings(gt_tracking_world_matrices):
    N = len(gt_tracking_world_matrices)

    # first make global tracking relative to image coords
    gt_tracking_glob = (
        invert_pose_matrix(gt_tracking_world_matrices[0])[None, ...]
        @ gt_tracking_world_matrices
    )

    # Generate relative gt trackings
    gt_tracking_local = np.zeros((N, 4, 4), dtype=np.float32)
    gt_tracking_local[0] = np.eye(4)

    for i in range(N - 1):
        gt_tracking_local[i + 1] = get_relative_pose(
            gt_tracking_glob[i], gt_tracking_glob[i + 1]
        )

    return gt_tracking_glob, gt_tracking_local


def get_relative_tracking_as_pose_vectors(gt_tracking_world_matrices):
    N = len(gt_tracking_world_matrices)
    relative_tracking = np.zeros((N - 1, 6))
    for i in range(N - 1):
        relative_tracking[i] = matrix_to_pose_vector(
            invert_pose_matrix(gt_tracking_world_matrices[i])
            @ gt_tracking_world_matrices[i + 1]
        )
    return relative_tracking


def get_global_and_relative_pred_trackings_from_vectors(
    model_outputs, outputs_mode="local"
):
    """Computes the global and relative tracking matrix sequence from a sequence of 6-d pose vectors"""

    N = len(model_outputs)

    # Generate predicted global and local trackings
    pred_tracking_glob = np.zeros((N + 1, 4, 4), dtype=np.float32)
    pred_tracking_glob[0] = np.eye(4)
    pred_tracking_loc = np.zeros((N + 1, 4, 4), dtype=np.float32)
    pred_tracking_loc[0] = np.eye(4)

    for i in range(N):
        if outputs_mode == "local":
            pred_rel_pose_matrix = pose_vector_to_matrix(model_outputs[i])
            pred_tracking_loc[i + 1] = pred_rel_pose_matrix
            pred_tracking_glob[i + 1] = pred_tracking_glob[i] @ pred_rel_pose_matrix
        else:
            pred_glob_pose_matrix = pose_vector_to_matrix(model_outputs[i])
            pred_tracking_loc[i + 1] = get_relative_pose(
                pred_tracking_glob[i], pred_glob_pose_matrix
            )
            pred_tracking_glob[i + 1] = pred_glob_pose_matrix

    return pred_tracking_glob, pred_tracking_loc


def cumulative_matrix_product_torch(matrices: torch.Tensor):
    *_, L, M, N = matrices.shape
    assert M == N, f"This only works with square matrices"

    output = torch.zeros_like(matrices)
    for i in range(L):
        if i == 0:
            output[..., i, :, :] = matrices[..., i, :, :]
        else:
            output[..., i, :, :] = torch.matmul(
                matrices[..., i, :, :], output[..., i - 1, :, :]
            )

    return output


def get_relative_transforms_torch(matrices: torch.Tensor, keepsize=False):
    *dims, L, M, N = matrices.shape

    output = torch.zeros([*dims, L - 1, M, N], dtype=float, device=matrices.device)
    for i in range(L - 1):
        output[..., i, :, :] = torch.matmul(
            torch.linalg.inv(matrices[..., i, :, :]), matrices[..., i + 1, :, :]
        )

    if keepsize:
        output = torch.cat(
            [
                torch.eye(4, device=matrices.device).expand_as(output[..., [0], :, :]),
                output,
            ],
            dim=-3,
        )

    return output


def rotation_matrix_to_euler_angles_torch(rotation_matrix, degrees=True):
    """
    Convert a rotation matrix to Euler angles (xyz order) in a differentiable way.
    """
    sy = torch.clamp(
        -rotation_matrix[..., 2, 0], -1.0, 1.0
    )  # Ensure within valid range

    singular = torch.abs(sy) == 1.0

    x = torch.where(
        singular,
        torch.atan2(-rotation_matrix[..., 0, 1], rotation_matrix[..., 1, 1]),
        torch.atan2(rotation_matrix[..., 2, 1], rotation_matrix[..., 2, 2]),
    )
    y = torch.asin(sy)
    z = torch.where(
        singular,
        torch.zeros_like(x),
        torch.atan2(rotation_matrix[..., 1, 0], rotation_matrix[..., 0, 0]),
    )

    euler_angles = torch.stack((x, y, z), dim=-1)

    if degrees:
        euler_angles = torch.rad2deg(euler_angles)

    return euler_angles


def rotation_matrix_to_pose_vector_torch(matrix, degrees=True):
    """
    Convert a 4x4 transformation matrix to a 6D pose vector (translation + Euler angles in xyz order).
    """
    if matrix.shape[-2:] != (4, 4):
        raise ValueError("Input matrix must have shape (..., 4, 4)")

    translation = matrix[..., :3, 3]
    rotation_matrix = matrix[..., :3, :3]
    euler_angles = rotation_matrix_to_euler_angles_torch(rotation_matrix, degrees)

    return torch.cat([translation, euler_angles], dim=-1)


def get_absolute_to_global_transforms_torch(rotation_matrices):
    ref = rotation_matrices[..., [0], :, :]
    return torch.linalg.inv(ref) @ rotation_matrices