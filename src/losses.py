import torch
from .dense_displacement_field import make_image_points
from .pose import pose_vector_to_rotation_matrix_torch
from torch import nn 


class TransformedImagePointsMSELoss:
    def __init__(self, image_shape_hw):
        self.image_shape_hw = image_shape_hw

        # n_points x 4 dimensions
        self.corner_points = make_image_points(
            self.image_shape_hw[0], self.image_shape_hw[1], "corners"
        )

    def __call__(self, input, target, px_to_image_matrix):
        """
        Computes the loss.

        Args:
            input (N x 6 dimensional torch.Tensor) - the predicted transformations as 6dof pose vector
            target - the target pose vector (same shape as input)
            calibration_matrix: the matrix that brings us from pixel coordinates to the coordinate system
                of the transforms (typically image coordinates)
        """

        points = (
            torch.tensor(self.corner_points, dtype=input.dtype, device=input.device)
        ).unsqueeze(0)
        # 1, 4 x n_points
        px_to_image_matrix = torch.tensor(
            px_to_image_matrix, dtype=input.dtype, device=input.device
        )  # B x 4 x 4
        points_img_coordinates = px_to_image_matrix @ points  # 1 x 4 x n_points

        pred_matrix = pose_vector_to_rotation_matrix_torch(input)  # B x 4 x 4
        true_matrix = pose_vector_to_rotation_matrix_torch(target)  # B x 4 x 4

        pred_points = pred_matrix @ points_img_coordinates  # B x 4 x n_points
        target_points = true_matrix @ points_img_coordinates  # B x 4 x n_points

        return torch.nn.functional.mse_loss(pred_points, target_points)


class PoseVectorMSELoss:
    """Wrapper of mse loss for API compatibility with the other losses."""

    def __call__(self, input, target, px_to_image_matrix):
        return torch.nn.functional.mse_loss(input, target)


class TrackingEstimationLoss(nn.Module): 
    """Supports 6dof loss calculation possibly with local/global predictions and padded inputs."""

    def forward(self, pred, targets, targets_global=None, targets_absolute=None, padding_size=None): 
        
        device = pred.device

        mse_loss = nn.MSELoss(reduction="none")

        def _get_loss(pred, targets):
            B, N, D = (
                pred.shape
                if isinstance(pred, torch.Tensor)
                else list(pred.values())[0].shape
            )

            loss = mse_loss(pred, targets)

            if padding_size is not None:
                mask = torch.ones(B, N, D, dtype=torch.bool, device=device)
                for i, padding_length in enumerate(padding_size):
                    if padding_length > 0:
                        mask[i, -padding_length:, :] = 0
                masked_loss = torch.where(mask, loss, torch.nan)
                mse_loss_val = masked_loss.nanmean()
                return mse_loss_val
            else: 
                return loss

        if isinstance(pred, dict):
            loss = torch.tensor(0.0, device=device)
            if "local" in pred:
                loss += _get_loss(pred["local"], targets)
            if "global" in pred:
                loss += _get_loss(pred["global"], targets_global.to(device))
            if "absolute" in pred:
                loss += _get_loss(pred["absolute"], targets_absolute.to(device))
            return loss
        else:
            return _get_loss(pred, targets)


