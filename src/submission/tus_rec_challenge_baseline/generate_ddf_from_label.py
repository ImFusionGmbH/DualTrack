# Functions used during ground truth DDF generation

import torch
from .transform import LabelTransform
from .plot_functions import (
    reference_image_points,
    read_calib_matrices,
    data_pairs_global,
    data_pairs_local,
)
from .Transf2DDFs import cal_global_ddfs, cal_local_ddfs


class generate_ddf_from_label:

    def __init__(self, data_path_calib, device, w=640, h=480):
        self.device = device
        # load calibration matrix
        self.tform_calib_scale, self.tform_calib_R_T, self.tform_calib = (
            read_calib_matrices(data_path_calib)
        )
        self.tform_calib_scale, self.tform_calib_R_T, self.tform_calib = (
            self.tform_calib_scale.to(self.device),
            self.tform_calib_R_T.to(self.device),
            self.tform_calib.to(self.device),
        )
        # image points coordinates in image coordinate system (in pixel), all pixel points
        self.image_points = reference_image_points([h, w], [h, w]).to(self.device)

    def calculate_GT_DDF(self, frames, tforms, landmark):
        # calculate DDFs of ground truth - label
        frames, tforms = (
            torch.tensor(t).to(self.device)[None, ...] for t in [frames, tforms]
        )
        tforms_inv = torch.linalg.inv(tforms)
        landmark = torch.from_numpy(landmark)

        # generate global and local transformations, based on recorded transformations (from tracker space to camera space) from NDI tracker.
        transformation_global, transformation_local = (
            self.get_global_local_transformations(tforms, tforms_inv)
        )

        # Global displacement vectors for pixel reconstruction and landmark reconstruction
        labels_global_allpts_DDF, labels_global_landmark_DDF = cal_global_ddfs(
            transformation_global.cpu(),
            self.tform_calib_scale.cpu(),
            self.image_points.cpu(),
            landmark,
        )
        # Local displacement vectors for pixel reconstruction and landmark reconstruction
        labels_local_allpts_DDF, labels_local_landmark_DDF = cal_local_ddfs(
            transformation_local.cpu(),
            self.tform_calib_scale.cpu(),
            self.image_points.cpu(),
            landmark,
        )

        return (
            labels_global_allpts_DDF,
            labels_global_landmark_DDF,
            labels_local_allpts_DDF,
            labels_local_landmark_DDF,
        )

    def get_global_local_transformations(self, tforms, tforms_inv):
        """
        This function generates global and local transformations for each frame in the scan

        Args:
            tforms (torch.Tensor): shape=(1, N, 4, 4), transformations from NDI tracker, from tracker tool space to camera space; where N is the number of frames in the scan.
            tforms_inv (torch.Tensor): shape=(1, N, 4, 4), inverse of tforms

        Returns:
            transformation_global (torch.Tensor): shape=(N-1, 4, 4), transformations from the current frame to the first frame, where N-1 is the number of frames in that scan (excluding the first frame)
            transformation_local (torch.Tensor): shape=(N-1, 4, 4), transformations from the current frame to the previous frame
        """

        data_pairs_all = data_pairs_global(tforms.shape[1])[1:, :]
        # convert transformations (from tracker space to camera space) to transformations (from the current frame to the first frame).
        transform_label_global_all = LabelTransform(
            "transform",
            pairs=data_pairs_all,
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T,
            tform_image_pixel_to_mm=self.tform_calib_scale,
        )
        transformation_global = torch.squeeze(
            transform_label_global_all(tforms, tforms_inv)
        )

        # convert transformations (from tracker space to camera space) to transformations (from the current frame to the immediate previous frame).
        data_pairs_local_all = data_pairs_local(tforms.shape[1] - 1)
        transform_label_local_all = LabelTransform(
            "transform",
            pairs=data_pairs_local_all,
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T,
            tform_image_pixel_to_mm=self.tform_calib_scale,
        )
        transformation_local = torch.squeeze(
            transform_label_local_all(tforms, tforms_inv)
        )

        return transformation_global, transformation_local
