# copy paste from https://github.com/QiLi111/tus-rec-challenge_baseline/blob/main/submission/utils/Transf2DDFs.py


# This script contains four functions, which can be used to generate required DDFs, using transformations
from typing import Literal
import torch
import numpy as np


def cal_global_allpts(transformation_global, tform_calib_scale, image_points):
    """
    This function generates global DDF for all pixels in a scan, using global transformations

    Args:
        transformation_global (torch.Tensor): shape=(N-1, 4, 4), global transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the first frame
        tform_calib_scale (torch.Tensor): shape=(4, 4), scale from image coordinate system (in pixel) to image coordinate system (in mm)
        image_points (torch.Tensor): shape=(4, 307200), point coordinate for all pixels, in image coordinate system (in pixel)

    Returns:
        global_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), global DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame)

    """
    # coordinates of points in current frame, with respect to the first frame
    global_allpts = torch.matmul(
        transformation_global, torch.matmul(tform_calib_scale, image_points)
    )
    # calculate DDF in mm, displacement from current frame to the first frame
    global_allpts_DDF = global_allpts[:, 0:3, :] - torch.matmul(
        tform_calib_scale, image_points
    )[0:3, :].expand(global_allpts.shape[0], -1, -1)

    return global_allpts_DDF


def cal_global_landmark(transformation_global, landmark, tform_calib_scale):
    """
    This function generates global DDF for landmark, using global transformations

    Args:
        transformation_global (torch.Tensor): shape=(N-1, 4, 4), global transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the first frame, where N-1 is the number of frames in that scan (excluding the first frame)
        landmark (torch.Tensor): shape=(20, 3), coordinates of landmark points in image coordinate system (in pixel)
        tform_calib_scale (torch.Tensor): shape=(4, 4), scale from image coordinate system (in pixel) to image coordinate system (in mm)

    Returns:
       global_landmark_DDF (numpy.ndarray): shape=(3, 20), global DDF for landmark
    """

    global_landmark = torch.zeros(3, len(landmark))
    for i in range(len(landmark)):
        # point coordinate in image coordinate system (in pixel)
        pts_coord = torch.cat(
            (landmark[i][1:], torch.FloatTensor([0, 1])), axis=0
        )
        # calculate global DDF in mm, displacement from current frame to the first frame
        global_landmark[:, i] = (
            torch.matmul(
                transformation_global[landmark[i][0] - 1],
                torch.matmul(tform_calib_scale, pts_coord),
            )[0:3]
            - torch.matmul(tform_calib_scale, pts_coord)[0:3]
        )

    global_landmark_DDF = global_landmark.cpu().numpy()

    return global_landmark_DDF


def cal_local_allpts(transformation_local, tform_calib_scale, image_points):
    """
    This function generates local DDF for all pixels in a scan, using local transformations

    Args:
        transformation_local (torch.Tensor): shape=(N-1, 4, 4), local transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the previous frame
        tform_calib_scale (torch.Tensor): shape=(4, 4), scale from image coordinate system (in pixel) to image coordinate system (in mm)
        image_points (torch.Tensor): shape=(4, 307200), point coordinate for all pixels, in image coordinate system (in pixel)

    Returns:
        local_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), local DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame)
    """

    # coordinates of points in current frame, with respect to the immediately previous frame
    local_allpts = torch.matmul(
        transformation_local, torch.matmul(tform_calib_scale, image_points)
    )
    # calculate DDF in mm, displacement from current frame to the immediately previous frame
    local_allpts_DDF = local_allpts[:, 0:3, :] - torch.matmul(
        tform_calib_scale, image_points
    )[0:3, :].expand(local_allpts.shape[0], -1, -1)

    return local_allpts_DDF


def cal_local_landmark(transformation_local, landmark, tform_calib_scale):
    """
    This function generates local DDF for landmark in a scan, using local transformations

    Args:
        transformation_local (torch.Tensor): shape=(N-1, 4, 4), local transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the previous frame, where N-1 is the number of frames in that scan (excluding the first frame)
        landmark (torch.Tensor): shape=(20, 3), coordinates of landmark points in image coordinate system (in pixel)
        tform_calib_scale (torch.Tensor): shape=(4, 4), scale from image coordinate system (in pixel) to image coordinate system (in mm)

    Returns:
        local_landmark_DDF (numpy.ndarray): shape=(3, 20), local DDF for landmarks
    """

    local_landmark = torch.zeros(3, len(landmark))
    for i in range(len(landmark)):
        # point coordinate in image coordinate system (in pixel)
        pts_coord = torch.cat(
            (landmark[i][1:], torch.FloatTensor([0, 1])), axis=0  # type:ignore
        )
        # calculate DDF in mm, displacement from current frame to the immediately previous frame
        local_landmark[:, i] = (
            torch.matmul(
                transformation_local[landmark[i][0] - 1],
                torch.matmul(tform_calib_scale, pts_coord),
            )[0:3]
            - torch.matmul(tform_calib_scale, pts_coord)[0:3]
        )

    local_landmark_DDF = local_landmark.cpu().numpy()

    return local_landmark_DDF


def make_image_points(
    H, W, mode: Literal["all-pts", "5pt-landmark", "corners"] = "all-pts"
):
    """
    Makes the array of pixel coordinates (in 4-d homogeneous format) for
    the image of the provided height and width.
    """

    if mode == "all-pts":
        ind = np.indices((W, H)).transpose(1, 2, 0).reshape(-1, 2)
    elif mode == "5pt-landmark":
        ind = np.array(
            [
                [0, 0],
                [W, 0],
                [0, H],
                [W, H],
                [W // 2, H // 2],
            ]
        )
    elif mode == "corners":
        ind = np.array([[0, 0], [W, 0], [0, H], [W, H]])
    else:
        raise ValueError(f"Unexpected mode {mode}")

    extra_coords = np.array([[0, 1]]).repeat(ind.shape[0], 0)
    ind = np.concatenate([ind, extra_coords], axis=-1)

    return ind.T


def make_example_landmark_points(
    N, H, W
):  

    WH_values = torch.tensor(make_image_points(H, W, mode='5pt-landmark')).long() # 5 x 4
    WH_values = WH_values[:2, :].T # 5 x 2

    t_values = torch.arange(0, N, (N // 4 + 1))
    TWH_values = []
    for i in range(len(t_values)):

        t = t_values[[i]].repeat_interleave(5, 0)[..., None]
        TWH_values.append(torch.cat([t, WH_values], dim=-1))

    TWH_values = torch.cat(TWH_values, dim=0)

    return TWH_values


if __name__ == "__main__": 
    print(make_example_landmark_points(10, 4, 4))
