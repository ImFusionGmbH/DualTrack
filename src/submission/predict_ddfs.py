# this is the function for generating DDF
import os
import json
from time import time
from omegaconf import OmegaConf
from .predictor import get_predictor
from .tus_rec_challenge_baseline.plot_functions import reference_image_points
from .tus_rec_challenge_baseline.Transf2DDFs import cal_global_ddfs, cal_local_ddfs
import torch
from src.utils.timer import timer


_predictor = None 


def predict_ddfs(frames, landmark, data_path_calib, device, config='dultrack_tus_rec_2025.yaml', convert_to_numpy=True):
    """
    Args:
        frames (numpy.ndarray): shape=(N, 480, 640),frames in the scan, where N is the number of frames in this scan
        landmark (numpy.ndarray): shape=(100,3), denoting the location of landmark. For example, a landmark with location of (10,200,100) denotes the landmark on the 10th frame, with coordinate of (200,100)
        data_path_calib (str): path to calibration matrix
        device (torch.device): device to run the model on, e.g., torch.device('cuda') or torch.device('cpu')

    Returns:
        pred_global_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), global DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame)
        pred_global_landmark_DDF (numpy.ndarray): shape=(3, 100), global DDF for landmark
        pred_local_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), local DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame)
        pred_local_landmark_DDF (numpy.ndarray): shape=(3, 100), local DDF for landmark

    """
    full_config_path = os.path.join(
        os.path.dirname(__file__), 'cfg', config
    )
    cfg = OmegaConf.load(full_config_path)

    global _predictor
    if _predictor is None:
        cfg.predictor.device = device
        _predictor = get_predictor(cfg.predictor, data_fmt='tus-rec-challenge')

    ddf_device = cfg.ddf_device

    # full_config_path = os.path.join(
    #     os.path.dirname(__file__), 'cfg', config
    # )
    # cfg = OmegaConf.load(full_config_path)
    # cfg.device = device
    #predictor = get_predictor(cfg, data_fmt='tus-rec-challenge')
    
    predictor = _predictor

    with timer('predictor run inference'):
        predictor.run_inference(frames)

    image_points = reference_image_points([480, 640], [480, 640])

    with timer('global DDF calculation'):
        pred_global_allpts_DDF, pred_global_landmark_DDF = cal_global_ddfs(
            torch.from_numpy(predictor.pred_tracking_matrices_glob).half().to(ddf_device, non_blocking=True),
            torch.from_numpy(predictor.pixel2img_matrix).half().to(ddf_device, non_blocking=True),
            image_points=image_points.to(ddf_device).half().to(ddf_device, non_blocking=True),
            landmark=torch.from_numpy(landmark).to(ddf_device, non_blocking=True),
        )

    with timer('local DDF calculation'):
        pred_local_allpts_DDF, pred_local_landmark_DDF = cal_local_ddfs(
            torch.from_numpy(predictor.pred_tracking_matrices_loc).half().to(ddf_device, non_blocking=True),
            torch.from_numpy(predictor.pixel2img_matrix).half().to(ddf_device, non_blocking=True),
            image_points=image_points.to(ddf_device).half().to(ddf_device, non_blocking=True),
            landmark=torch.from_numpy(landmark).to(ddf_device, non_blocking=True),
        )

    with timer('output transfer'):
        pred_global_allpts_DDF, pred_global_landmark_DDF, pred_local_allpts_DDF, pred_local_landmark_DDF = (
            pred_global_allpts_DDF.to('cpu', non_blocking=True), 
            pred_global_landmark_DDF.to('cpu', non_blocking=True),
            pred_local_allpts_DDF.to('cpu', non_blocking=True),
            pred_local_landmark_DDF.to('cpu', non_blocking=True)
        )

        if convert_to_numpy:
            pred_global_allpts_DDF = pred_global_allpts_DDF.numpy()
            pred_global_landmark_DDF = pred_global_landmark_DDF.numpy()
            pred_local_allpts_DDF = pred_local_allpts_DDF.numpy()
            pred_local_landmark_DDF = pred_local_landmark_DDF.numpy()

    return (
        pred_global_allpts_DDF, pred_global_landmark_DDF, pred_local_allpts_DDF, pred_local_landmark_DDF
    )
