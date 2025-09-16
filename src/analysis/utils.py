import math
import random
from matplotlib import pyplot as plt
from scipy import linalg
from src.utils.pose import matrix_to_pose_vector
import numpy as np


def plot_poses_single_parameter(parameter, trackings, ax=None):
    ax = ax or plt.gca()

    trackings = {
        k: np.stack([matrix_to_pose_vector(matrix) for matrix in v]) 
        for k, v in trackings.items()
    }

    tags = ["x", "y", "z", "pitch", "roll", "yaw"]
    i = tags.index(parameter)

    for name, tracking in trackings.items(): 
        if 'gt' in name or 'truth' in name:
            ax.plot(tracking[:, i], label='Ground Truth', linestyle='--', color='black', alpha=0.5)
        else:
            ax.plot(tracking[:, i], label=name, alpha=0.8)
    
    # ax_.set_title(f"mae={errors[i]:.2f}")
    if i <= 2:
        ax.set_ylabel(f"{tags[i]} (mm)")
    else:
        ax.set_ylabel(f"{tags[i]} (°)")
    ax.set_xlabel(f"timestep")


def plot_poses(trackings):
    trackings = {
        k: np.stack([matrix_to_pose_vector(matrix) for matrix in v]) 
        for k, v in trackings.items()
    }

    fig, ax = plt.subplots(2, 3)
    for i in range(6):
        ax_ = ax.flatten()[i]
        tags = ["x", "y", "z", "pitch", "roll", "yaw"]

        for name, tracking in trackings.items(): 
            if 'gt' in name or 'truth' in name:
                ...
                ax_.plot(tracking[:, i], label='Ground Truth', linestyle='--', color='black', alpha=0.5)
            else:
                ax_.plot(tracking[:, i], label=name, alpha=0.8)
        
        #ax_.set_title(f"mae={errors[i]:.2f}")
        if i <= 2:
            ax_.set_ylabel(f"{tags[i]} (mm)")
        else:
            ax_.set_ylabel(f"{tags[i]} (°)")
        ax_.set_xlabel(f"timestep")

        if i == 5:
            ax_.legend()

        ax_.set_xticks([])

    # access legend objects automatically created from data
    handles, labels = plt.gca().get_legend_handles_labels()

    #fig.tight_layout()
    return fig, ax


def plot_poses_norm_error(trackings, gt_tracking): 
    gt_tracking = _convert_to_6dof(gt_tracking)[0][0]
    trackings = _convert_to_6dof(**trackings)[1]

    fig, ax = plt.subplots(1, 1)
    for name, tracking in trackings.items(): 
        norm_diff = np.linalg.norm(tracking[:, :3] - gt_tracking[:, :3], ord=2, axis=-1)
        ax.plot(norm_diff, label='name')


def _convert_to_6dof(*trackings, **trackings_map): 

    def _convert(v): 
        if v[0].shape[-1] == 6: 
            return v 
        else: 
            return np.stack([matrix_to_pose_vector(matrix) for matrix in v])            

    trackings = [_convert(v) for v in trackings]
    trackings_map = {
        k: _convert(v) 
        for k, v in trackings_map.items()
    }
    return trackings, trackings_map


def get_random_us_sweep_as_numpy_array(dataset='tus-rec'):
    from src.datasets.sweeps_dataset_v2 import SweepsDataset
    ds = SweepsDataset(dataset)
    i = random.randint(0, len(ds) - 1)
    return ds[i]['images']