import multiprocessing as mp
import os
import shutil
import sys
from argparse import ArgumentParser
from copy import deepcopy

import h5py
import imfusion
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())


def main():
    parser = ArgumentParser(
        description="Script for preparing imfusion files to hdf5 files and corresponding metadata for this repo."
    )
    parser.add_argument(
        "--input_csv_file",
        "-i",
        help="Path to the input csv file. The file should have at least two columns: "
        "`sweep_id` and `raw_sweep_path` (path to the imfusion file of the sweep)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Path to the output directory where data will be saved",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of parallel processes to use.",
    )
    parser.add_argument(
        "--max_cases",
        default=None,
        type=int,
        help="Limits numbers of cases to be processed - good for debugging.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="If set, just print the names of the output files",
    )
    parser.add_argument(
        "--smooth_tracking",
        action="store_true",
        help="If set, will smooth the tracking sequences stored in the imfusion files.",
    )
    parser.add_argument(
        "--auto_calibrate_to_increasing_z",
        action="store_true",
        help="Will calibrate the sweep to make sure the z value is increasing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, will force-overwrite existing previously saved files.",
    )
    args = parser.parse_args()
    print(args)

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    shutil.copy(__file__, os.path.join(args.output_dir, "script.py"))
    import json

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)

    sweep_metadata = pd.read_csv(args.input_csv_file)
    if args.max_cases:
        sweep_metadata = sweep_metadata.head(args.max_cases)

    with mp.Manager() as manager:
        counter = manager.Value("i", 0)
        lock = manager.Lock()

        with mp.Pool(args.num_workers) as pool:
            outputs = pool.map(
                Processor(
                    args,
                    lock,
                    counter,
                    len(sweep_metadata),
                ),
                sweep_metadata.iloc,  # type:ignore
            )
            outputs = [output for output in outputs if output is not None]

            frame_metadata = pd.DataFrame(outputs)
            frame_metadata.to_csv(os.path.join(args.output_dir, "metadata.csv"))


def imfusion_sweep_to_h5(
    input_filepath,
    output_filepath,
    smooth_tracking=True,
    temporal_calibration=None,
    auto_calibrate_to_increasing_z=False,
) -> bool:

    try:
        (sweep,) = imfusion.load(input_filepath)
        if smooth_tracking:
            imfusion.execute_algorithm(
                "US.SweepProperties",
                [sweep],
                {
                    "maxTimestepBetweenSamples": 0,
                    "trackingFilterMode": 2,
                    "trackingFilterSize": 5,
                    "useTimestamps": 1,
                    "execute": 1,
                },
            )

        if temporal_calibration is not None:
            imfusion.execute_algorithm(
                "US.SweepProperties",
                [sweep],
                {
                    "maxTimestepBetweenSamples": 0,
                    "temporal": temporal_calibration,
                    "useTimestamps": 1,
                    "execute": 1,
                },
            )

        with h5py.File(output_filepath, "w") as F:
            N, _, H, W, _ = sweep.shape
            images = np.zeros((N, H, W), dtype=np.uint8)
            tracking = np.zeros((N, 4, 4), dtype=np.float32)

            for i in range(N):
                images[i] = np.array(sweep[i])[:, :, 0]
                tracking[i] = sweep.matrix(i)

            if auto_calibrate_to_increasing_z:
                # check if Z is increasing
                global_tracking = invert_pose_matrix(tracking[0])[None, ...] @ tracking
                trackings_vector = matrix_to_pose_vector(global_tracking)
                delta_z = trackings_vector[-1, 2] - trackings_vector[0, 2]
                if delta_z < 0:
                    # should be calibrated!

                    # equivalent to rotating the tracking chassis around the probe handle 180 deg.
                    rot = pose_vector_to_matrix(np.array([0, 0, 0, 0, 180, 0]))
                    # rot = np.eye(4)
                    # rot[2, 2] = -1

                    images = np.flip(images, axis=2)  # flip along x direction

                    tracking_rot = tracking @ rot[None, ...]
                    tracking = tracking_rot
                    # tracking = np.flip(tracking, axis=0)

            F.create_dataset("images", data=images)
            F.create_dataset("tracking", data=tracking)
            F.create_dataset("spacing", data=sweep.descriptor().spacing)
            F.create_dataset("dimensions", data=sweep.descriptor().dimensions)
            F.create_dataset("pixel_to_image", data=sweep.get().pixel_to_world_matrix)
            return True

    except Exception as e:
        print(e)
        return False


class Processor:
    def __init__(
        self,
        args,
        lock,
        counter,
        total,
        verbose=False,
    ):
        self.args = args
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.lock = lock
        self.counter = counter
        self.total = total
        self.verbose = verbose

    def __call__(self, row):
        row = row.to_dict()
        sweep_id = row["sweep_id"]
        sweep_path = row["raw_sweep_path"]
        temporal_calibration = row.get("temporal_calibration", None)
        smooth_tracking = row.get("smooth_tracking", self.args.smooth_tracking)

        output_path = os.path.join(self.args.output_dir, f"{sweep_id}.h5")

        if not os.path.exists(output_path) or self.args.force:
            if not self.args.dryrun:
                success = imfusion_sweep_to_h5(
                    sweep_path,
                    output_path,
                    smooth_tracking,
                    temporal_calibration,
                    self.args.auto_calibrate_to_increasing_z,
                    # auto_reverse_sweep=self.args.auto_reverse_sweep,
                )
            else:
                success = True
                print(f"Dry run: saving {output_path}")

            if not success:
                print(f"Warning - {output_path} unsuccessful")
                return None

        output_data_info = deepcopy(row)
        output_data_info["processed_sweep_path"] = output_path

        with self.lock:
            self.counter.value += 1
            print(f"Processed {self.counter.value}/{self.total}")

        return output_data_info

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


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


if __name__ == "__main__":
    main()
