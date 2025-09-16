import os
from random import choice
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.datasets import SweepsDataset
from src.pose import (
    get_global_and_relative_gt_trackings,
    get_relative_tracking_as_pose_vectors,
    matrix_to_pose_vector,
)


def main():
    parser = ArgumentParser()
    parser.add_argument("metadata_csv_path")
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Directory to save information about the dataset",
        default="dataset_info",
    )
    parser.add_argument("--split", choices=('train', 'val', 'all'), default='train')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # only compute statistics for the training set (mean, std) - this is best practice to make sure that
    # our model really generalizes without knowing test data statistics a priori
    dataset = SweepsDataset(metadata_csv_path=args.metadata_csv_path, split="all")
    per_sweep_info = []

    stats = defaultdict(list)

    for item in tqdm(dataset, desc="Calculating statistics"):
        per_sweep_info_i = {}

        tracking = item["tracking"][:]

        gt_tracking_global, gt_tracking_local = get_global_and_relative_gt_trackings(
            tracking
        )
        gt_tracking_global_as_vector = matrix_to_pose_vector(gt_tracking_global)

        for idx, name in enumerate(["x", "y", "z", "pitch", "roll", "yaw"]):
            per_sweep_info_i[f"delta_{name}"] = (
                gt_tracking_global_as_vector[-1][idx]
                - gt_tracking_global_as_vector[0][idx]
            )

        poses = get_relative_tracking_as_pose_vectors(tracking)
        stats["tracking"].append(poses)
        images = item["img"][:] / 255.0
        stats["img_mean"].append(images.mean())
        stats["img_std"].append(images.std())

        per_sweep_info.append(per_sweep_info_i)

    print(f"Collected stats for {len(dataset)} items.")

    outputs = {}
    pixel_mean = sum(stats["img_mean"]) / len(stats["img_std"])
    outputs["pixel_mean"] = float(pixel_mean)
    pixel_std = sum(stats["img_std"]) / len(stats["img_std"])
    outputs["pixel_std"] = float(pixel_std)
    tracking = np.concatenate(stats["tracking"])
    outputs["tracking_mean"] = tracking.mean(0).tolist()
    outputs["tracking_std"] = tracking.std(0).tolist()

    with (output_dir / "info.yaml").open("w") as f:
        yaml.dump(outputs, f, yaml.SafeDumper)
    pd.DataFrame(per_sweep_info).to_csv((output_dir / "sweep_info.csv"))

    print(outputs)


if __name__ == "__main__":
    main()
