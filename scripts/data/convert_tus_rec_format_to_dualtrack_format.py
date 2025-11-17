from argparse import ArgumentParser
import sys
import os
import shutil
import pandas as pd
from tqdm import tqdm
import h5py 
import logging 

sys.path.append(os.getcwd())
import numpy as np


INPUT_CSV_HELP = """Path to a csv file containing at least the following columns:
(i) sweep_id: Unique identifier for the sweep
(ii) raw_tus_rec_sweep_path: Path to the .tusrec file containing the raw TUS-REC sweep data. This will be an `.h5 file` with datasets `frames` (the images) and `tforms` (the tool to world tracking matrices).
In case the `frames` and `tforms` datasets are stored in separate files, you can also provide the following columns:
(iii) raw_tus_rec_frames_path: Path to the .tusrec file containing the `frames` dataset.
(iv) raw_tus_rec_tforms_path: Path to the .tusrec file containing the `tforms` dataset.
(v - OPTIONAL) split: Which data split this sweep belongs to (e.g., train/val/test). If not provided, the `split` cli argument will be used for all sweeps.
(vi - OPTIONAL) check_against: Path to an existing dualtrack format .h5 file to check the conversion against.
"""

OUTPUT_DIR_HELP = """Path to the output directory where converted dualtrack format files will be saved.
For each sweep in the input csv file, a corresponding .h5 file will be created in the output directory.
A metadata.csv file will also be created in the output directory summarizing the processed sweeps.
"""


def main():
    p = ArgumentParser()
    p.add_argument(
        "--input_csv_path",
        "-i",
        help=INPUT_CSV_HELP,
        required=True,
    )
    p.add_argument("--output_dir", "-o", help=OUTPUT_DIR_HELP, required=True)
    p.add_argument(
        "--dryrun",
        action="store_true",
        help="If set, will only print the names of the output files without creating them.",
    )
    p.add_argument('--split',
        help='Data split to assign to all sweeps (if the input csv does not have a `split` column).',
        default='train',
    )

    args = p.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(os.path.join(args.output_dir, "conversion.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    shutil.copy(__file__, os.path.join(args.output_dir, "script.py"))
    import json

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)

    converter = TUSRecToDualTrackConverter.from_hardcoded_tus_rec_2024_calibration_settings()
    input_table = pd.read_csv(args.input_csv_path)
    output_rows = []

    for i, row in tqdm(input_table.iterrows(), total=len(input_table)):
        logger.info(f"Processing sweep {i+1}/{len(input_table)}")
        logger.info(f"Row data: {row.to_dict()}")

        sweep_id = row["sweep_id"]
        if "raw_tus_rec_sweep_path" in row:
            input_file = row["raw_tus_rec_sweep_path"]
            f = h5py.File(input_file, "r")
            frames = f["frames"][:]
            tforms = f["tforms"][:]
        else:
            frames_file = row["raw_tus_rec_frames_path"]
            tforms_file = row["raw_tus_rec_tforms_path"]
            f_frames = h5py.File(frames_file, "r")
            f_tforms = h5py.File(tforms_file, "r")
            frames = f_frames["frames"][:]
            tforms = f_tforms["tforms"][:]

        data_dict = converter.convert_data_dict(frames, tforms)
        if "check_against" in row and not pd.isna(row["check_against"]):
            logger.info(f"Checking conversion against: {row['check_against']}")
            ref_file = row["check_against"]
            f_ref = h5py.File(ref_file, "r")
            for k in data_dict:
                assert np.allclose(
                    data_dict[k], f_ref[k][:]
                ), f"Mismatch found in sweep {sweep_id} for key {k}"
                logger.info(f"Key {k} matches reference file.")

        output_file = os.path.join(args.output_dir, f"{sweep_id}.h5")
        if not args.dryrun:
            logger.info(f"Creating file: {output_file}")
            with h5py.File(output_file, "w") as F_out:
                for k in data_dict:
                    F_out.create_dataset(k, data=data_dict[k], compression="gzip")
        else: 
            print(f"Would create file: {output_file}")
        output_rows.append({"processed_sweep_path": output_file, **row.to_dict()})
        output_rows[-1]["split"] = row.get("split", args.split)

    # Save the output rows to a CSV file
    output_table = pd.DataFrame(output_rows)
    output_table.to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)


class TUSRecToDualTrackConverter:
    def __init__(
        self,
        *,
        spacing,
        image_in_tus_rec_coords_to_tool,
        image_in_dualtrack_coords_to_image_in_tus_rec_coords,
        pixel_to_image_in_dualtrack_coords=None,
    ):
        self.spacing = spacing
        self.image_in_tus_rec_coords_to_tool = image_in_tus_rec_coords_to_tool
        self.image_in_dualtrack_coords_to_image_in_tus_rec_coords = (
            image_in_dualtrack_coords_to_image_in_tus_rec_coords
        )
        self.pixel_to_image_in_dualtrack_coords = pixel_to_image_in_dualtrack_coords

    def convert_tracking_sequence(self, tool_to_world_tracking_sequence):
        image_in_tus_rec_coords_to_world = (
            tool_to_world_tracking_sequence @ self.image_in_tus_rec_coords_to_tool
        )
        image_in_dualtrack_coords_to_world = (
            image_in_tus_rec_coords_to_world
            @ self.image_in_dualtrack_coords_to_image_in_tus_rec_coords
        )
        return image_in_dualtrack_coords_to_world

    def convert_data_dict(self, images, tool_to_world_tracking):
        output = {}
        output["images"] = images
        output["tracking"] = self.convert_tracking_sequence(tool_to_world_tracking)
        output["spacing"] = np.array([self.spacing[0], self.spacing[1], 1])
        N, H, W = images.shape[0:3]
        output["dimensions"] = np.array((W, H, 1))
        if self.pixel_to_image_in_dualtrack_coords is not None:
            output["pixel_to_image"] = self.pixel_to_image_in_dualtrack_coords
        return output

    @classmethod
    def from_hardcoded_tus_rec_2024_calibration_settings(cls):
        IMAGE_IN_TUS_REC_COORDS_TO_TOOL = np.asarray(
            [
                0.231064671309448,
                -0.21805203504134,
                0.948189025293473,
                -70.7413291931152,
                -0.190847036221471,
                -0.965787272832032,
                -0.175591436013112,
                -80.6505661010742,
                0.954036962825936,
                -0.140386087807861,
                -0.264773903381482,
                -46.1766223907471,
                0,
                0,
                0,
                1,
            ]
        ).reshape(4, 4)

        SPACING = (0.229389190673828, 0.220979690551758, 1)
        translation_x = (-640 / 2 + -0.5) * SPACING[0]
        translation_y = (-480 / 2 + -0.5) * SPACING[1]

        IMAGE_IN_TUS_REC_COORDS_TO_IMAGE_IN_DUALTRACK_COORDS = np.array(
            [
                [1, 0, 0, translation_x],
                [0, 1, 0, translation_y],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        PIXEL_TO_IMAGE_IN_DUALTRACK_COORDS = np.array(
            [
                [0.22938919, 0.0, 0.0, -73.28984642],
                [0.0, 0.22097969, 0.0, -52.92463589],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        return cls(
            spacing=SPACING,
            image_in_tus_rec_coords_to_tool=IMAGE_IN_TUS_REC_COORDS_TO_TOOL,
            image_in_dualtrack_coords_to_image_in_tus_rec_coords=np.linalg.inv(
                IMAGE_IN_TUS_REC_COORDS_TO_IMAGE_IN_DUALTRACK_COORDS
            ),
            pixel_to_image_in_dualtrack_coords=PIXEL_TO_IMAGE_IN_DUALTRACK_COORDS,
        )


if __name__ == "__main__":
    main()