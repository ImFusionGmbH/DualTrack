from argparse import ArgumentParser
from collections import defaultdict
import os

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def main(): 
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_csv_file', required=True) 
    parser.add_argument('-o', '--output_dir', required=True)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True) 

    table = pd.read_csv(args.input_csv_file, index_col=0)
    splits = defaultdict(list)

    for i, row in enumerate(table.iloc):

        path = row['processed_sweep_path']
        split = row['split']
        splits[split].append(i)

        case_name = f"Case{i:04}"
        os.makedirs(os.path.join(args.output_dir, case_name), exist_ok=True)

        with h5py.File(path) as f:
            # 1. Process the image frames
            input_frames = f['images']
            N, H, W = input_frames.shape 
            output_frames = np.zeros((224, 224, N), dtype=np.uint8)
            for frame_idx in range(N): 
                input_frame = input_frames[frame_idx] #type:ignore
                out = cv2.resize(input_frame, (224, 224)) #type:ignore
                output_frames[..., frame_idx] = out

            # save the frames together as array
            np.save(os.path.join(args.output_dir, case_name, f'{case_name}_frames.npy'), output_frames)

            # save the frames individually as jpeg
            frames_dir = os.path.join(args.output_dir, case_name, "Frames")
            os.makedirs(frames_dir, exist_ok=True)
            for frame_idx in range(N): 
                frame = output_frames[..., frame_idx] 
                cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:04}.jpg"), frame)

            # 2. Process the tracking data
            tracking_info = f['tracking']
            def tracking_matrix_to_custom_format(matrix): 
                """Converts the given affine transformation matrix to the proprietary format used by the authors of 
                `Sensorless Freehand 3D Ultrasound Reconstruction via Deep Contextual Learning`"""
        
                rotation_matrix = matrix[:3, :3]
                translation = matrix[:3, 3]

                rotation = Rotation.from_matrix(rotation_matrix)
                quaternion_representation = rotation.as_quat()

                out = np.zeros((9,)) 
                out[0:2] = 0 # these two entries are meaningless 
                out[2:5] = translation
                out[5:9] = quaternion_representation
                return out

            representation = []
            for frame_idx in range(N): 
                representation_i = tracking_matrix_to_custom_format(tracking_info[frame_idx])
                representation.append(representation_i)
            representation = np.stack(representation)
            np.savetxt(os.path.join(args.output_dir, case_name, f"{case_name}_pos.txt"), representation)

            calibration_matrix = f['pixel_to_image'][:]
            np.savetxt(os.path.join(args.output_dir, case_name, f'{case_name}_USCalib.txt'), calibration_matrix)

        print(path, split)

    # writing splits
    for split, ids in splits.items(): 
        with open(os.path.join(args.output_dir, f"{split}_ids.txt"), "w") as f: 
            [f.write(f"{id_}\n") for id_ in ids]

if __name__ == '__main__': 
    main()