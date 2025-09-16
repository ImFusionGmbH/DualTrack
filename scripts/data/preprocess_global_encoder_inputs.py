from argparse import ArgumentParser
import os
import h5py
import pandas as pd
import skimage
from pathlib import Path
from tqdm import tqdm 
import numpy as np 
import sys 


p = ArgumentParser()
p.add_argument('--metadata-path', help='Path to the metadata csv file. \
    Will add downsampled ultrasound images to the h5 datafiles corresponding to the metadata.')
args = p.parse_args()


print("Running")
metadata_path = args.metadata_path
processed_version_name = 'images_downsampled-224'
table = pd.read_csv(metadata_path)

for index in tqdm(table.index):
    try: 
        input_path = table.at[index, 'processed_sweep_path']

        with h5py.File(input_path, 'a') as f: 
            input_array = f['images'][:]

            N = len(input_array)
            output_array = np.zeros((N, 224, 224), dtype=np.uint8)

            for i in range(N): 
                image = input_array[i]
                image = skimage.util.crop(image, [(50, 50), (50, 50)])
                image = skimage.transform.resize(image, (224, 224))
                image = (image * 255).astype('uint8')
                output_array[i] = image
                # from PIL import Image 
                # img = Image.fromarray(image)
                # img.convert("RGB")
                # img.save(output_path)

            f[processed_version_name] = output_array
    except Exception as e: 
        print(e)
        

