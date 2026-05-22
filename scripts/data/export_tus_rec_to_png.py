import os
import h5py
import pandas as pd
import skimage
from pathlib import Path
from tqdm import tqdm 
import numpy as np 

print("Running")
metadata_path = '/h/pwilson/projects/trackerless-us-2/data/tus-rec-processed/metadata.csv'
target_dir = '/h/pwilson/projects/trackerless-us-2/data/tus-rec-processed_png/train/'

os.makedirs(target_dir, exist_ok=True)

table = pd.read_csv(metadata_path)
table = table.loc[table.split == 'train']
paths = table.processed_sweep_path.to_list()

for row in tqdm(list(table.iloc)):
    path = row['processed_sweep_path']

    with h5py.File(path) as f: 
        N = f['images'].shape[0]
        for i in range(0, N): 
            output_path = os.path.join(
                target_dir, 
                row['scan_shape'],
                os.path.basename(path).replace('.h5', f"_{i}.png")
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if os.path.exists(output_path): 
                continue

            image = f['images'][[i]]
            image = np.transpose(image, (1, 2, 0))
            image = skimage.util.crop(image, [(50, 50), (50, 50), (0, 0)])
            image = skimage.transform.resize(image, (224, 224))
            image = (image * 255).astype('uint8')

            from PIL import Image 

            img = Image.fromarray(image[..., 0])
            img.save(output_path)
