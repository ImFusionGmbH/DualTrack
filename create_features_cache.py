from argparse import ArgumentParser
import itertools
import os

import h5py
from omegaconf import OmegaConf
import torch
from tqdm import tqdm 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def export_features(args): 
    config = OmegaConf.load(args.config)
    assert config.data.batch_size == 1

    state = torch.load(config.checkpoint, weights_only=False) if 'checkpoint' in config else None

    if state:    
        print(f"Loaded checkpoint: ")
        print(f"Epoch: {state['epoch']}")
        print(f"Best score: {state['best_score']}")
        print(f"Training config: \n{OmegaConf.to_yaml(OmegaConf.create(state['config']))}")
    
    os.makedirs(args.output_folder, exist_ok=True)
    OmegaConf.save(config, os.path.join(args.output_folder, 'config.yaml'), resolve=True)
    torch.save(state, os.path.join(args.output_folder, 'checkpoint.pt'))

    from src.models import get_model
    model = get_model(**config.model)
    # print(model.load_state_dict(state['model']))
    model.eval().to(device)
    
    from src.datasets import get_dataloaders
    train_loader, val_loader = get_dataloaders(**config.data)

    with h5py.File(os.path.join(args.output_folder, 'features.h5'), 'a') as f:
        # Extract features from the model and save them to the HDF5 file
        for batch in tqdm(itertools.chain(train_loader or [], val_loader)):
            with torch.autocast(device_type='cuda', enabled=config.use_amp):
                features = model(batch['images'].to(device))[0]
            features = features.half().cpu().numpy()
            assert len(features) == batch['images'].shape[1]
            sweep_id = batch['sweep_id'][0]
            f.create_dataset(sweep_id, data=features)
            f.flush()


if __name__ == "__main__": 
    p = ArgumentParser()
    p.add_argument('config', help='Path to the configuration file')
    p.add_argument('output_folder', help='Folder to save features')
    args = p.parse_args()
    export_features(args)