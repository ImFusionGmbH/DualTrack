from argparse import ArgumentParser
import argparse
from omegaconf import OmegaConf
from src.models import get_model

parser = ArgumentParser()
parser.add_argument("--config", '-c', help='Path to config file')
args = parser.parse_args()

cfg = OmegaConf.load(args.config)

import torch 
model = get_model(**cfg.model)

from src.datasets.loader_factory.fusion_model_training import get_loaders
cfg.data.dataset = 'tus-rec'
train_loader, val_loader = get_loaders(**cfg.data)

batch = next(iter(train_loader))
model.eval()
with torch.no_grad():
    model.forward_dict(batch)['prediction']

from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

batch = next(iter(train_loader))
batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, profile_memory=True) as prof:
    with record_function("model_inference"):
        model.eval()
        with torch.no_grad():
            output = model.forward_dict(batch)['prediction']

print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))