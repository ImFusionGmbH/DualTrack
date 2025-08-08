import torch
from torch import nn


class RegularGridSampler(nn.Module):
    def __init__(self, grid_spacing):
        super().__init__()
        self.grid_spacing = grid_spacing

    def __call__(self, images, features):
        B, N, C, H, W = images.shape
        samples = []
        for _ in range(B):
            samples.append(torch.arange(0, N, self.grid_spacing, device=images.device))
        return torch.stack(samples, 0)


class SparseSampler(nn.Module):
    def __init__(self, n_samples=128):
        super().__init__()
        self.n_samples = n_samples

    def __call__(self, images, features):
        B, N, C, H, W = images.shape
        samples = []
        for _ in range(B):
            samples.append(
                torch.sort(
                    torch.randperm(N, device=images.device)[: self.n_samples]
                ).values
            )

        return torch.stack(samples, 0)


