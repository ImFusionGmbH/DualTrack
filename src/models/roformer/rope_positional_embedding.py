import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn


class RopePositionEmbedding1D(nn.Module):
    """
    1D RoPE (axial, no mixing) with optional coord jitter/shift/rescale.
    Returns (sin, cos) with shape [L or (1+L), D_head] where D_head = embed_dim // num_heads.

    If prepend_cls=True, the first row corresponds to a special token with identity rotation
    (sin=0, cos=1), i.e. “unrotated class token”.
    """
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        # For 1D, all normalize modes reduce to dividing by L; keep the API shape-compatible.
        normalize_coords: Literal["separate", "max", "min"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (2 * num_heads) == 0, "embed_dim must be divisible by 2*num_heads for 1D RoPE."
        both_periods = (min_period is not None) and (max_period is not None)
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or (`min_period` and `max_period`) must be provided, but not both.")

        self.D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.dtype = dtype
        # In 1D, frequencies count = D_head//2 (each pair uses one frequency).
        self.register_buffer(
            "periods",
            torch.empty(self.D_head // 2, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            # geometric progression over D_head//2 frequencies
            freqs = 2 * torch.arange(self.D_head // 2, device=device, dtype=dtype) / (self.D_head)
            periods = self.base ** freqs  # [D_head//2]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 2, device=device, dtype=dtype)
            periods = base ** exponents
            periods = (periods / base) * self.max_period  # range [min_period, max_period]
        self.periods.data = periods

    def forward(self, L: int) -> tuple[Tensor, Tensor]:
        """
        Args:
            L: sequence length (number of non-special tokens).
        Returns:
            sin, cos: tensors of shape [L] or [1+L], each with last-dim D_head.
        """
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # coords in [-1, +1], centered at token centers (0.5, 1.5, ..., L-0.5)/L
        coords = torch.arange(0.5, L, **dd) / L  # [L] in (0,1)
        coords = 2.0 * coords - 1.0              # [-1, 1]

        # augment (train-time only)
        if self.training and self.shift_coords is not None:
            shift = torch.empty((), **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords = coords + shift
        if self.training and self.jitter_coords is not None:
            jmax = math.log(self.jitter_coords)
            jitter = torch.empty((), **dd).uniform_(-jmax, jmax).exp()
            coords = coords * jitter
        if self.training and self.rescale_coords is not None:
            rmax = math.log(self.rescale_coords)
            rescale = torch.empty((), **dd).uniform_(-rmax, rmax).exp()
            coords = coords * rescale

        # angles: [L or 1+L, D_head//2]; then interleave to [*, D_head]
        angles = 2 * math.pi * coords[:, None] / self.periods[None, :]   # [S, D_head//2], S=L or 1+L
        # Build sin/cos for pairs (even, odd). We "tile(2)" to match per-head D.
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)             # [S, D_head]
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)             # [S, D_head]
        return sin, cos


def apply_rope_1d(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """
    Apply 1D RoPE to Q/K.
    x:   (B, H, S, D_head)
    sin,cos: (S, D_head) or broadcastable to that.
    Returns rotated x with same shape.
    """
    # Broadcast sin/cos to (B,H,S,D)
    while sin.dim() < x.dim():
        sin = sin.unsqueeze(0)
        cos = cos.unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    sin_e  = sin[..., 0::2]
    cos_e  = cos[..., 0::2]
    # 2x2 rotation on pairs
    x_rot_even = x_even * cos_e - x_odd * sin_e
    x_rot_odd  = x_even * sin_e + x_odd * cos_e
    # interleave back
    out = torch.empty_like(x)
    out[..., 0::2] = x_rot_even
    out[..., 1::2] = x_rot_odd
    return out
