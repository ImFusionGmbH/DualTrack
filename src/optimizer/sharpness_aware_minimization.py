# sam.py
from __future__ import annotations
import torch
from torch import nn


_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


class FreezeBNStats:
    """Temporarily freeze BN running stats (keeps Dropout active)."""

    def __init__(self, model: nn.Module):
        self.m = model
        self._modes = []

    def __enter__(self):
        self._bns = [m for m in self.m.modules() if isinstance(m, _BN)]
        self._modes = [bn.training for bn in self._bns]
        for bn in self._bns:
            bn.eval()  # freeze running stats for sharpness pass only
        return self

    def __exit__(self, exc_type, exc, tb):
        for bn, mode in zip(self._bns, self._modes):
            bn.train(mode)


class SAM:
    """
    Wrap any base optimizer (e.g., torch.optim.AdamW).
    - adaptive=False  -> SAM
    - adaptive=True   -> ASAM (scale by |w|)
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        eps: float = 1e-12,
    ):
        self.model = model
        self.opt = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.eps = eps

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        norms = []
        for g in (
            p.grad
            for group in self.opt.param_groups
            for p in group["params"]
            if p.grad is not None
        ):
            norms.append(g.norm(p=2))
        if not norms:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self):
        # compute perturbation scale
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + self.eps)

        # e = scale * (|w| ⊙ g) for ASAM, or scale * g for SAM
        for group in self.opt.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = p.grad
                if self.adaptive:
                    e = e * p.detach().abs()
                e = e * scale
                p.add_(e)  # w <- w + e
                p.state["sam_e"] = e  # save e

    @torch.no_grad()
    def second_step(self):
        # w <- w - e
        for group in self.opt.param_groups:
            for p in group["params"]:
                e = p.state.pop("sam_e", None)
                if e is not None:
                    p.add_(-e)
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()

    def state_dict(self):
        return {"opt": self.opt.state_dict()}

    def load_state_dict(self, state):
        self.opt.load_state_dict(state["opt"])


def sam_optimizer_step(sam, is_ddp, scaler, loss_closure, model):
    # ----------- pass 1: compute perturbation -----------
    sam.zero_grad()
    if is_ddp:
        # avoid an allreduce on the first backward
        no_sync_ctx = model.no_sync()
    else:
        # no-op context
        from contextlib import nullcontext as no_sync_ctx

    with no_sync_ctx():
        loss = loss_closure()
    scaler.scale(loss).backward()

    # unscale before accessing .grad (required with AMP)
    scaler.unscale_(sam.opt)
    # (optional) clip before perturbation
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    sam.first_step()  # w <- w + e

    # ----------- pass 2: sharpness-aware gradient -----------
    with FreezeBNStats(model):  # freeze BN running stats during the sharpness pass
        sam.zero_grad()
        loss_sharp = loss_closure()
        scaler.scale(loss_sharp).backward()

    # step at the *original* weights
    scaler.step(sam.opt)  # equivalent to: sam.second_step() with scaler support
    scaler.update()

    return loss