from dataclasses import dataclass
from attr import asdict
import torch
from torch import optim
import logging

from src.configuration import BaseConfig


@dataclass
class WarmupConfig:
    warmup_steps: int
    initial_lr_factor: float = 0.001


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Wraps another scheduler to add a linear warmup phase"""

    def __init__(self, scheduler, warmup_config: WarmupConfig):
        self.scheduler = scheduler
        self.warmup_config = warmup_config
        self.current_step = 0
        super().__init__(scheduler.optimizer)

    def get_lr(self):
        if self.current_step < self.warmup_config.warmup_steps:
            # Linear warmup from initial_lr_factor * base_lr to base_lr
            alpha = self.current_step / self.warmup_config.warmup_steps
            factor = (
                self.warmup_config.initial_lr_factor
                + (1 - self.warmup_config.initial_lr_factor) * alpha
            )
            return [base_lr * factor for base_lr in self.base_lrs]

        # After warmup, use wrapped scheduler's learning rates
        self.scheduler.step(self.current_step - self.warmup_config.warmup_steps)
        return self.scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is not None:
            self.current_step = epoch
        else:
            self.current_step += 1
        return super().step()


def setup_optimizer(
    model,
    optimizer_name='adam',
    scheduler_name='none',
    lr=1e-4,
    weight_decay=0.,
    total_steps=None,
    total_epochs=None,
    num_steps_per_epoch=None,
    warmup_steps=None,
    warmup_epochs=None,
    make_grad_scaler=False,
    state=None,
    device=None, 
    use_amp=None,
    **opt_kwargs,
):
    """
    Setup optimizer and scheduler for the model.

    Args:
        model: The model to optimize.
        optimizer_name: The name of the optimizer to use.
        scheduler_name: The name of the scheduler to use.
        lr: The learning rate.
        weight_decay: The weight decay.
        total_steps: The total number of steps.
        warmup_steps: The number of warmup steps.
        make_grad_scaler: Whether to make a gradient scaler.
        state: The state of the optimizer and scheduler.

    Returns:
        tuple: A tuple containing the optimizer, scheduler, and gradient scaler,
        or just the optimizer and scheduler if make_grad_scaler is False.
    """

    if total_steps is None and scheduler_name != "none":
        assert total_epochs is not None and num_steps_per_epoch is not None
        total_steps = total_epochs * num_steps_per_epoch
        logging.info(f"Total steps: {total_steps}")

    if warmup_steps is None and "warmup" in scheduler_name:
        assert warmup_epochs is not None and num_steps_per_epoch is not None
        warmup_steps = warmup_epochs * num_steps_per_epoch
        logging.info(f"Warmup steps: {warmup_steps}")

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    if optimizer_name == "adam":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **opt_kwargs)
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(
            params, lr, momentum=0.9, weight_decay=weight_decay, **opt_kwargs
        )
    elif optimizer_name == "adagrad":
        opt = torch.optim.Adagrad(params, lr, weight_decay=weight_decay, **opt_kwargs)
    else:
        raise ValueError()

    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s)
        for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        opt.add_param_group({"params": params, **hp})

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(opt.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        logging.info(
            " | ".join(
                [
                    f"Optimizer group {i}",
                    f"{len(g['params'])} tensors",
                ]
                + [f"{k} {v}" for k, v in group_hps.items()]
            )
        )

    if state is not None:
        opt.load_state_dict(state["optimizer"])

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    elif scheduler_name == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda iter: 1.0)
    elif scheduler_name == "warmup_cosine":
        scheduler = WarmupScheduler(
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps),
            WarmupConfig(warmup_steps=warmup_steps, initial_lr_factor=lr / 1e3),
        )
    else:
        raise ValueError(f"Unknown scheduler {scheduler_name}")

    if state is not None:
        scheduler.load_state_dict(state["scheduler"])

    if not make_grad_scaler:
        return opt, scheduler

    scaler = torch.GradScaler(device=device, enabled=use_amp)
    if state:
        scaler.load_state_dict(state["scaler"])

    return opt, scheduler, scaler
