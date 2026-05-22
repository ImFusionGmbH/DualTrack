import torch.nn as nn
import logging
import os

from src.utils import load_model_weights
from functools import wraps
import torch


_MODELS = {}


def lyric_dragon_checkpoint(drop_keys=[]):
    checkpoint_path = os.environ.get(
        "LYRIC_DRAGON_CHECKPOINT",
        "experiments/good_runs/lyric-dragon/checkpoint/best.pt",
    )
    orig_state = torch.load(checkpoint_path)["model"]
    state = {}
    for key in orig_state:
        should_drop = False
        for prefix in drop_keys:
            if key.startswith(prefix):
                should_drop = True

        if not should_drop:
            state[key] = orig_state[key]
    return state


_REGISTERED_CHECKPOINTS = {
    "lyric-dragon": lambda: os.environ.get(
        "LYRIC_DRAGON_CHECKPOINT",
        "experiments/good_runs/lyric-dragon/checkpoint/best.pt",
    ),
    "lyric_dragon_flexible_load": lyric_dragon_checkpoint,
}


def register_model(fn):
    name = fn.__name__

    @wraps(fn)
    def wrapper(**kwargs):
        model = fn(**kwargs)
        if model is None:
            return model
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logging.info(
            f"Built model {name} with {n_params/1e6:.2f}M params and {n_trainable_params/1e6:.2f}M trainable params"
        )
        config = dict(name=name, **kwargs)
        logging.info(f"model config {kwargs}")
        model.registry_config = config
        return model

    _MODELS[name] = wrapper
    return wrapper


def list_models():
    return list(_MODELS.keys())


def get_model(
    name=None,
    checkpoint=None,
    checkpoint_kw={},
    load_kw=dict(strict=False),
    config=None,
    config_path=None,
    **kwargs,
) -> nn.Module:

    if config is not None:
        assert config_path is None
        if "model" in config:
            return get_model(**config["model"])
        else: 
            return get_model(**config)

    if config_path is not None: 
        assert config is None 
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)
        return get_model(config=config)

    model = _MODELS[name](**kwargs)

    if checkpoint:
        if checkpoint in _REGISTERED_CHECKPOINTS:
            checkpoint = _REGISTERED_CHECKPOINTS[checkpoint](**checkpoint_kw)

        print(load_model_weights(model, checkpoint, **load_kw))
        logging.info(f"Loaded weights from {checkpoint}")

    return model
