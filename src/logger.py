import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import logging
import os
import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
import tempfile
from typing import Optional

import torch
import torch.distributed
import wandb
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


def _get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_default_log_dir():
    return datetime.now().strftime(
        os.path.join(
            "experiments",
            "%Y-%b-%d",
            "%-I:%M.%S%p",
        )
    )


class Logger(ABC):
    """Handles experiment logging"""

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        name = kwargs.pop("name")
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls

    @classmethod
    def get_logger(cls, name, dir, config, *args, **kwargs):
        out: Logger = cls._registry[name](dir, config, *args, **kwargs)
        return out

    def __init__(
        self,
        dir,
        config,
        script_filename=None,
        auto_symlink_checkpoint_dir=False,
        disable_checkpoint=False,
        log_level=20,
    ):
        self.dir = dir
        self.config = config
        self.disable_checkpoint = disable_checkpoint
        self._prefix = None
        self._suffix = None

        if not _get_rank() == 0:
            return

        os.makedirs(dir, exist_ok=True)
        log_name = Path(dir) / "output.log"
        logging.basicConfig(
            handlers=[logging.FileHandler(log_name), logging.StreamHandler(sys.stdout)],
            level=log_level,
        )

        if auto_symlink_checkpoint_dir:
            if (slurm_id := os.environ.get("SLURM_JOB_ID")) and not os.path.exists(
                os.path.join(dir, "checkpoint")
            ):
                os.symlink(
                    f"/checkpoint/{os.environ.get('USER')}/{slurm_id}",
                    os.path.join(dir, "checkpoint"),
                )

        if not os.path.exists(os.path.join(dir, "checkpoint")):
            os.makedirs(os.path.join(dir, "checkpoint"), exist_ok=True)

        if script_filename is None:
            try:
                import __main__

                script_filename = __main__.__file__
            except:
                script_filename = None

        if script_filename is not None:
            shutil.copy(script_filename, os.path.join(self.dir, "script.py"))

        # save the configuration
        if isinstance(config, argparse.Namespace):
            config = vars(config)
            config = config.copy()
            for k in config.keys():
                if is_dataclass(config[k]):
                    config[k] = asdict(config[k])
        elif is_dataclass(config):
            config = asdict(config)
        OmegaConf.save(config, os.path.join(dir, "config_resolved.yaml"), resolve=True)
        OmegaConf.save(config, os.path.join(dir, "config.yaml"), resolve=False)

        # save the command
        with open(os.path.join(self.dir, "command.sh"), "w") as f:
            cmd = " ".join(["python"] + sys.argv)
            cmd = cmd.replace("--", "\\\n\t--")
            f.write(cmd)

        # save the git commit hash
        git_hash = os.popen("git rev-parse HEAD").read().strip()
        with open(Path(dir) / "git_info.txt", "w") as f:
            f.write("GIT COMMIT: \n")
            f.write(git_hash)

        # save the slurm job id
        if slurm_id := os.environ.get("SLURM_JOB_ID"):
            with open(os.path.join(dir, f"SLURM_JOB-{slurm_id}"), "w") as f:
                f.write("")

    def log(self, d: dict, global_step: Optional[int] = None):
        if not _get_rank() == 0:
            return

        if self._suffix is not None:
            d = self.add_suffix(d, self._suffix)
        if self._prefix is not None:
            d = self.add_prefix(d, self._prefix)
        self.log_impl(d, global_step)

    @abstractmethod
    def log_impl(self, d: dict, global_step: Optional[int] = None): ...

    def add_prefix(self, d, prefix):
        return {prefix + k: v for k, v in d.items()}

    def add_suffix(self, d, suffix):
        return {k + suffix: v for k, v in d.items()}

    def save_checkpoint(self, obj, name="last.pt"):
        if not _get_rank() == 0:
            return
        if self.disable_checkpoint:
            return
        torch.save(obj, os.path.join(self.dir, "checkpoint", name))

    def get_checkpoint(self, name="last.pt"):
        if not os.path.exists(os.path.join(self.dir, "checkpoint", name)):
            logging.info(f"Checkpoint {name} not found")
            return None
        else:
            state = torch.load(
                os.path.join(self.dir, "checkpoint", name),
                map_location="cpu",
                weights_only=False,
            )
            logging.info(f"Loaded checkpoint {name} - keys: {state.keys()}")
            return state

    def set_prefix(self, prefix: Optional[str]):
        self._prefix = prefix

    def set_suffix(self, suffix: Optional[str]):
        self._suffix = suffix

    def get_config_as_dict(self):
        return OmegaConf.to_object(self.config)

    def _convert_config_to_dict(self, config):
        if isinstance(config, argparse.Namespace):
            config = vars(config)
            config = config.copy()
            for k in config.keys():
                if is_dataclass(config[k]):
                    config[k] = asdict(config[k])
        elif is_dataclass(config):
            config = asdict(config)
        elif OmegaConf.is_config(config):
            config = OmegaConf.to_object(config)
        return config


class NullLogger(Logger, name="null"):
    def __init__(self, dir, config, *args, **kwargs):
        dir = tempfile.mkdtemp()
        super().__init__(dir, config, *args, **kwargs)

    def log_impl(self, d: dict, global_step: Optional[int] = None):
        print(f"Logger - global step {global_step}:")
        print(d)


class ConsoleLogger(Logger, name="console"):
    def log_impl(self, d: dict, global_step: Optional[int] = None):
        print(f"Logger - global step {global_step}:")
        print(d)


class FileLogger(Logger, name="file"):

    def log_impl(self, d: dict, global_step: Optional[int] = None):
        with open(os.path.join(self.dir, "metrics.jsonl"), "a") as f:
            d["step"] = global_step
            f.write(json.dumps(d))
            f.write("\n")


class TensorBoardLogger(Logger, name="tensorboard"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = SummaryWriter(log_dir=self.dir)

    def log_impl(self, d: dict, global_step: Optional[int] = None):
        for key, value in d.items():
            if isinstance(value, Figure):
                self.writer.add_figure(key, value, global_step)
            else:
                self.writer.add_scalar(key, value, global_step)


class WandbLogger(Logger, name="wandb"):
    def __init__(
        self,
        dir,
        config,
        wandb_kwargs={},
        wandb_project="trackerless-ultrasound",
        *args,
        **kwargs,
    ):
        super().__init__(dir, config, *args, **kwargs)

        if not _get_rank() == 0:
            return

        import wandb

        config = self._convert_config_to_dict(config)
        self.run = wandb.init(
            # dir=self.dir,
            config=config,
            project=wandb_project,
            **wandb_kwargs,
        )

        try:
            with open(os.path.join(self.dir, "WANDB_URL"), "w") as f:
                f.write(self.run.url)
        except:
            pass

        self.run.save(
            os.path.join(self.dir, "config_resolved.yaml"), base_path=self.dir
        )
        # self.run.log_code(
        #     os.getcwd(),
        #     exclude_fn=lambda path: "experiment" in path,
        #     include_fn=lambda path: path.endswith(".py"),
        # )

    def log_impl(self, d, global_step: Optional[int] = None):
        to_log = {}
        if global_step is not None:
            to_log["epoch"] = global_step
        for key, value in d.items():
            if isinstance(value, Figure):
                to_log[key] = wandb.Image(value)
            else:
                to_log[key] = value
        self.run.log(to_log)


def get_logger(name, dir, config, *args, **kwargs) -> Logger:
    if name == "wandb":
        return WandbLogger(dir, config, *args, **kwargs)
    elif name == "tensorboard":
        return TensorBoardLogger(dir, config, *args, **kwargs)
    elif name == "console":
        return ConsoleLogger(dir, config, *args, **kwargs)
    else:
        raise ValueError(f"No logger called {name}")
