from dataclasses import dataclass, fields
import inspect
import logging
import os
import numpy as np
import argparse

try:
    import imfusion
except ImportError:
    imfusion = None
import h5py
from collections import UserDict
import torch
import matplotlib.pyplot as plt


@dataclass
class SweepData:
    images: np.ndarray
    tracking: np.ndarray
    spacing: np.ndarray
    dimensions: np.ndarray
    pixel_to_image: np.ndarray
    metadata: dict | None = None

    @classmethod
    def from_h5(cls, path_or_handle):
        if isinstance(path_or_handle, h5py.File):
            F = path_or_handle
            handle = True
        else:
            F = h5py.File(path_or_handle, "r")
            handle = False
        instance = cls(
            F["images"][:],
            F["tracking"][:],
            F["spacing"][:],
            F["dimensions"][:],
            F["pixel_to_image"][:],
        )
        if not handle:
            F.close()
        return instance

    def to_h5(self, path):
        with h5py.File(path, "w") as F:
            F.create_dataset("images", data=self.images)
            F.create_dataset("tracking", data=self.tracking)
            F.create_dataset("spacing", data=self.spacing)
            F.create_dataset("dimensions", data=self.dimensions)
            F.create_dataset("pixel_to_image", data=self.pixel_to_image)

    @classmethod
    def from_imfusion(cls, path):
        (sweep,) = imfusion.load(path)

        N, _, H, W, _ = sweep.shape
        images = np.zeros((N, H, W), dtype=np.uint8)
        for i in range(N):
            images[i] = np.array(sweep[i])[:, :, 0]

        tracking = np.zeros((N, 4, 4), dtype=np.float32)
        for i in range(N):
            tracking[i] = sweep.matrix(i)

        return cls(
            images,
            tracking,
            sweep.descriptor().spacing,
            sweep.descriptor().dimensions,
            sweep.get().pixel_to_world_matrix,
        )

    def to_imfusion(self, path):
        images = self.images
        tracking = self.tracking
        spacing = self.spacing
        dimensions = self.dimensions

        descriptor = imfusion.ImageDescriptor()
        descriptor.set_dimensions(dimensions)
        descriptor.set_spacing(spacing, True)

        N, H, W = images.shape
        sweep = imfusion.UltrasoundSweep()
        sweep.set_timestamp(False)

        ts = imfusion.TrackingSequence()
        for i in range(N):
            sweep_i = imfusion.SharedImage(descriptor)
            sweep_i.assign_array(images[i][:, :, None])
            sweep_i.descriptor.set_spacing(spacing, True)
            sweep.add(sweep_i)
            ts.add(tracking[i])

        sweep.add_tracking(ts)
        # sweep.descriptor().set_spacing(spacing, True)
        # sweep.descriptor().set_dimensions(dimensions)

        imfusion.save([sweep], path)


def load_model_weights(model, path_or_state, strict=True, handle_size_mismatch=True, state_dict_prefix=None):
    """
    Helper function to load model weights into a model.
    """

    if isinstance(path_or_state, str):
        state = torch.load(path_or_state, map_location="cpu", weights_only=False)
    else:
        state = path_or_state

    if "model" in state:
        state = state["model"]

    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
    consume_prefix_in_state_dict_if_present(state, '_orig_mod.')

    if state_dict_prefix:
        state = {
            k[len(state_dict_prefix):]: v for k, v in state.items() if k.startswith(state_dict_prefix)
        }

    if handle_size_mismatch: 
        model_state = model.state_dict()
        filtered_state_dict = {
            k: v for k, v in state.items()
            if k in model_state and v.size() == model_state[k].size()
        }
        state = filtered_state_dict

    out = model.load_state_dict(state, strict=strict)

    return out


def submit_slurm(
    log_dir,
    task_fn,
    wandb_id=None,
    task_args=(),
    task_kwargs={},
    mem_gb=48,
    qos="m",
    cpus=4,
):
    import os

    os.makedirs(log_dir, exist_ok=True)

    cwd = os.getcwd()

    import submitit
    from submitit.helpers import DelayedSubmission

    class Main:
        def __init__(self, task_fn):
            self.task_fn = task_fn

        def __call__(self, *args, **kwargs):
            if wandb_id:
                os.environ["WANDB_RUN_ID"] = wandb_id
            else:
                os.environ["WANDB_RUN_ID"] = os.environ["SLURM_JOB_ID"]
            os.environ["WANDB_RESUME"] = "allow"

            slurm_checkpoint_dir = os.path.join(
                "/checkpoint", os.environ["USER"], os.environ["SLURM_JOB_ID"]
            )
            if not os.path.exists(
                ckpt_path := os.path.abspath(os.path.join(log_dir, "checkpoint"))
            ):
                os.symlink(slurm_checkpoint_dir, ckpt_path)

            return self.task_fn(*args, **kwargs)

        def checkpoint(self, *args, **kwargs):
            return DelayedSubmission(Main(self.task_fn), *args, **kwargs)

    # Initialize the Submitit AutoExecutor
    executor = submitit.SlurmExecutor(
        folder=log_dir, max_num_timeout=10
    )  # Logs will be saved here

    # Update the executor parameters based on the SLURM script
    executor.update_parameters(
        job_name="tracking_estimation",
        stderr_to_stdout=True,
        nodes=1,
        ntasks_per_node=1,
        cpus_per_task=cpus,
        gpus_per_node=1,
        mem=f"{mem_gb}G",
        qos=qos,
        partition="a40,rtx6000,t4v2",
        time=8 * 60,
        signal_delay_s=240,
    )

    job = executor.submit(Main(task_fn), *task_args, **task_kwargs)
    print(f"Submitted job with ID: {job.job_id}")
    print(job.paths.stdout)


def get_current_function_args():
    frame = inspect.currentframe()
    assert frame is not None
    frame = frame.f_back
    assert frame is not None
    argnames, _, _, locals = inspect.getargvalues(frame)
    return {k: locals[k] for k in argnames}


class UnstructuredArgsAction(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest: str,
        nargs: int | str | None = None,
        required: bool = False,
        help: str | None = None,
        metavar: str | tuple[str, ...] | None = "KEY=VALUE|*.yaml",
    ) -> None:
        super().__init__(
            option_strings,
            dest,
            nargs="+",
            const=None,
            default={},
            type=None,
            choices=None,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        option_string: str | None = None,
    ) -> None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({})
        dotlist_elements = []
        for value in values:
            if os.path.isfile(value):
                cfg = OmegaConf.merge(cfg, OmegaConf.load(value))
            else:
                dotlist_elements.append(value)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist_elements))
        cfg = OmegaConf.to_object(cfg)
        setattr(namespace, self.dest, cfg)


def log_model_info(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        f"Model {model.__class__} with {n_params/1e6:.2f}M params and {n_trainable_params/1e6:.2f}M trainable params"
    )


def parse_dict_from_string(string):
    from omegaconf import OmegaConf

    if "," in string:
        dotlist = string.split(",")
    else:
        dotlist = [string]

    return OmegaConf.to_container(OmegaConf.from_dotlist(dotlist))


def optional_type(type):
    def inner(s):
        if s == "null":
            return None
        else:
            return type(s)

    return inner
