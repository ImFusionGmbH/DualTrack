from logging import getLogger
from torch import distributed as dist
import os
import torch


logger = getLogger()


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def init_distributed(port=40112, rank_and_world_size=(None, None)):

    for env_varname in [
        "SLURM_NTASKS",
        "SLURM_PROCID",
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "HOSTNAME",
    ]:
        print(f"{env_varname}={os.getenv(env_varname)}")

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ["MASTER_ADDR"] = "localhost"

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            os.environ["MASTER_ADDR"] = os.environ.get("HOSTNAME", "localhost")
        except Exception:
            logger.info("SLURM vars not set (distributed training not available)")
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ["MASTER_PORT"] = str(port)
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=rank
        )

        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            print(f"Setting CUDA device to {dist.get_rank() % len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))}")
            torch.cuda.set_device(dist.get_rank() % len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
        else:
            torch.cuda.set_device(0)

    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f"distributed training not available {e}")

    return world_size, rank
