import os
import submitit
from src.logger import get_default_log_dir
import train
import argparse
from omegaconf import OmegaConf


def main(config, task='train'):
    if task == 'train':
        import train  # or: from train import main as train_main
        train.train(config)  # or: train_main(config_path)
    elif task == 'sanity_check':
        import sanity_check_ddp
        sanity_check_ddp.train(config)


class Main:
    def __init__(self, cfg, task): 
        self.cfg = cfg 
        self.task = task

    def checkpoint(self): 
        return submitit.helpers.DelayedSubmission(self)

    def __call__(self): 
        os.environ['WANDB_RUN_ID'] = os.environ["SLURM_JOB_ID"]
        os.environ['WANDB_RESUME'] = 'allow'

        return main(self.cfg, self.task)


slurm_defaults = dict(
    nodes=1,
    slurm_job_name="tus-rec",
    cpus_per_task=10,
    tasks_per_node=1,
    slurm_gres="gpu:1",
    slurm_account="aip-medilab",
    timeout_min=4 * 60,
    mem_gb=128,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", default=get_default_log_dir())
    parser.add_argument("--config", "-c", help="Path to yaml configuration file")
    parser.add_argument("--resume_from_wandb", metavar='PATH', help="Resume from Weights & Biases")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Overrides to config")
    parser.add_argument("--task", default='train')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    if args.resume_from_wandb: 
        from wandb import Api
        api = Api() 
        run = api.run(args.resume_from_wandb)
        f = run.file('config_resolved.yaml')
        with f.download('/tmp', replace=True) as f:
            cfg = OmegaConf.load(f)

    else: 
        cfg = OmegaConf.create({"log_dir": args.log_dir})
        if args.config:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))
        if args.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    slurm_cfg = cfg.get('slurm', {})
    slurm_cfg = OmegaConf.merge(OmegaConf.create(slurm_defaults), slurm_cfg)

    executor = submitit.AutoExecutor(folder=args.log_dir, max_num_timeout=25)
    executor.update_parameters(
        **slurm_cfg
    )
    # Pass your config path as argument
    job = executor.submit(Main(cfg, args.task))
    print(f"Submitted job: {job.job_id}")
    print(f"{job.paths.stdout} for stdout")