from argparse import ArgumentParser
from datetime import datetime
import sys
from omegaconf import OmegaConf
import os

import omegaconf
from src.logger import get_default_log_dir


class ConfigPathArgumentParser(ArgumentParser):
    def __init__(
        self, default_config_paths=[], default_configs: list[dict] = [], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.default_config_paths = default_config_paths
        self.default_configs = default_configs

        self.add_argument("--log_dir", default=get_default_log_dir())
        self.add_argument(
            "--config",
            "-c",
            help="Path to yaml configuration",
            nargs="+",
            default=self.default_config_paths,
        )
        self.add_argument(
            "--overrides", "-ov", nargs="+", help="Overrides to config", default=[]
        )
        self.add_argument(
            "--print_config", action="store_true", help="print config and exit"
        )

    def parse_config(self, *args, **kwargs):
        args = self.parse_args(*args, **kwargs)
        if os.path.isfile(p := os.path.join(args.log_dir, "config.yaml")):
            print("Found config in log dir, using this config.")
            return OmegaConf.load(p)

        config = OmegaConf.create({})
        config = self._merge_configs(config, self.default_configs)
        # config = self._merge_config_files(config, self.default_config_paths)
        if args.config:
            config = self._merge_config_files(config, args.config)
        if args.overrides:
            config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
        config = OmegaConf.merge(OmegaConf.create({"log_dir": args.log_dir}), config)

        if args.print_config:
            print()
            print(OmegaConf.to_yaml(config))
            sys.exit(0)
        return config

    def _merge_config_files(self, cfg, files):
        for file in files:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(file))
        return cfg

    def _merge_configs(self, cfg, configs):
        for config in configs:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(config))
        return cfg


def add_config_path_args(parser):
    parser.add_argument(
        "--log_dir",
        default=get_default_log_dir(),
        help="Directory to save experiment logs, configuration and model checkpoints",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Path to yaml configuration",
    )
    parser.add_argument(
        "--options", "-opt", nargs="+", help="Specify config options from command line.", default=[],
    )
    parser.add_argument(
        "--print_config", action="store_true", help="print config and exit"
    )
    parser.add_argument(
        "--no_auto_reload_config", action='store_true', help='Disable automatically loading config from log_dir'
    )

    return parser


def parse_config(args):
    if os.path.isfile(p := os.path.join(args.log_dir, "config.yaml")) and not args.no_auto_reload_config:
        print("Found config in log dir, using this config.")
        return OmegaConf.load(p)

    config = OmegaConf.create(dict(log_dir=args.log_dir))
    if args.config:
        config = OmegaConf.merge(config, OmegaConf.load(args.config))
        config.pop("log_dir", None)
    if args.options:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.options))
    config = OmegaConf.merge(OmegaConf.create({"log_dir": args.log_dir}), config)

    if args.print_config:
        print()
        print(OmegaConf.to_yaml(config))
        sys.exit(0)
    return config
