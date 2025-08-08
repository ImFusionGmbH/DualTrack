import argparse
from copy import copy
import datetime
from enum import Enum
import inspect
import json
import os
from dataclasses import asdict, dataclass, field, fields
import dataclasses
from pathlib import Path
import sys
import types
from typing_extensions import deprecated
from omegaconf import OmegaConf

import __main__
from typing import Any, Callable, Literal, Optional, List, Union
import wandb
from sys import exit
import yaml

from src.logger import get_default_log_dir

# Make the datasets.yaml path configurable
DATASET_INFO_PATH = os.environ.get("DATASET_INFO_PATH", "data/datasets.yaml")

try:
    if os.path.exists(DATASET_INFO_PATH):
        DATASET_INFO = yaml.load(open(DATASET_INFO_PATH), yaml.FullLoader)
    else:
        # Fallback to empty dict if file doesn't exist
        DATASET_INFO = {}
        print(f"Warning: {DATASET_INFO_PATH} not found. Using empty dataset info.")
except Exception as e:
    DATASET_INFO = {}
    print(f"Warning: Could not load {DATASET_INFO_PATH}: {e}. Using empty dataset info.")


try:
    script_name = Path(__main__.__file__).name.replace(".py", "")
except:
    script_name = "run"


def get_log_dir():
    return datetime.datetime.now().strftime(
        os.path.join(
            "experiments",
            "%Y-%b-%d",
            "%-I%p:%M",
        )
    )


@dataclass
class BaseConfig:
    """Base class for all configs"""

    name: Optional[str] = field(default=None, init=False)
    script: Optional[str] = field(default=None, init=False)

    _ignore_parse = []

    def __post_init__(self):
        if not self.script:
            try:
                self.script = inspect.getfile(self.__class__)
            except:
                self.script = "unknown"

        if not self.name:
            self.name = os.path.basename(self.script).replace(".py", "")

    @classmethod
    def from_dict(cls, config: dict, strict=False):
        field_names_init = [f.name for f in fields(cls) if f.init]
        field_names_no_init = [f.name for f in fields(cls) if not f.init]
        field_names = field_names_init + field_names_no_init

        used = {k: v for k, v in config.items() if k in field_names}
        unused = {k: v for k, v in config.items() if k not in field_names}
        if unused:
            print(f"Unused config keys: {unused.keys()}")
        if strict:
            assert not unused, f"Unused config keys: {unused.keys()}"

        out = cls(**{k: v for k, v in used.items() if k in field_names_init})
        for k in field_names_no_init:
            setattr(out, k, used.get(k, getattr(out, k)))
        return out

    @classmethod
    def from_yaml(cls, path: str, new_log_dir=True):
        with open(path, "r") as f:
            config = yaml.load(f, yaml.FullLoader)
        return cls.from_dict(config, new_log_dir)

    @classmethod
    def from_wandb_path(cls, path: str, new_log_dir=True):
        import wandb

        run = wandb.Api().run(path)
        config = run.config
        return cls.from_dict(config, new_log_dir)

    @classmethod
    def from_cli(cls, args=None, parent_parsers=[]):
        parser = argparse.ArgumentParser(add_help=False, parents=parent_parsers)
        group = parser.add_argument_group("Configuration")
        group.add_argument(
            "-c", "--config", help="If set, will load the config from the given file"
        )
        group.add_argument(
            "--wandb_path",
            help="If set, try to load the config from the wandb run specified by wandb_path",
        )
        group.add_argument(
            "--resume_wandb",
            action="store_true",
            help="If set, will resume the wandb run specified by wandb_path",
        )
        group.add_argument(
            "--overrides",
            "-o",
            nargs="+",
            default=[],
            help="Override config values with the given key=value pairs",
        )
        group.add_argument(
            "--dump_config",
            type=str,
            help="If set, dumps the config as a yaml to the given path and exits.",
        )
        group.add_argument("--log_dir", default=get_log_dir())
        group.add_argument(
            "--keep_log_dir",
            action="store_true",
            help="Keep the log directory the same as the loaded config",
        )
        group.add_argument("--help", "-h", action="store_true")
        args = parser.parse_args(args)

        if args.resume_wandb:
            args.keep_log_dir = True
            os.environ["WANDB_RUN_ID"] = wandb.Api().run(args.wandb_path).id
            os.environ["WANDB_RESUME"] = "allow"

        conf = OmegaConf.create(asdict(cls()))
        if args.config:
            conf = OmegaConf.merge(conf, OmegaConf.load(args.config))

        if args.wandb_path:
            wandb_conf = OmegaConf.create(wandb.Api().run(args.wandb_path).config)
            conf = OmegaConf.merge(conf, wandb_conf)

        if args.overrides:
            conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args.overrides))

        if not args.keep_log_dir:
            conf.log_dir = args.log_dir

        # instantiate the dataclass to run any post init methods - then convert back to OmegaConf for printing
        # and for saving to disk
        print(conf.log_dir)
        conf = OmegaConf.create(asdict(cls.from_dict(OmegaConf.to_object(conf))))
        print(conf.log_dir)

        if args.help:
            parser.print_help()
            print("==========   CONFIG   ==========")
            print(OmegaConf.to_yaml(conf))
            print("==========   END CONFIG   ==========")
            exit()

        if args.dump_config:
            with open(args.dump_config, "w") as f:
                f.write(OmegaConf.to_yaml(conf))
            exit()

        return cls.from_dict(OmegaConf.to_object(conf))

    def print(self):
        print("==========   CONFIG   ==========")
        print(OmegaConf.to_yaml(asdict(self)))
        print("==========   END CONFIG   ==========")

    @classmethod
    def get_parser(cls, parser=None, add_help=False, **kwargs):
        parser = parser or argparse.ArgumentParser(
            add_help=add_help,
            **kwargs,
        )
        for field in fields(cls):
            extra_kwargs = {}

            if field.metadata.get("no_argparse"):
                continue
            if field.name in cls._ignore_parse:
                continue
            if not field.init:
                continue
            help = field.metadata.get("help") or ""
            extra_kwargs["help"] = f"{help} (default: %(default)s)"
            _parse_dataclass_field(parser, field, **extra_kwargs)
        return parser

    @classmethod
    def from_args(cls, args):
        return cls.from_dict(vars(args))


# class ArgumentParser(argparse.ArgumentParser):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.add_argument("--wandb_path")
#         self.add_argument("--config_path")
#         self.add_argument("--resume_wandb")
#
#     def parse_known_args(
#         self, arg_strings: List[str], namespace: argparse.Namespace
#     ) -> tuple[argparse.Namespace, list[str]]:
#         temp_parser = argparse.ArgumentParser(add_help=False)
#         temp_parser.add_argument("--wandb_path")
#         temp_parser.add_argument("--config_path")
#         temp_parser.add_argument("--resume_wandb")
#         temp_args, _ = temp_parser.parse_known_args()
#
#         if temp_args.wandb_path:
#             print("hello")
#             from wandb import Api
#
#             run = Api().run(temp_args.wandb_path)
#             cfg = run.config
#             self.set_defaults(**cfg)
#
#         return super().parse_known_args(arg_strings, namespace)


# from huggingface CLI
def _parse_dataclass_field(
    parser: argparse.ArgumentParser, field: dataclasses.Field, **extra_kwargs
):
    # Long-option strings are conventionlly separated by hyphens rather
    # than underscores, e.g., "--long-format" rather than "--long_format".
    # Argparse converts hyphens to underscores so that the destination
    # string is a valid attribute name. Hf_argparser should do the same.
    long_options = [f"--{field.name}"]
    if "_" in field.name:
        long_options.append(f"--{field.name.replace('_', '-')}")

    kwargs = field.metadata.copy()
    kwargs.update(extra_kwargs)
    # field.metadata is not used at all by Data Classes,
    # it is provided as a third-party extension mechanism.
    if isinstance(field.type, str):
        raise RuntimeError(
            "Unresolved type detected, which should have been done with the help of "
            "`typing.get_type_hints` method by default"
        )

    aliases = kwargs.pop("aliases", [])
    if isinstance(aliases, str):
        aliases = [aliases]

    origin_type = getattr(field.type, "__origin__", field.type)
    if origin_type is Union or (
        hasattr(types, "UnionType") and isinstance(origin_type, types.UnionType)
    ):
        if str not in field.type.__args__ and (
            len(field.type.__args__) != 2 or type(None) not in field.type.__args__
        ):
            raise ValueError(
                "Only `Union[X, NoneType]` (i.e., `Optional[X]`) is allowed for `Union` because"
                " the argument parser only supports one type per argument."
                f" Problem encountered in field '{field.name}'."
            )
        if type(None) not in field.type.__args__:
            # filter `str` in Union
            field.type = (
                field.type.__args__[0]
                if field.type.__args__[1] is str
                else field.type.__args__[1]
            )
            origin_type = getattr(field.type, "__origin__", field.type)
        elif bool not in field.type.__args__:
            # filter `NoneType` in Union (except for `Union[bool, NoneType]`)
            field.type = (
                field.type.__args__[0]
                if isinstance(None, field.type.__args__[1])
                else field.type.__args__[1]
            )
            origin_type = getattr(field.type, "__origin__", field.type)

    # A variable to store kwargs for a boolean field, if needed
    # so that we can init a `no_*` complement argument (see below)
    bool_kwargs = {}
    if origin_type is Literal or (
        isinstance(field.type, type) and issubclass(field.type, Enum)
    ):
        if origin_type is Literal:
            kwargs["choices"] = field.type.__args__
        else:
            kwargs["choices"] = [x.value for x in field.type]

        kwargs["type"] = make_choice_type_function(kwargs["choices"])

        if field.default is not dataclasses.MISSING:
            kwargs["default"] = field.default
        else:
            kwargs["required"] = True
    elif field.type is bool or field.type == Optional[bool]:
        # Copy the currect kwargs to use to instantiate a `no_*` complement argument below.
        # We do not initialize it here because the `no_*` alternative must be instantiated after the real argument
        bool_kwargs = copy(kwargs)

        # Note: Using string parsing for boolean values because argparse type=bool doesn't behave as expected
        kwargs["type"] = string_to_bool
        if field.type is bool or (
            field.default is not None and field.default is not dataclasses.MISSING
        ):
            # Default value is False if we have no default when of type bool.
            default = False if field.default is dataclasses.MISSING else field.default
            # This is the value that will get picked if we don't include --{field.name} in any way
            kwargs["default"] = default
            # This tells argparse we accept 0 or 1 value after --{field.name}
            kwargs["nargs"] = "?"
            # This is the value that will get picked if we do --{field.name} (without value)
            kwargs["const"] = True
    elif inspect.isclass(origin_type) and issubclass(origin_type, list):
        kwargs["type"] = field.type.__args__[0]
        kwargs["nargs"] = "+"
        if field.default_factory is not dataclasses.MISSING:
            kwargs["default"] = field.default_factory()
        elif field.default is dataclasses.MISSING:
            kwargs["required"] = True
    else:
        kwargs["type"] = field.type
        if field.default is not dataclasses.MISSING:
            kwargs["default"] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            kwargs["default"] = field.default_factory()
        else:
            kwargs["required"] = True
    parser.add_argument(*long_options, *aliases, **kwargs)

    # Add a complement `no_*` argument for a boolean field AFTER the initial field has already been added.
    # Order is important for arguments with the same destination!
    # We use a copy of earlier kwargs because the original kwargs have changed a lot before reaching down
    # here and we do not need those changes/additional keys.
    if field.default is True and (field.type is bool or field.type == Optional[bool]):
        bool_kwargs["default"] = False
        parser.add_argument(
            f"--no_{field.name}",
            f"--no-{field.name.replace('_', '-')}",
            action="store_false",
            dest=field.name,
            **bool_kwargs,
        )


def make_choice_type_function(choices: list) -> Callable[[str], Any]:
    """
    Creates a mapping function from each choices string representation to the actual value. Used to support multiple
    value types for a single argument.

    Args:
        choices (list): List of choices.

    Returns:
        Callable[[str], Any]: Mapping function from string representation to actual value for each choice.
    """
    str_to_choice = {str(choice): choice for choice in choices}
    return lambda arg: str_to_choice.get(arg, arg)


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


@dataclass
class BaseTrainingArgs(BaseConfig):

    log_dir: str = get_log_dir()
    logger: Literal["console", "tensorboard", "wandb"] = "wandb"

    dataset: str = field(default="tus-rec")
    val_datasets: Optional[List[str]] = None
    num_dataloader_workers: int = 4
    seed: int = 0
    batch_size: int = 32
    epochs: int = 50
    warmup_epochs: int = 0
    device: str = "cuda"
    use_bfloat: bool = False
    use_amp: bool = False
    scheduler: Literal["cosine", "none", "warmup_cosine"] = "cosine"
    clip_grad: float = 3.0
    optimizer: Literal["adam", "sgd", "adagrad"] = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    model_weights: Optional[str] = None
    disable_checkpoint: bool = False
    debug: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.debug:
            os.environ["WANDB_RUN_NAME"] = "debug"
            self.log_dir = self.log_dir.replace(
                "experiments", os.path.join("experiments", "debug")
            )
            self.disable_checkpoint = True


def get_training_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--name", default=script_name)
    parser.add_argument(
        "--dataset",
        choices=DATASET_INFO.keys(),
        default=next(iter(DATASET_INFO.keys())),
    )
    parser.add_argument(
        "--val_datasets",
        choices=DATASET_INFO.keys(),
        nargs="+",
        default=[],
        help="Specify the validation datasets to use. If not set, will use the validation set corresponding to DATASET.",
    )
    parser.add_argument("--num_dataloader_workers", default=8, type=int)
    parser.add_argument("--log_dir", default=get_default_log_dir())
    parser.add_argument(
        "--logger", choices=("wandb", "tensorboard", "console"), default="wandb"
    )

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_bfloat", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--scheduler",
        default="cosine",
        choices=["cosine", "none"],
        help="Learning rate scheduler.",
    )
    parser.add_argument("--clip_grad", type=float, default=3)
    parser.add_argument(
        "--optimizer", choices=("adam", "sgd", "adagrad"), default="adam"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument(
        "--model_weights", help="Path to the model weights path to load"
    )
    parser.add_argument("--debug", action="store_true")

    return parser


def post_process_training_args(args):
    if args.log_dir is None:
        args.log_dir = (
            (Path("debug") if args.debug else Path("experiments"))
            / args.dataset
            / "%Y-%b-%d"
            / f"%-I%p:%M_{args.name}"
        )
        args.log_dir = datetime.datetime.now().strftime(str(args.log_dir))

    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        args.logger = "console"
        args.num_dataloader_workers = 1
        args.name += "_debug"

    if args.val_datasets == []:
        args.val_datasets.append(args.dataset)

    import yaml

    print(json.dumps(vars(args), indent=4))

    return args
