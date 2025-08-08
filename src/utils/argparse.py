import argparse
import datetime
import inspect
import sys
import types
import typing
from argparse import Action, FileType, Namespace, ArgumentParser
from dataclasses import _HAS_DEFAULT_FACTORY_CLASS, fields, is_dataclass
from enum import Enum
from typing import Any, Container, Iterable, Literal, Sequence

import docstring_parser
import rich
from omegaconf import OmegaConf
from typing_extensions import deprecated


def add_args(
    parser: ArgumentParser,
    callable,
    group: str | None | Literal["auto"] | argparse._ArgumentGroup = "auto",
    ignore_keys=['self'],
    prefix="",
    case="snake",
    **defaults,
):
    """
    Adds arguments from the given callable to the parser.

    Inspired by libraries like Typer and SimpleParsing, lets the user define a CLI based on the call signature and
    doctring of a function or class.

    Args:
        parser: the argument parser to add the arguments to.
        callable: the object whose signature to inspect.
    """

    parsed_doc = docstring_parser.parse(callable.__doc__ or "")
    doc_params = {arg.arg_name: arg.description for arg in parsed_doc.params}

    if group is not None:
        if group == "auto":
            group = callable.__name__
            parser = parser.add_argument_group(group, parsed_doc.short_description)
        elif isinstance(group, str):
            parser = parser.add_argument_group(group, parsed_doc.short_description)
        elif isinstance(group, argparse._ArgumentGroup):
            parser = group

    parameters = inspect.signature(callable).parameters

    for key, param in parameters.items():

        parse_kw = {}

        if key in ignore_keys:
            continue

        if key in defaults:
            parse_kw["default"] = defaults[key]
            parse_kw["required"] = False
        else:
            if param.default is inspect._empty:
                parse_kw["default"] = None
                parse_kw["required"] = True
            elif type(param.default) is _HAS_DEFAULT_FACTORY_CLASS:
                fields_ = {field_.name: field_ for field_ in fields(callable)}
                parse_kw["default"] = fields_[key].default_factory()
                parse_kw["required"] = False
            else:
                parse_kw["default"] = param.default
                parse_kw["required"] = False

        parse_kw["help"] = doc_params.get(key, None)

        if param.annotation is inspect._empty:
            raise ValueError(f"Can't add args without annotations")

        # handle list etc.
        if type(param.annotation) == types.GenericAlias:
            if hasattr(param.annotation, "__origin__"):
                if issubclass(param.annotation.__origin__, Iterable):
                    parse_kw["nargs"] = "+"
                    parse_kw["type"] = param.annotation.__args__[0]
                else:
                    raise ValueError(f"Can't parse type annotation {param.annotation}")

        # hande Union type with None
        elif type(param.annotation) == types.UnionType:
            _args = param.annotation.__args__
            if len(_args) != 2:
                raise ValueError(f"Can't parse type annotation {param.annotation}")
            if _args[1] is type(None):
                # hasattr(param.annotation, "__args__") and param.annotation.__args__[1] is type(None):
                parse_kw["type"] = param.annotation.__args__[0]
                parse_kw["required"] = False
            else:
                raise ValueError(f"Can't parse type annotation {param.annotation}")

        # handle Literal types with choices
        elif type(param.annotation) == typing._LiteralGenericAlias:
            _literals = param.annotation.__args__
            _type = type(_literals[0])
            for l in _literals:
                if l is None:
                    parse_kw["required"] = False
                elif type(l) != _type:
                    raise ValueError(f"Can't parse type annotation {param.annotation}")
            parse_kw["type"] = _type
            parse_kw["choices"] = _literals

        # handle enum types
        elif issubclass(param.annotation, Enum):
            # raise NotImplementedError("Enum types are not yet supported")
            parse_kw["type"] = _ParseEnum(param.annotation)
            parse_kw["choices"] = list(param.annotation)
            parse_kw[
                "help"
            ] += f" - Choose by name {tuple(param.annotation.__members__.keys())}."

        elif param.annotation == bool:
            parse_kw["type"] = bool_flag

        else:
            parse_kw["type"] = param.annotation

        # handle passing "null" string from command line
        if not parse_kw["required"]:
            parse_kw["type"] = _ParseOptional(parse_kw["type"])

        parse_kw["dest"] = key

        if case == "kebab":
            key = key.replace("_", "-")

        parser.add_argument(f"--{prefix}{key}", **parse_kw)


@deprecated("Use add args on the class instead.")
def add_class_args(parser, cls, ignore_keys=[], group=None):
    group = group or cls.__name__
    ignore_keys = ignore_keys.copy()
    ignore_keys.append("self")
    add_args(parser, cls.__init__, ignore_keys, group)


def get_kwargs(args, callable, prefix=""):
    parameters = inspect.signature(callable).parameters
    kw = {}
    for key in parameters.keys():
        if hasattr(args, f"{prefix}{key}"):
            kw[key] = getattr(args, f"{prefix}{key}")
    return kw


def call_from_args(args, callable, prefix=""):
    kw = get_kwargs(args, callable, prefix)
    return callable(**kw)


def quick_cli(func):
    """Wrap a function to automatically parse arguments from the command line."""

    def wrapper():
        parser = argparse.ArgumentParser()
        add_args(parser, func)
        args = parser.parse_args()
        kw = get_kwargs(args, func)
        return func(**kw)

    return wrapper


def load_confs(paths):
    c = OmegaConf.create({})
    for value in paths:
        c = OmegaConf.merge(c, OmegaConf.load(value))
    return c


def load_structured_confs(dataclasses):
    conf = None
    for dataclass in dataclasses:
        if conf is None:
            conf = OmegaConf.structured(dataclass)
        else:
            conf = OmegaConf.merge(conf, OmegaConf.structured(conf))
    return conf


def format_dir_string(string):
    """replace the %d and %t with the current date and time"""
    new_output_dir = string.replace("%d", datetime.datetime.now().strftime("%Y-%m-%d"))
    new_output_dir = new_output_dir.replace(
        "%t", datetime.datetime.now().strftime("%H-%M-%S")
    )
    return new_output_dir


def bool_flag(string):
    """Parse boolean flags from the command line."""
    if string.lower() in ["true", "1", "yes", "y"]:
        return True
    elif string.lower() in ["false", "0", "no", "n"]:
        return False
    else:
        raise ValueError(f"Can't parse bool flag {string}")


class _ParseEnum:
    def __init__(self, enum_class):
        self.enum_class = enum_class

    def __call__(self, str):
        return self.enum_class.__members__[str]


class _ParseOptional:
    def __init__(self, parse_fn):
        self.parse_fn = parse_fn

    def __call__(self, str):
        if str in ["None", "none", "null"]:
            return None
        return self.parse_fn(str)


class LoadYamlConf(Action):
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        nargs: int | str | None = "+",
        default=[],
        required: bool = False,
        help: str | None = None,
        metavar: str | tuple[str, ...] | None = None,
        resolve: bool = False,
    ) -> None:
        self.resolve = resolve

        if not isinstance(default, Sequence):
            default = [default]
        if len(default) == 0 or isinstance(default[0], str):
            default = load_confs(default)
        elif is_dataclass(default[0]):
            default = load_structured_confs(default)
        if self.resolve:
            OmegaConf.resolve(default)

        super().__init__(
            option_strings,
            dest,
            nargs="+",
            default=default,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        from omegaconf import OmegaConf

        c = load_confs(values)
        c = OmegaConf.to_object(c)

        setattr(namespace, self.dest, c)


class UnstructuredConf(Action):
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        nargs: int | str | None = "+",
        default=[],
        required: bool = False,
        help: str | None = None,
        metavar: str | tuple[str, ...] | None = "key=value",
    ) -> None:
        super().__init__(
            option_strings,
            dest,
            nargs=nargs,
            default=default,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        assert hasattr(namespace, self.dest), f"Error in parsing arguments"
        if hasattr(namespace, self.dest):
            c = getattr(namespace, self.dest)
        else:
            c = OmegaConf.create({})
        c = OmegaConf.merge(c, OmegaConf.from_dotlist(values))
        setattr(namespace, self.dest, c)

        parser._defaults[self.dest] = c


def get_omegaconf_cli_args(default_configs=[], **parser_kw):
    parser = argparse.ArgumentParser(add_help=False, **parser_kw)
    parser.add_argument(
        "--config",
        "-c",
        nargs="+",
        default=default_configs,
        help="Path to one or more config files",
    )
    parser.add_argument(
        "--overrides", "-o", nargs="*", default=[], help="Overrides to config"
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and exit"
    )
    args = parser.parse_args()

    conf = OmegaConf.create({})
    for c in args.config:
        conf = OmegaConf.merge(conf, OmegaConf.load(c))
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args.overrides))

    if args.help:
        print("=========================================================")
        parser.print_help()
        print("\n====== CONFIG OPTIONS ======\n")
        rich.print(OmegaConf.to_yaml(conf))
        print("=========================================================")
        sys.exit(0)

    return conf


OmegaConf.register_new_resolver("dir", format_dir_string)


if __name__ == "__main__":
    # test case
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    from dataclasses import dataclass, field, fields

    @dataclass
    class A:
        b: list[str] = field(default_factory=list)

    def my_function(
        a: int | None,
        b: str = "a",
        c: Literal[1, 2, 3, None] = None,
        d: bool = False,
        e: float = 0.0,
        f: list[str] = [],
    ):
        """
        Short description of the function.

        Args:
            a: a help
            b: b help
            c: c help
            d: d help
            e: e help
            f: f help
        """
        pass

    add_args(parser, A, group="auto")
