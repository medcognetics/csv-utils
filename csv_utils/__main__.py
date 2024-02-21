#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Callable

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from .input import INPUT_REGISTRY
from .output import OUTPUT_REGISTRY
from .run import transform_csv


StrOrCallable = str | Callable


def main(args: Namespace) -> None:
    df = transform_csv(
        args.tables,
        args.inputs,
        args.aggregator,
        args.transforms,
        args.aggregation_groups,
    )
    output = OUTPUT_REGISTRY.get(args.output).instantiate_with_metadata()
    output(df, args.dest)


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="csv-utils")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("tables", nargs="+", type=Path, help="paths to files with table of entries")
    parser.add_argument("-d", "--dest", type=Path, default=None, help="destination for outputs")
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        default=["csv"],
        choices=INPUT_REGISTRY.available_keys(),
        type=StrOrCallable,
        help="registered names of input handlers",
    )
    parser.add_argument(
        "-a",
        "--aggregator",
        default="join-or-concat",
        type=StrOrCallable,
        help="registered names of aggregation handlers",
    )
    parser.add_argument(
        "-t",
        "--transforms",
        nargs="+",
        default=[],
        type=StrOrCallable,
        help="registered names of input transforms",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="print",
        type=StrOrCallable,
        help="registered names of output handlers",
    )
    parser.add_argument(
        "-g",
        "--aggregation-groups",
        default=[],
        type=int,
        nargs="+",
        help="indices of input sources to aggregate together",
    )
    cfg = parser.parse_args()
    return parser.instantiate_classes(cfg)


def entrypoint():
    main(parse_args())


if __name__ == "__main__":
    entrypoint()
