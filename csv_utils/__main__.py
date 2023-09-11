#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .input import AGGREGATOR_REGISTRY, INPUT_REGISTRY
from .output import OUTPUT_REGISTRY
from .run import transform_csv
from .transforms import TRANSFORM_REGISTRY


def main(args: Namespace) -> None:
    df = transform_csv(
        args.tables,
        args.inputs,
        args.aggregator,
        args.transforms,
    )
    output = OUTPUT_REGISTRY.get(args.output).instantiate_with_metadata()
    output(df, args.dest)


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="csv-utils")
    parser.add_argument("tables", nargs="+", type=Path, help="paths to files with table of entries")
    parser.add_argument("-d", "--dest", type=Path, default=None, help="destination for ouptuts")
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        default=["csv"],
        choices=INPUT_REGISTRY.available_keys(),
        help="registered names of input handlers",
    )
    parser.add_argument(
        "-a",
        "--aggregator",
        default="join-or-concat",
        choices=AGGREGATOR_REGISTRY.available_keys(),
        help="registered names of aggregation handlers",
    )
    parser.add_argument(
        "-t",
        "--transforms",
        nargs="+",
        default=[],
        choices=TRANSFORM_REGISTRY.available_keys(),
        help="registered names of input transforms",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="txt",
        choices=OUTPUT_REGISTRY.available_keys(),
        help="registered names of output handlers",
    )
    return parser.parse_args()


def entrypoint():
    main(parse_args())


if __name__ == "__main__":
    entrypoint()
