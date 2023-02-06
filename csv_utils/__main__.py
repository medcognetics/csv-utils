#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .input import AGGREGATOR_REGISTRY, INPUT_REGISTRY
from .run import transform_csv
from .transforms import TRANSFORM_REGISTRY


def main(args: Namespace) -> None:
    df = transform_csv(
        args.tables,
        args.inputs,
        args.aggregator,
        args.transforms,
    )

    if args.dest is None:
        print(df.to_string())
    else:
        path = Path(args.dest)
        if not path.parent.is_dir():
            raise NotADirectoryError(path.parent)
        df.to_csv(path)


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
        default="join",
        choices=AGGREGATOR_REGISTRY.available_keys(),
        help="registered names of aggregation handlers",
    )
    parser.add_argument(
        "-t",
        "--transforms",
        nargs="+",
        default=["noop"],
        choices=TRANSFORM_REGISTRY.available_keys(),
        help="registered names of input transforms",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
