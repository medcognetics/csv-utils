#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from registry import Registry


OutputFunction = Callable[[pd.DataFrame], Any]

OUTPUT_REGISTRY = Registry("output")


@OUTPUT_REGISTRY(name="print")
def print_output(df: pd.DataFrame, path: Optional[Path]) -> None:
    print(df.to_string())


@OUTPUT_REGISTRY(name="csv")
def to_csv(df: pd.DataFrame, path: Optional[Path], **kwargs) -> None:
    if path is None:
        raise ValueError("path must be provided to `to_csv`")
    df.to_csv(path, **kwargs)


@OUTPUT_REGISTRY(name="excel")
def to_excel(df: pd.DataFrame, path: Optional[Path], **kwargs) -> None:
    if path is None:
        raise ValueError("path must be provided to `to_excel`")
    df.to_excel(path, **kwargs)


@OUTPUT_REGISTRY(name="pdb")
def pdb(df: pd.DataFrame, path: Optional[Path]) -> None:
    import pdb

    pdb.set_trace()
