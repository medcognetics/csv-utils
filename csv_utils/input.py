#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Callable, Sequence, cast

import pandas as pd
from registry import Registry


InputFunction = Callable[[Path], pd.DataFrame]
Aggregator = Callable[[Sequence[pd.DataFrame]], pd.DataFrame]
TableTransform = Callable[[pd.DataFrame], pd.DataFrame]

INPUT_REGISTRY = Registry("inputs", bind_metadata=True)
AGGREGATOR_REGISTRY = Registry("aggregators")

INPUT_REGISTRY(pd.read_csv, name="csv")
INPUT_REGISTRY(pd.read_csv, name="stats-csv", index_col="Study Path", dtype={"Data Source Case ID": str})
INPUT_REGISTRY(pd.read_csv, name="scores-csv", index_col="cases")
INPUT_REGISTRY(lambda x: x, name="noop")


@INPUT_REGISTRY(name="stats-csv")
def stats_csv(path: Path) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(path, dtype={"Data Source Case ID": str, "Study Path": str}))
    df["case"] = df["Study Path"].apply(lambda p: Path(p).name)
    df.set_index("case", inplace=True)
    return df


@INPUT_REGISTRY(name="inference-csv")
def inference_csv(path: Path) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(path, index_col="case", dtype={"case": str}))
    df.rename(columns={"inference": "scores"}, inplace=True)
    return df


@AGGREGATOR_REGISTRY(name="join", how="inner")
def join(dataframes: Sequence[pd.DataFrame], **kwargs) -> pd.DataFrame:
    if len(dataframes) < 2:
        raise ValueError(f"Expected `len(dataframes)` >= 2, found {len(dataframes)}")
    df1 = next(iter(dataframes))
    return df1.join(list(dataframes[1:]), **kwargs)
