#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Callable, Sequence, cast

import pandas as pd
from registry import Registry


InputFunction = Callable[[Path], pd.DataFrame]
Aggregator = Callable[[Sequence[pd.DataFrame]], pd.DataFrame]
TableTransform = Callable[[pd.DataFrame], pd.DataFrame]

INPUT_REGISTRY = Registry("inputs", bound=Callable[..., pd.DataFrame])
AGGREGATOR_REGISTRY = Registry("aggregators", bound=Callable[..., pd.DataFrame])

INPUT_REGISTRY(lambda x: cast(pd.DataFrame, pd.read_csv(x)), name="csv")
INPUT_REGISTRY(
    lambda x: cast(pd.DataFrame, pd.read_csv(x, index_col="Study Path", dtype={"Data Source Case ID": str})),
    name="stats-csv",
)
INPUT_REGISTRY(lambda x: x, name="noop")


@INPUT_REGISTRY(name="stats-csv")
def stats_csv(path: Path, stem: bool = False) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(path, dtype={"Data Source Case ID": str, "Study Path": str}))
    if stem:
        df["case"] = df["Study Path"].apply(lambda p: Path(p).name)
        df.set_index("case", inplace=True)
    else:
        df.set_index("Study Path", inplace=True)
    return df


@INPUT_REGISTRY(name="scores-csv")
def scores_csv(path: Path, stem: bool = False) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(path, index_col="cases", dtype={"cases": str}))
    if stem:
        df.index = df.index.map(lambda x: Path(x).name)
    return df


INPUT_REGISTRY(stats_csv, name="stats-csv-stem", stem=True)
INPUT_REGISTRY(scores_csv, name="scores-csv-stem", stem=True)


@INPUT_REGISTRY(name="inference-csv")
def inference_csv(path: Path) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(path, index_col="case", dtype={"case": str}))
    df.rename(columns={"inference": "scores"}, inplace=True)
    return df


@AGGREGATOR_REGISTRY(name="join")
def join(dataframes: Sequence[pd.DataFrame], **kwargs) -> pd.DataFrame:
    if len(dataframes) < 2:
        raise ValueError(f"Expected `len(dataframes)` >= 2, found {len(dataframes)}")
    df1 = next(iter(dataframes))
    return df1.join(list(dataframes[1:]), **kwargs)


@AGGREGATOR_REGISTRY(name="concat")
def concat(dataframes: Sequence[pd.DataFrame], **kwargs) -> pd.DataFrame:
    if len(dataframes) < 2:
        raise ValueError(f"Expected `len(dataframes)` >= 2, found {len(dataframes)}")
    return pd.concat(dataframes, **kwargs)


@AGGREGATOR_REGISTRY(name="noop")
def noop_aggregator(dataframes: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if len(dataframes) != 1:
        raise ValueError("An aggregator must be provided when using multiple input DataFrames")
    return next(iter(dataframes))
