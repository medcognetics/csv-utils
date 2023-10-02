#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypeVar, cast

import pandas as pd
from registry import Registry


InputFunction = Callable[[Path], pd.DataFrame]
Aggregator = Callable[[Sequence[pd.DataFrame]], pd.DataFrame]
TableTransform = Callable[[pd.DataFrame], pd.DataFrame]

INPUT_REGISTRY = Registry("inputs", bound=Callable[..., pd.DataFrame])
AGGREGATOR_REGISTRY = Registry("aggregators", bound=Callable[..., pd.DataFrame])

INPUT_REGISTRY(lambda x: cast(pd.DataFrame, pd.read_csv(x)), name="csv")
INPUT_REGISTRY(lambda x: x, name="noop")

T = TypeVar("T", pd.DataFrame, pd.Series)


def get_study(s: str) -> str:
    return Path(s).parent.name


def get_patient(s: str) -> str:
    return Path(s).parents[1].name


@INPUT_REGISTRY(name="df")
def df_noop(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected `df` to be a DataFrame, found {type(df)}")
    return df


@INPUT_REGISTRY(name="series")
def series_noop(inp: pd.Series) -> pd.Series:
    if not isinstance(inp, pd.Series):
        raise TypeError(f"Expected `inp` to be a Series, found {type(inp)}")
    return inp


@INPUT_REGISTRY(name="df-or-series")
def df_or_series(inp: T) -> T:
    if not isinstance(inp, (pd.Series, pd.DataFrame)):
        raise TypeError(f"Expected `inp` to be a DataFrame or Series, found {type(inp)}")
    return inp


@INPUT_REGISTRY(name="stats-csv")
def stats_csv(path: Path, stem: bool = False) -> pd.DataFrame:
    r"""Reads a pre-data-organizer stats tool CSV"""
    df = cast(pd.DataFrame, pd.read_csv(path, dtype={"Data Source Case ID": str, "Study Path": str}))
    if stem:
        df["case"] = df["Study Path"].apply(lambda p: Path(p).name)
        df.set_index("case", inplace=True)
    else:
        df.set_index("Study Path", inplace=True)
    return df


@INPUT_REGISTRY(name="scores-csv")
def scores_csv(path: Path, stem: bool = False) -> pd.DataFrame:
    r"""Reads a pre-triage CLI output from medcog-efficientdet"""
    df = cast(pd.DataFrame, pd.read_csv(path, index_col="cases", dtype={"cases": str}))
    if stem:
        df.index = df.index.map(lambda x: Path(x).name)
    return df


@INPUT_REGISTRY(name="data-csv")
def data_organizer_csv(p: Path, index_col: str = "Patient", **kwargs) -> pd.DataFrame:
    r"""Reads CSVs from the data organizer."""
    df = pd.read_csv(p, index_col="Patient", **kwargs)
    return cast(pd.DataFrame, df)


@INPUT_REGISTRY(name="triage-csv")
def triage_csv(p: Path) -> pd.DataFrame:
    r"""Reads triage CSVs from medcog-efficientdet"""
    df = cast(pd.DataFrame, pd.read_csv(p))
    df["Patient"] = df["path"].apply(get_patient)
    df.set_index("Patient", inplace=True)
    df["Study"] = df["path"].apply(get_study)
    df["scores"] = df["malign_score"]
    return df


INPUT_REGISTRY(stats_csv, name="stats-csv-stem", stem=True)
INPUT_REGISTRY(scores_csv, name="scores-csv-stem", stem=True)


@INPUT_REGISTRY(name="inference-csv")
def inference_csv(path: Path) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(path, index_col="case", dtype={"case": str}))
    df.rename(columns={"inference": "scores"}, inplace=True)
    return df


@AGGREGATOR_REGISTRY(name="join")
@AGGREGATOR_REGISTRY(name="join-left", how="left")
@AGGREGATOR_REGISTRY(name="join-inner", how="inner")
@AGGREGATOR_REGISTRY(name="join-outer", how="outer")
def join(dataframes: Sequence[pd.DataFrame], **kwargs) -> pd.DataFrame:
    if len(dataframes) < 2:
        raise ValueError(f"Expected `len(dataframes)` >= 2, found {len(dataframes)}")
    first_dataframe, *remaining_dataframes = list(dataframes)
    return first_dataframe.join(remaining_dataframes, **kwargs)


@AGGREGATOR_REGISTRY(name="concat")
def concat(dataframes: Sequence[pd.DataFrame], **kwargs) -> pd.DataFrame:
    if len(dataframes) < 2:
        raise ValueError(f"Expected `len(dataframes)` >= 2, found {len(dataframes)}")
    return pd.concat(dataframes, **kwargs)


@AGGREGATOR_REGISTRY(name="join-or-concat")
def join_or_concat(
    dataframes: Sequence[pd.DataFrame],
    concat_kwargs: Dict[str, Any] = {},
    join_kwargs: Dict[str, Any] = {},
) -> pd.DataFrame:
    r"""Aggregates dataframes by joining or concatenating, depending on whether the dataframes have the same
    column set. Identical column sets will be concatenated, otherwise joined.
    """
    if len(dataframes) < 2:
        raise ValueError(f"Expected `len(dataframes)` >= 2, found {len(dataframes)}")

    # group dataframes by unique column set
    df_groups: Dict[Tuple[str, ...], List[pd.DataFrame]] = {}
    for df in dataframes:
        key = tuple(sorted(df.columns))
        df_groups.setdefault(key, []).append(df)

    # concat dataframes with unique column sets
    concat_dfs = [pd.concat(dfs, **concat_kwargs) for dfs in df_groups.values()]
    assert len(concat_dfs)
    df1 = next(iter(concat_dfs))
    return df1.join(concat_dfs[1:], **join_kwargs)


@AGGREGATOR_REGISTRY(name="noop")
def noop_aggregator(dataframes: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if len(dataframes) != 1:
        raise ValueError("An aggregator must be provided when using multiple input DataFrames")
    return next(iter(dataframes))
