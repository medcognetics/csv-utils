#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
from csv_utils import transform_csv, INPUT_REGISTRY, AGGREGATOR_REGISTRY, TRANSFORM_REGISTRY
from csv_utils.transforms import KeepColumns, DropWhere
from typing import cast, Sequence


# Step 1 - Define and register a reader for your CSV input. Assign index column or make other changes as needed.
# As written, this function is equivalent to the pre-registred function "csv"
@INPUT_REGISTRY(name="my-input")
def read(p: Path) -> pd.DataFrame:
    return cast(pd.DataFrame, pd.read_csv(p))


# Step 2 (Optional) - Define and register an aggregator that joins multiple input DataFrames into a single output.
# This is only needed when using multiple dataframes - otherwise use the pre-registered aggregator "noop".
# As written, this function is equivalent to the pre-registred function "join"
@AGGREGATOR_REGISTRY(name="join", how="inner")
def join(dataframes: Sequence[pd.DataFrame], **kwargs) -> pd.DataFrame:
    if len(dataframes) < 2:
        raise ValueError(f"Expected `len(dataframes)` >= 2, found {len(dataframes)}")
    df1 = next(iter(dataframes))
    return df1.join(list(dataframes[1:]), **kwargs)


# Step 3 - Define and register one or more transforms. These transforms will be applied to the aggregated DataFrame
TRANSFORM_REGISTRY(KeepColumns(columns=["col1", "col2"]), name="filter-cols")
TRANSFORM_REGISTRY(DropWhere(column="col1", value="N/A"), name="drop-col1-na")


# Step 4 - Call transform_csv using any of the callables you just registered, or use the csv-utils CLI interface
output_df = transform_csv(
    sources=[Path("file.csv")],
    types=["my-input"],
    transforms=["filter-cols", "drop-col1-na"],
)
