#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from csv_utils import TRANSFORM_REGISTRY, transform_csv
from csv_utils.transforms import KeepWhere, KeepColumns, DropWhere
from csv_utils.__main__ import main, parse_args
from functools import partial


Keep = KeepColumns(["Study Path", "Malignant", "scores", "predicted labels"])
TRANSFORM_REGISTRY(Keep, name="keep")


@TRANSFORM_REGISTRY(name="add-label", threshold=0.8)
def add_label(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    table["predicted labels"] = table["scores"] >= threshold
    table["predicted labels"] = table["predicted labels"].apply(lambda x: "malignant" if x else "benign")
    return table


@TRANSFORM_REGISTRY(name="rename-label")
def rename_label(table: pd.DataFrame) -> pd.DataFrame:
    table["predicted labels"] = table["predicted labels"].apply(lambda x: "malignant" if str(x) == "1" else "benign")
    return table


if __name__ == "__main__":
    main(parse_args())
