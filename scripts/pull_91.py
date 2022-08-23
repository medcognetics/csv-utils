#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from csv_utils import TRANSFORM_REGISTRY, transform_csv
from csv_utils.transforms import KeepWhere, KeepColumns, DropWhere
from csv_utils.__main__ import main, parse_args
from functools import partial


Keep = KeepColumns(["Study Path", "Malignant", "Ground Truther 1", "Ground Truther 2", "scores", "predicted labels"])
TRANSFORM_REGISTRY(Keep, name="keep")

Keep = KeepColumns(["Study Path", "Ground Truther 1", "scores", "predicted labels"])
TRANSFORM_REGISTRY(Keep, name="keep-gt1")

Keep = KeepColumns(["Study Path", "Ground Truther 2", "scores", "predicted labels"])
TRANSFORM_REGISTRY(Keep, name="keep-gt2")

Keep = KeepColumns(["Study Path", "scores", "predicted labels"])
TRANSFORM_REGISTRY(Keep, name="keep-alg")

@TRANSFORM_REGISTRY(name="rename-label")
def rename_label(table: pd.DataFrame) -> pd.DataFrame:
    table["predicted labels"] = table["predicted labels"].apply(lambda x: "malignant" if str(x) == "1" else "benign")
    return table


if __name__ == "__main__":
    main(parse_args())
