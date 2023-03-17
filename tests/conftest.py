#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def df_factory():
    def func(
        columns: Iterable[str] = ["col1", "col2", "col3"], num_rows: int = 10, as_str: bool = False
    ) -> pd.DataFrame:
        columns = list(columns)
        rows = []
        for r_idx in range(num_rows):
            entry = {}
            for c_idx, c in enumerate(columns):
                val = r_idx * len(columns) + c_idx
                if as_str:
                    val = f"entry-{val}"
                entry[c] = val
            rows.append(entry)
        return pd.DataFrame(rows)

    return func


@pytest.fixture
def setup_basic_test(tmp_path, df_factory):
    stats_df = df_factory(["Data Source Case ID", "Study Path", "Ground Truth"], as_str=True)
    scores_df = df_factory(["Study", "cases", "score"], as_str=True)

    stats_path = Path(tmp_path, "stats.csv")
    stats_df.to_csv(stats_path, index=False)

    scores_path = Path(tmp_path, "scores.csv")
    scores_df.to_csv(scores_path, index=False)

    return {
        "stats": stats_path,
        "scores": scores_path,
    }
