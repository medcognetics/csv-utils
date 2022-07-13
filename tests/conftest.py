#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Iterable

import pandas as pd
import pytest


@pytest.fixture
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
