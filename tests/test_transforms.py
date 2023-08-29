#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from csv_utils.transforms import (
    Discretize,
    DropWhere,
    GroupValues,
    KeepColumns,
    KeepWhere,
    NoopTransform,
    RenameColumn,
    RenameValue,
)


def test_noop_transform(df_factory):
    df = df_factory()
    result = NoopTransform()(df)
    assert result is df


@pytest.mark.parametrize(
    "cols",
    [
        ["col1"],
        ["col1", "col2"],
        ["col1", "NOT_PRESENT"],
    ],
)
def test_keep_columns(df_factory, cols):
    df = df_factory()
    result = KeepColumns(cols)(df)
    assert list(result.columns) == [c for c in cols if c in df.columns]


@pytest.mark.parametrize(
    "col,value,as_str,allow_empty,exp",
    [
        pytest.param("col1", 0, False, True, 1),
        pytest.param("col1", "0", True, True, 1),
        pytest.param("col1", "0", False, True, 0),
        pytest.param("col2", 1, False, True, 1),
        pytest.param("col1", 1, False, False, 0, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param("nocol", 1, False, True, 1, marks=pytest.mark.xfail(raises=KeyError)),
        pytest.param(["col1", "col2"], [0, 1], False, True, 1),
        pytest.param(["col1", "col2"], [0, 0], False, True, 0),
        pytest.param(["col1", "col2"], [1, 1], False, True, 0),
        pytest.param(["col1", "col2"], [1], False, True, 0, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(["col1", "col2"], 1, False, True, 0, marks=pytest.mark.xfail(raises=TypeError)),
    ],
)
def test_keep_where(df_factory, col, value, as_str, allow_empty, exp):
    df = df_factory()
    result = KeepWhere(col, value, as_str, allow_empty)(df)
    assert len(result) == exp


@pytest.mark.parametrize(
    "col,value,exp",
    [
        pytest.param("value", "val", True),
        pytest.param("value", "lue", True),
        pytest.param("value", "foo", False),
        pytest.param("1,2,3", "1", True),
    ],
)
def test_keep_where_contains(col, value, exp):
    df = pd.DataFrame({"colname": [col]})
    result = KeepWhere("colname", value, contains=True, allow_empty=True)(df)
    assert bool(len(result)) == exp


@pytest.mark.parametrize(
    "col,value,exp",
    [
        pytest.param(None, None, True),
        pytest.param(pd.NA, None, True),
        pytest.param("foo", None, False),
    ],
)
def test_keep_where_empty(col, value, exp):
    df = pd.DataFrame({"colname": [col]})
    result = KeepWhere("colname", value, allow_empty=True)(df)
    assert bool(len(result)) == exp


def test_keep_where_from_df(df_factory):
    df = df_factory()
    result = {KeepWhere.format_name(*k): v for k, v in KeepWhere.from_dataframe(df).items()}
    assert len(result) == 30
    assert all(isinstance(k, str) for k in result.keys())
    assert all(isinstance(v, KeepWhere) for v in result.values())


@pytest.mark.parametrize("allow_missing", [True, pytest.param(False, marks=pytest.mark.xfail(raises=KeyError))])
def test_keep_where_missing_column(df_factory, allow_missing):
    df = df_factory()
    result = {
        KeepWhere.format_name(*k): v
        for k, v in KeepWhere.from_dataframe(df, columns=["foo"], allow_missing_column=allow_missing).items()
    }
    assert not result


@pytest.mark.parametrize(
    "col,value,as_str,allow_empty,exp",
    [
        pytest.param("col1", 0, False, True, 9),
        pytest.param("col1", "0", True, True, 9),
        pytest.param("col1", "0", False, True, 10),
        pytest.param("col2", 1, False, True, 9),
        pytest.param("col1", 1, False, False, 10, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param("nocol", 1, False, True, 10, marks=pytest.mark.xfail(raises=KeyError)),
        pytest.param(["col1", "col2"], [0, 1], False, True, 9),
        pytest.param(["col1", "col2"], [0, 0], False, True, 10),
        pytest.param(["col1", "col2"], [1, 1], False, True, 10),
        pytest.param(["col1", "col2"], [1], False, True, 10, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(["col1", "col2"], 1, False, True, 10, marks=pytest.mark.xfail(raises=TypeError)),
    ],
)
def test_drop_where(df_factory, col, value, as_str, allow_empty, exp):
    df = df_factory()
    result = DropWhere(col, value, as_str, allow_empty)(df)
    assert len(result) == exp


@pytest.mark.parametrize(
    "col,value,exp",
    [
        pytest.param(None, None, False),
        pytest.param(pd.NA, None, False),
        pytest.param("foo", None, True),
    ],
)
def test_drop_where_empty(col, value, exp):
    df = pd.DataFrame({"colname": [col]})
    result = DropWhere("colname", value, allow_empty=True)(df)
    assert bool(len(result)) == exp


@pytest.mark.parametrize("allow_missing", [True, pytest.param(False, marks=pytest.mark.xfail(raises=KeyError))])
def test_drop_where_missing_col(df_factory, allow_missing):
    df = df_factory()
    DropWhere("NOT_HERE", 0, allow_missing_column=allow_missing)(df)


@pytest.mark.parametrize(
    "col,interval,output_colname,exp",
    [
        pytest.param("col1", [0, 2, 4, 6, 8, 10], None, "2 <= x < 4"),
        pytest.param("col1", [0, 1.5, 3.0, 6, 8, 10], None, "1.5 <= x < 3.0"),
        pytest.param("col1", [3, 4, 5], None, "< 3"),
        pytest.param("col1", [0, 1, 2], None, ">= 2"),
        pytest.param("col2", [0, 2, 4, 6, 8, 10], None, "2 <= x < 4"),
        pytest.param("col1", [0, 2, 4, 6, 8, 10], "Dest Column", "2 <= x < 4"),
        pytest.param("col1", [0, 2, 4, 6, 8, 10], "col1", "2 <= x < 4"),
        pytest.param("col1", [0, 2, 4, 6, 8, 10], "col1", "2 <= x < 4"),
    ],
)
def test_discretize(df_factory, col, interval, output_colname, exp):
    np.random.seed(42)
    df = df_factory()
    df[col] = np.random.rand(len(df[col])) * 10
    df.loc[0, col] = 2.0
    df.loc[1, col] = pd.NA
    transform = Discretize(col, interval, output_colname)
    result = transform(df)
    assert len(result) == len(df)

    output_colname = output_colname or col
    target_col = df[output_colname]
    assert target_col.loc[0] == exp


@pytest.mark.parametrize(
    "col,values,dest,exp",
    [
        pytest.param("col1", [0, 1, 2], "grouped", 1),
        pytest.param("col1", [0, 3, 6], "grouped", 3),
        pytest.param("col1", [0, 3, 6], "foo", 3),
        pytest.param("missing", [0, 3, 6], "foo", 3, marks=pytest.mark.xfail(raises=KeyError, strict=True)),
        pytest.param("col1", [1, 4, 7], "foo", 0),
    ],
)
def test_group_values(df_factory, col, values, dest, exp):
    df = df_factory()
    result = GroupValues(col, values, dest)(df)
    assert (result[col] == dest).sum() == exp


@pytest.mark.parametrize(
    "old_name,new_name,exp",
    [
        pytest.param("col1", "new_col1", ["new_col1", "col2", "col3"]),
        pytest.param("col2", "new_col2", ["col1", "new_col2", "col3"]),
        pytest.param("col3", "new_col3", ["col1", "col2", "new_col3"]),
        pytest.param("not_present", "new_col", ["col1", "col2", "col3"], marks=pytest.mark.xfail(raises=KeyError)),
    ],
)
def test_rename_column(df_factory, old_name, new_name, exp):
    df = df_factory()
    result = RenameColumn(old_name, new_name)(df)
    assert list(result.columns) == exp


@pytest.mark.parametrize(
    "col,old_value,new_value,exp",
    [
        pytest.param("col1", 0, "zero", ["zero", 3, 6, 9, 12, 15, 18, 21, 24, 27]),
        pytest.param("col1", 9, "nine", [0, 3, 6, "nine", 12, 15, 18, 21, 24, 27]),
        pytest.param("col2", 1, "one", ["one", 4, 7, 10, 13, 16, 19, 22, 25, 28]),
        pytest.param(
            "not_present", 0, "zero", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], marks=pytest.mark.xfail(raises=KeyError)
        ),
    ],
)
def test_rename_value(df_factory, col, old_value, new_value, exp):
    df = df_factory()
    result = RenameValue(col, old_value, new_value)(df)
    assert list(result[col]) == exp
