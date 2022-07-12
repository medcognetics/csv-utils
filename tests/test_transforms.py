#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest

from csv_utils.transforms import DropWhere, KeepColumns, KeepWhere, NoopTransform


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
        pytest.param("nocol", 1, False, True, 1, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_keep_where(df_factory, col, value, as_str, allow_empty, exp):
    df = df_factory()
    result = KeepWhere(col, value, as_str, allow_empty)(df)
    assert len(result) == exp


def test_keep_where_from_df(df_factory):
    df = df_factory()
    result = {KeepWhere.format_name(*k): v for k, v in KeepWhere.from_dataframe(df).items()}
    assert len(result) == 30
    assert all(isinstance(k, str) for k in result.keys())
    assert all(isinstance(v, KeepWhere) for v in result.values())


@pytest.mark.parametrize(
    "col,value,as_str,allow_empty,exp",
    [
        pytest.param("col1", 0, False, True, 9),
        pytest.param("col1", "0", True, True, 9),
        pytest.param("col1", "0", False, True, 10),
        pytest.param("col2", 1, False, True, 9),
        pytest.param("col1", 1, False, False, 10, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param("nocol", 1, False, True, 10, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_drop_where(df_factory, col, value, as_str, allow_empty, exp):
    df = df_factory()
    result = DropWhere(col, value, as_str, allow_empty)(df)
    assert len(result) == exp
