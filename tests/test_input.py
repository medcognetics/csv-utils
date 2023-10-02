#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import pytest

from csv_utils.input import (
    concat,
    data_organizer_csv,
    df_noop,
    join,
    join_or_concat,
    scores_csv,
    stats_csv,
    triage_csv,
)


def test_df_noop(df_factory):
    df = df_factory(columns=["col1", "col2"])
    result = df_noop(df)
    assert result.equals(df)


def test_stats_csv(tmp_path, df_factory):
    df = df_factory(columns=["Data Source Case ID", "Study Path"])
    path = Path(tmp_path, "df.csv")
    df.to_csv(path, index=False)
    result = stats_csv(path)
    assert result.index.name == "Study Path"
    assert "Data Source Case ID" in result.columns
    assert len(result) == len(df)


def test_scores_csv(tmp_path, df_factory):
    df = df_factory(columns=["cases", "scores"])
    df["cases"] = df["cases"].apply(lambda x: Path(f"case{x}"))
    path = Path(tmp_path, "df.csv")
    df.to_csv(path, index=False)
    result = scores_csv(path)
    assert result.index.name == "cases"
    assert "scores" in result.columns
    assert len(result) == len(df)


def test_data_organizer_csv(tmp_path, df_factory):
    df = df_factory(columns=["Patient", "Density"])
    path = Path(tmp_path, "df.csv")
    df.to_csv(path, index=False)
    result = data_organizer_csv(path)
    assert result.index.name == "Patient"
    assert "Density" in result.columns
    assert len(result) == len(df)


def test_triage_csv(tmp_path, df_factory):
    df = df_factory(columns=["path", "score"])
    df["path"] = df["path"].apply(lambda x: Path(f"/path/to/patient_{x}/study_{x}/file_{x}.dcm"))
    df["malign_score"] = 0.5
    path = Path(tmp_path, "df.csv")
    df.to_csv(path, index=False)
    result = triage_csv(path)
    assert result.index.name == "Patient"
    assert "malign_score" in result.columns
    assert len(result) == len(df)


@pytest.mark.parametrize("how", ["inner", "outer", "left"])
def test_join(df_factory, how):
    df1 = df_factory().loc[:, ("col1",)]
    df2 = df_factory().loc[:, ("col2",)]
    df3 = df_factory().loc[:, ("col3",)]
    df = join([df1, df2, df3], how=how)
    assert df.shape == (df1.shape[0], df1.shape[1] + df2.shape[1] + df3.shape[1])


def test_concat(df_factory):
    df1 = df_factory().iloc[:2]
    df2 = df_factory().iloc[2:4]
    df3 = df_factory().iloc[4:]
    df = concat([df1, df2, df3])
    assert (df == df_factory()).all().all()


def test_join_or_concat(df_factory):
    df1 = df_factory(columns=["col1", "col2"]).iloc[:4]
    df2 = df_factory(columns=["col1", "col2"]).iloc[4:]
    df3 = df_factory(columns=["col3"]).iloc[:4]
    df4 = df_factory(columns=["col3"]).iloc[4:]
    expected = pd.concat([df1, df2]).join(pd.concat([df3, df4]))
    df = join_or_concat([df1, df2, df3, df4])
    assert (df == expected).all().all()
