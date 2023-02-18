#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import pytest

from csv_utils.input import (
    concat,
    data_organizer_csv,
    first_index_csv,
    first_index_excel,
    join,
    join_or_concat,
    merge,
    pdb_agg,
    scores_csv,
    stats_csv,
    triage_csv,
)


def test_csv(tmp_path, df_factory):
    df = df_factory(columns=["Data Source Case ID", "Study Path"])
    path = Path(tmp_path, "df.csv")
    df.to_csv(path, index=False)
    result = first_index_csv(path)
    assert result.index.name == "Data Source Case ID"
    assert len(result) == len(df)


def test_excel(tmp_path, df_factory):
    df = df_factory(columns=["Data Source Case ID", "Study Path"])
    path = Path(tmp_path, "df.xlsx")
    df.to_excel(path, index=False)
    result = first_index_excel(path)
    assert result.index.name == "Data Source Case ID"
    assert len(result) == len(df)


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


def test_join(df_factory):
    df1 = df_factory().loc[:, ("col1",)]
    df2 = df_factory().loc[:, ("col2",)]
    df3 = df_factory().loc[:, ("col3",)]
    df = join([df1, df2, df3])
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


def test_merge(df_factory):
    df1 = df_factory(columns=["col1", "col2"]).iloc[:6]
    df2 = df_factory(columns=["col2", "col3"]).iloc[4:]
    expected = df1.join(df2, rsuffix="_r", how="outer")
    expected.update(df2, overwrite=False)
    expected.drop(expected.filter(regex="_r$").columns.tolist(), axis=1, inplace=True)
    df = merge([df1, df2])
    pd.testing.assert_frame_equal(df, expected)
    assert set(df.columns) == {"col1", "col2", "col3"}


def test_pdb(mocker, df_factory):
    m = mocker.MagicMock()
    mocker.patch("pdb.set_trace", m)
    df1 = df_factory().loc[:, ("col1",)]
    df2 = df_factory().loc[:, ("col2",)]
    with pytest.raises(AssertionError):
        pdb_agg([df1, df2])
    m.assert_called_once()
