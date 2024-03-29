#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import pytest

from csv_utils import INPUT_REGISTRY, TRANSFORM_REGISTRY, transform_csv
from csv_utils.transforms import Summarize


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


@pytest.mark.parametrize("transform", ["noop", lambda x: x])
@pytest.mark.parametrize(
    "types",
    [
        ("stats-csv", "scores-csv"),
        (INPUT_REGISTRY.get("stats-csv").fn, INPUT_REGISTRY.get("scores-csv").fn),
    ],
)
def test_basic_run(setup_basic_test, transform, types):
    stats = setup_basic_test["stats"]
    scores = setup_basic_test["scores"]
    paths = [stats, scores]
    types = types
    aggregator_name = "join"
    transform_names = [transform]

    output = transform_csv(paths, types, aggregator_name, transform_names)
    assert list(output.columns) == [
        "Data Source Case ID",
        "Ground Truth",
        "Study",
        "score",
    ]
    assert output.index.name == "Study Path"
    assert len(output) == 10


def test_raw_df_passthrough(df_factory):
    stats_df = df_factory(["Data Source Case ID", "Study Path", "Ground Truth"], as_str=True)
    scores_df = df_factory(["Study", "cases", "score"], as_str=True)
    stats_df.set_index("Data Source Case ID", inplace=True)
    scores_df.set_index("Study", inplace=True)

    output = transform_csv([stats_df, scores_df], ["df", "df"], aggregator="join")
    assert list(output.columns) == [
        "Study Path",
        "Ground Truth",
        "cases",
        "score",
    ]
    assert output.index.name == "Data Source Case ID"
    assert len(output) == 10


def test_summarize_latex(df_factory):
    stats_df = df_factory(["Data Source Case ID", "Study Path", "Ground Truth"], as_str=True)
    scores_df = df_factory(["Study", "cases", "score"], as_str=True)
    stats_df.set_index("Data Source Case ID", inplace=True)
    scores_df.set_index("Study", inplace=True)

    TRANSFORM_REGISTRY(Summarize, name="summary", column="Ground Truth")  # type: ignore

    output = transform_csv(
        [stats_df, scores_df],
        ["df", "df"],
        aggregator="join",
        transforms=["summary", "capitalize", "sanitize-latex", "sanitize-latex-index"],
    )
    assert "Overall $\\%$" in output.columns
    assert output.index.name == "Ground Truth"
    assert len(output) == 11


def test_aggregation_groups(df_factory):
    stats_df1 = df_factory(["Data Source Case ID", "Study Path", "Ground Truth"], as_str=True)
    stats_df2 = df_factory(["Data Source Case ID", "Ground Truth"], as_str=True, offset=len(stats_df1))

    scores_df1 = df_factory(["Study", "cases", "score"], as_str=True)
    scores_df2 = df_factory(["Study", "cases", "score"], as_str=True, offset=len(scores_df1))

    stats_df1.set_index("Data Source Case ID", inplace=True)
    stats_df2.set_index("Data Source Case ID", inplace=True)
    scores_df1.set_index("Study", inplace=True)
    scores_df2.set_index("Study", inplace=True)

    output = transform_csv(
        [stats_df1, stats_df2, scores_df1, scores_df2],
        ["df", "df", "df", "df"],
        aggregator=["join", "join"],
        aggregation_groups=[0, 1, 0, 1],
    )

    exp1 = transform_csv(
        [stats_df1, scores_df1],
        ["df", "df"],
        aggregator="join",
    )
    exp2 = transform_csv(
        [stats_df2, scores_df2],
        ["df", "df"],
        aggregator="join",
    )
    exp = pd.concat([exp1, exp2], join="outer")
    pd.testing.assert_frame_equal(output, exp)
