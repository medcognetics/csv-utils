#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest

from csv_utils import INPUT_REGISTRY, transform_csv


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

    output = transform_csv([stats_df, scores_df], ["df", "df"], aggregator_name="join")
    assert list(output.columns) == [
        "Study Path",
        "Ground Truth",
        "cases",
        "score",
    ]
    assert output.index.name == "Data Source Case ID"
    assert len(output) == 10
