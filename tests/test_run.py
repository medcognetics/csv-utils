#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest

from csv_utils import transform_csv


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


def test_basic_run(setup_basic_test):
    stats = setup_basic_test["stats"]
    scores = setup_basic_test["scores"]
    paths = [stats, scores]
    input_names = ["stats-csv", "scores-csv"]
    aggregator_name = "join"
    transform_names = ["noop"]

    output = transform_csv(paths, input_names, aggregator_name, transform_names)
    assert list(output.columns) == [
        "Data Source Case ID",
        "Study Path",
        "Ground Truth",
        "Study",
        "score",
    ]
    assert output.index.name == "case"
    assert len(output) == 10
