#!/usr/bin/env python
# -*- coding: utf-8 -*-
import builtins
import runpy
import sys
from pathlib import Path

import pytest

from csv_utils.output import OUTPUT_REGISTRY


@pytest.mark.parametrize("output", OUTPUT_REGISTRY.available_keys())
def test_main(mocker, tmp_path, setup_basic_test, output):
    stats_path = setup_basic_test["stats"]
    scores_path = setup_basic_test["scores"]
    dest_path = Path(tmp_path, "dest.csv")

    # mocks for outputs that don't write a file
    mocks = [
        mocker.patch("pdb.set_trace"),
        mocker.spy(builtins, "print"),
    ]

    sys.argv = [
        sys.argv[0],
        str(stats_path),
        str(scores_path),
        "-i",
        "stats-csv",
        "scores-csv",
        "-a",
        "join",
        "-t",
        "noop",
        "-o",
        output,
        "-d",
        str(dest_path),
    ]
    runpy.run_module("csv_utils", run_name="__main__", alter_sys=True)
    assert dest_path.is_file() or any(m.called for m in mocks)
