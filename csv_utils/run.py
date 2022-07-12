#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import PathLike
from pathlib import Path
from typing import Callable, Iterable, List, Union, cast

import pandas as pd

from .input import AGGREGATOR_REGISTRY, INPUT_REGISTRY
from .transforms import TRANSFORM_REGISTRY


def transform_csv(
    sources: Iterable[Union[pd.DataFrame, PathLike]],
    input_names: Iterable[str],
    aggregator_name: str,
    transform_names: Iterable[str] = [],
) -> pd.DataFrame:
    r"""Applies a transformation pipeline to one or more input CSVs

    The transformation pipeline follows these steps:
        * A list of DataFrames is built by invoking ``input_fn[i](sources[i])``, where
          ``input_fn[i]`` is the input handler associated with ``input_names[i]``.

        * The aggregator specified by ``aggregator_name`` is used to aggregate all input sources
          into a single DataFrame

        * The sequence of transforms specified by ``transform_names`` is run on the aggregate DataFrame.

    Args:
        sources:
            Iterable of :class:`pd.DataFrame` or source filepaths

        input_names:
            Iterable of registered names for input handlers registered in `INPUT_REGISTRY`.
            The condition ``len(input_names) == len(sources)`` must be satisfied.

        aggregator_name:
            Registered name of an aggregator registered in ``AGGREGATOR_REGISTRY``.

        transform_names:
            Iterable of registered names for transforms registered in ``TRANSFORM_REGISTRY``.

    Returns:
        Transformed dataframe
    """
    # prepare input targets
    sources = [s if isinstance(s, pd.DataFrame) else Path(s) for s in sources]
    inputs = [INPUT_REGISTRY.get(i) for i in input_names]
    if len(inputs) != len(sources):
        if len(inputs) != 1:
            raise ValueError("Only one input should be given, or the number of inputs should match the number of paths")
        else:
            inputs = inputs * len(sources)

    # load all inputs
    assert len(inputs) == len(sources)
    loaded_inputs: List[pd.DataFrame] = []
    for inp, s in zip(inputs, sources):
        fn = cast(Callable[..., pd.DataFrame], inp)
        df = fn(s)
        loaded_inputs.append(df)

    # if multiple inputs were loaded, run an aggregator to create a single dataframe
    if len(loaded_inputs) == 1:
        df = loaded_inputs[0]
    elif len(loaded_inputs) > 1:
        agg = AGGREGATOR_REGISTRY.get(aggregator_name)
        fn = cast(Callable[..., pd.DataFrame], agg)
        df = fn(loaded_inputs)
    else:
        raise RuntimeError("An aggregator must be provided when multiple inputs are used")

    # run transforms
    transforms = [TRANSFORM_REGISTRY.get_with_metadata(name) for name in transform_names]
    for t in transforms:
        fn = cast(Callable[..., pd.DataFrame], t.fn)
        df = fn(df, **t.metadata)

    return df
