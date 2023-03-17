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
    types: Iterable[Union[str, Callable]],
    aggregator_name: str = "noop",
    transforms: Iterable[Union[str, Callable]] = [],
) -> pd.DataFrame:
    r"""Applies a transformation pipeline to one or more input CSVs

    The transformation pipeline follows these steps:
        * A list of DataFrames is built by invoking ``input_fn[i](sources[i])``, where
          ``input_fn[i]`` is the input handler associated with ``types[i]``.

        * The aggregator specified by ``aggregator_name`` is used to aggregate all input sources
          into a single DataFrame

        * The sequence of transforms specified by ``transforms`` is run on the aggregate DataFrame.

    Args:
        sources:
            Iterable of :class:`pd.DataFrame` or source filepaths

        types:
            Iterable of input type handlers or registered names for input type handlers registered in `INPUT_REGISTRY`.
            The condition ``len(input_names) == len(sources)`` must be satisfied.

        aggregator_name:
            Registered name of an aggregator registered in ``AGGREGATOR_REGISTRY``.

        transforms:
            Iterable of transforms or registered names for transforms registered in ``TRANSFORM_REGISTRY``.

    Returns:
        Transformed dataframe
    """
    # prepare input targets
    sources = [s if isinstance(s, pd.DataFrame) else Path(s) for s in sources]
    inputs = [INPUT_REGISTRY.get(i).instantiate_with_metadata().bind_metadata() for i in types]
    if len(inputs) != len(sources):
        if len(inputs) != 1:
            raise ValueError("Only one input should be given, or the number of inputs should match the number of paths")
        else:
            inputs = inputs * len(sources)

    # load all inputs
    assert len(inputs) == len(sources)
    loaded_inputs: List[pd.DataFrame] = []
    for inp, s in zip(inputs, sources):
        df = inp(s)
        loaded_inputs.append(df)

    # if multiple inputs were loaded, run an aggregator to create a single dataframe
    if len(loaded_inputs) == 1:
        df = loaded_inputs[0]
    elif len(loaded_inputs) > 1:
        agg = AGGREGATOR_REGISTRY.get(aggregator_name).instantiate_with_metadata().bind_metadata()
        df = agg(loaded_inputs)
    else:
        raise RuntimeError("An aggregator must be provided when multiple inputs are used")

    # run transforms
    transforms = cast(
        List[Callable],
        [TRANSFORM_REGISTRY.get(name).instantiate_with_metadata().bind_metadata() for name in transforms],
    )
    for t in transforms:
        df = t(df)

    return df
