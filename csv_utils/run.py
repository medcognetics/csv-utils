#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import PathLike
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Union, cast

import pandas as pd

from .input import AGGREGATOR_REGISTRY, INPUT_REGISTRY
from .transforms import TRANSFORM_REGISTRY


def transform_csv(
    sources: Iterable[Union[pd.DataFrame, PathLike]],
    types: Iterable[Union[str, Callable]],
    aggregator: str | Callable | Sequence[str | Callable] = "noop",
    transforms: Iterable[Union[str, Callable]] = [],
    aggregation_groups: Sequence[int] = [],
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

        aggregator:
            Registered name of an aggregator registered in ``AGGREGATOR_REGISTRY``.

        transforms:
            Iterable of transforms or registered names for transforms registered in ``TRANSFORM_REGISTRY``.

        aggregation_groups:
            List of integers indicating how to group the input sources for aggregation. If empty, all sources are
            aggregated together. If not empty, the length of this list must be equal to the number of sources.
            When provided the aggregator must be a list of aggregators.

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

    # prepare the aggregator
    if isinstance(aggregator, (str, Callable)):
        agg = AGGREGATOR_REGISTRY.get(aggregator).instantiate_with_metadata().bind_metadata()
    elif isinstance(aggregator, Sequence):
        agg = [AGGREGATOR_REGISTRY.get(item).instantiate_with_metadata().bind_metadata() for item in aggregator]
    else:
        raise TypeError("Aggregator must be a string, callable, or sequence of such")

    # prepare the aggregation groups
    if aggregation_groups and not isinstance(agg, list):
        raise ValueError(
            "Aggregation groups can only be used with a list of aggregators. "
            "Pass `aggregator`=[...] to use multiple aggregators with aggregation groups."
        )
    elif not aggregation_groups:
        aggregation_groups = [0] * len(sources)
    elif len(aggregation_groups) != len(sources):
        raise ValueError("Length of aggregation groups must match the number of sources")

    # if multiple inputs were loaded, run an aggregator to create a single dataframe
    if len(loaded_inputs) == 1:
        df = loaded_inputs[0]
    elif len(loaded_inputs) > 1:
        if callable(agg):
            df = agg(loaded_inputs)
        else:
            df_groups = []
            for group in set(aggregation_groups):
                group_indices = [i for i, g in enumerate(aggregation_groups) if g == group]
                group_inputs = [loaded_inputs[i] for i in group_indices]
                df_groups.append(agg[group](group_inputs))
            df = pd.concat(df_groups, join="outer")
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
