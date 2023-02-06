#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from registry import Registry


TRANSFORM_REGISTRY = Registry("transforms", bound=Callable[..., pd.DataFrame])


class Transform(ABC):
    @abstractmethod
    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class NoopTransform(Transform):
    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        return table


TRANSFORM_REGISTRY(NoopTransform(), name="noop")


@dataclass
class Discretize(Transform):
    column: str
    intervals: Sequence[float]
    output_column: Optional[str] = None

    def __post_init__(self):
        if self.output_column is None:
            self.output_column = self.column

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.column not in table.columns:
            raise ValueError(f"column {self.column} not in table.columns {table.columns}")
        result = self.discretize(table)
        return result

    def discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        # get bin assignment for each item
        col = pd.to_numeric(df[self.column], errors="coerce")
        valid = ~col.isna()
        names = self._bins_to_index(self.intervals)
        name_dict = {i: v for i, v in enumerate(names)}
        groups = cast(List[Any], np.digitize(col[valid], self.intervals).tolist())

        # map bin assignments to clean string names
        discretized = [name_dict.get(g, "NA") for g in groups]
        df.loc[valid, self.output_column] = discretized
        return df

    @classmethod
    def _bins_to_index(cls, bins: Sequence[float]) -> Sequence[str]:
        result: List[str] = []
        for i, b in enumerate(bins):
            if i == 0:
                result.append(f"< {b}")
            else:
                result.append(f"{bins[i-1]} <= x < {b}")
        result.append(f">= {bins[-1]}")
        return result


@dataclass
class KeepWhere(Transform):
    column: str
    value: Any
    as_string: bool = True
    allow_empty: bool = False

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.column not in table.columns:
            raise ValueError(f"column {self.column} not in table.columns {table.columns}")
        result = cast(pd.DataFrame, table[self.get_mask(table)])
        if not len(result) and not self.allow_empty:
            raise ValueError(f"Filter {self} produced an empty result")
        return result

    def get_mask(self, table: pd.DataFrame) -> Any:
        value = str(self.value) if self.as_string else self.value
        column = table[self.column].astype(str) if self.as_string else table[self.column]
        return column == value

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        columns: Iterable[str] = [],
        as_string: bool = True,
        discretizers: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> Dict[Tuple[str, Any], "KeepWhere"]:
        columns = list(columns or df.columns)
        result: Dict[Tuple[str, Any], KeepWhere] = {}

        if discretizers is not None:
            df = df.copy()
            for dname in discretizers:
                func = TRANSFORM_REGISTRY.get(dname).instantiate_with_metadata()
                df = func(df)

        for colname in columns:
            for value in df[colname].unique():
                value = str(value) if as_string else value
                key = (colname, value)
                func = cls(column=colname, value=value, as_string=as_string, **kwargs)
                result[key] = func
        return result

    @classmethod
    def format_name(cls, colname: str, value: Any) -> str:
        def prepare_str(x: Any) -> str:
            return str(x).strip().lower().replace(" ", "_")

        return f"keep-{prepare_str(colname)}-{prepare_str(value)}"


@dataclass
class DropWhere(KeepWhere):
    def get_mask(self, table: pd.DataFrame) -> Any:
        return ~super().get_mask(table)


@dataclass
class KeepColumns(Transform):
    columns: Sequence[str]

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        columns = [c for c in self.columns if c in table.columns]
        return table[columns]
