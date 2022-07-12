#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple, cast

import pandas as pd
from registry import Registry


TRANSFORM_REGISTRY = Registry("transforms")


class Transform(ABC):
    @abstractmethod
    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class NoopTransform(Transform):
    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        return table


TRANSFORM_REGISTRY(NoopTransform(), name="noop")


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
        string_values: bool = False,
        as_string: bool = True,
        **kwargs,
    ) -> Dict[Tuple[str, Any], "KeepWhere"]:
        columns = list(columns or df.columns)
        result: Dict[Tuple[str, Any], KeepWhere] = {}

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
