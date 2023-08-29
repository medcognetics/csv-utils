#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

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
    r"""Keep rows where `column == value`. If `column` is a sequence, then
    `value` must be a sequence of the same length, and the mask is the
    logical and of the masks for each column/value pair.

    Args:
        column: column or columns to filter on
        value: value or values to filter on
        as_string: if True, convert value to string before comparison
        allow_empty: if True, allow empty result
        contains: if True, use `in` instead of `==`. String comparisons only.
    """
    column: Union[str, Sequence[str]]
    value: Any
    as_string: bool = True
    allow_empty: bool = False
    contains: bool = False

    def __post_init__(self):
        if isinstance(self.column, str):
            self.column = [self.column]
        else:
            if not isinstance(self.value, Sequence):
                raise TypeError(f"value {self.value} must be a sequence if column is a sequence")
            elif not len(self.column) == len(self.value):
                raise ValueError(f"columns {self.column} and values {self.value} must be same length")

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(self.column, Sequence) and not isinstance(self.column, str)
        for c in self.column:
            if c not in table.columns:
                raise KeyError(f"column {c} not in table.columns {table.columns}")
        result = cast(pd.DataFrame, table[self.get_mask(table)])
        if not len(result) and not self.allow_empty:
            raise ValueError(f"Filter {self} produced an empty result")
        return result

    def get_mask(self, table: pd.DataFrame) -> Any:
        if len(self.column) == 1:
            return self._get_mask_for_column(table, self.column[0], self.value)
        else:
            return np.logical_and.reduce(
                [self._get_mask_for_column(table, c, v) for c, v in zip(self.column, self.value)]
            )

    def _get_mask_for_column(self, table: pd.DataFrame, col: str, value: Any) -> Any:
        if value is None or value is pd.NA:
            return table[col].isna()
        value = str(value) if self.as_string else value
        column = table[col].astype(str) if self.as_string else table[col]
        return column == value if not self.contains else column.apply(lambda x: value in str(x))

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        columns: Iterable[str] = [],
        as_string: bool = True,
        discretizers: Optional[Iterable[str]] = None,
        allow_missing_column: bool = False,
        **kwargs,
    ) -> Dict[Tuple[str, Any], "KeepWhere"]:
        r"""Create a KeepWhere transform for each unique value in the given columns.

        Args:
            df: The dataframe to use for creating the transforms.
            columns: The columns to use for creating the transforms. If empty, all columns
                will be used.
            as_string: If True, the values will be converted to strings before comparison.
            discretizers: If given, the columns will be discretized using the given
                discretizers.
            allow_missing_column: If True, columns that are not in the dataframe will be
                ignored. Otherwise, a :class:`KeyError` will be raised.
            **kwargs: Additional keyword arguments to pass to the KeepWhere constructor.

        Returns:
            A dictionary mapping (column, value) pairs to KeepWhere transforms.
        """
        columns = list(columns or df.columns)
        result: Dict[Tuple[str, Any], KeepWhere] = {}

        if discretizers is not None:
            df = df.copy()
            for dname in discretizers:
                func = TRANSFORM_REGISTRY.get(dname).instantiate_with_metadata()
                df = func(df)

        for colname in columns:
            if colname not in df.columns:
                if allow_missing_column:
                    continue
                raise KeyError(f"column {colname} not in df.columns {df.columns}")
            for value in df[colname].unique():
                value = str(value) if as_string else value
                key = (colname, value)
                func = cls(column=colname, value=value, as_string=as_string, **kwargs)
                result[key] = func
        return result

    @classmethod
    def format_name(cls, colname: str, value: Any) -> str:
        r"""Format a column name and value into a human-readable string for use in a registry"""

        def prepare_str(x: Any) -> str:
            return str(x).strip().lower().replace(" ", "_")

        return f"keep-{prepare_str(colname)}-{prepare_str(value)}"


@dataclass
class DropWhere(KeepWhere):
    r"""Drop rows where `column == value`. If `column` is a sequence, then
    `value` must be a sequence of the same length, and the mask is the
    logical and of the masks for each column/value pair.

    Args:
        column: column or columns to filter on
        value: value or values to filter on
        as_string: if True, convert value to string before comparison
        allow_empty: if True, allow empty result
        contains: if True, use `in` instead of `==`. String comparisons only.
        allow_missing_column: if True, allow missing column
    """
    allow_missing_column: bool = False

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        for c in self.column:
            if c not in table.columns and self.allow_missing_column:
                return table
        return super().__call__(table)

    def get_mask(self, table: pd.DataFrame) -> Any:
        return ~super().get_mask(table)


@dataclass
class KeepColumns(Transform):
    columns: Sequence[str]

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        columns = [c for c in self.columns if c in table.columns]
        return table[columns]


@dataclass
class GroupValues(Transform):
    r"""Group values in a particular column.

    Args:
        colname: The column to group.
        sources: The values to group together.
        dest: The value to replace the sources with.
    """
    colname: str
    sources: Union[str, Sequence[str]]
    dest: str

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.colname not in table.columns:
            raise KeyError(f"column {self.colname} not in table.columns {table.columns}")
        sources = [self.sources] if isinstance(self.sources, str) else self.sources
        mask = table[self.colname].isin(sources)
        table.loc[mask, self.colname] = self.dest
        return table


@dataclass
class RenameColumn(Transform):
    old_name: str
    new_name: str

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.old_name not in table.columns:
            raise KeyError(f"column {self.old_name} not in table.columns {table.columns}")
        table = table.rename(columns={self.old_name: self.new_name})
        return table


@dataclass
class RenameValue(Transform):
    column: str
    old_value: Any
    new_value: Any

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.column not in table.columns:
            raise KeyError(f"column {self.column} not in table.columns {table.columns}")
        table.loc[table[self.column] == self.old_value, self.column] = self.new_value
        return table


@dataclass
class RenameTable(Transform):
    new_value: str

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        table.name = self.new_value
        return table


@dataclass
class RenameColumns(Transform):
    new_name: str

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        table.columns.name = self.new_name
        return table


@dataclass
class RenameIndex(Transform):
    new_value: str

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        table.index.name = self.new_value
        return table
