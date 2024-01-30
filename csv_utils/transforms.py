#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np
import pandas as pd
from registry import Registry


_ALPHABETICAL_RE = re.compile(r"^[a-zA-Z]")
_DIGIT_RE = re.compile(r"(\d+(\.\d+)?)")
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
        col = cast(pd.Series, pd.to_numeric(df[self.column], errors="coerce"))
        valid = ~col.isna()
        names = self._bins_to_index(self.intervals)
        name_dict = {i: v for i, v in enumerate(names)}
        groups = cast(List[Any], np.digitize(col[valid], self.intervals).tolist())

        # map bin assignments to clean string names
        discretized = [name_dict.get(g, "NA") for g in groups]
        df.loc[cast(Any, valid), self.output_column] = discretized
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
        columns = list(columns if columns else df.columns)
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
    """
    Renames a specific value in a given column of a DataFrame.
    It takes a DataFrame as input and returns a DataFrame with the specified value renamed.

    Args:
        column: The column in which the value to be renamed is located.
        old_value: The value to be renamed.
        new_value: The new name for the value.
        as_string: If True, compare values as strings.
    """

    column: str
    old_value: Any
    new_value: Any
    as_string: bool = False

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.column not in table.columns:
            raise KeyError(f"column {self.column} not in table.columns {table.columns}")
        if self.as_string:
            mask = table[self.column].astype(str) == str(self.old_value)
        else:
            mask = table[self.column] == self.old_value
        table.loc[mask, self.column] = self.new_value
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


T = TypeVar("T", str, pd.DataFrame)


@TRANSFORM_REGISTRY(name="capitalize")
@TRANSFORM_REGISTRY(name="capitalize-index", index=True)
def capitalize(x: T, index: bool = False) -> T:
    """
    This function takes a string and capitalizes the first letter of each word except for 'and' and 'or'.
    If a word is already in uppercase, it is left unchanged. Can also be applied to DataFrames, in which
    case it is applied to each column, or to the index, if `index` is True. DataFrame trasformation is
    applied in-place.

    Args:
        x: The input to be capitalized.
        index: If True, the index of a DataFrame is processed instead of the columns.

    Returns:
        str: The processed string or DataFrame with capitalized words.
    """
    if isinstance(x, str):
        return " ".join(
            word.capitalize() if not word.isupper() and word not in ["and", "or"] else word for word in x.split(" ")
        )
    elif isinstance(x, pd.DataFrame):
        x.columns
        if index:
            x.index = x.index.map(lambda i: capitalize(str(i)))
        else:
            x.columns = x.columns.map(lambda c: capitalize(str(c)))
        return x
    else:
        raise TypeError(f"capitalize() expected str or DataFrame, got {type(x)}")


@TRANSFORM_REGISTRY(name="sanitize-latex")
@TRANSFORM_REGISTRY(name="sanitize-latex-index", index=True)
def sanitize_latex(x: T, index: bool = False) -> T:
    r"""Sanitize a string or DataFrame for use in LaTeX.

    The following adjustments are made:
        * Replaces any comparison operators with their LaTeX equivalent. For example, "<=" becomes "$\\leq$".
        * Escapes any "%" characters
        * Replaces any underscore-separated words with their LaTeX subscript equivalent. For example,
            "hello_world" becomes "hello\\textsubscript{world}". If a word contains multiple underscores,
            a warning is issued and the word is left unchanged.

    Args:
        x: The input to sanitize.
        index: If True, the index of a DataFrame is processed instead of the columns.

    Returns:
        The sanitized string or DataFrame.
    """
    if isinstance(x, str):
        # Sanitize operators
        OPERATORS: Final = {"<=": "$\\leq$", ">=": "$\\geq$", "<": "$<$", ">": "$>$", "=": "$=$", "%": "$\\%$"}
        for op, latex in OPERATORS.items():
            x = x.replace(op, latex)

        # Sanitize underscores to subscripts
        result = []
        for word in x.split(" "):
            if "_" in word:
                try:
                    main, sub = word.split("_")
                    word = f"{main}\\textsubscript{{{sub}}}"
                except ValueError:
                    # We don't support multiple subscripts in one word
                    import warnings

                    warnings.warn(f"Multiple subscripts in one word not supported: {word}. Skipping.")
            result.append(word)
        return cast(T, " ".join(result))

    elif isinstance(x, pd.DataFrame):
        x.columns
        if index:
            x.index = x.index.map(lambda i: sanitize_latex(str(i)))
        else:
            x.columns = x.columns.map(lambda c: sanitize_latex(str(c)))
        return x
    else:
        raise TypeError(f"sanitize_latex() expected str or DataFrame, got {type(x)}")


@dataclass
class Summarize:
    r"""
    Summarize a table across each column.

    Args:
        col: The column to summarize.
        groupby: If provided, the summary will include an overall summary, plus summaries by
            values in the columns provided here.

    Raises:
        * KeyError: If ``col`` is not in ``df``.
        * KeyError: If ``groupby`` contains a column that is not in ``df``.
    """
    column: str
    groupby: List[str] = field(default_factory=list)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.column not in df.columns:
            raise KeyError(f"column {self.column} not in df.columns {df.columns}")
        if any(col not in df.columns for col in self.groupby):
            raise KeyError(f"One or more groupby columns not found in dataframe columns")

        total = len(df)

        def _create_cols(df: pd.DataFrame, col: str) -> pd.DataFrame:
            result = df.value_counts(self.column)
            result = result.sort_index(ascending=True).to_frame("Count")
            result["%"] = result / total * 100
            result.loc["Total"] = [result["Count"].sum(), 100.0]
            return result

        subtables = {"Overall": _create_cols(df, self.column)}
        for group in self.groupby:
            for category in df[group].unique():
                keep = df[group] == category
                subtables[category] = _create_cols(df[keep], self.column)
        summary = pd.concat(subtables, axis=1)
        summary.fillna(0, inplace=True)

        # Flatten multi-col
        summary.columns = [
            " ".join(str(c) for c in col).strip() if isinstance(col, tuple) else str(col)
            for col in summary.columns.values
        ]

        return summary


def sort(values: Sequence[Any], ascending: bool = True, numeric_first: bool = True) -> List[Any]:
    r"""
    Sorts a sequence of values.

    This function is robust to intervals with comparison operators.
    It uses a regular expression to extract the first numeric value (if any) from each key and
    sorts the keys based on these values. Non-numeric keys are sorted lexicographically.

    .. note::
        It is assumed that intervals always have the smaller value first.
        For example, "1 <= x < 2" or "1-2" is valid, but "2 < x <= 1" or "2-1" is not.

    Args:
        values: The values to sort.
        ascending: If True, sort in ascending order. Otherwise, sort in descending order.
        numeric_first: If True, sort numeric values before strings. Otherwise, sort strings before numeric values.

    Returns:
        The sorted values
    """

    def sort_key(val: str) -> float | str:
        # We don't want things like source_monthyy to be sorted as floats by year.
        # Instead we will consider them strs, though this doesn't give a good year sorting
        if _ALPHABETICAL_RE.search(val):
            return val
        match = _DIGIT_RE.search(val)
        result = float(match.group()) if match else val

        # We need to handle one-sided intervals, e.g. < 5.0 and 5.0 <= x < 10.0 in the same value set
        if match:
            if val.startswith("<"):
                result = sys.float_info.min
            elif val.startswith(">"):
                result = sys.float_info.max
        return result

    # The sort keys may be a mixture of float and str, so we need to compare them separately
    start_len = len(values)
    sort_keys = {sort_key(k): k for k in values}
    float_values = [key for key in sort_keys.keys() if isinstance(key, float)]
    str_values = [key for key in sort_keys.keys() if isinstance(key, str)]
    sorted_keys = (
        sorted(float_values) + sorted(str_values) if numeric_first else sorted(str_values) + sorted(float_values)
    )

    sorted_values = [sort_keys[key] for key in sorted_keys]
    assert len(sorted_values) == start_len, f"Expected {start_len} values, got {len(sorted_values)}"
    return sorted_values if ascending else sorted_values[::-1]


@TRANSFORM_REGISTRY(name="sort-cols")
@TRANSFORM_REGISTRY(name="sort-cols-descending", ascending=False)
def sort_columns(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    r"""
    Sorts the columns of a DataFrame.

    Args:
        df: The DataFrame to sort.
        ascending: If True, sort in ascending order. Otherwise, sort in descending order.

    Returns:
        The sorted DataFrame.
    """
    return df[sort(list(df.columns), ascending=ascending)]
