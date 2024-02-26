#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np
import pandas as pd
from registry import Registry


_ALPHABETICAL_RE: Final = re.compile(r"^[a-zA-Z]")
_DIGIT_RE: Final = re.compile(r"(\d+(\.\d+)?)")
_DIGIT_ALPHABETICAL_RE: Final = re.compile(r"(?P<digit>\d+(\.\d+)?)(?P<alphabetical>[a-zA-Z]+)")

TRANSFORM_REGISTRY = Registry("transforms", bound=Callable[..., pd.DataFrame])


class Transform(ABC):
    @abstractmethod
    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class NoopTransform(Transform):
    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        return table


TRANSFORM_REGISTRY(NoopTransform(), name="noop")


I = TypeVar("I")


def to_list(x: I | Iterable[I]) -> List[I]:
    match x:
        case str():
            return [cast(I, x)]
        case _ if isinstance(x, Iterable):
            return list(x)
        case _:
            return [cast(I, x)]


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
    value: Any | Sequence[Any]
    as_string: bool = True
    allow_empty: bool = False
    contains: bool = False

    def __post_init__(self):
        if isinstance(self.column, str):
            self.column = [self.column]
            self.value = [self.value]
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
            return self._get_mask_for_column(table, self.column[0], self.value[0])
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
    """
    Rename one or more columns in a DataFrame.

    This transform allows for renaming one or multiple columns in a given DataFrame. If `copy` is set to True,
    the original columns are retained alongside the new ones with updated names. Otherwise, the original columns
    are replaced by the new names.

    Attributes:
        old_name: A string or a list of strings representing the original column names. If None, all columns are considered.
        new_name: A string or a list of strings representing the new column names.
        copy: A boolean indicating whether to copy the original column(s) or replace them. Defaults to False.

    Raises:
        KeyError: If any of the `old_name` columns do not exist in the DataFrame.
        ValueError: If the length of `old_name` and `new_name` are not the same.
    """

    old_name: str | Sequence[str] | None
    new_name: str | Sequence[str]
    copy: bool = False

    def __post_init__(self):
        self.old_name = to_list(self.old_name) if self.old_name is not None else None
        self.new_name = to_list(self.new_name)

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        # Validate inputs
        old_name = self.old_name if self.old_name is not None else list(table.columns)
        new_name = self.new_name
        assert isinstance(old_name, list) and isinstance(new_name, list)
        if len(old_name) != len(new_name):
            raise ValueError(f"old_name and new_name must have the same length")
        for val in old_name:
            if val not in table.columns:
                raise KeyError(f"column {self.old_name} not in table.columns {table.columns}")

        # Rename columns
        if self.copy:
            for old, new in zip(old_name, new_name):
                table[new] = table[old]
        else:
            table = table.rename(columns={old: new for old, new in zip(old_name, new_name)})
        return table


@dataclass
class RenameValue(Transform):
    """
    Renames a value or values in a given column of a DataFrame.

    .. note::
        Passing ``mapping`` as a list of tuples is supported to enable jsonargparse instantiation when
        keys contain spaces. However, the mapping is converted to a dictionary internally.

    Args:
        column: The column on which to perform the value renaming.
        mapping: Defines the mapping of old to new values. Can be a dictionary mapping or a list of mapping tuples.
        default: The default value to use if the old value is not found in the mapping. If None, the original value is used.
        as_string: If True, compare values as strings.
        output_column: The name of the column to store the result. If None, the original column is updated in place.
        mask_column: The column or columns to use as a row mask for the transformation. If None, all rows are transformed.
            This is passed to :class:`KeepWhere` to filter the rows to transform.
        mask_value: The value or values to use as a row mask for the transformation. If None, all rows are transformed.
            This is passed to :class:`KeepWhere` to filter the rows to transform.
    """

    column: str
    mapping: Dict[Any, Any] | List[Tuple[Any, Any]]
    default: Any | None = None
    as_string: bool = False
    output_column: str | None = None
    mask_column: str | Sequence[str] | None = None
    mask_value: Any | Sequence[Any] | None = None

    def __post_init__(self):
        if not self.mapping:
            raise ValueError("mapping cannot be empty")
        self.mapping = dict(self.mapping) if isinstance(self.mapping, list) else self.mapping
        self.mask_column = to_list(self.mask_column) if self.mask_column is not None else None
        self.mask_value = to_list(self.mask_value) if self.mask_value is not None else None
        if self.mask_column and not self.mask_value:
            raise ValueError("`mask_column` cannot be set without `mask_value`")

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        # Validate inputs
        if self.column not in table.columns:
            raise KeyError(f"column {self.column} not in table.columns {table.columns}")

        mapping = self.mapping
        assert isinstance(mapping, dict)
        if self.as_string:
            mapping = {str(k): v for k, v in mapping.items()}

        def _rename_value(value: Any) -> Any:
            default = self.default if self.default is not None else value
            return mapping.get(str(value) if self.as_string else value, default)

        output_column = self.output_column if self.output_column is not None else self.column
        table.loc[self._get_mask(table), output_column] = table.loc[:, self.column].apply(_rename_value)
        return table

    def _get_mask(self, table: pd.DataFrame) -> Any:
        if self.mask_column:
            assert self.mask_value
            func = KeepWhere(self.mask_column, self.mask_value, allow_empty=True)
            table = func(table)
        return table.index


@dataclass
class RenameTable(Transform):
    new_value: str

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        table.name = self.new_value
        return table


# TODO: We should probably have a single RenameAxis transform that can rename both columns and index.
# Having RenameColumn and RenameColumns is a bit confusing.
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
        warnings.warn(
            "This transform is deprecated and will be removed in a future release. "
            "See `csv_utils.summary.Summarize instead.",
            DeprecationWarning,
        )
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


def sort(
    values: Sequence[Any],
    ascending: bool = True,
    numeric_first: bool = True,
    parse_dates: bool = True,
    **kwargs,
) -> List[Any]:
    r"""
    Sorts a sequence of unique values.

    This function is robust to intervals with comparison operators.
    It uses a regular expression to extract the first numeric value (if any) from each key and
    sorts the keys based on these values. Non-numeric keys are sorted lexicographically. If all
    inputs are dates they are parsed and sorted as dates.

    .. note::
        It is assumed that intervals always have the smaller value first.
        For example, "1 <= x < 2" or "1-2" is valid, but "2 < x <= 1" or "2-1" is not.

    Args:
        values: The values to sort. Values should be unique.
        ascending: If True, sort in ascending order. Otherwise, sort in descending order.
        numeric_first: If True, sort numeric values before strings. Otherwise, sort strings before numeric values.
        parse_dates: If True, attempt to interpret the values as dates and sort them accordingly.

    Keyword Args:
        Forwarded to :func:`pandas.to_datetime` for date parsing.

    Returns:
        The sorted values
    """

    def assign_sort_key(val: str) -> float | str:
        # If requested, first try to parse the value as a date.
        # When parsed we convert to a timestamp so the return value is still a float
        if parse_dates and (parsed_date := pd.to_datetime(val, errors="coerce", **kwargs)) is not pd.NaT:
            return cast(pd.Timestamp, parsed_date).timestamp()

        # We don't want things like source_monthyy to be sorted as floats by year.
        # Instead we will consider them strs, though this doesn't give a good year sorting
        if _ALPHABETICAL_RE.search(val):
            return val

        # Try to handle numeric values suffixed with alphabetical characters, e.g. 5a, 5b, 5c
        # To do this we will split the string into a digit and an alphabetical part. Both parts will be converted
        # to floats and combined into a single float. This will allow us to sort the values in the correct order.
        # This isn't perfect, but it lets us avoid a lot of custom sub-sorting logic.
        # If a more robust solution is later needed, consider recursive sub-sort calls.
        if re_match := _DIGIT_ALPHABETICAL_RE.search(val):
            digit = re_match.group("digit")
            ord_sum = sum(ord(letter) for letter in re_match.group("alphabetical"))
            BIG_DIVISOR: Final = 2**32
            return float(digit) + ord_sum / BIG_DIVISOR

        re_match = _DIGIT_RE.search(val)
        result = float(re_match.group()) if re_match else val

        # We need to handle one-sided intervals, e.g. < 5.0 and 5.0 <= x < 10.0 in the same value set
        if re_match:
            if val.startswith("<"):
                result = sys.float_info.min
            elif val.startswith(">"):
                result = sys.float_info.max
        return result

    if not len(values) == len(set(values)):
        raise ValueError("Values for sorting must be unique")

    # The sort keys may be a mixture of float and str, so we need to compare them separately
    sort_keys = {assign_sort_key(k): k for k in values}
    float_values = [key for key in sort_keys.keys() if isinstance(key, float)]
    str_values = [key for key in sort_keys.keys() if isinstance(key, str)]
    sorted_keys = (
        sorted(float_values) + sorted(str_values) if numeric_first else sorted(str_values) + sorted(float_values)
    )

    sorted_values = [sort_keys[key] for key in sorted_keys]
    assert len(sorted_values) == len(values), f"Expected {len(values)} values, got {len(sorted_values)}"
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


@dataclass
class Cast(Transform):
    """
    Casts or coerces a column to a new dtype.

    Args:
        column: Column or columns to cast. If ``None``, all columns will be cast.
        dtype: The new dtype to cast to. Will be passed to :func:`pandas.DataFrame.astype`.
        errors: How to handle errors in the cast. Will be passed to :func:`pandas.DataFrame.astype`.

    Returns:
        The DataFrame with the column or columns casted to the new dtype.
    """

    column: str | Sequence[str] | None
    dtype: Any = "str"
    errors: str = "raise"

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        columns = (
            [self.column]
            if isinstance(self.column, str)
            else list(self.column)
            if self.column is not None
            else list(table.columns)
        )

        for column in columns:
            if column not in table.columns:
                raise KeyError(f"column {column} not in table.columns {table.columns}")
            table[column] = table[column].astype(self.dtype, errors=cast(Any, self.errors))

        return table
