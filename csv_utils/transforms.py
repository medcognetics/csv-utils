#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union, cast

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

    Args:
        column: The column on which to perform the value renaming.
        mapping: A dictionary containing the old value(s) as keys and the new value(s) as values.
        as_string: If True, compare values as strings.
        output_column: The name of the column to store the result. If None, the original column is updated in place.
    """

    column: str
    mapping: Dict[Any, Any]
    default: Any | None = None
    as_string: bool = False
    output_column: str | None = None

    def __post_init__(self):
        if not self.mapping:
            raise ValueError("mapping cannot be empty")

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        # Validate inputs
        if self.column not in table.columns:
            raise KeyError(f"column {self.column} not in table.columns {table.columns}")

        mapping = self.mapping
        if self.as_string:
            mapping = {str(k): v for k, v in mapping.items()}

        def _rename_value(value: Any) -> Any:
            default = self.default if self.default is not None else value
            return mapping.get(str(value) if self.as_string else value, default)

        output_column = self.output_column if self.output_column is not None else self.column
        table.loc[:, output_column] = table.loc[:, self.column].apply(_rename_value)
        return table


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
