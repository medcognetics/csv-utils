from dataclasses import dataclass
from functools import partial
from typing import List, Sequence

import pandas as pd

from .transforms import Transform


@dataclass
class PivotTableCounts(Transform):
    r"""Transform to summarize counts of values in a table using a pivot table.

    Args:
        index: The column(s) to use as the row index in the summary.
        columns: The column(s) to use as the column index in the summary.
            If ``None``, all columns not in ``index`` will be used.
        total: The values in ``index`` to aggregate in order to produce a total row.
            If ``None``, no total row will be produced.
    """
    index: str | Sequence[str]
    columns: str | Sequence[str] | None = None
    total: str | Sequence[str] | None = None

    def __call__(self, table: pd.DataFrame) -> pd.DataFrame:
        index: List[str] = [self.index] if isinstance(self.index, str) else list(self.index)
        columns: List[str] = (
            [self.columns]
            if isinstance(self.columns, str)
            else list(self.columns)
            if self.columns is not None
            else [c for c in table.columns if c not in index]
        )

        # Create pivot table
        column_values = {col: sorted(table[col].unique()) for col in columns}
        agg_funcs = {
            # NOTE: partial lambda needed to capture the value of `v` at the time of the loop
            col: [partial(lambda x, y: (x == y).sum(), y=v) for v in values]
            for col, values in column_values.items()
        }
        agg_func_names = [(col, value) for col, values in column_values.items() for value in values]
        result = pd.pivot_table(table, index=index, values=columns, aggfunc=agg_funcs, fill_value=0)
        result.columns = pd.MultiIndex.from_tuples(agg_func_names)

        # Fill in missing index levels
        if isinstance(result.index, pd.MultiIndex):
            new_index = pd.MultiIndex.from_product(result.index.levels)
            result = result.reindex(new_index, fill_value=0)

        # Add total
        if self.total is not None:
            total_index = [self.total] if isinstance(self.total, str) else list(self.total)
            total_index = list(set(index) - set(total_index))
            other_index = list(set(index) - set(total_index))
            if total_index:
                total = (
                    result.groupby(total_index)
                    .sum()
                    .assign(**{k: "Total" for k in other_index})
                    .set_index(other_index, append=True)
                    .reorder_levels(result.index.names)
                )
            else:
                total = result.sum(axis=0).to_frame(name="Total").T
            result = pd.concat([result, total])

        return result
