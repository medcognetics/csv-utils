from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

from .transforms import Transform


@dataclass
class Summarize(Transform):
    r"""Transform to summarize counts of values a table.

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

        # Summarize each column individually
        column_summaries = {
            colname: table.groupby(index)[colname].value_counts().unstack().fillna(0).astype(int) for colname in columns
        }

        # Add column axis name as a top level in the column multi-index
        for colname, summary in column_summaries.items():
            summary.columns = pd.MultiIndex.from_product([[colname], summary.columns])

        # Build table
        result = pd.concat(column_summaries.values(), axis=1)

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
