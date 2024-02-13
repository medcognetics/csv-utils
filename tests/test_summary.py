import pandas as pd
import pytest

from csv_utils.summary import PivotTableCounts


@pytest.mark.parametrize(
    "index,columns,total,exp",
    [
        (
            "col1",
            None,
            None,
            pd.DataFrame.from_dict(
                {
                    ("col2", False): {False: 0, True: 5},
                    ("col2", True): {False: 5, True: 0},
                    ("col3", False): {False: 5, True: 0},
                    ("col3", True): {False: 0, True: 5},
                }
            ),
        ),
        (
            ["col1", "col2"],
            None,
            None,
            pd.DataFrame.from_dict(
                {
                    ("col3", False): {(False, False): 0, (False, True): 5, (True, False): 0, (True, True): 0},
                    ("col3", True): {(False, False): 0, (False, True): 0, (True, False): 5, (True, True): 0},
                }
            ),
        ),
        (
            "col1",
            "col3",
            None,
            pd.DataFrame.from_dict({("col3", False): {False: 5, True: 0}, ("col3", True): {False: 0, True: 5}}),
        ),
        (
            "col1",
            None,
            "col1",
            pd.DataFrame.from_dict(
                {
                    ("col2", False): {False: 0, True: 5, "Total": 5},
                    ("col2", True): {False: 5, True: 0, "Total": 5},
                    ("col3", False): {False: 5, True: 0, "Total": 5},
                    ("col3", True): {False: 0, True: 5, "Total": 5},
                }
            ),
        ),
        (
            ["col1", "col2"],
            None,
            "col1",
            pd.DataFrame.from_dict(
                {
                    ("col3", False): {
                        (False, False): 0,
                        (False, True): 5,
                        (True, False): 0,
                        (True, True): 0,
                        ("Total", False): 0,
                        ("Total", True): 5,
                    },
                    ("col3", True): {
                        (False, False): 0,
                        (False, True): 0,
                        (True, False): 5,
                        (True, True): 0,
                        ("Total", False): 5,
                        ("Total", True): 0,
                    },
                }
            ),
        ),
    ],
)
def test_summarize(df_factory, index, columns, total, exp):
    df = df_factory().apply(lambda x: x % 2 == 0)
    result = PivotTableCounts(index=index, columns=columns, total=total)(df)
    pd.testing.assert_frame_equal(result, exp, check_names=False)
