from pathlib import Path

from csv_utils.output import to_latex


def test_to_latex(tmp_path, df_factory):
    df = df_factory(["Data Source Case ID", "Study Path", "Ground Truth"], as_str=True)
    df.set_index("Data Source Case ID", inplace=True)
    path = tmp_path / Path("test_output.tex")
    to_latex(df, path)
    assert path.is_file()
