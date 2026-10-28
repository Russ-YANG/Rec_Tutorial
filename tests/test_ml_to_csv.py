import pandas as pd

from 召回.ml_to_csv import convert_movielens_to_csv


def test_转换后的列名和数据(tmp_path):
    source = tmp_path / "u.data"
    source.write_text("1\t10\t5\t100\n2\t20\t3\t200\n")
    target = tmp_path / "ratings.csv"
    convert_movielens_to_csv(str(source), str(target))
    df = pd.read_csv(target)
    assert list(df.columns) == ["userId", "movieId", "rating", "timestamp"]
    assert df.values.tolist() == [[1, 10, 5, 100], [2, 20, 3, 200]]
