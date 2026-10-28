import os
import random
import sys

import numpy as np
import pandas as pd
import pytest
import torch

# 让测试可以直接 import 召回.item_cf 这样的模块
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 测试只用 CPU，结果更稳定
CPU = torch.device("cpu")

# 手工构造的小数据集，方便手算期望值
# 物品: 10=A, 20=B, 30=C, 40=D
# 用户1: A(5分, t=1), B(3分, t=2)
# 用户2: A, B, C
# 用户3: A, C
# 用户4: B, D
TINY_RATINGS = [
    (1, 10, 5, 1),
    (1, 20, 3, 2),
    (2, 10, 4, 3),
    (2, 20, 4, 4),
    (2, 30, 5, 5),
    (3, 10, 3, 6),
    (3, 30, 4, 7),
    (4, 20, 2, 8),
    (4, 40, 5, 9),
]


@pytest.fixture(autouse=True)
def fixed_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def tiny_csv(tmp_path):
    path = tmp_path / "tiny.csv"
    df = pd.DataFrame(TINY_RATINGS, columns=["userId", "movieId", "rating", "timestamp"])
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def small_csv(tmp_path):
    """20 个用户，30 个物品，每个用户随机评 8 个物品。给需要训练集/测试集划分的模型用。"""
    rng = np.random.RandomState(0)
    rows = []
    timestamp = 0
    for user_id in range(1, 21):
        items = rng.choice(np.arange(1, 31), size=8, replace=False)
        for item_id in items:
            timestamp += 1
            rows.append((user_id, int(item_id) * 10, int(rng.randint(1, 6)), timestamp))
    path = tmp_path / "small.csv"
    pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def tiny_ratings(tiny_csv):
    return pd.read_csv(tiny_csv)


@pytest.fixture
def small_ratings(small_csv):
    return pd.read_csv(small_csv)
