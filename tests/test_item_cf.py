import math

import pytest

from 召回.item_cf import ItemCF


@pytest.fixture
def model(tiny_csv):
    model = ItemCF(dataset_path=tiny_csv, top_k=10, lastn=20)
    assert model.load_data(), "load_data 应该能读取 csv 并返回 True"
    return model.train()


def test_读取不存在的文件返回False(tmp_path):
    model = ItemCF(dataset_path=str(tmp_path / "missing.csv"))
    assert model.load_data() is False


def test_相似度公式(model):
    # sim(i, j) = |U_i ∩ U_j| / sqrt(|U_i| * |U_j|)
    assert model.item_sim[10][20] == pytest.approx(2 / math.sqrt(3 * 3))
    assert model.item_sim[10][30] == pytest.approx(2 / math.sqrt(3 * 2))
    assert model.item_sim[20][40] == pytest.approx(1 / math.sqrt(3 * 1))


def test_相似度对称(model):
    for item_i, sims in model.item_sim.items():
        for item_j, value in sims.items():
            assert model.item_sim[item_j][item_i] == pytest.approx(value)


def test_没有共同用户的物品不算相似度(model):
    assert 40 not in model.item_sim[10], "A 和 D 没有共同用户，不应该有相似度"


def test_相似物品按相似度从高到低排序(model):
    # A 的相似物品: C(0.816) > B(0.667)
    assert model.get_similar_items(10) == [30, 20]


def test_top_k限制相似物品数量(tiny_csv):
    model = ItemCF(dataset_path=tiny_csv, top_k=1, lastn=20)
    model.load_data()
    model.train()
    for item_id in [10, 20, 30, 40]:
        assert len(model.get_similar_items(item_id)) <= 1


def test_lastn只保留最近交互(tiny_csv):
    model = ItemCF(dataset_path=tiny_csv, top_k=10, lastn=2)
    model.load_data()
    model.train()
    # 用户2按时间顺序交互了 A, B, C
    assert model.get_user_interacted_items(2) == [20, 30]


def test_推荐分数公式(model):
    # score(j) = Σ rating(u, i) * sim(i, j)
    recommendations = dict(model.recommend_for_user(1, n_rec=10))
    expected_c = 5 * (2 / math.sqrt(6)) + 3 * (1 / math.sqrt(6))
    expected_d = 3 * (1 / math.sqrt(3))
    assert recommendations[30] == pytest.approx(expected_c)
    assert recommendations[40] == pytest.approx(expected_d)


def test_推荐结果排除已交互物品且有序(model):
    recommendations = model.recommend_for_user(1, n_rec=10)
    item_ids = [item_id for item_id, _ in recommendations]
    assert 10 not in item_ids and 20 not in item_ids
    scores = [score for _, score in recommendations]
    assert scores == sorted(scores, reverse=True)
    assert item_ids[0] == 30


def test_n_rec限制推荐数量(model):
    assert len(model.recommend_for_user(1, n_rec=1)) == 1
