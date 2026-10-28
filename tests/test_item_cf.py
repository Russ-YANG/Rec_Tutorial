import math

import pytest

from 召回.item_cf import ItemCF, build_similar_items, item_similarity, score_items


def test_相似度公式():
    # sim(i, j) = |U_i ∩ U_j| / sqrt(|U_i| * |U_j|)
    assert item_similarity({1, 2, 3}, {1, 2, 4}) == pytest.approx(2 / math.sqrt(3 * 3))
    assert item_similarity({1, 2, 3}, {2, 3}) == pytest.approx(2 / math.sqrt(3 * 2))


def test_没有共同用户相似度为0():
    assert item_similarity({1, 2}, {3}) == 0


def test_相似度可以直接传入评分字典():
    # 倒排表的值是 {用户: 评分}，只看用户
    assert item_similarity({1: 5, 2: 3}, {2: 4}) == pytest.approx(1 / math.sqrt(2))


def test_物品索引按相似度从高到低():
    item_users = {"A": {1, 2, 3}, "B": {1, 2, 4}, "C": {2, 3}, "D": {4}}
    similar_items = build_similar_items(item_users, item_similarity, top_k=10)
    assert [item for item, _ in similar_items["A"]] == ["C", "B"]
    assert similar_items["A"][0][1] == pytest.approx(2 / math.sqrt(6))


def test_物品索引对称且不含相似度为0的物品():
    item_users = {"A": {1, 2, 3}, "B": {1, 2, 4}, "C": {2, 3}, "D": {4}}
    similar_items = build_similar_items(item_users, item_similarity, top_k=10)
    assert "D" not in dict(similar_items["A"])
    for item_i, sims in similar_items.items():
        for item_j, sim in sims:
            assert dict(similar_items[item_j])[item_i] == pytest.approx(sim)


def test_物品索引用传入的相似度函数():
    item_users = {"A": {1}, "B": {2}}
    similar_items = build_similar_items(item_users, lambda users_i, users_j: 0.5, top_k=10)
    assert similar_items["A"] == [("B", 0.5)]


def test_top_k限制相似物品数量():
    item_users = {"A": {1, 2, 3}, "B": {1, 2, 4}, "C": {2, 3}, "D": {4}}
    similar_items = build_similar_items(item_users, item_similarity, top_k=1)
    assert all(len(sims) <= 1 for sims in similar_items.values())


def test_打分公式():
    # score(j) = Σ rating(u, i) * sim(i, j)
    similar_items = {"A": [("C", 0.8), ("B", 0.6)], "B": [("C", 0.4), ("D", 0.5)]}
    scores = score_items(["A", "B"], {"A": 5, "B": 3}, similar_items)
    assert scores["C"] == pytest.approx(5 * 0.8 + 3 * 0.4)
    assert scores["D"] == pytest.approx(3 * 0.5)


def test_打分只用最近交互的物品():
    similar_items = {"A": [("C", 0.8)], "B": [("D", 0.5)]}
    scores = score_items(["B"], {"A": 5, "B": 3}, similar_items)
    assert "C" not in scores


@pytest.fixture
def model(tiny_ratings):
    return ItemCF(top_k=10, lastn=20).fit(tiny_ratings)


def test_相似物品(model):
    # A 的相似物品: C(0.816) > B(0.667)
    assert model.get_similar_items(10) == [30, 20]


def test_推荐结果(model):
    recommendations = model.recommend(1, n_rec=10)
    expected_c = 5 * (2 / math.sqrt(6)) + 3 * (1 / math.sqrt(6))
    expected_d = 3 * (1 / math.sqrt(3))
    assert [item for item, _ in recommendations] == [30, 40]
    assert recommendations[0][1] == pytest.approx(expected_c)
    assert recommendations[1][1] == pytest.approx(expected_d)


def test_推荐排除所有交互过的物品(tiny_ratings):
    # lastn=1 时用户2只用 C 打分，但 A、B 也不能被推荐
    model = ItemCF(top_k=10, lastn=1).fit(tiny_ratings)
    item_ids = [item for item, _ in model.recommend(2, n_rec=10)]
    assert not {10, 20, 30} & set(item_ids)


def test_n_rec限制推荐数量(model):
    assert len(model.recommend(1, n_rec=1)) == 1
