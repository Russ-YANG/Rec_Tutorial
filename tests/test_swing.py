import pytest

from 召回.item_cf import ItemCF
from 召回.swing import Swing, swing_similarity

USER_ITEMS = {1: {"A", "B"}, 2: {"A", "B", "C"}, 3: {"A", "C"}, 4: {"B", "D"}}


def test_swing相似度公式():
    # sim(i, j) = Σ_{u1 < u2 ∈ U_i ∩ U_j} 1 / (alpha + |I_u1 ∩ I_u2|)
    # A 和 B 的共同用户是 1、2，两人共同交互了 {A, B}
    assert swing_similarity({1, 2, 3}, {1, 2, 4}, USER_ITEMS, alpha=1) == pytest.approx(1 / (1 + 2))


def test_多个用户对的贡献相加():
    user_items = {1: {"A", "B"}, 2: {"A", "B"}, 3: {"A", "B", "C", "D"}}
    # 用户对 (1,2) 重合 2 个，(1,3) 重合 2 个，(2,3) 重合 2 个
    assert swing_similarity({1, 2, 3}, {1, 2, 3}, user_items, alpha=1) == pytest.approx(3 / 3)


def test_用户越像贡献越小():
    loose = {1: {"A", "B"}, 2: {"A", "B"}}
    tight = {1: {"A", "B", "C", "D"}, 2: {"A", "B", "C", "D"}}
    assert swing_similarity({1, 2}, {1, 2}, tight, 1) < swing_similarity({1, 2}, {1, 2}, loose, 1)


def test_alpha越大相似度越小():
    assert swing_similarity({1, 2}, {1, 2}, USER_ITEMS, alpha=10) == pytest.approx(1 / (10 + 2))


def test_只有一个共同用户时没有相似度():
    # B 和 C 只有用户2一个共同用户，凑不出用户对
    assert swing_similarity({1, 2, 4}, {2, 3}, USER_ITEMS, alpha=1) == 0


def test_Swing是ItemCF的子类():
    assert issubclass(Swing, ItemCF)


@pytest.fixture
def model(tiny_ratings):
    return Swing(top_k=10, lastn=30, alpha=1).fit(tiny_ratings)


def test_相似物品(model):
    assert sorted(model.get_similar_items(10)) == [20, 30]
    assert model.get_similar_items(40) == []


def test_推荐结果(model):
    # 用户1: A(5分), B(3分)。只有 A-C 有 swing 相似度 1/3
    recommendations = model.recommend(1, n_rec=10)
    assert len(recommendations) == 1
    assert recommendations[0][0] == 30
    assert recommendations[0][1] == pytest.approx(5 / 3)
