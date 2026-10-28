import pytest

from 召回.swing import Swing


@pytest.fixture
def model(tiny_csv):
    model = Swing(dataset_path=tiny_csv, top_k=10, lastn=30, alpha=1)
    assert model.load_data()
    return model.train()


def test_overlap是集合交集(model):
    assert model.get_overlap([1, 2, 3], [2, 3, 4]) == {2, 3}


def test_物品的用户列表(model):
    assert sorted(model.item_users[10]) == [1, 2, 3]
    assert sorted(model.item_users[40]) == [4]


def test_swing相似度公式(model):
    # sim(i, j) = Σ_{u1 < u2 ∈ U_i ∩ U_j} 1 / (alpha + |I_u1 ∩ I_u2|)
    # A 和 B 的共同用户是 1、2，两人共同交互了 {A, B}
    assert model.item_sim[10][20] == pytest.approx(1 / (1 + 2))
    # A 和 C 的共同用户是 2、3，两人共同交互了 {A, C}
    assert model.item_sim[10][30] == pytest.approx(1 / (1 + 2))


def test_alpha越大相似度越小(tiny_csv):
    model = Swing(dataset_path=tiny_csv, top_k=10, lastn=30, alpha=10)
    model.load_data()
    model.train()
    assert model.item_sim[10][20] == pytest.approx(1 / (10 + 2))


def test_只有一个共同用户时没有相似度(model):
    # B 和 C 只有用户2一个共同用户，凑不出用户对
    assert model.item_sim[20].get(30, 0.0) == 0.0


def test_相似度对称(model):
    for item_i, sims in model.item_sim.items():
        for item_j, value in sims.items():
            assert model.item_sim[item_j][item_i] == pytest.approx(value)


def test_推荐分数公式(model):
    # 用户1: A(5分), B(3分)。只有 A-C 有 swing 相似度 1/3
    recommendations = model.recommend_for_user(1, n_rec=10)
    assert len(recommendations) == 1
    item_id, score = recommendations[0]
    assert item_id == 30
    assert score == pytest.approx(5 / 3)


def test_lastn只保留最近交互(tiny_csv):
    model = Swing(dataset_path=tiny_csv, top_k=10, lastn=2, alpha=1)
    model.load_data()
    model.train()
    assert model.get_user_interacted_items(2) == [20, 30]
