import math

import pytest

from 召回.user_cf import UserCF


@pytest.fixture
def model(tiny_csv):
    model = UserCF(dataset_path=tiny_csv, top_k=10, lastn=15)
    assert model.load_data()
    model.train()
    return model


def test_用户物品索引按时间排序(model):
    assert model.user_item_index[2] == [10, 20, 30]


def test_lastn只保留最近交互(tiny_csv):
    model = UserCF(dataset_path=tiny_csv, top_k=10, lastn=1)
    model.load_data()
    model.train()
    assert model.user_item_index[2] == [30]


def test_用户相似度公式(model):
    # sim(u1, u2) = |I_u1 ∩ I_u2| / sqrt(|I_u1| * |I_u2|)
    assert model.calculate_user_similarity(1, 2) == pytest.approx(2 / math.sqrt(2 * 3))
    assert model.calculate_user_similarity(1, 3) == pytest.approx(1 / math.sqrt(2 * 2))


def test_没有交集相似度为0(model):
    assert model.calculate_user_similarity(3, 4) == 0


def test_相似用户按相似度排序(model):
    similar_users = model.get_similar_users(1)
    assert similar_users[0][0] == 2
    scores = [score for _, score in similar_users]
    assert scores == sorted(scores, reverse=True)


def test_top_k限制相似用户数量(tiny_csv):
    model = UserCF(dataset_path=tiny_csv, top_k=2, lastn=15)
    model.load_data()
    model.train()
    for user_id in [1, 2, 3, 4]:
        assert len(model.user_user_index[user_id]) <= 2


def test_推荐分数是相似度之和(model):
    # 用户1的相似用户: 2(0.816), 3(0.5), 4(0.5)
    # C 被用户2和3交互过，D 被用户4交互过
    recommendations = dict(model.recommend_for_user(1, n_rec=10))
    assert recommendations[30] == pytest.approx(2 / math.sqrt(6) + 0.5)
    assert recommendations[40] == pytest.approx(0.5)


def test_推荐结果排除已交互物品(model):
    item_ids = [item_id for item_id, _ in model.recommend_for_user(1, n_rec=10)]
    assert 10 not in item_ids and 20 not in item_ids
    assert item_ids[0] == 30


def test_物品的用户数(model):
    assert model.get_item_users() == {10: 3, 20: 3, 30: 2, 40: 1}
