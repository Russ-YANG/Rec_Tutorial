import math

import pytest

from 召回.user_cf import UserCF, build_similar_users, score_items, user_similarity


def test_用户相似度公式():
    # sim(u, v) = |I_u ∩ I_v| / sqrt(|I_u| * |I_v|)
    assert user_similarity([10, 20], [10, 20, 30]) == pytest.approx(2 / math.sqrt(2 * 3))
    assert user_similarity([10, 20], [10, 30]) == pytest.approx(1 / 2)


def test_没有交集相似度为0():
    assert user_similarity([10, 30], [20, 40]) == 0


def test_用户索引按相似度从高到低():
    user_items = {1: [10, 20], 2: [10, 20, 30], 3: [10, 30], 4: [20, 40]}
    similar_users = build_similar_users(user_items, top_k=10)
    assert similar_users[1][0][0] == 2
    scores = [sim for _, sim in similar_users[1]]
    assert scores == sorted(scores, reverse=True)


def test_用户索引不含相似度为0的用户():
    user_items = {1: [10, 20], 2: [10, 20, 30], 3: [10, 30], 4: [20, 40]}
    similar_users = build_similar_users(user_items, top_k=10)
    assert 4 not in dict(similar_users[3])


def test_top_k限制相似用户数量():
    user_items = {1: [10, 20], 2: [10, 20, 30], 3: [10, 30], 4: [20, 40]}
    similar_users = build_similar_users(user_items, top_k=2)
    assert all(len(users) <= 2 for users in similar_users.values())


def test_打分是相似度之和():
    # score(j) = Σ_{v 交互过 j} sim(u, v)
    user_items = {2: [10, 30], 3: [30, 40]}
    scores = score_items([(2, 0.8), (3, 0.5)], user_items)
    assert scores[30] == pytest.approx(0.8 + 0.5)
    assert scores[40] == pytest.approx(0.5)


@pytest.fixture
def model(tiny_ratings):
    return UserCF(top_k=10, lastn=15).fit(tiny_ratings)


def test_相似用户(model):
    assert model.get_similar_users(1)[0][0] == 2


def test_推荐结果(model):
    # 用户1的相似用户: 2(0.816), 3(0.5), 4(0.5)
    recommendations = dict(model.recommend(1, n_rec=10))
    assert list(recommendations) == [30, 40]
    assert recommendations[30] == pytest.approx(2 / math.sqrt(6) + 0.5)
    assert recommendations[40] == pytest.approx(0.5)


def test_lastn只用最近交互算相似度(tiny_ratings):
    model = UserCF(top_k=10, lastn=1).fit(tiny_ratings)
    assert model.recent_items[2] == [30]
