import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.data import (
    PointwiseDataset, RatingDataset, build_id_maps, build_item_users, build_recent_items,
    build_user_items, load_ratings, sample_negative, split_ratings,
)
from common.recommend import recommend_by_embedding, top_n
from common.train import fit, run_epoch
from conftest import CPU


def test_读取评分数据(tiny_csv):
    ratings = load_ratings(tiny_csv)
    assert list(ratings.columns) == ["userId", "movieId", "rating", "timestamp"]
    assert len(ratings) == 9


def test_id映射从0开始连续编号(tiny_ratings):
    user_map, item_map = build_id_maps(tiny_ratings)
    assert sorted(user_map) == [1, 2, 3, 4]
    assert sorted(user_map.values()) == [0, 1, 2, 3]
    assert sorted(item_map) == [10, 20, 30, 40]
    assert sorted(item_map.values()) == [0, 1, 2, 3]


def test_训练集和测试集按比例划分(small_ratings):
    train, test = split_ratings(small_ratings, test_size=0.25)
    assert len(train) + len(test) == len(small_ratings)
    assert len(test) == pytest.approx(len(small_ratings) * 0.25, abs=1)
    assert not set(train.index) & set(test.index)


def test_用户物品表和倒排表(tiny_ratings):
    user_items = build_user_items(tiny_ratings)
    item_users = build_item_users(tiny_ratings)
    assert user_items[1] == {10: 5, 20: 3}
    assert item_users[30] == {2: 5, 3: 4}


def test_最近交互按时间排序并截断(tiny_ratings):
    shuffled = tiny_ratings.sample(frac=1, random_state=1)
    recent = build_recent_items(shuffled, lastn=2)
    assert recent[2] == [20, 30]
    assert recent[1] == [10, 20]


def test_负采样不会采到交互过的物品():
    for _ in range(100):
        assert sample_negative([1, 2, 3, 4], {1, 2, 3}) == 4


def test_显式评分样本(tiny_ratings):
    user_map, item_map = build_id_maps(tiny_ratings)
    dataset = RatingDataset(tiny_ratings, user_map, item_map)
    sample = dataset[0]
    assert len(dataset) == 9
    assert sample["user_idx"].item() == user_map[1]
    assert sample["item_idx"].item() == item_map[10]
    assert sample["rating"].item() == 5.0


def test_pointwise样本每个正样本跟若干负样本(small_ratings):
    user_map, item_map = build_id_maps(small_ratings)
    user_items = build_user_items(small_ratings)
    dataset = PointwiseDataset(small_ratings, user_map, item_map, user_items, negative_sampling=3)
    inverse_item_map = {index: item_id for item_id, index in item_map.items()}
    assert len(dataset) == len(small_ratings) * 4

    positive = dataset[0]
    assert positive["label"].item() == 1.0
    assert inverse_item_map[positive["item_idx"].item()] == small_ratings.iloc[0]["movieId"]

    user_id = small_ratings.iloc[0]["userId"]
    for idx in range(1, 4):
        negative = dataset[idx]
        assert negative["label"].item() == 0.0
        assert inverse_item_map[negative["item_idx"].item()] not in user_items[user_id]


def test_top_n排除物品并按分数排序():
    scores = {1: 0.5, 2: 0.9, 3: 0.7, 4: 0.1}
    assert top_n(scores, 2) == [(2, 0.9), (3, 0.7)]
    assert top_n(scores, 2, exclude={2}) == [(3, 0.7), (1, 0.5)]


def test_向量召回是内积排序():
    user_vector = torch.tensor([1.0, 0.0])
    item_vectors = torch.tensor([[0.2, 1.0], [0.9, 0.0], [0.5, 0.5]])
    result = recommend_by_embedding(user_vector, item_vectors, ["a", "b", "c"], n=2, exclude={"b"})
    assert [item for item, _ in result] == ["c", "a"]
    assert result[0][1] == pytest.approx(0.5)


def linear_loss(model, batch):
    return F.mse_loss(model(batch["x"]).squeeze(1), batch["y"])


def make_loader():
    x = torch.randn(64, 3)
    y = x @ torch.tensor([1.0, -2.0, 0.5])
    return DataLoader([{"x": x[i], "y": y[i]} for i in range(64)], batch_size=16)


def test_评估时不更新参数():
    model = nn.Linear(3, 1)
    before = model.weight.clone()
    run_epoch(model, make_loader(), linear_loss, CPU)
    assert torch.equal(model.weight, before)


def test_训练循环能降低损失():
    model = nn.Linear(3, 1)
    loader = make_loader()
    history = fit(model, loader, loader, linear_loss, CPU, n_epochs=30, learning_rate=0.05, log_every=0)
    assert len(history["train_loss"]) == 30
    assert history["train_loss"][-1] < history["train_loss"][0] * 0.1
