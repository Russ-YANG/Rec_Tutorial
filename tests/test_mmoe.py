import pytest
import torch

from conftest import CPU
from 排序.mmoe import MMoE, MMoERecommender, MovieLensDataset


def test_点击标签是评分大于等于4(small_csv):
    dataset = MovieLensDataset(small_csv, train=True)
    for idx in range(len(dataset)):
        rating = dataset.data.iloc[idx]["rating"]
        assert dataset[idx]["click_label"].item() == (1.0 if rating >= 4 else 0.0)


def test_评分用训练集均值方差归一化(small_csv):
    train = MovieLensDataset(small_csv, train=True)
    mean, std = train.get_rating_stats()
    assert mean == pytest.approx(train.data["rating"].mean())
    rating = train.data.iloc[0]["rating"]
    assert train[0]["rating_label"].item() == pytest.approx((rating - mean) / std, rel=1e-5)


def test_测试集沿用训练集的统计量(small_csv):
    train = MovieLensDataset(small_csv, train=True)
    mean, std = train.get_rating_stats()
    test = MovieLensDataset(small_csv, train=False, rating_mean=mean, rating_std=std)
    assert test.get_rating_stats() == (mean, std)


def make_model():
    model = MMoE(num_users=5, num_items=6, embedding_dim=8, expert_hidden_dims=[16, 8],
                 gate_hidden_dims=[8], task_hidden_dims=[8], num_experts=4, num_tasks=2)
    model.eval()
    return model


def test_专家和门控数量():
    model = make_model()
    assert len(model.experts) == 4
    assert len(model.gate_networks) == 2
    assert len(model.task_towers) == 2


def test_门控输出是专家上的概率分布():
    model = make_model()
    features = torch.randn(3, 16)
    for gate in model.gate_networks:
        weights = gate(features)
        assert weights.shape == (3, 4)
        assert torch.all(weights >= 0)
        assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)


def test_两个任务的输出():
    model = make_model()
    click, rating = model(torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    assert click.shape == (3,) and rating.shape == (3,)
    assert torch.all((click > 0) & (click < 1)), "点击率要过 sigmoid"


def test_推荐结果格式(small_csv):
    recommender = MMoERecommender(path=small_csv, embedding_dim=8, expert_hidden_dims=[16, 8],
                                  gate_hidden_dims=[8], task_hidden_dims=[8], device=CPU)
    recommender.load_data()
    recommendations = recommender.recommend_for_user(1, top_n=5)
    assert len(recommendations) == 5
    assert {"item_id", "click_prob", "rating", "score"} <= set(recommendations[0])
    scores = [row["score"] for row in recommendations]
    assert scores == sorted(scores, reverse=True)
