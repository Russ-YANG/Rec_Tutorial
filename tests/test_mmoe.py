import pytest
import torch
import torch.nn.functional as F

from common.data import build_id_maps
from conftest import CPU
from 排序.mmoe import MMoE, MMoERecommender, MultiTaskDataset, mix_experts, multitask_loss


def test_点击标签是评分大于等于4(small_ratings):
    user_map, item_map = build_id_maps(small_ratings)
    dataset = MultiTaskDataset(small_ratings, user_map, item_map, rating_mean=3.0, rating_std=1.0)
    for idx in range(len(dataset)):
        rating = small_ratings.iloc[idx]["rating"]
        assert dataset[idx]["click_label"].item() == (1.0 if rating >= 4 else 0.0)


def test_评分标签做了归一化(small_ratings):
    user_map, item_map = build_id_maps(small_ratings)
    dataset = MultiTaskDataset(small_ratings, user_map, item_map, rating_mean=3.0, rating_std=2.0)
    rating = small_ratings.iloc[0]["rating"]
    assert dataset[0]["rating_label"].item() == pytest.approx((rating - 3.0) / 2.0)


def test_专家混合是加权和():
    expert_outputs = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]])
    gate_weights = torch.tensor([[0.5, 0.25, 0.25]])
    assert torch.allclose(mix_experts(expert_outputs, gate_weights), torch.tensor([[1.0, 0.75]]))


def test_门控只选一个专家时就是该专家的输出():
    expert_outputs = torch.randn(2, 3, 4)
    gate_weights = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mixed = mix_experts(expert_outputs, gate_weights)
    assert torch.allclose(mixed[0], expert_outputs[0, 1])
    assert torch.allclose(mixed[1], expert_outputs[1, 2])


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


def test_两个任务的门控参数独立():
    model = make_model()
    assert model.gate_networks[0] is not model.gate_networks[1]


def test_两个任务的输出():
    model = make_model()
    click, rating = model(torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    assert click.shape == (3,) and rating.shape == (3,)
    assert torch.all((click > 0) & (click < 1)), "点击率要过 sigmoid"


def test_损失是两个任务加权求和():
    model = make_model()
    batch = {
        "user_idx": torch.tensor([0, 1]), "item_idx": torch.tensor([2, 3]),
        "click_label": torch.tensor([1.0, 0.0]), "rating_label": torch.tensor([0.5, -1.0]),
    }
    click, rating = model(batch["user_idx"], batch["item_idx"])
    expected = 0.3 * F.binary_cross_entropy(click, batch["click_label"]) + 0.7 * F.mse_loss(rating, batch["rating_label"])
    assert multitask_loss(model, batch, 0.3, 0.7).item() == pytest.approx(expected.item())


@pytest.fixture
def recommender(small_ratings):
    return MMoERecommender(embedding_dim=8, expert_hidden_dims=[16, 8], gate_hidden_dims=[8],
                           task_hidden_dims=[8], n_epochs=1, device=CPU).fit(small_ratings)


def test_评分统计量来自训练集(recommender, small_ratings):
    from common.data import split_ratings
    train, _ = split_ratings(small_ratings)
    assert recommender.rating_mean == pytest.approx(train["rating"].mean())
    assert recommender.rating_std == pytest.approx(train["rating"].std())


def test_预测评分还原到原始尺度(recommender):
    click, rating = recommender.predict(1)
    assert click.shape == rating.shape == (len(recommender.item_map),)
    user = torch.tensor([recommender.user_map[1]])
    item = torch.tensor([0])
    _, normalized = recommender.model(user, item)
    assert rating[0].item() == pytest.approx(normalized.item() * recommender.rating_std + recommender.rating_mean, abs=1e-5)


@pytest.mark.parametrize("by", ["click", "rating", "hybrid"])
def test_推荐排除已交互物品且有序(recommender, small_ratings, by):
    recommendations = recommender.recommend(1, n_rec=5, by=by)
    rated = set(small_ratings.query("userId == 1")["movieId"])
    assert len(recommendations) == 5
    assert not rated & {item_id for item_id, _ in recommendations}
    scores = [score for _, score in recommendations]
    assert scores == sorted(scores, reverse=True)
