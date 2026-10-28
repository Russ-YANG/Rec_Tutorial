import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import CPU
from 召回.dssm import DSSM, DSSMRecommender, bce_loss, build_tower


def test_塔是多层MLP():
    tower = build_tower(8, [16, 4])
    linears = [layer for layer in tower if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [(8, 16), (16, 4)]
    tower.eval()
    assert tower(torch.randn(3, 8)).shape == (3, 4)


def make_model():
    model = DSSM(num_users=5, num_items=6, embedding_dim=8, hidden_layers=[16, 8])
    model.eval()
    return model


def test_用户和物品向量做了L2归一化():
    model = make_model()
    user_vectors = model.get_user_embedding(torch.tensor([0, 1]))
    item_vectors = model.get_item_embedding(torch.tensor([2, 3]))
    assert user_vectors.shape == (2, 8)
    assert torch.allclose(user_vectors.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(item_vectors.norm(dim=1), torch.ones(2), atol=1e-5)


def test_输出是余弦相似度():
    model = make_model()
    scores = model(torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    assert scores.shape == (3,)
    assert torch.all(scores.abs() <= 1.0 + 1e-5)


def test_双塔可以分开算再做内积():
    model = make_model()
    users, items = torch.tensor([0, 1]), torch.tensor([2, 3])
    separate = (model.get_user_embedding(users) * model.get_item_embedding(items)).sum(dim=1)
    assert torch.allclose(model(users, items), separate, atol=1e-6)


def test_用户塔和物品塔参数独立():
    model = make_model()
    assert model.user_tower is not model.item_tower
    assert model.user_tower[0].weight is not model.item_tower[0].weight


def test_损失是二元交叉熵():
    model = make_model()
    batch = {"user_idx": torch.tensor([0, 1]), "item_idx": torch.tensor([2, 3]), "label": torch.tensor([1.0, 0.0])}
    scores = model(batch["user_idx"], batch["item_idx"])
    expected = F.binary_cross_entropy_with_logits(scores, batch["label"])
    assert bce_loss(model, batch).item() == pytest.approx(expected.item())


@pytest.fixture
def recommender(small_ratings):
    return DSSMRecommender(embedding_dim=8, hidden_layers=[16, 8], n_epochs=1, device=CPU).fit(small_ratings)


def test_推荐排除已交互物品且有序(recommender, small_ratings):
    recommendations = recommender.recommend(1, n_rec=5)
    rated = set(small_ratings.query("userId == 1")["movieId"])
    assert len(recommendations) == 5
    assert not rated & {item_id for item_id, _ in recommendations}
    scores = [score for _, score in recommendations]
    assert scores == sorted(scores, reverse=True)


def test_推荐分数就是余弦相似度(recommender):
    item_id, score = recommender.recommend(1, n_rec=1)[0]
    user = torch.tensor([recommender.user_map[1]])
    item = torch.tensor([recommender.item_map[item_id]])
    assert score == pytest.approx(recommender.model(user, item).item(), abs=1e-5)
