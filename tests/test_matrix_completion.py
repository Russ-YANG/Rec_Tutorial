import pytest
import torch

from common.data import build_id_maps
from conftest import CPU
from 召回.matrix_completion import MatrixCompletion, MatrixCompletionModel, mse_loss


def test_预测评分是两个向量的内积():
    model = MatrixCompletionModel(n_users=3, n_items=2, emb_size=2)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1.0, 2.0], [0.0, 1.0], [3.0, 0.0]]))
        model.item_embedding.weight.copy_(torch.tensor([[2.0, 1.0], [1.0, -1.0]]))
    prediction = model(torch.tensor([0, 2]), torch.tensor([0, 1]))
    assert torch.allclose(prediction, torch.tensor([4.0, 3.0]))


def test_嵌入维度():
    model = MatrixCompletionModel(n_users=5, n_items=7, emb_size=16)
    assert model.get_user_embedding(torch.tensor([0, 1])).shape == (2, 16)
    assert model.get_item_embedding(torch.tensor([3])).shape == (1, 16)


def test_损失是均方误差():
    model = MatrixCompletionModel(n_users=2, n_items=2, emb_size=2)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.item_embedding.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    batch = {"user_idx": torch.tensor([0, 1]), "item_idx": torch.tensor([0, 1]), "rating": torch.tensor([4.0, 3.0])}
    # 预测 [2, 3]，误差 [2, 0]
    assert mse_loss(model, batch).item() == pytest.approx(2.0)


@pytest.fixture
def model(small_ratings):
    return MatrixCompletion(emb_size=8, batch_size=32, learning_rate=0.05, n_epochs=30, device=CPU).fit(small_ratings)


def test_训练能降低损失(model):
    assert model.history["train_loss"][-1] < model.history["train_loss"][0], "训练集 MSE 应该下降"
    assert model.rmse == pytest.approx(model.history["test_loss"][-1] ** 0.5)


def test_predict和模型输出一致(model, small_ratings):
    user_map, item_map = build_id_maps(small_ratings)
    expected = model.model(torch.tensor([user_map[1]]), torch.tensor([item_map[10]])).item()
    assert model.predict(1, 10) == pytest.approx(expected)


def test_推荐排除已评分物品且有序(model, small_ratings):
    recommendations = model.recommend(1, n_rec=5)
    rated = set(small_ratings.query("userId == 1")["movieId"])
    assert len(recommendations) == 5
    assert not rated & {item_id for item_id, _ in recommendations}
    scores = [score for _, score in recommendations]
    assert scores == sorted(scores, reverse=True)


def test_推荐分数就是预测评分(model):
    item_id, score = model.recommend(1, n_rec=1)[0]
    assert score == pytest.approx(model.predict(1, item_id), abs=1e-5)
