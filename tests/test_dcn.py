import pytest
import torch
import torch.nn.functional as F

from conftest import CPU
from 特征交叉.dcn import DCN, CrossNetwork, DCNRecommender, bce_loss, cross_layer


def test_交叉层公式():
    # x_{l+1} = x_0 * (x_l · w_l) + b_l + x_l
    x0 = torch.tensor([[1.0, 2.0, 3.0]])
    xl = torch.tensor([[0.5, 0.0, 1.0]])
    weight = torch.tensor([1.0, 2.0, 1.0])
    bias = torch.tensor([0.1, 0.2, 0.3])
    # x_l · w = 0.5 + 0 + 1 = 1.5
    expected = x0 * 1.5 + bias + xl
    assert torch.allclose(cross_layer(x0, xl, weight, bias), expected)


def test_交叉层逐样本计算():
    x0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weight = torch.tensor([1.0, 1.0])
    output = cross_layer(x0, x0, weight, torch.zeros(2))
    assert torch.allclose(output, 2 * x0)


def test_交叉网络堆叠多层():
    network = CrossNetwork(input_dim=3, num_layers=2)
    x0 = torch.randn(4, 3)
    x1 = cross_layer(x0, x0, network.weights[0], network.biases[0])
    x2 = cross_layer(x0, x1, network.weights[1], network.biases[1])
    assert torch.allclose(network(x0), x2)


def test_权重和偏置为0时交叉网络是恒等映射():
    network = CrossNetwork(input_dim=4, num_layers=3)
    with torch.no_grad():
        for weight in network.weights:
            weight.zero_()
    x = torch.randn(5, 4)
    assert torch.allclose(network(x), x)


def test_DCN输出点击率():
    model = DCN(num_users=5, num_items=6, embedding_dim=4, num_cross_layers=2, deep_hidden_dims=[8, 4])
    model.eval()
    output = model(torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    assert output.shape == (3,)
    assert torch.all((output > 0) & (output < 1))


def test_预测层输入是cross和deep的拼接():
    model = DCN(num_users=5, num_items=6, embedding_dim=4, deep_hidden_dims=[8, 3])
    assert model.prediction_layer.in_features == 4 * 2 + 3


def test_损失是二元交叉熵():
    model = DCN(num_users=5, num_items=6, embedding_dim=4, deep_hidden_dims=[8])
    model.eval()
    batch = {"user_idx": torch.tensor([0, 1]), "item_idx": torch.tensor([2, 3]), "label": torch.tensor([1.0, 0.0])}
    expected = F.binary_cross_entropy(model(batch["user_idx"], batch["item_idx"]), batch["label"])
    assert bce_loss(model, batch).item() == pytest.approx(expected.item())


def test_推荐排除已交互物品且有序(small_ratings):
    recommender = DCNRecommender(embedding_dim=4, deep_hidden_dims=[8, 4], n_epochs=1, device=CPU).fit(small_ratings)
    recommendations = recommender.recommend(1, n_rec=5)
    rated = set(small_ratings.query("userId == 1")["movieId"])
    assert len(recommendations) == 5
    assert not rated & {item_id for item_id, _ in recommendations}
    scores = [score for _, score in recommendations]
    assert scores == sorted(scores, reverse=True)
