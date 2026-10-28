import pandas as pd
import torch

from conftest import CPU
from 特征交叉.dcn import DCN, CrossNetwork, DCNRecommender


def test_交叉层公式():
    # x_{l+1} = x_0 * (x_l · w_l) + b_l + x_l
    network = CrossNetwork(input_dim=3, num_layers=2)
    with torch.no_grad():
        network.weights[0].weight.copy_(torch.tensor([[1.0, 0.0, 1.0]]))
        network.weights[1].weight.copy_(torch.tensor([[0.0, 1.0, 0.0]]))
        network.biases[0].copy_(torch.tensor([0.1, 0.2, 0.3]))
        network.biases[1].copy_(torch.tensor([0.0, 0.0, 1.0]))
    x0 = torch.tensor([[1.0, 2.0, 3.0]])

    x1 = x0 * 4.0 + torch.tensor([0.1, 0.2, 0.3]) + x0
    x2 = x0 * x1[0, 1] + torch.tensor([0.0, 0.0, 1.0]) + x1
    assert torch.allclose(network(x0), x2)


def test_权重和偏置为0时交叉网络是恒等映射():
    network = CrossNetwork(input_dim=4, num_layers=3)
    with torch.no_grad():
        for layer in network.weights:
            layer.weight.zero_()
    x = torch.randn(5, 4)
    assert torch.allclose(network(x), x)


def test_交叉网络不改变维度():
    network = CrossNetwork(input_dim=6, num_layers=3)
    assert network(torch.randn(2, 6)).shape == (2, 6)


def test_DCN输出点击率():
    model = DCN(num_users=5, num_items=6, embedding_dim=4, num_cross_layers=2, deep_hidden_dims=[8, 4])
    model.eval()
    output = model(torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    assert output.shape == (3,)
    assert torch.all((output > 0) & (output < 1))


def test_预测层输入是cross和deep的拼接():
    model = DCN(num_users=5, num_items=6, embedding_dim=4, deep_hidden_dims=[8, 3])
    assert model.prediction_layer.in_features == 4 * 2 + 3


def test_推荐排除已交互物品(small_csv):
    recommender = DCNRecommender(path=small_csv, embedding_dim=4, deep_hidden_dims=[8, 4], device=CPU)
    recommender.load_data()
    recommendations = recommender.recommend_for_user(1, n_rec=5)
    rated = set(pd.read_csv(small_csv).query("userId == 1")["movieId"])
    assert len(recommendations) == 5
    assert not rated & {item_id for item_id, _ in recommendations}
