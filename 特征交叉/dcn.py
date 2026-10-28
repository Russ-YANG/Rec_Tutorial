"""
DCN (Deep & Cross Network)：交叉网络显式构造有限阶特征交叉，
MLP 学习隐式的非线性组合，两者并行，输出拼接后预测点击率。

1. cross_layer    x_{l+1} = x_0 * (x_l · w_l) + b_l + x_l
2. CrossNetwork   堆叠若干交叉层，第 l 层最高包含 l+1 阶交叉
3. DCN            Embedding -> [交叉网络, MLP] -> 拼接 -> sigmoid
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.data import PointwiseDataset, build_id_maps, build_user_items, load_ratings, split_ratings
from common.layers import build_mlp, init_linear_layers
from common.recommend import score_all_items, top_n
from common.train import fit, get_device


def cross_layer(x0, xl, weight, bias):
    """
    x0, xl: (batch, dim)
    weight, bias: (dim,)
    x_{l+1} = x_0 * (x_l · w_l) + b_l + x_l
    x_l · w_l 是一个标量，所以计算量和 dim 成线性关系
    """
    xl_w = (xl * weight).sum(dim=1, keepdim=True)  # (batch, 1)
    return x0 * xl_w + bias + xl


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim) * 0.01) for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])

    def forward(self, x):
        x0 = x
        xl = x
        for weight, bias in zip(self.weights, self.biases):
            xl = cross_layer(x0, xl, weight, bias)
        return xl


class DCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64,
                 num_cross_layers=3, deep_hidden_dims=[128, 64]):
        super(DCN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        input_dim = embedding_dim * 2
        self.cross_network = CrossNetwork(input_dim, num_cross_layers)
        self.deep_network = build_mlp(input_dim, deep_hidden_dims, dropout=0.2)
        # 交叉网络输出维度不变，MLP 输出 deep_hidden_dims[-1]
        self.prediction_layer = nn.Linear(input_dim + deep_hidden_dims[-1], 1)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        init_linear_layers(self)

    def forward(self, user_idx, item_idx):
        x = torch.cat([self.user_embedding(user_idx), self.item_embedding(item_idx)], dim=1)
        combined = torch.cat([self.cross_network(x), self.deep_network(x)], dim=1)
        return torch.sigmoid(self.prediction_layer(combined)).squeeze(1)


def bce_loss(model, batch):
    prediction = model(batch['user_idx'], batch['item_idx'])
    return F.binary_cross_entropy(prediction, batch['label'])


class DCNRecommender:
    def __init__(self,
                 embedding_dim=64,
                 num_cross_layers=3,
                 deep_hidden_dims=[128, 64],
                 batch_size=256,
                 learning_rate=0.001,
                 weight_decay=0.0001,
                 n_epochs=20,
                 test_size=0.2,
                 negative_sampling=4,
                 device=None):
        self.embedding_dim = embedding_dim
        self.num_cross_layers = num_cross_layers
        self.deep_hidden_dims = deep_hidden_dims
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.test_size = test_size
        self.negative_sampling = negative_sampling
        self.device = get_device(device)

    def fit(self, ratings):
        self.user_map, self.item_map = build_id_maps(ratings)
        self.user_items = build_user_items(ratings)

        train_data, test_data = split_ratings(ratings, self.test_size)
        train_set = PointwiseDataset(train_data, self.user_map, self.item_map, self.user_items, self.negative_sampling)
        test_set = PointwiseDataset(test_data, self.user_map, self.item_map, self.user_items, self.negative_sampling)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size)

        self.model = DCN(len(self.user_map), len(self.item_map), self.embedding_dim,
                         self.num_cross_layers, self.deep_hidden_dims)
        self.model.to(self.device)

        self.history = fit(self.model, train_loader, test_loader, bce_loss, self.device,
                           n_epochs=self.n_epochs,
                           learning_rate=self.learning_rate,
                           weight_decay=self.weight_decay)
        return self

    def recommend(self, user_id, n_rec=10):
        self.model.eval()
        scores = score_all_items(self.model, self.user_map[user_id], len(self.item_map), self.device)
        item_scores = dict(zip(self.item_map, scores.tolist()))
        return top_n(item_scores, n_rec, exclude=self.user_items[user_id])


if __name__ == "__main__":
    model = DCNRecommender(n_epochs=20).fit(load_ratings())
    print(f"Train Loss = {model.history['train_loss'][-1]:.4f}, "
          f"Test Loss = {model.history['test_loss'][-1]:.4f}")

    user_id = 1
    print(f"\n为用户 {user_id} 的推荐物品:")
    for i, (item_id, score) in enumerate(model.recommend(user_id, n_rec=10)):
        print(f"{i+1}. 物品ID: {item_id}, 预测得分: {score:.4f}")
