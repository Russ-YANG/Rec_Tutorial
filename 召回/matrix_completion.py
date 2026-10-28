"""
矩阵补全：用户和物品各学一个向量，内积拟合评分。
没评过分的格子用内积补全，分数高的就推荐。

1. MatrixCompletionModel  两个 Embedding，预测 = 内积
2. mse_loss               (真实评分 - 预测评分)^2
3. MatrixCompletion       训练 + 推荐
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.data import RatingDataset, build_id_maps, build_user_items, load_ratings, split_ratings
from common.recommend import recommend_by_embedding
from common.train import fit, get_device


class MatrixCompletionModel(nn.Module):
    def __init__(self, n_users, n_items, emb_size=50):
        super(MatrixCompletionModel, self).__init__()
        # 结构很简单，就是两个嵌入层
        self.user_embedding = nn.Embedding(n_users, emb_size)
        self.item_embedding = nn.Embedding(n_items, emb_size)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_idx, item_idx):
        user_embedding = self.user_embedding(user_idx)
        item_embedding = self.item_embedding(item_idx)
        return torch.sum(user_embedding * item_embedding, dim=1)  # 内积

    def get_user_embedding(self, user_idx):
        return self.user_embedding(user_idx)

    def get_item_embedding(self, item_idx):
        return self.item_embedding(item_idx)


def mse_loss(model, batch):
    prediction = model(batch['user_idx'], batch['item_idx'])
    return F.mse_loss(prediction, batch['rating'])


class MatrixCompletion:
    def __init__(self,
                 emb_size=50,
                 batch_size=64,
                 learning_rate=0.001,
                 weight_decay=0.00001,
                 n_epochs=20,
                 test_size=0.2,
                 device=None):
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.test_size = test_size
        self.device = get_device(device)

    def fit(self, ratings):
        self.user_map, self.item_map = build_id_maps(ratings)
        self.user_items = build_user_items(ratings)

        train_data, test_data = split_ratings(ratings, self.test_size)
        train_loader = DataLoader(RatingDataset(train_data, self.user_map, self.item_map),
                                  batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(RatingDataset(test_data, self.user_map, self.item_map),
                                 batch_size=self.batch_size)

        self.model = MatrixCompletionModel(len(self.user_map), len(self.item_map), self.emb_size)
        self.model.to(self.device)

        self.history = fit(self.model, train_loader, test_loader, mse_loss, self.device,
                           n_epochs=self.n_epochs,
                           learning_rate=self.learning_rate,
                           weight_decay=self.weight_decay)
        return self

    @property
    def rmse(self):
        return math.sqrt(self.history['test_loss'][-1])

    @torch.no_grad()
    def predict(self, user_id, item_id):
        self.model.eval()
        user_idx = torch.tensor([self.user_map[user_id]], device=self.device)
        item_idx = torch.tensor([self.item_map[item_id]], device=self.device)
        return self.model(user_idx, item_idx).item()

    @torch.no_grad()
    def recommend(self, user_id, n_rec=10):
        self.model.eval()
        user_idx = torch.tensor([self.user_map[user_id]], device=self.device)
        user_vector = self.model.get_user_embedding(user_idx)[0]
        # item_map 的顺序就是 Embedding 的行号顺序
        item_vectors = self.model.item_embedding.weight
        return recommend_by_embedding(user_vector, item_vectors, list(self.item_map), n_rec,
                                      exclude=self.user_items[user_id])


if __name__ == "__main__":
    model = MatrixCompletion(emb_size=50, batch_size=256, n_epochs=20).fit(load_ratings())
    print(f"训练结果: RMSE = {model.rmse:.4f}")

    user_id = 1
    print(f"\n为用户 {user_id} 的推荐物品:")
    for item_id, score in model.recommend(user_id, n_rec=10):
        print(f"物品ID: {item_id}, 预测评分: {score:.4f}")
