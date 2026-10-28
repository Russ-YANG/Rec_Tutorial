"""
DSSM 双塔：用户塔和物品塔各自把 id 变成向量，用余弦相似度衡量匹配程度。
两个塔互不依赖，线上可以提前算好所有物品向量，做向量召回。

1. build_mlp     每个塔都是 Embedding 后面接一个 MLP（在 common/layers.py）
2. DSSM          双塔 + 余弦相似度
3. bce_loss      pointwise 训练：正样本标签 1，随机负样本标签 0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.data import PointwiseDataset, build_id_maps, build_user_items, load_ratings, split_ratings
from common.layers import build_mlp, init_linear_layers
from common.recommend import recommend_by_embedding
from common.train import fit, get_device


class DSSM(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, hidden_layers=[512, 256]):
        super(DSSM, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_tower = build_mlp(embedding_dim, hidden_layers)
        self.item_tower = build_mlp(embedding_dim, hidden_layers)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        init_linear_layers(self.user_tower)
        init_linear_layers(self.item_tower)

    def get_user_embedding(self, user_idx):
        user_vector = self.user_tower(self.user_embedding(user_idx))
        return F.normalize(user_vector, p=2, dim=1)  # L2 归一化后，内积就是余弦相似度

    def get_item_embedding(self, item_idx):
        item_vector = self.item_tower(self.item_embedding(item_idx))
        return F.normalize(item_vector, p=2, dim=1)

    def forward(self, user_idx, item_idx):
        user_vector = self.get_user_embedding(user_idx)
        item_vector = self.get_item_embedding(item_idx)
        return torch.sum(user_vector * item_vector, dim=1)


def bce_loss(model, batch):
    # 把余弦相似度当作 logit
    scores = model(batch['user_idx'], batch['item_idx'])
    return F.binary_cross_entropy_with_logits(scores, batch['label'])


class DSSMRecommender:
    def __init__(self,
                 embedding_dim=128,
                 hidden_layers=[512, 256],
                 batch_size=256,
                 learning_rate=0.001,
                 weight_decay=0.0001,
                 n_epochs=20,
                 test_size=0.2,
                 negative_sampling=4,
                 device=None):
        """negative_sampling: 每个正样本配几个负样本"""
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
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

        self.model = DSSM(len(self.user_map), len(self.item_map), self.embedding_dim, self.hidden_layers)
        self.model.to(self.device)

        self.history = fit(self.model, train_loader, test_loader, bce_loss, self.device,
                           n_epochs=self.n_epochs,
                           learning_rate=self.learning_rate,
                           weight_decay=self.weight_decay)
        return self

    @torch.no_grad()
    def recommend(self, user_id, n_rec=10):
        self.model.eval()
        user_idx = torch.tensor([self.user_map[user_id]], device=self.device)
        user_vector = self.model.get_user_embedding(user_idx)[0]
        # 物品塔可以离线把所有物品向量算好
        all_items = torch.arange(len(self.item_map), device=self.device)
        item_vectors = self.model.get_item_embedding(all_items)
        return recommend_by_embedding(user_vector, item_vectors, list(self.item_map), n_rec,
                                      exclude=self.user_items[user_id])


if __name__ == "__main__":
    model = DSSMRecommender(n_epochs=20).fit(load_ratings())
    print(f"Train Loss = {model.history['train_loss'][-1]:.4f}, "
          f"Test Loss = {model.history['test_loss'][-1]:.4f}")

    user_id = 1
    print(f"\n为用户 {user_id} 的推荐物品:")
    for item_id, score in model.recommend(user_id, n_rec=10):
        print(f"物品ID: {item_id}, 相似度: {score:.4f}")
