"""
LightGCN：在用户-物品二部图上做图卷积。和 GCN 相比去掉了特征变换和非线性激活，
每一层只是把邻居的向量按度数归一化后加起来。

1. build_normalized_adjacency  Â = D^-1/2 A D^-1/2，用户节点在前，物品节点在后
2. propagate                    E^(k+1) = Â E^(k)
3. 最终向量 = 各层向量的平均
4. bpr_loss                     -log σ(score_正样本 - score_负样本)
5. LightGCNRecommender          BPR 训练 + 向量召回
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.data import PairwiseDataset, build_id_maps, build_user_items, load_ratings, split_ratings
from common.recommend import recommend_by_embedding
from common.train import fit, get_device


def build_normalized_adjacency(num_users, num_items, user_indices, item_indices):
    """
    A = [[0, R], [R^T, 0]]，R 是用户-物品交互矩阵
    Â_ui = 1 / sqrt(d_u * d_i)
    """
    shifted_item_indices = item_indices + num_users
    row = torch.cat([user_indices, shifted_item_indices])
    column = torch.cat([shifted_item_indices, user_indices])
    indices = torch.stack([row, column])
    values = torch.ones(indices.size(1), dtype=torch.float)

    node_count = num_users + num_items
    adjacency = torch.sparse_coo_tensor(indices, values, (node_count, node_count), check_invariants=False).coalesce()

    degree = torch.zeros(node_count)
    degree.scatter_add_(0, adjacency.indices()[0], adjacency.values())
    inverse_sqrt_degree = degree.pow(-0.5)

    row, column = adjacency.indices()
    normalized_values = adjacency.values() * inverse_sqrt_degree[row] * inverse_sqrt_degree[column]
    return torch.sparse_coo_tensor(adjacency.indices(), normalized_values, adjacency.size(), check_invariants=False).coalesce()


def propagate(adjacency, embeddings, num_layers):
    """返回 [E^(0), E^(1), ..., E^(num_layers)]"""
    layers = [embeddings]
    for _ in range(num_layers):
        embeddings = torch.sparse.mm(adjacency, embeddings)
        layers.append(embeddings)
    return layers


def bpr_loss(positive_scores, negative_scores):
    """正样本分数要比负样本高: -log σ(s_pos - s_neg)"""
    return -F.logsigmoid(positive_scores - negative_scores).mean()


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def compute_embeddings(self, adjacency_matrix):
        """返回所有用户和所有物品的最终向量"""
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layers = propagate(adjacency_matrix, all_embeddings, self.num_layers)
        final_embedding = torch.stack(layers, dim=1).mean(dim=1)
        return final_embedding[:self.num_users], final_embedding[self.num_users:]

    def forward(self, user_indices, item_indices, adjacency_matrix):
        user_embeddings, item_embeddings = self.compute_embeddings(adjacency_matrix)
        return user_embeddings[user_indices], item_embeddings[item_indices]

    def predict(self, user_indices, item_indices, adjacency_matrix):
        user_embeds, item_embeds = self.forward(user_indices, item_indices, adjacency_matrix)
        return torch.sum(user_embeds * item_embeds, dim=1)

    def bpr_loss(self, user_indices, positive_item_indices, negative_item_indices, adjacency_matrix):
        # 图传播只算一次，正负样本共用
        user_embeddings, item_embeddings = self.compute_embeddings(adjacency_matrix)
        users = user_embeddings[user_indices]
        positive_scores = torch.sum(users * item_embeddings[positive_item_indices], dim=1)
        negative_scores = torch.sum(users * item_embeddings[negative_item_indices], dim=1)
        return bpr_loss(positive_scores, negative_scores)


class LightGCNRecommender:
    def __init__(self,
                 embedding_dim=64,
                 num_layers=3,
                 batch_size=1024,
                 learning_rate=0.001,
                 weight_decay=0.0001,
                 n_epochs=20,
                 test_size=0.2,
                 device=None):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
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
        # 图只用训练集的边，避免测试集信息泄露
        user_indices = torch.tensor([self.user_map[user_id] for user_id in train_data['userId']])
        item_indices = torch.tensor([self.item_map[item_id] for item_id in train_data['movieId']])
        self.adjacency = build_normalized_adjacency(len(self.user_map), len(self.item_map),
                                                    user_indices, item_indices).to(self.device)

        train_loader = DataLoader(PairwiseDataset(train_data, self.user_map, self.item_map, self.user_items),
                                  batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(PairwiseDataset(test_data, self.user_map, self.item_map, self.user_items),
                                 batch_size=self.batch_size)

        self.model = LightGCN(len(self.user_map), len(self.item_map), self.embedding_dim, self.num_layers)
        self.model.to(self.device)

        def compute_loss(model, batch):
            return model.bpr_loss(batch['user_idx'], batch['pos_idx'], batch['neg_idx'], self.adjacency)

        self.history = fit(self.model, train_loader, test_loader, compute_loss, self.device,
                           n_epochs=self.n_epochs,
                           learning_rate=self.learning_rate,
                           weight_decay=self.weight_decay)
        return self

    @torch.no_grad()
    def recommend(self, user_id, n_rec=10):
        self.model.eval()
        user_embeddings, item_embeddings = self.model.compute_embeddings(self.adjacency)
        user_vector = user_embeddings[self.user_map[user_id]]
        return recommend_by_embedding(user_vector, item_embeddings, list(self.item_map), n_rec,
                                      exclude=self.user_items[user_id])


if __name__ == "__main__":
    model = LightGCNRecommender(n_epochs=20).fit(load_ratings())
    print(f"Train Loss = {model.history['train_loss'][-1]:.4f}, "
          f"Test Loss = {model.history['test_loss'][-1]:.4f}")

    user_id = 1
    print(f"\n为用户 {user_id} 的推荐物品:")
    for item_id, score in model.recommend(user_id, n_rec=10):
        print(f"物品ID: {item_id}, 分数: {score:.4f}")
