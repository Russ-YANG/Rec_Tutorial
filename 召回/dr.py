"""
Deep Retrieval：把每个物品放到若干条路径上，模型学习用户走到每条路径的概率。
召回时用 beam search 找出概率最高的几条路径，再取出路径上的物品。

路径有 num_levels 层，每层有 nodes_per_level 个节点。
节点用全局编号: 第 level 层的第 k 个节点编号为 level * nodes_per_level + k。

1. generate_item_paths  给每个物品随机分配路径（教学版，没有做 EM 更新路径）
2. DeepRetrieval        p(path | user) = Π_l p(c_l | user, c_1, ..., c_{l-1})
                        p(item | user) = Σ_{path ∈ item 的路径} p(path | user)
3. beam_search          逐层保留概率最高的 beam_size 条前缀
"""
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.data import build_id_maps, build_user_items, load_ratings, split_ratings
from common.recommend import top_n
from common.train import fit, get_device


def generate_item_paths(item_ids, num_paths, num_levels, nodes_per_level):
    """
    返回
    item_to_paths: {item: [path, ...]}
    path_to_items: {path: [item, ...]}，path 是节点编号的 tuple
    """
    item_to_paths = {}
    path_to_items = defaultdict(list)
    for item_id in item_ids:
        item_to_paths[item_id] = []
        for _ in range(num_paths):
            path = tuple(
                level * nodes_per_level + random.randint(0, nodes_per_level - 1)
                for level in range(num_levels)
            )
            item_to_paths[item_id].append(path)
            path_to_items[path].append(item_id)
    return item_to_paths, dict(path_to_items)


class PathDataset(Dataset):
    """每条交互: (用户, 交互物品对应的所有路径)"""

    def __init__(self, ratings, user_map, item_to_paths):
        self.users = [user_map[user_id] for user_id in ratings['userId']]
        self.paths = [item_to_paths[item_id] for item_id in ratings['movieId']]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return {
            'user_idx': torch.tensor(self.users[idx], dtype=torch.long),
            'target_paths': torch.tensor(self.paths[idx], dtype=torch.long)  # (num_paths, num_levels)
        }


class DeepRetrieval(nn.Module):
    def __init__(self, num_users, num_levels=3, nodes_per_level=100, embedding_dim=128, hidden_dim=256):
        super(DeepRetrieval, self).__init__()
        self.num_levels = num_levels
        self.nodes_per_level = nodes_per_level
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.node_embedding = nn.Embedding(num_levels * nodes_per_level, embedding_dim)

        # 第 l 层的输入: 用户向量 + 前 l 层已选节点的向量，输出这一层每个节点的 logit
        self.levels = nn.ModuleList()
        for level in range(num_levels):
            self.levels.append(nn.Sequential(
                nn.Linear(embedding_dim * (level + 1), hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, nodes_per_level)
            ))

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.node_embedding.weight, std=0.01)

    def level_log_probs(self, level, user_emb, prefix):
        """
        user_emb: (batch, dim)
        prefix: (batch, level)，前 level 层已选的节点
        返回 (batch, nodes_per_level)，这一层每个节点的对数概率
        """
        prefix_emb = self.node_embedding(prefix).reshape(prefix.size(0), prefix.size(1) * self.node_embedding.embedding_dim)
        level_input = torch.cat([user_emb, prefix_emb], dim=1)
        return F.log_softmax(self.levels[level](level_input), dim=1)

    def path_log_probs(self, user_idx, paths):
        """
        paths: (batch, num_paths, num_levels)
        返回 (batch, num_paths)，log p(path | user) = Σ_l log p(c_l | user, c_<l)
        """
        batch_size, num_paths, num_levels = paths.shape
        user_emb = self.user_embedding(user_idx)
        user_emb = user_emb.unsqueeze(1).expand(-1, num_paths, -1).reshape(batch_size * num_paths, -1)
        flat_paths = paths.reshape(batch_size * num_paths, num_levels)

        total = torch.zeros(batch_size * num_paths, device=paths.device)
        for level in range(num_levels):
            log_probs = self.level_log_probs(level, user_emb, flat_paths[:, :level])
            local_nodes = flat_paths[:, level] - level * self.nodes_per_level
            total = total + log_probs.gather(1, local_nodes.unsqueeze(1)).squeeze(1)
        return total.reshape(batch_size, num_paths)

    def forward(self, user_idx, target_paths):
        """log p(item | user) = log Σ_path p(path | user)"""
        return torch.logsumexp(self.path_log_probs(user_idx, target_paths), dim=1)

    @torch.no_grad()
    def beam_search(self, user_idx, beam_size=5):
        """
        返回
        paths: (batch, beam_size, num_levels)
        log_probs: (batch, beam_size)，从高到低
        """
        batch_size = user_idx.size(0)
        user_emb = self.user_embedding(user_idx)
        paths = torch.zeros(batch_size, 1, 0, dtype=torch.long, device=user_idx.device)
        scores = torch.zeros(batch_size, 1, device=user_idx.device)

        for level in range(self.num_levels):
            num_beams = paths.size(1)
            # 每条前缀往下扩展一层
            expanded_user = user_emb.unsqueeze(1).expand(-1, num_beams, -1).reshape(batch_size * num_beams, -1)
            log_probs = self.level_log_probs(level, expanded_user, paths.reshape(batch_size * num_beams, level))
            candidates = scores.unsqueeze(2) + log_probs.reshape(batch_size, num_beams, -1)

            # 所有候选里保留 beam_size 个
            k = min(beam_size, candidates[0].numel())
            scores, flat_index = candidates.reshape(batch_size, -1).topk(k, dim=1)
            beam_index = flat_index // self.nodes_per_level
            node = flat_index % self.nodes_per_level + level * self.nodes_per_level

            previous = paths.gather(1, beam_index.unsqueeze(2).expand(-1, -1, level))
            paths = torch.cat([previous, node.unsqueeze(2)], dim=2)

        return paths, scores


def path_loss(model, batch):
    # 最大化交互物品的概率
    return -model(batch['user_idx'], batch['target_paths']).mean()


class DeepRetrievalRecommender:
    def __init__(self,
                 embedding_dim=128,
                 hidden_dim=256,
                 batch_size=256,
                 learning_rate=0.001,
                 weight_decay=0.0001,
                 n_epochs=20,
                 test_size=0.2,
                 num_paths_per_item=3,
                 num_levels=3,
                 nodes_per_level=100,
                 device=None):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.test_size = test_size
        self.num_paths_per_item = num_paths_per_item
        self.num_levels = num_levels
        self.nodes_per_level = nodes_per_level
        self.device = get_device(device)

    def fit(self, ratings):
        self.user_map, self.item_map = build_id_maps(ratings)
        self.user_items = build_user_items(ratings)
        self.item_to_paths, self.path_to_items = generate_item_paths(
            list(self.item_map), self.num_paths_per_item, self.num_levels, self.nodes_per_level
        )

        train_data, test_data = split_ratings(ratings, self.test_size)
        train_loader = DataLoader(PathDataset(train_data, self.user_map, self.item_to_paths),
                                  batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(PathDataset(test_data, self.user_map, self.item_to_paths),
                                 batch_size=self.batch_size)

        self.model = DeepRetrieval(len(self.user_map), self.num_levels, self.nodes_per_level,
                                   self.embedding_dim, self.hidden_dim)
        self.model.to(self.device)

        self.history = fit(self.model, train_loader, test_loader, path_loss, self.device,
                           n_epochs=self.n_epochs,
                           learning_rate=self.learning_rate,
                           weight_decay=self.weight_decay)
        return self

    @torch.no_grad()
    def recommend(self, user_id, n_rec=10, beam_size=5):
        self.model.eval()
        user_idx = torch.tensor([self.user_map[user_id]], device=self.device)
        paths, log_probs = self.model.beam_search(user_idx, beam_size)

        # 物品分数 = 它所在的路径概率之和
        item_scores = defaultdict(float)
        for path, log_prob in zip(paths[0].tolist(), log_probs[0].exp().tolist()):
            for item_id in self.path_to_items.get(tuple(path), []):
                item_scores[item_id] += log_prob
        return top_n(item_scores, n_rec, exclude=self.user_items[user_id])


if __name__ == "__main__":
    model = DeepRetrievalRecommender(n_epochs=20).fit(load_ratings())
    print(f"Train Loss = {model.history['train_loss'][-1]:.4f}, "
          f"Test Loss = {model.history['test_loss'][-1]:.4f}")

    user_id = 1
    print(f"\n为用户 {user_id} 的推荐物品:")
    for item_id, score in model.recommend(user_id, n_rec=10, beam_size=5):
        print(f"物品ID: {item_id}, 得分: {score:.4f}")
