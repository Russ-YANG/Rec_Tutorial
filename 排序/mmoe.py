"""
MMoE 多任务排序：同时预测点击率和评分。

多个专家网络共享输入，每个任务有自己的门控，按门控权重混合专家的输出，
再接任务自己的塔。不同任务可以偏向不同的专家，避免互相拖累。

1. MultiTaskDataset  两个标签: 点击(评分 >= 4) 和归一化后的评分
2. mix_experts       Σ_k gate_k * expert_k
3. MMoE              专家 + 每个任务的门控 + 每个任务的塔
4. multitask_loss    点击用 BCE，评分用 MSE，加权求和
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.data import build_id_maps, build_user_items, load_ratings, split_ratings
from common.layers import build_mlp, init_linear_layers
from common.recommend import score_all_items, top_n
from common.train import fit, get_device


class MultiTaskDataset(Dataset):
    def __init__(self, ratings, user_map, item_map, rating_mean, rating_std):
        self.users = [user_map[user_id] for user_id in ratings['userId']]
        self.items = [item_map[item_id] for item_id in ratings['movieId']]
        self.ratings = ratings['rating'].tolist()
        # 评分用训练集的均值和标准差归一化，测试集也用同一套
        self.rating_mean = rating_mean
        self.rating_std = rating_std

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        rating = self.ratings[idx]
        return {
            'user_idx': torch.tensor(self.users[idx], dtype=torch.long),
            'item_idx': torch.tensor(self.items[idx], dtype=torch.long),
            'click_label': torch.tensor(1.0 if rating >= 4.0 else 0.0, dtype=torch.float),
            'rating_label': torch.tensor((rating - self.rating_mean) / self.rating_std, dtype=torch.float)
        }


def mix_experts(expert_outputs, gate_weights):
    """
    expert_outputs: (batch, num_experts, dim)
    gate_weights: (batch, num_experts)，每行和为 1
    返回 (batch, dim) = Σ_k gate_k * expert_k
    """
    return (expert_outputs * gate_weights.unsqueeze(2)).sum(dim=1)


class MMoE(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64,
                 expert_hidden_dims=[128, 64], gate_hidden_dims=[64],
                 task_hidden_dims=[64], num_experts=3, num_tasks=2):
        super(MMoE, self).__init__()
        self.num_tasks = num_tasks
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        input_dim = embedding_dim * 2

        # 专家: 所有任务共享
        self.experts = nn.ModuleList([
            build_mlp(input_dim, expert_hidden_dims) for _ in range(num_experts)
        ])

        # 门控: 每个任务一个，输出在专家上的概率分布
        self.gate_networks = nn.ModuleList()
        for _ in range(num_tasks):
            self.gate_networks.append(nn.Sequential(
                build_mlp(input_dim, gate_hidden_dims, batch_norm=False),
                nn.Linear(gate_hidden_dims[-1] if gate_hidden_dims else input_dim, num_experts),
                nn.Softmax(dim=1)
            ))

        # 任务塔: 每个任务一个
        self.task_towers = nn.ModuleList()
        for _ in range(num_tasks):
            self.task_towers.append(nn.Sequential(
                build_mlp(expert_hidden_dims[-1], task_hidden_dims, dropout=0.2),
                nn.Linear(task_hidden_dims[-1] if task_hidden_dims else expert_hidden_dims[-1], 1)
            ))

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        init_linear_layers(self)
        # 门控输出层用很小的权重，开始时各专家权重接近平均
        for gate in self.gate_networks:
            nn.init.normal_(gate[1].weight, std=0.01)

    def forward(self, user_idx, item_idx):
        features = torch.cat([self.user_embedding(user_idx), self.item_embedding(item_idx)], dim=1)
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)

        task_outputs = []
        for task_id in range(self.num_tasks):
            gate_weights = self.gate_networks[task_id](features)
            mixed = mix_experts(expert_outputs, gate_weights)
            task_outputs.append(self.task_towers[task_id](mixed).squeeze(-1))

        click_pred = torch.sigmoid(task_outputs[0])
        rating_pred = task_outputs[1]  # 预测的是归一化评分
        return click_pred, rating_pred


def multitask_loss(model, batch, click_weight=0.5, rating_weight=0.5):
    click_pred, rating_pred = model(batch['user_idx'], batch['item_idx'])
    click_loss = F.binary_cross_entropy(click_pred, batch['click_label'])
    rating_loss = F.mse_loss(rating_pred, batch['rating_label'])
    return click_weight * click_loss + rating_weight * rating_loss


class MMoERecommender:
    def __init__(self,
                 embedding_dim=64,
                 expert_hidden_dims=[128, 64],
                 gate_hidden_dims=[64],
                 task_hidden_dims=[64],
                 num_experts=3,
                 batch_size=256,
                 learning_rate=0.001,
                 weight_decay=0.0001,
                 n_epochs=20,
                 test_size=0.2,
                 device=None):
        self.embedding_dim = embedding_dim
        self.expert_hidden_dims = expert_hidden_dims
        self.gate_hidden_dims = gate_hidden_dims
        self.task_hidden_dims = task_hidden_dims
        self.num_experts = num_experts
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
        self.rating_mean = train_data['rating'].mean()
        self.rating_std = train_data['rating'].std()
        stats = (self.user_map, self.item_map, self.rating_mean, self.rating_std)
        train_loader = DataLoader(MultiTaskDataset(train_data, *stats), batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(MultiTaskDataset(test_data, *stats), batch_size=self.batch_size)

        self.model = MMoE(len(self.user_map), len(self.item_map), self.embedding_dim,
                          self.expert_hidden_dims, self.gate_hidden_dims, self.task_hidden_dims,
                          self.num_experts)
        self.model.to(self.device)

        self.history = fit(self.model, train_loader, test_loader, multitask_loss, self.device,
                           n_epochs=self.n_epochs,
                           learning_rate=self.learning_rate,
                           weight_decay=self.weight_decay)
        return self

    @torch.no_grad()
    def predict(self, user_id):
        """返回用户对所有物品的 (点击概率, 预测评分)，评分已还原到原始尺度"""
        self.model.eval()

        def score(user_idx, item_idx):
            return torch.stack(self.model(user_idx, item_idx), dim=1)

        scores = score_all_items(score, self.user_map[user_id], len(self.item_map), self.device)
        click_pred = scores[:, 0]
        rating_pred = scores[:, 1] * self.rating_std + self.rating_mean
        return click_pred, rating_pred

    def recommend(self, user_id, n_rec=10, by='hybrid'):
        """
        by='click'  按点击概率排序
        by='rating' 按预测评分排序
        by='hybrid' 点击概率 + 归一化评分
        """
        click_pred, rating_pred = self.predict(user_id)
        if by == 'click':
            scores = click_pred
        elif by == 'rating':
            scores = rating_pred
        else:
            scores = click_pred + (rating_pred - self.rating_mean) / self.rating_std
        item_scores = dict(zip(self.item_map, scores.tolist()))
        return top_n(item_scores, n_rec, exclude=self.user_items[user_id])


if __name__ == "__main__":
    model = MMoERecommender(n_epochs=20).fit(load_ratings())
    print(f"Train Loss = {model.history['train_loss'][-1]:.4f}, "
          f"Test Loss = {model.history['test_loss'][-1]:.4f}")

    user_id = 1
    click_pred, rating_pred = model.predict(user_id)
    print(f"\n为用户 {user_id} 的推荐物品:")
    for i, (item_id, score) in enumerate(model.recommend(user_id, n_rec=10)):
        idx = model.item_map[item_id]
        print(f"{i+1}. 物品ID: {item_id}, 点击概率: {click_pred[idx]:.4f}, 预测评分: {rating_pred[idx]:.2f}")
