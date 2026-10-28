"""
DIN (Deep Interest Network)：用候选物品对用户的历史行为做 attention，
和候选物品相关的历史行为权重更大，得到针对这个候选物品的兴趣向量。

1. attention_features  [query, key, query*key, query-key]
2. sequence_mask       历史序列里哪些位置是真实行为，哪些是 padding
3. masked_softmax      padding 位置的权重为 0
4. AttentionLayer      MLP 给每个历史行为打分，加权求和得到兴趣向量
5. DIN                 [用户, 候选物品, 兴趣向量] -> MLP -> 点击率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import build_mlp, init_linear_layers


def attention_features(query, keys):
    """
    query: (batch, dim)
    keys: (batch, seq_len, dim)
    返回 (batch, seq_len, dim * 4)
    """
    queries = query.unsqueeze(1).expand_as(keys)
    return torch.cat([queries, keys, queries * keys, queries - keys], dim=-1)


def sequence_mask(lengths, max_len):
    """lengths: (batch,) -> (batch, max_len)，前 length 个位置为 True"""
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def masked_softmax(scores, mask):
    """
    只在 mask 为 True 的位置做 softmax，其余位置权重为 0。
    整行都是 padding 时，权重全为 0。
    """
    scores = scores.masked_fill(~mask, -1e9)
    return F.softmax(scores, dim=1) * mask.float()


class AttentionLayer(nn.Module):
    # 论文里的激活函数是 Dice，这里简化成 ReLU
    def __init__(self, embedding_dim, attention_hidden_units=[80, 40]):
        super(AttentionLayer, self).__init__()
        self.attention_net = nn.Sequential(
            build_mlp(embedding_dim * 4, attention_hidden_units),
            nn.Linear(attention_hidden_units[-1], 1)
        )

    def forward(self, query, keys, keys_length):
        """
        返回
        interest: (batch, dim)，历史行为的加权和
        weights: (batch, seq_len)
        """
        batch_size, max_seq_len, dim = keys.shape
        features = attention_features(query, keys).reshape(batch_size * max_seq_len, dim * 4)
        scores = self.attention_net(features).reshape(batch_size, max_seq_len)

        weights = masked_softmax(scores, sequence_mask(keys_length, max_seq_len))
        interest = (keys * weights.unsqueeze(-1)).sum(dim=1)
        return interest, weights


class DIN(nn.Module):
    def __init__(self,
                 num_users,
                 num_items,
                 embedding_dim=64,
                 mlp_hidden_units=[200, 80],
                 attention_hidden_units=[80, 40],
                 dropout_rate=0.2):
        super(DIN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.attention_layer = AttentionLayer(embedding_dim, attention_hidden_units)
        self.mlp = nn.Sequential(
            build_mlp(embedding_dim * 3, mlp_hidden_units, dropout=dropout_rate),
            nn.Linear(mlp_hidden_units[-1], 1)
        )

        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        init_linear_layers(self.mlp)

    def forward(self, user_ids, item_ids, history_item_ids, history_length):
        """
        history_item_ids: (batch, max_seq_len)
        history_length: (batch,)
        返回 (点击率, 注意力权重)
        """
        user_emb = self.user_embedding(user_ids)
        target_item_emb = self.item_embedding(item_ids)
        history_emb = self.item_embedding(history_item_ids)

        interest_emb, attention_weights = self.attention_layer(target_item_emb, history_emb, history_length)

        features = torch.cat([user_emb, target_item_emb, interest_emb], dim=1)
        prediction = torch.sigmoid(self.mlp(features)).squeeze(1)
        return prediction, attention_weights
