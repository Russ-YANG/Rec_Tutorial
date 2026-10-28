"""
SIM (Search-based Interest Model)：用户的行为序列很长时，DIN 对全部行为做 attention 太慢。
SIM 先粗筛再精排：

1. hard_search   按类别从长序列里找出和候选物品同类的最近 top_k 个行为
2. exact_search  只对筛出来的行为做 DIN 式的 attention（复用 DIN 的函数）
3. SIM           [候选物品, 兴趣向量] -> MLP -> 点击率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from 序列.din import attention_features, masked_softmax, sequence_mask


def hard_search(target_category_ids, history_item_ids, history_category_ids, history_lengths, top_k):
    """
    返回
    searched_item_ids: (batch, top_k)，不足 top_k 的位置填 0
    searched_lengths: (batch,)
    """
    batch_size = history_item_ids.size(0)
    device = history_item_ids.device
    searched_item_ids = torch.zeros(batch_size, top_k, dtype=torch.long, device=device)
    searched_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    for batch_index in range(batch_size):
        history_length = int(history_lengths[batch_index].item())
        write_index = 0
        # 从最近的行为开始找
        for position in range(history_length - 1, -1, -1):
            if history_category_ids[batch_index, position] == target_category_ids[batch_index]:
                searched_item_ids[batch_index, write_index] = history_item_ids[batch_index, position]
                write_index += 1
                if write_index == top_k:
                    break
        searched_lengths[batch_index] = write_index

    return searched_item_ids, searched_lengths


class SIM(nn.Module):
    def __init__(self, num_items, embedding_dim=32, search_top_k=20):
        super(SIM, self).__init__()
        self.search_top_k = search_top_k
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        self.output_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def hard_search(self, target_category_ids, history_item_ids, history_category_ids, history_lengths):
        return hard_search(target_category_ids, history_item_ids, history_category_ids,
                           history_lengths, self.search_top_k)

    def exact_search(self, target_item_ids, searched_item_ids, searched_lengths):
        target_embedding = self.item_embedding(target_item_ids)
        history_embeddings = self.item_embedding(searched_item_ids)

        features = attention_features(target_embedding, history_embeddings)
        scores = self.attention_net(features).squeeze(2)
        weights = masked_softmax(scores, sequence_mask(searched_lengths, searched_item_ids.size(1)))

        interest = (history_embeddings * weights.unsqueeze(2)).sum(dim=1)
        return interest, weights

    def forward(self, target_item_ids, target_category_ids, history_item_ids,
                history_category_ids, history_lengths):
        """返回 (点击率, 注意力权重, hard search 选出的行为)"""
        searched_item_ids, searched_lengths = self.hard_search(
            target_category_ids, history_item_ids, history_category_ids, history_lengths
        )
        interest, attention_weights = self.exact_search(target_item_ids, searched_item_ids, searched_lengths)

        features = torch.cat([self.item_embedding(target_item_ids), interest], dim=1)
        prediction = torch.sigmoid(self.output_net(features)).squeeze(1)
        return prediction, attention_weights, searched_item_ids
