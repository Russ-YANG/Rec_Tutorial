import torch
import torch.nn as nn
import torch.nn.functional as F


class SIM(nn.Module):
    """
    Search-based Interest Model 的教学版。

    第一步用类别做 hard search，从长行为序列中找出与候选物品同类的行为；
    第二步用 attention 做 exact search，得到候选物品相关的兴趣向量。
    """

    def __init__(self, num_items, embedding_dim=32, search_top_k=20):
        super(SIM, self).__init__()
        self.search_top_k = search_top_k
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)

        self.attention_layer_1 = nn.Linear(embedding_dim * 4, embedding_dim)
        self.attention_layer_2 = nn.Linear(embedding_dim, 1)

        self.output_layer_1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.output_layer_2 = nn.Linear(embedding_dim, 1)

    def hard_search(self, target_category_ids, history_item_ids,
                    history_category_ids, history_lengths):
        batch_size = history_item_ids.size(0)
        searched_item_ids = torch.zeros(
            batch_size,
            self.search_top_k,
            dtype=torch.long,
            device=history_item_ids.device
        )
        searched_lengths = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=history_item_ids.device
        )

        for batch_index in range(batch_size):
            history_length = int(history_lengths[batch_index].item())
            write_index = 0

            # 从最近的行为开始搜索。
            for position in range(history_length - 1, -1, -1):
                history_category = history_category_ids[batch_index, position]
                if history_category == target_category_ids[batch_index]:
                    searched_item_ids[batch_index, write_index] = history_item_ids[batch_index, position]
                    write_index += 1
                    if write_index == self.search_top_k:
                        break

            searched_lengths[batch_index] = write_index

        return searched_item_ids, searched_lengths

    def exact_search(self, target_item_ids, searched_item_ids, searched_lengths):
        target_embedding = self.item_embedding(target_item_ids)
        history_embeddings = self.item_embedding(searched_item_ids)

        sequence_length = searched_item_ids.size(1)
        repeated_target = target_embedding.unsqueeze(1)
        repeated_target = repeated_target.expand(-1, sequence_length, -1)

        attention_input = torch.cat(
            [
                repeated_target,
                history_embeddings,
                repeated_target * history_embeddings,
                repeated_target - history_embeddings
            ],
            dim=2
        )

        hidden = self.attention_layer_1(attention_input)
        hidden = F.relu(hidden)
        scores = self.attention_layer_2(hidden).squeeze(2)

        positions = torch.arange(sequence_length, device=searched_item_ids.device)
        positions = positions.unsqueeze(0)
        mask = positions < searched_lengths.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        weights = weights * mask.float()

        interest = history_embeddings * weights.unsqueeze(2)
        interest = interest.sum(dim=1)
        return interest, weights

    def forward(self, target_item_ids, target_category_ids, history_item_ids,
                history_category_ids, history_lengths):
        searched_item_ids, searched_lengths = self.hard_search(
            target_category_ids,
            history_item_ids,
            history_category_ids,
            history_lengths
        )

        interest, attention_weights = self.exact_search(
            target_item_ids,
            searched_item_ids,
            searched_lengths
        )

        target_embedding = self.item_embedding(target_item_ids)
        features = torch.cat([target_embedding, interest], dim=1)
        hidden = F.relu(self.output_layer_1(features))
        prediction = torch.sigmoid(self.output_layer_2(hidden)).squeeze(1)

        return prediction, attention_weights, searched_item_ids
