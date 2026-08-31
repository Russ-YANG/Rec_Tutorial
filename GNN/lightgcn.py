import torch
import torch.nn as nn


def build_normalized_adjacency(num_users, num_items, user_indices, item_indices):
    # 用户节点排在前面，物品节点排在后面。
    shifted_item_indices = item_indices + num_users

    row = torch.cat([user_indices, shifted_item_indices])
    column = torch.cat([shifted_item_indices, user_indices])
    indices = torch.stack([row, column])

    values = torch.ones(indices.size(1), dtype=torch.float)
    node_count = num_users + num_items
    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        (node_count, node_count)
    ).coalesce()

    degree = torch.zeros(node_count)
    degree.scatter_add_(0, adjacency.indices()[0], adjacency.values())
    inverse_sqrt_degree = degree.pow(-0.5)

    normalized_values = (
        adjacency.values()
        * inverse_sqrt_degree[adjacency.indices()[0]]
        * inverse_sqrt_degree[adjacency.indices()[1]]
    )
    normalized_adjacency = torch.sparse_coo_tensor(
        adjacency.indices(),
        normalized_values,
        adjacency.size()
    )
    return normalized_adjacency.coalesce()


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_indices, item_indices, adjacency_matrix):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]

        for layer in range(self.num_layers):
            all_embeddings = torch.sparse.mm(adjacency_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        final_embedding = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)

        user_final_embeddings = final_embedding[:self.num_users]
        item_final_embeddings = final_embedding[self.num_users:]

        user_embeds = user_final_embeddings[user_indices]
        item_embeds = item_final_embeddings[item_indices]

        return user_embeds, item_embeds

    def predict(self, user_indices, item_indices, adjacency_matrix):
        user_embeds, item_embeds = self.forward(user_indices, item_indices, adjacency_matrix)
        scores = torch.sum(user_embeds * item_embeds, dim=1)
        return scores

    def bpr_loss(self, user_indices, positive_item_indices,
                 negative_item_indices, adjacency_matrix):
        user_embeddings, positive_embeddings = self.forward(
            user_indices,
            positive_item_indices,
            adjacency_matrix
        )
        _, negative_embeddings = self.forward(
            user_indices,
            negative_item_indices,
            adjacency_matrix
        )

        positive_scores = torch.sum(user_embeddings * positive_embeddings, dim=1)
        negative_scores = torch.sum(user_embeddings * negative_embeddings, dim=1)
        loss = -torch.log(torch.sigmoid(positive_scores - negative_scores)).mean()
        return loss
