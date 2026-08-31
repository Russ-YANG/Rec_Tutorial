import torch
import torch.nn as nn


class BilinearInteraction(nn.Module):

    def __init__(self, embedding_dim, num_fields, bilinear_type='field_interaction'):
        super(BilinearInteraction, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.bilinear_type = bilinear_type

        # field_all: 所有特征对共享一个矩阵。
        # field_each: 每个特征域使用一个矩阵。
        # field_interaction: 每个特征对使用一个矩阵。
        if bilinear_type == 'field_all':
            matrix_count = 1
        elif bilinear_type == 'field_each':
            matrix_count = num_fields
        else:
            matrix_count = num_fields * (num_fields - 1) // 2

        matrices = torch.empty(matrix_count, embedding_dim, embedding_dim)
        self.bilinear_matrices = nn.Parameter(matrices)
        nn.init.xavier_normal_(self.bilinear_matrices)

    def forward(self, inputs):
        interactions = []
        pair_index = 0

        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                if self.bilinear_type == 'field_all':
                    matrix_index = 0
                elif self.bilinear_type == 'field_each':
                    matrix_index = i
                else:
                    matrix_index = pair_index

                vector_i = inputs[:, i, :]
                vector_j = inputs[:, j, :]
                transformed_i = torch.matmul(vector_i, self.bilinear_matrices[matrix_index])
                interaction = transformed_i * vector_j
                interactions.append(interaction)
                pair_index += 1

        return torch.cat(interactions, dim=1)
