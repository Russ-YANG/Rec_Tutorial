"""
Bilinear 交叉：两个特征的哈达玛积之间加一个参数矩阵。

    p_ij = (v_i W) ⊙ v_j

参数矩阵有三种共享方式:
field_all          所有特征对共享一个矩阵
field_each         每个特征域一个矩阵，(i, j) 用 i 的矩阵
field_interaction  每个特征对一个矩阵
"""
import torch
import torch.nn as nn


def matrix_count(bilinear_type, num_fields):
    if bilinear_type == 'field_all':
        return 1
    if bilinear_type == 'field_each':
        return num_fields
    return num_fields * (num_fields - 1) // 2


def matrix_index(bilinear_type, i, pair_index):
    """特征对 (i, j) 用第几个矩阵，pair_index 是这个特征对的序号"""
    if bilinear_type == 'field_all':
        return 0
    if bilinear_type == 'field_each':
        return i
    return pair_index


def bilinear(vector_i, vector_j, matrix):
    """(v_i W) ⊙ v_j，vector: (batch, dim)，matrix: (dim, dim)"""
    return torch.matmul(vector_i, matrix) * vector_j


class BilinearInteraction(nn.Module):
    def __init__(self, embedding_dim, num_fields, bilinear_type='field_interaction'):
        super(BilinearInteraction, self).__init__()
        self.num_fields = num_fields
        self.bilinear_type = bilinear_type
        count = matrix_count(bilinear_type, num_fields)
        self.bilinear_matrices = nn.Parameter(torch.empty(count, embedding_dim, embedding_dim))
        nn.init.xavier_normal_(self.bilinear_matrices)

    def forward(self, inputs):
        """
        inputs: (batch, num_fields, dim)
        返回 (batch, num_pairs * dim)，所有 i < j 的特征对按顺序拼接
        """
        interactions = []
        pair_index = 0
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                matrix = self.bilinear_matrices[matrix_index(self.bilinear_type, i, pair_index)]
                interactions.append(bilinear(inputs[:, i, :], inputs[:, j, :], matrix))
                pair_index += 1
        return torch.cat(interactions, dim=1)
