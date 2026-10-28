"""
SENet：给每个特征域学一个 0 到 1 的权重，重要的特征放大，不重要的压低。

输入是 m 个特征域的 embedding，组成 (m, k) 的矩阵（每个域要先投影到相同维度 k）。
1. squeeze     每个域的 embedding 取平均，得到 m 维向量
2. excitation  FC + ReLU 压缩，再 FC + sigmoid 恢复到 m 维，得到每个域的权重
3. reweight    每个域的 embedding 乘上自己的权重
"""
import torch
import torch.nn as nn


class SENet(nn.Module):
    def __init__(self, num_fields, reduction_ratio=3):
        super(SENet, self).__init__()
        # 压缩后的维度至少为 1
        reduced_size = max(1, num_fields // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size),  # 压缩
            nn.ReLU(),
            nn.Linear(reduced_size, num_fields),  # 恢复
            nn.Sigmoid()                          # 归一化为权重
        )

    def field_weights(self, inputs):
        """inputs: (batch, num_fields, dim) -> (batch, num_fields)"""
        squeezed = torch.mean(inputs, dim=2)
        return self.excitation(squeezed)

    def forward(self, inputs):
        weights = self.field_weights(inputs)
        return inputs * weights.unsqueeze(2)
