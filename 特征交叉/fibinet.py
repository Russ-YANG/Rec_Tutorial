"""
FiBiNET = SENet + Bilinear。

1. 原始 embedding 做一次 Bilinear 交叉
2. SENet 加权后的 embedding 再做一次 Bilinear 交叉
3. 两路结果拼接，过 MLP 预测点击率
"""
import torch
import torch.nn as nn

from common.layers import build_mlp
from 特征交叉.bilinear_interaction import BilinearInteraction
from 特征交叉.senet import SENet


class FiBiNET(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 hidden_dims=[256, 128, 64],
                 dropout_rate=0.2,
                 reduction_ratio=3,
                 bilinear_type='field_interaction'):
        super(FiBiNET, self).__init__()
        self.senet = SENet(num_fields, reduction_ratio)
        # 两路交叉的参数不共享
        self.bilinear_interaction = BilinearInteraction(embedding_dim, num_fields, bilinear_type)
        self.se_bilinear_interaction = BilinearInteraction(embedding_dim, num_fields, bilinear_type)

        pair_count = num_fields * (num_fields - 1) // 2
        combined_dim = pair_count * embedding_dim * 2
        self.dnn_layers = build_mlp(combined_dim, hidden_dims, dropout=dropout_rate)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, feature_embeddings):
        """feature_embeddings: (batch, num_fields, dim)"""
        bilinear_output = self.bilinear_interaction(feature_embeddings)
        se_bilinear_output = self.se_bilinear_interaction(self.senet(feature_embeddings))
        combined = torch.cat([bilinear_output, se_bilinear_output], dim=1)
        output = torch.sigmoid(self.output_layer(self.dnn_layers(combined)))
        return output.squeeze(1)
