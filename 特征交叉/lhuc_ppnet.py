"""
LHUC / PPNet：用个性化特征（这里是用户特征）生成逐层的门值，调节主网络的隐藏表示。

1. lhuc_gate   gate = 2 * sigmoid(W u + b)，范围 (0, 2)，初始附近为 1
2. 主网络每一层: h = ReLU(W h) * gate(u)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def lhuc_gate(gate_layer, user_features):
    """乘 2 让门值以 1 为中心，既能放大也能缩小"""
    return 2.0 * torch.sigmoid(gate_layer(user_features))


class LHUC_PPNet(nn.Module):
    def __init__(self,
                 user_dim,
                 item_dim,
                 hidden_dims=[128, 64],
                 dropout_rate=0.2):
        super(LHUC_PPNet, self).__init__()
        self.main_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        input_dim = user_dim + item_dim
        for hidden_dim in hidden_dims:
            self.main_layers.append(nn.Linear(input_dim, hidden_dim))
            # 门控只看用户特征，输出和这一层一样宽
            self.gate_layers.append(nn.Linear(user_dim, hidden_dim))
            input_dim = hidden_dim

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, user_features, item_features):
        hidden = torch.cat([user_features, item_features], dim=1)
        for main_layer, gate_layer in zip(self.main_layers, self.gate_layers):
            hidden = F.relu(main_layer(hidden))
            hidden = hidden * lhuc_gate(gate_layer, user_features)
            hidden = self.dropout(hidden)
        return self.output_layer(hidden).squeeze(1)
