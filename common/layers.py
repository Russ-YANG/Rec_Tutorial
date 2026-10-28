"""
各个模型共用的网络层。
"""
import torch.nn as nn


def build_mlp(input_dim, hidden_dims, dropout=0.0, batch_norm=True):
    """
    多层感知机: 每层 Linear -> ReLU -> (BatchNorm) -> (Dropout)
    输出维度是 hidden_dims[-1]，不带最后的输出层
    """
    layers = []
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        input_dim = hidden_dim
    return nn.Sequential(*layers)


def init_linear_layers(module):
    """所有 Linear 层用 xavier 初始化，偏置置 0"""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
