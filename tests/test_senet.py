import torch

from 特征交叉.senet import SENet


def test_SENet不改变形状():
    senet = SENet(num_fields=4, reduction_ratio=2)
    inputs = torch.randn(3, 4, 5)
    assert senet(inputs).shape == (3, 4, 5)


def test_SENet给每个特征域乘一个0到1的权重():
    senet = SENet(num_fields=4, reduction_ratio=2)
    inputs = torch.rand(3, 4, 5) + 0.5
    ratio = senet(inputs) / inputs
    # 同一个特征域内，所有维度乘的是同一个权重
    assert torch.allclose(ratio, ratio[:, :, :1].expand_as(ratio), atol=1e-6)
    assert torch.all((ratio > 0) & (ratio < 1))


def test_SENet压缩用的是平均值():
    senet = SENet(num_fields=3, reduction_ratio=1)
    inputs = torch.randn(2, 3, 4)
    shifted = inputs - inputs.mean(dim=2, keepdim=True) + 1.0
    # 两个输入每个域均值相同，所以权重相同
    weights = senet(torch.ones_like(inputs))[:, :, :1]
    assert torch.allclose(senet(shifted), shifted * weights, atol=1e-6)
