import pytest
import torch

from 特征交叉.bilinear_interaction import BilinearInteraction


@pytest.mark.parametrize("bilinear_type, matrix_count", [
    ("field_all", 1),
    ("field_each", 4),
    ("field_interaction", 6),
])
def test_三种双线性的参数矩阵数量(bilinear_type, matrix_count):
    layer = BilinearInteraction(embedding_dim=5, num_fields=4, bilinear_type=bilinear_type)
    assert layer.bilinear_matrices.shape == (matrix_count, 5, 5)


def test_双线性输出维度是特征对数乘以嵌入维度():
    layer = BilinearInteraction(embedding_dim=5, num_fields=4)
    assert layer(torch.randn(2, 4, 5)).shape == (2, 6 * 5)


def test_矩阵为单位阵时退化成哈达玛积():
    layer = BilinearInteraction(embedding_dim=3, num_fields=3, bilinear_type="field_all")
    with torch.no_grad():
        layer.bilinear_matrices.copy_(torch.eye(3).unsqueeze(0))
    inputs = torch.randn(2, 3, 3)
    expected = torch.cat([
        inputs[:, 0] * inputs[:, 1],
        inputs[:, 0] * inputs[:, 2],
        inputs[:, 1] * inputs[:, 2],
    ], dim=1)
    assert torch.allclose(layer(inputs), expected, atol=1e-6)
