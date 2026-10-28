import pytest
import torch

from 特征交叉.bilinear_interaction import BilinearInteraction, bilinear, matrix_count, matrix_index


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


def test_双线性公式():
    # (v_i W) ⊙ v_j
    vector_i = torch.tensor([[1.0, 2.0]])
    vector_j = torch.tensor([[3.0, -1.0]])
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert torch.allclose(bilinear(vector_i, vector_j, matrix), torch.tensor([[6.0, -1.0]]))


@pytest.mark.parametrize("bilinear_type, count", [("field_all", 1), ("field_each", 5), ("field_interaction", 10)])
def test_矩阵数量(bilinear_type, count):
    assert matrix_count(bilinear_type, 5) == count


def test_特征对使用哪个矩阵():
    # 特征对 (2, 3) 是第 5 个特征对
    assert matrix_index("field_all", 2, 5) == 0
    assert matrix_index("field_each", 2, 5) == 2
    assert matrix_index("field_interaction", 2, 5) == 5
