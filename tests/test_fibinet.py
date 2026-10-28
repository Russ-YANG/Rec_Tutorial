import torch

from 特征交叉.fibinet import FiBiNET


def test_FiBiNET输出点击率():
    model = FiBiNET(num_fields=4, embedding_dim=5, hidden_dims=[16, 8])
    model.eval()
    output = model(torch.randn(3, 4, 5))
    assert output.shape == (3,)
    assert torch.all((output > 0) & (output < 1))


def test_原始嵌入和SENet加权嵌入各做一次双线性交叉():
    model = FiBiNET(num_fields=4, embedding_dim=5, hidden_dims=[16, 8])
    assert model.bilinear_interaction is not model.se_bilinear_interaction
    # 两路交叉结果拼起来: 2 * C(4, 2) * 5
    assert model.dnn_layers[0].in_features == 2 * 6 * 5


def test_DNN输入是两路交叉结果的拼接():
    model = FiBiNET(num_fields=3, embedding_dim=4, hidden_dims=[8])
    model.eval()
    inputs = torch.randn(2, 3, 4)
    combined = torch.cat([
        model.bilinear_interaction(inputs),
        model.se_bilinear_interaction(model.senet(inputs)),
    ], dim=1)
    expected = torch.sigmoid(model.output_layer(model.dnn_layers(combined))).squeeze(1)
    assert torch.allclose(model(inputs), expected, atol=1e-6)
