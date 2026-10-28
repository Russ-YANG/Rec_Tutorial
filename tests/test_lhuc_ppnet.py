import torch
import torch.nn.functional as F

from 特征交叉.lhuc_ppnet import LHUC_PPNet


def test_输出点击率():
    model = LHUC_PPNet(user_dim=4, item_dim=6, hidden_dims=[8, 4])
    model.eval()
    output = model(torch.randn(3, 4), torch.randn(3, 6))
    assert output.shape == (3,)
    assert torch.all((output > 0) & (output < 1))


def test_门控只看用户特征():
    model = LHUC_PPNet(user_dim=4, item_dim=6, hidden_dims=[8, 4])
    for gate, hidden_dim in zip(model.gate_layers, [8, 4]):
        assert gate.in_features == 4
        assert gate.out_features == hidden_dim


def test_门控为0时缩放系数是1():
    # gate = 2 * sigmoid(0) = 1，此时等价于普通 MLP
    model = LHUC_PPNet(user_dim=4, item_dim=6, hidden_dims=[8, 4])
    model.eval()
    with torch.no_grad():
        for gate in model.gate_layers:
            gate.weight.zero_()
            gate.bias.zero_()
    user, item = torch.randn(3, 4), torch.randn(3, 6)
    hidden = torch.cat([user, item], dim=1)
    for layer in model.main_layers:
        hidden = F.relu(layer(hidden))
    expected = model.output_layer(hidden).squeeze(1)
    assert torch.allclose(model(user, item), expected, atol=1e-6)


def test_门控范围在0到2之间():
    model = LHUC_PPNet(user_dim=4, item_dim=6, hidden_dims=[8])
    model.eval()
    with torch.no_grad():
        model.gate_layers[0].weight.zero_()
        model.gate_layers[0].bias.fill_(100.0)
    user, item = torch.randn(3, 4), torch.randn(3, 6)
    hidden = F.relu(model.main_layers[0](torch.cat([user, item], dim=1)))
    expected = model.output_layer(hidden * 2.0).squeeze(1)
    assert torch.allclose(model(user, item), expected, atol=1e-5)
