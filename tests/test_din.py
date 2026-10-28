import torch

from 序列.din import DIN, AttentionLayer, attention_features, masked_softmax, sequence_mask


def test_注意力特征是4部分拼接():
    query = torch.tensor([[1.0, 2.0]])
    keys = torch.tensor([[[3.0, 1.0], [0.0, 2.0]]])
    features = attention_features(query, keys)
    assert features.shape == (1, 2, 8)
    assert features[0, 0].tolist() == [1.0, 2.0, 3.0, 1.0, 3.0, 2.0, -2.0, 1.0]


def test_序列掩码():
    mask = sequence_mask(torch.tensor([2, 0, 3]), max_len=3)
    assert mask.tolist() == [[True, True, False], [False, False, False], [True, True, True]]


def test_带掩码的softmax():
    scores = torch.tensor([[1.0, 2.0, 100.0]])
    mask = torch.tensor([[True, True, False]])
    weights = masked_softmax(scores, mask)
    assert torch.allclose(weights, torch.cat([torch.softmax(scores[:, :2], dim=1), torch.zeros(1, 1)], dim=1))


def test_全是padding时权重全为0():
    weights = masked_softmax(torch.randn(1, 3), torch.zeros(1, 3, dtype=torch.bool))
    assert torch.allclose(weights, torch.zeros(1, 3))


def make_attention():
    layer = AttentionLayer(embedding_dim=4, attention_hidden_units=[8, 4])
    layer.eval()
    return layer


def test_注意力权重和为1():
    layer = make_attention()
    query, keys = torch.randn(2, 4), torch.randn(2, 5, 4)
    _, weights = layer(query, keys, torch.tensor([5, 3]))
    assert weights.shape == (2, 5)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)


def test_padding位置的权重为0():
    layer = make_attention()
    query, keys = torch.randn(2, 4), torch.randn(2, 5, 4)
    _, weights = layer(query, keys, torch.tensor([5, 3]))
    assert torch.allclose(weights[1, 3:], torch.zeros(2), atol=1e-6)


def test_兴趣向量是历史行为的加权和():
    layer = make_attention()
    query, keys = torch.randn(2, 4), torch.randn(2, 5, 4)
    interest, weights = layer(query, keys, torch.tensor([5, 3]))
    expected = (keys * weights.unsqueeze(-1)).sum(dim=1)
    assert torch.allclose(interest, expected, atol=1e-6)


def test_padding的内容不影响结果():
    layer = make_attention()
    query, keys = torch.randn(1, 4), torch.randn(1, 5, 4)
    lengths = torch.tensor([2])
    changed = keys.clone()
    changed[:, 2:] = torch.randn(1, 3, 4) * 100
    interest_1, _ = layer(query, keys, lengths)
    interest_2, _ = layer(query, changed, lengths)
    assert torch.allclose(interest_1, interest_2, atol=1e-5)


def test_注意力网络输入包含4种特征():
    layer = AttentionLayer(embedding_dim=4, attention_hidden_units=[8])
    assert layer.attention_net[0][0].in_features == 4 * 4


def test_DIN输出():
    model = DIN(num_users=3, num_items=10, embedding_dim=4, mlp_hidden_units=[8], attention_hidden_units=[8])
    model.eval()
    history = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    prediction, weights = model(torch.tensor([0, 1]), torch.tensor([6, 7]), history, torch.tensor([3, 2]))
    assert prediction.shape == (2,)
    assert torch.all((prediction > 0) & (prediction < 1))
    assert weights.shape == (2, 4)
