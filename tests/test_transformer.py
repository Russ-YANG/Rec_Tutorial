import math

import torch
import torch.nn.functional as F

from transformer import (
    Decoder, Encoder, LayerNorm, MultiHeadAttention, PositionalEncoding,
    SimpleTokenizer, Transformer, TransformerConfig, attention,
)


def small_config(**kwargs):
    values = dict(block_size=16, vocab_size=50, n_layer=2, n_head=2, n_embd=8, dropout=0.0)
    values.update(kwargs)
    return TransformerConfig(**values)


def test_缩放点积注意力公式():
    q, k, v = torch.randn(2, 3, 4), torch.randn(2, 5, 4), torch.randn(2, 5, 6)
    weights = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(4), dim=-1)
    assert torch.allclose(attention(q, k, v), weights @ v, atol=1e-6)


def test_多头注意力输出形状():
    layer = MultiHeadAttention(small_config())
    query, memory = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    assert layer(query, memory, memory).shape == (2, 3, 8)


def test_因果注意力看不到未来():
    layer = MultiHeadAttention(small_config(), is_causal=True)
    layer.eval()
    x = torch.randn(1, 6, 8)
    changed = x.clone()
    changed[:, 4:] = torch.randn(1, 2, 8)
    # 改动第4个位置之后的输入，前4个位置的输出不应该变
    assert torch.allclose(layer(x, x, x)[:, :4], layer(changed, changed, changed)[:, :4], atol=1e-6)


def test_非因果注意力能看到全部():
    layer = MultiHeadAttention(small_config(), is_causal=False)
    layer.eval()
    x = torch.randn(1, 6, 8)
    changed = x.clone()
    changed[:, 5:] = torch.randn(1, 1, 8)
    assert not torch.allclose(layer(x, x, x)[:, 0], layer(changed, changed, changed)[:, 0])


def test_LayerNorm():
    norm = LayerNorm(8, bias=True)
    output = norm(torch.randn(3, 8) * 5 + 2)
    assert torch.allclose(output.mean(dim=-1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(output.var(dim=-1, unbiased=False), torch.ones(3), atol=1e-3)


def test_正弦位置编码():
    encoding = PositionalEncoding(small_config())
    output = encoding(torch.zeros(1, 4, 8))
    for position in range(4):
        assert torch.isclose(output[0, position, 0], torch.tensor(math.sin(position)), atol=1e-6)
        assert torch.isclose(output[0, position, 1], torch.tensor(math.cos(position)), atol=1e-6)
        # 第 2i 维的频率是 1 / 10000^(2i / d)
        assert torch.isclose(output[0, position, 2], torch.tensor(math.sin(position / 10000 ** (2 / 8))), atol=1e-6)


def test_Encoder和Decoder形状():
    config = small_config()
    encoder, decoder = Encoder(config), Decoder(config)
    source, target = torch.randn(2, 5, 8), torch.randn(2, 3, 8)
    memory = encoder(source)
    assert memory.shape == (2, 5, 8)
    assert decoder(target, memory).shape == (2, 3, 8)


def test_语言模型前向():
    model = Transformer(small_config())
    idx = torch.randint(0, 50, (2, 6))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 6, 50)
    assert loss.dim() == 0
    # 随机初始化时交叉熵接近 log(vocab_size)
    assert abs(loss.item() - math.log(50)) < 1.0

    logits, loss = model(idx)
    assert logits.shape == (2, 1, 50) and loss is None


def test_语言模型是因果的():
    model = Transformer(small_config())
    model.eval()
    idx = torch.randint(0, 50, (1, 6))
    changed = idx.clone()
    changed[0, 5] = (idx[0, 5] + 1) % 50
    logits_1, _ = model(idx, targets=idx)
    logits_2, _ = model(changed, targets=changed)
    assert torch.allclose(logits_1[:, :5], logits_2[:, :5], atol=1e-5)


def test_生成指定数量的token():
    model = Transformer(small_config())
    model.eval()
    output = model.generate(torch.tensor([[1, 2]]), max_new_tokens=5, top_k=3)
    assert output.shape == (1, 7)
    assert output[0, :2].tolist() == [1, 2]


def test_能过拟合一个小序列():
    model = Transformer(small_config())
    optimizer = model.configure_optimizers(0.0, 1e-2, (0.9, 0.95), "cpu")
    idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    for _ in range(100):
        _, loss = model(idx[:, :-1], targets=idx[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5


def test_分词器编码解码():
    tokenizer = SimpleTokenizer()
    ids = tokenizer.encode("The cat is big.")
    assert ids[0] == tokenizer.vocab_to_id["<BOS>"]
    assert ids[-1] == tokenizer.vocab_to_id["<EOS>"]
    assert tokenizer.decode(ids) == "the cat is big"
    assert tokenizer.vocab_to_id["<UNK>"] in tokenizer.encode("zebra")
