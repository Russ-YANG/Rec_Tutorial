import torch

from transformer.model import Transformer, TransformerConfig
from transformer.torch_version import Transformer_Torch


def small_config():
    return TransformerConfig(block_size=16, vocab_size=50, n_layer=2, n_head=2, n_embd=8, dropout=0.0)


def test_参数量和手写版一致():
    assert Transformer_Torch(small_config()).get_num_params() == Transformer(small_config()).get_num_params()


def test_输出形状():
    model = Transformer_Torch(small_config())
    idx = torch.randint(0, 50, (2, 6))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 6, 50)
    assert loss.dim() == 0


def test_也是因果的():
    model = Transformer_Torch(small_config())
    model.eval()
    idx = torch.randint(0, 50, (1, 6))
    changed = idx.clone()
    changed[0, 5] = (idx[0, 5] + 1) % 50
    logits_1, _ = model(idx, targets=idx)
    logits_2, _ = model(changed, targets=changed)
    assert torch.allclose(logits_1[:, :5], logits_2[:, :5], atol=1e-5)
