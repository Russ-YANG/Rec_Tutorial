"""
训练手写 Transformer 和 PyTorch 版 Transformer，对比损失、速度和生成效果。

运行: python -m transformer.train
"""
import math
import time

import torch
from torch.utils.data import DataLoader

from common.train import fit, get_device
from transformer.data import EnglishDataset, SimpleTokenizer
from transformer.model import Transformer, TransformerConfig
from transformer.torch_version import Transformer_Torch


def lm_batch_loss(model, batch):
    _, loss = model(batch['x'], targets=batch['y'])
    return loss


def generate_text(model, tokenizer, prompt, device, max_new_tokens=10, temperature=0.8, top_k=10):
    model.eval()
    start_tokens = tokenizer.encode(prompt)[:-1]  # 去掉 EOS，让模型接着写
    idx = torch.tensor([start_tokens], device=device)
    generated = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(generated[0].tolist())


@torch.no_grad()
def benchmark_performance(model, config, device, batch_size=32, n_runs=100):
    """返回 (平均推理时间, 每秒处理的样本数)"""
    model.eval()
    test_input = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device=device)

    for _ in range(10):  # 预热
        model(test_input)

    start_time = time.time()
    for _ in range(n_runs):
        model(test_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    avg_time = (time.time() - start_time) / n_runs
    return avg_time, batch_size / avg_time


def compare_models(n_epochs=20):
    device = get_device()
    tokenizer = SimpleTokenizer()
    config = TransformerConfig(
        block_size=32,
        vocab_size=tokenizer.vocab_size,
        n_layer=3,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=True
    )
    train_loader = DataLoader(EnglishDataset(tokenizer, config.block_size, num_samples=100),
                              batch_size=8, shuffle=True)
    val_loader = DataLoader(EnglishDataset(tokenizer, config.block_size, num_samples=30), batch_size=8)
    print(f"设备: {device}，词表大小: {tokenizer.vocab_size}")

    models = {
        "手写Transformer": Transformer(config),
        "PyTorch原生Transformer": Transformer_Torch(config),
    }
    results = {}
    for name, model in models.items():
        print(f"\n{'=' * 60}\n{name}，参数量: {model.get_num_params():,}\n{'=' * 60}")
        model.to(device)
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95))

        start_time = time.time()
        history = fit(model, train_loader, val_loader, lm_batch_loss, device,
                      n_epochs=n_epochs, optimizer=optimizer)
        training_time = time.time() - start_time
        inference_time, throughput = benchmark_performance(model, config, device)

        for prompt in ["the cat", "i like to", "the dog runs", "she reads a"]:
            print(f"'{prompt}' -> '{generate_text(model, tokenizer, prompt, device)}'")

        results[name] = {
            '参数量': f"{model.get_num_params():,}",
            '训练损失': f"{history['train_loss'][-1]:.4f}",
            '验证损失': f"{history['test_loss'][-1]:.4f}",
            '验证困惑度': f"{math.exp(history['test_loss'][-1]):.2f}",
            '训练时间': f"{training_time:.1f}s",
            '推理时间': f"{inference_time * 1000:.2f}ms",
        }

    print(f"\n{'指标':<12}" + "".join(f"{name:<28}" for name in results))
    for metric in next(iter(results.values())):
        print(f"{metric:<12}" + "".join(f"{row[metric]:<28}" for row in results.values()))
    return results


if __name__ == "__main__":
    compare_models()
