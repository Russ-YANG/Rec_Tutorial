"""
从零实现 Transformer，并用它做一个预测下一个 token 的语言模型。

1. scaled_dot_product_attention  softmax(Q K^T / sqrt(d)) V，可选掩码
2. causal_mask                   下三角掩码，位置 t 只能看到 <= t 的位置
3. split_heads / merge_heads     多头注意力: 把向量拆成多个头分别算，再拼回来
4. MultiHeadAttention
5. MLP / LayerNorm
6. EncoderLayer / DecoderLayer   Pre-LN 残差块: x + f(LN(x))
7. sinusoidal_encoding           正弦位置编码
8. LanguageModel                 embedding + 若干层 + 输出层 + 生成，手写版和 PyTorch 版共用
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class TransformerConfig:
    def __init__(self, block_size=256, vocab_size=1000, n_layer=6,
                 n_head=8, n_embd=256, dropout=0.1, bias=True):
        self.block_size = block_size  # 最大序列长度
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """
    q: (..., query_len, d)  k: (..., key_len, d)  v: (..., key_len, d_v)
    mask: (query_len, key_len)，为 0 的位置不能被看到
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))  # k 要转置（交换最后两个维度）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v


def example_attention():
    # query: 水果
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    keys = torch.tensor([[1.0, 0.5, 0.0, 0.0],   # apple  - 强匹配
                         [0.9, 0.4, 0.0, 0.0],   # banana - 强匹配
                         [0.0, 0.0, 0.8, 1.0],   # cat    - 不匹配
                         [0.0, 0.0, 0.9, 0.8]])  # dog    - 不匹配
    # values: 实际的语义内容
    values = torch.tensor([[1.0, 0.0, 0.5, 0.0],   # apple  - 红色，圆形
                           [0.0, 1.0, 0.0, 0.8],   # banana - 黄色，长形
                           [0.5, 0.5, 0.0, 0.0],   # cat    - 灰色，动物
                           [0.3, 0.2, 0.0, 0.0]])  # dog    - 棕色，动物
    # 输出更接近苹果和香蕉的属性
    print(scaled_dot_product_attention(query, keys, values))


def causal_mask(query_length, key_length, device=None):
    """下三角矩阵，mask[i, j] = 1 表示位置 i 能看到位置 j"""
    return torch.tril(torch.ones(query_length, key_length, device=device))


def split_heads(x, n_head):
    """(batch, seq_len, n_embd) -> (batch, n_head, seq_len, head_dim)"""
    batch_size, seq_len, n_embd = x.shape
    return x.view(batch_size, seq_len, n_head, n_embd // n_head).transpose(1, 2)


def merge_heads(x):
    """(batch, n_head, seq_len, head_dim) -> (batch, seq_len, n_embd)"""
    batch_size, n_head, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, n_head * head_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, config, is_causal=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.is_causal = is_causal

    def forward(self, query, key, value):
        q = split_heads(self.q_proj(query), self.n_head)
        k = split_heads(self.k_proj(key), self.n_head)
        v = split_heads(self.v_proj(value), self.n_head)

        mask = causal_mask(q.size(2), k.size(2), query.device) if self.is_causal else None
        y = scaled_dot_product_attention(q, k, v, mask, self.attn_dropout)
        return self.resid_dropout(self.c_proj(merge_heads(y)))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.relu(self.c_fc(x))))


class LayerNorm(nn.Module):
    """可以关掉 bias 的 LayerNorm"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class EncoderLayer(nn.Module):
    def __init__(self, config, is_causal=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config, is_causal=is_causal)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        normalized_x = self.ln_1(x)
        # Self Attention: Q、K、V 都是 x
        x = x + self.attn(normalized_x, normalized_x, normalized_x)
        x = x + self.mlp(self.ln_2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config, is_causal=False):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(config, is_causal) for _ in range(config.n_layer)])
        self.norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.m_attn = MultiHeadAttention(config, is_causal=True)  # 只能看到前面
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config, is_causal=False)   # 和 encoder 的输出做 attention
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, enc_out):
        normalized_x = self.ln_1(x)
        x = x + self.m_attn(normalized_x, normalized_x, normalized_x)
        # Cross Attention: Q 来自 decoder，K、V 来自 encoder
        x = x + self.attn(self.ln_2(x), enc_out, enc_out)
        x = x + self.mlp(self.ln_3(x))
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])
        self.norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)


def sinusoidal_encoding(block_size, n_embd):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i / d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d))
    """
    position = torch.arange(0, block_size).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, n_embd, 2) * -(math.log(10000.0) / n_embd))
    encoding = torch.zeros(block_size, n_embd)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        # 固定的编码，不参与训练
        self.register_buffer("pe", sinusoidal_encoding(config.block_size, config.n_embd).unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def lm_loss(logits, targets):
    """每个位置预测下一个 token 的交叉熵，target 为 -1 的位置（padding）不算"""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


def sample_next_token(logits, temperature=1.0, top_k=None):
    """logits: (batch, vocab_size)。只在概率最高的 top_k 个 token 里按概率采样"""
    logits = logits / temperature
    if top_k is not None:
        threshold, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < threshold[:, [-1]], float('-inf'))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class LanguageModel(nn.Module):
    """
    token embedding + 位置编码 -> transform -> 输出层
    子类只需要实现 transform(x)，也就是中间的若干层
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = PositionalEncoding(config)
        self.drop = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def transform(self, x):
        raise NotImplementedError

    def forward(self, idx, targets=None):
        t = idx.size(1)
        assert t <= self.config.block_size, f"序列长度 {t} 超过最大长度 {self.config.block_size}"

        x = self.drop(self.wpe(self.wte(idx)))
        x = self.transform(x)

        if targets is not None:
            logits = self.lm_head(x)
            return logits, lm_loss(logits, targets)
        # 生成时只需要最后一个位置
        return self.lm_head(x[:, [-1], :]), None

    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wte.weight.numel()
        return n_params

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        return torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]  # 超过最大长度就截断
            logits, _ = self(idx_cond)
            idx_next = sample_next_token(logits[:, -1, :], temperature, top_k)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class Transformer(LanguageModel):
    """手写的 decoder-only 语言模型: 因果 self-attention 的 Encoder 层堆叠"""

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(config, is_causal=True)
        self.apply(init_weights)

    def transform(self, x):
        return self.encoder(x)


if __name__ == "__main__":
    example_attention()
