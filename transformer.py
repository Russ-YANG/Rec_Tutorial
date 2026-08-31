import math
from torch.nn import functional as F
import torch.nn as nn
import torch
import time


class TransformerConfig:
    def __init__(self, block_size=256, vocab_size=1000, n_layer=6,
                 n_head=8, n_embd=256, dropout=0.1, bias=True):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias




def attention(q, k, v):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # k要转置（交换最后两个维度）
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def example_attention():
    # query: fruit
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # (1, 4)

    # keys
    keys = torch.tensor([[1.0, 0.5, 0.0, 0.0],  # apple - 强匹配
                         [0.9, 0.4, 0.0, 0.0],  # banana - 强匹配
                         [0.0, 0.0, 0.8, 1.0],  # cat - 不匹配
                         [0.0, 0.0, 0.9, 0.8]])  # dog - 不匹配

    # values: 实际的语义内容
    values = torch.tensor([[1.0, 0.0, 0.5, 0.0],  # apple - 红色，圆形
                           [0.0, 1.0, 0.0, 0.8],  # banana - 黄色，长形
                           [0.5, 0.5, 0.0, 0.0],  # cat - 灰色，动物
                           [0.3, 0.2, 0.0, 0.0]])  # dog - 棕色，动物

    output = attention(query, keys, values)
    print(output) #输出的就是水果相关的属性信息 应该更接近苹果和香蕉的属性


class MultiHeadAttention(nn.Module):
    def __init__(self, config, is_causal=False):
        # config包含模型的配置参数，这里可以先不管
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attns = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd, bias=config.bias) for _ in range(3)]) # 原文备注：矩阵内积再拼接等价于拼接后再内积
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.is_causal = is_causal

        # 为因果注意力机制注册下三角mask
        if is_causal:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                               .view(1, 1, config.block_size, config.block_size))


    def forward(self, query, key, value):

        batch_size = query.size(0)
        query_length = query.size(1)
        key_length = key.size(1)
        embedding_dim = query.size(2)

        q = self.c_attns[0](query)
        k = self.c_attns[1](key)
        v = self.c_attns[2](value)

        head_dim = embedding_dim // self.n_head
        q = q.view(batch_size, query_length, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, key_length, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, key_length, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.is_causal:
            mask = self.bias[:, :, :query_length, :key_length]
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, query_length, embedding_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y



class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias) # 第一个线性层
        self.relu = nn.ReLU()  # 加一个relu
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias) # 第二个线性层
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5) # torch自带的layer_norm函数 我们加上了bias


class EncoderLayer(nn.Module):
    def __init__(self, config, is_causal=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config, is_causal=is_causal)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        normalized_x = self.ln_1(x)
        # Encoder 使用 Self Attention，所以 Q、K、V 都是 x
        x = x + self.attn(normalized_x, normalized_x, normalized_x)
        x = x + self.mlp(self.ln_2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config, is_causal=False):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(config.n_layer):
            self.layers.append(EncoderLayer(config, is_causal))
        self.norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)



class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.m_attn = MultiHeadAttention(config, is_causal=True) # decoder这里的att是需要加掩码的（只能看到前面）
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        self.attn = MultiHeadAttention(config, is_causal=False) # 这里不加掩码 因为是和encoder的输出做attention
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, enc_out):
        normalized_x = self.ln_1(x)
        x = x + self.m_attn(normalized_x, normalized_x, normalized_x)

        normalized_x = self.ln_2(x)
        x = x + self.attn(normalized_x, enc_out, enc_out)
        x = x + self.mlp(self.ln_3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)]) # 这里要用多层DecoderLayer
        self.norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)

class PositionalEncoding(nn.Module):

    def __init__(self, config):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(config.block_size, config.n_embd)
        position = torch.arange(0, config.block_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.n_embd, 2) * -(math.log(10000.0) / config.n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x




class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 必须输入词表大小和 block size
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = PositionalEncoding(config),
            drop = nn.Dropout(config.dropout),
            blocks = Encoder(config, is_causal=True),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)


    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.config.block_size}"


        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(tok_emb)
        x = self.transformer.drop(pos_emb)
        x = self.transformer.blocks(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )
        return optimizer

    @torch.no_grad() # 不需要计算梯度
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:] #做截断
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# 直接用torch封装好的Transformer模块是下面这样写
class Transformer_Torch(nn.Module):
    def __init__(self, config):
        super(Transformer_Torch, self).__init__()
        self.config = config

        # Embedding层
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = PositionalEncoding(config)
        self.drop = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4*config.n_embd,
            dropout=config.dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layer)

        # 输出层
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wte.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"序列长度 {t} 超过最大长度 {self.config.block_size}"

        # Embedding
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(tok_emb)
        x = self.drop(pos_emb)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(t).to(device)
        output = self.transformer(x, mask=causal_mask)

        if targets is not None:
            logits = self.lm_head(output)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(output[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



class SimpleTokenizer:
    def __init__(self):
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'

        basic_vocab = [
            self.pad_token, self.unk_token, self.bos_token, self.eos_token,
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those',
            'is', 'am', 'are', 'was', 'were', 'be', 'being', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'may', 'might',
            'cat', 'dog', 'bird', 'fish', 'house', 'car', 'tree', 'book', 'water', 'food',
            'man', 'woman', 'child', 'boy', 'girl', 'people', 'person', 'family', 'friend',
            'good', 'bad', 'big', 'small', 'hot', 'cold', 'new', 'old', 'red', 'blue', 'green',
            'go', 'come', 'see', 'look', 'hear', 'say', 'tell', 'know', 'think', 'want', 'like',
            'eat', 'drink', 'sleep', 'work', 'play', 'run', 'walk', 'sit', 'stand', 'live',
            'today', 'yesterday', 'tomorrow', 'now', 'here', 'there', 'yes', 'no', 'please', 'thank'
        ]

        # 创建词汇表映射
        self.vocab_to_id = {word: idx for idx, word in enumerate(basic_vocab)}
        self.id_to_vocab = {idx: word for idx, word in enumerate(basic_vocab)}
        self.vocab_size = len(basic_vocab)

    def encode(self, text):
        """将文本转换为token id列表"""
        words = text.lower().split()
        tokens = [self.bos_token]  # 添加开始token
        for word in words:
            # 简单处理标点符号
            word = word.strip('.,!?;:"()[]')
            if word in self.vocab_to_id:
                tokens.append(word)
            else:
                tokens.append(self.unk_token)
        tokens.append(self.eos_token)  # 添加结束token

        return [self.vocab_to_id[token] for token in tokens]

    def decode(self, token_ids):
        """将token id列表转换为文本"""
        tokens = []
        for token_id in token_ids:
            if token_id < len(self.id_to_vocab):
                token = self.id_to_vocab[token_id]
                if token not in [self.pad_token, self.bos_token, self.eos_token]:
                    tokens.append(token)
        return ' '.join(tokens)


class EnglishDataset:
    def __init__(self, tokenizer, seq_length, num_samples=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length


        sentences = [
            "the cat is sleeping on the bed",
            "i like to eat good food",
            "the dog runs in the park",
            "she reads a book every day",
            "we go to school in the morning",
            "the bird sings in the tree",
            "he drives a red car",
            "they play with their friends",
            "the house is big and beautiful",
            "water is cold and fresh",
            "the boy likes to play games",
            "she has a small cat",
            "we see the blue sky",
            "the man works in the office",
            "children like to eat ice cream",
            "the woman cooks good food",
            "birds fly in the sky",
            "fish swim in the water",
            "the car is new and fast",
            "people walk in the street",
            "the girl reads a good book",
            "cats like to sleep here",
            "dogs run and play outside",
            "the family lives in a house",
            "friends come to visit today",
            "the tree is green and tall",
            "i want to go home now",
            "you can see the red bird",
            "we will eat dinner soon",
            "they should go to bed",
        ]


        if num_samples:
            sentences = (sentences * (num_samples // len(sentences) + 1))[:num_samples]

        # 编码所有句子
        self.encoded_data = []
        for sentence in sentences:
            encoded = tokenizer.encode(sentence)
            if len(encoded) <= seq_length + 1:
                # 填充到指定长度
                padded = encoded + [tokenizer.vocab_to_id[tokenizer.pad_token]] * (seq_length + 1 - len(encoded))
                self.encoded_data.append(padded[:seq_length + 1])

        print(f"数据集创建完成: {len(self.encoded_data)} 个样本")
        if len(self.encoded_data) > 0:
            print(f"样本示例: {tokenizer.decode(self.encoded_data[0][:-1])}")

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        seq = self.encoded_data[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)  # 输入序列
        target = seq[1:]
        pad_id = self.tokenizer.vocab_to_id[self.tokenizer.pad_token]
        for i in range(len(target)):
            if target[i] == pad_id:
                target[i] = -1
        y = torch.tensor(target, dtype=torch.long)   # 目标序列（右移1位）
        return x, y

def create_dataloader(dataset, batch_size, shuffle=True):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_step(model, batch, optimizer, device):
    """单个训练步骤"""
    model.train()
    x, y = batch
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    logits, loss = model(x, targets=y)
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            logits, loss = model(x, targets=y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10):
    print(f"设备: {device}")

    train_losses = []
    val_losses = []
    val_perplexities = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # 训练
        model.train()
        epoch_train_loss = 0
        num_batches = 0

        for batch in train_loader:
            loss = train_step(model, batch, optimizer, device)
            epoch_train_loss += loss
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # 验证
        val_loss, val_perplexity = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_perplexities.append(val_perplexity)

        # 打印进度
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Perplexity: {val_perplexity:.2f} | "
              f"Time: {elapsed:.1f}s")

    return train_losses, val_losses, val_perplexities


def test_generation(model, tokenizer, device):
    """测试英语文本生成能力"""
    model.eval()

    # 测试不同的起始文本
    test_prompts = [
        "the cat",
        "i like to",
        "the dog runs",
        "she reads a"
    ]

    print(f"\n英语文本生成测试:")
    print("-" * 50)

    for prompt in test_prompts:
        # 编码起始文本
        start_tokens = tokenizer.encode(prompt)
        start_seq = torch.tensor([start_tokens[:-1]], device=device)  # 去掉EOS token

        print(f"起始文本: '{prompt}'")
        print(f"起始tokens: {start_tokens}")

        # 生成文本
        with torch.no_grad():
            generated = model.generate(start_seq, max_new_tokens=10, temperature=0.8, top_k=10)

        # 解码生成的文本
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"生成文本: '{generated_text}'")
        print("-" * 30)


def benchmark_performance(model, config, device):
    """性能基准测试"""
    model.eval()

    # 测试推理速度
    batch_size = 32
    seq_length = config.block_size
    test_input = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)

    # 实际测试
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(100):
            _ = model(test_input)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()

    avg_time = (end_time - start_time) / 100
    throughput = batch_size / avg_time

    print(f"\n性能基准测试:")
    print(f"批次大小: {batch_size}")
    print(f"序列长度: {seq_length}")
    print(f"平均推理时间: {avg_time*1000:.2f}ms")
    print(f"吞吐量: {throughput:.1f} samples/sec")

    return avg_time, throughput


def compare_models():
    """对比手写Transformer和PyTorch原生Transformer的性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建tokenizer
    tokenizer = SimpleTokenizer()
    print(f"词汇表大小: {tokenizer.vocab_size}")

    # 创建配置
    config = TransformerConfig(
        block_size=32,
        vocab_size=tokenizer.vocab_size,
        n_layer=3,  # 减少层数以加快训练
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=True
    )

    # 创建数据集
    train_dataset = EnglishDataset(tokenizer, config.block_size, num_samples=100)
    val_dataset = EnglishDataset(tokenizer, config.block_size, num_samples=30)

    train_loader = create_dataloader(train_dataset, batch_size=8, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=8, shuffle=False)

    models = {
        "手写Transformer": Transformer(config),
        "PyTorch原生Transformer": Transformer_Torch(config)
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"测试模型: {model_name}")
        print(f"{'='*60}")

        model = model.to(device)
        print(f"模型参数量: {model.get_num_params():,}")

        # 创建优化器
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            device_type=device.type
        )

        # 训练模型
        start_time = time.time()
        train_losses, val_losses, val_perplexities = train_model(
            model, train_loader, val_loader, optimizer, device, num_epochs=20
        )
        training_time = time.time() - start_time

        # 性能测试
        avg_inference_time, throughput = benchmark_performance(model, config, device)

        # 文本生成测试
        print(f"\n{model_name} - 文本生成测试:")
        test_generation(model, tokenizer, device)

        # 记录结果
        results[model_name] = {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_perplexity': val_perplexities[-1],
            'training_time': training_time,
            'inference_time': avg_inference_time,
            'throughput': throughput,
            'num_params': model.get_num_params()
        }

    # 打印对比结果
    print(f"\n{'='*80}")
    print("模型性能对比总结")
    print(f"{'='*80}")
    print(f"{'指标':<20} {'手写Transformer':<25} {'PyTorch原生':<25} {'差异':<15}")
    print(f"{'-'*80}")

    metrics = [
        ('参数量', 'num_params', '{:,}'),
        ('最终训练损失', 'final_train_loss', '{:.4f}'),
        ('最终验证损失', 'final_val_loss', '{:.4f}'),
    ]

    custom_results = results["手写Transformer"]
    torch_results = results["PyTorch原生Transformer"]

    for metric_name, key, fmt in metrics:
        custom_val = custom_results[key]
        torch_val = torch_results[key]

        if key == 'inference_time':
            custom_val *= 1000  # 转换为毫秒
            torch_val *= 1000

        custom_str = fmt.format(custom_val)
        torch_str = fmt.format(torch_val)

        # 计算差异
        if key in ['final_train_loss', 'final_val_loss', 'final_perplexity', 'training_time', 'inference_time']:
            diff = ((custom_val - torch_val) / torch_val * 100) if torch_val != 0 else 0
            diff_str = f"{diff:+.1f}%"
        elif key == 'throughput':
            diff = ((custom_val - torch_val) / torch_val * 100) if torch_val != 0 else 0
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = "N/A"

        print(f"{metric_name:<20} {custom_str:<25} {torch_str:<25} {diff_str:<15}")

    return results


def main():
    """主函数"""
    compare_models()


if __name__ == "__main__":
    main()
