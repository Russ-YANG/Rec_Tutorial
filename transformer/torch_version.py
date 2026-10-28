"""
用 PyTorch 自带的 nn.TransformerEncoder 实现同样的语言模型，和手写版做对比。
embedding、输出层和生成都复用 LanguageModel，只替换中间的层。
"""
import torch.nn as nn

from transformer.model import LanguageModel, init_weights


class Transformer_Torch(LanguageModel):
    def __init__(self, config):
        super().__init__(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.dropout,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-LN，和手写版一致
        )
        # 和手写版一样，最后再接一个 LayerNorm
        self.encoder = nn.TransformerEncoder(encoder_layer, config.n_layer,
                                             norm=nn.LayerNorm(config.n_embd),
                                             enable_nested_tensor=False)
        self.apply(init_weights)

    def transform(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        return self.encoder(x, mask=mask, is_causal=True)
