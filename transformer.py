import math

import numpy as np
import torch
from torch import nn
from torch.nn import Transformer


class PositionalEncoding(nn.Module):
    def __init__(self, d_embedding, max_context_window):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_context_window, d_embedding)
        position = torch.arange(0, max_context_window, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embedding, 2).float() * -(math.log(10000.0) / d_embedding))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def get_causal_mask(for_size: int):
    return ((1 - torch.triu(torch.ones(for_size, for_size), diagonal=1)).bool() != True).requires_grad_(False).cuda()


def _init_weights(module):
    """ Initialize the weights """
    if isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
        n = module.in_features
        y = 1.0 / np.sqrt(n)
        module.weight.data.uniform_(-y, y)


class StdDETransformer(nn.Module):
    def __init__(self,
                 d_embedding: int = 1024,
                 vocab_size: int = 1024,
                 transformer_depth: int = 6,
                 max_context_window: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embedding = d_embedding
        self.max_context_window = max_context_window
        self.encoder_embedding = nn.Embedding(self.vocab_size, self.d_embedding, dtype=torch.float32)
        self.decoder_embedding = nn.Embedding(self.vocab_size, self.d_embedding, dtype=torch.float32)
        self.positional = PositionalEncoding(d_embedding, max_context_window=max_context_window)
        self.transformer = Transformer(
            d_model=self.d_embedding,
            batch_first=True,
            dtype=torch.float32,
            activation=nn.GELU(),
            dropout=dropout,
            num_encoder_layers=transformer_depth,
            num_decoder_layers=transformer_depth
        )

        self.classifier = nn.Linear(self.d_embedding, vocab_size, dtype=torch.float32)
        self.normalization_factor = torch.sqrt(torch.tensor(self.max_context_window)).requires_grad_(False)

        self.apply(_init_weights)

        self.training_causal_mask = get_causal_mask(for_size=self.max_context_window)

    def forward(self, encoder_input, decoder_input):
        enc_embedding = self.encoder_embedding(encoder_input)
        enc_input_embedding = self.positional(enc_embedding)

        dec_embedding = self.decoder_embedding(decoder_input)
        dec_input_embedding = self.positional(dec_embedding)

        if self.training:
            dec_out_embedding = self.transformer(src=enc_input_embedding,
                                                 src_key_padding_mask=(encoder_input == 0),
                                                 tgt=dec_input_embedding,
                                                 tgt_key_padding_mask=(decoder_input == 0),
                                                 tgt_mask=self.training_causal_mask,
                                                 tgt_is_causal=True)
        else:
            dec_out_embedding = self.transformer(src=enc_input_embedding,
                                                 src_key_padding_mask=(encoder_input == 0),
                                                 tgt=dec_input_embedding,
                                                 tgt_key_padding_mask=(decoder_input == 0))

        return self.classifier(dec_out_embedding)
