# Cell
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim

from layers import Encoder, EncoderLayer, ConvLayer
from selfattention import FullAttention, AttentionLayer


class Encoder_model(nn.Module):
    def __init__(self, output_attention,
                 d_model, dropout,
                 factor, n_heads, d_ff, activation, e_layers,):
        super(Encoder_model, self).__init__()
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forward(self, x, enc_self_mask=None):

        enc_out, attns = self.encoder(x, attn_mask=enc_self_mask)

        return enc_out, attns


def main():
    d_model = 32
    n_heads = 2
    dropout = 0
    factor = 5
    activation = 'relu'
    output_attention = True
    d_ff = None
    e_layers = 2

    src = torch.rand(60, 120, d_model)
    model = Encoder(
        [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, factor, attention_dropout=dropout,
                                  output_attention=output_attention), d_model, n_heads),
                d_model,
                d_ff,
                dropout=dropout,
                activation=activation
            ) for l in range(e_layers)
        ],
        norm_layer=torch.nn.LayerNorm(d_model)
    )
    # model = Encoder_model(True, d_model, 0, 5, 2, None, "relu", 2)

    out, attn = model(src)
    print(attn)
    print(out.shape)


if __name__ == '__main__':
    main()
