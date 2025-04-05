from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import normal

import sys
sys.path.append('../')

import os
import copy

# =============================================================================
# Token processing blocks
# =============================================================================

# =============================================================================
# Processing blocks
# X-attention input
#   Q/z_input         -> (#latent_embs, batch_size, embed_dim)
#   K/V/x             -> (#events, batch_size, embed_dim)
#   key_padding_mask  -> (batch_size, #event)
# output -> (#latent_embs, batch_size, embed_dim)
# =============================================================================
class AttentionBlock(nn.Module):  # PerceiverAttentionBlock
    def __init__(self, opt_dim, heads, dropout, att_dropout, **args):
        super(AttentionBlock, self).__init__()

        self.layer_norm_x = nn.LayerNorm([opt_dim])
        self.layer_norm_1 = nn.LayerNorm([opt_dim])
        self.layer_norm_att = nn.LayerNorm([opt_dim])

        self.attention = nn.MultiheadAttention(
            opt_dim,  # embed_dim
            heads,  # num_heads
            dropout=att_dropout,
            bias=True,
            add_bias_kv=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(opt_dim, opt_dim)
        self.layer_norm_2 = nn.LayerNorm([opt_dim])
        self.linear2 = nn.Linear(opt_dim, opt_dim)
        self.linear3 = nn.Linear(opt_dim, opt_dim)

    def forward(self, x, z_input, mask=None, q_mask=None, **args):
        x = self.layer_norm_x(x)
        z = self.layer_norm_1(z_input)

        z_att, _ = self.attention(z, x, x, key_padding_mask=mask, attn_mask=q_mask)  # Q, K, V

        z_att = z_att + z_input
        z = self.layer_norm_att(z_att)

        z = self.dropout(z)
        z = self.linear1(z)
        z = torch.nn.GELU()(z)

        z = self.layer_norm_2(z)
        z = self.linear2(z)
        z = torch.nn.GELU()(z)
        z = self.dropout(z)
        z = self.linear3(z)

        return z + z_att


class TransformerBlock(nn.Module):
    def __init__(self, opt_dim, latent_blocks, dropout, att_dropout, att_heads, cross_heads, **args):
        super(TransformerBlock, self).__init__()

        self.cross_attention = AttentionBlock(opt_dim=opt_dim, heads=cross_heads, dropout=dropout,
                                              att_dropout=att_dropout)
        self.latent_attentions = nn.ModuleList([
            AttentionBlock(opt_dim=opt_dim, heads=att_heads, dropout=dropout, att_dropout=att_dropout) for _ in
            range(latent_blocks)
        ])

    def forward(self, x_input, z, mask=None, q_mask=None, **args):
        z = self.cross_attention(x_input, z, mask=mask, q_mask=q_mask)
        for latent_attention in self.latent_attentions:
            z = latent_attention(z, z, q_mask=q_mask)
        return z