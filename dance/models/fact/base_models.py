# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Basic building blocks for the multi-modal model."""

from einops.layers.torch import Rearrange
# from mint.core import base_model_util
# import tensorflow as tf

import numpy as np

import torch
import torch.nn as nn


class Norm(nn.Module):
    """Layer normalization."""

    def __init__(self, fn, hidden_dim: int):
      super().__init__()
      self.norm = nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-5, elementwise_affine=True)
      # self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
      self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class Residual(nn.Module):
    """Residual layer."""

    def __init__(self, fn):
      super().__init__()
      self.fn = fn

    def forward(self, x):
      return self.fn(x) + x


class MLP(nn.Module):
    """Feedforward layer."""

    def __init__(self, in_features, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            ])
        # self.net = tf.keras.Sequential([
            # tf.keras.layers.Dense(
            # hidden_dim, activation=base_model_util.get_activation("gelu")),
            # tf.keras.layers.Dense(out_dim)
        # ])

    def forward(self, x):
      return self.net(x)


class Attention(nn.Module):
  """Attention layer."""

  def __init__(self, in_features, dim, heads=8):
    super().__init__()
    self.heads = heads
    self.scale = dim**-0.5

    self.to_qkv = nn.Linear(in_features, dim * 3, bias=False)
    self.to_out = nn.Linear(dim, dim)

    # self.to_qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
    # self.to_out = tf.keras.layers.Dense(dim)

    self.rearrange_qkv = Rearrange(
        "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
    self.rearrange_out = Rearrange("b h n d -> b n (h d)")

  def forward(self, x):
    qkv = self.to_qkv(x)
    qkv = self.rearrange_qkv(qkv)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
    attn = nn.functional.softmax(dots, dim=-1)

    out = torch.einsum("bhij,bhjd->bhid", attn, v)
    out = self.rearrange_out(out)
    out = self.to_out(out)
    return out


class Transformer(nn.Module):
    """Transformer Encoder."""
    
    def __init__(self,
        in_features: int,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        ):
        super().__init__()
        blocks = []
        blocks.extend([
              Residual(Norm(Attention(in_features, hidden_size, heads=num_attention_heads), hidden_size)),
              Residual(Norm(MLP(hidden_size, intermediate_size, hidden_size), hidden_size))
          ])
        for _ in range(num_hidden_layers - 1):
          blocks.extend([
              Residual(Norm(Attention(hidden_size, hidden_size, heads=num_attention_heads), hidden_size)),
              Residual(Norm(MLP(hidden_size, intermediate_size, hidden_size), hidden_size))
          ])
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class PatchEmbedding(nn.Module):
    """Images patch embedding layer."""

    def __init__(self, in_features, config):
        super().__init__()
        self.patch_embed_layer = nn.Linear(in_features, config.hidden_size)
        self.rearrange = Rearrange(
            "b (h p1) (w p2) c -> b (h w) (p1 p2 c)",
            p1=config.patch_size,
            p2=config.patch_size,
            c=config.num_channel,
            )

    def forward(self, x):
      x = self.rearrange(x)
      return self.patch_embed_layer(x)


class LinearEmbedding(nn.Module):
    """Linear projection."""

    def __init__(self, in_features, dim):
        super().__init__()
        self.net = nn.Linear(in_features, dim)

    def forward(self, x):
        out = self.net(x)
        return out


class PositionEmbedding(nn.Module):
  """Position Embedding layer."""

  def __init__(self, seq_length, dim):
    super().__init__()
    w = torch.empty(seq_length, dim)
    nn.init.trunc_normal_(w, std=0.02, a=-0.02*2, b=0.02*2)
    self.pos_embedding = nn.Parameter(w)

    # pos_initializer = base_model_util.create_initializer(0.02)
    # self.pos_embedding = self.add_weight(
        # "position_embedding",
        # shape=[seq_length, dim],
        # initializer=pos_initializer,
        # dtype=tf.float32)

  def forward(self, x):
    """Call embedding layer."""
    return x + self.pos_embedding


class CrossModalLayer(nn.Module):
  """Cross-modal layer."""

  def __init__(self, config, out_dim):
    super().__init__()
    self.config = config

    model_config = self.config.transformer
    
    self.transformer_layer = Transformer(
        in_features=model_config.hidden_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        )

    self.cross_output_layer = nn.Linear(
        in_features=model_config.hidden_size,
        out_features=out_dim)
    nn.init.trunc_normal_(self.cross_output_layer.weight, std=0.02, a=-0.02*2, b=0.02*2)
    nn.init.trunc_normal_(self.cross_output_layer.bias, std=0.02, a=-0.02*2, b=0.02*2)


  def forward(self, modal_a_sequences, modal_b_sequences):
    """Get loss for the cross-modal tasks."""
    # _, _, modal_a_width = base_model_util.get_shape_list(modal_a_sequences)
    # _, _, modal_b_width = base_model_util.get_shape_list(modal_b_sequences)
    # if modal_a_width != modal_b_width:
    #   raise ValueError(
        #   "The modal_a hidden size (%d) should be the same with the modal_b "
        #   "hidden size (%d)" % (modal_a_width, modal_b_width))

    # [batch_size, modal_a_seq_len + modal_b_seq_len, width]
    merged_sequences = torch.cat([modal_a_sequences, modal_b_sequences], dim=1)

    # [batch_size, modal_a_seq_len + modal_b_seq_len, width]
    merged_sequences = self.transformer_layer(merged_sequences)
    logits = self.cross_output_layer(merged_sequences)

    return logits
