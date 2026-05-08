# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import clip
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class ContextAwareTextEncoder(nn.Module):

  def __init__(
      self,
      clip_model_name='ViT-B/32',
      num_layers=1,
      num_heads=8,
      dropout=0.1,
      max_len=100,
      device='cuda',
  ):
    super().__init__()
    self.device = device
    self.clip_model, _ = clip.load(clip_model_name, device=device)
    for param in self.clip_model.parameters():
      param.requires_grad = False

    self.clip_dim = self.clip_model.ln_final.normalized_shape[
        0
    ]  # 512 for ViT-B/32
    self.ln = nn.LayerNorm(self.clip_dim)

    # Positional encoding
    pe = torch.zeros(max_len, self.clip_dim)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, self.clip_dim, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / self.clip_dim)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    self.register_buffer('pos_table', pe.unsqueeze(0))  # (1, T, d)
    self.dropout = nn.Dropout(dropout)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=self.clip_dim,
        nhead=num_heads,
        dim_feedforward=2048,
        dropout=dropout,
        batch_first=True,
    )
    self.transformer = nn.TransformerEncoder(
        encoder_layer, num_layers=num_layers
    )
    self.to(device)

  def forward(self, batch):
    """batch: List of (select_idx, List[str]) where List[str] is of variable length"""
    batch_idx = [group[0] for group in batch]
    batch_sentences = [group[1] for group in batch]

    # Flatten all sentences
    flat_sentences = [s for group in batch_sentences for s in group]
    tokens = clip.tokenize(flat_sentences).to(self.device)

    # CLIP text encoding (frozen)
    with torch.no_grad():
      clip_feats = self.clip_model.encode_text(tokens)  # (sum(T_i), 512)

    clip_feats = clip_feats.to(
        self.ln.weight.dtype
    )  # match transformer precision
    group_lens = [len(group) for group in batch_sentences]
    split_feats = torch.split(clip_feats, group_lens)  # list of (T_i, 512)

    # Pad to max T
    padded = pad_sequence(split_feats, batch_first=True)  # (B, T_max, 512)
    max_len = padded.size(1)
    mask = (
        torch.arange(max_len, device=self.device)[None, :]
        >= torch.tensor(group_lens, device=self.device)[:, None]
    )

    # Add position
    pos_encoding = self.pos_table[:, :max_len].to(padded.device)
    x = self.dropout(padded + pos_encoding)

    # Transformer + LN
    x = self.transformer(x, src_key_padding_mask=mask)
    x = self.ln(x)  # (B, T_max, 512)

    # Select output at index
    batch_idx = torch.tensor(batch_idx, device=self.device)
    idx = batch_idx.view(-1, 1, 1).expand(-1, 1, x.size(2))
    selected = torch.gather(x, dim=1, index=idx).squeeze(1)  # (B, 512)
    return selected
