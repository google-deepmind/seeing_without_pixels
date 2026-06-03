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

import copy
import math
import torch
import torch.nn as nn

# Input dimensionality per pose encoding (see utils/dataset_utils.transform_camera_pose
# and the --pose_encoding flag). For rel9d_grav (11) this is the pose part only; the
# 3-D gravity vector is projected by a separate head.
encode_pose_to_dim = {
    0: 7,  # abs7d
    2: 9,  # rel9d
    11: 9,  # rel9d_grav (pose part)
}


class CameraPoseSeqEncoder(nn.Module):
  """Camera-pose sequence -> encoded features.

  encode_pose controls the input representation (see utils/dataset_utils.py::
  transform_camera_pose). encode_pose=11 (the paper's default) splits the
  9-D relative pose from a 3-D gravity vector, projects each separately, and
  prepends the gravity token to the sequence before the transformer.
  """

  def __init__(
      self,
      d_model: int = 128,
      encode_pose: int = 0,
      output_dim: int = None,
      nhead: int = 4,
      num_layers: int = 4,
      dim_feedforward: int = 256,
      dropout: float = 0.1,
      max_len: int = 10000,
      pooling: str = "mean",
      use_scenario_label: bool = False,
      use_learnable_gravity: bool = False,
  ):
    super().__init__()
    self.encode_pose = encode_pose
    self.pooling = pooling.lower()
    assert self.pooling in {"mean", "last", "no"}
    self.gravity_dropout_prob = 0.5
    self.use_learnable_gravity = use_learnable_gravity
    if self.encode_pose == 11:
      self.learnable_gravity = nn.Parameter(torch.randn(1, 1, d_model))

    self.output_dim = output_dim if output_dim is not None else d_model

    self.in_dim = encode_pose_to_dim[self.encode_pose]
    self.in_proj = nn.Linear(self.in_dim, d_model)
    if self.encode_pose == 11:
      self.in_proj_gravity = nn.Linear(3, d_model)

    # sine-cosine positional table
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    self.register_buffer("pos_table", pe.unsqueeze(0))
    self.dropout = nn.Dropout(dropout)

    enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
    )
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    self.use_scenario_label = use_scenario_label
    if self.use_scenario_label:
      self.label_embed = nn.Embedding(8, d_model)

    self.norm = nn.LayerNorm(d_model)
    if self.output_dim != d_model:
      self.out_proj = nn.Linear(d_model, self.output_dim)
    else:
      self.out_proj = nn.Identity()

  def forward(
      self,
      x_padded,
      pad_mask,
      lengths,
      scenario_labels=None,
      return_middle_h=False,
  ):
    if self.encode_pose == 11:
      x, gravity_vecs = x_padded[:, :, :9], x_padded[:, 0:1, 9:]
      h1 = self.in_proj(x)
      h2 = self.in_proj_gravity(gravity_vecs)
      if self.training and not self.use_learnable_gravity:
        drop_mask = (
            torch.rand(x.size(0), 1, 1, device=x.device)
            > self.gravity_dropout_prob
        ).float()
        h2 = drop_mask * h2 + (1 - drop_mask) * self.learnable_gravity
      else:
        h2 = (
            self.learnable_gravity.expand(h1.size(0), 1, -1)
            if self.use_learnable_gravity
            else h2
        )
      h = torch.cat([h2, h1], dim=1)
      first_col = torch.zeros(
          (pad_mask.size(0), 1), dtype=torch.bool, device=pad_mask.device
      )
      pad_mask = torch.cat([first_col, pad_mask], dim=1)
    else:
      h = self.in_proj(x_padded)

    if self.use_scenario_label:
      label_token = self.label_embed(scenario_labels).unsqueeze(1)
      h = torch.cat([label_token, h], dim=1)
      pad_mask = torch.cat(
          [
              torch.zeros(
                  pad_mask.size(0), 1, dtype=torch.bool, device=pad_mask.device
              ),
              pad_mask,
          ],
          dim=1,
      )

    h = h + self.pos_table[:, : h.size(1)]
    h = self.dropout(h)

    h = self.encoder(h, src_key_padding_mask=pad_mask)
    if return_middle_h:
      middle_h = copy.deepcopy(h)

    if self.pooling == "mean":
      # NOTE: when a gravity token (encode_pose==11) or a scenario token is
      # prepended, this sum includes that token while the denominator is the
      # pose-frame count (`lengths`). This is the exact behavior the released
      # checkpoints were trained and evaluated with (e.g. the base-task Nymeria
      # zero-shot path), so it is kept as-is for reproducibility rather than
      # pooling only the real pose tokens. The contextualized longseq task uses
      # pooling='no' and pools the labeled window itself, so it is unaffected.
      h_sum = h.masked_fill(pad_mask.unsqueeze(-1), 0.0).sum(1)
      h = h_sum / lengths.unsqueeze(-1)
    elif self.pooling == "last":
      idx = (lengths - 1).unsqueeze(-1).expand(-1, h.size(-1))
      h = h.gather(1, idx.unsqueeze(1)).squeeze(1)
    else:  # 'no': keep (B,T,d) but strip the prepended tokens
      if self.use_scenario_label:
        h = h[:, 1:]
      if self.encode_pose == 11:
        h = h[:, 1:]

    h = self.norm(h)
    features = self.out_proj(h)
    if return_middle_h:
      return features, middle_h
    return features
