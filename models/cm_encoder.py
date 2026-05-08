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

encode_pose_to_dim = {
    0: 7,
    1: 7,
    2: 9,
    3: 12,
    4: 7,
    5: 14,
    6: 7,
    7: 3,
    8: 9,
    9: 13,
    10: 12,
    11: 9,
}


class CameraPoseSeqEncoder(nn.Module):
  """Camera-pose sequence → encoded features   (all in one class)

  This encoder shares the same architecture as CameraPoseSeqClassifier's encoder
  for weight transfer after pre-training.

  Args
  ----
  d_model     : int
      Transformer feature width (input → encoder).
  output_dim  : int or None
      Dimension of output features. If None, will be same as d_model.
  nhead       : int
      Number of attention heads.
  num_layers  : int
      Transformer-encoder depth.
  dim_feedforward : int
      Hidden size of the MLP inside each encoder block.
  dropout     : float
      Drop-out applied after positional encoding and inside blocks.
  max_len     : int
      Maximum sequence length you expect (for the positional table).
  pooling     : str  {"mean", "last"}
      How to collapse the time dimension into a single vec
  Inputs
  ------
  x_padded : Tensor, shape (B, T_max, 7 or 9)
      Zero-padded batch of (tx,ty,tz,qx,qy,qz,qw).
  pad_mask : BoolTensor, shape (B, T_max)
      True on PAD tokens, False on real ones (for `src_key_padding_mask`).
  lengths  : LongTensor, shape (B,)
      Original (un-padded) sequence lengths.

  Returns
  -------
  features : Tensor, shape (B, output_dim)
      Encoded camera pose features.
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

    # Set output dimension (default to d_model if not specified)
    self.output_dim = output_dim if output_dim is not None else d_model

    # 1) 7-D or 9-D → model width
    self.in_dim = encode_pose_to_dim[self.encode_pose]
    self.in_proj = nn.Linear(self.in_dim, d_model)
    if self.encode_pose == 11:
      self.in_proj_gravity = nn.Linear(3, d_model)

    # 2) sine-cosine positional table (no extra class needed)
    pe = torch.zeros(max_len, d_model)  # (T,d)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    self.register_buffer("pos_table", pe.unsqueeze(0))  # (1,T,d)
    self.dropout = nn.Dropout(dropout)

    # 3) Transformer encoder (batch_first → (B,T,E))
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

    # 4) Output processing
    self.norm = nn.LayerNorm(d_model)
    # Add projection layer if output_dim is different from d_model
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
    # (B,T,7 or 9) → (B,T,d_model)
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

    # Label token (B, d_model) → (B,1,d_model)
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
      )  # (B, T+1)

    # add positions
    h = h + self.pos_table[:, : h.size(1)]
    h = self.dropout(h)

    # Transformer
    h = self.encoder(h, src_key_padding_mask=pad_mask)  # (B,T,d)
    if return_middle_h:
      middle_h = copy.deepcopy(h)
    # Pool time → vector or keep sequence
    if self.pooling == "mean":
      h_sum = h.masked_fill(pad_mask.unsqueeze(-1), 0.0).sum(1)
      h = h_sum / lengths.unsqueeze(-1)  # (B,d)
    elif self.pooling == "last":
      idx = (lengths - 1).unsqueeze(-1).expand(-1, h.size(-1))
      h = h.gather(1, idx.unsqueeze(1)).squeeze(1)  # (B,d)
    else:  #'no', keep h as (B,T,d)
      if self.use_scenario_label:
        h = h[:, 1:]
      if self.encode_pose == 11:
        h = h[:, 1:]

    # Normalize and project to output dimension
    h = self.norm(h)
    features = self.out_proj(h)
    if return_middle_h:
      return features, middle_h
    return features

  def get_encoder_state_dict(self) -> dict:
    """Returns a state dict containing only the encoder-related parameters

    that can be used to initialize the classifier's encoder.
    """
    encoder_keys = [
        "in_proj.weight",
        "in_proj.bias",
        "pos_table",
        "encoder.layers",
    ]
    return {
        k: v
        for k, v in self.state_dict().items()
        if any(key in k for key in encoder_keys)
    }


class RaymapPoseEncoder(nn.Module):
  """Encodes a sequence of dense raymaps into features.

  Input Shape: (Batch, Time, Height, Width, Channels) -> (B, T, 448, 448, 6)
  Output Shape: (B, output_dim) or (B, T, output_dim) depending on pooling.

  This model consists of two main parts:
  1. A CNN-based Spatial Encoder that processes each raymap individually
     to extract a feature vector.
  2. A Transformer-based Temporal Encoder that processes the sequence of
     feature vectors to capture the dynamics of the camera motion.

  Args
  ----
  in_channels : int
      Number of channels in the input raymap (e.g., 6 for (ox,oy,oz,dx,dy,dz)).
  d_model     : int
      The feature width used throughout the Transformer (and the output of the
      CNN).
  output_dim  : int or None
      Dimension of the final output features. Defaults to d_model if None.
  nhead       : int
      Number of attention heads for the Transformer.
  num_layers  : int
      Number of layers for the Transformer encoder.
  dim_feedforward : int
      Hidden size of the MLP inside each Transformer block.
  dropout     : float
      Dropout rate applied in the Transformer and after positional encoding.
  max_len     : int
      Maximum sequence length for the positional encoding table.
  pooling     : str, {"mean", "last", "no"}
      Method to collapse the time dimension into a single vector after the
      Transformer.
      - "mean": Averages the features over time.
      - "last": Takes the features from the last time step.
      - "no": Returns the full sequence of features.
  """

  def __init__(
      self,
      in_channels: int = 6,
      d_model: int = 256,
      output_dim: int = None,
      nhead: int = 8,
      num_layers: int = 4,
      dim_feedforward: int = 512,
      dropout: float = 0.1,
      max_len: int = 10000,
      pooling: str = "mean",
  ):
    super().__init__()
    self.pooling = pooling.lower()
    assert self.pooling in {"mean", "last", "no"}
    self.output_dim = output_dim if output_dim is not None else d_model

    # --- 1. Spatial Encoder (CNN) ---
    # This part processes each (448, 448, 6) map.
    # It progressively reduces spatial dimensions and increases channel depth.
    self.spatial_encoder = nn.Sequential(
        # Input: (C, H, W) = (6, 448, 448)
        nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3
        ),  # -> (64, 224, 224)
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1
        ),  # -> (128, 112, 112)
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(
            128, 256, kernel_size=3, stride=2, padding=1
        ),  # -> (256, 56, 56)
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(
            256, d_model, kernel_size=3, stride=2, padding=1
        ),  # -> (d_model, 28, 28)
        nn.BatchNorm2d(d_model),
        nn.ReLU(),
        # Global average pooling collapses spatial dimensions to a single vector
        nn.AdaptiveAvgPool2d((1, 1)),  # -> (d_model, 1, 1)
        nn.Flatten(),  # -> (d_model)
    )

    # --- 2. Temporal Encoder (Transformer) ---
    # This part processes the sequence of feature vectors from the CNN.

    # Sine-cosine positional table for the time dimension
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    self.register_buffer(
        "pos_table", pe.unsqueeze(0)
    )  # Shape: (1, max_len, d_model)
    self.pos_dropout = nn.Dropout(dropout)

    # Standard Transformer Encoder
    enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
    )
    self.temporal_encoder = nn.TransformerEncoder(
        enc_layer, num_layers=num_layers
    )

    # --- 3. Output Processing ---
    self.norm = nn.LayerNorm(d_model)
    self.out_proj = (
        nn.Linear(d_model, self.output_dim)
        if self.output_dim != d_model
        else nn.Identity()
    )

  def forward(self, x_padded, pad_mask, lengths, scenario_labels=None):
    # Input shape: (B, T, H, W, C) e.g., (8, 50, 448, 448, 6)
    B, T, H, W, C = x_padded.shape

    # Reshape for CNN: view all raymaps in the batch as a single collection of images
    # (B, T, H, W, C) -> (B*T, C, H, W)
    x_reshaped = x_padded.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)

    # 1. Spatial Encoding
    # Pass all (B*T) images through the CNN
    # h_spatial shape: (B*T, d_model)
    h_spatial = self.spatial_encoder(x_reshaped)

    # Reshape back to a sequence for the Transformer
    # h shape: (B, T, d_model)
    h = h_spatial.view(B, T, -1)

    # 2. Add positional encoding to the sequence
    h = h + self.pos_table[:, :T]
    h = self.pos_dropout(h)

    # 3. Temporal Encoding (Transformer)
    # The pad_mask ensures attention is not computed over padded time steps
    h = self.temporal_encoder(
        h, src_key_padding_mask=pad_mask
    )  # Shape: (B, T, d_model)

    # 4. Pool time dimension to get a single feature vector per sequence
    if self.pooling == "mean":
      # Mask out padded values before summing to get a correct mean
      h_sum = h.masked_fill(pad_mask.unsqueeze(-1), 0.0).sum(1)
      h = h_sum / lengths.unsqueeze(-1)  # Shape: (B, d_model)
    elif self.pooling == "last":
      # Gather the features from the last valid time step for each sequence
      idx = (lengths - 1).view(-1, 1, 1).expand(-1, -1, h.size(-1))
      h = h.gather(1, idx).squeeze(1)  # Shape: (B, d_model)
    # If pooling is 'no', h remains (B, T, d_model)

    # 5. Final normalization and projection to the desired output dimension
    h = self.norm(h)
    features = self.out_proj(h)

    return features


class CameraPoseSeqClassifier(nn.Module):
  """Camera-pose sequence → class-logits, with optional encoder pre-initialization

  This classifier uses the same encoder architecture as CameraPoseSeqEncoder
  and can be initialized from a pre-trained encoder checkpoint.

  Args
  ----
  num_classes : int
      Number of target categories.
  d_model     : int
      Transformer feature width (input → encoder → head).
  feature_dim : int or None
      Dimension of encoder features. If None, will be same as d_model.
  nhead       : int
      Number of attention heads.
  num_layers  : int
      Transformer-encoder depth.
  dim_feedforward : int
      Hidden size of the MLP inside each encoder block.
  dropout     : float
      Drop-out applied after positional encoding and inside blocks.
  max_len     : int
      Maximum sequence length you expect (for the positional table).
  pooling     : str  {"mean", "last"}
      How to collapse the time dimension into a single vector.
  init_ckpt   : str or None
      Path to encoder checkpoint (.ckpt) to initialize from.

  Inputs
  ------
  x_padded : Tensor, shape (B, T_max, 7 or 9)
      Zero-padded batch of (tx,ty,tz,qx,qy,qz,qw).
  pad_mask : BoolTensor, shape (B, T_max)
      True on PAD tokens, False on real ones (for `src_key_padding_mask`).
  lengths  : LongTensor, shape (B,)
      Original (un-padded) sequence lengths.

  Returns
  -------
  logits : Tensor, shape (B, num_classes)
      Un-normalised class scores (use `CrossEntropyLoss`).
  """

  def __init__(
      self,
      num_classes: int,
      encode_pose: int = 0,
      d_model: int = 128,
      feature_dim: int = None,
      nhead: int = 4,
      num_layers: int = 4,
      dim_feedforward: int = 256,
      dropout: float = 0.1,
      max_len: int = 10000,
      pooling: str = "mean",
      init_ckpt: str = None,
      use_learnable_gravity: bool = False,
  ):
    super().__init__()
    self.pooling = pooling.lower()
    assert self.pooling in {"mean", "last", "no"}

    # 1) Create encoder with same architecture
    self.encoder = CameraPoseSeqEncoder(
        d_model=d_model,
        encode_pose=encode_pose,
        output_dim=feature_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        pooling=pooling,
        use_learnable_gravity=use_learnable_gravity,
    )

    # Get actual feature dimension
    feature_dim = feature_dim if feature_dim is not None else d_model

    # 2) Initialize from checkpoint if provided
    if init_ckpt is not None:
      print(f"Initializing encoder from checkpoint: {init_ckpt}")
      try:
        # First try loading with weights_only=True (safer)
        ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=True)
      except Exception as e:
        print(f"Warning: Failed to load checkpoint with weights_only=True: {e}")
        print("Attempting to load with weights_only=False (less secure)")
        # If that fails, try with weights_only=False
        ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)

      # Handle both full state dict and encoder-only state dict
      if "state_dict" in ckpt:
        # If it's a full model checkpoint, extract just the encoder part
        ckpt = {
            k.replace("model.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }

      # Load the state dict and print statistics
      missing_keys, unexpected_keys = self.encoder.load_state_dict(
          ckpt, strict=False
      )
      print(
          f"\nCheckpoint loading: {len(ckpt) - len(missing_keys)}/{len(ckpt)}"
          " parameters loaded successfully"
      )
      if missing_keys:
        print("Missing keys:", ", ".join(missing_keys))

    # 3) Classification head
    self.head = nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, num_classes),
    )

  def forward(
      self,
      x_padded: torch.Tensor,
      pad_mask: torch.Tensor,
      lengths: torch.Tensor,
      return_middle_h=False,
  ) -> torch.Tensor:
    # Get encoded features
    if return_middle_h:
      features, h = self.encoder(
          x_padded, pad_mask, lengths, return_middle_h=True
      )
    else:
      features = self.encoder(
          x_padded, pad_mask, lengths, return_middle_h=False
      )

    # Class logits
    logits = self.head(features)  # (B,C)
    if return_middle_h:
      return logits, h
    return logits
