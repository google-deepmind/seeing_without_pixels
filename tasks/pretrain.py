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

import os
import clip
from loader import create_loader
from models.cm_encoder import CameraPoseSeqEncoder, RaymapPoseEncoder
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):

  def __init__(self, temperature=0.07):
    super().__init__()
    self.temperature = temperature

  def forward(self, motion_features, text_features):
    # Normalize features
    motion_features = F.normalize(motion_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Compute similarity matrix
    logits = torch.matmul(motion_features, text_features.t()) / self.temperature

    # Labels are diagonal (matching pairs)
    labels = torch.arange(len(motion_features), device=motion_features.device)

    # Compute loss in both directions
    loss_m = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)

    return (loss_m + loss_t) / 2


class CameraPoseTextPretraining(pl.LightningModule):

  def __init__(self, args):
    super().__init__()
    self.args = args

    # Camera motion encoder
    if False:
      self.model = RaymapPoseEncoder(
          in_channels=6,
          d_model=args.d_model,
          output_dim=4096 if args.use_text_embeds else 512,
          nhead=args.nhead,
          num_layers=args.num_layers,
          dim_feedforward=args.dim_feedforward,
          dropout=args.dropout,
          pooling='mean',
      )
    else:
      self.model = CameraPoseSeqEncoder(
          d_model=args.d_model,
          encode_pose=args.encode_pose,
          output_dim=4096
          if args.use_text_embeds
          else 512,  # Match CLIP's feature dimension
          nhead=args.nhead,
          num_layers=args.num_layers,
          dim_feedforward=args.dim_feedforward,
          dropout=args.dropout,
          pooling='mean',
          use_scenario_label=args.use_scenario_label,
          use_learnable_gravity=args.use_learnable_gravity,
      )

    # Load CLIP model
    if not args.use_text_embeds:
      self.clip_model, _ = clip.load('ViT-B/32', device=self.device)
      # Freeze CLIP parameters
      for param in self.clip_model.parameters():
        param.requires_grad = False

    # Loss function
    self.criterion = CLIPLoss()

    # Save hyperparameters for logging
    self.save_hyperparameters()

    if args.init_ckpt != '':
      print(f'Initializing model weights from {args.init_ckpt}')
      checkpoint = torch.load(
          args.init_ckpt, map_location='cpu', weights_only=False
      )
      # If using PyTorch Lightning checkpoint, use 'state_dict'
      state_dict = (
          checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
      )
      # Remove 'model.' prefix if present in keys
      new_state_dict = {}
      for k, v in state_dict.items():
        if k.startswith('model.'):
          new_state_dict[k[len('model.') :]] = v
        else:
          new_state_dict[k] = v
      self.model.load_state_dict(new_state_dict, strict=False)

    self.test_motion_features = []
    self.test_text_features = []

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=self.args.lr,
        weight_decay=self.args.weight_decay,
    )
    return optimizer

  def training_step(self, batch, batch_idx):
    poses, pad_mask, texts, lengths = batch

    # Get motion features
    motion_features = self.model(poses, pad_mask, lengths)

    # Handle text based on whether it's raw text or pre-computed embeddings
    if self.args.use_text_embeds:
      text_features = texts.detach().to(self.device)
    else:
      text_tokens = clip.tokenize(texts).to(self.device)
      with torch.no_grad():
        text_features = self.clip_model.encode_text(text_tokens)

    # Compute loss
    loss = self.criterion(motion_features, text_features)

    # Log metrics
    self.log(
        'train_loss',
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )
    return loss

  def validation_step(self, batch, batch_idx):
    poses, pad_mask, texts, lengths = batch

    # Get motion features
    motion_features = self.model(poses, pad_mask, lengths)

    # Handle text based on whether it's raw text or pre-computed embeddings
    if self.args.use_text_embeds:
      text_features = texts.detach().to(self.device)
    else:
      text_tokens = clip.tokenize(texts).to(self.device)
      with torch.no_grad():
        text_features = self.clip_model.encode_text(text_tokens)

    # Compute loss
    loss = self.criterion(motion_features, text_features)

    # Compute similarity matrix for retrieval metrics
    motion_features = F.normalize(motion_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # if self.args.test:
    #
    sim_matrix = torch.matmul(motion_features, text_features.t())

    # Calculate retrieval metrics
    motion_to_text_ranks = self._calculate_ranks(sim_matrix)
    text_to_motion_ranks = self._calculate_ranks(sim_matrix.t())

    metrics = {
        'val_loss': loss,
        'm2t_r1': (motion_to_text_ranks < 1).float().mean(),
        'm2t_r5': (motion_to_text_ranks < 5).float().mean(),
        'm2t_r10': (motion_to_text_ranks < 10).float().mean(),
        'm2t_median_rank': motion_to_text_ranks.median(),
        't2m_r1': (text_to_motion_ranks < 1).float().mean(),
        't2m_r5': (text_to_motion_ranks < 5).float().mean(),
        't2m_r10': (text_to_motion_ranks < 10).float().mean(),
        't2m_median_rank': text_to_motion_ranks.median(),
    }

    # Log all metrics
    for name, value in metrics.items():
      self.log(
          name, value, on_step=False, on_epoch=True, prog_bar=True, logger=True
      )

    return metrics

  def test_step(self, batch, batch_idx):
    # metrics = self.validation_step(batch, batch_idx)
    # # Rename metrics from val_ to test_ prefix
    # return {k.replace('val_', 'test_'): v for k, v in metrics.items()}

    poses, pad_mask, texts, lengths = batch
    motion_features = self.model(poses, pad_mask, lengths)
    text_tokens = clip.tokenize(texts).to(self.device)
    with torch.no_grad():
      text_features = self.clip_model.encode_text(text_tokens)

    motion_features = F.normalize(motion_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    self.test_motion_features.append(motion_features.detach().cpu())
    self.test_text_features.append(text_features.detach().cpu())

  def on_test_end(self):
    all_features = torch.cat(
        self.test_motion_features, dim=0
    )  # Shape: [dataset_size, d]
    all_text_features = torch.cat(
        self.test_text_features, dim=0
    )  # Shape: [dataset_size, d]

    if self.args.dataset in ['egoexo4d_pretrain', 'dynpose_pretrain']:
      save_dir = f"final_data/retrieval_features/ours/{self.args.dataset}/{self.args.eval_data}_{self.args.ckpt.split('/')[-3]}"  # _{self.args.eval_data}/egovisible{self.args.ego_visible}'
    elif self.args.dataset == 'nymeria_pretrain':
      save_dir = f"final_data/retrieval_features/ours/{self.args.dataset}/{self.args.ckpt.split('/')[-3]}_gravity{self.args.use_learnable_gravity}/text_{self.args.text_column}"
    elif self.args.dataset == 'demo':
      save_dir = f"final_data/retrieval_features/ours/{self.args.dataset}/{self.args.ckpt.split('/')[-3]}"
    else:
      save_dir = 'tmp/cam_viz/oursfeatures_full/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(all_features, os.path.join(save_dir, 'frames.pt'))
    torch.save(all_text_features, os.path.join(save_dir, 'text.pt'))
    print(
        f'Saved features {all_features.shape}, {all_text_features.shape} to'
        f' {save_dir}'
    )

  def _calculate_ranks(self, sim_matrix):
    # For each row, get the indices that would sort it in descending order
    sorted_indices = torch.argsort(sim_matrix, dim=-1, descending=True)
    # Create a tensor of correct indices (diagonal)
    correct_indices = torch.arange(len(sim_matrix), device=sim_matrix.device)
    # Find where the correct indices appear in the sorted list
    ranks = torch.where(sorted_indices == correct_indices[:, None])[1]
    return ranks

  def train_dataloader(self):
    return create_loader(self.args, 'train')

  def val_dataloader(self):
    return create_loader(self.args, 'val')

  def test_dataloader(self):
    test_loader, _ = create_loader(self.args, 'test')
    return test_loader
