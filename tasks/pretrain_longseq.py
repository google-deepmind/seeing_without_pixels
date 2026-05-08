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
import matplotlib.pyplot as plt
from models.cm_encoder import CameraPoseSeqEncoder, RaymapPoseEncoder
from models.text_encoder import ContextAwareTextEncoder
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from tasks.pretrain import CLIPLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

class_names = [
    'Basketball',
    'Dance',
    'Music',
    'Soccer',
    'Cooking',
    'Rock Climbing',
    'Health',
    'Bike Repair',
]


class CameraPoseTextPretrainingLongSeq(pl.LightningModule):

  def __init__(self, args):
    super().__init__()
    self.args = args

    # Initialize test feature accumulation
    self.test_motion_features = []
    self.test_text_features = []
    # For classification evaluation in test mode
    self.test_cls_preds = []
    self.test_cls_labels = []
    # For classification evaluation in val mode
    self.val_cls_preds = []
    self.val_cls_labels = []

    # Camera motion encoder
    self.output_dim = 4096 if args.use_text_embeds else 512
    if False:
      self.model = RaymapPoseEncoder(
          in_channels=6,
          d_model=args.d_model,
          output_dim=self.output_dim,
          nhead=args.nhead,
          num_layers=args.num_layers,
          dim_feedforward=args.dim_feedforward,
          dropout=args.dropout,
          pooling='no',
      )
    else:
      self.model = CameraPoseSeqEncoder(
          d_model=args.d_model,
          encode_pose=args.encode_pose,
          output_dim=self.output_dim,  # Match CLIP's feature dimension
          nhead=args.nhead,
          num_layers=args.num_layers,
          dim_feedforward=args.dim_feedforward,
          dropout=args.dropout,
          pooling='no',
          use_scenario_label=args.use_scenario_label,
          use_learnable_gravity=args.use_learnable_gravity,
      )

    # Load text model
    if not args.use_text_embeds:
      if self.args.hier_text:
        self.text_encoder = ContextAwareTextEncoder()
      else:
        self.clip_model, _ = clip.load('ViT-B/32', device=self.device)
        if not args.finetune_clip:
          for param in self.clip_model.parameters():
            param.requires_grad = False

    # Loss function
    self.criterion = CLIPLoss()

    # Auxiliary classification model
    self.test_cls = False
    if args.cls_loss_weight > 0:
      self.num_scenarios = 8
      self.cls_model = nn.Linear(self.output_dim, self.num_scenarios)
      self.cls_criterion = nn.CrossEntropyLoss()
      self.test_cls = True

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

  def configure_optimizers(self):
    params = list(self.model.parameters())
    if self.args.cls_loss_weight > 0:
      params.extend(list(self.cls_model.parameters()))

    optimizer = torch.optim.AdamW(
        params, lr=self.args.lr, weight_decay=self.args.weight_decay
    )
    return optimizer

  def encode_motion(self, poses, pad_mask, text_list, lengths, scenario_labels):
    motion_features = self.model(poses, pad_mask, lengths, scenario_labels)
    motion_means = [
        motion_features[j, t[0] : t[1]].mean(dim=0)
        for j, t in enumerate(text_list)
    ]
    features = torch.stack(motion_means)
    return motion_features, features

  def encode_text(self, text_list, full_text_list):
    if self.args.hier_text:
      text_features = self.text_encoder(full_text_list)
    else:
      all_texts = [text[2] for text in text_list]
      text_tokens = clip.tokenize(all_texts).to(self.device)
      if self.args.finetune_clip:
        text_features = self.clip_model.encode_text(text_tokens)
      else:
        with torch.no_grad():
          text_features = self.clip_model.encode_text(text_tokens)
    return text_features

  def training_step(self, batch, batch_idx):
    poses, pad_mask, text_list, lengths, scenario_labels, full_text_list = batch

    motion_features, features = self.encode_motion(
        poses, pad_mask, text_list, lengths, scenario_labels
    )
    text_features = self.encode_text(text_list, full_text_list)

    assert not torch.isnan(features).any(), 'NaN in features'
    assert not torch.isinf(features).any(), 'Inf in features'
    assert not torch.isnan(text_features).any(), 'NaN in text_features'
    assert not torch.isinf(text_features).any(), 'Inf in text_features'

    # Contrastive loss
    contrastive_loss = self.criterion(features, text_features)
    total_loss = contrastive_loss

    # Classification loss on motion features
    if self.args.cls_loss_weight > 0:
      # Average motion_features along temporal dimension
      motion_features_avg = motion_features.mean(
          dim=1
      )  # Shape: [batch_size, d_model]

      # Use averaged features for classification
      cls_logits = self.cls_model(
          motion_features_avg
      )  # Shape: [batch_size, num_scenarios]

      # Direct classification loss (scenario_labels already matches batch_size)
      cls_loss = self.cls_criterion(cls_logits, scenario_labels)
      total_loss = contrastive_loss + self.args.cls_loss_weight * cls_loss

      # Log classification metrics
      self.log(
          'train_cls_loss',
          cls_loss,
          on_step=True,
          on_epoch=True,
          prog_bar=True,
          logger=True,
      )
      self.log(
          'train_total_loss',
          total_loss,
          on_step=True,
          on_epoch=True,
          prog_bar=True,
          logger=True,
      )

    # Log metrics
    self.log(
        'train_loss',
        contrastive_loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )
    return total_loss

  def validation_step(self, batch, batch_idx):
    poses, pad_mask, text_list, lengths, scenario_labels, full_text_list = batch

    motion_features, features = self.encode_motion(
        poses, pad_mask, text_list, lengths, scenario_labels
    )
    text_features = self.encode_text(text_list, full_text_list)

    loss = self.criterion(features, text_features)
    features = F.normalize(features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    sim_matrix = torch.matmul(features, text_features.t())

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

    # Classification metrics if enabled
    if self.args.cls_loss_weight > 0:
      # Average motion_features along temporal dimension
      motion_features_avg = motion_features.mean(
          dim=1
      )  # Shape: [batch_size, d_model]

      # Use averaged features for classification
      cls_logits = self.cls_model(
          motion_features_avg
      )  # Shape: [batch_size, num_scenarios]

      # Direct classification loss and accuracy
      cls_loss = self.cls_criterion(cls_logits, scenario_labels)
      cls_preds = torch.argmax(cls_logits, dim=1)
      cls_acc = (cls_preds == scenario_labels).float().mean()

      metrics.update({
          'val_cls_loss': cls_loss,
          'val_cls_acc': cls_acc,
      })
      # Accumulate for confusion matrix/accuracy
      self.val_cls_preds.append(cls_preds.detach().cpu())
      self.val_cls_labels.append(scenario_labels.detach().cpu())

    # Log all metrics
    for name, value in metrics.items():
      self.log(
          name, value, on_step=False, on_epoch=True, prog_bar=True, logger=True
      )

    return metrics

  def test_step(self, batch, batch_idx):
    poses, pad_mask, text_list, lengths, scenario_labels, full_text_list = batch

    motion_features, features = self.encode_motion(
        poses, pad_mask, text_list, lengths, scenario_labels
    )
    text_features = self.encode_text(text_list, full_text_list)

    # Accumulate features from this batch
    # motion_features shape: [batch_size, d]
    self.test_motion_features.append(features.detach().cpu())
    self.test_text_features.append(text_features.detach().cpu())

    # Classification evaluation if enabled
    if self.test_cls:
      motion_features_avg = motion_features.mean(dim=1)
      cls_logits = self.cls_model(motion_features_avg)
      cls_preds = torch.argmax(cls_logits, dim=1)
      self.test_cls_preds.append(cls_preds.detach().cpu())
      self.test_cls_labels.append(scenario_labels.detach().cpu())

    return {}

  def on_test_end(self):
    all_features = torch.cat(
        self.test_motion_features, dim=0
    )  # Shape: [dataset_size, d]
    all_text_features = torch.cat(
        self.test_text_features, dim=0
    )  # Shape: [dataset_size, d]

    # Save features to file
    if self.args.scenario == 'all':
      fn = f'all_gravity{self.args.use_learnable_gravity}'
    else:
      fn = (
          f'{self.args.scenario}_egovisible{self.args.ego_visible}'
          if self.args.scenario != ''
          else f'egovisible{self.args.ego_visible}'
      )
    save_dir = f"final_data/retrieval_features/ours/{self.args.dataset}/{fn}/{self.args.ckpt.split('/')[-3]}/{self.args.test_time_ratio}"  # _nopool
    if self.args.sample_dur:
      save_dir = save_dir + f'/testdur{self.args.test_take_duration}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(all_features, os.path.join(save_dir, 'frames.pt'))
    torch.save(all_text_features, os.path.join(save_dir, 'text.pt'))
    print(f'Saved features with shape {all_features.shape} to {save_dir}')

    # Classification evaluation if enabled
    if self.test_cls:
      self._compute_and_save_confusion_matrix(
          self.test_cls_preds, self.test_cls_labels, mode='test'
      )
      # Clear classification accumulators
      self.test_cls_preds = []
      self.test_cls_labels = []

    # Clear the accumulated features
    self.test_motion_features = []
    self.test_text_features = []

  def on_validation_end(self):
    # Classification evaluation if enabled
    if self.test_cls:
      self._compute_and_save_confusion_matrix(
          self.val_cls_preds, self.val_cls_labels, mode='val'
      )
      # Clear classification accumulators
      self.val_cls_preds = []
      self.val_cls_labels = []

  def _calculate_ranks(self, sim_matrix):
    # For each row, get the indices that would sort it in descending order
    sorted_indices = torch.argsort(sim_matrix, dim=-1, descending=True)
    # Create a tensor of correct indices (diagonal)
    correct_indices = torch.arange(len(sim_matrix), device=sim_matrix.device)
    # Find where the correct indices appear in the sorted list
    ranks = torch.where(sorted_indices == correct_indices[:, None])[1]
    return ranks

  def _compute_and_save_confusion_matrix(
      self, preds_list, labels_list, mode='test'
  ):
    preds = torch.cat(preds_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    acc = accuracy_score(labels, preds)
    print(f'{mode.capitalize()} classification accuracy: {acc:.4f}')
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs('tmp', exist_ok=True)
    fn = f'_egovisible{self.args.ego_visible}' if mode == 'test' else ''
    plt.savefig(f'tmp/{mode}_confusion_matrix{fn}.png')
    plt.close()

    # Save normalized confusion matrix (ratios)
    cm_norm = confusion_matrix(labels, preds, normalize='true')
    disp_norm = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm, display_labels=class_names
    )
    disp_norm.plot(cmap=plt.cm.Blues, values_format='.2f')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'tmp/{mode}_confusion_matrix_normalized{fn}.png')
    plt.close()

    print('Confusion matrix saved to tmp')

  def train_dataloader(self):
    return create_loader(self.args, 'train')

  def val_dataloader(self):
    return create_loader(self.args, 'val')

  def test_dataloader(self):
    test_loader, _ = create_loader(self.args, 'test')
    return test_loader
