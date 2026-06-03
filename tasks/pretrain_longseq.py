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
from models.cm_encoder import CameraPoseSeqEncoder
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

    self.test_motion_features = []
    self.test_text_features = []
    self.test_cls_preds = []
    self.test_cls_labels = []
    self.val_cls_preds = []
    self.val_cls_labels = []

    self.output_dim = 512
    self.model = CameraPoseSeqEncoder(
        d_model=args.d_model,
        encode_pose=args.encode_pose,
        output_dim=self.output_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pooling='no',
        use_scenario_label=args.use_scenario_label,
        use_learnable_gravity=args.use_learnable_gravity,
    )

    if self.args.hier_text:
      self.text_encoder = ContextAwareTextEncoder()
    else:
      self.clip_model, _ = clip.load('ViT-B/32', device=self.device)
      if not args.finetune_clip:
        for param in self.clip_model.parameters():
          param.requires_grad = False

    self.criterion = CLIPLoss()

    self.test_cls = False
    if args.cls_loss_weight > 0:
      self.num_scenarios = 8
      self.cls_model = nn.Linear(self.output_dim, self.num_scenarios)
      self.cls_criterion = nn.CrossEntropyLoss()
      self.test_cls = True

    self.save_hyperparameters()

    if args.init_ckpt != '':
      print(f'Initializing model weights from {args.init_ckpt}')
      checkpoint = torch.load(
          args.init_ckpt, map_location='cpu', weights_only=False
      )
      state_dict = (
          checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
      )
      new_state_dict = {}
      for k, v in state_dict.items():
        if k.startswith('model.'):
          new_state_dict[k[len('model.') :]] = v
        else:
          new_state_dict[k] = v
      result = self.model.load_state_dict(new_state_dict, strict=False)
      if result.unexpected_keys:
        print(
            f'[init_ckpt] ignored {len(result.unexpected_keys)} non-encoder'
            f' key(s) (e.g. {result.unexpected_keys[:3]})'
        )
      if result.missing_keys:
        raise RuntimeError(
            f"init_ckpt '{args.init_ckpt}' is missing"
            f' {len(result.missing_keys)} encoder weight(s) (e.g.'
            f' {result.missing_keys[:5]}); this usually means --pose_encoding'
            ' does not match the checkpoint (Ego-Exo4D = rel9d_grav, DynPose ='
            ' rel9d).'
        )

  def configure_optimizers(self):
    params = list(self.model.parameters())
    if self.args.cls_loss_weight > 0:
      params.extend(list(self.cls_model.parameters()))
    optimizer = torch.optim.AdamW(
        params,
        lr=self.args.lr,
        weight_decay=self.args.weight_decay,
    )
    return optimizer

  def encode_motion(self, poses, pad_mask, text_list, lengths, scenario_labels):
    motion_features = self.model(poses, pad_mask, lengths, scenario_labels)
    # Average only the frames inside each sample's [t0:t1] window.
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

    contrastive_loss = self.criterion(features, text_features)
    total_loss = contrastive_loss

    if self.args.cls_loss_weight > 0:
      motion_features_avg = motion_features.mean(dim=1)
      cls_logits = self.cls_model(motion_features_avg)
      cls_loss = self.cls_criterion(cls_logits, scenario_labels)
      total_loss = contrastive_loss + self.args.cls_loss_weight * cls_loss
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

    if self.args.cls_loss_weight > 0:
      motion_features_avg = motion_features.mean(dim=1)
      cls_logits = self.cls_model(motion_features_avg)
      cls_loss = self.cls_criterion(cls_logits, scenario_labels)
      cls_preds = torch.argmax(cls_logits, dim=1)
      cls_acc = (cls_preds == scenario_labels).float().mean()
      metrics.update({'val_cls_loss': cls_loss, 'val_cls_acc': cls_acc})
      self.val_cls_preds.append(cls_preds.detach().cpu())
      self.val_cls_labels.append(scenario_labels.detach().cpu())

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

    self.test_motion_features.append(features.detach().cpu())
    self.test_text_features.append(text_features.detach().cpu())

    if self.test_cls:
      motion_features_avg = motion_features.mean(dim=1)
      cls_logits = self.cls_model(motion_features_avg)
      cls_preds = torch.argmax(cls_logits, dim=1)
      self.test_cls_preds.append(cls_preds.detach().cpu())
      self.test_cls_labels.append(scenario_labels.detach().cpu())

    return {}

  def on_test_end(self):
    all_features = torch.cat(self.test_motion_features, dim=0)
    all_text_features = torch.cat(self.test_text_features, dim=0)

    if self.args.scenario == 'all':
      fn = f'all_gravity{self.args.use_learnable_gravity}'
    else:
      fn = (
          f'{self.args.scenario}_egovisible{self.args.ego_visible}'
          if self.args.scenario != ''
          else f'egovisible{self.args.ego_visible}'
      )
    ref = self.args.ckpt or self.args.init_ckpt
    tag = os.path.splitext(os.path.basename(ref))[0] if ref else 'eval'
    save_dir = (
        f'final_data/retrieval_features/ours/{self.args.dataset}/'
        f'{fn}/{tag}/{self.args.test_time_ratio}'
    )
    if self.args.sample_dur:
      save_dir = save_dir + f'/testdur{self.args.test_take_duration}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(all_features, os.path.join(save_dir, 'frames.pt'))
    torch.save(all_text_features, os.path.join(save_dir, 'text.pt'))
    print(f'Saved features with shape {all_features.shape} to {save_dir}')

    if self.test_cls:
      self._compute_and_save_confusion_matrix(
          self.test_cls_preds, self.test_cls_labels, mode='test'
      )
      self.test_cls_preds = []
      self.test_cls_labels = []

    self.test_motion_features = []
    self.test_text_features = []

  def on_validation_end(self):
    if self.test_cls:
      self._compute_and_save_confusion_matrix(
          self.val_cls_preds, self.val_cls_labels, mode='val'
      )
      self.val_cls_preds = []
      self.val_cls_labels = []

  def _calculate_ranks(self, sim_matrix):
    sorted_indices = torch.argsort(sim_matrix, dim=-1, descending=True)
    correct_indices = torch.arange(len(sim_matrix), device=sim_matrix.device)
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
