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

"""Base CamFormer contrastive pretraining task.

This is the non-contextualized ("short sequence") variant: the camera-pose
encoder internally mean-pools each trajectory to a single motion embedding,
which is aligned with the CLIP text embedding via symmetric InfoNCE.

`train.py` selects this task for datasets whose name does NOT contain
`longseq` (e.g. `dynpose_pretrain`, `nymeria_pretrain`). The contextualized
Ego-Exo4D variant lives in `tasks/pretrain_longseq.py` and reuses `CLIPLoss`
from this module.
"""

import os

import clip
from loader import create_loader
from models.cm_encoder import CameraPoseSeqEncoder
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
  """Symmetric InfoNCE over matched (motion, text) pairs in a batch."""

  def __init__(self, temperature=0.07):
    super().__init__()
    self.temperature = temperature

  def forward(self, motion_features, text_features):
    motion_features = F.normalize(motion_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = torch.matmul(motion_features, text_features.t()) / self.temperature
    labels = torch.arange(len(motion_features), device=motion_features.device)
    loss_m = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_m + loss_t) / 2


class CameraPoseTextPretraining(pl.LightningModule):

  def __init__(self, args):
    super().__init__()
    self.args = args
    self.output_dim = 512  # CLIP ViT-B/32 text-embed dim

    self.model = CameraPoseSeqEncoder(
        d_model=args.d_model,
        encode_pose=args.encode_pose,
        output_dim=self.output_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pooling='mean',
        use_scenario_label=args.use_scenario_label,
        use_learnable_gravity=args.use_learnable_gravity,
    )

    # Frozen CLIP text tower.
    self.clip_model, _ = clip.load('ViT-B/32', device=self.device)
    for param in self.clip_model.parameters():
      param.requires_grad = False

    self.criterion = CLIPLoss()
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

    self.test_motion_features = []
    self.test_text_features = []

  def configure_optimizers(self):
    return torch.optim.AdamW(
        self.model.parameters(),
        lr=self.args.lr,
        weight_decay=self.args.weight_decay,
    )

  def _encode_text(self, texts):
    text_tokens = clip.tokenize(texts).to(self.device)
    with torch.no_grad():
      return self.clip_model.encode_text(text_tokens)

  def training_step(self, batch, batch_idx):
    poses, pad_mask, texts, lengths = batch
    motion_features = self.model(poses, pad_mask, lengths)
    text_features = self._encode_text(texts)
    loss = self.criterion(motion_features, text_features)
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
    motion_features = self.model(poses, pad_mask, lengths)
    text_features = self._encode_text(texts)
    loss = self.criterion(motion_features, text_features)

    motion_features = F.normalize(motion_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    sim_matrix = torch.matmul(motion_features, text_features.t())

    m2t = self._calculate_ranks(sim_matrix)
    t2m = self._calculate_ranks(sim_matrix.t())
    metrics = {
        'val_loss': loss,
        'm2t_r1': (m2t < 1).float().mean(),
        'm2t_r5': (m2t < 5).float().mean(),
        'm2t_r10': (m2t < 10).float().mean(),
        'm2t_median_rank': m2t.median(),
        't2m_r1': (t2m < 1).float().mean(),
        't2m_r5': (t2m < 5).float().mean(),
        't2m_r10': (t2m < 10).float().mean(),
        't2m_median_rank': t2m.median(),
    }
    for name, value in metrics.items():
      self.log(
          name, value, on_step=False, on_epoch=True, prog_bar=True, logger=True
      )
    return metrics

  def test_step(self, batch, batch_idx):
    poses, pad_mask, texts, lengths = batch
    motion_features = self.model(poses, pad_mask, lengths)
    text_features = self._encode_text(texts)
    motion_features = F.normalize(motion_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    self.test_motion_features.append(motion_features.detach().cpu())
    self.test_text_features.append(text_features.detach().cpu())

  def on_test_end(self):
    all_features = torch.cat(self.test_motion_features, dim=0)
    all_text_features = torch.cat(self.test_text_features, dim=0)

    ref = self.args.ckpt or self.args.init_ckpt
    tag = os.path.splitext(os.path.basename(ref))[0] if ref else 'eval'
    if self.args.dataset == 'dynpose_pretrain':
      save_dir = f'final_data/retrieval_features/ours/{self.args.dataset}/{tag}_{self.args.pose_source}'
    elif self.args.dataset == 'nymeria_pretrain':
      save_dir = f'final_data/retrieval_features/ours/{self.args.dataset}/{tag}/text_{self.args.text_column}'
    else:
      save_dir = 'tmp/retrieval_features/ours_full/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(all_features, os.path.join(save_dir, 'frames.pt'))
    torch.save(all_text_features, os.path.join(save_dir, 'text.pt'))
    print(
        f'Saved features {all_features.shape}, {all_text_features.shape} to'
        f' {save_dir}'
    )

  def _calculate_ranks(self, sim_matrix):
    sorted_indices = torch.argsort(sim_matrix, dim=-1, descending=True)
    correct_indices = torch.arange(len(sim_matrix), device=sim_matrix.device)
    return torch.where(sorted_indices == correct_indices[:, None])[1]

  def train_dataloader(self):
    return create_loader(self.args, 'train')

  def val_dataloader(self):
    return create_loader(self.args, 'val')

  def test_dataloader(self):
    test_loader, _ = create_loader(self.args, 'test')
    return test_loader
