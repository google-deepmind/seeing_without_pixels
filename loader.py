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

from datasets.dynpose import DynPoseCamTextPair
from datasets.egoexo4d_pretrain import EgoExo4DCameraPoseLongSeqForPretraining
from datasets.nymeria import NymeriaCamMotionTextPair
import torch
from torch.nn.utils.rnn import pad_sequence


def pose_text_collate(batch):
  """Flat (trajectory, text) collate for the base pretraining task.

  batch = [(cam_motion_i (Ti, D), text_i), ...]
  Returns:
    poses_padded : (B, T_max, D)
    pad_mask     : (B, T_max)   True on PAD positions
    texts        : list[str]    length B
    lengths      : (B,)         original sequence lengths
  """
  seqs, texts = zip(*batch)
  seqs = [torch.as_tensor(s, dtype=torch.float32) for s in seqs]
  lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
  poses_padded = pad_sequence(seqs, batch_first=True)
  max_T = poses_padded.size(1)
  pad_mask = (
      torch.arange(max_T)[None, :].expand(len(seqs), max_T) >= lengths[:, None]
  )
  return poses_padded, pad_mask, list(texts), lengths


def longseq_pose_text_collate(batch):
  """Contextualized collate for the Ego-Exo4D long-sequence task.

  Each sample carries its [t0, t1, text] window and scenario label so the
  LightningModule can mean-pool only the labeled sub-window.
  """
  seqs, text_list, scenario_labels, description_list = zip(*batch)
  scenario_labels = torch.tensor(scenario_labels, dtype=torch.long)

  seqs = [torch.as_tensor(s, dtype=torch.float32) for s in seqs]
  lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
  poses_padded = pad_sequence(seqs, batch_first=True)
  max_T = poses_padded.size(1)
  pad_mask = (
      torch.arange(max_T)[None, :].expand(len(seqs), max_T) >= lengths[:, None]
  )
  return (
      poses_padded,
      pad_mask,
      text_list,
      lengths,
      scenario_labels,
      description_list,
  )


def create_loader(args, mode):
  if args.dataset == 'egoexo4d_pretrain_longseq':
    dataset = EgoExo4DCameraPoseLongSeqForPretraining(args, mode)
    collate_fn = longseq_pose_text_collate
  elif args.dataset == 'dynpose_pretrain':
    dataset = DynPoseCamTextPair(args, mode)
    collate_fn = pose_text_collate
  elif args.dataset == 'nymeria_pretrain':
    dataset = NymeriaCamMotionTextPair(args, mode)
    collate_fn = pose_text_collate
  else:
    raise ValueError(f'Invalid dataset: {args.dataset}')

  shuffle = mode == 'train'
  drop_last = mode == 'train'
  loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=shuffle,
      collate_fn=collate_fn,
      num_workers=args.num_workers,
      drop_last=drop_last,
  )
  if mode == 'test':
    return loader, None
  return loader
