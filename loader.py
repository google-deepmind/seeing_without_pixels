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

import math, torch, torch.nn as nn
import random
from datasets.action_dataset import FineGymCameraPoseSeq, UCF101CameraPoseSeq
from datasets.demo_data import DemoCamTextPair
from datasets.dynpose import DynPoseCamTextPair
from datasets.egoexo4d import EgoExo4DCameraPoseLongSeqForPretraining, EgoExo4DCameraPoseSeqForActionCls, EgoExo4DCameraPoseSeqForPretraining, EgoExo4DCameraPoseSeqForProficiencyCls, EgoExo4DCameraPoseSeqForScenarioCls, EgoExo4DCameraPoseSeqForScenarioClsSubset
from datasets.hdepic import HdEpicCameraPoseSeq
from datasets.nymeria import NymeriaCamMotionTextPair
from datasets.viz_dataset import VizCamMotionTextPair
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def pose_collate(batch):
  """batch = [(cam_motion_i (Ti,7) torch/np), label_i, id_i, ...]

  Returns
    poses_padded : (B,T_max,7)
    pad_mask     : (B,T_max)  True on PAD positions
    labels       : (B,)
    lengths      : (B,)  original sequence lengths
    ids          : (B,)  list of ids
  """
  # Check if batch[0] is a list (test mode, multiple clips per sample)
  if isinstance(batch[0], list):
    # Flatten all clips
    flat = [item for sublist in batch for item in sublist]
    seqs, labels, ids = zip(*flat)
    seqs = [torch.as_tensor(s, dtype=torch.float32) for s in seqs]
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    poses_padded = pad_sequence(seqs, batch_first=True)
    max_T = poses_padded.size(1)
    pad_mask = (
        torch.arange(max_T)[None, :].expand(len(seqs), max_T)
        >= lengths[:, None]
    )
    return poses_padded, pad_mask, torch.tensor(labels), lengths, list(ids)
  else:
    seqs, labels, ids = zip(*batch)
    seqs = [torch.as_tensor(s, dtype=torch.float32) for s in seqs]
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    poses_padded = pad_sequence(seqs, batch_first=True)  # zero-pads to max-T
    max_T = poses_padded.size(1)
    # Transformer expects mask=True where PAD
    pad_mask = (
        torch.arange(max_T)[None, :].expand(len(seqs), max_T)
        >= lengths[:, None]
    )
    return poses_padded, pad_mask, torch.tensor(labels), lengths, list(ids)


def pose_text_collate(batch):
  """batch = [(cam_motion_i (Ti,7) torch/np), text_i, [neg_cam_motion_1, ...], [neg_text_1, ...]]

  Returns
    poses_padded : (B*(1+K),T_max,7) where K is num_negatives
    pad_mask     : (B*(1+K),T_max)  True on PAD positions
    texts        : List[str] or torch.Tensor (B*(1+K), d) depending on
    use_text_embeds
    lengths      : (B*(1+K),)  original sequence lengths
  """
  # Check if we're using text embeddings (first item's text is numpy array) or raw text (first item's text is string)
  use_text_embeds = isinstance(batch[0][1], (np.ndarray, torch.Tensor))

  # Unpack the batch - now each item might have negative samples
  if len(batch[0]) == 4:  # If we have negative samples and texts
    seqs, texts, neg_seqs, neg_texts = zip(*batch)
    # Flatten positive and negative sequences
    all_seqs = []
    all_texts = []
    for seq, text, negs, neg_txts in zip(seqs, texts, neg_seqs, neg_texts):
      all_seqs.append(seq)
      all_texts.append(text)
      all_seqs.extend(negs)
      all_texts.extend(neg_txts)
    seqs = all_seqs
    texts = all_texts
  else:  # Original case without negatives
    seqs, texts = zip(*batch)

  seqs = [torch.as_tensor(s, dtype=torch.float32) for s in seqs]
  lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
  poses_padded = pad_sequence(seqs, batch_first=True)  # zero-pads to max-T
  max_T = poses_padded.size(1)
  # Transformer expects mask=True where PAD
  pad_mask = (
      torch.arange(max_T)[None, :].expand(len(seqs), max_T) >= lengths[:, None]
  )

  # Handle text based on whether we're using embeddings or raw text
  if use_text_embeds:
    # Convert text embeddings to tensor
    text_tensors = []
    for t in texts:
      if isinstance(t, np.ndarray):
        text_tensors.append(
            torch.from_numpy(t).float().detach().requires_grad_(False)
        )
      elif isinstance(t, torch.Tensor):
        text_tensors.append(t.float().detach().requires_grad_(False))
      else:
        raise TypeError(f'Unexpected text type: {type(t)}')
    texts = torch.stack(text_tensors)

  return poses_padded, pad_mask, texts, lengths


def longseq_pose_text_collate(batch):
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
  if args.dataset == 'egoexo4d_pretrain':
    dataset = EgoExo4DCameraPoseSeqForPretraining(args, mode)
    collate_fn = pose_text_collate
  elif args.dataset == 'demo':
    dataset = DemoCamTextPair(args)
    collate_fn = pose_text_collate
  elif args.dataset == 'egoexo4d_pretrain_longseq':
    dataset = EgoExo4DCameraPoseLongSeqForPretraining(args, mode)
    collate_fn = longseq_pose_text_collate
  elif args.dataset == 'dynpose_pretrain':
    dataset = DynPoseCamTextPair(args, mode)
    collate_fn = pose_text_collate
  elif args.dataset == 'nymeria_pretrain':
    dataset = NymeriaCamMotionTextPair(args, mode)
    collate_fn = pose_text_collate
  elif args.dataset == 'viz_pretrain':
    dataset = VizCamMotionTextPair()
    collate_fn = pose_text_collate

  elif args.dataset == 'egoexo4d_scenario':
    dataset = EgoExo4DCameraPoseSeqForScenarioCls(args, mode)
    collate_fn = pose_collate
  elif args.dataset == 'egoexo4d_scenario_subset':
    dataset = EgoExo4DCameraPoseSeqForScenarioClsSubset(args, mode)
    collate_fn = pose_collate
  elif args.dataset == 'egoexo4d_prof':
    dataset = EgoExo4DCameraPoseSeqForProficiencyCls(args, mode)
    collate_fn = pose_collate
  elif args.dataset == 'egoexo4d_action':
    # dataset = EgoExo4DCameraPoseSeqForAction(args, mode)
    dataset = EgoExo4DCameraPoseSeqForActionCls(args, mode)
    collate_fn = pose_collate
  # elif args.dataset == 'hdepic':
  #     dataset = HdEpicCameraPoseSeq(args, mode)
  #     collate_fn = pose_collate
  elif args.dataset == 'ucf101':
    dataset = UCF101CameraPoseSeq(args, mode)
    collate_fn = pose_collate
  elif args.dataset == 'finegym':
    dataset = FineGymCameraPoseSeq(args, mode)
    collate_fn = pose_collate
  else:
    raise ValueError(f'Invalid dataset: {args.dataset}')

  # Use original batch_size - the actual batch will be batch_size * (1 + num_negatives)
  batch_size = len(dataset) if args.use_label_id != '' else args.batch_size

  shuffle = mode == 'train'  # test loader just for visualization
  drop_last = mode == 'train'
  loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      collate_fn=collate_fn,
      num_workers=args.num_workers,
      drop_last=drop_last,
  )
  if mode == 'test' and args.dataset == 'egoexo4d_scenario':
    return loader, dataset.label_mapping
  if mode == 'test' and args.dataset == 'ucf101':
    return loader, dataset.label_name_mapping
  elif mode == 'test':
    return loader, None
  else:
    return loader


if __name__ == '__main__':
  dataset = EgoExo4DCameraPoseLongSeqForPretraining(None, 'train')
  # dataset = EgoExo4DCameraPoseSeqForPretraining(None, 'train')
  loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=128,
      shuffle=False,
      collate_fn=longseq_pose_text_collate,
  )
  from models.cm_encoder import CameraPoseSeqEncoder

  model = CameraPoseSeqEncoder(encode_pose=11, pooling='no')
  for i, batch in enumerate(loader):
    poses, pad_mask, text_list, lengths, scenario_labels, full_text_list = batch
    motion_features = model(poses, pad_mask, lengths, scenario_labels)
