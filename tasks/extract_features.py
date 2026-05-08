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

import argparse
from collections import defaultdict
import json
import os
from datasets.action_dataset import FineGymCameraPoseLongSeq
from datasets.egoexo4d import EgoExo4DCameraPoseSeqForActionContextFeature, EgoExo4DCameraPoseSeqForActionFeature, EgoExo4DCameraPoseSeqForActionFeatureSubset, EgoExo4DCameraPoseSeqForLocalization
from models.cm_encoder import CameraPoseSeqEncoder
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def get_args():
  parser = argparse.ArgumentParser(
      description='Extract Camera Pose Features for Action Cls/Loc'
  )
  parser.add_argument('--task', type=str, default='cls', help='cls or loc')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=256,
      help='Batch size for feature extraction',
  )
  parser.add_argument(
      '--fps',
      type=int,
      default=20,
      help='Frames per second for feature extraction',
  )
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=1,
      help='Sample rate for feature extraction',
  )
  parser.add_argument(
      '--encode_pose',
      type=int,
      default=2,
      help='Encoding method for camera poses',
  )
  parser.add_argument(
      '--context_ratio', type=float, default=0.0, help='Context ratio'
  )
  parser.add_argument(
      '--num_workers', type=int, default=8, help='Number of workers'
  )
  parser.add_argument(
      '--max_dur', type=int, default=8, help='Max duration for features'
  )
  parser.add_argument(
      '--window_size', type=int, default=80, help='Window size for features'
  )
  parser.add_argument(
      '--window_stride', type=int, default=10, help='Window stride for features'
  )
  parser.add_argument(
      '--pool_len', type=int, default=0, help='Pool length for features'
  )
  parser.add_argument(
      '--ref_frame_idx',
      type=str,
      default='middle',
      choices=['middle', 'start'],
      help='Reference frame index for features',
  )
  parser.add_argument(
      '--method',
      type=str,
      default='gt',
      choices=['gt', 'megasam', 'pi3', 'vipe'],
      help='Method to obtain camera poses',
  )
  parser.add_argument(
      '--umeyama_transform',
      action='store_true',
      help='Apply Umeyama transformation to predictions',
  )
  parser.add_argument(
      '--init_ckpt',
      type=str,
      default='',
      help='Path to checkpoint for model initialization',
  )
  parser.add_argument(
      '--output_dir', type=str, default='', help='Output directory for features'
  )
  parser.add_argument(
      '--use_learnable_gravity',
      action='store_true',
      help='Use learnable gravity in model',
  )
  args = parser.parse_args()
  return args


def pose_collate_localization(batch):
  """Custom collate function for localization dataset

  batch = [(cam_motion_i (Ti,9) torch/np), take_name_i]
  Returns
    poses_padded : (B,T_max,9)
    pad_mask     : (B,T_max)  True on PAD positions
    take_names   : (B,)  list of take names
    lengths      : (B,)  original sequence lengths
  """
  seqs, take_names, ranges = zip(*batch)
  seqs = [torch.as_tensor(s, dtype=torch.float32) for s in seqs]
  lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
  poses_padded = pad_sequence(seqs, batch_first=True)  # zero-pads to max-T
  max_T = poses_padded.size(1)
  # Transformer expects mask=True where PAD
  pad_mask = (
      torch.arange(max_T)[None, :].expand(len(seqs), max_T) >= lengths[:, None]
  )
  return poses_padded, pad_mask, list(take_names), lengths, ranges


def load_architecture_args_from_json(ckpt_path):
  # Try to find args.json in the checkpoint directory or its parent
  ckpt_dir = os.path.dirname(ckpt_path)
  candidates = [
      os.path.join(ckpt_dir, 'args.json'),
      os.path.join(os.path.dirname(ckpt_dir), 'args.json'),
  ]
  for path in candidates:
    if os.path.exists(path):
      with open(path, 'r') as f:
        args_dict = json.load(f)
      return args_dict
  raise FileNotFoundError(
      f'Could not find args.json in {ckpt_dir} or its parent directory.'
  )


def extract_features(args):
  """Extract features from camera pose sequences using trained model"""
  print(f'Saving to {args.output_dir}')

  # Load model architecture from args.json
  arch_args = load_architecture_args_from_json(args.init_ckpt)

  # Initialize dataset
  if args.task == 'loc':
    dataset = EgoExo4DCameraPoseSeqForLocalization(args)
  elif args.task == 'cls':
    dataset = EgoExo4DCameraPoseSeqForActionFeature(args)
  elif args.task == 'cls_context':
    dataset = EgoExo4DCameraPoseSeqForActionContextFeature(args)
  elif args.task == 'cls_subset':
    dataset = EgoExo4DCameraPoseSeqForActionFeatureSubset(args)
  elif args.task == 'cls_finegym':
    dataset = FineGymCameraPoseLongSeq(args)
  else:
    raise ValueError(f'Unknown task: {args.task}')
  # Create dataloader
  dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=False,  # Keep order for analysis
      collate_fn=pose_collate_localization,
      num_workers=args.num_workers,
      drop_last=False,
  )

  # Initialize model with loaded architecture
  model = CameraPoseSeqEncoder(
      d_model=arch_args['d_model'],
      encode_pose=arch_args['encode_pose'],
      output_dim=arch_args.get('output_dim', 512),
      nhead=arch_args['nhead'],
      num_layers=arch_args['num_layers'],
      dim_feedforward=arch_args['dim_feedforward'],
      dropout=arch_args['dropout'],
      pooling='no',
      use_scenario_label=False,
      use_learnable_gravity=args.use_learnable_gravity,
  )

  # Load trained weights
  if args.init_ckpt:
    print(f'Loading model weights from {args.init_ckpt}')
    checkpoint = torch.load(
        args.init_ckpt, map_location='cpu', weights_only=False
    )
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
      state_dict = checkpoint['state_dict']
    else:
      state_dict = checkpoint

    # Remove 'model.' prefix if present in keys
    new_state_dict = {}
    for k, v in state_dict.items():
      if k.startswith('model.'):
        new_state_dict[k[len('model.') :]] = v
      else:
        new_state_dict[k] = v

    # Load state dict and get missing/unexpected keys
    missing_keys, unexpected_keys = model.load_state_dict(
        new_state_dict, strict=False
    )
    if len(missing_keys) > 0:
      print(f'Missing keys: {missing_keys}')
    if len(unexpected_keys) > 0:
      print(f'Unexpected keys: {unexpected_keys}')
    print('Model weights loaded successfully')

  # Move model to device
  device = 'cuda:0'
  model = model.to(device)
  model.eval()

  # Group features by take_name
  take_features = defaultdict(list)
  print(f'Extracting features from {len(dataset)} sequences...')
  with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader)):
      poses, pad_mask, take_names, lengths, ranges = batch
      poses = poses.to(device)
      pad_mask = pad_mask.to(device)
      lengths = lengths.to(device)

      # Extract features
      print(batch_idx, poses.shape, pad_mask.shape, lengths.shape)
      features = model(poses, pad_mask, lengths)  # Shape: (B, output_dim)

      # Store results
      features = features.cpu().numpy()
      if args.pool_len != 0:
        center = features.shape[1] // 2
        features = features[
            :, center - args.pool_len : center + args.pool_len, :
        ]
      if args.task == 'cls' and args.context_ratio > 0:
        features_mean = [
            features[j, t[0] : t[1]].mean(axis=0) for j, t in enumerate(ranges)
        ]
        features = np.stack(features_mean)
      elif 'finegym' not in args.task:
        features = features.mean(axis=1)
      for i, take_name in enumerate(take_names):
        take_features[take_name].append(features[i])

  # Create output directory
  os.makedirs(args.output_dir, exist_ok=True)

  # Save each take's features as <take_name>.npy
  for take_name, feats in take_features.items():
    if 'finegym' in args.task:
      max_seq_len = args.window_stride * len(feats) + args.window_size
      feats_arr = np.zeros((max_seq_len, feats[0].shape[-1]), dtype=np.float32)
      for i, feat in enumerate(feats):
        print(take_name, i, len(feats), feat.shape)
        start_idx = i * args.window_stride
        end_idx = start_idx + feat.shape[0]
        feats_arr[start_idx:end_idx] += feat
        feats_arr[start_idx:end_idx] /= 2
        print(f'setting {start_idx}:{end_idx} feat shape {feat.shape}')
    else:
      feats_arr = np.stack(feats, axis=0)  # (num_chunks, output_dim)

    np.save(os.path.join(args.output_dir, f'{take_name}.npy'), feats_arr)
    # print(f"Saved features for {take_name} with shape {feats_arr.shape}")
  print(f'Saved all features to {args.output_dir}')


if __name__ == '__main__':
  args = get_args()

  if args.task == 'loc':
    args.init_ckpt = os.path.expanduser(
        '~/data/logs/egoexo4d_pretrain_longseq/bs1024_dur4_pose2/checkpoints/best-epoch=490-val_loss=5.6580.ckpt'
    )
    args.init_ckpt = os.path.expanduser(
        '~/data/logs/egoexo4d_pretrain/bs1024_nn0/encodepose2/checkpoints/best-epoch=493-val_loss=5.9971.ckpt'
    )
    args.window_size = 21
    args.window_stride = 11
    args.ref_frame_idx = 'start'
    args.output_dir = os.path.join(
        os.path.expanduser('~/local_data/egoexo4d/features/camera_motion'),
        args.task,
        f'w{args.window_size}_s{args.window_stride}_p{args.pool_len}',
    )
  elif 'cls' in args.task:
    if 'finegym' in args.task:
      args.ckpt = os.path.expanduser(
          '~/data/logs/dynpose_pretrain/v1_vipe/checkpoints/best-epoch=315-val_loss=6.0542.ckpt'
      )
      args.output_dir = os.path.expanduser(
          f'~/local_data/misc/finegym_camera_pose_features/w{args.window_size}_s{args.window_stride}'
      )
    else:
      # args.init_ckpt = os.path.expanduser('~/data/logs/egoexo4d_pretrain_longseq/bs1025_sampledur16_pose2/checkpoints/best-epoch=499-val_loss=5.6426.ckpt')
      save_dir = os.path.expanduser(
          '~/local_data/misc/egoexo4d_action_features'
      )
      save_dir = os.path.expanduser('~/final_data/egoexo4d_action_features')
      ckpt = args.init_ckpt.split('/')[-3]
      fn = (
          f'{args.method}_context{args.context_ratio}'
          if args.task in ['cls', 'cls_context']
          else f'{args.method}_gravity{args.use_learnable_gravity}_transform{args.umeyama_transform}_encodepose{args.encode_pose}'
      )
      fn2 = (
          f'maxdur{args.max_dur}'
          if args.task == 'cls_context'
          else f'{args.fps}fps_w{args.window_size}_s{args.window_stride}'
      )
      # args.output_dir = os.path.join(save_dir, args.task, ckpt, fn, fn2)
      args.output_dir = os.path.join(save_dir, args.task, fn + '_' + fn2, ckpt)
  extract_features(args)
