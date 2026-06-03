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

"""DynPose-100K camera-pose / text pairs for exocentric CamFormer pretraining.

This is the exocentric counterpart to `egoexo4d_pretrain.py`. Each sample is a
(camera-pose trajectory, video caption) pair, where the trajectory is either the
dataset's original pose (`--pose_source original`) or a ViPE-estimated pose
(`--pose_source vipe`). Pretraining runs through the base `tasks/pretrain.py`
LightningModule (the dataset name has no `longseq`, so `train.py` selects the
base task), with the flat `pose_text_collate` from `loader.py`.
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
from utils.dataset_utils import matrix_to_pose7d, transform_camera_pose, w2c_to_c2w

# Packaged metadata CSVs shipped alongside this module (if present).
_PKG_DATA_FILES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data_files',
    'dynpose100k',
)
# Base dir holding `data_files/dynpose100k/metadata_*.csv` and the per-clip pose
# files referenced (relative to this dir) by the metadata. Overridable via
# DYNPOSE_DATA_DIR; defaults to the public `~/final_data` bundle.
_DEFAULT_DATA_DIR = os.environ.get(
    'DYNPOSE_DATA_DIR',
    os.path.expanduser('~/final_data'),
)


def _resolve_csv(fn):
  """Prefer the bundled data_files/dynpose100k/<fn>; fall back to the bundle."""
  local = os.path.join(_PKG_DATA_FILES_DIR, fn)
  if os.path.exists(local):
    return local
  return os.path.join(_DEFAULT_DATA_DIR, 'data_files', 'dynpose100k', fn)


class DynPoseCamTextPair(torch.utils.data.Dataset):
  """DynPose-100K (exocentric) camera-pose / caption pairs.

  Each sample yields: (cam_traj (N, D), caption). The pose source is chosen by
  `args.pose_source`: 'original' (the dataset's own poses) or 'vipe'
  (ViPE-estimated). Clips whose pose scale exceeds a threshold are dropped as
  unreliable. Pose files are read relative to DYNPOSE_DATA_DIR.
  """

  def __init__(self, args, mode):
    self.args = args
    self.encode_pose = args.encode_pose if args is not None else 2
    self.sample_rate = args.sample_rate if args is not None else 1
    self.mode = mode
    self.pose_source = 'original' if args is None else args.pose_source
    assert self.pose_source in [
        'original',
        'vipe',
    ], f'Unsupported pose_source: {self.pose_source}'
    self.scale_threshold = 20
    # CLIP's context is 77 tokens; truncate text to keep it within budget.
    self.approx_max_chars = 300
    print(
        f'DynPose: encode_pose={self.encode_pose},'
        f' sample_rate={self.sample_rate}, pose_source={self.pose_source}'
    )
    self.build_dataset()

  def build_dataset(self):
    # Test = the 5-way MCQ set (samevideo_test5000); train/val use the full splits.
    fn = 'samevideo_test5000' if self.mode == 'test' else f'{self.mode}_v1'
    df = pd.read_csv(_resolve_csv(f'metadata_{fn}.csv'))
    if self.pose_source == 'vipe':
      df = df[df['vipe_pose_path'].notnull()].reset_index(drop=True)
    col_name = (
        'pose_scale' if self.pose_source == 'original' else 'vipe_pose_scale'
    )
    self.df = df[df[col_name] < self.scale_threshold].reset_index(drop=True)
    print(f'DynPose mode={self.mode}: loaded {len(self.df)}/{len(df)} samples')

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    if self.pose_source == 'original':
      pose_path = os.path.join(
          _DEFAULT_DATA_DIR, self.df.iloc[idx]['pose_path']
      )
      with open(pose_path, 'rb') as f:
        poses = pickle.load(f)['poses']
        poses = w2c_to_c2w(poses)
    else:
      pose_path = os.path.join(
          _DEFAULT_DATA_DIR, self.df.iloc[idx]['vipe_pose_path']
      )
      poses = np.load(pose_path)['data']  # (N, 4, 4)
    poses = matrix_to_pose7d(poses)
    poses = poses[:: self.sample_rate]
    poses = transform_camera_pose(poses, self.encode_pose)
    caption = self.df.iloc[idx]['description_text']
    return poses, caption[: self.approx_max_chars]
