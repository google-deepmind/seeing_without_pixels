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

"""Nymeria camera-motion / text pairs.

In the paper, Nymeria is used for *zero-shot* egocentric text retrieval: the
CamFormer pretrained on Ego-Exo4D is evaluated directly on Nymeria without any
fine-tuning (Table 2). The four narration types (body posture / hands-arms /
legs-feet / focus attention) are selected via `--text_column {a,b,c,d}`.

The test (eval-1000) split CSVs are bundled under data_files/nymeria/eval1000/;
the per-sequence camera-motion .npz bundles live under
<NYMERIA_DATA_DIR>/data/nymeria_eval/ and are resolved at runtime.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from utils.dataset_utils import transform_camera_pose

# Packaged eval CSVs shipped alongside this module (if present).
_PKG_DATA_FILES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data_files',
    'nymeria',
    'eval1000',
)
# Base dir holding `data_files/nymeria/...` and `data/nymeria_eval/...`.
# Overridable via NYMERIA_DATA_DIR; defaults to the public `~/final_data` bundle.
_DEFAULT_DATA_DIR = os.environ.get(
    'NYMERIA_DATA_DIR',
    os.path.expanduser('~/final_data'),
)
# Raw Nymeria release (only needed to (re)build the train split; the zero-shot
# evaluation uses the bundled eval CSVs and does not require this).
_RAW_DATA_DIR = os.environ.get(
    'NYMERIA_RAW_DIR',
    os.path.expanduser('~/data/nymeria'),
)

text_column_mapping = {
    'a': 'Describe my body posture',
    'b': 'Describe my hands/arms motion',
    'c': 'Describe my legs/feet motion',
    'd': 'Describe my focus attention',
}


def _resolve_eval_csv(fn):
  """Prefer the bundled data_files/nymeria/eval1000/<fn>; fall back to bundle."""
  local = os.path.join(_PKG_DATA_FILES_DIR, fn)
  if os.path.exists(local):
    return local
  return os.path.join(
      _DEFAULT_DATA_DIR, 'data_files', 'nymeria', 'eval1000', fn
  )


class NymeriaCamMotionTextPair(torch.utils.data.Dataset):
  """Nymeria camera-motion / narration pairs (zero-shot eval).

  Each sample yields: (cam_motion (N, D), caption).
  """

  def __init__(self, args, mode):
    self.args = args
    self.mode = mode
    self.sample_rate = 1 if args is None else args.sample_rate
    self.text_column = 'a' if args is None else args.text_column
    self.select_column = text_column_mapping[self.text_column]
    self.scenario = '' if args is None else args.scenario
    self.encode_pose = 2 if args is None else args.encode_pose
    # CLIP's context is 77 tokens; truncate text to keep it within budget.
    self.approx_max_chars = 300
    if mode == 'test':
      self.build_test_dataset()
    else:
      self.build_dataset()

  def build_dataset(self):
    """Build the (optional) train/val split from the raw Nymeria release.

    Not needed for the paper's zero-shot evaluation; requires the raw
    Nymeria download at NYMERIA_RAW_DIR.
    """
    df = pd.read_csv(os.path.join(_RAW_DATA_DIR, 'metadata.csv'))
    self.df = df.dropna(subset=[self.select_column])
    with open(os.path.join(_RAW_DATA_DIR, 'dataset_metadata.json'), 'r') as f:
      metadata = json.load(f)
    keys = self.df['motion_file'].str.split('/').str[-1].str.split('.').str[0]
    self.df['script'] = keys.map(lambda key: metadata[key]['script'])
    self.df['scenario'] = self.df['script'].str.split('-').str[0]
    print(f'Nymeria: loaded {len(self.df)}/{len(df)} samples')
    if self.scenario != '':
      self.df = self.df[self.df['scenario'] == self.scenario]
      print(
          f'Nymeria: loaded {len(self.df)} samples (scenario={self.scenario})'
      )

  def build_test_dataset(self):
    key_name = self.select_column.replace('/', '_').replace(' ', '_')
    self.df = pd.read_csv(_resolve_eval_csv(f'split_by_{key_name}.csv'))

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row_idx = self.df.iloc[idx]['row_idx']
    motion_file = os.path.join(
        _DEFAULT_DATA_DIR,
        'data',
        'nymeria_eval',
        self.df.iloc[idx]['motion_file'],
    )
    cam_motion_dict = np.load(motion_file)
    cam_motion = cam_motion_dict[str(row_idx)]
    cam_motion = cam_motion[:: self.sample_rate]
    if len(cam_motion) == 0:
      return self.__getitem__((idx + 1) % len(self))
    cam_motion = transform_camera_pose(cam_motion, self.encode_pose)
    caption = self.df.iloc[idx][self.select_column]
    return cam_motion, caption[: self.approx_max_chars]
