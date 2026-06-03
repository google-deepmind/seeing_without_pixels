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
import random
import numpy as np
import pandas as pd
import torch
from utils.dataset_utils import matrix_to_pose7d, transform_camera_pose

# Packaged CSV files (train_v2.csv etc.) shipped alongside this module.
_PKG_DATA_FILES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data_files',
    'egoexo4d',
)
# Where the large per-take trajectory bundles live. Overridable via
# EGOEXO4D_PRETRAIN_TRAJ_DIR for users who store them outside the repo.
_DEFAULT_TRAJ_DIR = os.environ.get(
    'EGOEXO4D_PRETRAIN_TRAJ_DIR',
    os.path.expanduser('~/final_data/data/egoexo4d_pretrain'),
)
# Where per-take Pi3 pose .npy files live (one subfolder per take). Overridable
# via EGOEXO4D_PI3_TRAJ_DIR; defaults to the bundled data_files/pi3_preds/.
_DEFAULT_PI3_TRAJ_DIR = os.environ.get(
    'EGOEXO4D_PI3_TRAJ_DIR',
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data_files',
        'pi3_preds',
    ),
)


def _resolve_csv(fn):
  """Prefer the bundled data_files/egoexo4d/<fn>; fall back to ~/final_data/..."""
  local = os.path.join(_PKG_DATA_FILES_DIR, fn)
  if os.path.exists(local):
    return local
  return os.path.expanduser(f'~/final_data/data_files/egoexo4d/{fn}')


class EgoExo4DCameraPoseLongSeqForPretraining(torch.utils.data.Dataset):
  """EgoExo4D camera-pose / text pairs for contrastive pretraining.

  Two pose sources are supported, selected by `args.use_pi3_pose`:

  - Aria GT poses (default, `--use_pi3_pose=False`):
      CSV: data_files/egoexo4d/{train,val}_v2.csv / test_alltasks_mcqv0.csv
      Traj: <EGOEXO4D_PRETRAIN_TRAJ_DIR>/{train,val}_presr50_v2/<take>.npz
            (each npz entry keyed by `save_id`; (N, 7) pose arrays)

  - Pi3 predicted poses (`--use_pi3_pose=True`):
      CSV: data_files/egoexo4d/{train,val}_v2_pi3pose.csv
      Traj: <EGOEXO4D_PI3_TRAJ_DIR>/<take>/partNNNN.npy
            (each .npy holds an (N, 3, 4) camera matrix). The CSV's
            `pose_file` column stores the original relative path; only its
            last two components (<take>/<file>) are used, so the CSV works
            regardless of where the .npy files actually live.

  Take -> scenario labels come from data_files/egoexo4d/scenario_cls.csv
  (its `task` column), removing the dependency on the raw EgoExo4D
  takes.json release file.

  Each sample yields: (cam_traj_slice, [t0, t1, description_text],
                      scenario_label, []).
  """

  def __init__(self, args, mode):
    self.args = args
    self.mode = mode

    self.use_pi3_pose = False if args is None else args.use_pi3_pose
    self.pre_sample_rate = 50 if args is None else args.pre_sample_rate
    self.sample_rate = 1 if args is None else args.sample_rate
    # Pi3 poses are stored at 5 Hz; Aria default is 1e3 / pre_sample_rate.
    self.sample_fps = 5 if self.use_pi3_pose else 1e3 / self.pre_sample_rate
    self.encode_pose = 11 if args is None else args.encode_pose
    self.ego_visible = False if args is None else args.ego_visible
    self.sample_dur = True if args is None else args.sample_dur
    self.scenario = 'all' if args is None else args.scenario
    self.take_duration = 4 if args is None else args.take_duration
    self.test_take_duration = 4 if args is None else args.test_take_duration
    self.dur_points = int(self.take_duration / 2 * self.sample_fps)
    self.test_dur_points = int(self.test_take_duration / 2 * self.sample_fps)
    self.start_ratio = 0.2 if args is None else args.start_ratio
    self.test_time_ratio = 0.5 if args is None else args.test_time_ratio

    mode_key = 'train' if mode == 'train' else 'val'
    self.cache_dir = os.path.join(
        _DEFAULT_TRAJ_DIR,
        f'{mode_key}_presr{self.pre_sample_rate}_v2',
    )
    self.pi3_dir = _DEFAULT_PI3_TRAJ_DIR

    self.load_take_mapping()
    self.build_dataset()

  def load_take_mapping(self):
    self.name2label = {
        'Basketball': 0,
        'Dance': 1,
        'Music': 2,
        'Soccer': 3,
        'Cooking': 4,
        'Rock Climbing': 5,
        'Health': 6,
        'Bike Repair': 7,
    }
    scenario_csv = _resolve_csv('scenario_cls.csv')
    scenario_df = pd.read_csv(scenario_csv)
    self.take_mapping = {
        row['take_name']: self.name2label[row['task']]
        for _, row in scenario_df.iterrows()
    }

  def build_dataset(self):
    if self.mode == 'test':
      # Pi3 pose doesn't ship a test CSV; fall back to the Aria test set.
      if self.scenario == 'all':
        fn = 'test_alltasks_mcqv0.csv'
      else:
        fn = (
            f'test_{self.scenario}_egovisible{self.ego_visible}_mcqv0.csv'
            if self.scenario != ''
            else f'test_egovisible{self.ego_visible}_v2.csv'
        )
    else:
      fn = (
          f'{self.mode}_v2_pi3pose.csv'
          if self.use_pi3_pose
          else f'{self.mode}_v2.csv'
      )

    data_file = _resolve_csv(fn)
    self.df = pd.read_csv(data_file)

    if self.use_pi3_pose and self.mode != 'test':
      # pi3 CSVs use a `pose_file` path in place of save_id; the
      # take_name is the second-to-last path component.
      self.df['take_name'] = self.df['pose_file'].apply(
          lambda x: x.split('/')[-2]
      )
    else:
      self.df['take_name'] = self.df['save_id'].apply(
          lambda x: '_'.join(x.split('_')[:-2])
      )

    # Drop any rows whose take isn't in the scenario label mapping.
    missing = self.df[~self.df['take_name'].isin(self.take_mapping)]
    if len(missing) > 0:
      print(
          f'[EgoExo4DCameraPoseLongSeqForPretraining] dropping {len(missing)}'
          ' rows '
          f"for {missing['take_name'].nunique()} take(s) missing from"
          ' scenario_cls.csv'
      )
      self.df = self.df[
          self.df['take_name'].isin(self.take_mapping)
      ].reset_index(drop=True)

  def _load_cam_traj(self, row):
    """Return the full (N, 7) trajectory for this row."""
    if self.use_pi3_pose and self.mode != 'test':
      # Join the last two components of CSV's pose_file with the user-
      # configurable pi3 root (EGOEXO4D_PI3_TRAJ_DIR).
      rel_path = os.path.join(*row['pose_file'].split('/')[-2:])
      cam = np.load(os.path.join(self.pi3_dir, rel_path))
      return matrix_to_pose7d(cam)
    file_path = os.path.join(self.cache_dir, row['take_name'] + '.npz')
    return np.load(file_path)[row['save_id']]

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    row = self.df.iloc[index]
    cam_traj = self._load_cam_traj(row)

    if (
        self.mode == 'test' and self.test_take_duration == 0
    ) or self.dur_points == 0:
      random_start = max(0, row['t0_idx'])
      random_end = min(row['t1_idx'], row['cam_traj_len'])
    else:
      if self.sample_dur:
        bound = min((row['t1_idx'] - row['t0_idx']) // 2, self.dur_points)
        dur_points = (
            self.test_dur_points
            if self.mode == 'test'
            else random.randint(bound, self.dur_points)
        )
      else:
        dur_points = self.dur_points
      ratio = (
          self.test_time_ratio
          if self.mode == 'test'
          else random.uniform(self.start_ratio, 1 - self.start_ratio)
      )
      random_start = max(0, int(row['timestamp_idx'] - dur_points * ratio))
      random_start = min(random_start, row['t0_idx'])
      random_end = min(random_start + dur_points * 2, row['cam_traj_len'])

    cam_traj_slice = cam_traj[random_start:random_end]
    cam_traj_slice = cam_traj_slice[:: self.sample_rate]
    if len(cam_traj_slice) == 0:
      print(
          f'Invalid indices {index}, cam_traj_slice.shape:'
          f' {cam_traj_slice.shape}'
      )
      return self.__getitem__(random.randint(0, len(self) - 1))

    cam_traj_slice = transform_camera_pose(cam_traj_slice, self.encode_pose)

    t0 = (row['t0_idx'] - random_start) // self.sample_rate
    t1 = (row['t1_idx'] - random_start) // self.sample_rate
    t1 = max(t1, t0 + 1)
    t1 = min(t1, len(cam_traj_slice))

    if not 0 <= t0 < t1 <= len(cam_traj_slice):
      print(
          f'Invalid indices {index}, t0: {t0}, t1: {t1}, cam_traj.shape:'
          f' {cam_traj_slice.shape}'
      )
      return self.__getitem__(random.randint(0, len(self) - 1))

    scenario_label = self.take_mapping[row['take_name']]
    return (
        cam_traj_slice,
        [int(t0), int(t1), row['description_text']],
        scenario_label,
        [],
    )
