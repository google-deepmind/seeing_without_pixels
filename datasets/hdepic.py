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
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils.dataset_utils import extract_video_frames, poses_to_cam2world

DATA_DIR = os.path.expanduser('~/data/hd-epic_action')


class HdEpicCameraPoseSeq(torch.utils.data.Dataset):

  def __init__(self, args, mode):
    self.args = args
    self.use_label_id = '' if args is None else args.use_label_id
    self.mode = 'train' if mode == 'train' else 'test'
    self.build_dataset()
    if mode == 'test':
      self.load_label_mapping()

  def build_dataset(self):
    fn = '_byparticipant' if self.args.by_participant else ''
    data_file = os.path.join(
        DATA_DIR, f'ann_files/HD_EPIC_Narrations_{self.mode}{fn}.csv'
    )
    df = pd.read_csv(data_file)
    self.data_paths = []
    self.labels = []
    # valid_rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
      camera_pose_path = os.path.join(
          DATA_DIR,
          'camera_motion',
          row['participant_id'],
          row['video_id']
          + f"_start{row['start_timestamp']}_end{row['end_timestamp']}_relative.npy",
      )
      if not os.path.exists(camera_pose_path):
        # print(f"Camera pose path {camera_pose_path} does not exist")
        continue
      main_action_classes = eval(row['main_action_classes'])
      if len(main_action_classes) != 1:
        # print(f"Main action classes {main_action_classes} do not exist")
        continue
      verb_class = main_action_classes[0][0]
      if self.use_label_id != '' and verb_class != int(self.use_label_id):
        continue
      # valid_rows.append(row)
      self.data_paths.append(camera_pose_path)
      self.labels.append(verb_class)
    # df_valid = pd.DataFrame(valid_rows)
    # df_valid.to_csv(os.path.join(DATA_DIR, f'ann_files/HD_EPIC_Narrations_{self.mode}_filtered.csv'), index=False)
    print(
        f'Loaded {len(self.data_paths)} camera motion data from {len(df)} data,'
        f' mode = {self.mode}'
    )

  def load_label_mapping(self):
    df = pd.read_csv(
        os.path.join(DATA_DIR, 'ann_files/HD_EPIC_verb_classes.csv')
    )
    self.label_mapping = {row['id']: row['key'] for _, row in df.iterrows()}

  def __len__(self):
    return len(self.data_paths)

  def __getitem__(self, index):
    camera_pose_path = self.data_paths[index]
    cam_trajectory = np.load(camera_pose_path)
    label = self.labels[index]
    id = os.path.basename(camera_pose_path).replace('_relative.npy', '')
    return cam_trajectory, label, id


class HdEpicVideoAndCameraPoseSeq(
    torch.utils.data.Dataset
):  # for visualization

  def __init__(self):
    self.build_dataset()
    self.load_label_mapping()

  def build_dataset(self):
    data_file = os.path.join(DATA_DIR, 'ann_files/HD_EPIC_Narrations_test.csv')
    df = pd.read_csv(data_file)
    self.data_paths = []
    self.labels = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
      camera_pose_path = os.path.join(
          DATA_DIR,
          'camera_motion',
          row['participant_id'],
          row['video_id']
          + f"_start{row['start_timestamp']}_end{row['end_timestamp']}.csv",
      )
      video_path = camera_pose_path.replace('camera_motion', 'clips').replace(
          '.csv', '.mp4'
      )
      # if not os.path.exists(video_path) or not os.path.exists(camera_pose_path):
      #     continue
      main_action_classes = eval(row['main_action_classes'])
      if len(main_action_classes) != 1:
        continue
      # if 'P08-20240614-085000_start468.15_end469.11' not in video_path:
      #     continue
      self.data_paths.append((video_path, camera_pose_path))
      self.labels.append(main_action_classes[0][0])
    print(f'Loaded {len(self.data_paths)} video and camera motion data')

  def load_label_mapping(self):
    df = pd.read_csv(
        os.path.join(DATA_DIR, 'ann_files/HD_EPIC_verb_classes.csv')
    )
    self.label_mapping = {row['id']: row['key'] for _, row in df.iterrows()}

  def __len__(self):
    return len(self.data_paths)

  def __getitem__(self, index):
    video_path, cache_path = self.data_paths[index]
    cam_trajectory = pd.read_csv(cache_path)[[
        'tx_world_device',
        'ty_world_device',
        'tz_world_device',
        'qx_world_device',
        'qy_world_device',
        'qz_world_device',
        'qw_world_device',
    ]].values
    cam2world = poses_to_cam2world(cam_trajectory)  # shape: (T, 3, 4)
    # cam2world = cam2world[::100]
    n_frames = cam2world.shape[0]
    frames = extract_video_frames(video_path, n_frames)  # shape: (T, H, W, 3)
    id = os.path.basename(cache_path).replace('.csv', '')
    label = self.labels[index]  # int
    return frames, cam2world, id, label, self.label_mapping[label]


if __name__ == '__main__':
  # dataset = HdEpicCameraPoseSeq(None, 'train')
  dataset = HdEpicCameraPoseSeq(None, 'test')
  # for i in range(5):
  # data = dataset[i]
