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
from utils.dataset_utils import absolute_to_relative, extract_video_frames, matrix_to_pose7d, pose7_to_pose9, poses_to_cam2world, transform_camera_pose
from utils.pose_est import align_pose_and_compute_metrics


class UCF101CameraPoseSeq(torch.utils.data.Dataset):

  def __init__(self, args, mode):
    self.DATA_DIR = os.path.expanduser('~/data/UCF101-ZIP')
    self.LOCAL_DATA_DIR = os.path.expanduser('~/local_data/ucf101')

    self.args = args
    self.version = self.args.ucf_version
    self.method = 'megasam' if args is None else self.args.method
    self.split = '01' if args is None else args.split
    self.mode = 'train' if mode == 'train' else 'test'
    self.build_dataset()

  def load_label_mapping(self):
    if self.version in [2, 3]:
      self.label_mapping = {
          'Skijet': 0,
          'SkateBoarding': 1,
          'IceDancing': 2,
          'Rafting': 3,
          'SkyDiving': 4,
          'LongJump': 5,
          'Biking': 6,
          'Skiing': 7,
          'Kayaking': 8,
          'JavelinThrow': 9,
      }
    elif self.version == 4:
      self.label_mapping = {
          'Skijet': 0,
          'SkateBoarding': 1,
          'Knitting': 2,
          'MoppingFloor': 3,
          'WalkingWithDog': 4,
          'Lunges': 5,
          'MilitaryParade': 6,
          'SoccerPenalty': 7,
      }
    else:
      with open(f'{self.DATA_DIR}/ucfTrainTestlist/classInd.txt', 'r') as f:
        self.label_mapping = {}
        self.label_name_mapping = {}
        for line in f:
          line = line.strip().split(' ')
          self.label_mapping[line[1]] = int(line[0]) - 1  # make it 0-indexed
          self.label_name_mapping[int(line[0]) - 1] = line[1]

  def build_dataset(self):
    self.load_label_mapping()
    self.data_paths, self.label_list = [], []
    with open(
        f'{self.DATA_DIR}/ucfTrainTestlist/{self.mode}list{self.split}_filtered_v{self.version}.txt',
        'r',
    ) as f:
      for line in f:
        line = line.strip().split(' ')
        if self.method == 'pi3':
          postscript = '.npy'
        elif self.method == 'vipe':
          postscript = '.npz'
        else:
          postscript = '_droid.npz'
        pose_path = os.path.join(
            self.LOCAL_DATA_DIR,
            f'{self.method}_pose',
            line[0].split('/')[-1].replace('.avi', postscript),
        )
        if not os.path.exists(pose_path):
          # print(f"Missing file: {pose_path}")
          continue
        self.data_paths.append(pose_path)
        self.label_list.append(
            self.label_mapping[line[0].split('/')[0]]
        )  # labels are 1-indexed
    print(
        f'Mode = {self.mode}, loaded {len(self.data_paths)} samples, label'
        f' range = [{min(self.label_list)}, {max(self.label_list)}]'
    )

  def __len__(self):
    return len(self.data_paths)

  def __getitem__(self, idx):
    pose_path = self.data_paths[idx]
    if self.method == 'pi3':
      poses = np.load(pose_path)
      poses = poses[::5]
    elif self.method == 'vipe':
      poses = np.load(pose_path)['data']  # (N, 4, 4)
      poses = poses[::5]
    else:
      poses = np.load(pose_path)['cam_c2w']  # (N, 4, 4)
    poses = matrix_to_pose7d(poses)
    poses = transform_camera_pose(poses, encode_pose=2)
    label = self.label_list[idx]
    return (poses, label, idx)


class UCF101VideoAndCamPoseSeq(torch.utils.data.Dataset):

  def __init__(self, split, method='pi3'):
    self.DATA_DIR = os.path.expanduser('~/data/UCF101-ZIP')
    self.LOCAL_DATA_DIR = os.path.expanduser('~/local_data/ucf101')
    self.method = method
    self.split = split
    self.mode = 'train'
    self.build_dataset()

  def build_dataset(self):
    self.data_paths = []
    with open(
        f'{self.DATA_DIR}/ucfTrainTestlist/{self.mode}list{self.split}.txt', 'r'
    ) as f:
      for line in f:
        line = line.strip().split(' ')
        # if 'v_BenchPress_g01_c01' not in line[0]:
        #     continue
        label = line[0].split('/')[0]
        if label not in ['BenchPress']:
          continue
        vipe_pose_path = os.path.join(
            self.LOCAL_DATA_DIR,
            'vipe_pose',
            line[0].split('/')[-1].replace('.avi', '.npz'),
        )
        pi3_pose_path = os.path.join(
            self.LOCAL_DATA_DIR,
            'pi3_pose',
            line[0].split('/')[-1].replace('.avi', '.npy'),
        )
        megasam_pose_path = os.path.join(
            self.LOCAL_DATA_DIR,
            'megasam_pose',
            line[0].split('/')[-1].replace('.avi', '_droid.npz'),
        )
        video_path = os.path.join(
            self.LOCAL_DATA_DIR,
            'mp4_videos',
            line[0].split('/')[-1].replace('.avi', '.mp4'),
        )
        if (
            not os.path.exists(pi3_pose_path)
            or not os.path.exists(video_path)
            or not os.path.exists(vipe_pose_path)
            or not os.path.exists(megasam_pose_path)
        ):
          continue
        self.data_paths.append((
            video_path,
            vipe_pose_path,
            pi3_pose_path,
            megasam_pose_path,
            line[0].split('/')[-1].split('.')[0],
        ))  # (video_path, pose_path, label)
    print(f'Mode = {self.mode}, loaded {len(self.data_paths)} samples')

  def __len__(self):
    return len(self.data_paths)

  def _transform_pose(self, poses):
    poses = matrix_to_pose7d(poses)
    poses = absolute_to_relative(poses, ref_frame_idx=0)
    poses = poses_to_cam2world(poses)
    return poses

  def _resize_sequence(self, poses, target_len):
    if len(poses) == target_len:
      return poses
    idxs = np.linspace(0, len(poses) - 1, target_len, dtype=int)
    return poses[idxs]

  def __getitem__(self, idx):
    video_path, vipe_pose_path, pi3_pose_path, megasam_pose_path, label = (
        self.data_paths[idx]
    )
    pi3_poses = np.load(pi3_pose_path)[:, :3, :]
    vipe_poses = np.load(vipe_pose_path)['data'][:, :3, :]
    megasam_poses = np.load(megasam_pose_path)['cam_c2w'][:, :3, :]

    pi3_poses = self._transform_pose(pi3_poses)
    vipe_poses = self._transform_pose(vipe_poses)
    megasam_poses = self._transform_pose(megasam_poses)

    pose1 = vipe_poses
    pose2 = pi3_poses if self.method == 'pi3' else megasam_poses
    pose1 = self._resize_sequence(pose1, len(pose2))

    ate, *_ = align_pose_and_compute_metrics(pose1, pose2)
    # if ate < 0.3:
    #     return self.__getitem__(random.randint(0, len(self)-1))

    frames = extract_video_frames(video_path, len(pose2))
    pose = pose1 if self.method == 'vipe' else pose2
    return frames, pose1, label, '', ''
    # return frames, pose1, pose2, label + f"_ate{ate:.3f}"


class FineGymCameraPoseSeq(torch.utils.data.Dataset):

  def __init__(self, args, mode):
    self.DATA_DIR = os.path.expanduser('~/data/FineGym')
    self.LOCAL_DATA_DIR = os.path.expanduser('~/local_data/FineGym')
    self.args = args
    self.mode = 'train' if mode == 'train' else 'val'
    self.min_sample_points = 5
    self.sample_dur = True if args is None else args.sample_dur
    self.test_ratio = 0.2 if args is None else args.test_ratio
    self.build_dataset()

  def build_dataset(self):
    self.df = pd.read_csv(
        f'{self.DATA_DIR}/annotations/finegym_{self.mode}_split.csv'
    )
    print(f'Mode = {self.mode}, loaded {len(self.df)} samples')

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    pi3_pose = np.load(row['pi3_path'])[:, :3, :]
    if self.sample_dur:
      ratio = (
          random.uniform(0.0, 1.0) if self.mode == 'train' else self.test_ratio
      )
      target_len = max(int(len(pi3_pose) * ratio), self.min_sample_points)
      start = (
          random.randint(0, max(0, len(pi3_pose) - target_len))
          if self.mode == 'train'
          else (len(pi3_pose) - target_len) // 2
      )
      pi3_pose = pi3_pose[start : start + target_len]
    pose = matrix_to_pose7d(pi3_pose)
    pose = transform_camera_pose(pose, encode_pose=2)
    return (pose, row['event_label'] - 1, idx)


class FineGymVideoAndCamPoseSeq(torch.utils.data.Dataset):

  def __init__(self):
    self.DATA_DIR = os.path.expanduser('~/data/FineGym')
    self.LOCAL_DATA_DIR = os.path.expanduser('~/local_data/FineGym')
    self.mode = 'val'
    self.build_dataset()

  def build_dataset(self):
    self.df = pd.read_csv(
        f'{self.DATA_DIR}/annotations/finegym_{self.mode}_split.csv'
    )
    print(f'Mode = {self.mode}, loaded {len(self.df)} samples')

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    print(row['video'])
    pose = np.load(row['pi3_path'])[:, :3, :]
    pose = matrix_to_pose7d(pose)
    pose = absolute_to_relative(pose, ref_frame_idx=0)
    pose = poses_to_cam2world(pose)
    frames = extract_video_frames(row['video'], len(pose))
    return frames, pose, '', '', ''


class FineGymCameraPoseLongSeq(torch.utils.data.Dataset):

  def __init__(self, args, mode):
    self.DATA_DIR = os.path.expanduser('~/data/FineGym')
    self.LOCAL_DATA_DIR = os.path.expanduser('~/local_data/FineGym')

    self.vipe_pred_dir = 'baselines/vipe/vipe_demo_results/pose'
    self.window_size = (
        args.window_size if args is not None else 120
    )  # 4 seconds at 30fps
    self.window_stride = args.window_stride if args is not None else 10
    self.build_dataset()

  def build_dataset(self):
    df = pd.read_csv(f'{self.DATA_DIR}/annotations/finegym_shots_stat.csv')
    df['fn'] = df['video'].apply(
        lambda x: os.path.basename(x).replace('.mp4', '')
    )
    self.data_list = []
    for i, row in df.iterrows():
      if 'event10' not in row['fn']:
        continue
      vipe_pose_path = os.path.join(self.vipe_pred_dir, row['fn'] + '.npz')
      if not os.path.exists(vipe_pose_path):
        continue
      vipe_pose = np.load(vipe_pose_path)['data']
      for start_idx in range(0, len(vipe_pose), self.window_stride):
        end_idx = min(start_idx + self.window_size, len(vipe_pose))
        cam_pose = vipe_pose[start_idx:end_idx]
        self.data_list.append((cam_pose, row['fn']))

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    poses, fn = self.data_list[idx]
    poses = matrix_to_pose7d(poses)
    poses = absolute_to_relative(poses, ref_frame_idx=len(poses) // 2)
    poses = pose7_to_pose9(poses)
    return poses, fn, (0, len(poses))


if __name__ == '__main__':
  dataset = UCF101CameraPoseSeq(None, 'train')
  for i in range(len(dataset)):
    dataset[i]
