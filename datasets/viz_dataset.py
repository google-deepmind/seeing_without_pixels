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

import glob
import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
from utils.dataset_utils import absolute_to_relative, custom_to_opencv_c2w, extract_video_frames, matrix_to_pose7d, poses_to_cam2world

log_dir = 'tmp/cam_viz'


class VizCamMotionTextPair(torch.utils.data.Dataset):

  def __init__(self):
    self.build_dataset()
    self.approx_max_chars = 300  # CLIP's context length is 77 tokens, so we need to truncate the text to 300 characters

  def build_dataset(self):
    self.data_list = []
    self.text_list = []
    for dataset in ['nymeria_full', 'egoexo4d_full']:
      file = f'{log_dir}/{dataset}/caption.csv'
      caption_list = pd.read_csv(file, header=None)[0].tolist()
      self.text_list.extend(caption_list)
      print(f'Loading {len(caption_list)} rows from {dataset}')
      for i in range(len(caption_list)):
        cam_traj = np.load(f'{log_dir}/{dataset}/{i}.npy')
        self.data_list.append(cam_traj)
    print(f'Loading {len(self.data_list)} {len(self.text_list)} in total')

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    return self.data_list[idx], self.text_list[idx][: self.approx_max_chars]


class VizVideoAndMegaSAMPose(torch.utils.data.Dataset):

  def __init__(self, scale_transform=False, rel_pose=False):
    self.input_dir = (  #'baselines/mega-sam/data/DAVIS'
        'data/egoexo4d/scenario/frames_5fps'
    )
    self.megasam_output_dir = 'baselines/mega-sam/outputs_cvd'
    self.gt_campose_dir = os.path.expanduser('~/local_data/egoexo4d')
    self.scale_transform = scale_transform
    self.rel_pose = rel_pose
    self.sample_fps = 20
    self.build_dataset()

  def build_dataset(self):
    self.df = pd.read_csv(
        'data/egoexo4d/annotations/downstream/scenario_cls.csv'
    )
    print(f'Loading {len(self.df)} rows from dataset')

  def _load_cam_estimated(self, path):
    if self.scale_transform:
      cam_c2w = np.load(path.replace('outputs', 'local_outputs_new'))[
          'cam_c2w_scaled'
      ]
      cam_c2w = custom_to_opencv_c2w(cam_c2w)
    else:
      cam_c2w = np.load(path)['cam_c2w']
    cam_traj = matrix_to_pose7d(cam_c2w)
    if self.rel_pose:
      cam_traj = absolute_to_relative(cam_traj, ref_frame_idx=0)  # N, 7
    cam_c2w = poses_to_cam2world(cam_traj)  # N, 3, 4
    return cam_c2w

  def _load_cam_gt(self, row, n_points, sample_points=80):
    file_path = os.path.join(
        self.gt_campose_dir,
        'takes',
        row['take_name'],
        'trajectory_presr50/closed_loop_trajectory.csv',
    )
    trajectory_df = pd.read_csv(file_path)
    cam_gt = trajectory_df[[
        'tx_world_device',
        'ty_world_device',
        'tz_world_device',
        'qx_world_device',
        'qy_world_device',
        'qz_world_device',
        'qw_world_device',
    ]].values
    center = cam_gt.shape[0] // 2
    start = max(0, center - sample_points // 2)
    cam_gt = cam_gt[start : start + sample_points]

    cam_gt = poses_to_cam2world(cam_gt)
    cam_gt = custom_to_opencv_c2w(cam_gt)
    idxs = np.linspace(0, len(cam_gt) - 1, n_points, dtype=int)
    cam_gt = cam_gt[idxs]
    cam_gt = matrix_to_pose7d(cam_gt)
    if self.rel_pose:
      cam_gt = absolute_to_relative(cam_gt, ref_frame_idx=0)
    cam_gt = poses_to_cam2world(cam_gt)
    return cam_gt

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    seq = row['take_name']

    frames_dir = os.path.join(self.input_dir, seq)
    frame_files = sorted(glob.glob(f'{frames_dir}/*.png'))
    frames = [
        cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in frame_files
    ]
    frames = np.stack(frames, axis=0)

    path = os.path.join(
        'baselines/mega-sam/outputs_cvd', f"{row['take_name']}_sgd_cvd_hr.npz"
    )
    cam_est = self._load_cam_estimated(path)
    cam_gt = self._load_cam_gt(row, n_points=len(frames))

    return frames, cam_gt, cam_est, seq


class VizVideoAndVipePose(VizVideoAndMegaSAMPose):

  def __init__(self, *args, **kwargs):
    self.vipe_output_dir = 'baselines/vipe/vipe_results/pose'
    super().__init__(*args, **kwargs)

  def build_dataset(self):
    df = pd.read_csv('data/egoexo4d/annotations/downstream/scenario_cls.csv')
    for i, row in df.iterrows():
      file_path = os.path.join(self.vipe_output_dir, f"{row['take_name']}.npz")
      if not os.path.exists(file_path):
        df = df.drop(i)
    self.df = df.reset_index(drop=True)
    print(f'Loading {len(self.df)} rows from dataset')

  def _load_cam_estimated(self, path, n_points=None):
    cam_c2w = np.load(path)['data']
    cam_traj = matrix_to_pose7d(cam_c2w)
    if n_points is not None:
      idxs = np.linspace(0, len(cam_traj) - 1, n_points, dtype=int)
      cam_traj = cam_traj[idxs]
    cam_traj = absolute_to_relative(cam_traj, ref_frame_idx=0)  # N, 7
    cam_estimated = poses_to_cam2world(cam_traj)  # N, 3, 4
    return cam_estimated

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    seq = row['take_name']

    frames_dir = os.path.join(self.input_dir, seq)
    frame_files = sorted(glob.glob(f'{frames_dir}/*.png'))
    frames = [
        cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in frame_files
    ]
    frames = np.stack(frames, axis=0)

    path = os.path.join(self.vipe_output_dir, f"{row['take_name']}.npz")
    cam_est = self._load_cam_estimated(path, n_points=len(frames))
    cam_gt = self._load_cam_gt(row, n_points=len(frames))
    return frames, cam_gt, cam_est, seq


class VizVipeResultDir(VizVideoAndVipePose):

  def __init__(self):
    self.result_dir = 'baselines/vipe/vipe_results2/'
    self.build_dataset()

  def build_dataset(self):
    self.file_list = sorted(glob.glob(f'{self.result_dir}/pose/*.npz'))
    print(f'Loading {len(self.file_list)} files from {self.result_dir}')

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, idx):
    path = self.file_list[idx]
    seq = os.path.basename(path).replace('.npz', '')
    cam_est = self._load_cam_estimated(path)

    video_path = path.replace('/pose/', '/rgb/').replace('.npz', '.mp4')
    assert os.path.exists(video_path), f'{video_path} not exists'
    frames = extract_video_frames(video_path, len(cam_est))
    return frames, cam_est, seq


class VizVideoAndPi3Pose(VizVideoAndMegaSAMPose):

  def __init__(self, *args, **kwargs):
    self.pi3_output_dir = 'baselines/Pi3/preds'
    super().__init__(*args, **kwargs)

  def build_dataset(self):
    df = pd.read_csv('data/egoexo4d/annotations/downstream/scenario_cls.csv')
    for i, row in df.iterrows():
      file_path = os.path.join(self.pi3_output_dir, f"{row['take_name']}.npy")
      if not os.path.exists(file_path):
        df = df.drop(i)
    self.df = df.reset_index(drop=True)
    print(f'Loading {len(self.df)} rows from dataset')

  def _load_cam_estimated(self, path, n_points=None):
    cam_c2w = np.load(path)
    cam_traj = matrix_to_pose7d(cam_c2w)
    if n_points is not None:
      idxs = np.linspace(0, len(cam_traj) - 1, n_points, dtype=int)
      cam_traj = cam_traj[idxs]
    cam_traj = absolute_to_relative(cam_traj, ref_frame_idx=0)  # N, 7
    cam_estimated = poses_to_cam2world(cam_traj)  # N, 3, 4
    return cam_estimated

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    seq = row['take_name']

    frames_dir = os.path.join(self.input_dir, seq)
    frame_files = sorted(glob.glob(f'{frames_dir}/*.png'))
    frames = [
        cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in frame_files
    ]
    frames = np.stack(frames, axis=0)

    path = os.path.join(self.pi3_output_dir, f"{row['take_name']}.npy")
    cam_est = self._load_cam_estimated(path, n_points=len(frames))
    cam_gt = self._load_cam_gt(row, n_points=len(frames))
    return frames, cam_gt, cam_est, seq


if __name__ == '__main__':
  dataset = VizVideoAndPi3Pose()
  dataset[0]
