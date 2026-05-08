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
import numpy as np
import torch
from utils.dataset_utils import extract_video_frames, matrix_to_pose7d, transform_camera_pose


class DemoCamTextPair(torch.utils.data.Dataset):

  def __init__(self, args):
    self.DATA_DIR = os.path.expanduser('~/data/demo_video_clips')
    self.encode_pose = args.encode_pose if args is not None else 2
    self.method = args.method if args is not None else 'vipe'
    assert self.method in ['vipe', 'pi3']
    self.build_dataset()

  def build_dataset(self):
    postscript = 'npy' if self.method == 'pi3' else 'npz'
    self.pose_files = sorted(
        glob.glob(f'{self.DATA_DIR}/{self.method}poses_v0/*.{postscript}')
    )
    print(f'Loading {len(self.pose_files)} pose files')

  def __len__(self):
    return len(self.pose_files)

  def __getitem__(self, idx):
    pose_path = self.pose_files[idx]
    if self.method == 'pi3':
      poses = np.load(pose_path)
    else:
      poses = np.load(pose_path)['data']
    poses = matrix_to_pose7d(poses)
    poses = transform_camera_pose(poses, encode_pose=self.encode_pose)
    return poses, ''


def apply_relative_to_absolute(a, b):
  """a, b: (T, 3, 4). b is relative pose (e.g., from frame 0 or accumulated)."""
  out = []
  T_a0 = a[0]
  for i in range(len(b)):
    T_bi = b[i]
    T_out = T_a0 @ T_bi  # compose
    out.append(T_out[:3, :4])
  return np.stack(out, axis=0)  # (T, 3, 4)


class DemoVideoAndCamTextPair(torch.utils.data.Dataset):

  def __init__(self, method, transform=False):
    self.DATA_DIR = os.path.expanduser('~/data/demo_video_clips')
    self.encode_pose = 2
    self.sample_rate = 1
    self.method = method
    self.transform = transform
    assert self.method in ['vipe', 'pi3']
    self.build_dataset()

  def build_dataset(self):
    self.pose_files = sorted(
        glob.glob(f'{self.DATA_DIR}/{self.method}poses_v0/*.npz')
    )
    print(f'Loading {len(self.pose_files)} pose files')

  def __len__(self):
    return len(self.pose_files)

  def __getitem__(self, idx):
    pose_path = self.pose_files[idx]
    pi3_pose_path = pose_path.replace(
        f'{self.method}poses_', 'pi3poses_'
    ).replace('.npz', '.npy')
    video_path = (
        pose_path.replace(f'{self.method}poses_', 'videos_')
        .replace('.npy', '.mp4')
        .replace('.npz', '.mp4')
    )

    vipe_pose = np.load(pose_path)['data']
    pi3_pose = np.load(pi3_pose_path)
    if self.method == 'pi3':
      poses = pi3_pose[:, :3, :]
    else:
      if self.transform:
        poses = apply_relative_to_absolute(pi3_pose, vipe_pose)
      else:
        poses = vipe_pose[:, :3, :]
    poses = poses[:: self.sample_rate]
    frames = extract_video_frames(video_path, poses.shape[0])
    return frames, poses, idx, '', ''


if __name__ == '__main__':
  dataset = DemoVideoAndCamTextPair()
  dataset[6]
