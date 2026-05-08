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
import pickle
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils.dataset_utils import absolute_to_relative, absolute_to_relative_from_t0, custom_to_opencv_c2w, extract_video_frames, matrix_to_pose7d, opencv_to_custom_c2w, pose7_to_pose9, poses_to_cam2world, transform_camera_pose, w2c_to_c2w, w2c_to_c2w
from utils.pose_est import align_pose_and_compute_metrics

DATA_DIR = os.path.expanduser('~/data/dynpose-100k/')
LOCAL_DATA_DIR = os.path.expanduser('~/local_data/dynpose-100k/')
FINAL_DATA_DIR = os.path.expanduser('~/final_data/')


class DynPoseCamTextPair(torch.utils.data.Dataset):

  def __init__(self, args, mode):
    self.args = args
    self.encode_pose = args.encode_pose if args is not None else 2
    self.sample_rate = args.sample_rate if args is not None else 1
    print(
        f'Using encode_pose = {self.encode_pose}, sample_rate ='
        f' {self.sample_rate}'
    )
    self.eval_data = (
        'val' if args is None else args.eval_data
    )  #'samevideo' , 'val
    # assert self.args.eval_data in ['randvideo', 'samevideo', 'val']
    self.mode = mode
    self.method = 'vipe' if args is None else args.method
    assert self.method in ['gt', 'vipe']
    self.scale_threshold = 20
    self.approx_max_chars = 300  # CLIP's context length is 77 tokens, so we need to truncate the text to 300 characters
    self.build_dataset()

  def build_dataset(self):
    fn = (
        f'{self.eval_data}_test5000'
        if self.mode == 'test'
        else f'{self.mode}_v1'
    )
    fn = (
        'val_v1' if self.mode == 'test' and self.eval_data == 'val' else fn
    )  # val_v1 samevideo_test5000'
    # df = pd.read_csv(f"{DATA_DIR}/dynpose_100k/metadata_{fn}.csv")
    df = pd.read_csv(
        f'{FINAL_DATA_DIR}/data_files/dynpose100k/metadata_{fn}.csv'
    )
    if self.method == 'vipe':
      df = df[df['vipe_pose_path'].notnull()].reset_index(drop=True)
    col_name = 'pose_scale' if self.method == 'gt' else 'vipe_pose_scale'
    self.df = df[df[col_name] < self.scale_threshold].reset_index(drop=True)
    print(f'Mode = {self.mode}, loaded {len(self.df)}/{len(df)} samples')

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    if self.method == 'gt':
      pose_path = os.path.join(FINAL_DATA_DIR, self.df.iloc[idx]['pose_path'])
      with open(pose_path, 'rb') as f:
        poses = pickle.load(f)['poses']
        poses = w2c_to_c2w(poses)
    else:
      pose_path = os.path.join(
          FINAL_DATA_DIR, self.df.iloc[idx]['vipe_pose_path']
      )
      poses = np.load(pose_path)['data']  # (N, 4, 4)
    poses = matrix_to_pose7d(poses)
    poses = poses[:: self.sample_rate]
    poses = transform_camera_pose(poses, self.encode_pose)
    caption = self.df.iloc[idx]['description_text']
    return poses, caption[: self.approx_max_chars]


class DynPoseVideoAndCamTextPair(DynPoseCamTextPair):

  def __init__(self):
    super().__init__(None, 'test')

  def _resize_sequence(self, poses, num_points):
    idx = np.linspace(0, poses.shape[0] - 1, num_points).astype(int)
    poses = poses[idx]
    return poses

  def _transform_pose(self, poses):
    poses = matrix_to_pose7d(poses)
    poses = absolute_to_relative(poses, ref_frame_idx=0)
    poses = poses_to_cam2world(poses)
    return poses

  def __getitem__(self, idx):
    pose_path = self.df.iloc[idx]['pose_path']
    with open(pose_path, 'rb') as f:
      poses = pickle.load(f)['poses']
      poses = w2c_to_c2w(poses)

    vipe_pose_path = self.df.iloc[idx]['vipe_pose_path']
    vipe_poses = np.load(vipe_pose_path)['data'][:, :3, :]  # (N, 3, 4)

    n_points = min(poses.shape[0], vipe_poses.shape[0])
    poses = self._resize_sequence(poses, n_points)
    vipe_poses = self._resize_sequence(vipe_poses, n_points)

    vipe_poses = align_pose_and_compute_metrics(
        poses, vipe_poses, return_aligned=True
    )
    poses = self._transform_pose(poses)
    vipe_poses = self._transform_pose(vipe_poses)

    vid_path = self.df.iloc[idx]['video_path']
    frames = extract_video_frames(vid_path, n_points)

    poses = poses if self.method == 'gt' else vipe_poses
    print(frames.shape, poses.shape, vipe_poses.shape)
    return (
        frames,
        poses,
        self.df.iloc[idx]['description_text'].replace(' ', '_'),
        '',
        '',
    )
    # return frames, poses, vipe_poses, self.df.iloc[idx]['description_text']


def save_data_for_viz(save_dir):
  os.makedirs(save_dir, exist_ok=True)
  dataset = DynPoseCamTextPair(None, 'test', transform=True)
  text_list = []
  for i in tqdm(range(len(dataset))):
    cam_traj, caption = dataset[i]
    text_list.append(caption)
    np.save(f'{save_dir}/{i}.npy', cam_traj)
  pd.Series(text_list).to_csv(
      f'{save_dir}/caption.csv', index=False, header=False
  )


def save_campose_data():
  save_dir = 'data/misc/retrieval_features/ours/dynpose_pretrain/val_v1_vipe'

  def preprocess_data(x, T0):
    T = x.shape[0]
    if T == T0:
      return x
    elif T > T0:
      idx = np.linspace(0, T - 1, T0).astype(int)
      return x[idx]
    else:
      pad = np.zeros((T0 - T, x.shape[1]), dtype=x.dtype)
      return np.concatenate([x, pad], axis=0)

  dataset = DynPoseCamTextPair(None, 'test')
  all_trajs = []
  for i in tqdm(range(len(dataset)), total=len(dataset)):
    cam_traj, caption = dataset[i]
    cam_traj = preprocess_data(cam_traj, 80)
    all_trajs.append(cam_traj.reshape(-1))

  all_trajs = np.stack(all_trajs)
  print(all_trajs.shape)  # (len(dataset), T0*9)
  np.save(f'{save_dir}/input_cam.npy', all_trajs)


if __name__ == '__main__':
  # save_campose_data()
  dataset = DynPoseCamTextPair(None, 'val')
  print(len(dataset))
  # avg_len = []
  # for i in tqdm(range(len(dataset))):
  #     idx = i
  #     # idx = random.randint(0, len(dataset)-1)
  #     cam_traj, caption = dataset[idx]
  #     # print(f"Sample {i}, idx={idx}, cam_traj.shape={cam_traj.shape}, caption={caption}")
  #     avg_len.append(cam_traj.shape[0])
  # print(f"Avg len = {np.mean(avg_len)}, min = {np.min(avg_len)}, max = {np.max(avg_len)}")
