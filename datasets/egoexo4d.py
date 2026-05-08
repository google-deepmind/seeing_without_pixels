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

import json
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dataset_utils import absolute_to_relative, absolute_to_relative_sequential, custom_to_opencv_c2w, extract_video_frames, matrix_to_pose7d, opencv_to_custom_c2w, pose7_to_pose9, poses_to_cam2world, raymap_naive_batch, transform_camera_pose
from utils.pose_est import align_pose_and_compute_metrics

DATA_DIR = os.path.expanduser('~/data/egoexo4d')
LOCAL_DATA_DIR = os.path.expanduser('~/local_data/egoexo4d')
FINAL_DATA_DIR = os.path.expanduser('~/final_data/')


class EgoExo4DCameraPoseSeqForPretraining(torch.utils.data.Dataset):

  def __init__(self, args, mode):
    self.args = args
    self.mode = mode
    self.pre_sample_rate = 50
    self.train_concat_data = False if args is None else args.train_concat_data
    self.use_text_embeds = True if args is None else args.use_text_embeds
    self.encode_pose = 9 if args is None else args.encode_pose

    self.num_negatives = 0 if args is None else args.num_negatives
    self.ego_visible = False if args is None else args.ego_visible
    self.eval_data = 'mcqv0' if args is None else args.eval_data
    self.scenario = 'all' if args is None else args.scenario

    mode = 'train' if self.mode == 'train' else 'val'
    self.cache_dir = os.path.join(
        LOCAL_DATA_DIR,
        'camera_motion_cache/egoexo4d_pretrain',
        mode + '_presr' + str(self.pre_sample_rate),
    )
    self.cache_dir2 = os.path.join(
        LOCAL_DATA_DIR,
        'camera_motion_cache/egoexo4d_pretrain',
        mode + '_presr' + str(self.pre_sample_rate) + '_egoexovlpv2',
    )
    self.raymap_cache_dir = os.path.join(
        DATA_DIR,
        'camera_motion_cache_raymap/egoexo4d_pretrain',
        f'{mode}_presr{self.pre_sample_rate}',
    )
    self.build_dataset()
    if self.num_negatives > 0:
      negative_mapping_file = os.path.join(
          DATA_DIR,
          'annotations/pretraining',
          f'{self.mode}_presr{self.pre_sample_rate}_cleanv1_components_neg_mapping.pkl',
      )
      self.load_negative_mapping(negative_mapping_file)

  def build_dataset(self):
    if self.scenario == 'all' and self.mode == 'test':
      fn = f'sampled500_alltasks_{self.eval_data}'
    else:
      fn = (
          f'sampled1000_{self.ego_visible}_{self.eval_data}'
          if self.mode == 'test'
          else 'cleanv1_components'
      )
    data_file = os.path.join(
        DATA_DIR,
        'annotations/pretraining',
        f'{self.mode}_presr{self.pre_sample_rate}_{fn}.csv',
    )
    text_embeds_fn = (
        os.path.basename(data_file).replace('.csv', '')
        if self.mode == 'test'
        else 'train_val'
    )
    self.text_embeds_dir = os.path.join(
        LOCAL_DATA_DIR, 'text_embeds', text_embeds_fn
    )
    self.df = pd.read_csv(data_file)
    self.df['take_name'] = self.df['save_id'].apply(
        lambda x: '_'.join(x.split('_')[:-2])
    )
    self.df['version'] = 'v1'  # Add version column
    if self.train_concat_data and self.mode == 'train':
      df2 = pd.read_csv(
          os.path.join(
              DATA_DIR,
              'annotations/pretraining',
              f'{self.mode}_presr{self.pre_sample_rate}_egoexovlpv2.csv',
          )
      )
      self.df = pd.concat([self.df, df2])
      print(f'Concatenated {len(df2)} camera motion data')
    print(f'Loaded {len(self.df)} camera motion data ({self.mode})')

  def load_negative_mapping(self, negative_mapping_file):
    with open(negative_mapping_file, 'rb') as f:
      self.negative_mapping = pickle.load(f)
    print(f'Loaded negative mapping from {negative_mapping_file}')

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    row = self.df.iloc[index]
    cache_dir = self.cache_dir if row['version'] == 'v1' else self.cache_dir2
    # if self.encode_pose == 9:
    #     cache_dir = self.raymap_cache_dir
    try:
      cache_path = os.path.join(cache_dir, row['save_id'] + '_absolute.npy')
      trajectory = np.load(cache_path)
      cam_trajectory = transform_camera_pose(trajectory, self.encode_pose)
    except Exception as e:
      print(f'Error loading camera trajectory: {e}')
      return self.__getitem__(random.randint(0, len(self) - 1))

    if self.use_text_embeds:
      if self.mode == 'test':
        text = (
            torch.load(
                os.path.join(self.text_embeds_dir, row['save_id'] + '.pt')
            )
            .squeeze(0)
            .detach()
        )
      else:
        text = torch.load(
            os.path.join(self.text_embeds_dir, row['take_name'] + '.pt')
        )
        take_df = self.df[self.df['take_name'] == row['take_name']]
        take_df_init_idx = take_df.index[0]
        text = text[index - take_df_init_idx].detach()
    else:
      text = row['description_text']

    if self.num_negatives > 0:
      negative_indices = self.negative_mapping[index]
      # Randomly sample num_negatives indices from negative_indices
      sampled_neg_indices = random.sample(
          negative_indices, min(self.num_negatives, len(negative_indices))
      )
      neg_trajectories, neg_texts = [], []
      for neg_idx in sampled_neg_indices:
        neg_row = self.df.iloc[neg_idx]
        neg_cache_dir = (
            self.cache_dir if neg_row['version'] == 'v1' else self.cache_dir2
        )
        neg_trajectory = self._load_camera_trajectory(neg_row, neg_cache_dir)
        neg_trajectories.append(neg_trajectory)

        # Handle negative text based on use_text_embeds
        if self.use_text_embeds:
          neg_take_name = neg_row['take_name']
          neg_text_embeds = torch.load(
              os.path.join(self.text_embeds_dir, neg_take_name + '.pt')
          )
          neg_take_df = self.df[self.df['take_name'] == neg_take_name]
          neg_take_df_init_idx = neg_take_df.index[0]
          neg_text = neg_text_embeds[neg_idx - neg_take_df_init_idx].detach()
        else:
          neg_text = neg_row['description_text']
        neg_texts.append(neg_text)
      return cam_trajectory, text, neg_trajectories, neg_texts
    return cam_trajectory, text


class EgoExo4DVideoAndCameraPoseSeqForPretraining(torch.utils.data.Dataset):

  def __init__(self, ego_visible):
    self.mode = 'val'
    self.pre_sample_rate = 50
    self.ego_visible = ego_visible
    self.cache_dir = os.path.join(
        DATA_DIR,
        'camera_motion_cache/egoexo4d_pretrain',
        self.mode + '_presr' + str(self.pre_sample_rate),
    )
    self.load_mapping()
    self.build_dataset()

  def load_mapping(self):
    self.camera_name_mapping = {}
    data_file = os.path.join(DATA_DIR, 'takes.json')
    with open(data_file, 'r') as f:
      data_list = json.load(f)
    for data in data_list:
      keys = list(data['frame_aligned_videos'].keys())
      self.camera_name_mapping[data['take_name']] = data[
          'frame_aligned_videos'
      ][keys[0]]['rgb']['relative_path']

  def build_dataset(self):
    data_file = os.path.join(
        DATA_DIR,
        'annotations/pretraining',
        f'test_presr{self.pre_sample_rate}_sampled1000_{self.ego_visible}_mcqv0.csv',
    )
    self.df = pd.read_csv(data_file)
    # self.df = df[df['ego_visible'] == self.ego_visible]
    print(
        f'Loaded {len(self.df)} camera motion data ({self.mode},'
        f' ego_visible={self.ego_visible})'
    )

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    row = self.df.iloc[index]
    cache_path = os.path.join(self.cache_dir, row['save_id'] + '_absolute.npy')
    id = row['save_id']
    take_id = '_'.join(id.split('_')[:-2])
    video_path = os.path.join(
        DATA_DIR,
        'takes',
        take_id,
        self.camera_name_mapping[take_id].replace(
            'frame_aligned_videos', 'frame_aligned_videos/downscaled/448/'
        ),
    )  #'frame_aligned_videos/downscaled/448/ego_preview.mp4')
    if not os.path.exists(video_path):
      print(f'Video file not found: {video_path}')
      return
    cam_trajectory = np.load(cache_path)
    cam2world = poses_to_cam2world(cam_trajectory)
    n_frames = cam2world.shape[0]
    text = row['description_text']
    frames = extract_video_frames(
        video_path, n_frames, row['start_time'], row['end_time']
    )
    return (
        frames,
        cam2world,
        text + '_egovisible' + str(self.ego_visible),
        '',
        '',
    )


class EgoExo4DCameraPoseLongSeqForPretraining(torch.utils.data.Dataset):

  def __init__(self, args, mode):
    self.args = args
    self.mode = mode
    self.intrinsics = [
        610.524378190068,
        610.524378190068,
        712.688057683927,
        710.828602211728,
        1408,
        1408,
    ]
    self.raymap_ds_dim = 448
    self.pre_sample_rate = 50 if args is None else args.pre_sample_rate
    self.sample_rate = 1 if args is None else args.sample_rate
    self.use_pi3_pose = False if args is None else args.use_pi3_pose
    self.sample_fps = 5 if self.use_pi3_pose else 1e3 / self.pre_sample_rate
    self.encode_pose = 11 if args is None else args.encode_pose
    self.ego_visible = False if args is None else args.ego_visible
    self.sample_dur = True if args is None else args.sample_dur
    self.hier_text = True if args is None else args.hier_text
    self.eval_data = 'mcqv0' if args is None else args.eval_data
    self.scenario = 'all' if args is None else args.scenario
    self.take_duration = 4 if args is None else args.take_duration
    self.test_take_duration = 4 if args is None else args.test_take_duration
    self.dur_points = int(self.take_duration / 2 * self.sample_fps)
    self.test_dur_points = int(self.test_take_duration / 2 * self.sample_fps)
    self.start_ratio = 0.2 if args is None else args.start_ratio
    self.test_time_ratio = 0.5 if args is None else args.test_time_ratio
    mode = 'train' if mode == 'train' else 'val'
    # self.cache_dir = os.path.join(LOCAL_DATA_DIR, 'camera_motion_cache/egoexo4d_pretrain', f'{mode}_presr{self.pre_sample_rate}_v2')
    self.cache_dir = os.path.join(
        FINAL_DATA_DIR,
        'data/egoexo4d_pretrain',
        f'{mode}_presr{self.pre_sample_rate}_v2',
    )
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
    self.take_mapping = {}
    with open(os.path.join(DATA_DIR, 'takes.json'), 'r') as f:
      data_list = json.load(f)
    for data in data_list:
      self.take_mapping[data['take_name']] = self.name2label[
          data['parent_task_name']
      ]
    # unique_values = list(set(self.take_mapping.values()))
    # print(f"{len(unique_values)} unique values in take_mapping: {unique_values}")

  def build_dataset(self):
    if self.mode == 'test':
      if self.scenario == 'all':
        fn = (
            f'test_alltasks_mcqv0_pi3pose.csv'
            if self.use_pi3_pose
            else f'test_alltasks_mcqv0.csv'
        )
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
    # data_file = os.path.join(DATA_DIR, 'annotations/pretraining', fn)
    data_file = os.path.join(FINAL_DATA_DIR, 'data_files/egoexo4d', fn)
    self.df = pd.read_csv(data_file)
    if self.use_pi3_pose:
      self.df['take_name'] = self.df['pose_file'].apply(
          lambda x: x.split('/')[-2]
      )
    else:
      self.df['take_name'] = self.df['save_id'].apply(
          lambda x: '_'.join(x.split('_')[:-2])
      )

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    row = self.df.iloc[index]
    take_name = row['take_name']
    if self.use_pi3_pose:
      cam = np.load(row['pose_file'])
      cam_traj = matrix_to_pose7d(cam)
    else:
      file_path = os.path.join(self.cache_dir, take_name + '.npz')
      cam_traj = np.load(file_path)[row['save_id']]

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
        # print(f'Sampling duration from {bound} - {self.dur_points}, getting len {dur_points}')
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

    scenario_label = self.take_mapping[take_name]
    return (
        cam_traj_slice,
        [int(t0), int(t1), row['description_text']],
        scenario_label,
        [],
    )


class EgoExo4DVideoAndCameraPoseLongSeq(torch.utils.data.Dataset):

  def __init__(self, take_duration=4, opencv_cord=False):
    self.cache_dir = os.path.join(
        LOCAL_DATA_DIR, 'camera_motion_cache/egoexo4d_pretrain/val_presr50_v2'
    )
    self.take_duration = take_duration
    self.opencv_cord = opencv_cord
    self.sample_fps = 20
    self.dur_points = int(self.take_duration / 2 * self.sample_fps)
    self.build_dataset()
    self.load_mapping()

  def load_mapping(self):
    self.camera_name_mapping = {}
    data_file = os.path.join(DATA_DIR, 'takes.json')
    with open(data_file, 'r') as f:
      data_list = json.load(f)
    for data in data_list:
      keys = list(data['frame_aligned_videos'].keys())
      self.camera_name_mapping[data['take_name']] = data[
          'frame_aligned_videos'
      ][keys[0]]['rgb']['relative_path']

  def build_dataset(self):
    data_file = os.path.join(
        DATA_DIR, 'annotations/pretraining/test_alltasks_mcqv0.csv'
    )
    self.df = pd.read_csv(data_file)

    self.df['take_name'] = self.df['save_id'].apply(
        lambda x: '_'.join(x.split('_')[:-2])
    )
    print(f'Loaded {len(self.df)} camera motion data (test)')

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    row = self.df.iloc[index]
    text = row['description_text']
    # if 'transfer' not in text:
    # if 'cooking' not in row['save_id']:
    #     return self.__getitem__(random.randint(0, len(self)-1))
    take_name = row['take_name']
    file_path = os.path.join(self.cache_dir, take_name + '.npz')
    cam_traj = np.load(file_path)[row['save_id']]
    start = max(0, int(row['timestamp_idx'] - self.dur_points))
    start = min(start, row['t0_idx'])
    end = min(start + self.dur_points * 2, row['cam_traj_len'])
    cam_traj_slice = cam_traj[start:end]
    if self.opencv_cord:
      cam_gt = poses_to_cam2world(cam_traj_slice)
      cam_gt = custom_to_opencv_c2w(cam_gt)
      cam_gt = matrix_to_pose7d(cam_gt)
      cam_traj_slice = absolute_to_relative(cam_gt, ref_frame_idx=0)
    cam2world = poses_to_cam2world(cam_traj_slice)
    n_frames = cam2world.shape[0]

    video_path = os.path.join(
        DATA_DIR,
        'takes',
        take_name,
        self.camera_name_mapping[take_name].replace(
            'frame_aligned_videos', 'frame_aligned_videos/downscaled/448/'
        ),
    )  #'frame_aligned_videos/downscaled/448/ego_preview.mp4')
    start_time = max(0, row['timestamp'] - 0.5 * self.take_duration)
    end_time = start_time + self.take_duration

    if (
        abs(end_time - start_time - self.take_duration) > 1e-3
        or n_frames != self.dur_points * 2
    ):
      print(
          f'Invalid time range for {index}, start_time: {start_time}, end_time:'
          f' {end_time}, start idx {start}, end idx {end}, n_frames: {n_frames}'
      )
      return self.__getitem__(random.randint(0, len(self) - 1))

    frames = extract_video_frames(video_path, n_frames, start_time, end_time)

    return frames, cam2world, text, index, ''


class EgoExo4DCameraPoseSeqForScenarioCls(torch.utils.data.Dataset):

  def __init__(self, args, mode):
    self.args = args
    self.mode = mode
    self.sample_fps = 20
    self.encode_pose = 11 if args is None else args.encode_pose
    self.sample_rate = 1 if args is None else args.sample_rate
    print(
        f'Using encode_pose = {self.encode_pose}, sample_rate ='
        f' {self.sample_rate}'
    )
    self.sample_dur = True if args is None else args.sample_dur
    self.take_duration = 4 if args is None else args.take_duration
    self.test_take_duration = 4 if args is None else args.test_take_duration
    self.min_sample_points = int(self.sample_fps * 0.5)  # min: 0.5s
    self.sample_points = int(self.sample_fps * self.take_duration)
    self.test_sample_points = int(self.sample_fps * self.test_take_duration)
    self.num_test_clips = 10 if args is None else args.num_test_clips
    self.build_dataset()

  def build_dataset(self):
    # df = pd.read_csv(f"{DATA_DIR}/annotations/downstream/scenario_cls.csv")
    df = pd.read_csv(f'{FINAL_DATA_DIR}/data_files/egoexo4d/scenario_cls.csv')
    self.df = df[df['split'] == self.mode]
    print(f'Loading {len(self.df)}/{len(df)} data (mode = {self.mode})')
    self.label_mapping = df.set_index('label')['task'].to_dict()

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    row = self.df.iloc[index]
    # file_path = os.path.join(LOCAL_DATA_DIR, 'takes', row['take_name'], 'trajectory_presr50/closed_loop_trajectory.csv')
    file_path = os.path.join(
        FINAL_DATA_DIR,
        'data/egoexo4d_downstream/trajectory_presr50',
        row['take_name'],
        'closed_loop_trajectory.csv',
    )
    trajectory_df = pd.read_csv(file_path)
    cam_trajectory = trajectory_df[[
        'tx_world_device',
        'ty_world_device',
        'tz_world_device',
        'qx_world_device',
        'qy_world_device',
        'qz_world_device',
        'qw_world_device',
    ]].values

    if self.mode == 'train':
      sample_points = (
          random.randint(self.min_sample_points, self.sample_points)
          if self.sample_dur
          else self.sample_points
      )
    else:
      sample_points = (
          self.test_sample_points if self.sample_dur else self.sample_points
      )

    if self.mode == 'train':
      start = [
          random.randint(0, max(cam_trajectory.shape[0] - sample_points, 0))
      ]
    else:
      if self.num_test_clips > 1 and len(cam_trajectory) > sample_points:
        start = np.linspace(
            0,
            len(cam_trajectory) - sample_points,
            self.num_test_clips,
            dtype=int,
        )
      else:
        center = cam_trajectory.shape[0] // 2
        start = [max(0, center - sample_points // 2)]
    clips = []
    for st in start:
      sampled_cam_trajectory = cam_trajectory[st : st + sample_points]
      sampled_cam_trajectory = sampled_cam_trajectory[:: self.sample_rate]
      sampled_cam_trajectory = transform_camera_pose(
          sampled_cam_trajectory, self.encode_pose
      )
      clips.append((sampled_cam_trajectory, row['label'], index))
    return clips


class EgoExo4DCameraPoseSeqForScenarioClsSubset(torch.utils.data.Dataset):

  def __init__(self, args, mode):
    self.mode = mode
    self.encode_pose = args.encode_pose
    self.method = 'megasam' if args is None else args.method
    assert self.method in ['gt', 'megasam', 'vipe', 'pi3', 'd4rt']
    self.umeyama_transform = False if args is None else args.umeyama_transform
    self.build_dataset()

  def build_dataset(self):
    df = pd.read_csv(
        f'{DATA_DIR}/annotations/downstream/scenario_cls_with_all3_pred.csv'
    )
    # df = pd.read_csv(f"{FINAL_DATA_DIR}/data_files/egoexo4d/scenario_cls_with_all3_pred.csv")
    # data = np.load(f"{FINAL_DATA_DIR}/data/egoexo4d_downstream/cam_est_cache/scenario_cls_with_all3_pred_transformed.npz", allow_pickle=True)
    data = np.load(
        f'{DATA_DIR}/cam_est_cache/scenario_cls_with_all4_pred_transformed.npz',
        allow_pickle=True,
    )
    self.data_list = []
    for i, row in df.iterrows():
      if row['split'] != self.mode:
        continue
      key = f'{i:04d}_{self.method}'
      key = key + '_transformed' if self.umeyama_transform else key
      cam_traj = matrix_to_pose7d(data[key])
      self.data_list.append((cam_traj, row['label']))
    print(
        f'Loading {len(self.data_list)}/{len(df)} data (mode = {self.mode},'
        f' method = {self.method}, umeyama_transform ='
        f' {self.umeyama_transform})'
    )

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, index):
    cam_traj, label = self.data_list[index]
    cam_traj = transform_camera_pose(cam_traj, encode_pose=self.encode_pose)
    return (cam_traj, label, index)


class EgoExo4DCameraPoseSeqForProficiencyCls(
    EgoExo4DCameraPoseSeqForScenarioCls
):

  def __init__(self, args, mode):
    self.scenario = (
        'Rock Climbing' if args is None else args.scenario_name
    )  #'Dance'
    super().__init__(args, mode)

  def build_dataset(self):
    df = pd.read_csv(
        f'{DATA_DIR}/annotations/downstream/proficiency_cls_new.csv'
    )
    mode = 'train' if self.mode == 'train' else 'val'
    self.df = df[df['split'] == mode]
    if self.scenario != 'all':
      self.df = self.df[self.df['scenario_name'] == self.scenario]
    print(
        f'Loading {len(self.df)}/{len(df)} data (mode = {self.mode}, scenario ='
        f' {self.scenario})'
    )


class EgoExo4DProficiencyVideoAndCameraPoseLongSeq(torch.utils.data.Dataset):

  def __init__(self):
    self.mode = 'val'
    self.scenario = 'Rock Climbing'
    self.take_duration = 16
    self.sample_fps = 20
    self.sample_points = int(self.sample_fps * self.take_duration)
    self.load_mapping()
    self.build_dataset()

  def load_mapping(self):
    self.camera_name_mapping = {}
    data_file = os.path.join(DATA_DIR, 'takes.json')
    with open(data_file, 'r') as f:
      data_list = json.load(f)
    for data in data_list:
      keys = list(data['frame_aligned_videos'].keys())
      self.camera_name_mapping[data['take_name']] = data[
          'frame_aligned_videos'
      ][keys[0]]['rgb']['relative_path']

  def build_dataset(self):
    df = pd.read_csv(
        f'{DATA_DIR}/annotations/downstream/proficiency_cls_new.csv'
    )
    self.df = df[df['split'] == self.mode]
    self.df = self.df[
        self.df['proficiency_score'].isin(['Early Expert'])
    ]  #'Late Expert',
    if self.scenario != 'all':
      self.df = self.df[self.df['scenario_name'] == self.scenario]
    print(
        f'Loading {len(self.df)}/{len(df)} data (mode = {self.mode}, scenario ='
        f' {self.scenario})'
    )

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    row = self.df.iloc[index]
    file_path = os.path.join(
        LOCAL_DATA_DIR,
        'takes',
        row['take_name'],
        'trajectory_presr50/closed_loop_trajectory.csv',
    )
    trajectory_df = pd.read_csv(file_path)
    cam_trajectory = trajectory_df[[
        'tx_world_device',
        'ty_world_device',
        'tz_world_device',
        'qx_world_device',
        'qy_world_device',
        'qz_world_device',
        'qw_world_device',
    ]].values
    center = cam_trajectory.shape[0] // 2
    start = max(0, center - self.sample_points // 2)
    cam_traj_slice = cam_trajectory[start : start + self.sample_points]
    cam2world = poses_to_cam2world(cam_traj_slice)
    n_frames = cam2world.shape[0]

    take_name = row['take_name']
    video_path = os.path.join(
        DATA_DIR,
        'takes',
        take_name,
        self.camera_name_mapping[take_name].replace(
            'frame_aligned_videos', 'frame_aligned_videos/downscaled/448/'
        ),
    )  #'frame_aligned_videos/downscaled/448/ego_preview.mp4')
    start_time = max(
        0,
        0.5 * cam_trajectory.shape[0] / self.sample_fps
        - 0.5 * self.take_duration,
    )
    end_time = start_time + self.take_duration
    print(
        f'video {video_path}, start_time: {start_time}, end_time: {end_time},'
        f' n_frames: {n_frames}'
    )
    frames = extract_video_frames(video_path, n_frames, start_time, end_time)
    return (
        frames,
        cam2world,
        row['scenario_name'] + '_' + row['proficiency_score'],
        '',
        '',
    )


class EgoExo4DCameraPoseSeqForActionCls(EgoExo4DCameraPoseSeqForScenarioCls):

  def __init__(self, args, mode):
    self.action_mode = 'a' if args is None else args.action_mode
    super().__init__(args, mode)

  def build_dataset(self):
    df = pd.read_csv(f'{DATA_DIR}/annotations/downstream/action_cls.csv')
    mode = 'train' if self.mode == 'train' else 'val'
    self.df = df[df['split'] == mode]
    print(f'Loading {len(self.df)}/{len(df)} data (mode = {self.mode})')

  def __getitem__(self, index):
    row = self.df.iloc[index]
    file_path = os.path.join(
        LOCAL_DATA_DIR,
        'takes',
        row['take_name'],
        'trajectory_presr50/closed_loop_trajectory.csv',
    )
    trajectory_df = pd.read_csv(file_path)
    cam_trajectory = trajectory_df[[
        'tx_world_device',
        'ty_world_device',
        'tz_world_device',
        'qx_world_device',
        'qy_world_device',
        'qz_world_device',
        'qw_world_device',
    ]].values
    if self.action_mode == 'a':
      start = int(row['start_time'] * self.sample_fps)
      end = int(row['end_time'] * self.sample_fps)
      sampled_cam_trajectory = cam_trajectory[start:end]
    else:
      raise NotImplementedError
    cam_trajectory = transform_camera_pose(
        sampled_cam_trajectory, encode_pose=2
    )
    return cam_trajectory, row['label'], index


class EgoExo4DVideoAndCameraPoseSeqForActionCounting(torch.utils.data.Dataset):

  def __init__(self, mode, use_label_id=''):
    self.mode = mode
    self.pre_sample_rate = 100
    self.use_label_id = use_label_id
    self.cache_dir = os.path.join(
        DATA_DIR,
        'camera_motion_cache',
        mode + '_presr' + str(self.pre_sample_rate),
    )
    self.load_label_mapping()
    self.build_dataset()

  def build_dataset(self):
    data_file = os.path.join(
        DATA_DIR, 'keystep_files', 'json', f'keystep_segment_{self.mode}.json'
    )
    with open(data_file, 'r') as f:
      segments_list = json.load(f)['segments']
    self.data_paths = []
    self.labels = []
    for segment in tqdm(segments_list):
      start, end = segment['start_time'], segment['end_time']
      cache_path = os.path.join(
          self.cache_dir, segment['take_name'], f'start{start}_end{end}.csv'
      )
      # if not os.path.exists(cache_path):
      #     print(f"Cache file not found: {cache_path}")
      #     continue
      video_path = os.path.join(
          DATA_DIR, 'keystep/clips_448p', segment['ego_segment_name']
      )
      # if not os.path.exists(video_path):
      #     print(f"Video file not found: {video_path}")
      #     continue
      # if 'indiana_cooking_23_3/start588.503_end648.11093' not in video_path:
      #     continue
      label_id = self.labeling_grouping[segment['label_id']]
      if self.use_label_id != '' and label_id != int(self.use_label_id):
        continue
      self.data_paths.append((video_path, cache_path))
      self.labels.append(label_id)
    print(f'Loaded {len(self.data_paths)} video and camera motion data')

  def load_label_mapping(self):
    df = pd.read_csv(
        os.path.join(DATA_DIR, 'keystep_files', 'label_mapping_v1.csv')
    )
    self.label_mapping = {
        row['verb_id']: row['verb'] for _, row in df.iterrows()
    }
    self.labeling_grouping = {
        row['label_id']: row['verb_id'] for _, row in df.iterrows()
    }

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
    n_frames = cam2world.shape[0]
    frames = extract_video_frames(video_path, n_frames)  # shape: (T, H, W, 3)
    id = '_'.join(cache_path.split('/')[-2:]).replace('.csv', '')
    label = self.labels[index]  # int
    return frames, cam2world, id, label, self.label_mapping[label]


class EgoExo4DVideoAndCameraPoseSeqForAction(torch.utils.data.Dataset):

  def __init__(self, label_id, use_relative=False):
    self.mode = 'val'
    self.label_id = label_id
    self.fps = 20
    self.use_relative = use_relative
    self.build_dataset()

  def build_dataset(self):
    df = pd.read_csv(f'{DATA_DIR}/annotations/downstream/action_cls.csv')
    self.df = df[df['label'] == self.label_id]
    print(f'Loading {len(self.df)}/{len(df)} data (mode = {self.mode})')

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    row = self.df.iloc[index]
    file_path = os.path.join(
        LOCAL_DATA_DIR,
        'takes',
        row['take_name'],
        'trajectory_presr50/closed_loop_trajectory.csv',
    )
    trajectory_df = pd.read_csv(file_path)
    cam_trajectory = trajectory_df[[
        'tx_world_device',
        'ty_world_device',
        'tz_world_device',
        'qx_world_device',
        'qy_world_device',
        'qz_world_device',
        'qw_world_device',
    ]].values
    start = int(row['start_time'] * self.fps)
    end = int(row['end_time'] * self.fps)
    sampled_cam_trajectory = cam_trajectory[start:end]
    if self.use_relative:
      cam_gt = poses_to_cam2world(sampled_cam_trajectory)
      cam_gt = custom_to_opencv_c2w(cam_gt)
      cam_gt = matrix_to_pose7d(cam_gt)
      sampled_cam_trajectory = absolute_to_relative(
          cam_gt, ref_frame_idx=len(sampled_cam_trajectory) // 2
      )
    cam2world = poses_to_cam2world(sampled_cam_trajectory)
    video_path = os.path.join(f'{DATA_DIR}/keystep/clips_448p', row['video_fp'])
    frames = extract_video_frames(video_path, cam2world.shape[0])
    return frames, cam2world, row['take_name'], row['label'], row['step_name']


class EgoExo4DCameraPoseSeqForActionFeature(torch.utils.data.Dataset):

  def __init__(self, args):
    self.fps = 20
    self.window_size = 80 if args is None else args.window_size
    self.window_stride = 10 if args is None else args.window_stride
    self.context_ratio = 0.25 if args is None else args.context_ratio
    self.build_dataset()

  def build_dataset(self):
    df = pd.read_csv(f'{DATA_DIR}/annotations/downstream/action_cls.csv')
    self.data_list = []
    self.name_list = []
    self.ranges = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
      cam_trajectory_path = os.path.join(
          LOCAL_DATA_DIR,
          'takes',
          row['take_name'],
          'trajectory_presr50/closed_loop_trajectory.csv',
      )
      take_df = pd.read_csv(cam_trajectory_path)
      cam_traj = take_df[[
          'tx_world_device',
          'ty_world_device',
          'tz_world_device',
          'qx_world_device',
          'qy_world_device',
          'qz_world_device',
          'qw_world_device',
      ]].values

      duration = row['end_time'] - row['start_time']
      start_time = row['start_time'] - 0.5 * self.context_ratio * duration
      end_time = row['end_time'] + 0.5 * self.context_ratio * duration

      range_start = max(0, int(start_time * self.fps))
      range_end = min(len(cam_traj), int(end_time * self.fps))
      true_start = max(0, int(row['start_time'] * self.fps))
      true_end = min(len(cam_traj), int(end_time * self.fps))

      for start_idx in range(range_start, range_end, self.window_stride):
        end_idx = min(start_idx + self.window_size, range_end)
        cam_traj_sampled = cam_traj[start_idx:end_idx]
        if len(cam_traj_sampled) == 0:
          cam_traj_sampled = np.random.randn(10, 7).astype(np.float32)
        # intersection range
        mask_start = max(start_idx, true_start) - start_idx
        mask_end = min(end_idx, true_end) - start_idx
        if mask_start >= mask_end:
          # print(i, range_start, range_end, start_idx, end_idx, cam_traj.shape, cam_traj_sampled.shape, true_start, true_end, mask_start, mask_end)
          cam_traj_sampled = np.random.randn(10, 7).astype(np.float32)

        self.data_list.append(cam_traj_sampled)
        self.name_list.append(i)
        self.ranges.append((mask_start, mask_end))

    print(
        f'Total {len(self.data_list)} samples from {len(df)} action instances'
    )

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, index):
    cam_traj = self.data_list[index]
    cam_traj = transform_camera_pose(cam_traj, encode_pose=2)
    return cam_traj, self.name_list[index], self.ranges[index]


class EgoExo4DCameraPoseSeqForActionContextFeature(torch.utils.data.Dataset):

  def __init__(self, args):
    self.fps = 20
    self.encode_pose = 2 if args is None else args.encode_pose
    self.sample_rate = 1 if args is None else args.sample_rate
    self.max_dur = 8 if args is None else args.max_dur
    self.context_ratio = 0.5 if args is None else args.context_ratio
    self.build_dataset()

  def build_dataset(self):
    self.df = pd.read_csv(f'{DATA_DIR}/annotations/downstream/action_cls.csv')

  def __len__(self):
    data_num = len(self.df) * 2 if self.context_ratio > 0 else len(self.df)
    return data_num

  def __getitem__(self, index):
    idx = index // 2 if self.context_ratio > 0 else index
    row = self.df.iloc[idx]
    cam_trajectory_path = os.path.join(
        LOCAL_DATA_DIR,
        'takes',
        row['take_name'],
        'trajectory_presr50/closed_loop_trajectory.csv',
    )
    take_df = pd.read_csv(cam_trajectory_path)
    cam_traj = take_df[[
        'tx_world_device',
        'ty_world_device',
        'tz_world_device',
        'qx_world_device',
        'qy_world_device',
        'qz_world_device',
        'qw_world_device',
    ]].values

    duration = row['end_time'] - row['start_time']
    if self.context_ratio == 0:
      if duration > self.max_dur:
        start_time = (
            0.5 * (row['start_time'] + row['end_time']) - self.max_dur / 2
        )
        end_time = start_time + self.max_dur
      else:
        start_time = row['start_time']
        end_time = row['end_time']

    else:
      if index % 2 == 0:
        start_time = row['start_time'] - 0.5 * self.context_ratio * duration
        end_time = row['start_time']
      else:
        start_time = row['end_time']
        end_time = row['end_time'] + (0.5 + self.context_ratio) * duration
      if end_time - start_time > self.max_dur:
        center = 0.5 * (start_time + end_time)
        start_time = center - self.max_dur / 2
        end_time = start_time + self.max_dur

    start_idx = min(max(0, int(start_time * self.fps)), len(cam_traj))
    end_idx = max(min(int(end_time * self.fps), len(cam_traj)), 0)
    # print(f'{index}, {row["take_name"]}, {start_time:.2f}-{end_time:.2f}s, idx: {start_idx}-{end_idx}, dur: {end_time-start_time:.2f}s, max_dur: {self.max_dur}s')
    cam_traj = cam_traj[start_idx:end_idx]
    cam_traj = cam_traj[:: self.sample_rate]
    if len(cam_traj) == 0:
      cam_traj = np.random.randn(10, 7).astype(np.float32)
      # print(f"Invalid indices {index}, cam_traj.shape: {cam_traj.shape}")
    cam_traj = transform_camera_pose(cam_traj, encode_pose=self.encode_pose)
    if len(cam_traj) > 160:
      print(index, cam_traj.shape)
    return cam_traj, idx, (0, len(cam_traj))


class EgoExo4DCameraPoseSeqForActionFeatureSubset(torch.utils.data.Dataset):

  def __init__(self, args):
    self.window_size = 20 if args is None else args.window_size
    self.window_stride = 2 if args is None else args.window_stride
    self.encode_pose = 2 if args is None else args.encode_pose
    self.method = 'megasam' if args is None else args.method
    assert self.method in ['gt', 'megasam', 'vipe', 'pi3']
    self.umeyama_transform = False if args is None else args.umeyama_transform
    self.build_dataset()

  def build_dataset(self):
    # df = pd.read_csv(f"{DATA_DIR}/annotations/downstream/action_cls_with_all3_pred.csv")
    df = pd.read_csv(
        f'{FINAL_DATA_DIR}/data_files/egoexo4d/action_cls_with_all3_pred.csv'
    )
    print(f'Loading {len(df)} data')
    # data = np.load(f"{DATA_DIR}/cam_est_cache/action_cls_with_all3_pred_transformed.npz", allow_pickle=True)
    data = np.load(
        f'{FINAL_DATA_DIR}/data/egoexo4d_downstream/cam_est_cache/action_cls_with_all3_pred_transformed.npz',
        allow_pickle=True,
    )
    self.data_list = []
    self.name_list = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
      key = f'{i:04d}_{self.method}'
      key = key + '_transformed' if self.umeyama_transform else key
      cam_traj = matrix_to_pose7d(data[key])
      for start_idx in range(0, len(cam_traj), self.window_stride):
        end_idx = min(start_idx + self.window_size, len(cam_traj))
        cam_traj_sampled = cam_traj[start_idx:end_idx]
        self.data_list.append(cam_traj_sampled)
        self.name_list.append(i)

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, index):
    cam_traj = self.data_list[index]
    cam_traj = transform_camera_pose(cam_traj, encode_pose=self.encode_pose)
    return cam_traj, self.name_list[index], (0, len(cam_traj))


class EgoExo4DCameraPoseSeqForLocalization(torch.utils.data.Dataset):

  def __init__(self, args):
    self.pre_sample_rate = 50
    self.fps = 1000 / self.pre_sample_rate
    self.window_size = 80 if args is None else args.window_size
    self.window_stride = 10 if args is None else args.window_stride
    self.ref_frame_idx = 'middle' if args is None else args.ref_frame_idx
    self.load_take_mapping()
    # self.export_json()
    self.load_data()

  def load_take_mapping(self):
    with open(os.path.join(DATA_DIR, 'takes.json'), 'r') as f:
      data_list = json.load(f)
    self.takeid2name = {
        data['take_uid']: data['take_name'] for data in data_list
    }

  def export_json(self):
    self.data_list = []
    self.take_ids = []
    train_dict = self.build_dataset('train')
    val_dict = self.build_dataset('val')
    with open(
        os.path.join(
            DATA_DIR, 'annotations/downstream/localization_keystep_camera.json'
        ),
        'w',
    ) as f:
      json.dump({'train': train_dict, 'val': val_dict}, f, indent=4)

  def build_dataset(self, mode):
    data_file = os.path.join(
        DATA_DIR, 'annotations/downstream/localization_keystep.json'
    )
    with open(data_file, 'r') as f:
      data_dict = json.load(f)[mode]
    save_dict = {}
    for take_id, take_info in tqdm(data_dict.items(), total=len(data_dict)):
      trajectory_path = os.path.join(
          DATA_DIR,
          'takes',
          self.takeid2name[take_id],
          'trajectory/closed_loop_trajectory.csv',
      )
      if not os.path.exists(trajectory_path):
        print(f'Trajectory file not found: {trajectory_path}')
        continue
      save_trajectory_path = trajectory_path.replace(
          'trajectory', 'trajectory_presr' + str(self.pre_sample_rate)
      )
      if not os.path.exists(save_trajectory_path):
        print(f'Subsampled Trajectory file not found: {save_trajectory_path}')
      take_df = pd.read_csv(save_trajectory_path)
      cam_traj = take_df[[
          'tx_world_device',
          'ty_world_device',
          'tz_world_device',
          'qx_world_device',
          'qy_world_device',
          'qz_world_device',
          'qw_world_device',
      ]].values
      num_frames = cam_traj.shape[0]
      num_clips = max(
          0, (num_frames - self.window_size) // self.window_stride + 1
      )
      take_info['fps'] = self.fps
      take_info['num_frames'] = num_frames
      take_info['duration'] = take_info['num_frames'] / self.fps
      take_info['num_clips'] = num_clips
      save_dict[take_id] = take_info
    return save_dict

  def load_data(self):
    with open(
        os.path.join(
            DATA_DIR, 'annotations/downstream/localization_keystep_camera.json'
        ),
        'r',
    ) as f:
      self.data_dict = json.load(f)
    self.data_list = []
    for mode in ['train', 'val']:
      for take_id, take_info in tqdm(
          self.data_dict[mode].items(), total=len(self.data_dict[mode])
      ):
        cam_trajectory_path = os.path.join(
            DATA_DIR,
            'takes',
            self.takeid2name[take_id],
            'trajectory_presr' + str(self.pre_sample_rate),
            'closed_loop_trajectory_presr50.csv',
        )
        num_frames = take_info['num_frames']
        for start_idx in range(
            0, num_frames - self.window_size + 1, self.window_stride
        ):
          end_idx = start_idx + self.window_size
          self.data_list.append({
              'take_id': take_id,
              'start_idx': start_idx,
              'end_idx': end_idx,
              'cam_trajectory_path': cam_trajectory_path,
          })
    print(f'Loaded {len(self.data_list)} camera motion data')

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, index):
    data_dict = self.data_list[index]
    cam_trajectory_path = data_dict['cam_trajectory_path']
    take_df = pd.read_csv(cam_trajectory_path)
    cam_traj = take_df[[
        'tx_world_device',
        'ty_world_device',
        'tz_world_device',
        'qx_world_device',
        'qy_world_device',
        'qz_world_device',
        'qw_world_device',
    ]].values
    cam_traj = cam_traj[data_dict['start_idx'] : data_dict['end_idx']]
    ref_frame_idx = (
        len(cam_traj) // 2
        if self.ref_frame_idx == 'middle'
        else 0
        if self.ref_frame_idx == 'start'
        else len(cam_traj) - 1
    )
    cam_traj = absolute_to_relative(cam_traj, ref_frame_idx=ref_frame_idx)
    cam_traj = pose7_to_pose9(cam_traj)
    return cam_traj, data_dict['take_id']


def sample_data_for_viz(save_dir, sampled_k=1000):
  os.makedirs(save_dir, exist_ok=True)
  dataset = EgoExo4DCameraPoseLongSeqForPretraining(None, 'test')
  # idx_list = random.sample(range(len(dataset)), k=sampled_k)
  text_list = []
  # for i, idx in enumerate(idx_list):
  for idx in range(len(dataset)):
    data = dataset[idx]
    cam_traj = data[0]
    caption = data[-1][-1][0]
    text_list.append(caption)
    np.save(f'{save_dir}/{idx}.npy', cam_traj)
    # print(idx, cam_traj.shape, caption)
  pd.Series(text_list).to_csv(
      f'{save_dir}/caption.csv', index=False, header=False
  )


def save_campose_data():
  save_dir = 'data/misc/retrieval_features/ours/egoexo4d_pretrain_longseq/all/bs1024_sampledur8_pose5/0.5/testdur4'

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

  dataset = EgoExo4DCameraPoseLongSeqForPretraining(None, 'test')
  all_trajs = []
  for i in tqdm(range(len(dataset)), total=len(dataset)):
    data = dataset[i]
    cam_traj = preprocess_data(data[0], 80)
    all_trajs.append(cam_traj.reshape(-1))

  all_trajs = np.stack(all_trajs)
  print(all_trajs.shape)  # (len(dataset), T0*9)
  np.save(f'{save_dir}/input_cam.npy', all_trajs)


if __name__ == '__main__':
  dataset = EgoExo4DCameraPoseSeqForProficiencyCls(None, 'val')
  print(len(dataset))
  # for i in range(10):
  #     data = dataset[i]
  #     print(i, data[0].shape)
