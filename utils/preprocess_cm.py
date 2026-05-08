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

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from dataset_utils import raymap_naive_batch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

data_dir = os.path.expanduser('~/data/egoexo4d')
local_data_dir = os.path.expanduser('~/local_data/egoexo4d')


def preprocess_egoexo4d_forpretraining_v2(
    mode, input_fn, output_fn, duration=30.0, sample_rate=50, ego_visible=False
):
  # fn = f'{mode}_presr50_sampled1000_{ego_visible}_mcqv0.csv' if mode == 'test' else f'{mode}_presr50_cleanv1.csv'
  fn = input_fn
  file_path = os.path.join(data_dir, 'annotations', 'pretraining', fn)
  df = pd.read_csv(file_path)
  df['take_name'] = df['save_id'].str.split('_').str[:-2].str.join('_')
  take_name_list = df['take_name'].unique()

  # fn2 = f"{mode}_presr{sample_rate}_egovisible{ego_visible}_v2" if mode == 'test' else f"{mode}_presr{sample_rate}_v2"
  # save_dir = os.path.join(data_dir, 'camera_motion_cache', 'egoexo4d_pretrain', fn2)
  # os.makedirs(save_dir, exist_ok=True)

  # List to store the new data
  new_data = []

  for take_name in tqdm(take_name_list):
    take_df = df[df['take_name'] == take_name]
    cam_traj_file = os.path.join(
        data_dir, 'takes', take_name, 'trajectory', 'closed_loop_trajectory.csv'
    )
    cam_traj = pd.read_csv(cam_traj_file)
    cam_traj = cam_traj[[
        'tx_world_device',
        'ty_world_device',
        'tz_world_device',
        'qx_world_device',
        'qy_world_device',
        'qz_world_device',
        'qw_world_device',
    ]].values
    save_dict = {}
    for i, row in take_df.iterrows():
      timestamp = row['timestamp']
      start_time = max(0, timestamp - duration)
      end_time = min(len(cam_traj) / 1e3, timestamp + duration)
      save_id = f'{take_name}_{start_time:.2f}_{end_time:.2f}'

      cam_traj_slice = cam_traj[
          int(start_time * 1e3) : int(end_time * 1e3) : sample_rate
      ]
      # save_dict[save_id] = cam_traj_slice
      timestamp_idx = int((timestamp - start_time) * 1e3 / sample_rate)
      start_idx = max(
          0, int((row['start_time'] - start_time) * 1e3 / sample_rate)
      )
      end_idx = min(
          len(cam_traj_slice),
          int((row['end_time'] - start_time) * 1e3 / sample_rate),
      )
      # print(start_time, end_time, timestamp, cam_traj_slice.shape, timestamp_idx, start_idx, end_idx)

      # Add data to the list
      new_data.append({
          'save_id': save_id,
          'description_text': row['description_text'],
          'start_time': start_time,
          'end_time': end_time,
          'timestamp': timestamp,
          't0': row['start_time'],
          't1': row['end_time'],
          'timestamp_idx': timestamp_idx,
          't0_idx': start_idx,
          't1_idx': end_idx,
          'cam_traj_len': cam_traj_slice.shape[0],
          'ego_visible': row['ego_visible'],
      })
      # if save_id not in save_dict:
      # save_dict[save_id] = cam_traj_slice

    # np.savez(os.path.join(save_dir, f"{take_name}.npz"), **save_dict)

  # Create new DataFrame
  new_df = pd.DataFrame(new_data)

  # Save the new DataFrame
  # fn3 = f'{mode}_egovisible{ego_visible}_v2.csv' if mode == 'test' else f'{mode}_v2.csv'
  fn3 = output_fn
  output_path = os.path.join(data_dir, 'annotations', 'pretraining', fn3)
  new_df.to_csv(output_path, index=False)
  print(f'Saved processed indices to: {output_path}')
  print(f'DataFrame shape: {new_df.shape}')


def _resize_numpy(raymap, dim=448):
  raymap = torch.from_numpy(raymap).float()
  raymap = raymap.permute(0, 3, 1, 2)
  resized_raymap = F.interpolate(
      raymap, size=(dim, dim), mode='bilinear', align_corners=False
  )
  resized_raymap = resized_raymap.permute(0, 2, 3, 1)  # -> (T, 448, 448, 6)
  return resized_raymap.numpy()


def _process_raymap_row(row_tuple, mode, intrinsics):
  i, row = row_tuple
  try:
    cache_path = os.path.join(
        local_data_dir,
        'camera_motion_cache/egoexo4d_pretrain',
        mode + '_presr50',
        row['save_id'] + '_absolute.npy',
    )
    save_path = (
        cache_path.replace('_absolute.npy', '_raymap.npy')
        .replace(local_data_dir, data_dir)
        .replace('camera_motion_cache', 'camera_motion_cache_raymap')
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
      return
    pose_seq = np.load(cache_path)
    if pose_seq.shape[0] == 0:
      return
    raymap = raymap_naive_batch(intrinsics, pose_seq)
    raymap = _resize_numpy(raymap)
    # print(f"Saving {raymap.shape} to {save_path}")
    np.save(save_path, raymap)
  except FileNotFoundError:
    pass  # Some files might be missing, which is expected.
  except Exception as e:
    print(f"Error processing row {i} ({row['save_id']}): {e}")


def preprocess_egoexo4d_raymap(mode, num_workers=8):
  intrinsics = [
      610.524378190068,
      610.524378190068,
      712.688057683927,
      710.828602211728,
      1408,
      1408,
  ]
  df = pd.read_csv(
      f'{data_dir}/annotations/pretraining/{mode}_presr50_cleanv1_components.csv'
  )
  print(
      f'Processing {len(df)} rows (mode = {mode}) with {num_workers} workers.'
  )
  tasks = list(df.iterrows())
  process_func = partial(_process_raymap_row, mode=mode, intrinsics=intrinsics)
  with ProcessPoolExecutor(max_workers=num_workers) as executor:
    list(
        tqdm(
            executor.map(process_func, tasks),
            total=len(tasks),
            desc=f'Generating raymaps for {mode}',
        )
    )
  print('Finished processing all rows.')


if __name__ == '__main__':
  # for task_name in ['Rock_Climbing', 'Bike_Repair', 'Basketball', 'Soccer', 'Music']:  #'Cooking', 'Health', 'Dance',
  #     for ego_visible in [True, False]:
  #         preprocess_egoexo4d_forpretraining_v2('', f'test_presr50_sampled500_{task_name}_egovisible{ego_visible}_mcqv0.csv', f'test_{task_name}_egovisible{ego_visible}_mcqv0.csv')
  preprocess_egoexo4d_raymap('val', num_workers=16)
