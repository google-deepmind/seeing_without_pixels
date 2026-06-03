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

"""Step 1: Turn EgoExo4D atomic descriptions into per-segment camera-pose .npy files

and a flat CSV of (save_id, description_text, start_time, end_time, ...).

Input:
  - ~/data/egoexo4d/takes.json
  - ~/data/egoexo4d/annotations/atomic_descriptions_{mode}.json
  - ~/data/egoexo4d/takes/<take_name>/trajectory/closed_loop_trajectory.csv

Output:
  - ~/data/egoexo4d/camera_motion_cache/egoexo4d_pretrain/{mode}_presr50/
        <save_id>_absolute.npy, <save_id>_relative.npy
  - ~/data/egoexo4d/annotations/pretraining/{mode}_presr50.csv

Notes:
  - `train_alpha` controls the half-window around each description timestamp,
    scaled by the local mean spacing between adjacent descriptions. This is the
    value used for the paper.
  - Trajectories are sub-sampled by `sample_rate` (50 -> 20 Hz after
  sub-sampling
    from 1 kHz Aria output).
"""

import argparse
from functools import partial
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.dataset_utils import absolute_to_relative_from_t0

TRAIN_ALPHA = 2.2724306610481197


def _process_take(chunk_data, data_dir, save_dir, sample_rate, train_alpha):
  take_dict, take_uid, data_list = chunk_data
  results = []

  take_info = take_dict[take_uid]
  take_name = take_info['take_name']
  traj_path = os.path.join(
      data_dir, 'takes', take_name, 'trajectory/closed_loop_trajectory.csv'
  )
  if not os.path.exists(traj_path):
    return results

  df = pd.read_csv(traj_path)
  df['relative_time'] = (
      df['tracking_timestamp_us'] - df['tracking_timestamp_us'].iloc[0]
  ) / 1e6

  for data in data_list:
    timestamp_list = [d['timestamp'] for d in data['descriptions']]
    if len(timestamp_list) == 0:
      continue
    diffs = [
        timestamp_list[i + 1] - timestamp_list[i]
        for i in range(len(timestamp_list) - 1)
    ]
    beta = np.mean(diffs)
    if np.isnan(beta):
      continue
    interval = beta / (2 * train_alpha)

    for description in data['descriptions']:
      ts = description['timestamp']
      text = description['text']
      start_time = ts - interval
      end_time = ts + interval
      save_id = f'{take_name}_{start_time:.2f}_{end_time:.2f}'
      results.append([
          save_id,
          text,
          ts,
          interval,
          start_time,
          end_time,
          description['ego_visible'],
      ])

      abs_path = os.path.join(save_dir, f'{save_id}_absolute.npy')
      if os.path.exists(abs_path):
        continue
      try:
        start_idx = df[df['relative_time'] >= start_time].index[0]
        end_idx = df[df['relative_time'] <= end_time].index[-1]
      except Exception:
        continue

      sub_df = df.iloc[start_idx : end_idx + 1]
      absolute_pose = sub_df[[
          'tx_world_device',
          'ty_world_device',
          'tz_world_device',
          'qx_world_device',
          'qy_world_device',
          'qz_world_device',
          'qw_world_device',
      ]].values
      absolute_pose = absolute_pose[::sample_rate]
      if absolute_pose.shape[0] < 2:
        continue

      relative_pose = absolute_to_relative_from_t0(absolute_pose)
      np.save(os.path.join(save_dir, f'{save_id}_relative.npy'), relative_pose)
      np.save(abs_path, absolute_pose)

  return results


def main(mode, data_dir, sample_rate, num_processes):
  save_dir = os.path.join(
      data_dir,
      'camera_motion_cache',
      'egoexo4d_pretrain',
      f'{mode}_presr{sample_rate}',
  )
  os.makedirs(save_dir, exist_ok=True)

  with open(os.path.join(data_dir, 'takes.json'), 'r') as f:
    take_list = json.load(f)
  take_dict = {t['take_uid']: t for t in take_list}
  print(f'Loaded {len(take_dict)} takes')

  with open(
      os.path.join(data_dir, 'annotations', f'atomic_descriptions_{mode}.json'),
      'r',
  ) as f:
    data_dict = json.load(f)['annotations']

  chunks = [
      (take_dict, uid, dl) for uid, dl in data_dict.items() if uid in take_dict
  ]
  proc = partial(
      _process_take,
      data_dir=data_dir,
      save_dir=save_dir,
      sample_rate=sample_rate,
      train_alpha=TRAIN_ALPHA,
  )

  all_results = []
  with mp.Pool(num_processes) as pool:
    for res in tqdm(
        pool.imap_unordered(proc, chunks),
        total=len(chunks),
        desc=f'Processing takes with {num_processes} processes',
    ):
      all_results.extend(res)

  out_df = pd.DataFrame(
      all_results,
      columns=[
          'save_id',
          'description_text',
          'timestamp',
          'interval',
          'start_time',
          'end_time',
          'ego_visible',
      ],
  )
  out_dir = os.path.join(data_dir, 'annotations', 'pretraining')
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, f'{mode}_presr{sample_rate}.csv')
  out_df.to_csv(out_path, index=False)
  print(f'Wrote {len(out_df)} rows to {out_path}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--mode', type=str, required=True, choices=['train', 'val']
  )
  parser.add_argument(
      '--data_dir', type=str, default=os.path.expanduser('~/data/egoexo4d')
  )
  parser.add_argument('--sample_rate', type=int, default=50)
  parser.add_argument(
      '--num_processes', type=int, default=max(1, mp.cpu_count() - 1)
  )
  args = parser.parse_args()
  main(args.mode, args.data_dir, args.sample_rate, args.num_processes)
