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

"""Step 3: Build the long-sequence training files the pretraining loader reads.

For each description row, take a +/- `duration`-second window of the take's
trajectory (sub-sampled at `sample_rate`) and:
  - store all windows of a given take together in one .npz keyed by save_id
  - record (save_id, description_text, start_time, end_time, timestamp, t0, t1,
            timestamp_idx, t0_idx, t1_idx, cam_traj_len, ego_visible).

Produces the `train_v2.csv` / `val_v2.csv` that the dataset class expects, plus
per-take `<take>.npz` bundles under `{mode}_presr{sample_rate}_v2/`.

Expected input: the cleaned CSV from step 2 (e.g. `train_presr50_cleanv1.csv`)
OR any CSV with (save_id, description_text, timestamp, start_time, end_time,
ego_visible) columns.
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def main(
    mode, data_dir, input_csv, output_csv, duration, sample_rate, save_npz
):
  in_path = os.path.join(data_dir, 'annotations', 'pretraining', input_csv)
  df = pd.read_csv(in_path)
  df['take_name'] = df['save_id'].str.split('_').str[:-2].str.join('_')
  take_names = df['take_name'].unique()
  print(f'Loaded {len(df)} rows / {len(take_names)} takes from {in_path}')

  npz_dir = None
  if save_npz:
    npz_dir = os.path.join(
        data_dir,
        'camera_motion_cache',
        'egoexo4d_pretrain',
        f'{mode}_presr{sample_rate}_v2',
    )
    os.makedirs(npz_dir, exist_ok=True)

  new_rows = []
  for take_name in tqdm(take_names):
    take_df = df[df['take_name'] == take_name]
    traj_path = os.path.join(
        data_dir, 'takes', take_name, 'trajectory', 'closed_loop_trajectory.csv'
    )
    if not os.path.exists(traj_path):
      continue
    traj = pd.read_csv(traj_path)[[
        'tx_world_device',
        'ty_world_device',
        'tz_world_device',
        'qx_world_device',
        'qy_world_device',
        'qz_world_device',
        'qw_world_device',
    ]].values

    save_dict = {}
    for _, row in take_df.iterrows():
      ts = row['timestamp']
      start_time = max(0.0, ts - duration)
      end_time = min(len(traj) / 1e3, ts + duration)
      save_id = f'{take_name}_{start_time:.2f}_{end_time:.2f}'

      traj_slice = traj[
          int(start_time * 1e3) : int(end_time * 1e3) : sample_rate
      ]
      timestamp_idx = int((ts - start_time) * 1e3 / sample_rate)
      t0_idx = max(0, int((row['start_time'] - start_time) * 1e3 / sample_rate))
      t1_idx = min(
          len(traj_slice),
          int((row['end_time'] - start_time) * 1e3 / sample_rate),
      )

      new_rows.append({
          'save_id': save_id,
          'description_text': row['description_text'],
          'start_time': start_time,
          'end_time': end_time,
          'timestamp': ts,
          't0': row['start_time'],
          't1': row['end_time'],
          'timestamp_idx': timestamp_idx,
          't0_idx': t0_idx,
          't1_idx': t1_idx,
          'cam_traj_len': traj_slice.shape[0],
          'ego_visible': row['ego_visible'],
      })

      if save_npz and save_id not in save_dict:
        save_dict[save_id] = traj_slice

    if save_npz and save_dict:
      np.savez(os.path.join(npz_dir, f'{take_name}.npz'), **save_dict)

  out_df = pd.DataFrame(new_rows)
  out_path = os.path.join(data_dir, 'annotations', 'pretraining', output_csv)
  out_df.to_csv(out_path, index=False)
  print(f'Wrote {len(out_df)} rows to {out_path}')
  if save_npz:
    print(f'Wrote per-take npz bundles to {npz_dir}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--mode', type=str, required=True, choices=['train', 'val']
  )
  parser.add_argument(
      '--data_dir', type=str, default=os.path.expanduser('~/data/egoexo4d')
  )
  parser.add_argument(
      '--input_csv',
      type=str,
      default=None,
      help='Defaults to {mode}_presr50_cleanv1.csv.',
  )
  parser.add_argument(
      '--output_csv', type=str, default=None, help='Defaults to {mode}_v2.csv.'
  )
  parser.add_argument(
      '--duration',
      type=float,
      default=30.0,
      help="Half-window in seconds around each description's timestamp.",
  )
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=50,
      help='Sub-sampling of the raw 1 kHz trajectory (50 -> 20 Hz).',
  )
  parser.add_argument(
      '--no_npz',
      action='store_true',
      help='Skip writing per-take npz bundles (CSV only).',
  )
  args = parser.parse_args()

  input_csv = (
      args.input_csv or f'{args.mode}_presr{args.sample_rate}_cleanv1.csv'
  )
  output_csv = args.output_csv or f'{args.mode}_v2.csv'
  main(
      args.mode,
      args.data_dir,
      input_csv,
      output_csv,
      args.duration,
      args.sample_rate,
      save_npz=not args.no_npz,
  )
