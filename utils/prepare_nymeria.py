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

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import glob
import json
import os
import os, subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = os.path.expanduser('~/local_data/nymeria/')


def filter_download_data_json():
  with open(f'{DATA_DIR}/Nymeria_download_urls.json', 'r') as f:
    data = json.load(f)
  new_data = {}
  for key, value in data['sequences'].items():
    if (
        'narration_motion_narration_csv' not in value
    ):  # narration_atomic_action_csv
      continue
    new_data[key] = value
  print(f'Found {len(new_data)} sequences with motion narration')
  save_dict = {'sequences': new_data}
  with open(
      f'{DATA_DIR}/Nymeria_download_urls_motion_narration.json', 'w'
  ) as f:
    json.dump(save_dict, f)


def prepare_data(pre_sample_rate=50):
  with open(
      f'{DATA_DIR}/Nymeria_download_urls_motion_narration.json', 'r'
  ) as f:
    data = json.load(f)
  print(f"Found {len(data['sequences'])} sequences")
  motion_save_dir = f'{DATA_DIR}/cam_motion_cache/presr{pre_sample_rate}'
  os.makedirs(motion_save_dir, exist_ok=True)
  cnt = 0
  for key, value in tqdm(data['sequences'].items()):
    save_file = f'{motion_save_dir}/{key}.npz'
    if os.path.exists(save_file):
      continue
    ann_file = f'{DATA_DIR}/motion_data/{key}/narration/motion_narration.csv'
    motion_file = f'{DATA_DIR}/motion_data/{key}/recording_head/mps/slam/closed_loop_trajectory.csv'
    if not os.path.exists(ann_file) or not os.path.exists(motion_file):
      continue
    ann_df = pd.read_csv(ann_file)
    motion_df = pd.read_csv(motion_file)
    save_dict = {}
    for i, row in ann_df.iterrows():
      start_time = row['start_time']
      end_time = row['end_time']
      motion_df_slice = motion_df[
          (motion_df['tracking_timestamp_us'] >= start_time * 1e6)
          & (motion_df['tracking_timestamp_us'] <= end_time * 1e6)
      ]
      motion_df_slice = motion_df_slice.iloc[::pre_sample_rate]
      cam_trajectory = motion_df_slice[[
          'tx_world_device',
          'ty_world_device',
          'tz_world_device',
          'qx_world_device',
          'qy_world_device',
          'qz_world_device',
          'qw_world_device',
      ]].values
      save_dict[str(i)] = cam_trajectory
    np.savez(save_file, **save_dict)
    # print(f"Processed {key}")


def get_video_duration(path):
  result = subprocess.run(
      [
          'ffprobe',
          '-v',
          'error',
          '-show_entries',
          'format=duration',
          '-of',
          'default=noprint_wrappers=1:nokey=1',
          path,
      ],
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
  )
  return float(result.stdout)


def _process_video_sequence(item, video_save_dir):
  """Helper function to process a single video sequence."""
  key, value = item
  os.makedirs(f'{video_save_dir}/{key}', exist_ok=True)
  ann_file = f'{DATA_DIR}/motion_data/{key}/narration/motion_narration.csv'
  video_file = f'{DATA_DIR}/motion_data/{key}/video_main_rgb/preview_rgb.mp4'
  motion_file = f'{DATA_DIR}/motion_data/{key}/recording_head/mps/slam/closed_loop_trajectory.csv'
  if (
      not os.path.exists(ann_file)
      or not os.path.exists(video_file)
      or not os.path.exists(motion_file)
  ):
    return None

  video_dur = get_video_duration(video_file)
  motion_df = pd.read_csv(motion_file)
  t0 = motion_df.iloc[0]['tracking_timestamp_us'] / 1e6
  tn = motion_df.iloc[-1]['tracking_timestamp_us'] / 1e6
  ann_df = pd.read_csv(ann_file)
  align_error = abs(video_dur - (tn - t0))

  for i, row in ann_df.iterrows():
    video_path = f'{video_save_dir}/{key}/{i}.mp4'
    if os.path.exists(video_path):
      continue

    start_time = row['start_time'] - t0
    end_time = row['end_time'] - t0
    duration = min(end_time, video_dur) - start_time
    if duration <= 0:
      continue
    # text = row['Describe my body posture'].replace(' ', '_')

    cmd = (  # _st{start_time:.2f}_end{end_time:.2f}_{text}.mp4"
        f'ffmpeg -y -ss {start_time} -i {video_file} -t {duration} -c copy'
        f' {video_path}'
    )
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

  return align_error


def prepare_video(num_workers=None):
  if num_workers is None:
    num_workers = os.cpu_count() or 1
  with open(
      f'{DATA_DIR}/Nymeria_download_urls_motion_narration.json', 'r'
  ) as f:
    data = json.load(f)
  print(f"Found {len(data['sequences'])} sequences")
  video_save_dir = f'{DATA_DIR}/video_cache/'
  os.makedirs(video_save_dir, exist_ok=True)

  tasks = list(data['sequences'].items())
  align_errors = []

  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_task = {
        executor.submit(_process_video_sequence, task, video_save_dir): task
        for task in tasks
    }
    for future in tqdm(
        as_completed(future_to_task), total=len(tasks), desc='Processing videos'
    ):
      task = future_to_task[future]
      try:
        result = future.result()
        if result is not None:
          align_errors.append(result)
      except Exception as exc:
        print(f'Task {task[0]} generated an exception: {exc}')

  print(f'Alignment errors: {np.mean(align_errors)}')


text_column_mapping = {
    'a': 'Describe my body posture',
    'b': 'Describe my hands/arms motion',
    'c': 'Describe my legs/feet motion',
    'd': 'Describe my focus attention',
}


def _process_imu_row(row, local_data_dir):
  """Helper function to process a single row from the dataframe."""
  try:
    take_name = os.path.basename(row['motion_file']).replace('.npz', '')

    motion_file = os.path.join(
        local_data_dir,
        take_name,
        'recording_head/mps/slam/closed_loop_trajectory.csv',
    )
    motion_df = pd.read_csv(motion_file)
    t0 = motion_df.iloc[0]['tracking_timestamp_us'] / 1e6

    ann_file = os.path.join(
        local_data_dir, take_name, 'narration/motion_narration.csv'
    )
    ann_df = pd.read_csv(ann_file)
    ann_row = ann_df.iloc[row['row_idx']]
    start_time = ann_row['start_time']
    end_time = ann_row['end_time']
    row_dict = row.to_dict()
    row_dict.update({
        'take_name': take_name,
        't0': t0,
        'start_time': start_time,
        'end_time': end_time,
    })
    return row_dict
  except Exception as e:
    print(
        'Error processing row for motion file'
        f" {row.get('motion_file', 'N/A')}: {e}"
    )
    return None


def prepare_imu(text_column, num_workers=None):
  if num_workers is None:
    num_workers = os.cpu_count() or 1
  data_dir = os.path.expanduser('~/data/nymeria')
  local_data_dir = os.path.expanduser('~/local_data/nymeria/motion_data')
  select_column = text_column_mapping[text_column]
  key_name = select_column.replace('/', '_').replace(' ', '_')
  data_file = f'{data_dir}/eval1000/split_by_{key_name}.csv'
  df = pd.read_csv(data_file)

  tasks = [row for _, row in df.iterrows()]
  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    worker_func = partial(_process_imu_row, local_data_dir=local_data_dir)
    results = list(
        tqdm(
            executor.map(worker_func, tasks),
            total=len(tasks),
            desc='Preparing IMU data',
        )
    )

  new_rows = [res for res in results if res is not None]

  new_df = pd.DataFrame(new_rows)
  new_df.to_csv(data_file.replace('.csv', '_new.csv'), index=False)


if __name__ == '__main__':
  # filter_download_data_json()
  # prepare_data()
  # prepare_video(32)
  prepare_imu('b', num_workers=32)
  prepare_imu('c', num_workers=32)
  prepare_imu('d', num_workers=32)
