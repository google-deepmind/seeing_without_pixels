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

import bisect
from collections import Counter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import glob
import json
import json
import math
import multiprocessing as mp
import os
import random

# import spacy
import shutil
import subprocess
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import torch
from tqdm import tqdm
from utils.dataset_utils import absolute_to_relative_from_t0

# from pydub import AudioSegment


def split_by_participant(data_dir):
  random.seed(0)
  df = pd.read_pickle(
      os.path.join(data_dir, 'ann_files/HD_EPIC_Narrations.pkl')
  )
  unique_participants = list(df['participant_id'].unique())
  print(f'There are {len(unique_participants)} unique participants')
  test_participants = random.sample(unique_participants, 2)
  train_participants = [
      p for p in unique_participants if p not in test_participants
  ]
  print(f'Test participants: {test_participants}')
  print(f'Train participants: {train_participants}')

  train_df = df[df['participant_id'].isin(train_participants)]
  test_df = df[df['participant_id'].isin(test_participants)]
  train_df.to_csv(
      os.path.join(
          data_dir, 'ann_files/HD_EPIC_Narrations_train_byparticipant.csv'
      ),
      index=False,
  )
  test_df.to_csv(
      os.path.join(
          data_dir, 'ann_files/HD_EPIC_Narrations_test_byparticipant.csv'
      ),
      index=False,
  )


def label_dist(data_file, pred_file):
  df = pd.read_csv(data_file)
  pred_df = pd.read_csv(pred_file)
  pred_df['annotation_id'] = pred_df['annotation_id'].str.replace('_', '-')
  pred_df = pred_df[pred_df['annotation_id'].isin(df['unique_narration_id'])]
  print(pred_df['gt_verb'].value_counts())
  print(len(pred_df))


def reformat_egoexo4d_label(data_file):
  df = pd.read_csv(data_file)
  df['verb'] = df['step_name'].apply(lambda x: x.split(' ')[0])
  df['change'] = 0
  df.to_csv(data_file.replace('.csv', '_reformat.csv'), index=False)


def update_egoexo4d_labelv1():
  data_dir = os.path.expanduser('~/data/egoexo4d')
  df = pd.read_csv(
      os.path.join(data_dir, 'keystep_files', 'label_mapping_v1.csv')
  )
  df['step_name+scenario'] = df['step_name'] + ' (' + df['scenario_name'] + ')'
  unique_verbs = df['verb'].unique()
  for verb in unique_verbs:
    sub_df = df[df['verb'] == verb]
    unique_step_names = sub_df['step_name+scenario'].unique()
    if len(unique_step_names) > 1:
      print(f'{verb} ({len(unique_step_names)}): {unique_step_names}')

  # delete column change
  df = df.drop(columns=['change', 'step_name+scenario'])
  df['verb'] = df['verb'] + '_' + df['scenario_name']

  # Create new verb_id mapping based on unique verbs
  unique_verbs = df['verb'].unique()
  verb_id_mapping = {verb: idx for idx, verb in enumerate(sorted(unique_verbs))}
  df['verb_id'] = df['verb'].map(verb_id_mapping)
  print(
      f'{len(unique_verbs)} unique verbs,'
      f" {df['verb_id'].min()}-{df['verb_id'].max()}"
  )
  df.to_csv(
      os.path.join(data_dir, 'keystep_files', 'label_mapping_v2.csv'),
      index=False,
  )


def ana_label(data_file):
  df_new = pd.read_csv(data_file)
  unique_verbs = df_new['verb'].unique()
  for verb in unique_verbs:
    sub_df = df_new[df_new['verb'] == verb]
    unique_step_names = sub_df['step_name'].unique()
    print(f'{verb} ({len(unique_step_names)}): {unique_step_names}')
    print('-' * 100)
  print(f'Unique verbs: {len(unique_verbs)}')
  # create a mapping verb to verb_id
  verb_id_mapping = {}
  for i, verb in enumerate(unique_verbs):
    verb_id_mapping[verb] = i
  # update df_new with verb_id
  df_new['verb_id'] = df_new['verb'].map(verb_id_mapping)
  df_new.to_csv(data_file.replace('.csv', '_reformat_with_id.csv'), index=False)


def hdepic_label_dist(mode):
  df = pd.read_csv(
      f'data/hd-epic_action/ann_files/HD_EPIC_Narrations_{mode}.csv'
  )
  # Filter rows and add verb_id column
  df['verb_id'] = df['main_action_classes'].apply(
      lambda x: eval(x)[0][0] if len(eval(x)) == 1 else None
  )
  df = df.dropna(subset=['verb_id'])
  df['verb_id'] = df['verb_id'].astype(int)

  # Add main action labels
  df['action_label'] = df['main_actions'].apply(
      lambda x: eval(x)[0][0] if len(eval(x)) == 1 else None
  )

  # Print distribution
  print(f'\nVerb ID distribution for {mode} set:')
  print(df['verb_id'].value_counts())

  # Plot distribution
  plt.figure(figsize=(20, 8))
  verb_counts = df['verb_id'].value_counts()
  # Get action labels for each verb_id
  x_labels = [
      df[df['verb_id'] == vid]['action_label'].iloc[0]
      for vid in verb_counts.index
  ]

  plt.bar(range(len(verb_counts)), verb_counts.values)
  plt.xticks(
      range(len(verb_counts)), x_labels, rotation=45, ha='right', fontsize=12
  )
  plt.yticks(fontsize=12)
  plt.xlabel('Action', fontsize=14)
  plt.ylabel('Count', fontsize=14)
  plt.title(f'Action Distribution for {mode.capitalize()} Set', fontsize=16)

  # Adjust layout
  plt.tight_layout()
  plt.savefig(
      f'data/hd-epic_action/ann_files/verb_id_dist_{mode}.png',
      bbox_inches='tight',
      dpi=300,
  )
  plt.close()
  return df


def egoexo4d_label_dist(mode):
  df = pd.read_csv('data/egoexo4d/keystep_files/label_mapping_v1.csv')
  labeling_grouping = {
      row['label_id']: row['verb_id'] for _, row in df.iterrows()
  }
  label_mapping = {row['verb_id']: row['verb'] for _, row in df.iterrows()}
  data_dir = os.path.expanduser('~/data/egoexo4d')
  data_file = os.path.join(
      data_dir, 'keystep_files', 'json', f'keystep_segment_{mode}.json'
  )
  with open(data_file, 'r') as f:
    segments_list = json.load(f)['segments']
  label_dist = {}
  for segment in segments_list:
    label = labeling_grouping[segment['label_id']]
    if label not in label_dist:
      label_dist[label] = 0
    label_dist[label] += 1
  # sort by value
  label_dist = dict(
      sorted(label_dist.items(), key=lambda item: item[1], reverse=True)
  )
  for verb_id, count in label_dist.items():
    print(f'{verb_id}-{label_mapping[verb_id]}: {count}')
  print(f'{len(label_dist)} unique labels')
  # plot
  plt.figure(figsize=(20, 8))
  x_labels = [f'{label_mapping[verb_id]}' for verb_id in label_dist.keys()]
  plt.bar(x_labels, label_dist.values())
  plt.xticks(rotation=45, ha='right', fontsize=12)
  plt.yticks(fontsize=12)
  plt.xlabel('Verb', fontsize=14)
  plt.ylabel('Count', fontsize=14)
  plt.title(f'Label Distribution for {mode.capitalize()} Set', fontsize=16)
  # Adjust layout to prevent label cutoff
  plt.subplots_adjust(bottom=0.2)
  plt.tight_layout()
  plt.savefig(
      os.path.join(data_dir, 'keystep_files', f'label_dist_{mode}.png'),
      bbox_inches='tight',
      dpi=300,
  )
  plt.close()


def create_egoexo4d_video_train_file(version='v1'):
  data_dir = os.path.expanduser('~/data/egoexo4d')
  df = pd.read_csv(
      os.path.join(data_dir, 'keystep_files', f'label_mapping_{version}.csv')
  )
  labeling_grouping = {
      row['label_id']: row['verb_id'] for _, row in df.iterrows()
  }
  print(f"Label range: {df['verb_id'].min()}-{df['verb_id'].max()}")
  for mode in ['train', 'val']:
    file_path = os.path.join(data_dir, 'keystep_files', 'csv', f'{mode}.csv')
    df = pd.read_csv(file_path, header=None)
    rows = []
    for i, row in df.iterrows():
      label = labeling_grouping[row[1]]
      rows.append([row[0], label])
    df = pd.DataFrame(rows)
    os.makedirs(
        os.path.join(data_dir, 'keystep_files', f'csv_{version}'), exist_ok=True
    )
    df.to_csv(
        os.path.join(
            data_dir, 'keystep_files', f'csv_{version}', f'{mode}.csv'
        ),
        index=False,
        header=False,
    )


def create_hdepic_video_train_file():
  data_dir = os.path.expanduser('~/data/hd-epic_action')
  for mode in ['train', 'test']:
    df = pd.read_csv(
        os.path.join(data_dir, f'ann_files/HD_EPIC_Narrations_{mode}.csv')
    )
    new_df = []
    for i, row in df.iterrows():
      camera_pose_path = os.path.join(
          data_dir,
          'camera_motion',
          row['participant_id'],
          row['video_id']
          + f"_start{row['start_timestamp']}_end{row['end_timestamp']}.csv",
      )
      video_path = camera_pose_path.replace('camera_motion', 'clips').replace(
          '.csv', '.mp4'
      )
      relative_video_path = os.path.relpath(video_path, f'{data_dir}/clips')
      main_action_classes = eval(row['main_action_classes'])
      if len(main_action_classes) != 1:
        continue
      new_df.append([relative_video_path, main_action_classes[0][0]])
    os.makedirs(
        os.path.join(data_dir, 'ann_files', 'video_train'), exist_ok=True
    )
    new_df = pd.DataFrame(new_df)
    new_df.to_csv(
        os.path.join(data_dir, 'ann_files', 'video_train', f'{mode}.csv'),
        index=False,
        header=False,
    )


def _process_take_chunk(
    chunk_data, data_dir1, data_dir2, save_dir, sample_rate, train_alpha
):
  """Helper function to process a chunk of takes in parallel"""
  take_dict, take_uid, data_list = chunk_data
  results = []

  take_info = take_dict[take_uid]
  take_name = take_info['take_name']
  take_trajectory_path = os.path.join(
      data_dir2, 'takes', take_name, 'trajectory/closed_loop_trajectory.csv'
  )
  if not os.path.exists(take_trajectory_path):
    return results

  df = pd.read_csv(take_trajectory_path)
  df['relative_time'] = (
      df['tracking_timestamp_us'] - df['tracking_timestamp_us'].iloc[0]
  ) / 1e6

  for data in data_list:
    timestamp_list = [d['timestamp'] for d in data['descriptions']]
    if len(timestamp_list) == 0:
      continue
    timestamp_diff_list = [
        timestamp_list[i + 1] - timestamp_list[i]
        for i in range(len(timestamp_list) - 1)
    ]
    beta = np.mean(timestamp_diff_list)
    if np.isnan(beta):
      continue

    interval = beta / (2 * train_alpha)

    for description in data['descriptions']:
      timestamp = description['timestamp']
      description_text = description['text']
      start_time = timestamp - interval
      end_time = timestamp + interval
      save_id = f'{take_name}_{start_time:.2f}_{end_time:.2f}'
      results.append([
          save_id,
          description_text,
          timestamp,
          interval,
          start_time,
          end_time,
          description['ego_visible'],
      ])

      save_path = os.path.join(save_dir, f'{save_id}_absolute.npy')
      if os.path.exists(save_path):
        continue
      try:
        start_idx = df[df['relative_time'] >= start_time].index[0]
        end_idx = df[df['relative_time'] <= end_time].index[-1]
      except:
        print(f'Start time: {start_time}, end time: {end_time}')
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
      np.save(save_path, absolute_pose)

  return results


def preprocess_egoexo4d_forpretraining(
    mode, sample_rate=50, num_processes=None
):
  if num_processes is None:
    num_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU free

  data_dir1 = os.path.expanduser('~/data/egoexo4d')
  data_dir2 = os.path.expanduser('~/projects/camera_motion_modeling/egoexo4d')
  save_dir = os.path.join(
      data_dir1,
      'camera_motion_cache',
      'egoexo4d_pretrain',
      f'{mode}_presr{sample_rate}',
  )
  os.makedirs(save_dir, exist_ok=True)

  take_dict = {}
  take_file = os.path.join(data_dir1, 'takes.json')
  with open(take_file, 'r') as f:
    take_list = json.load(f)
  for take_info in take_list:
    take_id = take_info['take_uid']
    take_dict[take_id] = take_info
  print(f'Loading {len(take_dict)} takes')

  train_alpha = 2.2724306610481197
  interval_list = []

  data_file = os.path.join(
      data_dir1, 'annotations', f'atomic_descriptions_{mode}.json'
  )
  with open(data_file, 'r') as f:
    data_dict = json.load(f)['annotations']

    # Prepare data for parallel processing
    chunk_data = [
        (take_dict, take_uid, data_list)
        for take_uid, data_list in data_dict.items()
        if take_uid in take_dict
    ]

    # Process chunks in parallel
    process_func = partial(
        _process_take_chunk,
        data_dir1=data_dir1,
        data_dir2=data_dir2,
        save_dir=save_dir,
        sample_rate=sample_rate,
        train_alpha=train_alpha,
    )

    all_results = []
    with mp.Pool(num_processes) as pool:
      for results in tqdm(
          pool.imap_unordered(process_func, chunk_data),
          total=len(chunk_data),
          desc=f'Processing takes with {num_processes} processes',
      ):
        all_results.extend(results)
        # Update interval list
        intervals = [result[3] for result in results]
        interval_list.extend(intervals)

    # Save results
    save_df = pd.DataFrame(
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
    os.makedirs(
        os.path.join(data_dir1, 'annotations', 'pretraining'), exist_ok=True
    )
    save_df.to_csv(
        os.path.join(
            data_dir1,
            'annotations',
            'pretraining',
            f'{mode}_presr{sample_rate}.csv',
        ),
        index=False,
    )

    print(
        f'Interval mean: {np.mean(interval_list)}, std: {np.std(interval_list)}'
    )


def preprocess_egoexo4d_forpretraining_mp(
    mode, sample_rate=50, num_processes=None
):
  """Multi-process version of preprocess_egoexo4d_forpretraining"""
  if num_processes is None:
    num_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU free

  data_dir1 = os.path.expanduser('~/data/egoexo4d')
  data_dir2 = os.path.expanduser('~/projects/camera_motion_modeling/egoexo4d')
  save_dir = os.path.join(
      data_dir1,
      'camera_motion_cache',
      'egoexo4d_pretrain',
      f'{mode}_presr{sample_rate}',
  )
  os.makedirs(save_dir, exist_ok=True)

  take_dict = {}
  take_file = os.path.join(data_dir1, 'takes.json')
  with open(take_file, 'r') as f:
    take_list = json.load(f)
  for take_info in take_list:
    take_id = take_info['take_uid']
    take_dict[take_id] = take_info
  print(f'Loading {len(take_dict)} takes')

  train_alpha = 2.2724306610481197
  interval_list = []

  data_file = os.path.join(
      data_dir1, 'annotations', f'atomic_descriptions_{mode}.json'
  )
  with open(data_file, 'r') as f:
    data_dict = json.load(f)['annotations']

  # Prepare data for parallel processing
  chunk_data = [
      (take_dict, take_uid, data_list)
      for take_uid, data_list in data_dict.items()
      if take_uid in take_dict
  ]

  # Process chunks in parallel
  process_func = partial(
      _process_take_chunk,
      data_dir1=data_dir1,
      data_dir2=data_dir2,
      save_dir=save_dir,
      sample_rate=sample_rate,
      train_alpha=train_alpha,
  )

  all_results = []
  with mp.Pool(num_processes) as pool:
    for results in tqdm(
        pool.imap_unordered(process_func, chunk_data),
        total=len(chunk_data),
        desc=f'Processing takes with {num_processes} processes',
    ):
      all_results.extend(results)
      # Update interval list
      intervals = [result[3] for result in results]
      interval_list.extend(intervals)

  # Save results
  save_df = pd.DataFrame(
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
  os.makedirs(
      os.path.join(data_dir1, 'annotations', 'pretraining'), exist_ok=True
  )
  save_df.to_csv(
      os.path.join(
          data_dir1,
          'annotations',
          'pretraining',
          f'{mode}_presr{sample_rate}.csv',
      ),
      index=False,
  )

  print(
      f'Interval mean: {np.mean(interval_list)}, std: {np.std(interval_list)}'
  )


def exclude_action_egoexo4d_pretraining():
  data_dir = os.path.expanduser('~/data/egoexo4d')
  df = pd.read_csv(
      os.path.join(
          data_dir, 'annotations', 'pretraining', 'train_presr50_filtered.csv'
      )
  )

  # Create a dictionary mapping take_id to list of [start_time, end_time, index]
  segment_dict = defaultdict(list)
  for i, row in tqdm(df.iterrows(), total=len(df)):
    take_name = '_'.join(row['save_id'].split('_')[:-2])
    segment_dict[take_name].append([row['start_time'], row['end_time'], i])

  # Load action segments
  action_file = os.path.join(
      data_dir, 'keystep_files', 'json', 'keystep_segment_train.json'
  )
  with open(action_file, 'r') as f:
    data_list = json.load(f)['segments']

  # Keep track of indices to drop
  indices_to_drop = set()

  # Check for overlaps
  for item in tqdm(data_list, total=len(data_list)):
    if item['take_name'] not in segment_dict:
      continue
    action_start = item['start_time']
    action_end = item['end_time']
    # Check all segments in this take for overlap
    for segment in segment_dict[item['take_name']]:
      seg_start, seg_end, idx = segment
      # Check if segments overlap
      if not (seg_end < action_start or seg_start > action_end):
        indices_to_drop.add(idx)

  # Drop overlapping segments and save filtered DataFrame
  filtered_df = df.drop(index=list(indices_to_drop))
  print(f'Filtered DataFrame size: {len(filtered_df)}')
  print(f'Dropped {len(indices_to_drop)} overlapping segments')

  # Save filtered DataFrame
  save_path = os.path.join(
      data_dir, 'annotations', 'pretraining', 'train_presr50_filtered.csv'
  )
  filtered_df.to_csv(save_path, index=False)
  print(f'Saved filtered DataFrame to {save_path}')


def clean_egoexo4d_pretraining_file():
  data_dir = os.path.expanduser('~/data/egoexo4d')
  for mode in ['train', 'val']:
    fn = '_filtered_clean' if mode == 'train' else ''
    df = pd.read_csv(
        os.path.join(
            data_dir, 'annotations', 'pretraining', f'{mode}_presr50{fn}.csv'
        )
    )
    print(f'Original DataFrame size: {len(df)}')
    df = df[df['description_text'].apply(lambda x: isinstance(x, str))]
    print(f'After text filtering DataFrame size: {len(df)}')

    # Create a mask for rows where cache file exists
    valid_rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
      cache_path = os.path.join(
          data_dir,
          'camera_motion_cache',
          'egoexo4d_pretrain',
          f'{mode}_presr50',
          f"{row['save_id']}_relative.npy",
      )
      valid_rows.append(os.path.exists(cache_path))

    # Filter DataFrame to keep only rows where cache exists
    df = df[valid_rows]
    print(f'Final DataFrame size after cache check: {len(df)}')
    df.to_csv(
        os.path.join(
            data_dir,
            'annotations',
            'pretraining',
            f'{mode}_presr50_cleanv1.csv',
        ),
        index=False,
    )


def clean_egovlpv2_egoexo4d_pretraining_file():
  data_dir = os.path.expanduser('~/data/egoexo4d')
  data_file = os.path.join(
      data_dir, 'annotations', 'egoclip_egoexo_v2_paraphrased_seq-20.csv'
  )
  df_orig = pd.read_csv(data_file, sep='\t')
  df = df_orig[
      df_orig['video_uid'].str.contains('aria|Aria', case=False, na=False)
  ]
  print(
      f'Original DataFrame size: {len(df_orig)}, Filtered DataFrame size:'
      f' {len(df)}'
  )
  df['take_name'] = df['video_uid'].str.split('_aria|_Aria').str[0]

  # Filter out rows where trajectory path doesn't exist
  valid_rows = []
  for i, row in tqdm(df.iterrows(), total=len(df)):
    trajectory_path = os.path.join(
        data_dir,
        'takes',
        row['take_name'],
        'trajectory/closed_loop_trajectory.csv',
    )
    if os.path.exists(trajectory_path):
      valid_rows.append(True)
    else:
      valid_rows.append(False)
  df = df[valid_rows]
  print(f'After filtering missing trajectories, DataFrame size: {len(df)}')
  print(f"Number of unique takes: {len(df['take_name'].unique())}")
  df.to_csv(data_file.replace('.csv', '_filtered.csv'), index=False)


def _process_egovlpv2_chunk(chunk_data, data_dir, save_dir, sample_rate):
  """Helper function to process a chunk of data in parallel"""
  df_chunk = chunk_data
  results = []
  save_dict = {}

  for _, row in df_chunk.iterrows():
    take_name = row['take_name']
    description_text = row['clip_text'].replace('#c c', 'C')
    trajectory_path = os.path.join(
        data_dir, 'takes', take_name, 'trajectory/closed_loop_trajectory.csv'
    )
    start_time, end_time = row['clip_start'], row['clip_end']
    save_id = f'{take_name}_{start_time:.2f}_{end_time:.2f}'

    if save_id not in save_dict:
      try:
        take_df = pd.read_csv(trajectory_path)
        take_df['relative_time'] = (
            take_df['tracking_timestamp_us']
            - take_df['tracking_timestamp_us'].iloc[0]
        ) / 1e6

        start_idx = take_df[take_df['relative_time'] >= start_time].index[0]
        end_idx = take_df[take_df['relative_time'] >= end_time].index[0]

        sub_df = take_df.iloc[start_idx:end_idx]
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
        np.save(
            os.path.join(save_dir, f'{save_id}_relative.npy'), relative_pose
        )
        save_dict[save_id] = relative_pose

      except Exception as e:
        print(
            f'Error processing take {take_name}: {e}, start_time: {start_time},'
            f' end_time: {end_time}'
        )
        continue

    results.append(
        [save_id, description_text, start_time, end_time, take_name, 'v2']
    )

  return results, save_dict


def process_egovlpv2_egoexo4d_pretraining_file_mp(
    sample_rate=50, num_processes=None
):
  """Multi-process version of process_egovlpv2_egoexo4d_pretraining_file"""
  if num_processes is None:
    num_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU free

  save_dir = os.path.expanduser(
      f'~/camera_motion_cache/egoexo4d_pretrain/train_presr{sample_rate}_egoexovlpv2'
  )
  os.makedirs(save_dir, exist_ok=True)

  data_dir = os.path.expanduser('~/data/egoexo4d')
  data_file = os.path.join(
      data_dir,
      'annotations',
      'egoclip_egoexo_v2_paraphrased_seq-20_filtered.csv',
  )
  df = pd.read_csv(data_file)
  df['take_name'] = df['video_uid'].str.split('_aria|_Aria').str[0]

  # Split dataframe into chunks for parallel processing
  chunk_size = len(df) // num_processes
  chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

  # Process chunks in parallel
  all_results = []
  all_save_dicts = {}

  with mp.Pool(num_processes) as pool:
    process_func = partial(
        _process_egovlpv2_chunk,
        data_dir=data_dir,
        save_dir=save_dir,
        sample_rate=sample_rate,
    )

    for results, save_dict in tqdm(
        pool.imap_unordered(process_func, chunks),
        total=len(chunks),
        desc=f'Processing chunks with {num_processes} processes',
    ):
      all_results.extend(results)
      all_save_dicts.update(save_dict)

  # Save results
  save_df = pd.DataFrame(
      all_results,
      columns=[
          'save_id',
          'description_text',
          'start_time',
          'end_time',
          'take_name',
          'version',
      ],
  )
  save_df.to_csv(
      os.path.join(
          data_dir,
          'annotations',
          'pretraining',
          f'train_presr{sample_rate}_egoexovlpv2.csv',
      ),
      index=False,
  )

  print(f'Processed {len(all_results)} samples')
  print(f'Saved {len(all_save_dicts)} unique motion sequences')


def preprocess_egoexo4d_audio():
  data_dir = os.path.expanduser('~/data/egoexo4d')
  # df = pd.read_csv(os.path.join(data_dir, 'annotations/pretraining/test_presr50_sampled1000_True_mcqv0.csv'))
  df = pd.read_csv(
      os.path.join(data_dir, 'annotations/pretraining/test_alltasks_mcqv0.csv')
  )
  df['take_name'] = df['save_id'].str.split('_').str[:-2].str.join('_')

  with open(f'{data_dir}/takes.json', 'r') as f:
    takes = json.load(f)
  take_name_to_uid = {}
  for take in takes:
    take_name_to_uid[take['take_name']] = take['take_uid']

  # Create audio cache directory
  audio_cache_dir = os.path.join(data_dir, 'audio_cache', 'test_alltasks')
  os.makedirs(audio_cache_dir, exist_ok=True)

  for i, row in tqdm(df.iterrows(), total=len(df)):
    take_name = row['take_name']
    segment_path = os.path.join(audio_cache_dir, f"{row['save_id']}.wav")
    start, end = row['t0'], row['t1']  # row['start_time'], row['end_time']
    if os.path.exists(segment_path) and start > 0:
      continue
    start = max(start, 0)
    end = max(end, start + 0.1)

    audio_files = glob.glob(
        os.path.join(data_dir, 'takes', take_name, 'audio', '*.wav')
    )
    if len(audio_files) != 1:
      print(f'Error! Found {len(audio_files)} from {take_name}')
      continue

    # Load audio file
    audio = AudioSegment.from_wav(audio_files[0])
    # Convert start and end to milliseconds
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    # Extract segment
    audio_segment = audio[start_ms:end_ms]
    # Save segment to cache
    audio_segment.export(segment_path, format='wav')


def load_camera_name_mapping(data_file):
  camera_name_mapping = {}
  with open(data_file, 'r') as f:
    data_list = json.load(f)
  for data in data_list:
    keys = list(data['frame_aligned_videos'].keys())
    camera_name_mapping[data['take_name']] = data['frame_aligned_videos'][
        keys[0]
    ]['rgb']['relative_path']
  return camera_name_mapping


def split_dynpose_idx():
  data_file = os.path.expanduser(
      '~/local_data/dynpose-100k/dynpose_100k/valid_indices.npy'
  )
  valid_indices = np.load(data_file)
  print(f'{len(valid_indices)} valid indices')
  np.random.seed(42)
  val_indices = np.random.choice(valid_indices, size=5000, replace=False)
  train_indices = np.setdiff1d(valid_indices, val_indices)
  print(f'Train indices: {len(train_indices)}, Val indices: {len(val_indices)}')
  np.save(
      data_file.replace('valid_indices.npy', 'train_indices.npy'), train_indices
  )
  np.save(
      data_file.replace('valid_indices.npy', 'val_indices.npy'), val_indices
  )


def _copy_file_task(task):
  """Helper function for multiprocessing file copying"""
  source_file, target_file = task
  try:
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    shutil.copy2(source_file, target_file)
    return True
  except Exception as e:
    print(f'Error copying {source_file}: {e}')
    return False


def copy_data_to_local():
  data_dir = os.path.expanduser('~/data/egoexo4d/')
  with open(os.path.join(data_dir, 'takes.json'), 'r') as f:
    data_list = json.load(f)
  take_list = [data['take_name'] for data in data_list]
  take_list = list(set(take_list))
  files = [
      os.path.join(
          data_dir, 'takes', take, 'trajectory', 'closed_loop_trajectory.csv'
      )
      for take in take_list
  ]

  # Prepare copy tasks
  copy_tasks = []
  for file in files:
    target_file = file.replace('data/egoexo4d', 'local_data/egoexo4d')
    if not os.path.exists(target_file):
      copy_tasks.append((file, target_file))

  print(f'Need to copy {len(copy_tasks)} files')

  if not copy_tasks:
    print('All files already exist locally')
    return

  # Use multiprocessing to copy files in parallel
  num_processes = min(mp.cpu_count(), len(copy_tasks))
  print(f'Using {num_processes} processes')

  with mp.Pool(num_processes) as pool:
    results = list(
        tqdm(
            pool.imap_unordered(_copy_file_task, copy_tasks),
            total=len(copy_tasks),
            desc='Copying files',
        )
    )

  successful_copies = sum(results)
  print(f'Successfully copied {successful_copies}/{len(copy_tasks)} files')


def check_egoexo4d_gravity_direction():
  data_dir = os.path.expanduser('~/data/egoexo4d/')
  with open(os.path.join(data_dir, 'takes.json'), 'r') as f:
    data_list = json.load(f)
  take_list = [data['take_name'] for data in data_list]
  take_list = list(set(take_list))
  files = [
      os.path.join(
          data_dir.replace('data', 'local_data'),
          'takes',
          take,
          'trajectory_presr50',
          'closed_loop_trajectory.csv',
      )
      for take in take_list
  ]
  err_cnt, cnt = 0, 0
  for file in tqdm(files):
    if not os.path.exists(file):
      continue
    df = pd.read_csv(file)
    check_x = (df['gravity_x_world'] == 0).all()
    check_y = (df['gravity_y_world'] == 0).all()
    check_z = (df['gravity_z_world'] == -9.81).all()
    if int(check_x) + int(check_y) + int(check_z) != 3:
      print(f'Problem for {file}, {check_x}, {check_y}, {check_z}')
      err_cnt += 1
    cnt += 1
  print(f'{err_cnt} / {cnt} files have gravity direction issues.')


def check_nymeria_gravity_direction():
  data_dir = os.path.expanduser('~/local_data/nymeria/')
  with open(
      f'{data_dir}/Nymeria_download_urls_motion_narration.json', 'r'
  ) as f:
    data = json.load(f)
  err_cnt, cnt = 0, 0
  for key, value in tqdm(data['sequences'].items()):
    file = f'{data_dir}/motion_data/{key}/recording_head/mps/slam/closed_loop_trajectory.csv'
    df = pd.read_csv(file)
    check_x = (df['gravity_x_world'] == 0).all()
    check_y = (df['gravity_y_world'] == 0).all()
    check_z = (df['gravity_z_world'] == -9.81).all()
    if int(check_x) + int(check_y) + int(check_z) != 3:
      print(f'Problem for {file}, {check_x}, {check_y}, {check_z}')
      err_cnt += 1
    cnt += 1
  print(f'{err_cnt} / {cnt} files have gravity direction issues.')


def _extract_frame_worker(row, data_dir, fps):
  """Helper function to process a single video for frame extraction."""
  video_fp = os.path.join(data_dir, 'keystep/clips_448p', row['video_fp'])
  output_dir = os.path.join(
      data_dir,
      f'keystep/frames_{fps}fps',
      row['video_fp'].replace('/', '_').replace('.mp4', ''),
  )
  if os.path.exists(output_dir):
    return False  # Already processed
  assert os.path.exists(video_fp), f'Video file not found: {video_fp}'
  os.makedirs(output_dir, exist_ok=True)
  cmd = f'ffmpeg -i {video_fp} -r {fps} {output_dir}/%06d.png'
  try:
    # Using DEVNULL to suppress output, check=True will raise on error
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True
  except subprocess.CalledProcessError as e:
    print(f'ffmpeg failed for {video_fp}: {e}')
    # Clean up partially created directory if ffmpeg fails
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    return False


def extract_keystep_frames(fps=5, num_workers=None):
  if num_workers is None:
    num_workers = mp.cpu_count()
  data_dir = os.path.expanduser('~/data/egoexo4d')
  df = pd.read_csv(f'{data_dir}/annotations/downstream/action_cls.csv')
  df['dur'] = df['end_time'] - df['start_time']
  # df = df[df['dur'] < 10]
  df = df.sample(frac=1, random_state=42).reset_index(drop=True)
  tasks = [row for _, row in df.iterrows()]
  worker_func = partial(_extract_frame_worker, data_dir=data_dir, fps=fps)
  with mp.Pool(num_workers) as pool:
    results = list(
        tqdm(
            pool.imap_unordered(worker_func, tasks),
            total=len(tasks),
            desc='Extracting frames',
        )
    )
  print(f'Finished. Processed {sum(results)} new videos.')


def _extract_ucf_frame_worker(video_file, data_dir, fps):
  """Helper function to process a single UCF101 video for frame extraction."""
  save_dir = f'{data_dir}/frames_{fps}fps'
  seq = os.path.basename(video_file).replace('.mp4', '')
  seq_save_dir = os.path.join(save_dir, seq)
  if os.path.exists(seq_save_dir):
    return f'Skipping {seq}, directory already exists.'

  os.makedirs(seq_save_dir, exist_ok=True)
  cmd = f'ffmpeg -i {video_file} -r {fps} {seq_save_dir}/%06d.png'
  try:
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return f'Extracted frames for {seq} to {seq_save_dir}'
  except subprocess.CalledProcessError as e:
    return f'ffmpeg failed for {video_file}: {e}'


def extract_ucf_frames(fps=5, num_workers=None):
  if num_workers is None:
    num_workers = os.cpu_count() or 1
  data_dir = 'local_data/ucf101'
  video_files = glob.glob(f'{data_dir}/mp4_videos/*.mp4')
  worker_func = partial(_extract_ucf_frame_worker, data_dir=data_dir, fps=fps)
  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    list(
        tqdm(
            executor.map(worker_func, video_files),
            total=len(video_files),
            desc='Extracting UCF101 frames',
        )
    )


def crop_clip_mid(in_fp, out_fp, clip_len, video_dur=None):
  """Extract a clip of length `clip_len` (seconds) centered at the video's midpoint.

  Fast path (stream copy); may cut on nearest keyframe.

  Args:
      in_fp: input video path
      out_fp: output clip path
      clip_len: desired clip length in seconds (e.g., 2, 4, 6, ...)
      video_dur: total duration of input video in seconds (optional; if None,
        probes)
  """
  if clip_len <= 0:
    raise ValueError('clip_len must be > 0')

  if video_dur is None:
    video_dur = float(
        subprocess.check_output([
            'ffprobe',
            '-v',
            'error',
            '-show_entries',
            'format=duration',
            '-of',
            'default=noprint_wrappers=1:nokey=1',
            in_fp,
        ])
        .decode()
        .strip()
    )

  mid = video_dur / 2.0
  start = max(0.0, mid - clip_len / 2.0)
  length = min(clip_len, max(0.0, video_dur - start))

  subprocess.run(
      [
          'ffmpeg',
          '-y',
          '-ss',
          f'{start:.3f}',
          '-i',
          in_fp,
          '-t',
          f'{length:.3f}',
          '-c',
          'copy',
          '-reset_timestamps',
          '1',
          out_fp,
      ],
      check=True,
  )


def _process_scenario_take(take_name, take_mapping, data_dir, fps):
  """Helper function to process a single take for scenario frame extraction."""
  video_name = take_mapping[take_name].replace(
      'frame_aligned_videos/', 'frame_aligned_videos/downscaled/448/'
  )
  video_fp = os.path.join(data_dir, 'takes', take_name, video_name)
  if not os.path.exists(video_fp):
    return f'Warning: video file not found for {take_name}: {video_fp}'

  output_fp = f'{data_dir}/scenario/clips/{take_name}.mp4'
  output_dir = f'{data_dir}/scenario/frames_{fps}fps/{take_name}'
  if os.path.exists(output_fp) and os.path.exists(output_dir):
    return None  # Success, but already done

  try:
    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    crop_clip_mid(video_fp, output_fp, clip_len=4)
    os.makedirs(output_dir, exist_ok=True)
    cmd = f'ffmpeg -i {output_fp} -r {fps} {output_dir}/%06d.png'
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return f'Successfully processed {take_name}'
  except Exception as e:
    return f'Error processing {take_name}: {e}'


def extract_scenario_frames(fps=5, num_workers=None):
  if num_workers is None:
    num_workers = os.cpu_count() or 1

  data_dir = os.path.expanduser('~/data/egoexo4d')
  take_mapping = {}
  with open(f'{data_dir}/takes.json') as f:
    data_list = json.load(f)
  for data_dict in data_list:
    keys = sorted(data_dict['frame_aligned_videos'].keys())
    key = keys[0]
    assert 'aria' in key or 'Aria' in key, print(key)
    take_mapping[data_dict['take_name']] = data_dict['frame_aligned_videos'][
        key
    ]['rgb']['relative_path']
  df = pd.read_csv(f'{data_dir}/annotations/downstream/scenario_cls.csv')
  take_name_list = df['take_name'].unique()
  worker_func = partial(
      _process_scenario_take,
      take_mapping=take_mapping,
      data_dir=data_dir,
      fps=fps,
  )

  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(
        tqdm(
            executor.map(worker_func, take_name_list),
            total=len(take_name_list),
            desc='Extracting scenario frames',
        )
    )

  for result in results:
    if result:
      print(result)


def save_alltake_video():
  data_dir = os.path.expanduser('~/data/egoexo4d')
  with open(f'{data_dir}/takes.json') as f:
    data_list = json.load(f)
  print(len(data_list))
  video_name_list = []
  for i, data_dict in tqdm(enumerate(data_list), total=len(data_list)):
    keys = sorted(data_dict['frame_aligned_videos'].keys())
    key = keys[0]
    assert 'aria' in key or 'Aria' in key, print(key)
    video_name = data_dict['frame_aligned_videos'][key]['rgb'][
        'relative_path'
    ].replace('frame_aligned_videos/', 'frame_aligned_videos/downscaled/448/')
    video_fp = os.path.join(
        data_dir, 'takes', data_dict['take_name'], video_name
    )
    if not os.path.exists(video_fp):
      print(f'Missing {video_fp}')
      continue
    video_name_list.append(video_fp)
  with open(f'{data_dir}/all_take_videos.json', 'w') as f:
    json.dump(video_name_list, f, indent=4)


def _probe_duration_seconds(video_fp: str) -> float:
  cmd = [
      'ffprobe',
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=noprint_wrappers=1:nokey=1',
      video_fp,
  ]
  out = subprocess.check_output(cmd).decode('utf-8').strip()
  return float(out)


def crop_video_into_clips(s):
  """Slice each video into s-second clips under ~/local_data/egoexo4d/clips/<video_stem>/.

  If the last window is shorter than s, keep it as-is.
  """
  data_dir = os.path.expanduser('~/data/egoexo4d')
  local_data_dir = os.path.expanduser('~/local_data/egoexo4d')
  clips_root = os.path.join(local_data_dir, 'clips')

  with open(f'{data_dir}/all_take_videos.json', 'r') as f:
    video_list = json.load(f)

  for in_fp in video_list:
    duration = _probe_duration_seconds(in_fp)
    stem = in_fp.split('/')[-5]
    out_dir = os.path.join(clips_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    n_segs = math.ceil(duration / s)
    print(f'[info] {in_fp} | {duration:.2f}s -> {n_segs} clip(s) | {out_dir}')

    for i in range(n_segs):
      start = i * s
      clip_len = min(s, duration - start)
      if clip_len <= 0:
        continue

      out_fp = os.path.join(out_dir, f'part{i:04d}.mp4')

      cmd = [
          'ffmpeg',
          '-y',
          '-i',
          in_fp,
          '-c',
          'copy',
          '-map',
          '0',
          '-f',
          'segment',
          '-segment_time',
          str(s),
          '-reset_timestamps',
          '1',
          '-movflags',
          '+faststart',
          os.path.join(out_dir, 'part%04d.mp4'),
      ]
      subprocess.run(cmd, check=True)

  print('Done.')


def check_megasam_results():
  data_dir = 'data/egoexo4d/annotations/downstream'
  result_dir = 'baselines/mega-sam/outputs_cvd'
  for fn in ['action']:  # ['action', 'scenario']:
    data_file = f'{data_dir}/{fn}_cls.csv'
    df = pd.read_csv(data_file)

    if fn == 'action':
      names = (
          df['video_fp']
          .str.replace('/', '_', regex=False)
          .str.replace('.mp4', '', regex=False)
      )
    else:
      names = df['take_name']

    paths = [f'{result_dir}/{n}_sgd_cvd_hr.npz' for n in names]
    exists = pd.Series([os.path.exists(p) for p in paths])
    miss_df = df.loc[~exists].copy()
    out_file = data_file.replace('.csv', '_remaining.csv')
    miss_df.to_csv(out_file, index=False)
    print(
        f'{fn}: {exists.sum()}/{len(df)} present; {len(miss_df)} missing ->'
        f' saved to {out_file}'
    )


def check_megasam_results_scenario():
  data_dir = 'data/egoexo4d/annotations/downstream'
  result_dir = 'baselines/mega-sam/outputs_cvd'
  data_file = f'{data_dir}/scenario_cls.csv'
  df = pd.read_csv(data_file)
  for i, row in tqdm(df.iterrows(), total=len(df)):
    name = row['take_name']
    path = f'{result_dir}/{name}_sgd_cvd_hr.npz'
    if not os.path.exists(path):
      print(f'Row {i} missing')


def check_megasam_results_action():
  data_dir = 'data/egoexo4d/annotations/downstream'
  result_dir1 = 'baselines/mega-sam/outputs'
  result_dir2 = 'baselines/mega-sam/outputs_cvd'
  data_file = f'{data_dir}/action_cls.csv'
  df = pd.read_csv(data_file)
  # miss_rows = []
  pred_paths = []
  for i, row in tqdm(df.iterrows(), total=len(df)):
    name = row['video_fp'].replace('/', '_').replace('.mp4', '')
    path = f'{result_dir2}/{name}_sgd_cvd_hr.npz'
    if not os.path.exists(path):
      path = f'{result_dir1}/{name}_droid.npz'
      if not os.path.exists(path):
        # miss_rows.append(row)
        path = None
    pred_paths.append(path)

  # miss_df = pd.DataFrame(miss_rows)
  # miss_df.to_csv(data_file.replace('.csv', '_remaining_2.csv'), index=False)
  # print(f'{len(miss_rows)} rows missing')

  df['pred_path'] = pred_paths
  new_df = df[df['pred_path'].notnull()].copy()
  new_df.to_csv(
      data_file.replace('.csv', '_with_megasam_pred.csv'), index=False
  )
  print(f'Original df len = {len(df)}, with pred len = {len(new_df)}')


def check_vipeandpi3_results_action():
  data_dir = 'data/egoexo4d/annotations/downstream'
  data_file = f'{data_dir}/action_cls_with_megasam_pred.csv'
  df = pd.read_csv(data_file)
  cnt = 0
  pred_paths = []

  # df["name"] = df["video_fp"].apply(lambda x: os.path.basename(x).replace(".mp4", ""))
  # unique_names = df["name"].unique()
  # print(f"Unique video names: {len(unique_names)}, Total rows: {len(df)}")

  for i, row in df.iterrows():
    name = os.path.basename(row['video_fp']).replace('.mp4', '')
    path = f'baselines/vipe/vipe_results/pose/{name}.npz'
    pi3_path = os.path.join(
        'baselines/Pi3/preds_action_new',
        row['video_fp'].replace('.mp4', '').replace('/', '_') + '.npy',
    )
    if not os.path.exists(path) or not os.path.exists(pi3_path):
      # print(f"Row {i} missing: {path}")
      cnt += 1
      path = None
    pred_paths.append(path)

  df['vipe_pred_path'] = pred_paths
  new_df = df[df['vipe_pred_path'].notnull()].copy()
  new_df.to_csv(
      data_file.replace('_with_megasam_pred.csv', '_with_all3_pred.csv'),
      index=False,
  )
  print(f'Original df len = {len(df)}, saving vipe pred len = {len(new_df)}')


def check_camest_results_scenario():
  data_file = 'data/egoexo4d/annotations/downstream/scenario_cls.csv'
  df = pd.read_csv(data_file)
  rows = []
  for i, row in tqdm(df.iterrows(), total=len(df)):
    megasam_path = os.path.join(
        'baselines/mega-sam/outputs_cvd', f"{row['take_name']}_sgd_cvd_hr.npz"
    )
    vipe_path = os.path.join(
        'baselines/vipe/vipe_results/pose', f"{row['take_name']}.npz"
    )
    pi3_path = os.path.join(
        'baselines/Pi3/preds_scenario', f"{row['take_name']}.npy"
    )
    if (
        os.path.exists(megasam_path)
        and os.path.exists(vipe_path)
        and os.path.exists(pi3_path)
    ):
      row = row.copy()
      row['megasam_path'] = megasam_path
      row['vipe_path'] = vipe_path
      row['pi3_path'] = pi3_path
      rows.append(row)
    else:
      print(
          f'Row {i} missing: megasam: {os.path.exists(megasam_path)}, vipe:'
          f' {os.path.exists(vipe_path)}, pi3: {os.path.exists(pi3_path)}'
      )

  new_df = pd.DataFrame(rows)
  print(f'Found {len(new_df)} valid rows out of {len(df)}')
  new_df.to_csv(
      'data/egoexo4d/annotations/downstream/scenario_cls_with_all3_pred.csv',
      index=False,
  )


def reformat_ucf_video():
  data_dir = os.path.expanduser('~/local_data/ucf101/UCF-101')
  dest_path = data_dir.replace('UCF-101', 'mp4_videos')
  os.makedirs(dest_path, exist_ok=True)
  video_files = glob.glob(os.path.join(data_dir, '*/*.avi'))
  print(f'Found {len(video_files)} video files')
  for video_file in video_files:
    # save_folder = os.path.dirname(video_file).split('/')[-1]
    # os.makedirs(os.path.join(dest_path, save_folder), exist_ok=True)
    cmd = (
        f'ffmpeg -i {video_file} -c:v libx264 -c:a aac'
        f" {dest_path}/{os.path.basename(video_file).replace('.avi', '.mp4')}"
    )
    os.system(cmd)


def filter_by_action(df):
  # Compute mean metrics
  df['mean_ate'] = df[
      ['ate_megasam_pi3', 'ate_megasam_vipe', 'ate_pi3_vipe']
  ].mean(axis=1)
  df['mean_rpe_t'] = df[
      ['rpe_t_megasam_pi3', 'rpe_t_megasam_vipe', 'rpe_t_pi3_vipe']
  ].mean(axis=1)
  df['mean_rpe_r'] = df[
      ['rpe_r_deg_megasam_pi3', 'rpe_r_deg_megasam_vipe', 'rpe_r_deg_pi3_vipe']
  ].mean(axis=1)

  # Apply 0.9 quantile filtering within each action group
  def filter_group(g):
    q_ate = g['mean_ate'].quantile(0.9)
    q_rpe_t = g['mean_rpe_t'].quantile(0.9)
    q_rpe_r = g['mean_rpe_r'].quantile(0.9)
    return g[
        (g['mean_ate'] < q_ate)
        & (g['mean_rpe_t'] < q_rpe_t)
        & (g['mean_rpe_r'] < q_rpe_r)
    ]

  df_filtered = df.groupby('action', group_keys=False).apply(filter_group)
  return df_filtered


def filter_ucf(split='01'):
  data_dir = os.path.expanduser('~/data/UCF101-ZIP/ucfTrainTestlist')
  df = pd.read_csv(f'{data_dir}/cmp_3pose.csv')
  action = [
      'Skijet',
      'SkateBoarding',
      'IceDancing',
      'Rafting',
      'SkyDiving',
      'LongJump',
      'Biking',
      'Skiing',
      'Kayaking',
      'JavelinThrow',
  ]
  action_v4 = [
      'Skijet',
      'SkateBoarding',
      'Knitting',
      'MoppingFloor',
      'WalkingWithDog',
      'Lunges',
      'MilitaryParade',
      'SoccerPenalty',
  ]
  df['action'] = df['seq'].apply(lambda x: x.split('_')[1])
  df = df[df['action'].isin(action_v4)].copy()

  # df["mean_ate"] = df[["ate_megasam_pi3","ate_megasam_vipe","ate_pi3_vipe"]].mean(axis=1)
  # df["mean_rpe_t"] = df[["rpe_t_megasam_pi3","rpe_t_megasam_vipe","rpe_t_pi3_vipe"]].mean(axis=1)
  # df["mean_rpe_r"] = df[["rpe_r_deg_megasam_pi3","rpe_r_deg_megasam_vipe","rpe_r_deg_pi3_vipe"]].mean(axis=1)
  # keep_ate = df["mean_ate"] < df["mean_ate"].quantile(0.9)
  # keep_rpe_t = df["mean_rpe_t"] < df["mean_rpe_t"].quantile(0.9)
  # keep_rpe_r = df["mean_rpe_r"] < df["mean_rpe_r"].quantile(0.9)
  # sub_df = df[keep_ate & keep_rpe_t & keep_rpe_r]

  sub_df = filter_by_action(df)
  print(sub_df['action'].value_counts())
  print(f'Filtered {len(sub_df)} videos from {len(df)}')

  for mode in ['train', 'test']:
    list_file = f'{data_dir}/{mode}list{split}.txt'
    with open(list_file, 'r') as f:
      lines = f.readlines()
    print(f'Original {mode} list has {len(lines)} videos')
    filtered_lines = [
        line
        for line in lines
        if line.split('/')[1].split('.')[0] in sub_df['seq'].values
    ]
    print(f'Filtered {mode} list has {len(filtered_lines)} videos')
    with open(list_file.replace('.txt', '_filtered_v4.txt'), 'w') as f:
      f.writelines(filtered_lines)


def get_pennaction_video():
  data_dir = os.path.expanduser('~/local_data/Penn_Action')
  for dir_name in os.listdir(f'{data_dir}/frames'):
    cmd = (
        f'ffmpeg -y -framerate 25 -i {data_dir}/frames/{dir_name}/%06d.jpg -c:v'
        f' libx264 {data_dir}/videos/{dir_name}.mp4'
    )
    os.system(cmd)


def check_egoexo4d_pi3pred():
  video_dir = 'local_data/egoexo4d/clips'
  pi3_pred_dir = 'baselines/pi3/preds_all'
  save_dict = {}
  for take_name in tqdm(os.listdir(pi3_pred_dir)):
    files = sorted(glob.glob(f'{pi3_pred_dir}/{take_name}/*.npy'))
    segments = [0]
    for file in files:
      video_file = os.path.join(
          video_dir, take_name, os.path.basename(file).replace('.npy', '.mp4')
      )
      duration = float(
          subprocess.check_output([
              'ffprobe',
              '-v',
              'error',
              '-show_entries',
              'format=duration',
              '-of',
              'default=noprint_wrappers=1:nokey=1',
              video_file,
          ])
          .decode()
          .strip()
      )
      # pred = np.load(file)
      # sr = pred.shape[0] / duration
      segments.append(segments[-1] + duration)
    save_dict[take_name] = segments
  with open('data/egoexo4d/annotations/pi3_pose_segments.json', 'w') as f:
    json.dump(save_dict, f, indent=4)


def prepare_egoexo4d_pi3file():
  data_dir = os.path.expanduser('~/data/egoexo4d/annotations/')
  pi3_pred_dir = 'baselines/pi3/preds_all'
  with open(f'{data_dir}/pi3_pose_segments.json', 'r') as f:
    segment_dict = json.load(f)
  files = ['test_alltasks_mcqv0.csv']  # ['train_v2.csv', 'val_v2.csv']

  for file in files:
    save_rows = []
    df = pd.read_csv(f'{data_dir}/pretraining/{file}')
    df['take_name'] = df['save_id'].apply(lambda x: '_'.join(x.split('_')[:-2]))
    for i, row in tqdm(df.iterrows(), total=len(df)):
      segment_list = segment_dict[row['take_name']]
      ts = row['timestamp']
      i = bisect.bisect_right(segment_list, ts) - 1
      if i < 0 or i >= len(segment_list) - 1:
        print(
            f"segment list error for {row['take_name']}, ts: {ts},"
            f' segment_list: {segment_list}, i: {i}'
        )
        continue
      if ts == segment_list[i + 1]:
        i = max(0, i - 1)
      pose_file = os.path.join(
          pi3_pred_dir, row['take_name'], f'part{i:04d}.npy'
      )
      assert os.path.exists(pose_file), (
          f'Pose file not found: {pose_file}, segment_list: {segment_list}, ts:'
          f' {ts}, i: {i}'
      )
      pose_len = np.load(pose_file).shape[0]
      sr = pose_len / (segment_list[i + 1] - segment_list[i])

      t_start = segment_list[i]
      t0_idx = max(0, int((row['t0'] - t_start) * sr))
      t1_idx = min(int((row['t1'] - t_start) * sr), pose_len - 1)
      if t0_idx == 0 and t1_idx == 0:
        t1_idx = 1
      if t0_idx == pose_len - 1 and t1_idx == pose_len - 1:
        t0_idx = pose_len - 2
      if t0_idx >= pose_len - 1 or t1_idx <= 0 or t0_idx >= t1_idx:
        print(
            f"Error: t0 {row['t0']}, t1 {row['t1']}, t0_idx {t0_idx}, t1_idx"
            f' {t1_idx}, pose_len {pose_len}'
        )
        continue
      ts_idx = int((row['timestamp'] - t_start) * sr)

      save_rows.append({
          'pose_file': pose_file,
          't0_idx': t0_idx,
          't1_idx': t1_idx,
          'timestamp_idx': ts_idx,
          'sr': sr,
          'cam_traj_len': pose_len,
          'description_text': row['description_text'],
      })
    save_df = pd.DataFrame(save_rows)
    print(f'Processed {len(save_df)}/{len(df)} rows for {file}')
    save_df.to_csv(
        f"{data_dir}/pretraining/{file.replace('.csv', '_pi3pose.csv')}",
        index=False,
    )


if __name__ == '__main__':
  # clean_egovlpv2_egoexo4d_pretraining_file()
  # process_egovlpv2_egoexo4d_pretraining_file_mp()

  # data_dir = os.path.expanduser('~/hd-epic_action')
  # label_dist(os.path.join(data_dir, 'ann_files/HD_EPIC_Narrations_test_filtered.csv'), os.path.join(data_dir, 'predictions/motionformer_predictions.csv'))
  # split_by_participant(data_dir)
  # egoexo4d_label_dist('train')
  # hdepic_label_dist('test')
  # update_egoexo4d_labelv1()

  # reformat_egoexo4d_label('data/egoexo4d/keystep_files/label_mapping.csv')
  # ana_label('data/egoexo4d/keystep_files/label_mapping_reformat.csv')
  # create_egoexo4d_video_train_file('v2')
  # create_hdepic_video_train_file()

  # Use single-process version
  # preprocess_egoexo4d_forpretraining('train')
  # Or use multi-process version
  # preprocess_egoexo4d_forpretraining_mp('train', num_processes=None)  # Adjust number of processes as needed

  # exclude_action_egoexo4d_pretraining()
  # clean_egoexo4d_pretraining_file()
  # preprocess_vlm_mcq_csv_to_json_nymeria()
  # preprocess_dynpose_to_json()
  # preprocess_camerabench_mcq()
  # check_camerabench_textoverlap()
  # split_dynpose_idx()
  # preprocess_dynpose_val_to_mcq()
  # preprocess_dynpose_to_json3()
  # preprocess_camerabench_to_vqa()
  # read_dynpose_similartext()
  # read_dynpose_vtalign()

  # preprocess_egoexo4d_audio()
  # copy_data_to_local()

  # check_egoexo4d_gravity_direction()
  # check_nymeria_gravity_direction()
  # check_megasam_results_scenario()
  # check_megasam_results_action()
  # check_vipeandpi3_results_action()
  # check_camest_results_scenario()

  # extract_keystep_frames(num_workers=32)
  # extract_scenario_frames(num_workers=32)
  # extract_ucf_frames(num_workers=32)
  # save_alltake_video()
  # crop_video_into_clips(60)

  # reformat_ucf_video()
  filter_ucf()
  # get_pennaction_video()
  # check_egoexo4d_pi3pred()
  # prepare_egoexo4d_pi3file()
