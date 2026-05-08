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

from collections import Counter, defaultdict
import glob
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = os.path.expanduser('~/data/FineGym')
LOCAL_DATA_DIR = os.path.expanduser('~/local_data/FineGym')


def read_ann(download=False):
  # 1: vault, 2: floor exercise, 3: balance beam, 4, uneven bar
  df = pd.read_csv(f'{DATA_DIR}/annotations/links.csv')
  videoid_to_gid = {
      row['name'].split('.')[0]: row['id'] for _, row in df.iterrows()
  }
  videoid_to_name = {
      row['name'].split('.')[0]: row['name'] for _, row in df.iterrows()
  }
  with open(
      f'{DATA_DIR}/annotations/finegym_annotation_info_v1.0.json', 'r'
  ) as f:
    ann = json.load(f)

  print(f'Loading {len(ann)} annotations')
  duration_dict = defaultdict(list)
  video_id_list = []
  for video_id, event_dict in ann.items():
    event_list = []
    for key, value in event_dict.items():
      if value['event'] not in event_list:
        event_list.append(value['event'])
      assert len(value['timestamps']) == 1
      duration = value['timestamps'][0][1] - value['timestamps'][0][0]
      duration_dict[value['event']].append(duration)
    common_list = set(event_list) & set(range(1, 5))
    if len(common_list) != 4:
      continue
    video_id_list.append(video_id)

  duration_dict = dict(sorted(duration_dict.items(), key=lambda item: item[0]))
  total_duration = 0
  for event, durations in duration_dict.items():
    if event in [1, 2, 3, 4]:
      total_duration += sum(durations)
    avg_duration = sum(durations) / len(durations)
    print(
        f'Event: {event}, Average Duration: {avg_duration:.2f} seconds over'
        f' {len(durations)} instances'
    )
  print(f'Total Duration: {total_duration:.2f} seconds')
  return

  print(f'Found {len(video_id_list)} videos with all 4 events')
  video_name_list = [videoid_to_name[vid] for vid in video_id_list]
  with open(f'{DATA_DIR}/annotations/video_name_list.json', 'w') as f:
    json.dump(video_name_list, f, indent=2)

  if download:
    for video_id in video_id_list:
      fn = videoid_to_name[video_id]
      if os.path.exists(f'{DATA_DIR}/videos/{fn}'):
        print(f'File exists: {fn}')
        continue
      gid = videoid_to_gid[video_id]
      print(f'{video_id} -> {gid}')
      cmd = f'gdrive files download {gid} --destination {DATA_DIR}/videos/'
      os.system(cmd)


def crop_video():
  with open(f'{DATA_DIR}/annotations/video_name_list.json', 'r') as f:
    video_name_list = json.load(f)
  with open(
      f'{DATA_DIR}/annotations/finegym_annotation_info_v1.0.json', 'r'
  ) as f:
    ann = json.load(f)
  for fn in video_name_list:
    video_file = f'{DATA_DIR}/videos/{fn}'
    if not os.path.exists(video_file):
      continue
    save_dir = f'{LOCAL_DATA_DIR}/clips/{fn.split(".")[0]}'
    os.makedirs(save_dir, exist_ok=True)
    ann_dict = ann[fn.split('.')[0]]
    for key, value in ann_dict.items():
      event = value['event']
      start_time, end_time = value['timestamps'][0]
      save_name = f'{key}_event{event}.mp4'
      save_path = os.path.join(save_dir, save_name)
      if os.path.exists(save_path):
        print(f'File exists: {save_path}')
        continue
      cmd = (
          f'ffmpeg -y -ss {start_time} -i {video_file} -t'
          f' {end_time - start_time} -c copy {save_path}'
      )
      os.system(cmd)


def scene_detect(data_dir):
  from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg

  video_files = glob.glob(f'{data_dir}/*/*.mp4')
  for video_file in video_files:
    print(f'Processing {video_file}')
    shot_path = os.path.dirname(video_file).replace('clips', 'shots')
    os.makedirs(shot_path, exist_ok=True)
    scene_list = detect(video_file, AdaptiveDetector())
    if len(scene_list) == 0:
      print(f'No scenes detected for {video_file}')
      cmd = f'cp {video_file} {shot_path}'
      os.system(cmd)
      continue
    print(f'Detected {len(scene_list)} scenes')
    split_video_ffmpeg(video_file, scene_list, output_dir=shot_path)


def check_video_len():
  videos = glob.glob(f'{LOCAL_DATA_DIR}/shots/*/*.mp4')
  print(f'Found {len(videos)} videos')
  rows = []
  for video in tqdm(videos):
    event_label = (
        video.split('_event')[1].split('-Scene')[0].replace('.mp4', '')
    )
    video_len = (
        os.popen(
            'ffprobe -v error -select_streams v:0 -show_entries'
            f' stream=duration -of default=noprint_wrappers=1:nokey=1 "{video}"'
        )
        .read()
        .strip()
    )
    rows.append({
        'video': video,
        'event_label': event_label,
        'video_length': float(video_len),
    })
  df = pd.DataFrame(rows)
  print(df['video_length'].describe())
  df.to_csv(f'{DATA_DIR}/finegym_shots_stat.csv', index=False)


def split():
  df = pd.read_csv(f'{DATA_DIR}/annotations/finegym_shots_stat.csv')
  df = df[df['event_label'].isin([1, 2, 3, 4])]
  df['pi3_path'] = df['video'].apply(
      lambda x: os.path.join(
          LOCAL_DATA_DIR,
          'pi3_poses',
          os.path.basename(x).replace('.mp4', '.npy'),
      )
  )
  df = df[df['pi3_path'].apply(lambda x: os.path.exists(x))]
  df['video_id'] = df['video'].apply(lambda x: x.split('/')[-2])
  unique_videos = df['video_id'].unique()
  print(
      f'{len(unique_videos)} unique videos, {len(df)} total clips,'
      f" {df['video_length'].mean():.2f} seconds"
  )
  np.random.seed(42)
  np.random.shuffle(unique_videos)
  num_videos = len(unique_videos)
  train_split = int(0.8 * num_videos)
  train_videos = unique_videos[:train_split]
  val_videos = unique_videos[train_split:]
  train_df = df[df['video_id'].isin(train_videos)]
  val_df = df[df['video_id'].isin(val_videos)]
  print(
      f'Train set: {len(train_df)} clips from {len(train_videos)} videos, Val'
      f' set: {len(val_df)} clips from {len(val_videos)} videos'
  )
  train_df.to_csv(
      f'{DATA_DIR}/annotations/finegym_train_split.csv', index=False
  )
  val_df.to_csv(f'{DATA_DIR}/annotations/finegym_val_split.csv', index=False)
  print(val_df['event_label'].value_counts())


def crop_gym99_video_plmm():
  with open(
      f'{DATA_DIR}/annotations/finegym_annotation_info_v1.0.json', 'r'
  ) as f:
    ann = json.load(f)
  df = pd.read_csv(
      f'{DATA_DIR}/annotations/gym99_train_element_v1.0.txt',
      sep=' ',
      header=None,
      names=['name', 'label'],
  )
  df['element'] = df['name'].apply(lambda x: '_'.join(x.split('_')[:4]))
  print(
      f"Total {len(df['element'].unique())} unique elements,"
      f" {len(df['name'].unique())} unique clips"
  )
  save_dir = f'{DATA_DIR}/gym99_clips'
  for i, row in df.iterrows():
    if row['label'] not in [94, 95, 96, 97, 98]:  # [0, 1, 2, 3, 4, 5]:
      continue
    save_path = os.path.join(
        save_dir, f"label_{row['label']}", f"{row['name']}.mp4"
    )
    if os.path.exists(save_path):
      continue
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    video = row['name'].split('_')[0]
    video_file = glob.glob(f'{DATA_DIR}/videos/{video}*')
    if len(video_file) == 0:
      print(f'Video file not found for {video}')
      continue
    video_file = video_file[0]
    event_id = '_'.join(row['name'].split('_')[1:4])
    action_id = '_'.join(row['name'].split('_')[4:])
    ann_dict = ann[video][event_id]
    action_ann = ann_dict['segments'][action_id]
    assert len(ann_dict['timestamps']) == 1
    start = ann_dict['timestamps'][0][0] + action_ann['timestamps'][0][0]
    end = ann_dict['timestamps'][0][0] + action_ann['timestamps'][-1][-1]
    print(
        f"{i}: {row['name']}, {row['label']}, {ann_dict['timestamps'][0]}"
        f' {start}-{end}'
    )
    cmd = (
        f'ffmpeg -y -ss {start} -i {video_file} -t {end - start} -c copy'
        f' {save_path}'
    )
    os.system(cmd)


def prepare_vlm_query_plmm(sampled_num=50):
  query = """Given a video, select the option that best matches its content.
    Choices:
    1. round-off, flic-flac on, stretched salto backward with 2 turn off
    2. round-off, flic-flac on, stretched salto backward with 1 turn off
    3. round-off, flic-flac on, stretched salto backward with 1.5 turn off
    4. round-off, flic-flac on, stretched salto backward with 2.5 turn off
    5. round-off, flic-flac on, stretched salto backward off
    Only reply with the option number (1-5).
    """
  output_file = f'{DATA_DIR}/vlm_queries/input/label1_5.json'
  os.makedirs(os.path.dirname(output_file), exist_ok=True)
  save_list = []
  cnt = 0
  for label in range(1, 6):
    video_dir = f'{DATA_DIR}/gym99_clips/label_{label}'
    video_files = glob.glob(f'{video_dir}/*.mp4')
    print(f'Label {label}: Found {len(video_files)} video files')
    sampled_video_files = np.random.choice(
        video_files, min(sampled_num, len(video_files)), replace=False
    )
    for video_file in sampled_video_files:
      save_list.append({
          'qa_idx': f'{cnt:04d}_label{label}',
          'video_path': video_file,
          'query': query,
          'answer': str(label),
      })
      cnt += 1
  with open(output_file, 'w') as f:
    json.dump(save_list, f, indent=4)


def split_gym99_plmm(train_ratio=0.8):
  for label in [97, 98]:  # 1, 2,
    clip_dir = f'{DATA_DIR}/gym99_clips/label_{label}'
    video_files = glob.glob(f'{clip_dir}/*.mp4')
    print(f'Label {label}: Found {len(video_files)} video files')

    np.random.seed(42)
    np.random.shuffle(video_files)
    n_train = int(len(video_files) * train_ratio)
    train_files = video_files[:n_train]
    test_files = video_files[n_train:]

    save_dir = f'{DATA_DIR}/gym99_clips_exp1/label_{label}'
    os.makedirs(f'{save_dir}/train', exist_ok=True)
    os.makedirs(f'{save_dir}/test', exist_ok=True)
    for video_file in train_files:
      cmd = f'cp {video_file} {save_dir}/train/'
      os.system(cmd)
    for video_file in test_files:
      cmd = f'cp {video_file} {save_dir}/test/'
      os.system(cmd)


if __name__ == '__main__':
  # read_ann()
  # crop_video()
  # scene_detect('local_data/FineGym/clips/')
  # check_video_len()
  # split()

  # crop_gym99_video_plmm()
  # prepare_vlm_query_plmm()
  split_gym99_plmm()
