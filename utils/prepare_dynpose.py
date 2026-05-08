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
import json
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.dataset_utils import poses_to_scale

DATA_DIR = os.path.expanduser('~/data/dynpose-100k/')
LOCAL_DATA_DIR = os.path.expanduser('~/local_data/dynpose-100k/')


def load_dynpose():
  df = pd.read_csv(f'{LOCAL_DATA_DIR}/dynpose_100k/metadata.csv')
  df_vipe = pd.read_parquet(f'{DATA_DIR}/vipe-dynpose-100kpp/meta.parquet')
  vipe_mapping = {
      row['sequence']: row['tar_name'] for _, row in df_vipe.iterrows()
  }

  uid_mapping = {}
  with open(f'{LOCAL_DATA_DIR}/dynpose_100k/uid_mapping.csv') as f:
    next(f)  # Skip the first row
    for i, line in enumerate(f):
      parts = line.strip().split(',')
      uid_mapping[i] = parts

  valid_rows = []
  for i, row in tqdm(df.iterrows(), total=len(df)):
    uid_list = uid_mapping[i]
    caption_list = eval(row['caption'])
    video_uid = row['videoID']
    for j, (uid, caption) in enumerate(zip(uid_list, caption_list)):
      video_path = os.path.join(
          LOCAL_DATA_DIR,
          'dynpose_100k/videos',
          video_uid[0],
          f'{video_uid}_{j:03d}.mp4',
      )
      pose_path = os.path.join(
          LOCAL_DATA_DIR, 'dynpose_100k/cameras', f'{uid}.pkl'
      )
      video_path = video_path if os.path.exists(video_path) else None

      with open(pose_path, 'rb') as f:
        poses = pickle.load(f)['poses']
      scale = poses_to_scale(poses, 'w2c')

      vipe_path = (
          os.path.join(
              LOCAL_DATA_DIR, 'vipe-dynpose-100kpp/poses', f'{uid}.npz'
          )
          if uid in vipe_mapping
          else None
      )
      vipe_scale = None
      if vipe_path is not None:
        vipe_pose = np.load(vipe_path)['data']  # (N, 4, 4)
        vipe_scale = poses_to_scale(vipe_pose, 'c2w')

      valid_row = {
          'video_uid': video_uid,
          'uid': uid,
          'segment_id': j,
          'video_path': video_path,
          'pose_path': pose_path,
          'vipe_pose_path': vipe_path,
          'description_text': caption,
          'pose_scale': scale,
          'vipe_pose_scale': vipe_scale,
      }
      valid_rows.append(valid_row)

  new_df = pd.DataFrame(valid_rows)

  print('-' * 10, 'pose_scale stats', '-' * 10)
  print(new_df['pose_scale'].describe())
  print('-' * 10, 'vipe_scale stats', '-' * 10)
  print(new_df['vipe_pose_scale'].describe())

  new_df.to_csv(
      f'{LOCAL_DATA_DIR}/dynpose_100k/metadata_with_path.csv', index=False
  )
  print(f'Saving {len(new_df)} rows to metadata_with_path.csv')


def untar_file():
  dir_name = f'{DATA_DIR}/vipe-dynpose-100kpp/payload'
  dest_path = f'{LOCAL_DATA_DIR}/vipe-dynpose-100kpp/poses'
  os.makedirs(dest_path, exist_ok=True)
  for sub_dir in os.listdir(dir_name):
    file_path = os.path.join(dir_name, sub_dir, 'pose.tar')
    assert os.path.exists(file_path), f'{file_path} does not exist'
    cmd = f'tar -xvf {file_path} -C {dest_path}'
    os.system(cmd)


def split_file(val_num=5000, threshold=20):
  df = pd.read_csv(f'{DATA_DIR}/dynpose_100k/metadata_with_path.csv')
  valid_df = df[df['pose_scale'] < threshold]
  print(f'{len(valid_df)}/{len(df)} samples have pose scale < {threshold}')
  valid_df = valid_df[valid_df['vipe_pose_scale'] < threshold]
  print(f'{len(valid_df)} samples have vipe pose scale < {threshold}')
  valid_df = valid_df[valid_df['video_path'].notnull()]
  print(f'{len(valid_df)}/{len(df)} samples have valid video path')
  valid_df = valid_df[valid_df['vipe_pose_path'].notnull()]
  print(f'{len(valid_df)} samples have valid vipe pose path')

  sampled_uids = (
      valid_df['video_uid'].drop_duplicates().sample(n=val_num, random_state=42)
  )
  val_df = valid_df[valid_df['video_uid'].isin(sampled_uids)]
  train_df = df.drop(val_df.index)
  print(f'Train size: {len(train_df)}, Val size: {len(val_df)}')
  train_df.to_csv(f'{DATA_DIR}/dynpose_100k/metadata_train_v1.csv', index=False)
  val_df.to_csv(f'{DATA_DIR}/dynpose_100k/metadata_val_v1.csv', index=False)


def gen_vlm_query_file_parta():
  data_file = f'{DATA_DIR}/dynpose_100k/metadata_samevideo_test5000.csv'
  df = pd.read_csv(data_file)
  save_list = []
  for i, row in df.iterrows():
    if i % 5 != 0:
      continue
    save_list.append({
        'qa_idx': f'{i:04d}',
        'video_path': row['video_path'],
        'query': 'Describe the camera motion in this video.',
    })
  save_file = (
      f"{DATA_DIR}/dynpose_100k/vlm_queries/input/camera_description_{os.path.basename(data_file).replace('.csv', '.json')}"
  )
  with open(save_file, 'w') as f:
    json.dump(save_list, f, indent=4)
  print(f'Saved {len(save_list)} VLM querides to {save_file}')


def gen_vlm_query_file_partb1():
  response_files = glob.glob(
      f'{DATA_DIR}/dynpose_100k/vlm_queries/output/camera_description_metadata_samevideo_test5000/*.jsonl'
  )
  for response_file in response_files:
    save_list = []
    with open(response_file, 'r') as f:
      for line in f:
        tmp_dict = json.loads(line)
        query = (
            'Rewrite the given caption to keep only the camera description'
            ' (e.g., angle, motion, shot type, framing). Remove all semantic'
            ' content (e.g., mentions of objects, people, or actions). Here is'
            f" the caption: {tmp_dict['response']}"
        )
        save_list.append({'qa_idx': tmp_dict['idx'], 'query': query})
        print(query)

    save_file = (
        f"{DATA_DIR}/dynpose_100k/vlm_queries/input/camera_filter_{os.path.basename(response_file).replace('.jsonl', '.json')}"
    )
    with open(save_file, 'w') as f:
      json.dump(save_list, f, indent=4)
    print(f'Saved {len(save_list)} queries to {save_file}')


def gen_vlm_query_file_partb2():
  data_file = f'{DATA_DIR}/dynpose_100k/metadata_samevideo_test5000.csv'
  df = pd.read_csv(data_file)

  response_files = glob.glob(
      f'{DATA_DIR}/dynpose_100k/vlm_queries/output/camera_description_metadata_samevideo_test5000/*.jsonl'
  )
  response_files = [
      'data/dynpose-100k/dynpose_100k/vlm_queries/output/camera_filter_qwen2.5-vl-7b-cam-motion/gemini-2.5-flash.jsonl',
      'data/dynpose-100k/dynpose_100k/vlm_queries/output/camera_filter_ShotVL-7B/gemini-2.5-flash.jsonl',
  ]
  for response_file in response_files:
    response_dict = {}
    with open(response_file, 'r') as f:
      for line in f:
        tmp_dict = json.loads(line)
        response_dict[tmp_dict['idx']] = tmp_dict['response']
    print(f'Loaded {len(response_dict)} responses from {response_file}')

    save_list = []
    for i, row in df.iterrows():
      if i % 5 != 0:
        continue
      sub_df = df.iloc[i : i + 5]

      caption_list = sub_df['description_text'].tolist()
      vid_options = caption_list.copy()
      correct_answer_vid = vid_options[0]
      random.shuffle(vid_options)
      answer_vid_letter = chr(65 + vid_options.index(correct_answer_vid))

      camera_desc = response_dict[f'{i:04d}']
      query = f"""The following describes the motion and focus of a camera while filming a scene:"{camera_desc}"
                Which of the following events or scene descriptions is most likely being filmed with this camera movement?\n"""
      for j, option in enumerate(vid_options):
        query += f'{chr(65 + j)}. {option}\n'
      query += 'Reply with just the letter (A, B, C, D, or E).'
      save_list.append({
          'qa_idx': f'{i:04d}',
          'video_path': row['video_path'],
          'query': query,
          'answer': answer_vid_letter,
      })
    # fn = os.path.basename(response_file).replace('.jsonl', '.json')
    fn = response_file.split('/')[-2] + '.jsonl'
    save_file = f'{DATA_DIR}/dynpose_100k/vlm_queries/input/camera_qa_{fn}'
    with open(save_file, 'w') as f:
      json.dump(save_list, f, indent=4)


if __name__ == '__main__':
  # load_dynpose()
  # split_file()
  # untar_file()
  # gen_vlm_query_file_partb()
  gen_vlm_query_file_partb2()
