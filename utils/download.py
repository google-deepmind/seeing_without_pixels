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
import random
import pandas as pd
from tqdm import tqdm


def download_egoexo4d():
  data_dir = os.path.expanduser('~/data/egoexo4d/')
  for mode in ['val', 'train']:
    file = os.path.join(
        data_dir, f'annotations/proficiency_demonstrator_{mode}.json'
    )
    with open(file, 'r') as f:
      data = json.load(f)['annotations']
    take_uids = []
    for item in data:
      if item['scenario_name'] != 'Rock Climbing':
        continue
      cmd = (
          f'egoexo -y -o {data_dir} --parts take_trajectory --uids'
          f" {item['take_uid']}"
      )
      os.system(cmd)


def download_egoexo4d_audio():
  data_dir = os.path.expanduser('~/data/egoexo4d/')
  data_dir_vrs = os.path.expanduser('~/data/egoexo4d_vrs/')
  # df = pd.read_csv(os.path.join(data_dir, 'annotations/pretraining/test_presr50_sampled1000_True_mcqv0.csv'))
  df = pd.read_csv(
      os.path.join(data_dir, 'annotations/pretraining/test_alltasks_mcqv0.csv')
  )
  df['take_name'] = df['save_id'].str.split('_').str[:-2].str.join('_')
  take_uids = df['take_name'].unique().tolist()
  print(f'Found {len(take_uids)} takes')

  with open(f'{data_dir}/takes.json', 'r') as f:
    takes = json.load(f)
  take_name_to_uid = {}
  for take in takes:
    take_name_to_uid[take['take_name']] = take['take_uid']

  for take_name in tqdm(take_uids):
    uid = take_name_to_uid[take_name]
    # cmd = f"egoexo -y -o {data_dir} --parts take_audio --uids {uid}"
    cmd = (
        f'egoexo -y -o {data_dir_vrs} --parts take_vrs_noimagestream --uids'
        f' {uid}'
    )
    os.system(cmd)


def mv_take():
  source_dir = os.path.expanduser(
      '~/projects/camera_motion_modeling/egoexo4d/takes'
  )
  target_dir = os.path.expanduser('~/data/egoexo4d/')
  for mode in ['train', 'val']:
    file = os.path.join(
        target_dir, f'annotations/proficiency_demonstrator_{mode}.json'
    )
    with open(file, 'r') as f:
      data = json.load(f)['annotations']
    take_uids = []
    cnt1, cnt2, total_cnt = 0, 0, 0
    for item in tqdm(data):
      # if item['scenario_name'] != 'Rock Climbing':
      #     continue
      take_name = item['video_paths']['ego'].split('/')[1]
      source_file = os.path.join(
          source_dir, take_name, 'trajectory/closed_loop_trajectory.csv'
      )
      total_cnt += 1
      if not os.path.exists(source_file):
        # print(f"File {source_file} does not exist")
        # cmd = f"egoexo -y -o {target_dir} --parts take_trajectory --uids {item['take_uid']}"
        # os.system(cmd)
        continue
      cnt1 += 1
      target_file = os.path.join(
          target_dir,
          'takes',
          take_name,
          'trajectory/closed_loop_trajectory.csv',
      )
      if not os.path.exists(target_file):
        os.system(f'cp -r {source_dir}/{take_name} {target_dir}/takes')
        # continue
      # cnt2 += 1
    print(f'Mode {mode}: {cnt1}/{total_cnt}, {cnt2}/{total_cnt}')


if __name__ == '__main__':
  # download_egoexo4d()
  # download_egoexo4d_pretrain()
  # mv_take()
  download_egoexo4d_audio()
