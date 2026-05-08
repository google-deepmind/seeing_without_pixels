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

from collections import defaultdict
import json
import os
import numpy as np

DATA_DIR = os.path.expanduser('~/data/egoexo4d')


def read_ann_file():
  with open(
      f'{DATA_DIR}/annotations/proficiency_demonstrator_train.json', 'r'
  ) as f:
    ann_data = json.load(f)['annotations']

  # {'Soccer', 'Music', 'Dance', 'Bike Repair', 'Cooking', 'Basketball', 'Health', 'Rock Climbing'}
  p_dict = defaultdict(list)
  unique_task = set()
  cnt = 0
  for ann_dict in ann_data:
    if ann_dict['scenario_name'] != 'Basketball':
      continue
    unique_task.add(ann_dict['task_name'])
    if 'Reverse Layup' not in ann_dict['task_name']:
      continue
    # if 'unc' not in ann_dict['task_name']:
    #     continue
    video_path = os.path.join(
        DATA_DIR,
        ann_dict['video_paths']['exo1'].replace(
            'frame_aligned_videos', 'frame_aligned_videos/downscaled/448/'
        ),
    )
    if not os.path.exists(video_path):
      continue
    p_dict[ann_dict['origin_participant_id']].append(ann_dict)
    cnt += 1
  print(f'Total {cnt} clips from {len(p_dict)} participants')
  for pid, ann_list in p_dict.items():
    print(
        f"Participant {pid} ({ann_list[0]['proficiency_score']}),"
        f' {len(ann_list)} clips'
    )
    video_path = os.path.join(
        DATA_DIR,
        ann_list[0]['video_paths']['exo1'].replace(
            'frame_aligned_videos', 'frame_aligned_videos/downscaled/448/'
        ),
    )
    print(video_path, os.path.exists(video_path))
    print('-' * 20)


def read_task_name():
  with open(
      f'{DATA_DIR}/annotations/proficiency_demonstrator_train.json', 'r'
  ) as f:
    ann_data = json.load(f)['annotations']
  task_dict = defaultdict(list)
  for ann_dict in ann_data:
    if ann_dict['task_name'] not in task_dict[ann_dict['scenario_name']]:
      task_dict[ann_dict['scenario_name']].append(ann_dict['task_name'])
  for scenario, task_list in task_dict.items():
    print(f"{scenario}: {', '.join(task_list)}")


if __name__ == '__main__':
  # read_ann_file()
  read_task_name()
