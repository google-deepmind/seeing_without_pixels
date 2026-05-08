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

from collections import Counter
import json
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils.dataset_utils import absolute_to_relative, custom_to_opencv_c2w, extract_video_frames, matrix_to_pose7d, pose7_to_pose9, poses_to_cam2world, transform_camera_pose

DATA_DIR = os.path.expanduser('~/data/nymeria/')
LOCAL_DATA_DIR = os.path.expanduser('~/local_data/nymeria/')
FINAL_DATA_DIR = os.path.expanduser('~/final_data/')

text_column_mapping = {
    'a': 'Describe my body posture',
    'b': 'Describe my hands/arms motion',
    'c': 'Describe my legs/feet motion',
    'd': 'Describe my focus attention',
}


class NymeriaCamMotionTextPair(torch.utils.data.Dataset):

  def __init__(self, args, mode):
    self.args = args
    self.mode = mode
    self.sample_rate = 1 if args is None else args.sample_rate
    self.text_column = 'a' if args is None else args.text_column
    self.select_column = text_column_mapping[self.text_column]
    self.scenario = '' if args is None else args.scenario
    self.encode_pose = 2 if args is None else args.encode_pose
    self.approx_max_chars = 300  # CLIP's context length is 77 tokens, so we need to truncate the text to 300 characters
    # self.export_to_csv()
    if mode == 'test':
      self.build_test_dataset()
    else:
      self.build_dataset()

  def build_dataset(self):
    df = pd.read_csv(f'{DATA_DIR}/metadata.csv')
    self.df = df.dropna(subset=[self.select_column])
    # Add script column to the DataFrame
    with open(f'{DATA_DIR}/dataset_metadata.json', 'r') as f:
      metadata = json.load(f)
    # Extract keys from motion_file paths and map to scripts
    keys = self.df['motion_file'].str.split('/').str[-1].str.split('.').str[0]
    self.df['script'] = keys.map(lambda key: metadata[key]['script'])
    self.df['scenario'] = self.df['script'].str.split('-').str[0]
    print(f'Loaded {len(self.df)}/{len(df)} samples')
    # print('Script distribution', self.df['script'].value_counts())
    if self.scenario != '':
      self.df = self.df[self.df['scenario'] == self.scenario]
      print(f'Loaded {len(self.df)} samples (scenario = {self.scenario})')

  def build_test_dataset(self):
    key_name = self.select_column.replace('/', '_').replace(' ', '_')
    # self.df = pd.read_csv(f"{DATA_DIR}/eval1000/split_by_{key_name}.csv")
    self.df = pd.read_csv(
        f'{FINAL_DATA_DIR}/data_files/nymeria/eval1000/split_by_{key_name}.csv'
    )

  def export_to_csv(self):
    metadata_rows = []
    with open(
        f'{DATA_DIR}/Nymeria_download_urls_motion_narration.json', 'r'
    ) as f:
      data = json.load(f)
    for key, value in tqdm(data['sequences'].items()):
      ann_file = f'{DATA_DIR}/motion_data/{key}/narration/motion_narration.csv'
      motion_file = f'{DATA_DIR}/cam_motion_cache/presr50/{key}.npz'
      if not os.path.exists(ann_file) or not os.path.exists(motion_file):
        print(f'Skipping {key} because it does not exist')
        continue
      ann_df = pd.read_csv(ann_file)
      cam_motion_dict = np.load(motion_file)
      for i, row in ann_df.iterrows():
        cam_motion = cam_motion_dict[str(i)]
        if len(cam_motion) == 0:
          print(f'Skipping {i} because it does not exist')
          continue
        # Create row with motion file, row index, and all four text columns
        row_data = {'motion_file': motion_file, 'row_idx': i}
        # Add all four text columns from the mapping
        for col in text_column_mapping.values():
          row_data[col] = row[col]
        metadata_rows.append(row_data)

    # Save to metadata.csv
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(f'{DATA_DIR}/metadata.csv', index=False)
    print(f'Saved {len(metadata_rows)} samples to metadata.csv')

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row_idx = self.df.iloc[idx]['row_idx']
    motion_file = os.path.join(
        FINAL_DATA_DIR, 'data/nymeria_eval', self.df.iloc[idx]['motion_file']
    )
    cam_motion_dict = np.load(motion_file)
    cam_motion = cam_motion_dict[str(row_idx)]
    cam_motion = cam_motion[:: self.sample_rate]
    if len(cam_motion) == 0:
      print(f'Skipping {idx} because it does not exist')
      return self.__getitem__(random.randint(0, len(self) - 1))
    cam_motion = transform_camera_pose(cam_motion, self.encode_pose)
    caption = self.df.iloc[idx][self.select_column]
    return cam_motion, caption[: self.approx_max_chars]


class NymeriaVideoAndCamMotionTextPair(torch.utils.data.Dataset):

  def __init__(self, text_mode, opencv_cord=False):
    self.text_column = text_mode
    self.opencv_cord = opencv_cord
    self.select_column = text_column_mapping[self.text_column]
    self.build_test_dataset()

  def build_test_dataset(self):
    key_name = self.select_column.replace('/', '_').replace(' ', '_')
    self.df = pd.read_csv(f'{DATA_DIR}/eval1000/split_by_{key_name}.csv')

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row_idx = self.df.iloc[idx]['row_idx']
    motion_file = self.df.iloc[idx]['motion_file']
    video_file = (
        motion_file.replace('cam_motion_cache/presr50', 'video_cache').replace(
            '.npz', '/'
        )
        + str(row_idx)
        + '.mp4'
    )
    if not os.path.exists(video_file):
      print(f'{video_file} does not exist')
      return self.__getitem__(random.randint(0, len(self) - 1))
    cam_motion_dict = np.load(motion_file)
    cam_motion = cam_motion_dict[str(row_idx)]
    if len(cam_motion) == 0:
      print(f'Skipping {idx} because it does not exist')
      return self.__getitem__(random.randint(0, len(self) - 1))

    cam_c2w = poses_to_cam2world(cam_motion)
    if self.opencv_cord:
      cam_c2w = custom_to_opencv_c2w(cam_c2w)
      cam_c2w = matrix_to_pose7d(cam_c2w)
      cam_c2w = absolute_to_relative(cam_c2w)
      cam_c2w = poses_to_cam2world(cam_c2w)

    n_frames = cam_c2w.shape[0]
    caption = self.df.iloc[idx][self.select_column]
    frames = extract_video_frames(video_file, n_frames)
    return frames, cam_c2w, caption, idx, ''


def sample_data_for_viz(save_dir, sampled_k=1000):
  os.makedirs(save_dir, exist_ok=True)
  dataset = NymeriaCamMotionTextPair(None, 'test')
  idx_list = random.sample(range(len(dataset)), k=sampled_k)
  text_list = []
  # for i, idx in enumerate(idx_list):
  for idx in tqdm(range(len(dataset))):
    cam_traj, caption = dataset[idx]
    text_list.append(caption)
    np.save(f'{save_dir}/{idx}.npy', cam_traj)
  pd.Series(text_list).to_csv(
      f'{save_dir}/caption.csv', index=False, header=False
  )


if __name__ == '__main__':
  sample_data_for_viz('tmp/cam_viz/nymeria')
  # dataset = NymeriaCamMotionTextPair(None, 'test')
  # print(len(dataset))
  # for i in range(50):
  #     idx = random.randint(0, len(dataset)-1)
  #     cam_traj, caption = dataset[idx]
  #     print(i, cam_traj.shape)
