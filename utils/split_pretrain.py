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

import ast
import json
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm


def split_egoexo4d_pretraining_val(sample_num=1000):
  data_dir = os.path.expanduser('~/data/egoexo4d')
  df = pd.read_csv(
      os.path.join(
          data_dir, 'annotations', 'pretraining', 'val_presr50_cleanv1.csv'
      )
  )
  df['take_name'] = df['save_id'].apply(lambda x: '_'.join(x.split('_')[:-2]))
  for ego_visible in [True, False]:
    sampled_rows = []
    # Get subset for current ego_visible category
    df_subset = df[df['ego_visible'] == ego_visible]

    # First sample one row from each unique take
    for take in df_subset['take_name'].unique():
      take_rows = df_subset[df_subset['take_name'] == take]
      sampled_rows.append(take_rows.sample(n=1, random_state=42))

    # If we need more samples, randomly sample from remaining rows
    if len(sampled_rows) < sample_num:
      remaining = df_subset[
          ~df_subset.index.isin(pd.concat(sampled_rows).index)
      ]
      additional_samples = remaining.sample(
          n=sample_num - len(sampled_rows), random_state=42
      )
      sampled_rows.append(additional_samples)

    sampled_df = pd.concat(sampled_rows)
    print(f'Total sampled rows: {len(sampled_df)}')
    print(
        'Number of unique takes in sampled data:'
        f" {len(sampled_df['take_name'].unique())}"
    )
    # sampled_df.to_csv(os.path.join(data_dir, 'annotations', 'pretraining', f'test_presr50_sampled{sample_num}_{ego_visible}.csv'), index=False)


def split_egoexo4d_pretraining_val_v2(
    total_samples=1000, num_takes=50, ego_visible=False
):
  """Sample validation data with a new strategy: 1.

  Filter takes to only include those with enough unique rows (>=
  samples_per_take) 2. Sample num_takes from the filtered takes 3. For each
  take, sample samples_per_take rows from non-duplicate description texts 4. If
  total samples is less than requested, fill in with random unique rows from any
  take

  Args:
      total_samples (int): Total number of samples to generate (default: 1000)
      num_takes (int): Number of unique takes to sample (default: 50)
  """
  samples_per_take = total_samples // num_takes
  data_dir = os.path.expanduser('~/data/egoexo4d')
  df = pd.read_csv(
      os.path.join(
          data_dir, 'annotations', 'pretraining', 'val_presr50_cleanv1.csv'
      )
  )
  df['take_name'] = df['save_id'].apply(lambda x: '_'.join(x.split('_')[:-2]))
  df = df[df['ego_visible'] == ego_visible]
  df = df.drop_duplicates(subset=['description_text'])

  # First filter takes to only include those with enough unique rows
  valid_takes = []
  for take in df['take_name'].unique():
    take_rows = df[df['take_name'] == take]
    if len(take_rows) >= samples_per_take:
      valid_takes.append(take)

  print(
      f'Found {len(valid_takes)} takes with at least {samples_per_take} unique'
      ' rows'
  )

  if len(valid_takes) < num_takes:
    print(
        f'Warning: Only {len(valid_takes)} takes have enough unique rows, which'
        f' is less than requested {num_takes}'
    )
    num_takes = len(valid_takes)

  # Sample num_takes from valid takes
  sampled_takes = np.random.choice(valid_takes, size=num_takes, replace=False)

  sampled_rows = []
  used_indices = set()  # Keep track of used row indices

  for take in sampled_takes:
    # Get rows for this take
    take_rows = df[df['take_name'] == take]

    # Remove rows with duplicate description text
    take_rows = take_rows.drop_duplicates(subset=['description_text'])

    # Sample samples_per_take rows
    sampled_take_rows = take_rows.sample(n=samples_per_take, random_state=42)
    sampled_rows.append(sampled_take_rows)
    used_indices.update(sampled_take_rows.index)

  # Combine all sampled rows
  sampled_df = pd.concat(sampled_rows)

  print(f'Total sampled rows: {len(sampled_df)}')
  print(
      'Number of unique takes in sampled data:'
      f" {len(sampled_df['take_name'].unique())}"
  )
  print(
      'Number of unique description texts:'
      f" {len(sampled_df['description_text'].unique())}"
  )

  # Save the sampled data
  sampled_df.to_csv(
      os.path.join(
          data_dir,
          'annotations',
          'pretraining',
          f'test_presr50_sampled{total_samples}_{ego_visible}_v2.csv',
      ),
      index=False,
  )


def split_egoexo4d_pretraining_val_v3(
    total_samples=1000, ego_visible=False, scenario=None
):
  """Sample validation data with a simple strategy: 1.

  Filter for ego_visible 2. Optionally filter for specific scenario (e.g.
  'bouldering') 3. Remove duplicate description texts 4. Sample total_samples
  rows

  Args:
      total_samples (int): Total number of samples to generate (default: 1000)
      ego_visible (bool): Whether to sample ego-visible or ego-invisible samples
        (default: False)
      scenario (str): Optional scenario to filter for (e.g. 'bouldering'). If
        None, samples from all scenarios.
  """
  data_dir = os.path.expanduser('~/data/egoexo4d')
  df = pd.read_csv(
      os.path.join(
          data_dir, 'annotations', 'pretraining', 'val_presr50_cleanv1.csv'
      )
  )
  df['take_name'] = df['save_id'].apply(lambda x: '_'.join(x.split('_')[:-2]))

  # Filter for ego_visible
  df = df[df['ego_visible'] == ego_visible]

  # Filter for specific scenario if provided
  if scenario is not None:
    df = df[df['take_name'].str.contains(scenario, case=False)]
    print(f'Filtered to {len(df)} rows containing scenario: {scenario}')

  # Remove duplicates
  df = df.drop_duplicates(subset=['description_text'])

  # Sample total_samples rows
  sampled_df = df.sample(n=min(total_samples, len(df)), random_state=42)

  print(f'Total sampled rows: {len(sampled_df)}')
  print(
      'Number of unique takes in sampled data:'
      f" {len(sampled_df['take_name'].unique())}"
  )
  print(
      'Number of unique description texts:'
      f" {len(sampled_df['description_text'].unique())}"
  )

  # Save the sampled data
  scenario_suffix = f'_{scenario}' if scenario else ''
  sampled_df.to_csv(
      os.path.join(
          data_dir,
          'annotations',
          'pretraining',
          f'test_presr50_sampled{total_samples}_{ego_visible}{scenario_suffix}_v3.csv',
      ),
      index=False,
  )


def split_egoexo4d_pretraining_val_v4(
    feature_path, total_samples=1000, ego_visible=False
):
  """Sample validation data using text feature embeddings and clustering: 1.

  Load text features from feature_path 2. Filter out rows with ego_visible=True
  and their corresponding features 3. Use K-means clustering to get
  total_samples clusters 4. Sample one row from each cluster to get diverse
  samples

  Args:
      total_samples (int): Total number of samples to generate (default: 1000)
      feature_path (str): Path to the text feature embeddings file
  """

  data_dir = os.path.expanduser('~/data/egoexo4d')
  df = pd.read_csv(
      os.path.join(
          data_dir,
          'annotations',
          'pretraining',
          'val_presr50_cleanv1_components_motionstd.csv',
      )
  )
  df['take_name'] = df['save_id'].apply(lambda x: '_'.join(x.split('_')[:-2]))

  # Filter out rows with ego_visible=True
  print(f'Before filtering: {len(df)} rows')
  df = df[df['ego_visible'] == ego_visible]
  df = df.drop_duplicates(subset=['description_text'])
  df = df[df['motion_std'] > 0.03]
  print(f'Filtered to {len(df)} rows with motion_std > 0.03')

  # Load text features
  features = torch.load(feature_path)
  if isinstance(features, torch.Tensor):
    features = features.cpu().numpy()

  # Filter features to match filtered dataframe
  features = features[df.index]

  # Ensure features match dataframe length
  assert len(features) == len(
      df
  ), f"Feature length {len(features)} doesn't match dataframe length {len(df)}"

  # Perform K-means clustering
  kmeans = KMeans(n_clusters=total_samples, random_state=42, n_init=10)
  cluster_labels = kmeans.fit_predict(features)

  # Sample one row from each cluster
  sampled_indices = []
  for cluster_id in range(total_samples):
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    if len(cluster_indices) > 0:
      # Randomly sample one index from this cluster
      sampled_idx = np.random.choice(cluster_indices)
      sampled_indices.append(sampled_idx)

  # Get sampled rows
  sampled_df = df.iloc[sampled_indices].copy()

  print(f'Total sampled rows: {len(sampled_df)}')
  print(
      'Number of unique takes in sampled data:'
      f" {len(sampled_df['take_name'].unique())}"
  )
  print(
      'Number of unique description texts:'
      f" {len(sampled_df['description_text'].unique())}"
  )

  # Save the sampled data
  sampled_df.to_csv(
      os.path.join(
          data_dir,
          'annotations',
          'pretraining',
          f'test_presr50_sampled{total_samples}_{ego_visible}_v7.csv',
      ),
      index=False,
  )


def split_eval_v5(data_file):
  df = pd.read_csv(data_file)
  df['first_verb'] = df['verbs'].apply(
      lambda x: x.split(';')[0] if not pd.isna(x) else None
  )
  df = df[df['ego_visible'] == False]
  unique_verbs = df['verbs'].unique()
  unique_first_verbs = df['first_verb'].unique()
  print(
      f'Found {len(unique_verbs)} unique verbs, {len(unique_first_verbs)}'
      f' unique first verbs, {len(df)} total rows'
  )

  rows = []
  for i, verb in enumerate(unique_first_verbs):
    sub_df = df[df['first_verb'] == verb]
    if len(sub_df) == 0:
      continue
    print(i, verb, len(sub_df))
    row = sub_df.sample(1)
    rows.append(row)
  save_file = os.path.join(
      os.path.dirname(data_file), 'test_presr50_sampled1000_False_v5.csv'
  )
  pd.concat(rows).to_csv(save_file, index=False)
  print(f'Saved {len(rows)} rows to {save_file}')


def split_eval_v6(data_file):
  df = pd.read_csv(data_file)
  df = df[df['ego_visible'] == False]
  unique_verbs = df['verbs'].unique()
  sampled_unique_verbs = np.random.choice(
      unique_verbs, size=1001, replace=False
  )
  print(
      f'Sampled {len(sampled_unique_verbs)} unique verbs from'
      f' {len(unique_verbs)} verbs'
  )
  rows = []
  for verb in sampled_unique_verbs:
    sub_df = df[df['verbs'] == verb]
    if len(sub_df) == 0:
      continue
    print(verb, len(sub_df))
    row = sub_df.sample(1)
    rows.append(row)
  save_file = os.path.join(
      os.path.dirname(data_file), 'test_presr50_sampled1000_False_v6.csv'
  )
  pd.concat(rows).to_csv(save_file, index=False)
  print(f'Saved {len(rows)} rows to {save_file}')


def load_take_mapping():
  data_dir = os.path.expanduser('~/data/egoexo4d')
  with open(os.path.join(data_dir, 'takes.json'), 'r') as f:
    data_list = json.load(f)
  take_mapping = {}
  for data in data_list:
    take_mapping[data['take_name']] = data['parent_task_name']
  return take_mapping


def split_mcq(
    data_file,
    ego_visible=False,
    sample_num=1000,
    scenario=None,
    motion_thresh=0.03,
):
  take_mapping = load_take_mapping()
  df = pd.read_csv(data_file)
  # Convert verbs column to lists of verbs for all rows at once
  df['verb_list'] = df['verbs'].apply(
      lambda x: set(x.split(';')) if isinstance(x, str) else set()
  )
  df['take_name'] = df['save_id'].apply(lambda x: '_'.join(x.split('_')[:-2]))

  # df = df.drop_duplicates(subset=['description_text'])
  # print(f"Filtered to {len(df)} rows with unique description texts")
  if scenario is not None:
    df['task_name'] = df['take_name'].map(take_mapping)
    selected_df = df[df['task_name'] == scenario]
    print(
        f'Filtered to {len(selected_df)} rows containing scenario: {scenario}'
    )
    selected_df = selected_df[selected_df['ego_visible'] == ego_visible]
    print(
        f'Filtered to {len(selected_df)} rows with ego_visible = {ego_visible}'
    )
  else:
    selected_df = df[df['ego_visible'] == ego_visible]
    print(
        f'Filtered to {len(selected_df)} rows with ego_visible = {ego_visible}'
        f' and scenario = {scenario}'
    )
  selected_df = selected_df[selected_df['motion_std'] > motion_thresh]
  print(
      f'Filtered to {len(selected_df)} rows with motion_std > {motion_thresh}'
  )
  selected_df = selected_df
  sample_row_num = min(sample_num + 500, len(selected_df))
  sampled_df = selected_df.sample(sample_row_num)
  save_rows = []
  cnt = 0
  for i, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
    if cnt >= sample_num:
      break
    current_verbs = row['verb_list']
    current_take = row['take_name']

    # First filter for same take, then check for non-overlapping verbs
    same_take_mask = df['take_name'] == current_take
    same_take_df = df[
        same_take_mask
    ].copy()  # Make a copy to avoid SettingWithCopyWarning

    # Among same take rows, find those with non-overlapping verbs
    non_overlap_mask = same_take_df['verb_list'].apply(
        lambda x: x.isdisjoint(current_verbs)
    )
    sub_df = same_take_df[
        non_overlap_mask
    ].copy()  # Make a copy to avoid SettingWithCopyWarning

    if len(sub_df) < 4:
      continue

    # Find 5 rows with mutually disjoint verb lists
    selected_indices = []
    remaining_df = sub_df.copy()

    while len(selected_indices) < 4 and len(remaining_df) > 0:
      # Get the first row
      current_idx = remaining_df.index[0]
      selected_indices.append(current_idx)

      # Remove rows that have any overlapping verbs with any selected row
      remaining_df = remaining_df.iloc[
          1:
      ].copy()  # Make a copy to avoid SettingWithCopyWarning

      # Create a mask for rows that are disjoint from all selected rows
      disjoint_mask = pd.Series(True, index=remaining_df.index)
      for selected_idx in selected_indices:
        selected_verbs = df.loc[selected_idx, 'verb_list']
        disjoint_mask &= remaining_df['verb_list'].apply(
            lambda x: x.isdisjoint(selected_verbs)
        )

      remaining_df = remaining_df[disjoint_mask]

    if len(selected_indices) < 4:
      continue

    # Get the selected rows using the original DataFrame
    sub_df = df.loc[[i] + selected_indices].copy()
    # Add gt column - first row is True, others are False
    sub_df['gt'] = [True] + [False] * (len(sub_df) - 1)
    # text_list = sub_df['description_text'].tolist()
    # print(i, row['description_text'])
    # for j, text in enumerate(text_list):
    #     print(f"{j+1}. {text}")
    # print('-'*50)

    save_rows.append(sub_df)
    cnt += 1

  save_df = pd.concat(save_rows)
  # print(save_df['ego_visible'].value_counts())
  fn = (
      ego_visible if scenario is None else f'{scenario}_egovisible{ego_visible}'
  )
  save_df.to_csv(
      os.path.join(
          os.path.dirname(data_file),
          f'test_presr50_sampled{sample_num}_{fn}_mcqv0.csv',
      ),
      index=False,
  )
  print(f'Saved {len(save_df)} rows')


def combine_files():
  data_dir = os.path.expanduser('~/data/egoexo4d/annotations/pretraining')
  all_dfs = []

  for task_name in [
      'Cooking',
      'Health',
      'Dance',
      'Rock_Climbing',
      'Bike_Repair',
      'Basketball',
      'Soccer',
      'Music',
  ]:
    for ego_visible in [True, False]:
      fn = f'{task_name}_egovisible{ego_visible}'
      # file = os.path.join(data_dir, f'test_presr50_sampled500_{fn}_mcqv0.csv')
      file = os.path.join(data_dir, f'test_{fn}_mcqv0.csv')
      assert os.path.exists(file), f'File not found: {file}'
      df = pd.read_csv(file)
      print(f'Loaded {len(df)} rows from {file}')
      all_dfs.append(df)

  # Combine all dataframes
  combined_df = pd.concat(all_dfs, ignore_index=True)
  print(f'Combined {len(combined_df)} total rows from {len(all_dfs)} files')

  # Save the combined dataframe
  # output_file = os.path.join(data_dir, 'test_presr50_sampled500_alltasks_mcqv0.csv')
  output_file = os.path.join(data_dir, 'test_alltasks_mcqv0.csv')
  combined_df.to_csv(output_file, index=False)
  print(f'Saved combined data to {output_file}')


def split_dynpose_pretraining_val(data_file, num_samples=1000):
  df = pd.read_csv(data_file)
  cnt = 0
  for i, row in df.iterrows():
    caption = eval(row['caption'])
    if len(caption) > 4:
      print(i, caption)
      cnt += 1
  print(cnt, len(df))


def split_nymeria_retrieval(key, sample_num=500):
  data_dir = os.path.expanduser('~/data/nymeria/')
  local_data_dir = os.path.expanduser('~/local_data/nymeria/')
  df = pd.read_csv(f'{data_dir}/metadata_components.csv')
  print('before', len(df))
  df = df.dropna(subset=['motion_file', f'{key}_verbs'])
  print('after filtering', len(df))
  df['verb_list'] = df[f'{key}_verbs'].apply(
      lambda x: set(x.split(';')) if isinstance(x, str) else set()
  )
  with open(f'{data_dir}/dataset_metadata.json', 'r') as f:
    metadata = json.load(f)
  keys = df['motion_file'].str.split('/').str[-1].str.split('.').str[0]
  df['script'] = keys.map(lambda key: metadata[key]['script'])
  df['scenario'] = df['script'].str.split('-').str[0]
  scenario_list = df['scenario'].unique()
  save_rows = []
  # for scenario in scenario_list:
  #     selected_df = df[df['scenario'] == scenario]
  #     # sample_row_num = min(sample_num + 100, len(selected_df))
  #     sampled_df = selected_df.sample(len(selected_df), random_state=42)

  if True:
    selected_df = df.sample(len(df), random_state=42)
    sampled_df = selected_df

    cnt = 0
    for i, row in sampled_df.iterrows():
      if cnt >= sample_num:
        break
      current_verbs = row['verb_list']
      same_take_df = selected_df[
          selected_df['motion_file'] == row['motion_file']
      ]
      non_overlap_mask = same_take_df['verb_list'].apply(
          lambda x: x.isdisjoint(current_verbs)
      )
      sub_df = same_take_df[
          non_overlap_mask
      ].copy()  # Make a copy to avoid SettingWithCopyWarning
      # print(i, current_verbs)

      if len(sub_df) < 4:
        continue

      # Find 5 rows with mutually disjoint verb list
      selected_indices = []
      remaining_df = sub_df.copy()

      while len(selected_indices) < 4 and len(remaining_df) > 0:
        # Get the first row
        current_idx = remaining_df.index[0]
        selected_indices.append(current_idx)

        # Remove rows that have any overlapping verbs with any selected row
        remaining_df = remaining_df.iloc[
            1:
        ].copy()  # Make a copy to avoid SettingWithCopyWarning

        # Create a mask for rows that are disjoint from all selected rows
        disjoint_mask = pd.Series(True, index=remaining_df.index)
        for selected_idx in selected_indices:
          selected_verbs = df.loc[selected_idx, 'verb_list']
          disjoint_mask &= remaining_df['verb_list'].apply(
              lambda x: x.isdisjoint(selected_verbs)
          )

        remaining_df = remaining_df[disjoint_mask]

      if len(selected_indices) < 4:
        continue

      # Get the selected rows using the original DataFrame
      sub_df = df.loc[[i] + selected_indices].copy()
      # Add gt column - first row is True, others are False
      sub_df['gt'] = [True] + [False] * (len(sub_df) - 1)

      text_list = sub_df[key].tolist()
      text_verb_list = sub_df['verb_list'].tolist()
      # print(i, row[key], row['verb_list'])
      # for j, text in enumerate(text_list):
      #     print(f"{j+1}. {text}, {text_verb_list[j]}")
      # print('-'*50)

      save_rows.append(sub_df)
      cnt += 1

  print(f'Saving {len(save_rows)} rows')
  save_df = pd.concat(save_rows)
  save_key = key.replace('/', '_').replace(' ', '_')
  os.makedirs(f'{data_dir}/eval{sample_num}', exist_ok=True)
  save_df.to_csv(f'{data_dir}/eval{sample_num}/split_by_{save_key}.csv')


def split_dynpose_retrieval(sample_num=1000, same_video=False):
  df = pd.read_csv(
      'data/dynpose-100k/dynpose_100k/metadata_val_v1_with_verbs.csv'
  )
  df['verb_list'] = df['verb_lemmas'].apply(lambda x: set(ast.literal_eval(x)))
  if same_video:
    video_uid_list = df['video_uid'].value_counts()
    video_uid_list = video_uid_list[video_uid_list > 1].index.tolist()
    sampled_df = df[df['video_uid'].isin(video_uid_list)]
    print(
        f'Filtered to {len(sampled_df)} rows from {len(df)} with same video'
        ' uids'
    )
    sampled_df = sampled_df.sample(sample_num + 200, random_state=42)
  else:
    sampled_df = df.sample(sample_num + 200, random_state=42)

  save_rows = []
  for i, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
    if len(save_rows) == sample_num:
      break
    current_verbs = row['verb_list']
    non_overlap_mask = df['verb_list'].apply(
        lambda x: x.isdisjoint(current_verbs)
    )
    sub_df = df[non_overlap_mask].copy()
    sub_df = sub_df.sample(len(sub_df), random_state=42)
    if len(sub_df) < 4:
      continue

    if same_video:
      same_video_mask = sub_df['video_uid'] == row['video_uid']
      selected_df = sub_df[same_video_mask]
      selected_indices = [selected_df.index[0]] if len(selected_df) > 0 else []
    else:
      selected_indices = []
    remaining_df = sub_df.copy()

    while len(selected_indices) < 4 and len(remaining_df) > 0:
      # Get the first row
      current_idx = remaining_df.index[0]
      selected_indices.append(current_idx)

      # Remove rows that have any overlapping verbs with any selected row
      remaining_df = remaining_df.iloc[
          1:
      ].copy()  # Make a copy to avoid SettingWithCopyWarning

      # Create a mask for rows that are disjoint from all selected rows
      disjoint_mask = pd.Series(True, index=remaining_df.index)
      for selected_idx in selected_indices:
        selected_verbs = df.loc[selected_idx, 'verb_list']
        disjoint_mask &= remaining_df['verb_list'].apply(
            lambda x: x.isdisjoint(selected_verbs)
        )

      remaining_df = remaining_df[disjoint_mask]

    if len(selected_indices) < 4:
      continue

    # Get the selected rows using the original DataFrame
    sub_df = df.loc[[i] + selected_indices].copy()
    # Add gt column - first row is True, others are False
    sub_df['gt'] = [True] + [False] * (len(sub_df) - 1)

    text_list = sub_df['description_text'].tolist()
    video_uid_list = sub_df['video_uid'].tolist()
    text_verb_list = sub_df['verb_list'].tolist()

    # for j, text in enumerate(text_list):
    #     print(f"{j+1}. {text}, {video_uid_list[j]}, {text_verb_list[j]}")
    # print('-'*50)

    save_rows.append(sub_df)

  print(f'Saving {len(save_rows)} rows')
  save_df = pd.concat(save_rows)
  fn = 'samevideo_' if same_video else ''
  save_df.to_csv(
      f'data/dynpose-100k/dynpose_100k/metadata_{fn}test5000.csv', index=False
  )


if __name__ == '__main__':
  # split_egoexo4d_pretraining_val()
  # split_egoexo4d_pretraining_val_v2(ego_visible=False)
  # split_egoexo4d_pretraining_val_v3(ego_visible=False)  # Sample from all scenarios
  # split_egoexo4d_pretraining_val_v3(ego_visible=False, scenario='bouldering')
  # split_egoexo4d_pretraining_val_v4('baselines/logs/text_features.pt', total_samples=1000)

  # for task_name in ['Cooking', 'Health', 'Dance', 'Rock Climbing', 'Bike Repair', 'Basketball', 'Soccer', 'Music']:
  #     for ego_visible in [True, False]:
  #         split_mcq('data/egoexo4d/annotations/pretraining/val_presr50_cleanv1_components_motionstd.csv', ego_visible, 500, task_name, 0.01)
  #         print('-'*50)
  # combine_files()

  # split_dynpose_pretraining_val('local_data/dynpose-100k/dynpose_100k/metadata.csv')
  # for key in ['Describe my body posture', 'Describe my hands/arms motion', 'Describe my legs/feet motion']: # 'Describe my focus attention']:
  #     split_nymeria_retrieval(key, 1000)
  # split_nymeria_retrieval('Describe my focus attention', 1000)

  split_dynpose_retrieval(same_video=True)
