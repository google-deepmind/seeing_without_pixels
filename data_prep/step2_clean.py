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

"""Step 2: Drop rows whose trajectory cache is missing or whose description is not

a valid string, producing `{mode}_presr50_cleanv1.csv`.
"""

import argparse
import os

import pandas as pd
from tqdm import tqdm


def main(mode, data_dir, sample_rate, suffix):
  in_path = os.path.join(
      data_dir,
      'annotations',
      'pretraining',
      f'{mode}_presr{sample_rate}{suffix}.csv',
  )
  df = pd.read_csv(in_path)
  print(f'Loaded {len(df)} rows from {in_path}')

  df = df[df['description_text'].apply(lambda x: isinstance(x, str))]
  print(f'After text filtering: {len(df)} rows')

  cache_root = os.path.join(
      data_dir,
      'camera_motion_cache',
      'egoexo4d_pretrain',
      f'{mode}_presr{sample_rate}',
  )
  keep = []
  for _, row in tqdm(df.iterrows(), total=len(df)):
    keep.append(
        os.path.exists(
            os.path.join(cache_root, f"{row['save_id']}_relative.npy")
        )
    )
  df = df[keep]
  print(f'After cache check: {len(df)} rows')

  out_path = os.path.join(
      data_dir,
      'annotations',
      'pretraining',
      f'{mode}_presr{sample_rate}_cleanv1.csv',
  )
  df.to_csv(out_path, index=False)
  print(f'Wrote {out_path}')


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
      '--suffix',
      type=str,
      default='',
      help=(
          "Extra suffix of the step1 CSV (e.g. '_filtered' if you"
          ' pre-filtered).'
      ),
  )
  args = parser.parse_args()
  main(args.mode, args.data_dir, args.sample_rate, args.suffix)
