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
import glob
import json
import random
from downstream.linearSVM import run_svm
import numpy as np
from sklearn.model_selection import train_test_split
import torch


def sample_balanced(features, labels, per_class, seed=42, allow_smaller=True):
  """Returns (X_bal, y_bal, idx_bal) where each class {0,1} has up to `per_class` samples.

  If a class has fewer than `per_class` and allow_smaller=True, it uses all
  available from that class. Set allow_smaller=False to raise an error instead.
  """
  assert features.shape[0] == labels.shape[0], 'Mismatched features/labels.'
  rng = np.random.default_rng(seed)

  idx0 = np.where(labels == 0)[0]
  idx1 = np.where(labels == 1)[0]
  n0, n1 = len(idx0), len(idx1)

  def pick(idx, want):
    if len(idx) < want:
      if allow_smaller:
        return idx  # take all
      else:
        raise ValueError(
            f'Not enough samples in class for want={want}: have {len(idx)}.'
        )
    return rng.choice(idx, size=want, replace=False)

  # Choose counts
  want0 = min(per_class, n0) if allow_smaller else per_class
  want1 = min(per_class, n1) if allow_smaller else per_class

  sel0 = pick(idx0, want0)
  sel1 = pick(idx1, want1)

  idx_bal = np.concatenate([sel0, sel1])
  rng.shuffle(idx_bal)

  X_bal = features[idx_bal]
  y_bal = labels[idx_bal]

  print('Original label counts:', Counter(labels.tolist()))
  print(f'Selected counts: {{0: {len(sel0)}, 1: {len(sel1)}}}')
  return X_bal, y_bal, idx_bal


def run(input_file, feature_file, seed=42):
  idx_list, label_list = [], []
  answer_map1 = {'A': 0, 'B': 1}
  answer_map2 = {'A': 0, 'B': 0, 'C': 1, 'D': 1}
  answer_map = answer_map2 if 'peoplecount' in input_file else answer_map1
  with open(input_file, 'r') as f:
    for line in f:
      tmp = json.loads(line)
      if tmp['response'][0] not in answer_map.keys():
        # print(f"Invalid response: {tmp['response']}")
        continue
      idx_list.append(int(tmp['idx']))
      label_list.append(answer_map[tmp['response'][0]])

  features = torch.load(feature_file).numpy()
  features = features[idx_list]
  print('Original:', Counter(label_list), features.shape)
  labels = np.array(label_list)

  features, labels, _ = sample_balanced(
      features, labels, per_class=1500, seed=seed, allow_smaller=True
  )
  print('Balanced:', Counter(labels), features.shape)

  X_train, X_val, y_train, y_val = train_test_split(
      features,
      labels,
      test_size=0.2,
      stratify=labels,  # preserve class balance
      random_state=42,  # reproducibility
  )
  print(f'Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}')

  # Majority baseline
  majority_class = Counter(y_train).most_common(1)[0][0]
  majority_preds = np.full_like(y_val, majority_class)
  majority_acc = (majority_preds == y_val).mean()
  print(f'Majority baseline accuracy: {majority_acc:.3f}')

  run_svm(X_train, X_val, y_train, y_val)


if __name__ == '__main__':
  feature_dir = (
      'data/misc/retrieval_features/ours/dynpose_pretrain/val_v1_gt/frames.pt'
  )
  response_dir = 'data/dynpose-100k/dynpose_100k/vlm_queries/output'
  response_files = glob.glob(f'{response_dir}/*/*.jsonl')
  # for rf in response_files:
  #     if 'vtalign' in rf:
  #         continue
  #     print('*' * 20, rf, '*' * 20)
  #     run(rf, feature_dir, seed=42)
  #     print('*' * 100)
  run(
      'data/dynpose-100k/dynpose_100k/vlm_queries/output/walking/gemini-2.5-flash_4frames_nothinking.jsonl',
      feature_dir,
      seed=9908,
  )
