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

"""Compute 5-way multiple-choice retrieval accuracy from dumped CamFormer features.

`train.py --test` writes `frames.pt` (motion embeddings) and `text.pt` (text
embeddings) to a feature directory, in dataset order, and prints that directory.
The MCQ test sets are laid out as consecutive groups of 5 rows, where the first
row of each group is the correct (motion, text) match. This script reports top-1
accuracy in both directions:

    motion -> text : among the 5 candidate texts, is the correct one ranked
    first?
    text -> motion : among the 5 candidate motions, is the correct one ranked
    first?

Usage:
    python eval_retrieval.py <feature_dir>
"""

import argparse

import torch
import torch.nn.functional as F


def mcq_accuracy(feature_dir, group_size=5):
  frames = torch.load(f'{feature_dir}/frames.pt', map_location='cpu').float()
  texts = torch.load(f'{feature_dir}/text.pt', map_location='cpu').float()
  n = frames.shape[0]
  if n % group_size != 0:
    raise ValueError(
        f'{n} features is not a multiple of {group_size}; this does not look'
        f' like a {group_size}-way MCQ set (e.g. DynPose val_v1 is'
        ' full-retrieval).'
    )

  frames = F.normalize(frames, dim=-1)
  texts = F.normalize(texts, dim=-1)

  n_groups = n // group_size
  m2t_correct = 0
  t2m_correct = 0
  for g in range(n_groups):
    sl = slice(g * group_size, (g + 1) * group_size)
    sim = (
        frames[sl] @ texts[sl].t()
    )  # (group_size, group_size), row/col 0 = correct
    m2t_correct += int(sim[0].argmax().item() == 0)
    t2m_correct += int(sim[:, 0].argmax().item() == 0)

  m2t, t2m = m2t_correct / n_groups, t2m_correct / n_groups
  print(
      f'Loaded {n} features ({n_groups} groups of {group_size}) from'
      f' {feature_dir}'
  )
  print(f'Motion->Text MCQ acc: {m2t:.4f}')
  print(f'Text->Motion MCQ acc: {t2m:.4f}')
  return m2t, t2m


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='5-way MCQ retrieval accuracy from dumped features.'
  )
  parser.add_argument(
      'feature_dir',
      help=(
          'Directory containing frames.pt and text.pt (printed by train.py'
          ' --test).'
      ),
  )
  parser.add_argument('--group_size', type=int, default=5)
  args = parser.parse_args()
  mcq_accuracy(args.feature_dir, args.group_size)
