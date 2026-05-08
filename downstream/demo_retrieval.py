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
import pandas as pd
import torch


def v2t_retrieval(setting):
  if setting == 'egoexo4d':
    df = pd.read_csv(
        'data/egoexo4d/annotations/pretraining/test_alltasks_mcqv0.csv'
    )
    cm_feat_path = 'data/misc/retrieval_features/ours/demo/bs1024_sampledur8_pose5_pi3pose/frames.pt'
    text_feat_path = 'data/misc/retrieval_features/ours/egoexo4d_pretrain_longseq/all/bs1024_sampledur8_pose5_pi3pose/0.5/testdur4/text.pt'
  else:
    df = pd.read_csv('data/dynpose-100k/dynpose_100k/metadata_val_v1.csv')
    # df = pd.read_csv('data/dynpose-100k/dynpose_100k/metadata_with_path.csv')
    cm_feat_path = 'data/misc/retrieval_features/ours/demo/v1_vipe/frames.pt'
    text_feat_path = (
        'data/misc/retrieval_features/ours/dynpose_pretrain/val_v1_vipe/text.pt'
    )
    # text_feat_path = './logs/text_features.pt'
    pose_files = sorted(glob.glob(f'data/demo_video_clips/vipeposes_v0/*.npz'))

  cm_feat = torch.load(cm_feat_path)
  text_feat = torch.load(text_feat_path).cpu()
  caption_list = df['description_text'].tolist()
  print(cm_feat.shape, text_feat.shape, len(caption_list), len(pose_files))

  cm_feat = torch.nn.functional.normalize(cm_feat, dim=-1)
  text_feat = torch.nn.functional.normalize(text_feat, dim=-1)

  # Compute similarity
  # sim = cm_feat @ text_feat.T   # [147, 35395]
  sim = cm_feat @ text_feat.to(torch.float32).T

  # For each camera feature row
  for i in range(cm_feat.shape[0]):
    topk_values, topk_indices = torch.topk(
        sim[i], k=20
    )  # get more to allow filtering
    seen = set()
    unique_results = []

    for idx, score in zip(topk_indices.tolist(), topk_values.tolist()):
      caption = caption_list[idx].strip()
      if caption not in seen:
        seen.add(caption)
        unique_results.append((caption, score))
      if len(unique_results) == 10:  # stop after 5 unique captions
        break
    # if unique_results[0][1] < 0.2:
    #     continue
    # if 'IMG' not in pose_files[i]:
    # continue
    if i not in [2, 4, 75, 92]:
      continue
    print(f'=== Camera feature {i}, {pose_files[i]} ===')
    for rank, (caption, score) in enumerate(unique_results, start=1):
      print(f'Top {rank}: {caption}  (score={score:.4f})')
    print('=' * 50)


def t2v_retrieval(mode):
  from baselines.run_clip import demo_text_dict

  text_list = demo_text_dict[mode]
  df = pd.read_csv('data/dynpose-100k/dynpose_100k/metadata_val_v1.csv')
  text_feat = torch.load(
      f'data/misc/retrieval_features/demo/text_{mode}.pt'
  ).cpu()
  cm_feat = torch.load(
      'data/misc/retrieval_features/ours/dynpose_pretrain/val_v1_vipe/frames.pt'
  ).cpu()

  text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
  cm_feat = torch.nn.functional.normalize(cm_feat, dim=-1)

  sim = cm_feat.float() @ text_feat.float().T
  print(cm_feat.shape, text_feat.shape, sim.shape, len(df))

  for i in range(text_feat.shape[0]):
    topk_values, topk_indices = torch.topk(
        sim[:, i], k=10
    )  # get more to allow filtering
    print(f'Text query {i}: {text_list[i]}')
    for rank, (idx, score) in enumerate(
        zip(topk_indices.tolist(), topk_values.tolist()), start=1
    ):
      caption = df['description_text'].tolist()[idx]
      print(
          f'Top {rank}: Camera feature {idx}  (score={score:.4f}), caption:'
          f' {caption}'
      )
    print('=' * 50)


def t2v_retrieval_egoexo4d():
  feature_dir = 'data/misc/retrieval_features/ours/egoexo4d_pretrain_longseq/all/bs1025_sampledur8_pose2/0.5/testdur4'
  cm_feat = torch.load(f'{feature_dir}/frames.pt')
  text_feat = torch.load(f'{feature_dir}/text.pt')
  cm_feat = torch.nn.functional.normalize(cm_feat, dim=-1)
  text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
  sim = torch.matmul(text_feat, cm_feat.T)

  df = pd.read_csv(
      'data/egoexo4d/annotations/pretraining/test_alltasks_mcqv0.csv'
  )
  caption_list = df['description_text'].tolist()
  for i, (text, count) in enumerate(
      df['description_text'].value_counts().head(20).items(), 1
  ):
    query_idx = df.index[df['description_text'] == text][0]
    query_similarities = sim[query_idx]
    topk_scores, topk_indices = torch.topk(query_similarities, k=10)
    print('*' * 5, f'{i}. {count}\t{text}', '*' * 5)
    for rank, (idx, score) in enumerate(
        zip(topk_indices.tolist(), topk_scores.tolist()), start=1
    ):
      caption = caption_list[idx]
      print(
          f'Top {rank}: Camera feature {idx}  (score={score:.4f}), caption:'
          f' {caption}'
      )
    print([query_idx.item()] + topk_indices.tolist())
    print('-' * 100)


if __name__ == '__main__':
  v2t_retrieval('dynpose')
  # t2v_retrieval('a')
  # t2v_retrieval_egoexo4d()
