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
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_features(feature_dir):
  frame_feat = F.normalize(torch.load(f'{feature_dir}/frames.pt').cpu(), dim=-1)
  text_feat = F.normalize(torch.load(f'{feature_dir}/text.pt').cpu(), dim=-1)
  return frame_feat, text_feat


def cross_dataset_retrieval():
  data_file_ego = (
      'data/egoexo4d/annotations/pretraining/test_alltasks_mcqv0.csv'
  )
  df_ego = pd.read_csv(data_file_ego)
  caption_list_ego = df_ego['description_text'].tolist()
  feature_dir_ego = 'data/misc/retrieval_features/ours/egoexo4d_pretrain_longseq/all/bs1024_sampledur8_pose5/0.5/testdur4'
  feature_dir_exo = (
      'data/misc/retrieval_features/ours/dynpose_pretrain/val_v1_vipe'
  )

  data_file_exo = f'data/dynpose-100k/dynpose_100k/metadata_val_v1.csv'
  df_exo = pd.read_csv(data_file_exo)
  caption_list_exo = df_exo['description_text'].tolist()
  frame_feat_ego, text_feat_ego = load_features(feature_dir_ego)
  frame_feat_exo, text_feat_exo = load_features(feature_dir_exo)
  print(
      frame_feat_ego.shape,
      text_feat_ego.shape,
      frame_feat_exo.shape,
      text_feat_exo.shape,
      len(caption_list_ego),
      len(caption_list_exo),
  )

  topk = 5  # number of top exo matches for each ego frame
  results = []

  batch_size = 512
  for i in range(0, frame_feat_ego.size(0), batch_size):
    f_ego = frame_feat_ego[i : i + batch_size]  # (B, 512)
    t_ego = text_feat_ego[i : i + batch_size]

    # Compute cosine similarity with all exo frames
    sim_f = f_ego @ frame_feat_exo.T  # (B, N_exo)
    sim_t = t_ego @ text_feat_exo.T  # (B, N_exo)

    # top-k visual matches
    top_sim_f, top_idx_f = torch.topk(sim_f, topk, dim=-1)

    for b in range(f_ego.size(0)):
      ego_idx = i + b
      exo_indices = top_idx_f[b]
      f_scores = top_sim_f[b]
      t_scores = sim_t[b, exo_indices]

      for rank in range(topk):
        results.append({
            'ego_idx': ego_idx,
            'exo_idx': exo_indices[rank].item(),
            'rank': rank,
            'frame_sim': f_scores[rank].item(),
            'text_sim': t_scores[rank].item(),
            'top5_exo_indices': exo_indices.tolist(),
            'f_scores': f_scores.tolist(),
            't_scores': t_scores.tolist(),
        })

  w0, w1 = 0.55, 0.45
  sorted_results = sorted(
      results,
      key=lambda x: (w0 * x['frame_sim'] + w1 * x['text_sim']),
      reverse=True,
  )

  for r in sorted_results[:20]:
    print(
        f"Ego {r['ego_idx']} ↔ Exo {r['exo_idx']} | f_sim={r['frame_sim']:.3f},"
        f" t_sim={r['text_sim']:.3f} | rank={r['rank']}"
    )
    print(f"Ego query {r['ego_idx']}: {caption_list_ego[r['ego_idx']]}")
    for j, exo_idx in enumerate(r['top5_exo_indices']):
      frame_score = r['f_scores'][j]
      text_score = r['t_scores'][j]
      indicator = '***' if exo_idx == r['exo_idx'] else ''
      print(
          f'-> {indicator} Exo {exo_idx}: {caption_list_exo[exo_idx]}, frame'
          f' sim score = {frame_score:.3f}, text sim score = {text_score:.3f}'
      )
    print('-' * 30)


def cross_dataset_retrieval_baseline():
  data_file_exo = f'data/dynpose-100k/dynpose_100k/metadata_val_v1.csv'
  df_exo = pd.read_csv(data_file_exo)
  caption_list_exo = df_exo['description_text'].tolist()

  ego_feat = np.load(
      'data/misc/retrieval_features/ours/egoexo4d_pretrain_longseq/all/bs1024_sampledur8_pose5/0.5/testdur4/input_cam.npy'
  )
  exo_feat = np.load(
      'data/misc/retrieval_features/ours/dynpose_pretrain/val_v1_vipe/input_cam.npy'
  )
  ego_norm = ego_feat / np.linalg.norm(ego_feat, axis=1, keepdims=True)
  exo_norm = exo_feat / np.linalg.norm(exo_feat, axis=1, keepdims=True)
  print(ego_feat.shape, exo_feat.shape, len(caption_list_exo))

  ego_idx = [12565, 9379]
  ego_subset = ego_norm[ego_idx]
  sim = ego_subset @ exo_norm.T  # (B, N_exo)

  topk = 5
  topk_idx = np.argsort(sim, axis=1)[:, -topk:][:, ::-1]  # top-5 per query
  topk_sim = np.take_along_axis(sim, topk_idx, axis=1)

  for n, idx in enumerate(ego_idx):
    print(f'Ego idx {idx}:')
    for rank in range(topk):
      exo_id = topk_idx[n, rank]
      score = topk_sim[n, rank]
      print(
          f'  Top {rank+1}: Exo idx {exo_id}: {caption_list_exo[exo_id]},'
          f' sim={score:.4f}'
      )


def dynpose_retrieval():
  feature_dir_exo = (
      'data/misc/retrieval_features/ours/dynpose_pretrain/val_v1_vipe'
  )
  data_file_exo = f'data/dynpose-100k/dynpose_100k/metadata_val_v1.csv'
  df_exo = pd.read_csv(data_file_exo)
  caption_list_exo = df_exo['description_text'].tolist()
  frame_feat_exo, text_feat_exo = load_features(feature_dir_exo)

  # Compute similarity matrices
  sim_f = frame_feat_exo @ frame_feat_exo.T  # (N, N)
  sim_t = text_feat_exo @ text_feat_exo.T  # (N, N)

  # Avoid self-matching
  sim_f.fill_diagonal_(-1.0)

  # Top-5 visual matches
  top_sim_f, top_idx_f = torch.topk(sim_f, k=5, dim=-1)

  # Gather corresponding text similarities
  results = []
  for i in range(frame_feat_exo.size(0)):
    match_indices = top_idx_f[i]
    f_scores = top_sim_f[i]
    t_scores = sim_t[i, match_indices]

    for k in range(5):
      results.append({
          'query_idx': i,
          'match_idx': match_indices[k].item(),
          'frame_sim': f_scores[k].item(),
          'text_sim': t_scores[k].item(),
          'top5_match_indices': match_indices.tolist(),
          'f_scores': f_scores.tolist(),
          't_scores': t_scores.tolist(),
      })

  w0, w1 = 0.5, 0.5
  sorted_results = sorted(
      results,
      key=lambda x: (w0 * x['frame_sim'] + w1 * x['text_sim']),
      reverse=True,
  )
  for r in sorted_results[:20]:
    print(
        f"Query {r['query_idx']} ↔ Match {r['match_idx']} |"
        f" f_sim={r['frame_sim']:.3f}, t_sim={r['text_sim']:.3f}"
    )
    print(f"Query {r['query_idx']}: {caption_list_exo[r['query_idx']]}")
    for j, match_idx in enumerate(r['top5_match_indices']):
      frame_score = r['f_scores'][j]
      text_score = r['t_scores'][j]
      indicator = '***' if match_idx == r['match_idx'] else ''
      print(
          f'-> {indicator} Match {match_idx}: {caption_list_exo[match_idx]},'
          f' frame sim score = {frame_score:.3f}, text sim score ='
          f' {text_score:.3f}'
      )
    print('-' * 30)


def egoexo4d_retrieval(num=10):
  from utils.read_result import load_mapping

  feature_dir = 'data/misc/retrieval_features/ours/egoexo4d_pretrain_longseq/all_gravityFalse/bs1024_sampleddur8_pose11/0.5/testdur4'
  data_file = 'data/egoexo4d/annotations/pretraining/test_alltasks_mcqv0.csv'
  df = pd.read_csv(data_file)

  df['take_name'] = df['save_id'].apply(lambda x: '_'.join(x.split('_')[:-2]))
  take_name_mapping = load_mapping()
  df['parent_task_name'] = df['take_name'].apply(
      lambda x: take_name_mapping.get(x, 'Unknown')
  )
  # df.to_csv('data/egoexo4d/annotations/pretraining/test_alltasks_mcqv0_with_taskname.csv', index=False)

  caption_list = df['description_text'].tolist()
  frame_feat, text_feat = load_features(feature_dir)

  sim_f = frame_feat @ frame_feat.T  # (N, N)
  sim_t = text_feat @ text_feat.T  # (N, N)
  sim_f.fill_diagonal_(-1.0)
  top_sim_f, top_idx_f = torch.topk(sim_f, k=num, dim=-1)

  for i in range(frame_feat.size(0)):
    match_indices = top_idx_f[i]
    f_scores = top_sim_f[i]
    t_scores = sim_t[i, match_indices]
    print_first_line = False
    for k in range(num):
      match_idx = match_indices[k].item()
      frame_score = f_scores[k].item()
      text_score = t_scores[k].item()
      task_name = df.iloc[match_idx]['parent_task_name']
      if task_name == df.iloc[i]['parent_task_name'] or text_score < 0.9:
        continue
      if not print_first_line:
        print_first_line = True
        print('-' * 50)
        print(
            f"Query {i} (Task: {df.iloc[i]['parent_task_name']}):"
            f' {caption_list[i]}'
        )
      print(
          f'-> Match {match_idx} (Task:'
          f" {df.iloc[match_idx]['parent_task_name']}):"
          f' {caption_list[match_idx]}, frame sim score = {frame_score:.3f},'
          f' text sim score = {text_score:.3f}'
      )


def dynpose_stat():
  data_file_exo = (
      f'data/dynpose-100k/dynpose_100k/metadata_val_v1_with_verbs.csv'
  )
  df = pd.read_csv(data_file_exo)
  df['verb_list'] = df['verb_lemmas'].apply(lambda x: set(ast.literal_eval(x)))
  verb_list = []
  cnt = 0
  for i, row in df.iterrows():
    if 'dog' in row['description_text']:
      cnt += 1
      # print(i, row['description_text'])
    verb_list.extend(list(row['verb_list']))
  verb_freq = pd.Series(verb_list).value_counts()
  # print(cnt)
  print('Top 20 verbs:')
  print(verb_freq.head(20))


if __name__ == '__main__':
  egoexo4d_retrieval()
  # cross_dataset_retrieval_baseline()
  # cross_dataset_retrieval()
  # dynpose_retrieval()
  # dynpose_stat()
