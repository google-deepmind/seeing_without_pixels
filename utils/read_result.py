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

# import seaborn as sns
# import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter


def read_hdepic_pred(test_file, pred_dir):
  df = pd.read_csv(test_file)
  annotation_ids = df['unique_narration_id'].unique()
  print(f'There are {len(annotation_ids)} unique annotation_ids')
  pred_files = glob.glob(os.path.join(pred_dir, '*.csv'))
  for pred_file in pred_files:
    pred_df = pd.read_csv(pred_file)
    pred_df['annotation_id'] = pred_df['annotation_id'].str.replace('_', '-')
    pred_subdf = pred_df[pred_df['annotation_id'].isin(annotation_ids)]

    # check acc by comparing pred_verb column with gt_verb column
    verb_acc = (pred_subdf['pred_verb'] == pred_subdf['gt_verb']).mean() * 100
    noun_acc = (pred_subdf['pred_noun'] == pred_subdf['gt_noun']).mean() * 100
    all_acc = (
        (pred_subdf['pred_verb'] == pred_subdf['gt_verb'])
        & (pred_subdf['pred_noun'] == pred_subdf['gt_noun'])
    ).mean() * 100

    # Calculate accuracy for entire dataframe
    df_verb_acc = (pred_df['pred_verb'] == pred_df['gt_verb']).mean() * 100
    df_noun_acc = (pred_df['pred_noun'] == pred_df['gt_noun']).mean() * 100
    df_all_acc = (
        (pred_df['pred_verb'] == pred_df['gt_verb'])
        & (pred_df['pred_noun'] == pred_df['gt_noun'])
    ).mean() * 100
    print(pred_file)
    print(
        'Verb range',
        pred_subdf['pred_verb'].min(),
        pred_subdf['pred_verb'].max(),
    )
    print(
        f'Subset Accuracies ({len(pred_subdf)} samples) - Verb: {verb_acc:.2f},'
        f' Noun: {noun_acc:.2f}, Overall: {all_acc:.2f}'
    )
    print(
        f'Full Dataset Accuracies ({len(pred_df)} samples) - Verb:'
        f' {df_verb_acc:.2f}, Noun: {df_noun_acc:.2f}, Overall:'
        f' {df_all_acc:.2f}'
    )
    print('-' * 100)


def get_confusion_matrix(pred_file, top_k=15, show_ratio=False):
  # Read the JSON file
  with open(pred_file, 'r') as f:
    predictions = json.load(f)

  # Extract ground truth and predicted labels
  gt_labels = []
  pred_labels = []

  for key, pred in predictions.items():
    gt_labels.append(str(pred['gt_label']) + '_' + str(pred['gt_label_name']))
    pred_labels.append(
        str(pred['top1_pred']['label'])
        + '_'
        + str(pred['top1_pred']['label_name'])
    )

  # Get label frequencies and select top k labels
  label_freq = pd.Series(gt_labels).value_counts()
  top_k_labels = label_freq.head(top_k).index.tolist()

  # Filter data to include only top k labels
  filtered_indices = [
      i for i, label in enumerate(gt_labels) if label in top_k_labels
  ]
  filtered_gt = [gt_labels[i] for i in filtered_indices]
  filtered_pred = [pred_labels[i] for i in filtered_indices]

  # Compute confusion matrix for top k classes
  normalize_option = 'true' if show_ratio else None
  cm = confusion_matrix(
      filtered_gt,
      filtered_pred,
      labels=top_k_labels,
      normalize=normalize_option,
  )

  # Adjust font sizes based on number of classes
  base_font_size = 8 if top_k > 30 else 12
  annotation_size = 10 if top_k > 30 else 14
  title_size = 14 if top_k > 30 else 16
  label_size = 12 if top_k > 30 else 14

  # Set font sizes
  plt.rcParams.update({'font.size': base_font_size})

  # Create a figure with larger size
  plt.figure(figsize=(16, 12))

  # Create heatmap using seaborn with larger font sizes
  fmt = '.2f' if show_ratio else 'd'
  sns.heatmap(
      cm,
      annot=True,
      fmt=fmt,
      cmap='Blues',
      xticklabels=top_k_labels,
      yticklabels=top_k_labels,
      annot_kws={'size': annotation_size},
  )

  title = f'Confusion Matrix (Top {top_k} Most Frequent Classes)'
  if show_ratio:
    title = 'Normalized ' + title
  plt.title(title, fontsize=title_size, pad=20)
  plt.xlabel('Predicted Label', fontsize=label_size, labelpad=10)
  plt.ylabel('True Label', fontsize=label_size, labelpad=10)

  # Rotate x-axis labels for better readability
  plt.xticks(rotation=45, ha='right', fontsize=base_font_size)
  plt.yticks(rotation=0, fontsize=base_font_size)

  # Adjust layout to prevent label cutoff
  plt.tight_layout()

  # Save the plot with higher DPI for better quality
  output_dir = os.path.dirname(pred_file)
  filename = f'confusion_matrix_top{top_k}'
  if show_ratio:
    filename += '_normalized'
  output_path = os.path.join(output_dir, f'{filename}.png')
  plt.savefig(output_path, dpi=300, bbox_inches='tight')
  plt.close()
  print(f'Confusion matrix saved to: {output_path}')


def reformat_result(pred_file, caption_file):
  result_list = []
  with open(pred_file, 'r') as f:
    for line in f:
      ans_dict = json.loads(line)
      result_list.append(ans_dict['response'][0])

  df = pd.read_csv(caption_file)
  caption_list = df['description_text'].tolist()

  with open('tmp/result.tsv', 'w') as f:
    f.write('Caption\tLabel\n')
    for caption, item in zip(caption_list, result_list):
      f.write(f'{caption}\t{item}\n')


def read_mcq_pred(pred_file, by_type=False, print_correct=False):
  total, correct = 0, 0
  correct_dict = defaultdict(int)
  total_dict = defaultdict(int)
  if by_type or print_correct:
    input_file = (
        os.path.dirname(pred_file).replace('output', 'input')
        + '_byscenariotype.json'
    )  # '_new.json', '_byscenariotype.json'
    with open(input_file, 'r') as f:
      input_data_list = json.load(f)
    input_data_dict = {d['qa_idx']: d for d in input_data_list}
  result_list = []
  res_dist = defaultdict(int)
  ans_dist = defaultdict(int)
  with open(pred_file, 'r') as f:
    for line in f:
      ans_dict = json.loads(line)
      idx = int(ans_dict['idx'].split('_')[-1])
      is_correct = ans_dict['response'] == ans_dict['answer'][0]
      correct += is_correct
      total += 1
      result_list.append(ans_dict['response'][0])
      res_dist[ans_dict['answer'][0]] += is_correct
      ans_dist[ans_dict['answer'][0]] += 1
      if by_type:
        # q_type = ans_dict['idx'].split('_')[1]
        q_type = input_data_dict[ans_dict['idx']]['q_type']
        correct_dict[q_type] += is_correct
        total_dict[q_type] += 1
      if print_correct:
        if is_correct:
          print(input_data_dict[ans_dict['idx']]['query'])
          print(ans_dict['response'], ans_dict['answer'])
          print('-' * 50)
  print(
      f'Total: {total}, Correct: {correct}, Accuracy: {correct/total*100:.2f}%'
  )
  if by_type:
    for k, v in correct_dict.items():
      print(f'{k}: {v}/{total_dict[k]} ({v/total_dict[k]*100:.2f}%)')

  for k, v in res_dist.items():
    print(f'{k}: {v}/{ans_dist[k]} ({v/ans_dist[k]*100:.2f}%)')


def load_mapping():
  data_file = 'data/egoexo4d/takes.json'
  with open(data_file, 'r') as f:
    take_list = json.load(f)
  take_name_mapping = {}
  for data_dict in take_list:
    take_name_mapping[data_dict['take_name']] = data_dict['parent_task_name']
  return take_name_mapping


def sanitize(text):
  if not isinstance(text, str):
    text = str(text)
  # Replace tabs, newlines, and carriage returns with a space
  return text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')


def viz_um_embedding(feature_path, file_path, log_dir):
  take_name_mapping = load_mapping()
  os.system(f'rm -rf {log_dir}')
  features = torch.load(f'{feature_path}/frames.pt')
  df = pd.read_csv(file_path)
  df['take_name'] = df['save_id'].apply(lambda x: '_'.join(x.split('_')[:-2]))
  df['parent_task_name'] = df['take_name'].apply(
      lambda x: take_name_mapping.get(x, 'Unknown')
  )
  captions = df['description_text'].tolist()
  take_names = df['take_name'].tolist()
  task_name_list = df['parent_task_name'].tolist()

  metadata = [
      [f'{i}. {sanitize(caption)}', task_name, take_name]
      for i, (caption, task_name, take_name) in enumerate(
          zip(captions, task_name_list, take_names)
      )
  ]
  print(features.shape, len(captions), len(take_names), len(metadata))
  writer = SummaryWriter(log_dir)
  writer.add_embedding(
      mat=features,
      metadata=metadata,
      metadata_header=['caption', 'task_name', 'take_name'],
      tag='camera_features',
  )
  writer.close()


def viz_mm_embedding(feature_path, file_path, log_dir, sampled_k=10000):
  cam_features = torch.load(f'{feature_path}/frames.pt')
  text_features = torch.load(f'{feature_path}/text.pt')
  df = pd.read_csv(file_path)
  captions = df['description_text'].tolist()

  idx = torch.randperm(len(df))[:sampled_k]
  cam_features = cam_features[idx]
  text_features = text_features[idx]
  captions = [captions[i] for i in idx]

  features = torch.cat([cam_features, text_features], dim=0)
  both_captions = captions + captions
  modalities = ['cam'] * len(captions) + ['text'] * len(captions)
  multi_meta = [[c, m] for c, m in zip(both_captions, modalities)]

  writer = SummaryWriter(log_dir)
  writer.add_embedding(
      mat=features,
      metadata=multi_meta,
      metadata_header=['caption', 'modality'],
      tag='cam_vs_text',
  )
  writer.close()


def viz_mm_embedding2(
    feature_path1,
    file_path1,
    feature_path2,
    file_path2,
    log_dir,
    sampled_k=1000,
):
  df = pd.read_csv(file_path1)
  idx = torch.randperm(len(df))[:sampled_k]

  cam_features = torch.load(f'{feature_path1}/frames.pt')[idx]
  text_features = torch.load(f'{feature_path1}/text.pt')[idx]
  cam_features2 = torch.load(
      f"{feature_path1.replace('0.5', '0.5')}/frames.pt"
  )[idx]
  print(
      cam_features.shape,
      text_features.shape,
      cam_features2.shape,
      text_features.shape,
  )

  # df2 = pd.read_csv(file_path2)
  text_features_list = []
  for mode in ['a', 'b', 'c', 'd']:
    df2 = pd.read_csv(f'{file_path2}/split_by_Describe_my_focus_attention.csv')
    if mode == 'a':
      idx2 = torch.randperm(len(df2))[:sampled_k]
    cam_features3 = torch.load(f'{feature_path2}/frames.pt')[
        idx2
    ]  # text_{mode}
    t = torch.load(f'{feature_path2}/text.pt')[idx2]  # /text_{mode}
    text_features_list.append(t)
    break
  print(
      cam_features3.shape, len(text_features_list), text_features_list[0].shape
  )

  features = torch.cat(
      [cam_features, cam_features2, cam_features3, text_features]
      + text_features_list,
      dim=0,
  )
  modalities = (
      ['cam_egoexo4d'] * sampled_k
      + ['cam_egoexo4d_nopool'] * sampled_k
      + ['cam_nymeria'] * sampled_k
      + ['text_egoexo4d'] * sampled_k
      + ['text_nymeria_a'] * sampled_k
  )  # + ['text_nymeria_b'] * sampled_k + ['text_nymeria_c'] * sampled_k + ['text_nymeria_d'] * sampled_k
  print(features.shape, len(modalities))

  os.system(f'rm -rf {log_dir}')
  writer = SummaryWriter(log_dir)
  writer.add_embedding(
      mat=features,
      metadata=[[m] for m in modalities],
      tag='cam_vs_text',
  )
  writer.close()


def viz_cm_embedding(data_dir, log_dir, sampled_k=1000):
  feat1_list, feat2_list, feat3_list, feat4_list = [], [], [], []
  for i in range(sampled_k):
    feat1 = np.load(f'{data_dir}/egoexo4d/{i}.npy').mean(axis=0)
    feat2 = np.load(f'{data_dir}/nymeria/{i}.npy').mean(axis=0)
    feat3 = np.load(f'{data_dir}/dynpose_transform/{i}.npy').mean(axis=0)
    feat4 = np.load(f'{data_dir}/dynpose/{i}.npy').mean(axis=0)
    feat1_list.append(feat1)
    feat2_list.append(feat2)
    feat3_list.append(feat3)
    feat4_list.append(feat4)
  in_feat1 = np.stack(feat1_list, axis=0)
  in_feat2 = np.stack(feat2_list, axis=0)
  in_feat3 = np.stack(feat3_list, axis=0)
  in_feat4 = np.stack(feat4_list, axis=0)
  print(in_feat1.shape, in_feat2.shape, in_feat3.shape, in_feat4.shape)

  features = np.concatenate([in_feat1, in_feat2, in_feat3, in_feat4], axis=0)
  datasets = (
      ['egoexo4d'] * sampled_k
      + ['nymeria'] * sampled_k
      + ['dynpose_transform'] * sampled_k
      + ['dynpose'] * sampled_k
  )

  os.system(f'rm -rf {log_dir}')
  writer = SummaryWriter(log_dir)
  writer.add_embedding(
      mat=features,
      metadata=[[m] for m in datasets],
      tag='cam_input',
  )

  # cam_feat = torch.load(f"{data_dir}/oursfeatures/frames.pt")
  # text_feat = torch.load(f"{data_dir}/oursfeatures/text.pt")

  # cam_feat2 = torch.load(f"{data_dir}/oursfeatures_dynpose/frames.pt")
  # text_feat2 = torch.load(f"{data_dir}/oursfeatures_dynpose/text.pt")
  # print(cam_feat.shape, text_feat.shape, cam_feat2.shape, text_feat2.shape)

  # features = torch.cat([cam_feat, text_feat, cam_feat2, text_feat2], dim=0)
  # name = ['cam_egoexo4d'] * sampled_k + ['cam_nymeria'] * sampled_k + ['text_egoexo4d'] * sampled_k + ['text_nymeria'] * sampled_k + ['cam_dynpose'] * sampled_k + ['text_dynpose'] * sampled_k

  # writer.add_embedding(
  #     mat=features,
  #     metadata=[[m] for m in name],
  #     tag="feat_output",
  # )
  writer.close()


def check_embedding_label_alignment(feature_path, pred_file):
  from sklearn.metrics import silhouette_score
  from sklearn.cluster import KMeans

  embeddings = torch.load(feature_path).cpu().numpy()
  mapping = {'A': 0, 'B': 1, 'C': 2}
  label_list = []
  with open(pred_file, 'r') as f:
    for line in f:
      ans_dict = json.loads(line)
      label_list.append(mapping[ans_dict['response'][0]])
  label_list = np.array(label_list)
  # 1. Silhouette Score using true labels
  score_true = silhouette_score(embeddings, label_list)
  print(f'Silhouette Score w.r.t. true labels: {score_true:.2f}')

  # 2. KMeans clustering
  kmeans = KMeans(n_clusters=3, random_state=42)
  cluster_preds = kmeans.fit_predict(embeddings)

  # 3. Silhouette Score using predicted clusters
  score_kmeans = silhouette_score(embeddings, cluster_preds)
  print(f'Silhouette Score w.r.t. KMeans clusters: {score_kmeans:.2f}')


def read_ds_result(log_dir):
  def get_accuracy_from_path(file_path):
    return (
        float(match.group(1))
        if (match := re.search(r'val_acc=([0-9]+\.[0-9]+)', file_path))
        else None
    )

  ckpt_list = glob.glob(f'{log_dir}/**/*.ckpt', recursive=True)
  group_result = defaultdict(list)
  name_mapping = {}
  print(f'Found {len(ckpt_list)} checkpoints in {log_dir}')
  for ckpt in ckpt_list:
    # if 'pi3' not in ckpt:
    #     continue
    category = ckpt.split('/')[-3]
    acc = get_accuracy_from_path(ckpt)
    group_result[category].append(acc)
    name_mapping[acc] = ckpt
  for key, result_list in group_result.items():
    print('-' * 10, f'Category: {key}', '-' * 10)
    result_list = np.array(result_list)
    print(
        f'Found {len(result_list)} results, mean {result_list.mean():.2%}, min'
        f' {result_list.min():.2%}, max {result_list.max():.2%}'
    )
    print(name_mapping[result_list.max()])


if __name__ == '__main__':
  # data_dir = os.path.expanduser('~/hd-epic_action')
  # read_hdepic_pred(os.path.join(data_dir, 'ann_files/HD_EPIC_Narrations_test_filtered.csv'), os.path.join(data_dir, 'predictions'))
  # get_confusion_matrix('logs/predictions/bs64_best-epoch=298-val_acc=0.2924.json')
  # for k in [10, 30, 50, 74]:
  #     get_confusion_matrix('logs/predictions/2gpu_best-epoch=58-val_acc=0.3050.json', top_k=k)
  # read_mcq_pred('data/egoexo4d/annotations/pretraining/vlm_baseline/output/test_alltasks_mcqv0_sampled/gemini-2.5-pro_8frames_nothinking.jsonl', by_type=True)
  read_mcq_pred(
      'data/egoexo4d/annotations/pretraining/vlm_baseline/output/test_alltasks_mcqv0_new/gemini-2.5-pro_8frames_nothinking.jsonl',
      by_type=True,
  )
  # read_mcq_pred('data/nymeria/eval1000/vlm_baseline/output/test_all/gemini-2.5-pro_8frames_nothinking.jsonl', by_type=True)
  # read_mcq_pred('data/egoexo4d/annotations/pretraining/vlm_baseline/output/test_alltasks_mcqv0_new/8frames_Qwen2.5-VL-7B-Instruct.jsonl', by_type=True)
  # read_mcq_pred('data/dynpose-100k/dynpose_100k/vlm_queries/output/camera_qa_camera_filter_qwen2.5-vl-7b-cam-motionl/gemini-2.5-flash_nothinking.jsonl')
  # read_mcq_pred('data/dynpose-100k/dynpose_100k/vlm_queries/output/camera_qa_camera_filter_ShotVL-7Bl/gemini-2.5-flash_nothinking.jsonl')

  # tb_embedding_project('baselines/logs/dynpose_pretrain/oursfeatures_mcqv0/egovisibleFalse/frames.pt', 'local_data/dynpose-100k/dynpose_100k/metadata_val_withvideo_sampled1000.csv', 'baselines/logs/dynpose_pretrain/tb_embedding')
  # process_tsv('data/egoexo4d/annotations/pretraining/test_alltasks_mcqv0.csv')
  # viz_um_embedding('data/misc/retrieval_features/ours/egoexo4d_pretrain_longseq/all/bs1025_sampledur8_pose2/0.5/testdur4', 'data/egoexo4d/annotations/pretraining/test_alltasks_mcqv0.csv', 'baselines/logs/ego_embedding_viz')
  # viz_mm_embedding2('baselines/logs/egoexo4d_pretrain_longseq/oursfeatures_all/bs1024_dur4_pose2/0.5', 'data/egoexo4d/annotations/pretraining/test_alltasks_mcqv0.csv', 'baselines/logs/nymeria_pretrain/oursfeatures/', 'local_data/nymeria/eval/', 'baselines/logs/ego_embedding_viz2')
  # viz_mm_embedding2('tmp/cam_viz/oursfeatures_full/egoexo4d', 'data/egoexo4d/annotations/pretraining/test_alltasks_mcqv0.csv', 'tmp/cam_viz/oursfeatures_full/nymeria', 'local_data/nymeria/eval/', 'baselines/logs/ego_embedding_viz2')
  # viz_cm_embedding('tmp/cam_viz/', 'tmp/tb/ego_embedding_viz')

  # reformat_result('local_data/dynpose-100k/dynpose_100k/output/cmlabel_query0/gemini-2.5-pro_8frames.jsonl', 'local_data/dynpose-100k/dynpose_100k/metadata_val_withvideo_sampled1000.csv')
  # check_embedding_label_alignment('baselines/logs/dynpose_pretrain/oursfeatures_mcqv0/egovisibleFalse/frames.pt', 'local_data/dynpose-100k/dynpose_100k/output/cmlabel_query0/gemini-2.5-pro_8frames.jsonl')
  # read_ds_result('/home/sherryxue_google_com/data/logs/egoexo4d_prof_4label/Dance/dur16')
  # read_ds_result('/home/sherryxue_google_com/data/logs/egoexo4d_scenario_8label/pose5_sr4')
  # read_ds_result('/home/sherryxue_google_com/data/logs/ucf101_101label')
  # read_ds_result('/home/sherryxue_google_com/data/logs/egoexo4d_scenario_8label/bs128_new')
  # read_ds_result('/home/sherryxue_google_com/data/logs/egoexo4d_scenario_subset_8label')
  # read_ds_result('data/logs/finegym_4label')

  # get_confusion_matrix('logs/predictions/init_best-epoch=94-val_acc=0.8864.json', show_ratio=True)
  # check_megasam_results()
