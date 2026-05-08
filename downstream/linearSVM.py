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
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


def run_svm(train_feat, val_feat, train_labels, val_labels):
  # Ensure numpy arrays and correct shapes
  Xtr = np.asarray(train_feat)
  Xva = np.asarray(val_feat)
  ytr = np.asarray(train_labels).ravel()
  yva = np.asarray(val_labels).ravel()

  # Quick diagnostics
  assert (
      Xtr.ndim == 2 and Xva.ndim == 2
  ), 'Features must be 2D (n_samples, n_features).'
  assert ytr.ndim == 1 and yva.ndim == 1, 'Labels must be 1D.'
  if not np.isfinite(Xtr).all() or not np.isfinite(Xva).all():
    raise ValueError('NaN/inf in features. Clean or impute first.')
  if not np.isfinite(ytr).all() or not np.isfinite(yva).all():
    raise ValueError('NaN/inf in labels.')

  scaler = StandardScaler()
  train_feat_scaled = scaler.fit_transform(train_feat)
  val_feat_scaled = scaler.transform(val_feat)

  # Train a linear SVM
  svm_classifier = SVC(
      kernel='linear', C=1.0, random_state=42
  )  # class_weight='balanced')
  svm_classifier.fit(train_feat_scaled, train_labels)

  # Evaluate on validation set
  y_pred = svm_classifier.predict(val_feat_scaled)
  # train acc
  train_pred = svm_classifier.predict(train_feat_scaled)
  train_acc = accuracy_score(train_labels, train_pred)

  accuracy = accuracy_score(val_labels, y_pred)
  # f1 = f1_score(val_labels, y_pred)
  print(f'\n--- Results ---')
  # print(f"Training Accuracy: {train_acc * 100:.2f}%")
  print(f'Validation Accuracy: {accuracy * 100:.2f}%')
  # print(f"F1 Score: {f1:.3f}")


def run_linear_svc(
    train_feat,
    val_feat,
    train_labels,
    val_labels,
    C=1.0,
    balanced=True,
    tol=1e-3,
    max_iter=5000,
):
  # Arrays + light cast
  Xtr = np.asarray(train_feat, dtype=np.float32)
  Xva = np.asarray(val_feat, dtype=np.float32)
  ytr = np.asarray(train_labels).ravel()
  yva = np.asarray(val_labels).ravel()

  # Scale with train stats
  scaler = StandardScaler()
  Xtr = scaler.fit_transform(Xtr)
  Xva = scaler.transform(Xva)

  # Deterministic linear SVM (liblinear)
  clf = LinearSVC(
      C=C,
      dual=False,  # n_samples > n_features
      class_weight='balanced' if balanced else None,
      tol=tol,
      max_iter=max_iter,
  )
  clf.fit(Xtr, ytr)

  # Eval
  y_pred = clf.predict(Xva)
  acc = accuracy_score(yva, y_pred)
  print(f'\n--- Results ---\nValidation Accuracy: {acc*100:.2f}%')


def replace_nan_with_zero(arr: np.ndarray) -> np.ndarray:
  """Replace NaN and Inf values in a numpy array with 0,

  printing a warning if replacements are made.

  Args:
      arr (np.ndarray): Input array.

  Returns:
      np.ndarray: Cleaned array with NaNs/Infs replaced by 0.
  """
  nan_count = np.isnan(arr).sum()
  posinf_count = np.isposinf(arr).sum()
  neginf_count = np.isneginf(arr).sum()
  total_count = nan_count + posinf_count + neginf_count

  if total_count > 0:
    print(
        f'Warning: found {nan_count} NaNs, {posinf_count} +Infs, {neginf_count}'
        ' -Infs. Replacing with 0.'
    )

  return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def main(data_file, feature_dir):
  print(f'Reading features from {feature_dir}')
  df = pd.read_csv(data_file)
  train_feat, val_feat = [], []
  train_labels, val_labels = [], []
  for i, row in df.iterrows():
    feat_path = os.path.join(feature_dir, str(i) + '.npy')
    if not os.path.exists(feat_path):
      # print('error', i)
      continue
    feat = np.load(feat_path)
    if feat.size == 0:
      # print('error', i)
      continue
    feat = feat.mean(axis=0)
    if row['split'] == 'train':
      train_feat.append(feat)
      train_labels.append(row['label'])
    else:
      val_feat.append(feat)
      val_labels.append(row['label'])

  train_feat = np.vstack(train_feat)
  val_feat = np.vstack(val_feat)
  train_labels = np.array(train_labels)
  val_labels = np.array(val_labels)

  train_feat = replace_nan_with_zero(train_feat)
  val_feat = replace_nan_with_zero(val_feat)

  print(train_feat.shape, val_feat.shape, train_labels.shape, val_labels.shape)
  run_svm(train_feat, val_feat, train_labels, val_labels)


def main_combine_feat(data_file, feature_dir1, feature_dir2, mode='c'):
  print(f'Reading combined features from {feature_dir1} and {feature_dir2}')
  df = pd.read_csv(data_file)
  train_feat, val_feat = [], []
  train_labels, val_labels = [], []
  for i, row in df.iterrows():
    feat_path = os.path.join(feature_dir1, str(i) + '.npy')
    feat = np.load(feat_path)
    if feat.size == 0:
      # print('error', i)
      continue
    feat1 = feat.mean(axis=0)
    feat_path = os.path.join(feature_dir2, str(i) + '.npy')
    feat2 = np.load(feat_path).mean(axis=0)
    if mode == 'a':
      feat = feat1
    elif mode == 'b':
      feat = feat2
    else:
      feat = np.concatenate([feat1, feat2])
    if row['split'] == 'train':
      train_feat.append(feat)
      train_labels.append(row['label'])
    else:
      val_feat.append(feat)
      val_labels.append(row['label'])

  train_feat = np.vstack(train_feat)
  val_feat = np.vstack(val_feat)
  train_labels = np.array(train_labels)
  val_labels = np.array(val_labels)

  # majority label baseline
  # labels, counts = np.unique(val_labels, return_counts=True)
  # label_range = (labels.min(), labels.max())
  # majority_idx = np.argmax(counts)
  # majority_label = labels[majority_idx]
  # majority_count = counts[majority_idx]
  # print("Label range:", label_range)
  # print("Majority label:", majority_label, "with count:", majority_count)

  train_feat = replace_nan_with_zero(train_feat)
  val_feat = replace_nan_with_zero(val_feat)

  print(train_feat.shape, val_feat.shape, train_labels.shape, val_labels.shape)
  run_svm(train_feat, val_feat, train_labels, val_labels)


def read_combined_result():
  data_file = 'data/egoexo4d/annotations/downstream/action_cls.csv'
  feature_dir1 = 'local_data/egoexo4d/action_cls_features/egovlpv2_fps1.87'
  feature_dir2 = 'local_data/misc/egoexo4d_action_features/bs1025_sampledur16_pose2/gt_context0.75/20fps_w80_s10'
  main_combine_feat(data_file, feature_dir1, feature_dir2)


def read_combined_result_subset():
  # data_file = 'data/egoexo4d/annotations/downstream/action_cls_with_all3_pred.csv' # with_megasam_pred.csv
  data_file = 'final_data/data_files/egoexo4d/action_cls_with_all3_pred.csv'
  # feature_dir = 'local_data/misc/egoexo4d_action_features/cls_subset'
  feature_dir = 'final_data/egoexo4d_action_features/cls_subset'

  feature_dir1 = f'{feature_dir}/egovlpv2_fps1.87'
  feature_dir2 = f'{feature_dir}/bs1024_dur4_pose2_sr4_new/'
  for mode in ['b', 'c']:
    for method in ['megasam', 'pi3', 'vipe', 'gt']:
      print(
          '-' * 10, f'mode={mode}, method={method} (transform False)', '-' * 10
      )
      main_combine_feat(
          data_file,
          feature_dir1,
          f'{feature_dir2}/{method}_transformFalse/20fps_w20_s2',
          mode=mode,
      )
      # if method == 'gt':
      #     continue
      # print('-' * 10, f'mode={mode}, method={method} (transform True)', '-' * 10)
      # main_combine_feat(data_file, feature_dir1, f"{feature_dir2}/{method}_transformTrue/20fps_w20_s2", mode=mode)


def read_combined_result_subset_new():
  data_file = (  # with_megasam_pred.csv
      'data/egoexo4d/annotations/downstream/action_cls_with_all3_pred.csv'
  )
  feature_dir1 = (
      'local_data/misc/egoexo4d_action_features/cls_subset/egovlpv2_fps1.87'
  )
  feature_dir2 = 'local_data/misc/egoexo4d_action_features/cls_subset/'
  for mode in ['b', 'c']:
    for method in [
        'megasam',
        'pi3',
        'vipe',
        'gt_gravityTrue',
        'gt_gravityFalse',
    ]:
      print('-' * 10, f'mode={mode}, method={method}', '-' * 10)
      # main(data_file, f"{feature_dir2}/{method}_transformFalse_encodepose11_20fps_w80_s10/bs1024_sampleddur8_pose11")
      main_combine_feat(
          data_file,
          feature_dir1,
          f'{feature_dir2}/{method}_transformFalse_encodepose11_20fps_w80_s10/bs1024_sampleddur8_pose11',
          mode=mode,
      )
      # main_combine_feat(data_file, feature_dir1, f"{feature_dir2}/{method}_gravityFalse_transformFalse_encodepose2_20fps_w80_s10/bs1025_sampledur8_pose2", mode=mode)


def read_windowlen_result():
  data_file = 'data/egoexo4d/annotations/downstream/action_cls.csv'
  feature_dir = 'data/misc/egoexo4d_action_features/gt/'
  # for stride in [10, 20, 40, 80]:
  #     print('-' * 10, f'stride={stride}', '-' * 10)
  #     main(data_file, f"{feature_dir}/20fps_w80_s{stride}")
  for window in [40, 80, 120, 160, 320]:
    print('-' * 10, f'window={window}', '-' * 10)
    main(data_file, f'{feature_dir}/20fps_w{window}_s10')


def read_camest_result():
  data_file = (
      'data/egoexo4d/annotations/downstream/action_cls_with_megasam_pred.csv'
  )
  feature_dir = 'local_data/misc/egoexo4d_action_features/cls_subset/bs1024_dur4_pose2_sr4_new'
  for m in ['vipe', 'pi3', 'megasam', 'gt']:
    print('-' * 10, m, '-' * 10)
    main(data_file, f'{feature_dir}/{m}_scaleFalse/20fps_w80_s10_new')


def read_contextlen_result():
  data_file = 'data/egoexo4d/annotations/downstream/action_cls.csv'
  feature_dir = (
      'local_data/misc/egoexo4d_action_features/bs1025_sampledur8_pose2/'
  )
  for context_rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
    print('-' * 10, context_rate, '-' * 10)
    main(data_file, f'{feature_dir}/gt_context{context_rate}/20fps_w80_s10')


def read_contextlen_result_new():
  data_file = 'data/egoexo4d/annotations/downstream/action_cls.csv'
  base_feature_dir = 'local_data/misc/egoexo4d_action_features/cls/bs1025_sampledur8_pose2/gt_context0.0/20fps_w80_s10_new'
  feature_dir = 'local_data/misc/egoexo4d_action_features/cls_context/bs1025_sampledur8_pose2'
  main(data_file, base_feature_dir)
  for context_rate in [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0]:
    print('-' * 10, context_rate, '-' * 10)
    main_combine_feat(
        data_file,
        base_feature_dir,
        f'{feature_dir}/gt_context{context_rate}/maxdur8',
    )


def read_sample_rate_result():
  data_file = 'data/egoexo4d/annotations/downstream/action_cls.csv'
  feature_dir = 'local_data/misc/egoexo4d_action_features/cls_context/gt_context0.0_maxdur8/'
  for sr in [10, 20]:
    print('-' * 10, sr, '-' * 10)
    main(data_file, feature_dir + f'bs1024_sampledur8_pose5_sr{sr}')


if __name__ == '__main__':
  read_combined_result_subset()
  # read_sample_rate_result()
  # read_combined_result_subset_new()
  # read_contextlen_result_new()
  # read_windowlen_result()
  # read_camest_result()
