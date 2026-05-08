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

import json
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from utils.dataset_utils import extract_video_frames

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'sans-serif',  # or "sans-serif"
    'font.serif': ['Helvetica'],  # or ["Times New Roman"]
    'font.size': 15,  # 18
})


def plot_dynpose_retrieval():
  values = [20.0, 39.2, 46.3]
  labels = ['Random', 'DynPose-100K', 'Vipe']
  colors = ['#d9d9d9', '#F0F0FE', '#F0F0FE']
  edgecolor = '#636363'

  plt.figure(figsize=(6, 6))
  bars = plt.bar(
      labels, values, color=colors, edgecolor=edgecolor, width=0.6, alpha=0.8
  )
  plt.xticks(rotation=30)
  # for bar, value in zip(bars, values):
  #     plt.text(
  #         bar.get_x() + bar.get_width() / 2,
  #         bar.get_height(),
  #         f"{value:.1f}",
  #         ha='center', va='bottom'
  #     )

  plt.xlim(-0.6, 2.6)
  # plt.ylabel("Acc. (%)")

  ax = plt.gca()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  plt.tight_layout()
  plt.savefig('tmp/dynpose.png', dpi=300)


def plt_egoexo4d_pretrain_dur_hs_results(ego_visible):
  dur = [2, 4, 6, 8, 16, 32]
  if ego_visible:
    egovlp_m2t = 54.7
    egovlp_t2m = 60.8
    atomic_m2t = 39.8
    atomic_t2m = 37.5

    longcontext_m2t = [45.2, 46.1, 43.7, 42.9, 40.5, 36.5]
    longcontext_m2t_combined = [48.8, 51.0, 49.0, 49.0, 46.4, 46.0]

    longcontext_t2m = [37.5, 41.5, 39.4, 37.0, 36.4, 39.8]
    longcontext_t2m_combined = [44.0, 46.1, 43.0, 44.4, 42.8, 44.9]

  else:
    egovlp_m2t = 26.9
    egovlp_t2m = 30.3
    atomic_m2t = 40.4
    atomic_t2m = 45.5

    longcontext_m2t = [44.3, 42.9, 39.8, 38.1, 35.9, 33.7]
    longcontext_m2t_combined = [50.8, 49.4, 46.9, 46.4, 45.0, 42.9]

    longcontext_t2m = [44.6, 45.9, 43.9, 39.9, 38.6, 39.0]
    longcontext_t2m_combined = [54.8, 53.8, 55.2, 52.0, 52.2, 49.4]

  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  colors = ['#016c59', '#67a9cf', '#bdc9e1', '#fc8d59']

  # m2t subplot
  ax = axes[0]
  ax.plot(
      dur, longcontext_m2t, marker='o', label='LongContext', color=colors[0]
  )
  ax.plot(
      dur,
      longcontext_m2t_combined,
      marker='o',
      label='LongContext+Atomic',
      color=colors[1],
  )
  ax.hlines(
      atomic_m2t,
      0,
      dur[-1],
      colors=colors[2],
      linestyles='dotted',
      label='Atomic',
  )
  ax.hlines(
      egovlp_m2t,
      0,
      dur[-1],
      colors=colors[3],
      linestyles='dotted',
      label='EgoVLPv2 video baseline',
  )
  ax.set_title(f'Motion to Text (Ego visible = {ego_visible})')
  ax.set_xlabel('Duration (s)')
  ax.set_ylabel('MCQ Acc. (%)')
  ax.set_xticks(dur)
  ax.set_xlim(0, 32)
  ax.set_ylim(20, 70)
  ax.legend()

  # t2m subplot
  ax = axes[1]
  ax.plot(
      dur, longcontext_t2m, marker='o', label='LongContext', color=colors[0]
  )
  ax.plot(
      dur,
      longcontext_t2m_combined,
      marker='o',
      label='LongContext+Atomic',
      color=colors[1],
  )
  ax.hlines(
      atomic_t2m,
      0,
      dur[-1],
      colors=colors[2],
      linestyles='dotted',
      label='Atomic',
  )
  ax.hlines(
      egovlp_t2m,
      0,
      dur[-1],
      colors=colors[3],
      linestyles='dotted',
      label='EgoVLPv2 video baseline',
  )
  ax.set_title(f'Text to Motion (Ego visible = {ego_visible})')
  ax.set_xlabel('Duration (s)')
  ax.set_xticks(dur)
  ax.set_xlim(0, 32)
  ax.set_ylim(20, 70)
  ax.legend()

  plt.tight_layout()
  plt.savefig(f'tmp/egoexo4d_pretrain_dur_hs_egovisible{ego_visible}.png')


def plt_egoexo4d_scenario_cls_results():
  dur = [1, 2, 4, 8, 16]
  val_acc_base = [50.0, 63.04, 70.45, 80.74, 86.36]
  val_acc_init = [59.57, 70.57, 80.74, 84.57, 90.43]
  test_acc_base = [51.58, 59.24, 68.62, 73.94, 79.89]
  test_acc_init = [55.91, 66.64, 74.30, 80.70, 84.67]

  # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  colors = ['#016c59', '#67a9cf', '#bdc9e1', '#fc8d59']
  fig, axes = plt.subplots(1, 1, figsize=(6, 4))
  ax = axes if isinstance(axes, plt.Axes) else axes[0]
  # ax.plot(dur, val_acc_base, marker='o', label='Train from scratch', color='#bcbddc')
  # ax.plot(dur, val_acc_init, marker='o', label='Initialize from our ckpt', color='#756bb1')
  # ax.set_title('Scenario Cls. Val Acc. (%)')
  # ax.set_xlabel('Test sample duration (s)')
  # ax.set_xticks(dur)
  # ax.set_xlim(0, 16)
  # ax.set_ylim(40, 100)
  # ax.legend()

  # ax = axes[1]
  ax.plot(
      dur,
      test_acc_base,
      marker='o',
      label='Train from scratch',
      color='#cccccc',
  )
  ax.plot(
      dur,
      test_acc_init,
      marker='o',
      label='Initialize from CamFormer',
      color='#636363',
  )
  # ax.set_title('Scenario Cls. Test Acc. (%)')
  ax.set_xlabel('(seconds)', labelpad=10)
  ax.set_ylabel('Accuracy (\%)', labelpad=10)
  ax.set_xticks(dur)
  ax.set_xlim(0, 18)
  ax.set_ylim(40, 90)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.7)
  ax.legend()

  plt.tight_layout()
  plt.savefig(f'tmp/egoexo4d_scenario_cls_results.png', dpi=300)
  # plt.savefig(f'tmp/egoexo4d_scenario_cls_plot.pdf', dpi=300)


def plt_finegym_cls_results():
  dur = [20, 40, 60, 80, 100]
  val_acc_base = [55.68, 60.65, 63.35, 64.14, 64.62]
  val_acc_init = [59.43, 65.37, 67.80, 68.59, 69.94]

  colors = ['#016c59', '#67a9cf', '#bdc9e1', '#fc8d59']
  fig, axes = plt.subplots(1, 1, figsize=(6, 4))
  ax = axes if isinstance(axes, plt.Axes) else axes[0]
  ax.plot(
      dur, val_acc_base, marker='o', label='Train from scratch', color='#cccccc'
  )
  ax.plot(
      dur,
      val_acc_init,
      marker='o',
      label='Initialize from CamFormer',
      color='#636363',
  )
  ax.set_xlabel('(% of event segment)', labelpad=10)
  # ax.set_ylabel('Accuracy (\%)', labelpad=10)
  ax.set_xticks(dur)
  ax.set_ylim(50, 75)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.legend(loc='lower right')
  ax.grid(axis='y', linestyle='--', alpha=0.7)

  plt.tight_layout()
  plt.savefig(f'tmp/finegym_cls_results.png', dpi=300)
  # plt.savefig(f'tmp/finegym_cls_results.pdf', dpi=300)


def plt_egoexo4d_keystep_context_results():
  context_window = [0, 100, 200, 300, 400]
  x_label = ['atomic', '100', '200', '300', '400']
  acc = [14.66, 14.74, 15.50, 15.34, 15.25]
  fig, ax = plt.subplots(figsize=(6, 4))

  ax.plot(context_window, acc, marker='o', color='#636363')
  # ax.set_title('Val Acc. (%)')
  ax.set_xlabel('(% of keystep segment)', labelpad=10)
  ax.set_xticks(context_window)
  ax.set_xticklabels(x_label)
  ax.set_ylim(14, 16)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout()
  plt.savefig(f'tmp/egoexo4d_keystep_context_results.png', dpi=300)


def plt_egoexo4d_text_retrieval_results():
  dur = [0, 2, 4, 6, 8]
  x_label = ['atomic', '2', '4', '6', '8']
  acc = [38.71, 43.04, 44.12, 43.72, 42.69]
  fig, ax = plt.subplots(figsize=(6, 4))
  ax.plot(dur, acc, marker='o', color='#636363')
  ax.set_xlabel('(seconds)', labelpad=10)
  ax.set_xticks(dur)
  ax.set_xticklabels(x_label)
  ax.set_ylim(36, 46)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout()
  plt.savefig(f'tmp/egoexo4d_text_retrieval_results.png', dpi=300)


def plt_dynpose_scenario(draw_two_methods=False):
  labels = [
      'day/night',
      'animal?',
      'text?',
      'urban/rural',
      'male/female',
      'food?',
      'sports?',
      'indoor/outdoor',
      '> 1 people?',
      'walking?',
  ]
  dynpose = [
      50.00,
      56.67,
      56.50,
      53.33,
      56.17,
      61.67,
      59.67,
      64.67,
      61.33,
      73.83,
  ]
  vipe = [53.12, 55.50, 55.67, 56.50, 57.83, 62.67, 65.83, 67.00, 68.17, 80.33]

  # y = np.arange(len(labels))  # y positions
  spacing = 80
  y = np.arange(0, len(labels) * spacing, spacing)

  fig, ax = plt.subplots(figsize=(12, 8))

  if draw_two_methods:
    # Plot side-by-side bars
    h = 0.6  # bar thickness
    bars1 = ax.barh(
        y - h / 2,
        dynpose,
        height=h,
        color='#E5F3EA',  #'#E8F5F9',
        edgecolor='#bdbdbd',
        label='DynPose',
        alpha=0.8,
    )
    bars2 = ax.barh(
        y + h / 2,
        vipe,
        height=h,
        color='#A7C1CD',  #'#A6D6C9',
        edgecolor='#bdbdbd',
        label='ViPE',
        alpha=0.5,
    )

    # Add values
    for bars in [bars1, bars2]:
      for bar in bars:
        width = bar.get_width()
        ax.text(
            width - 2.5,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.1f}',
            va='center',
            fontsize=12,
        )

    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

  else:
    # Only plot ViPE
    bars = ax.barh(
        y,
        vipe,
        height=55,
        color='#E8F5F9',  #'#f0f0f0'
        edgecolor='#bdbdbd',
        alpha=0.8,
        label='ViPE',
        linewidth=0.1,
    )

    for bar in bars:
      width = bar.get_width()
      ax.text(
          width - 2,
          bar.get_y() + bar.get_height() / 2,
          f'{width:.1f}',
          va='center',
      )

    ax.set_yticks([])  # y
    # ax.set_yticks(y)
    ax.set_xlim(50, 87)
    # ax.set_yticklabels(labels)

  ax.set_xlabel('Accuracy (%)', labelpad=10)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.invert_yaxis()

  plt.tight_layout()
  plt.savefig('tmp/dynpose_scenario.pdf', dpi=300)


def plt_egoexo4d_confusion_matrix():
  data_file = 'data/misc/predictions/init16s_best-epoch=167-val_acc=0.9043.json'
  with open(data_file, 'r') as f:
    data = json.load(f)

  # Extract ground truth and predicted labels
  y_true = []
  y_pred = []

  for key, entry in data.items():
    y_true.append(entry['gt_label'])
    y_pred.append(entry['top1_pred']['label'])

  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  # Original label order
  orig_labels = [
      'Health',
      'Dance',
      'Basketball',
      'Bouldering',
      'Cooking',
      'Soccer',
      'Music',
      'Bike Repair',
  ]

  # Your desired order
  new_order = [
      'Bike Repair',
      'Health',
      'Cooking',
      'Music',
      'Soccer',
      'Dance',
      'Basketball',
      'Bouldering',
  ]

  # Compute confusion matrix (in original order)
  cm = confusion_matrix(y_true, y_pred, normalize='true') * 100

  # Reorder rows and columns
  reorder_idx = [orig_labels.index(lbl) for lbl in new_order]
  cm = cm[np.ix_(reorder_idx, reorder_idx)]

  # Plot
  plt.figure(figsize=(8, 6))
  cmap = sns.cubehelix_palette(
      start=0.5, rot=-0.5, light=0.9, dark=0.5, as_cmap=True
  )
  sns.heatmap(
      cm,
      annot=True,
      fmt='.1f',
      cmap=cmap,
      xticklabels=new_order,
      yticklabels=new_order,
      alpha=0.7,
  )
  plt.xlabel('Predicted Label', labelpad=20)
  plt.ylabel('True Label', labelpad=20)
  plt.tight_layout()
  plt.savefig('tmp/confusion_matrix_egoexo4d.pdf', dpi=300)
  plt.close()


def plot_finegym_confusion_matrix():
  data_file = (
      'logs/predictions/init_sampledur_best-epoch=177-val_acc=0.6823.json'
  )
  with open(data_file, 'r') as f:
    data = json.load(f)

  # Extract ground truth and predicted labels
  y_true = []
  y_pred = []

  for key, entry in data.items():
    y_true.append(entry['gt_label'])
    y_pred.append(entry['top1_pred']['label'])

  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  labels = ['Vault', 'Floor Exercise', 'Balance Beam', 'Uneven Bars']

  # Compute confusion matrix
  cm = confusion_matrix(y_true, y_pred, normalize='true') * 100

  plt.figure(figsize=(8, 6))
  cmap = sns.cubehelix_palette(
      start=0.5, rot=-0.5, light=0.9, dark=0.5, as_cmap=True
  )
  sns.heatmap(
      cm,
      annot=True,
      fmt='.1f',
      cmap=cmap,
      xticklabels=labels,
      yticklabels=labels,
      alpha=0.7,
  )
  plt.xlabel('Predicted Label', labelpad=20)
  plt.ylabel('True Label', labelpad=20)
  plt.tight_layout()
  plt.savefig('tmp/confusion_matrix_finegym.pdf', dpi=300)
  plt.close()


def _center_zoom_and_resize(frames, out_h, out_w, zoom_frac=0.6):
  """frames: (N, H, W, 3), zoom into central zoom_frac region, then resize to (out_h, out_w)."""
  zoom_frac = float(np.clip(zoom_frac, 1e-6, 1.0))  # avoid 0 or negative
  N, H, W, C = frames.shape
  crop_h = max(1, int(round(H * zoom_frac)))
  crop_w = max(1, int(round(W * zoom_frac)))
  y0 = (H - crop_h) // 2
  x0 = (W - crop_w) // 2
  cropped = frames[
      :, y0 : y0 + crop_h, x0 : x0 + crop_w
  ]  # (N, crop_h, crop_w, 3)

  # resize back to (out_h, out_w) without matplotlib (use PIL)
  try:
    from PIL import Image

    resized = []
    for i in range(N):
      img = Image.fromarray(cropped[i].astype(np.uint8))
      img = img.resize((out_w, out_h), resample=Image.BILINEAR)
      resized.append(np.asarray(img))
    return np.stack(resized, axis=0)  # (N, out_h, out_w, 3)
  except Exception:
    # fallback to numpy-only nearest neighbor (simple and dependency-free)
    yy = (np.linspace(0, crop_h - 1, out_h)).astype(int)
    xx = (np.linspace(0, crop_w - 1, out_w)).astype(int)
    return cropped[:, yy][:, :, xx]


def plot_from_video(video_path, save_path=None, zoom_frac=0.6):
  frames = extract_video_frames(video_path, 4)  # (4, H, W, 3)
  H, W = frames.shape[1], frames.shape[2]

  # split halves
  left_frames = frames[:, :, : W // 2]  # (4, H, W/2, 3)
  right_frames = frames[:, :, W // 2 :]  # (4, H, W/2, 3)

  print(frames.shape, left_frames.shape, right_frames.shape)

  # zoom into central x% on the right half, then resize back to (H, W/2)
  right_frames_zoomed = _center_zoom_and_resize(
      right_frames, out_h=H, out_w=W // 2, zoom_frac=zoom_frac
  )

  # build grid
  top_row = np.concatenate(
      list(left_frames.astype(np.uint8)), axis=1
  )  # (H, 4*W/2, 3)
  bottom_row = np.concatenate(
      list(right_frames_zoomed.astype(np.uint8)), axis=1
  )  # (H, 4*W/2, 3)
  grid_image = np.concatenate([top_row, bottom_row], axis=0)  # (2H, 4*W/2, 3)

  # save
  if save_path is None:
    save_path = video_path.replace('.mp4', '_grid.pdf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.imsave(save_path, grid_image.astype(np.uint8))
  print(f'Saved visualization to {save_path}')


def plot_from_video_singlecolumn(video_path, save_path=None):
  frames = extract_video_frames(video_path, 7)  # (4, H, W, 3)
  H, W = frames.shape[1], frames.shape[2]

  # build grid
  grid_image = np.concatenate(
      list(frames.astype(np.uint8)), axis=1
  )  # (H, 4*W, 3)

  # save
  if save_path is None:
    save_path = video_path.replace('.mp4', '_grid.pdf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
  img = Image.fromarray(grid_image)
  img.save(save_path)
  print(f'Saved visualization to {save_path}')


def plt_result_overview_difference():
  """Generates a chart where the smaller bar is plotted on top of the

  larger bar, clearly showing the difference.
  """

  categories = [
      'Text Retrieval (Ego-Exo4D)*',
      'Text Retrieval (Nymeria)*',
      'Text Retrieval (DynPose-100k)',
      'Proficiency Estimation (Ego-Exo4D)',
      'Keystep Recognition (Ego-Exo4D)*',
      'Keystep Localization (Ego-Exo4D)*',
      'Activity Classification (Ego-Exo4D)',
      'Scene Arribute Classification (DynPose-100K)$\diamond$',
      'Event Classification (FineGym)$\diamond$',
      'Action Recognition (UCF101-Dynamic)$\diamond$',
  ]
  baseline_scores = [
      38.40,
      29.10,
      33.10,
      62.64,
      29.17,
      31.81,
      79.89,
      50.0,
      64.62,
      64.18,
  ]
  ours_scores = [
      45.30,
      35.10,
      46.30,
      68.07,
      32.37,
      34.68,
      84.67,
      62.26,
      69.94,
      68.16,
  ]
  y_positions = np.arange(len(categories))
  bar_height = 0.6

  fig, ax = plt.subplots(figsize=(11, 7))

  # Plot bars category by category
  for i in range(len(categories)):
    baseline_val = baseline_scores[i]
    ours_val = ours_scores[i]
    gain = ours_val - baseline_val
    ax.barh(y_positions[i], ours_val, bar_height, color='#7fcdbb')
    ax.barh(y_positions[i], baseline_val, bar_height, color='#E8F5F9')
    ax.text(
        ours_val + 1, y_positions[i], f'+{gain:.1f}', va='center', ha='left'
    )

  ax.set_xlim(0, 100)
  ax.set_yticks(y_positions)
  ax.set_yticklabels(categories)
  ax.invert_yaxis()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  filename = './tmp/result_overview_difference.pdf'
  plt.tight_layout()
  plt.savefig(filename, dpi=300)
  plt.close(fig)


def plt_egoexo4d_scenario_acc():
  categories = [
      'Bouldering',
      'Basketball',
      'Dance',
      'Soccer',
      'Music',
      'Cooking',
      'Health',
      'Bike Repair',
  ]
  ours = [97.8, 94.6, 87.8, 85.9, 81.7, 70.3, 64.8, 54.8]
  categories = categories[::-1]
  ours = ours[::-1]

  categories = [
      'day/night',
      'animal?',
      'text?',
      'urban/rural',
      'male/female',
      'food?',
      'sports?',
      'indoor/outdoor',
      '> 1 people?',
      'walking?',
  ]
  ours = [53.12, 55.50, 55.67, 56.50, 57.83, 62.67, 65.83, 67.00, 68.17, 80.33]

  y_positions = np.arange(len(categories))
  bar_height = 0.6

  fig, ax = plt.subplots(figsize=(10, 8))

  # Plot bars category by category
  for i in range(len(categories)):
    ours_val = ours[i]
    ax.barh(y_positions[i], ours_val, bar_height, color='#d9d9d9', alpha=0.5)
    ax.text(
        ours_val + 1, y_positions[i], f'{ours_val:.1f}', va='center', ha='left'
    )
  ax.set_xlabel('Per-class Accuracy (%)', labelpad=10)
  # ax.set_xlim(30, 100)
  ax.set_xlim(40, 100)
  ax.set_yticks(y_positions)
  ax.set_yticklabels(categories)
  ax.invert_yaxis()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  # filename = './tmp/egoexo4d_scenario_acc.pdf'
  filename = './tmp/dynpose_scenario_acc.pdf'
  plt.tight_layout()
  plt.savefig(filename, dpi=300)
  plt.close(fig)


def plot_slides_numbers():
  # data = [64.62, 69.94]
  data = [61.69, 64.18]
  fig, ax = plt.subplots(figsize=(4, 4))
  colors = ['#EFF7FA', '#EFF7FA']
  ax.bar(
      [0.1, 1.0],
      data,
      color=colors,
      width=0.4,
      edgecolor='#bdbdbd',
      linewidth=0.1,
  )
  ax.set_xlim(-0.4, 1.6)
  # ax.set_ylim(60, 75)
  ax.set_ylim(60, 66)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.set_xticks([0.1, 1.0])
  ax.set_xticklabels(
      []
  )  # (['Train from scratch', 'Initialize from CamFormer'])
  ax.set_yticklabels([])
  plt.savefig('tmp/slides_tmp.pdf', dpi=300)


def plot_proficiency():
  labels = ['TimeSformer', 'CamFormer\n(No Pretrain)', 'CamFormer\n(Pretrain)']
  bouldering_scores = [
      60.0,
      64.6,
      69.9,
  ]  # [29.17, 14.07, 32.37] #[55.35, 63.52, 65.41]
  dancing_scores = [
      60.0,
      64.18,
      68.16,
  ]  # [31.81, 21.29, 34.68] #[69.92, 66.67, 70.73]

  # Setup the plot framework (1 row, 2 columns)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  # fig.suptitle('Performance Comparison by Method', fontsize=16, fontweight='bold', y=1.05)

  # Color palette: Gray for TimeSformer, Blues for CamFormer variants
  colors = ['#d9d9d9', '#EAF4EC', '#78c679']

  # --- Plot 1: Bouldering ---
  bars1 = ax1.bar(labels, bouldering_scores, color=colors, width=0.5)
  # ax1.set_title('Bouldering', fontsize=14, pad=15)
  # ax1.set_ylim(50, 75)
  # ax1.set_ylim(10, 40)
  ax1.set_ylim(60, 72)
  # ax1.set_ylabel('Accuracy / Score', fontsize=12)
  ax1.grid(axis='y', linestyle='--', alpha=0.5)
  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)
  ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
  # Add value labels
  for bar in bars1:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.5,
        f'{height:.2f}',
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold',
    )

  # --- Plot 2: Dancing ---
  bars2 = ax2.bar(labels, dancing_scores, color=colors, width=0.5)
  # ax2.set_title('Dancing', fontsize=14, pad=15)
  # ax2.set_ylim(60, 75)
  # ax2.set_ylim(15, 40)
  ax2.set_ylim(60, 72)
  ax2.grid(axis='y', linestyle='--', alpha=0.5)
  ax2.spines['top'].set_visible(False)
  ax2.spines['right'].set_visible(False)
  ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

  # Add value labels
  for bar in bars2:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.5,
        f'{height:.2f}',
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold',
    )

  legend_labels = [l.replace('\n', ' ') for l in labels]
  patches = [
      mpatches.Patch(color=colors[i], label=legend_labels[i])
      for i in range(len(labels))
  ]
  fig.legend(
      handles=patches,
      loc='lower center',
      bbox_to_anchor=(0.5, -0.1),
      ncol=3,
      fontsize=12,
      frameon=False,
  )

  plt.tight_layout()
  plt.savefig('tmp/proficiency_comparison.pdf', dpi=300)


if __name__ == '__main__':
  # plt_egoexo4d_scenario_acc()
  # plt_result_overview_difference()
  # plt_egoexo4d_confusion_matrix()
  # plt_egoexo4d_pretrain_dur_hs_results(True)
  # plt_egoexo4d_pretrain_dur_hs_results(False)
  # plt_egoexo4d_scenario_cls_results()
  # plt_finegym_cls_results()
  # plt_egoexo4d_keystep_context_results()
  # plt_egoexo4d_text_retrieval_results()
  # plt_dynpose_scenario(False)
  # plot_finegym_confusion_matrix()
  # plot_from_video_singlecolumn('viz/dynpose/success_1example/420_gt_A_man_in_a_suit_walks_down_a_long_hallway_in_a_building._camera.mp4')
  # plot_from_video('viz/nymeria/val_a/390_C slightly bends forward while doing lunges .mp4', zoom_frac=1.0)
  # plot_from_video('viz/egoexo4d_action/36_Stir_the_tea/3_iiith_cooking_43_1.mp4', zoom_frac=1.0)
  # plot_from_video_singlecolumn('viz/egoexo4d_action/68_Tap_patient_to_confirm_consciousness/13_nus_cpr_07_1_camera.mp4')
  # plot_slides_numbers()
  plot_proficiency()
