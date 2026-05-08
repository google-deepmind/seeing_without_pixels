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
import os
from datasets.action_dataset import FineGymVideoAndCamPoseSeq
from datasets.egoexo4d import EgoExo4DVideoAndCameraPoseSeqForActionCounting
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from utils.visualize import plot_camera

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'sans-serif',  # or "sans-serif"
    'font.serif': ['Helvetica'],  # or ["Times New Roman"]
    'font.size': 18,
})


def normalize(data_array):
  min_val = np.min(data_array)
  max_val = np.max(data_array)
  range_val = max_val - min_val
  if range_val == 0:
    return np.zeros(data_array.shape)
  normalized_array = (data_array - min_val) / range_val
  return normalized_array


def viz_one_sim_matrix(feature_path):
  sim_matrix_np = np.load(feature_path)  # [t, t]
  print(sim_matrix_np.shape)
  plt.figure(figsize=(8, 8))
  cmap = plt.cm.Purples_r
  new_cmap = mcolors.LinearSegmentedColormap.from_list(
      'lighter_purples', cmap(np.linspace(0.3, 1.0, 256))
  )
  print(sim_matrix_np.shape)
  if '824' in feature_path:
    sim_matrix_np = sim_matrix_np[90:162, 90:162]
  if '812' in feature_path:
    sim_matrix_np = sim_matrix_np[70:145, 70:145]
  fps = 5 if 'finegym' in feature_path else 20
  duration = sim_matrix_np.shape[0] / fps
  sim_matrix_np = normalize(sim_matrix_np)
  plt.imshow(
      sim_matrix_np,
      cmap='viridis',
      extent=[0, duration, duration, 0],
      vmin=0,
      vmax=1,
  )  #'viridis'
  print(sim_matrix_np.min(), sim_matrix_np.max())
  # plt.colorbar()
  # plt.title(f'Self-similarity Map - Sample {i} (Length: {actual_length})')
  plt.xlabel('Time (seconds)', labelpad=10)
  plt.ylabel('Time (seconds)', labelpad=10)
  # Save figure
  save_path = feature_path.replace('.npy', '.pdf')
  plt.savefig(save_path, dpi=300)
  plt.close()


def viz_sim_matrix(feature_path, save_path, csv_file=None):
  if csv_file is not None:
    base_idx = int(os.path.basename(feature_path).split('idx')[1][0]) * 1000
    print('Base idx', base_idx)
    df = pd.read_csv(csv_file)
  h = torch.load(feature_path)  # [batch_size, t, 128]
  fps = 5 if 'finegym' in feature_path else 20

  # Create output directory if it doesn't exist
  os.makedirs(save_path, exist_ok=True)

  # For each item in the batch

  for i in range(h.shape[0]):
    if i not in [163]:  # [163, 371, 824, 981]
      continue
    # Get features for this item
    features = h[i]  # [t, 128]

    if csv_file is not None:
      label = df.iloc[base_idx + i]['event_label']
      if label != 4:
        continue

    # Find actual sequence length by checking where features become zero
    # Compute feature magnitudes
    print(features.shape)
    if i == 981:
      features = features[30:65, 30:65]
    if i == 824:
      features = features[90:162, 90:162]
    if i == 812:
      features = features[70:145, 70:145]
    feature_magnitudes = torch.norm(features, dim=1)  # [t]
    # Find where magnitudes are close to zero (considering numerical precision)
    is_padding = feature_magnitudes < 1e-6
    # Get actual sequence length
    actual_length = (
        torch.where(is_padding)[0][0]
        if torch.any(is_padding)
        else features.shape[0]
    )

    # Only use the non-padded portion
    features = features[:actual_length]  # [actual_length, 128]
    if features.shape[0] < 50:
      continue

    duration = features.shape[0] / fps
    print('Duration:', duration)
    # Compute similarity matrix
    # Normalize features
    features_norm = features / torch.norm(features, dim=1, keepdim=True)
    # Compute cosine similarity
    sim_matrix = torch.mm(
        features_norm, features_norm.t()
    )  # [actual_length, actual_length]

    # Convert to numpy for plotting
    sim_matrix_np = sim_matrix.cpu().numpy()
    sim_matrix_np = normalize(sim_matrix_np)
    print(
        sim_matrix_np.shape, duration, sim_matrix_np.min(), sim_matrix_np.max()
    )

    # Create figure
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.Purples_r
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'lighter_purples', cmap(np.linspace(0.3, 1.0, 256))
    )
    plt.imshow(
        sim_matrix_np,
        cmap='viridis',
        extent=[0, duration, duration, 0],
        vmin=0,
        vmax=1,
    )  #'viridis'
    # plt.colorbar()
    # plt.title(f'Self-similarity Map - Sample {i} (Length: {actual_length})')
    plt.xlabel('Time (seconds)', labelpad=10)
    plt.ylabel('Time (seconds)', labelpad=10)

    # Save figure
    plt.savefig(f'{save_path}/{i}.pdf', dpi=300)
    plt.close()


def create_sim_matrix_video(
    save_path, feature_path, sample_idx, fps=10, fig_size=(10, 8)
):
  """Create a video showing a red dot moving along the diagonal of a similarity matrix.

  Args:
      save_path (str): Directory to save the visualization
      feature_path (str): Path to the feature tensor file
      sample_idx (int): Index of the sample to visualize
      fps (int): Frames per second for the output video
      fig_size (tuple): Figure size in inches (width, height)
  """
  h = torch.load(feature_path)  # [batch_size, t, 128]

  # Create output directory if it doesn't exist
  os.makedirs(save_path, exist_ok=True)

  # Get features for the specified sample
  features = h[sample_idx]  # [t, 128]

  # Find actual sequence length
  feature_magnitudes = torch.norm(features, dim=1)  # [t]
  is_padding = feature_magnitudes < 1e-6
  actual_length = (
      torch.where(is_padding)[0][0]
      if torch.any(is_padding)
      else features.shape[0]
  )

  # Only use the non-padded portion
  features = features[:actual_length]  # [actual_length, 128]

  # Compute similarity matrix
  features_norm = features / torch.norm(features, dim=1, keepdim=True)
  sim_matrix = torch.mm(
      features_norm, features_norm.t()
  )  # [actual_length, actual_length]

  # Convert to numpy for plotting
  sim_matrix_np = sim_matrix.cpu().numpy()
  fps = 5 if 'finegym' in feature_path else 20
  duration = sim_matrix_np.shape[0] / fps
  print('Duration:', duration)

  # Create frames for the video
  frames = []
  for i in range(actual_length):
    plt.figure(figsize=fig_size)
    # im = plt.imshow(sim_matrix_np, cmap='viridis', vmin=-1, vmax=1)  # Explicitly set value range
    # plt.colorbar(im)
    cmap = plt.cm.Purples_r
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'lighter_purples', cmap(np.linspace(0.3, 1.0, 256))
    )
    im = plt.imshow(
        sim_matrix_np, cmap='viridis', extent=[0, duration, duration, 0]
    )
    plt.colorbar(im)
    # plt.title(f'Self-similarity Map - Sample {sample_idx} (Step {i}/{actual_length})')
    plt.xlabel('Time (seconds)', labelpad=10)
    plt.ylabel('Time (seconds)', labelpad=10)

    # Add red dot at current position
    plt.plot(i / fps, i / fps, 'ro', markersize=10)

    # Convert plot to image
    plt.tight_layout()
    canvas = plt.gcf().canvas
    canvas.draw()
    frame = np.array(canvas.renderer.buffer_rgba())
    frame = frame[:, :, :3]  # Remove alpha channel
    frames.append(frame)
    plt.close()

  # Convert frames to numpy array
  frames = np.stack(frames)

  # # Save video
  # save_name = f'{save_path}/{sample_idx}_sim_matrix.mp4'
  # mediapy.write_video(save_name, frames, fps=fps)
  # print(f"Saved similarity matrix video to {save_name}")
  return frames


def create_combined_visualization(
    data, feature_path, save_path, sample_idx, fps=10
):
  """Create a combined visualization showing frames, camera motion, and similarity matrix side by side.

  Args:
      data: Tuple containing (frames, cam2world, id, label, label_name)
      feature_path (str): Path to the feature tensor file
      save_path (str): Directory to save the visualization
      fps (int): Frames per second for the output video
  """
  frames, cam2world, *_ = data
  print(sample_idx, frames.shape, cam2world.shape)

  save_name = os.path.join(save_path, f'combined_{sample_idx}.mp4')

  # if os.path.exists(save_name):
  #     print(f"Video already exists at {save_name}")
  #     return

  # Create camera visualization
  intrinsics = np.array([0.8660254, 0.8660254, 0.5, 0.5])
  fig_size = (frames.shape[2] / 100, frames.shape[1] / 100)
  cord = 'opencv' if 'finegym' in feature_path else ''
  video_camera = plot_camera(
      cam2world, intrinsics, fig_size=fig_size, cord=cord
  )

  # Create similarity matrix visualization with matching fig_size
  sim_matrix_frames = create_sim_matrix_video(
      save_path, feature_path, sample_idx=sample_idx, fps=fps, fig_size=fig_size
  )

  # Ensure all videos have the same number of frames
  min_frames = min(len(frames), len(video_camera), len(sim_matrix_frames))
  frames = frames[:min_frames]
  video_camera = video_camera[:min_frames]
  sim_matrix_frames = sim_matrix_frames[:min_frames]

  # Normalize all components to [0,1] range
  frames_norm = frames / 255.0
  sim_matrix_frames_norm = sim_matrix_frames / 255.0

  # Combine all visualizations side by side
  side_by_side = np.concatenate(
      [frames_norm, video_camera, sim_matrix_frames_norm], axis=2
  )

  # Debug print for final concatenated result
  print(
      'Side by side shape:',
      side_by_side.shape,
      'Range:',
      side_by_side.min(),
      side_by_side.max(),
  )

  # Save combined video
  mediapy.write_video(save_name, side_by_side, fps=fps)
  print(f'Saved combined visualization to {save_name}')


if __name__ == '__main__':
  # viz_dir = './viz/egoexo4d_action/val_verblabel15'
  # dataset = EgoExo4DVideoAndCameraPoseSeqForActionCounting('val', use_label_id='15')
  # idx = 163
  # data = dataset[idx]
  # viz_one_sim_matrix('tmp/egoexo4d_cut/idx163_dino.npy')
  # create_combined_visualization(data, 'tmp/egoexo4d_cut/h.pt', 'tmp/egoexo4d_cut/viz_seq', sample_idx=idx)
  viz_sim_matrix('tmp/egoexo4d_cut/h.pt', 'tmp/egoexo4d_cut/temp_sim_map', None)
  # viz_one_sim_matrix('tmp/finegym/idx824_dino.npy')
  # viz_sim_matrix('data/misc/finegym/idx0_1000_1000.pt', 'tmp/finegym/idx0_1000_1000', 'data/FineGym/annotations/finegym_val_split.csv')
  # dataset = FineGymVideoAndCamPoseSeq()
  # for idx in [812, 824, 371, 981]:
  #     data = dataset[idx]
  #     create_combined_visualization(data, 'data/misc/finegym/idx0_1000_1000.pt', 'tmp/finegym_viz', sample_idx=idx, fps=5)
  # break
