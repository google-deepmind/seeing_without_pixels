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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from utils.dataset_utils import matrix_to_pose7d, opencv_to_custom_c2w, poses_to_cam2world


def umeyama_similarity(X, Y, eps=1e-8):
  """Solve Y ≈ s R X + t for X,Y in R^{N×3}. Returns (s, R, t)."""
  X = np.asarray(X)
  Y = np.asarray(Y)
  assert X.ndim == 2 and Y.ndim == 2 and X.shape == Y.shape and X.shape[1] == 3

  muX = X.mean(axis=0)
  muY = Y.mean(axis=0)
  Xc = X - muX
  Yc = Y - muY

  # covariance
  Sigma = (Yc.T @ Xc) / X.shape[0]
  U, D, Vt = np.linalg.svd(Sigma)

  # det correction
  S_corr = np.eye(3)
  if np.linalg.det(U @ Vt) < 0:
    S_corr[2, 2] = -1.0

  R = U @ S_corr @ Vt

  varX = (Xc**2).sum() / X.shape[0]
  # scale numerator: sum(D * diag(S_corr))  (no .trace() on 1-D!)
  s = (D * np.diag(S_corr)).sum() / (varX + eps)

  t = muY - s * (R @ muX)
  return float(s), R, t


def se3_from_c2w(c2w):
  T = np.eye(4)
  T[:3, :3] = c2w[:3, :3]
  T[:3, 3] = c2w[:3, 3]
  return T


def rot_angle_deg(R):
  tr = np.clip((np.trace(R) - 1) * 0.5, -1.0, 1.0)
  return float(np.degrees(np.arccos(tr)))


def align_pose_and_compute_metrics(
    cam_gt, cam_est, delta=1, with_scaling=True, return_aligned=False
):
  """cam_gt, cam_est: (N,3,4) cam2world [R|t].

  Returns metrics dict and aligned estimates.
  """
  cam_gt = np.asarray(cam_gt)
  cam_est = np.asarray(cam_est)
  assert cam_gt.shape == cam_est.shape and cam_gt.shape[1:] == (3, 4)
  N = cam_gt.shape[0]

  # centers
  C_gt = cam_gt[:, :, 3]  # (N,3)
  C_est = cam_est[:, :, 3]  # (N,3)

  if with_scaling:
    s, R_a, t_a = umeyama_similarity(C_est, C_gt)
  else:
    s, R_a, t_a = 1.0, np.eye(3), np.zeros(3)

  # apply Sim(3) to full cam2world
  cam_est_aligned = cam_est.copy()
  cam_est_aligned[:, :3, :3] = R_a @ cam_est[:, :3, :3]
  cam_est_aligned[:, :3, 3] = (R_a @ (s * cam_est[:, :3, 3].T)).T + t_a

  if return_aligned:
    return cam_est_aligned

  # ATE-RMSE
  C_est_aligned = cam_est_aligned[:, :, 3]
  ate_rmse = float(
      np.sqrt(np.mean(np.sum((C_est_aligned - C_gt) ** 2, axis=1)))
  )

  # RPE (relative motions over step delta)
  T_gt = np.stack([se3_from_c2w(p) for p in cam_gt], axis=0)
  T_es = np.stack([se3_from_c2w(p) for p in cam_est_aligned], axis=0)

  rpe_t, rpe_r_deg = [], []
  for i in range(N - delta):
    Q_gt = np.linalg.inv(T_gt[i]) @ T_gt[i + delta]
    Q_es = np.linalg.inv(T_es[i]) @ T_es[i + delta]
    E = np.linalg.inv(Q_gt) @ Q_es
    rpe_t.append(np.linalg.norm(E[:3, 3]))
    rpe_r_deg.append(rot_angle_deg(E[:3, :3]))

  rpe_t = np.asarray(rpe_t)
  rpe_r_deg = np.asarray(rpe_r_deg)

  # return {
  #     "scale": s,
  #     "R_align": R_a,
  #     "t_align": t_a,
  #     "ATE_RMSE": ate_rmse,
  #     "RPE_T_RMSE": float(np.sqrt(np.mean(rpe_t**2))),
  #     "RPE_R_deg_mean": float(np.mean(rpe_r_deg)),
  #     "RPE_T_per_step": rpe_t,
  #     "RPE_R_deg_per_step": rpe_r_deg,
  # }
  return ate_rmse, float(np.sqrt(np.mean(rpe_t**2))), float(np.mean(rpe_r_deg))


DATA_DIR = os.path.expanduser('~/data/egoexo4d')
LOCAL_DATA_DIR = os.path.expanduser('~/local_data/egoexo4d')


def _load_megasam_estimated(path):
  cam_c2w = np.load(path)['cam_c2w']
  cam_c2w = opencv_to_custom_c2w(cam_c2w)
  return cam_c2w[:, :3, :]


def _load_vipe_estimated(path):
  cam_c2w = np.load(path)['data']
  cam_c2w = opencv_to_custom_c2w(cam_c2w)
  return cam_c2w[:, :3, :]


def _load_pi3_estimated(path):
  cam_c2w = np.load(path)
  cam_c2w = opencv_to_custom_c2w(cam_c2w)
  return cam_c2w[:, :3, :]


def _load_d4rt_estimated(path):
  cam_c2w = np.load(path)['poses_c2w']
  cam_c2w = opencv_to_custom_c2w(cam_c2w)
  return cam_c2w[:, :3, :]


def _load_scenario_gt(row, sample_points=80):
  file_path = os.path.join(
      LOCAL_DATA_DIR,
      'takes',
      row['take_name'],
      'trajectory_presr50/closed_loop_trajectory.csv',
  )
  trajectory_df = pd.read_csv(file_path)
  cam_gt = trajectory_df[[
      'tx_world_device',
      'ty_world_device',
      'tz_world_device',
      'qx_world_device',
      'qy_world_device',
      'qz_world_device',
      'qw_world_device',
  ]].values
  center = cam_gt.shape[0] // 2
  start = max(0, center - sample_points // 2)
  cam_gt = cam_gt[start : start + sample_points]
  cam_gt = poses_to_cam2world(cam_gt)
  return cam_gt


def _load_action_gt(row):
  fps = 20
  cam_trajectory_path = os.path.join(
      LOCAL_DATA_DIR,
      'takes',
      row['take_name'],
      'trajectory_presr50/closed_loop_trajectory.csv',
  )
  take_df = pd.read_csv(cam_trajectory_path)
  cam_traj = take_df[[
      'tx_world_device',
      'ty_world_device',
      'tz_world_device',
      'qx_world_device',
      'qy_world_device',
      'qz_world_device',
      'qw_world_device',
  ]].values
  range_start = max(0, int(row['start_time'] * fps))
  range_end = min(len(cam_traj), int(row['end_time'] * fps))
  cam_traj = cam_traj[range_start:range_end]
  cam_gt = poses_to_cam2world(cam_traj)
  return cam_gt


def resize_sequence(poses, target_len):
  if len(poses) == target_len:
    return poses
  idxs = np.linspace(0, len(poses) - 1, target_len, dtype=int)
  return poses[idxs]


def run(scenario=True, with_scaling=True):
  name = 'scenario' if scenario else 'action'
  print(f'Evaluating {name}, with_scaling={with_scaling}')
  df = pd.read_csv(
      f'{DATA_DIR}/annotations/downstream/{name}_cls_with_all3_pred.csv'
  )
  preds_dict = {}
  results_ate, results_rpe_t, results_rpe_r_deg = [], [], []

  for i, row in tqdm(df.iterrows(), total=len(df)):
    megasam_path = row['megasam_path'] if scenario else row['pred_path']
    poses_megasam = _load_megasam_estimated(megasam_path)
    pi3_path = (
        row['pi3_path']
        if scenario
        else os.path.join(
            'baselines/Pi3/preds_action_new',
            row['video_fp'].replace('/', '_').replace('.mp4', '.npy'),
        )
    )
    poses_pi3 = _load_pi3_estimated(pi3_path)
    vipe_path = row['vipe_path'] if scenario else row['vipe_pred_path']
    poses_vipe = _load_vipe_estimated(vipe_path)
    poses_gt = _load_scenario_gt(row) if scenario else _load_action_gt(row)
    d4rt_path = os.path.join('d4rt_preds', row['take_name'] + '.npz')
    poses_d4rt = _load_d4rt_estimated(d4rt_path)

    min_len = min(
        len(poses_megasam), len(poses_pi3), len(poses_vipe), len(poses_gt)
    )
    poses_megasam = resize_sequence(poses_megasam, min_len)
    poses_pi3 = resize_sequence(poses_pi3, min_len)
    poses_vipe = resize_sequence(poses_vipe, min_len)
    poses_gt = resize_sequence(poses_gt, min_len)
    poses_d4rt = resize_sequence(poses_d4rt, min_len)

    preds_dict[f'{i:04d}_megasam'] = poses_megasam
    preds_dict[f'{i:04d}_pi3'] = poses_pi3
    preds_dict[f'{i:04d}_vipe'] = poses_vipe
    preds_dict[f'{i:04d}_gt'] = poses_gt
    preds_dict[f'{i:04d}_d4rt'] = poses_d4rt

    preds_dict[f'{i:04d}_megasam_transformed'] = align_pose_and_compute_metrics(
        poses_gt, poses_megasam, return_aligned=True
    )
    preds_dict[f'{i:04d}_pi3_transformed'] = align_pose_and_compute_metrics(
        poses_gt, poses_pi3, return_aligned=True
    )
    preds_dict[f'{i:04d}_vipe_transformed'] = align_pose_and_compute_metrics(
        poses_gt, poses_vipe, return_aligned=True
    )
    preds_dict[f'{i:04d}_d4rt_transformed'] = align_pose_and_compute_metrics(
        poses_gt, poses_d4rt, return_aligned=True
    )

    ate_0, rpe_t0, rpe_r_deg0 = align_pose_and_compute_metrics(
        poses_gt, poses_megasam, with_scaling=with_scaling
    )
    ate_1, rpe_t1, rpe_r_deg1 = align_pose_and_compute_metrics(
        poses_gt, poses_vipe, with_scaling=with_scaling
    )
    ate_2, rpe_t2, rpe_r_deg2 = align_pose_and_compute_metrics(
        poses_gt, poses_pi3, with_scaling=with_scaling
    )
    ate_3, rpe_t3, rpe_r_deg3 = align_pose_and_compute_metrics(
        poses_gt, poses_d4rt, with_scaling=with_scaling
    )

    results_ate.append([ate_0, ate_1, ate_2, ate_3])
    results_rpe_t.append([rpe_t0, rpe_t1, rpe_t2, rpe_t3])
    results_rpe_r_deg.append([rpe_r_deg0, rpe_r_deg1, rpe_r_deg2, rpe_r_deg3])

  results_ate = np.array(results_ate)
  results_rpe_t = np.array(results_rpe_t)
  results_rpe_r_deg = np.array(results_rpe_r_deg)

  print('Methods: Megasam, ViPE, PI3, D4RT')
  print('ATE', results_ate.shape, np.mean(results_ate, axis=0))
  print(
      'RPE-T',
      results_rpe_t.shape,
      np.nanmean(results_rpe_t, axis=0),
      np.isnan(results_rpe_t).sum(),
  )
  print(
      'RPE-R-deg',
      results_rpe_r_deg.shape,
      np.nanmean(results_rpe_r_deg, axis=0),
      np.isnan(results_rpe_r_deg).sum(),
  )

  np.savez(
      f'{DATA_DIR}/cam_est_cache/{name}_cls_with_all4_pred_transformed.npz',
      **preds_dict,
  )


def return_2pose_metrics(pose1, pose2):
  ate_0, rpe_t0, rpe_r_deg0 = align_pose_and_compute_metrics(
      pose1, pose2, with_scaling=True
  )
  ate_1, rpe_t1, rpe_r_deg1 = align_pose_and_compute_metrics(
      pose2, pose1, with_scaling=True
  )
  return (
      (ate_0 + ate_1) / 2,
      (rpe_t0 + rpe_t1) / 2,
      (rpe_r_deg0 + rpe_r_deg1) / 2,
  )


def cmp_ucf():
  data_dir = os.path.expanduser('~/local_data/ucf101')
  pi3_pose_paths = glob.glob(f'{data_dir}/pi3_pose/*.npy')
  results_df = []
  for i, pi3_pose_path in tqdm(
      enumerate(pi3_pose_paths), total=len(pi3_pose_paths)
  ):
    seq = os.path.basename(pi3_pose_path).replace('.npy', '')
    vipe_pose_path = pi3_pose_path.replace('pi3_pose', 'vipe_pose').replace(
        '.npy', '.npz'
    )
    megasam_pose_path = pi3_pose_path.replace(
        'pi3_pose', 'megasam_pose'
    ).replace('.npy', '_droid.npz')
    if not os.path.exists(vipe_pose_path) or not os.path.exists(
        megasam_pose_path
    ):
      continue
    megasam_pose = np.load(megasam_pose_path)['cam_c2w'][:, :3, :]
    pi3_pose = np.load(pi3_pose_path)[:, :3, :]
    vipe_pose = np.load(vipe_pose_path)['data'][:, :3, :]

    min_len = min(len(megasam_pose), len(pi3_pose), len(vipe_pose))
    megasam_pose = resize_sequence(megasam_pose, min_len)
    pi3_pose = resize_sequence(pi3_pose, min_len)
    vipe_pose = resize_sequence(vipe_pose, min_len)

    ate_0, rpe_t0, rpe_r0 = return_2pose_metrics(megasam_pose, pi3_pose)
    ate_1, rpe_t1, rpe_r1 = return_2pose_metrics(megasam_pose, vipe_pose)
    ate_2, rpe_t2, rpe_r2 = return_2pose_metrics(pi3_pose, vipe_pose)
    results_df.append({
        'seq': seq,
        'ate_megasam_pi3': ate_0,
        'rpe_t_megasam_pi3': rpe_t0,
        'rpe_r_deg_megasam_pi3': rpe_r0,
        'ate_megasam_vipe': ate_1,
        'rpe_t_megasam_vipe': rpe_t1,
        'rpe_r_deg_megasam_vipe': rpe_r1,
        'ate_pi3_vipe': ate_2,
        'rpe_t_pi3_vipe': rpe_t2,
        'rpe_r_deg_pi3_vipe': rpe_r2,
    })

  results_df = pd.DataFrame(results_df)
  results_df.to_csv(f'{data_dir}/cmp_3pose.csv', index=False)


def motion_dynamics_score(poses, w_trans=1.0, w_rot=1.0):
  """Compute a single scalar representing how dynamic a camera trajectory is.

  Args:
      poses: (N, 3, 4) array of camera-to-world or world-to-camera poses
      w_trans: weight for translation magnitude
      w_rot: weight for rotation magnitude

  Returns:
      float: motion dynamics score (higher = more dynamic)
  """
  t = poses[:, :3, 3]  # (N, 3)
  Rm = poses[:, :3, :3]  # (N, 3, 3)
  rot = R.from_matrix(Rm)

  # --- translation displacement per frame ---
  trans_disp = np.linalg.norm(np.diff(t, axis=0), axis=1)
  trans_mean = trans_disp.mean()

  # --- rotation angle change per frame ---
  ang_disp = []
  for i in range(1, len(rot)):
    rel_rot = rot[i - 1].inv() * rot[i]
    ang_disp.append(rel_rot.magnitude())  # radians
  ang_disp = np.array(ang_disp)
  rot_mean = ang_disp.mean()

  # --- combine into single scalar ---
  score = w_trans * np.log1p(trans_mean) + w_rot * np.log1p(rot_mean)
  return float(score)


def ucf_motion_dynamics():
  data_dir = os.path.expanduser('~/local_data/ucf101')
  pi3_pose_paths = glob.glob(f'{data_dir}/pi3_pose/*.npy')
  results_df = []
  for i, pi3_pose_path in tqdm(
      enumerate(pi3_pose_paths), total=len(pi3_pose_paths)
  ):
    seq = os.path.basename(pi3_pose_path).replace('.npy', '')
    vipe_pose_path = pi3_pose_path.replace('pi3_pose', 'vipe_pose').replace(
        '.npy', '.npz'
    )
    megasam_pose_path = pi3_pose_path.replace(
        'pi3_pose', 'megasam_pose'
    ).replace('.npy', '_droid.npz')
    if not os.path.exists(vipe_pose_path) or not os.path.exists(
        megasam_pose_path
    ):
      continue
    megasam_pose = np.load(megasam_pose_path)['cam_c2w'][:, :3, :]
    pi3_pose = np.load(pi3_pose_path)[:, :3, :]
    vipe_pose = np.load(vipe_pose_path)['data'][:, :3, :]

    min_len = min(len(megasam_pose), len(pi3_pose), len(vipe_pose))
    megasam_pose = resize_sequence(megasam_pose, min_len)
    pi3_pose = resize_sequence(pi3_pose, min_len)
    vipe_pose = resize_sequence(vipe_pose, min_len)

    score_megasam = motion_dynamics_score(poses=megasam_pose)
    score_pi3 = motion_dynamics_score(poses=pi3_pose)
    score_vipe = motion_dynamics_score(poses=vipe_pose)
    results_df.append({
        'seq': seq,
        'score_megasam': score_megasam,
        'score_pi3': score_pi3,
        'score_vipe': score_vipe,
    })
  results_df = pd.DataFrame(results_df)
  results_df.to_csv(f'{data_dir}/ucf101_motion_dynamics.csv', index=False)


def ucf_stat():
  df = pd.read_csv('local_data/ucf101/cmp_3pose.csv')
  df['action'] = df['seq'].apply(lambda x: x.split('_')[1])
  df['mean_ate'] = df[
      ['ate_megasam_pi3', 'ate_megasam_vipe', 'ate_pi3_vipe']
  ].mean(axis=1)
  df['mean_rpe_t'] = df[
      ['rpe_t_megasam_pi3', 'rpe_t_megasam_vipe', 'rpe_t_pi3_vipe']
  ].mean(axis=1)
  df['mean_rpe_r'] = df[
      ['rpe_r_deg_megasam_pi3', 'rpe_r_deg_megasam_vipe', 'rpe_r_deg_pi3_vipe']
  ].mean(axis=1)

  # Keep ~80% best (lowest errors)
  keep_ate = df['mean_ate'] < df['mean_ate'].quantile(0.8)
  keep_rpe_t = df['mean_rpe_t'] < df['mean_rpe_t'].quantile(0.8)
  keep_rpe_r = df['mean_rpe_r'] < df['mean_rpe_r'].quantile(0.8)

  df_filtered = df[keep_ate & keep_rpe_t & keep_rpe_r]
  print(f'Kept {len(df_filtered)/len(df):.1%} sequences')

  # Compute per-action mean ATE
  action_means = df_filtered.groupby('action')['mean_ate'].mean().sort_values()

  # Plot histogram / bar chart
  plt.figure(figsize=(10, 5))
  plt.bar(
      action_means.index,
      action_means.values,
      color='#99d8c9',
      edgecolor='gray',
      alpha=0.8,
  )
  plt.xticks(rotation=90, fontsize=8)
  plt.ylabel('Mean ATE')
  plt.xlabel('Action')
  plt.title('Mean ATE per Action (UCF101)')
  plt.tight_layout()
  plt.savefig('tmp/ucf101_action_ate.png', dpi=300)


def ucf_motion_stat(top_k=101):
  df = pd.read_csv('local_data/ucf101/ucf101_motion_dynamics.csv')
  df['action'] = df['seq'].apply(lambda x: x.split('_')[1])
  for col in ['score_megasam', 'score_pi3', 'score_vipe']:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)

  grouped = df.groupby('action')[
      ['score_megasam', 'score_pi3', 'score_vipe']
  ].mean()
  # sort by average dynamics (mean across all methods)
  grouped['mean_score'] = grouped.mean(axis=1)
  grouped = grouped.sort_values('mean_score', ascending=True)
  top_actions = grouped.tail(top_k).iloc[::-1]  # reverse so highest first
  # print(f"\nTop {top_k} most dynamic actions:")
  # print(top_actions['mean_score'].round(3))
  for action, score in top_actions['mean_score'].round(6).items():
    print(f'{action}: {score}')

  grouped = grouped.drop(columns='mean_score')

  actions = grouped.index.tolist()
  x = np.arange(len(actions))
  width = 0.25

  fig, ax = plt.subplots(figsize=(12, 5))
  ax.bar(x - width, grouped['score_megasam'], width, label='MegaSAM')
  ax.bar(x, grouped['score_pi3'], width, label='PI3')
  ax.bar(x + width, grouped['score_vipe'], width, label='ViPE')

  ax.set_xticks(x)
  ax.set_xticklabels(actions, rotation=90)
  ax.set_ylabel('Mean Motion Dynamics Score')
  ax.set_xlabel('Action Category')
  ax.legend()
  ax.grid(axis='y', linestyle='--', alpha=0.5)

  plt.tight_layout()
  plt.savefig('tmp/ucf101_motion_dynamics.png', dpi=300)


if __name__ == '__main__':
  run()
  # cmp_ucf()
  # ucf_stat()
  # ucf_motion_dynamics()
  # ucf_motion_stat()
