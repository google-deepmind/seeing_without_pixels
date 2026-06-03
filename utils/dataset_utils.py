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

import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_R(q):
  """qx, qy, qz, qw -> 3x3 rotation matrix (right-handed, unit quaternion)."""
  qx, qy, qz, qw = q
  n = np.linalg.norm([qx, qy, qz, qw])
  qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
  x2, y2, z2 = 2 * qx, 2 * qy, 2 * qz
  xx, yy, zz = qx * x2, qy * y2, qz * z2
  xy, xz, yz = qx * y2, qx * z2, qy * z2
  wx, wy, wz = qw * x2, qw * y2, qw * z2
  R = np.array(
      [
          [1 - (yy + zz), xy - wz, xz + wy],
          [xy + wz, 1 - (xx + zz), yz - wx],
          [xz - wy, yz + wx, 1 - (xx + yy)],
      ],
      dtype=float,
  )
  return R


def pose7d_to_matrix(pose):
  t = pose[:3]
  q = pose[3:]
  R = Rotation.from_quat(q).as_matrix()
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = t
  return T


def matrix_to_pose7d(T):
  """Convert (3,4)/(4,4) or (N,3,4)/(N,4,4) pose matrix to 7D pose [tx,ty,tz,qx,qy,qz,qw]."""
  is_batched = T.ndim == 3
  if not is_batched:
    T = T[None, ...]
  if T.shape[-2:] == (4, 4):
    T = T[:, :3, :]
  Rs = T[:, :3, :3]
  ts = T[:, :3, 3]
  qs = Rotation.from_matrix(Rs).as_quat()
  poses7d = np.concatenate([ts, qs], axis=1)
  return poses7d[0] if not is_batched else poses7d


def w2c_to_c2w(poses_w2c):
  """Convert world-to-camera extrinsics to camera-to-world.

  Args:
      poses_w2c: np.ndarray of shape (T, 3, 4) or (T, 4, 4), [R|t]
        world-to-camera.

  Returns:
      poses_c2w: np.ndarray of shape (T, 3, 4), [R|t] camera-to-world.
  """
  out = []
  for P in poses_w2c:
    R, t = P[:3, :3], P[:3, 3]
    R_c2w = R.T
    t_c2w = -R.T @ t
    out.append(np.hstack([R_c2w, t_c2w[:, None]]))
  return np.stack(out, axis=0)


def absolute_to_relative_from_t0(absolute_pose):
  T0 = pose7d_to_matrix(absolute_pose[0])
  rel_poses = []
  for i in range(len(absolute_pose)):
    Ti = pose7d_to_matrix(absolute_pose[i])
    T_rel = np.linalg.inv(T0) @ Ti
    rel_poses.append(matrix_to_pose7d(T_rel))
  return np.stack(rel_poses, axis=0)


def absolute_to_relative(absolute_pose, ref_frame_idx=0):
  """Make all poses relative to absolute_pose[ref_frame_idx]."""
  N = len(absolute_pose)
  if ref_frame_idx < 0 or ref_frame_idx >= N:
    raise ValueError(
        f"ref_frame_idx {ref_frame_idx} is out of bounds [0, {N - 1}]"
    )
  T_ref = pose7d_to_matrix(absolute_pose[ref_frame_idx])
  rel_poses = []
  for i in range(N):
    Ti = pose7d_to_matrix(absolute_pose[i])
    T_rel = np.linalg.inv(T_ref) @ Ti
    rel_poses.append(matrix_to_pose7d(T_rel))
  return np.stack(rel_poses, axis=0)


def pose7_to_pose9(cam_motion):
  """(N,7) [tx,ty,tz,qx,qy,qz,qw] -> (N,9) [tx,ty,tz, 6D rot (Zhou et al.)]."""
  translations = cam_motion[:, :3]
  quaternions = cam_motion[:, 3:]
  assert np.all(
      np.isclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-6)
  ), "Quaternions are not normalized"
  rot_matrices = Rotation.from_quat(quaternions).as_matrix()
  rot_6d = rot_matrices[:, :, :2].reshape(-1, 6)
  return np.concatenate([translations, rot_6d], axis=1)


def get_gravity_in_camera(quat):
  """Return gravity vector (3D) in camera coordinates."""
  R_cw = quat_to_R(quat)
  g_w = np.array([0, 0, -1.0])
  g_c = R_cw.T @ g_w
  return g_c


def transform_camera_pose(cam_traj, encode_pose):
  """Map a raw camera trajectory to the representation CamFormer consumes.

  encode_pose is the integer that the --pose_encoding flag maps to:
      0  (abs7d)      raw absolute 7D pose
      2  (rel9d)      pose relative to the middle frame, 9D (6D rotation +
      translation)
      11 (rel9d_grav) rel9d plus the gravity-in-camera vector
  """
  if encode_pose == 0:
    return cam_traj
  if encode_pose in (2, 11):
    ref_frame = cam_traj[len(cam_traj) // 2]
    cam_traj = absolute_to_relative(cam_traj, ref_frame_idx=len(cam_traj) // 2)
    cam_traj = pose7_to_pose9(cam_traj)
    if encode_pose == 11:
      gravity = get_gravity_in_camera(ref_frame[3:])
      cam_traj = np.hstack([cam_traj, np.tile(gravity, (cam_traj.shape[0], 1))])
    return cam_traj
  raise ValueError(f"Unknown encode_pose {encode_pose}")
