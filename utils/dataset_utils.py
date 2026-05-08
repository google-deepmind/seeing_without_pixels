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

import subprocess
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_R(q):
  """qx, qy, qz, qw -> 3x3 rotation matrix (right-handed, unit quaternion)."""
  qx, qy, qz, qw = q
  # normalize just in case
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


def raymap_naive(intrinsics, pose, convention="camera_to_world"):
  """pose: (tx, ty, tz, qx, qy, qz, qw)

  returns: (H, W, 6) array with [o_x, o_y, o_z, d_x, d_y, d_z] per pixel
  """
  fx, fy, cx, cy, H, W = intrinsics
  tx, ty, tz, qx, qy, qz, qw = pose
  R = quat_to_R((qx, qy, qz, qw))

  if convention == "camera_to_world":
    R_c2w = R
    t_c2w = np.array([tx, ty, tz], dtype=float)
    o_world = t_c2w.copy()  # camera center in world
    R_world_from_cam = R_c2w  # rotate directions into world
  elif convention == "world_to_camera":
    R_w2c = R
    t_w2c = np.array([tx, ty, tz], dtype=float)
    o_world = -R_w2c.T @ t_w2c  # camera center in world
    R_world_from_cam = R_w2c.T  # rotate directions into world
  else:
    raise ValueError(
        "convention must be 'camera_to_world' or 'world_to_camera'"
    )

  # pixel grid (u to the right, v down). Shape (H,W)
  u = np.arange(W, dtype=float)[None, :].repeat(H, 0)
  v = np.arange(H, dtype=float)[:, None].repeat(W, 1)

  # directions in camera coordinates (pinhole, no distortion), shape (H,W,3)
  x = (u - cx) / fx
  y = (v - cy) / fy
  z = np.ones_like(x)

  d_cam = np.stack([x, y, z], axis=-1)  # (H,W,3)
  d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)

  # rotate to world coordinates
  d_world = d_cam @ R_world_from_cam.T  # (H,W,3)

  # fill raymap
  raymap = np.zeros((H, W, 6), dtype=np.float32)
  raymap[..., :3] = o_world[None, None, :]  # same origin for all pixels
  raymap[..., 3:] = d_world.astype(np.float32)  # per-pixel unit direction
  return raymap


def quat_to_R_batch(q):
  """q: (T,4) with (qx,qy,qz,qw) -> (T,3,3) rotation matrices"""
  qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  n = np.linalg.norm(q, axis=1, keepdims=True)
  qx, qy, qz, qw = qx / n[:, 0], qy / n[:, 0], qz / n[:, 0], qw / n[:, 0]
  x2, y2, z2 = 2 * qx, 2 * qy, 2 * qz
  xx, yy, zz = qx * x2, qy * y2, qz * z2
  xy, xz, yz = qx * y2, qx * z2, qy * z2
  wx, wy, wz = qw * x2, qw * y2, qw * z2
  R = np.empty((len(qx), 3, 3), dtype=float)
  R[:, 0, 0] = 1 - (yy + zz)
  R[:, 0, 1] = xy - wz
  R[:, 0, 2] = xz + wy
  R[:, 1, 0] = xy + wz
  R[:, 1, 1] = 1 - (xx + zz)
  R[:, 1, 2] = yz - wx
  R[:, 2, 0] = xz - wy
  R[:, 2, 1] = yz + wx
  R[:, 2, 2] = 1 - (xx + yy)
  return R


def raymap_naive_batch(intrinsics, cam_pose_traj, convention="camera_to_world"):
  """intrinsics: (fx, fy, cx, cy, H, W)

  cam_pose_traj: (T,7) [tx, ty, tz, qx, qy, qz, qw]
  returns: (T, H, W, 6) raymaps
  """
  fx, fy, cx, cy, H, W = intrinsics
  tx, ty, tz = cam_pose_traj[:, 0], cam_pose_traj[:, 1], cam_pose_traj[:, 2]
  quats = cam_pose_traj[:, 3:7]

  R_batch = quat_to_R_batch(quats)  # (T,3,3)

  if convention == "camera_to_world":
    o_world = np.stack([tx, ty, tz], axis=1)  # (T,3)
    R_world_from_cam = R_batch
  elif convention == "world_to_camera":
    # origin = -R^T @ t
    o_world = -(R_batch.transpose(0, 2, 1) @ cam_pose_traj[:, 0:3, None])[
        ..., 0
    ]
    R_world_from_cam = R_batch.transpose(0, 2, 1)
  else:
    raise ValueError(
        "convention must be 'camera_to_world' or 'world_to_camera'"
    )

  # 1. Precompute pixel grid in camera coords (H,W,3)
  u = np.arange(W, dtype=float)[None, :].repeat(H, 0)
  v = np.arange(H, dtype=float)[:, None].repeat(W, 1)
  x = (u - cx) / fx
  y = (v - cy) / fy
  z = np.ones_like(x)
  d_cam = np.stack([x, y, z], axis=-1)  # (H,W,3)
  d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)

  # 2. Rotate all directions for all frames (T,H,W,3)
  # First flatten pixel grid to (H*W,3) for efficient matmul
  d_cam_flat = d_cam.reshape(-1, 3)  # (HW,3)
  # Apply each R_world_from_cam to the same set of d_cam vectors
  d_world = R_world_from_cam @ d_cam_flat.T[None, :, :]  # (T,3,HW)
  d_world = d_world.transpose(0, 2, 1).reshape(len(cam_pose_traj), H, W, 3)

  # 3. Fill raymaps
  raymaps = np.zeros((len(cam_pose_traj), H, W, 6), dtype=np.float32)
  raymaps[..., :3] = o_world[:, None, None, :]  # broadcast origins
  raymaps[..., 3:] = d_world.astype(np.float32)  # per-pixel directions
  return raymaps


def pose7d_to_matrix(pose):
  t = pose[:3]
  q = pose[3:]
  R = Rotation.from_quat(q).as_matrix()  # shape (3, 3)
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = t
  return T


def matrix_to_pose7d(T):
  """Convert (3, 4) or (N, 3, 4) pose matrix to 7D pose vector(s): tx, ty, tz, qx, qy, qz, qw

  Args:
      T (np.ndarray): Pose matrix of shape (3, 4) or (N, 3, 4)

  Returns:
      pose7d: (7,) if input is (3, 4), or (N, 7) if input is (N, 3, 4)
  """
  is_batched = T.ndim == 3
  if not is_batched:
    T = T[None, ...]  # convert to shape (1, 3, 4)
  if T.shape[-2:] == (4, 4):
    T = T[:, :3, :]
  Rs = T[:, :3, :3]  # (N, 3, 3)
  ts = T[:, :3, 3]  # (N, 3)
  qs = Rotation.from_matrix(Rs).as_quat()  # (N, 4)
  poses7d = np.concatenate([ts, qs], axis=1)  # (N, 7)

  return poses7d[0] if not is_batched else poses7d


def absolute_to_relative_from_t0(absolute_pose):  # shape: (N, 7)
  T0 = pose7d_to_matrix(absolute_pose[0])
  rel_poses = []
  for i in range(len(absolute_pose)):
    Ti = pose7d_to_matrix(absolute_pose[i])
    T_rel = np.linalg.inv(T0) @ Ti
    rel_pose = matrix_to_pose7d(T_rel)
    rel_poses.append(rel_pose)
  return np.stack(rel_poses, axis=0)  # shape: (N, 7)


def absolute_to_relative(absolute_pose, ref_frame_idx=0):  # shape: (N, 7)
  """Convert absolute poses to relative poses using a specified reference frame.

  Args:
      absolute_pose (np.ndarray): Array of absolute poses with shape (N, 7)
      ref_frame_idx (int): Index of the reference frame. Default is 0 (first
        frame). Use N//2 for middle frame, or any valid index.

  Returns:
      np.ndarray: Array of relative poses with shape (N, 7)
  """
  N = len(absolute_pose)
  if ref_frame_idx < 0 or ref_frame_idx >= N:
    raise ValueError(
        f"ref_frame_idx {ref_frame_idx} is out of bounds [0, {N-1}]"
    )

  T_ref = pose7d_to_matrix(absolute_pose[ref_frame_idx])
  rel_poses = []
  for i in range(N):
    Ti = pose7d_to_matrix(absolute_pose[i])
    T_rel = np.linalg.inv(T_ref) @ Ti
    rel_pose = matrix_to_pose7d(T_rel)
    rel_poses.append(rel_pose)
  return np.stack(rel_poses, axis=0)  # shape: (N, 7)


def absolute_to_relative_sequential(absolute_pose):
  """Convert absolute poses to relative poses where each pose t is relative to absolute pose t-1.

  rel_pose[t] = inv(abs_pose[t-1]) @ abs_pose[t] The first pose is set to
  identity: [0, 0, 0, 0, 0, 0, 1]

  Args:
      absolute_pose (np.ndarray): Array of absolute poses with shape (N, 7)

  Returns:
      np.ndarray: Array of relative poses with shape (N, 7)
  """
  N = len(absolute_pose)
  rel_poses = [
      np.array([0, 0, 0, 0, 0, 0, 1], dtype=absolute_pose.dtype)
  ]  # identity pose
  for t in range(1, N):
    T_prev = pose7d_to_matrix(absolute_pose[t - 1])
    T_curr = pose7d_to_matrix(absolute_pose[t])
    T_rel = np.linalg.inv(T_prev) @ T_curr
    rel_pose = matrix_to_pose7d(T_rel)
    rel_poses.append(rel_pose)
  return np.stack(rel_poses, axis=0)  # shape: (N, 7)


def custom_to_opencv_c2w(poses_in: np.ndarray) -> np.ndarray:
  """Convert c2w (camera-to-world) poses from a custom coordinate system

  (x=down, y=left, z=forward) to the OpenCV convention (x=right, y=down,
  z=forward).

  A c2w pose transforms a point from the camera's coordinate system to the
  world's coordinate system: p_world = R @ p_camera + t.

  This function adjusts the rotation matrix so that it correctly transforms
  a point from the OpenCV camera frame, rather than the custom camera frame.

  Args:
      poses_in: np.ndarray of shape (T, 3, 4), representing [R|t] c2w poses in
        the custom convention. R is the orientation of the camera in the world,
        and t is the position of the camera in the world.

  Returns:
      poses_cv: np.ndarray of shape (T, 3, 4), representing [R|t] c2w poses
                that now operate on points in the OpenCV camera convention.
  """
  # This is the change-of-basis matrix that transforms a point from the
  # custom camera frame to the OpenCV camera frame.
  # p_opencv = M @ p_custom
  M = np.array([
      [0, -1, 0],  # OpenCV x-axis is the custom negative y-axis
      [1, 0, 0],  # OpenCV y-axis is the custom x-axis
      [0, 0, 1],  # OpenCV z-axis is the custom z-axis
  ])

  # For a c2w transformation, the input point is in the camera frame. We are
  # changing the convention of this input frame.
  # Original pose equation: p_world = R_custom @ p_custom + t
  # We want a new pose [R_cv | t_cv] such that: p_world = R_cv @ p_opencv + t_cv
  # We know p_custom = M.T @ p_opencv (inverse of M is M.T)
  # Substituting this into the original equation gives:
  # p_world = R_custom @ (M.T @ p_opencv) + t
  # p_world = (R_custom @ M.T) @ p_opencv + t
  #
  # From this, we can see:
  # R_cv = R_custom @ M.T
  # t_cv = t (The camera's position in world coordinates does not change)

  converted_poses = []
  for P_in in poses_in:
    R_custom, t = P_in[:, :3], P_in[:, 3]

    # Adjust the rotation matrix to post-multiply by the inverse change-of-basis.
    R_cv = R_custom @ M.T

    # The translation vector (camera's world position) remains unchanged.
    t_cv = t

    # Combine the new rotation and original translation.
    converted_poses.append(np.hstack([R_cv, t_cv[:, None]]))

  return np.stack(converted_poses, axis=0)


def opencv_to_custom_c2w(poses_cv: np.ndarray) -> np.ndarray:
  """Convert c2w (camera-to-world) poses from OpenCV convention

  (x=right, y=down, z=forward) to the custom convention
  (x=down, y=left, z=forward).

  Given:
      p_opencv = M @ p_custom

  OpenCV c2w:  p_world = R_cv @ p_opencv + t_cv
  Substitute:  p_world = R_cv @ (M @ p_custom) + t_cv
               => R_custom = R_cv @ M,  t_custom = t_cv

  Args:
      poses_cv: (T, 3, 4) array of [R|t] c2w in OpenCV convention.

  Returns:
      poses_custom: (T, 3, 4) array of [R|t] c2w in the custom convention.
  """
  M = np.array([
      [0, -1, 0],  # OpenCV x = custom -y
      [1, 0, 0],  # OpenCV y = custom  x
      [0, 0, 1],  # OpenCV z = custom  z
  ])

  out = []
  for P in poses_cv:
    R_cv, t = P[:, :3], P[:, 3]
    R_custom = R_cv @ M
    t_custom = t  # same camera position in world coords
    out.append(np.hstack([R_custom, t_custom[:, None]]))
  return np.stack(out, axis=0)


def poses_to_cam2world(poses):
  """Convert [T,7] pose vectors to [T,3,4] camera-to-world transforms.

  Args:
      poses (np.ndarray): [T, 7] where each row is [tx, ty, tz, qx, qy, qz, qw]

  Returns:
      np.ndarray: [T, 3, 4] camera-to-world extrinsics
  """
  T = poses.shape[0]
  cam2world = np.zeros((T, 3, 4))

  translations = poses[:, :3]  # (T, 3)
  quaternions = poses[:, 3:]  # (T, 4) in (x, y, z, w)

  # Convert quaternion to rotation matrix
  rot_mats = Rotation.from_quat(quaternions).as_matrix()  # (T, 3, 3)

  cam2world[:, :, :3] = rot_mats
  cam2world[:, :, 3] = translations

  return cam2world


def pose7_to_pose9(cam_motion):
  """Convert (N, 7) pose [tx, ty, tz, qx, qy, qz, qw] to (N, 9) representation

  with 6D rotation (Zhou et al.) + 3D translation.
  """
  # Split translation and quaternion
  translations = cam_motion[:, :3]  # (N, 3)
  quaternions = cam_motion[:, 3:]  # (N, 4)

  is_normalized = np.isclose(
      np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-6
  )
  assert np.all(is_normalized), "Quaternions are not normalized"

  # Convert to rotation matrices
  r = Rotation.from_quat(quaternions)
  rot_matrices = r.as_matrix()  # (N, 3, 3)

  # Take first two columns of each matrix and flatten → 6D
  rot_6d = rot_matrices[:, :, :2].reshape(-1, 6)  # (N, 6)

  # Concatenate with translation
  pose_9d = np.concatenate([translations, rot_6d], axis=1)  # (N, 9)
  return pose_9d


def pose7_to_pose12(cam_motion):
  """Convert (N,7) poses [tx, ty, tz, qx, qy, qz, qw] to (N,12) row-major [R|t].

  Output order: [r11 r12 r13 r21 r22 r23 r31 r32 r33 tx ty tz]
  """
  cam_motion = np.asarray(cam_motion)
  assert (
      cam_motion.ndim == 2 and cam_motion.shape[1] == 7
  ), "Input must be (N,7)"
  t = cam_motion[:, :3]  # (N,3)
  q = cam_motion[:, 3:]  # (N,4) [x,y,z,w]

  norms = np.linalg.norm(q, axis=1, keepdims=True)
  if not np.allclose(norms, 1.0, atol=1e-6):
    q = q / norms  # safely normalize

  Rm = Rotation.from_quat(q).as_matrix()  # (N,3,3)
  mat34 = np.concatenate([Rm, t[:, :, None]], axis=2)  # (N,3,4)
  pose12 = mat34.reshape(-1, 12)  # row-major flatten
  return pose12


def w2c_to_c2w(poses_w2c: np.ndarray) -> np.ndarray:
  """Convert world-to-camera extrinsics to camera-to-world.

  Args:
      poses_w2c: np.ndarray of shape (T, 3, 4), [R|t] world-to-camera.

  Returns:
      poses_c2w: np.ndarray of shape (T, 3, 4), [R|t] camera-to-world.
  """
  out = []
  for P in poses_w2c:
    R, t = P[:, :3], P[:, 3]
    R_c2w = R.T
    t_c2w = -R.T @ t
    out.append(np.hstack([R_c2w, t_c2w[:, None]]))
  return np.stack(out, axis=0)


def get_precise_frame_count_ffprobe(video_path):
  cmd = [
      "ffprobe",
      "-v",
      "error",
      "-count_frames",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=nb_read_frames",
      "-of",
      "default=nokey=1:noprint_wrappers=1",
      video_path,
  ]
  output = subprocess.check_output(cmd).decode().strip()
  return int(output)


def extract_video_frames(video_path, n_frames, start_time=None, end_time=None):
  """Extract frames from a video file at equal intervals.

  Args:
      video_path (str): Path to the video file
      n_frames (int): Number of frames to extract
      start_time (float, optional): Start time in seconds. If None, starts from
        beginning.
      end_time (float, optional): End time in seconds. If None, ends at video
        end.

  Returns:
      np.ndarray: Array of frames with shape [n_frames, height, width, 3]
  """
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise ValueError(f"Could not open video file: {video_path}")

  fps = cap.get(cv2.CAP_PROP_FPS)
  total_frames = get_precise_frame_count_ffprobe(video_path)

  # Convert time to frame indices if provided
  start_frame = 0
  end_frame = total_frames - 1

  if start_time is not None:
    start_frame = int(start_time * fps)
  if end_time is not None:
    end_frame = int(end_time * fps)

  if start_frame < 0:
    start_frame = 0
  if end_frame >= total_frames:
    end_frame = total_frames - 1
  if end_frame < start_frame:
    raise ValueError(
        f"End frame {end_frame} is before start frame {start_frame}"
    )

  # Calculate frame indices to sample at equal intervals within the specified range
  if n_frames == 1:
    frame_indices = [(start_frame + end_frame) // 2]
  else:
    frame_indices = np.linspace(start_frame, end_frame, n_frames, dtype=int)

  if len(frame_indices) < n_frames:
    raise ValueError(
        f"Video segment has fewer frames ({len(frame_indices)}) than requested"
        f" ({n_frames})"
    )

  frames = []
  for frame_idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
      raise ValueError(f"Could not read frame {frame_idx} from video")
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

  cap.release()
  return np.stack(frames, axis=0)  # shape: [n_frames, height, width, 3]


def poses_to_scale(poses, convention="w2c"):
  """Compute a global scale from camera centers.

  Args:
      poses: np.ndarray with shape (N, 3, 4) or (N, 4, 4). - w2c: x_cam = R
        x_world + t - c2w: x_world = R x_cam + t
      convention: "w2c" or "c2w"

  Returns:
      float: max(||camera_center_world||) over all N poses.
  """
  poses = np.asarray(poses)
  if poses.ndim != 3 or poses.shape[0] == 0:
    raise ValueError("poses must be (N,3,4) or (N,4,4) with N>0")

  if poses.shape[1:] == (3, 4):
    R = poses[:, :, :3]
    t = poses[:, :, 3]
  elif poses.shape[1:] == (4, 4):
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
  else:
    raise ValueError("Unsupported pose shape: expected (N,3,4) or (N,4,4)")

  if convention.lower() == "w2c":
    # camera center in world coords: C = -R^T t
    centers = -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)
  elif convention.lower() == "c2w":
    # camera center in world coords is the translation directly
    centers = t
  else:
    raise ValueError("convention must be 'w2c' or 'c2w'")

  return float(np.linalg.norm(centers, axis=1).max())


def get_gravity_in_camera(quat):
  """Return gravity vector (3D) in camera coordinates."""
  R_cw = quat_to_R(quat)
  g_w = np.array([0, 0, -1.0])  # gravity in world frame
  g_c = R_cw.T @ g_w  # gravity in camera frame
  return g_c  # [gx, gy, gz]


def transform_camera_pose(cam_traj, encode_pose):
  if encode_pose == 0:
    return cam_traj
  elif encode_pose in [1, 2, 3]:
    ref_frame_idx = len(cam_traj) // 2
    cam_traj = absolute_to_relative(cam_traj, ref_frame_idx=ref_frame_idx)
    if encode_pose == 2:
      cam_traj = pose7_to_pose9(cam_traj)
    elif encode_pose == 3:
      cam_traj = pose7_to_pose12(cam_traj)
  elif encode_pose == 4:
    cam_traj = absolute_to_relative_sequential(cam_traj)
  elif encode_pose == 5:
    cam_traj1 = absolute_to_relative_sequential(cam_traj)
    cam_traj2 = absolute_to_relative(cam_traj, ref_frame_idx=len(cam_traj) // 2)
    cam_traj = np.hstack([cam_traj1, cam_traj2])
  elif encode_pose == 6:
    ref_frame_idx = np.random.randint(0, len(cam_traj))
    cam_traj = absolute_to_relative(cam_traj, ref_frame_idx=ref_frame_idx)
  elif encode_pose == 7:
    cam_traj = cam_traj[:, :3]
  elif encode_pose == 8:
    cam_traj = pose7_to_pose9(cam_traj)
  elif encode_pose in [9, 10, 11]:
    ref_frame = cam_traj[len(cam_traj) // 2]
    cam_traj = absolute_to_relative(cam_traj, ref_frame_idx=len(cam_traj) // 2)
    cam_traj = pose7_to_pose9(cam_traj)
    quat = ref_frame[3:]
    vec = quat if encode_pose == 9 else get_gravity_in_camera(quat)
    cam_traj = np.hstack([cam_traj, np.tile(vec, (cam_traj.shape[0], 1))])
  else:
    raise ValueError(f"Unknown encode_pose {encode_pose}")
  return cam_traj
