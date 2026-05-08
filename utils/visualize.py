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
from math import radians, tan
import os
import shutil
import tempfile
from typing import Tuple
from datasets.action_dataset import UCF101VideoAndCamPoseSeq
from datasets.demo_data import DemoVideoAndCamTextPair
from datasets.dynpose import DynPoseVideoAndCamTextPair
from datasets.egoexo4d import EgoExo4DProficiencyVideoAndCameraPoseLongSeq, EgoExo4DVideoAndCameraPoseLongSeq, EgoExo4DVideoAndCameraPoseSeqForAction, EgoExo4DVideoAndCameraPoseSeqForPretraining
from datasets.hdepic import HdEpicVideoAndCameraPoseSeq
from datasets.nymeria import NymeriaVideoAndCamMotionTextPair
from datasets.viz_dataset import VizVideoAndMegaSAMPose, VizVideoAndPi3Pose, VizVideoAndVipePose, VizVipeResultDir
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import mediapy
import mpl_toolkits.mplot3d.art3d
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio
import tqdm
from utils.visualize_pose import (
    cam2world,
    get_camera_mesh,
    get_xyz_indicators,
    merge_meshes,
    merge_wireframes_plotly,
    merge_xyz_indicators_plotly,
    unbind_np,
)


def plotly_visualize_pose(
    poses,
    vis_depth=1,
    xyz_length=0.2,
    center_size=0.01,
    xyz_width=2,
    mesh_opacity=0.05,
    viz_dir=None,
    filename='pose.png',
):
  """Create plotly visualization traces for camera poses and save as PNG if viz_dir is provided.

  Args:
      poses: Camera poses to visualize [N,3,4]
      vis_depth: Size of camera frustum visualization
      xyz_length: Length of coordinate axis indicators
      center_size: Size of camera center markers
      xyz_width: Width of coordinate axis lines
      mesh_opacity: Opacity of camera frustum mesh
      viz_dir: Directory to save the PNG (optional)
      filename: Name of the PNG file (default: 'pose.png')

  Returns:
      plotly_traces: List of plotly visualization traces
  """
  N = len(poses)
  centers_cam = np.zeros([N, 1, 3])
  centers_world = cam2world(centers_cam, poses)
  centers_world = centers_world[:, 0]
  # Get the camera wireframes.
  vertices, faces, wireframe = get_camera_mesh(poses, depth=vis_depth)
  xyz = get_xyz_indicators(poses, length=xyz_length)
  vertices_merged, faces_merged = merge_meshes(vertices, faces)
  wireframe_merged = merge_wireframes_plotly(wireframe)
  xyz_merged = merge_xyz_indicators_plotly(xyz)
  # Break up (x,y,z) coordinates.
  wireframe_x, wireframe_y, wireframe_z = unbind_np(wireframe_merged, axis=-1)
  xyz_x, xyz_y, xyz_z = unbind_np(xyz_merged, axis=-1)
  centers_x, centers_y, centers_z = unbind_np(centers_world, axis=-1)
  vertices_x, vertices_y, vertices_z = unbind_np(vertices_merged, axis=-1)
  # Set the color map for the camera trajectory and the xyz indicators.base_cmap = plt.get_cmap("cool")

  color_map = plt.get_cmap(
      'cool'
  )  # gnuplot2("gist_rainbow")  # red -> yellow -> green -> blue -> purple
  center_color = []
  faces_merged_color = []
  wireframe_color = []
  xyz_color = []
  x_color, y_color, z_color = (*np.eye(3).T,)
  for i in range(N):
    r, g, b, _ = color_map(i / (N - 1))
    rgb = np.array([r, g, b]) * 0.8
    wireframe_color += [rgb] * 11
    center_color += [rgb]
    faces_merged_color += [rgb] * 6
    xyz_color += [x_color] * 3 + [y_color] * 3 + [z_color] * 3
  # Plot in plotly.
  plotly_traces = [
      go.Scatter3d(
          x=wireframe_x,
          y=wireframe_y,
          z=wireframe_z,
          mode='lines',
          line=dict(color=wireframe_color, width=1),
      ),
      go.Scatter3d(
          x=xyz_x,
          y=xyz_y,
          z=xyz_z,
          mode='lines',
          line=dict(color=xyz_color, width=xyz_width),
      ),
      go.Scatter3d(
          x=centers_x,
          y=centers_y,
          z=centers_z,
          mode='markers',
          marker=dict(color=center_color, size=center_size, opacity=1),
      ),
      go.Mesh3d(
          x=vertices_x,
          y=vertices_y,
          z=vertices_z,
          i=[f[0] for f in faces_merged],
          j=[f[1] for f in faces_merged],
          k=[f[2] for f in faces_merged],
          facecolor=faces_merged_color,
          opacity=mesh_opacity,
      ),
  ]

  # Save as PNG if viz_dir is provided
  if viz_dir is not None:
    os.makedirs(viz_dir, exist_ok=True)
    layout2 = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            dragmode='orbit',
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='data',
        ),
        height=400,
        width=600,
        showlegend=False,
    )
    fig = go.Figure(data=plotly_traces, layout=layout2)
    # fig.write_image(save_path)
    # print(f"Saved plotly pose visualization to {save_path}")
    save_path = os.path.join(viz_dir, filename.replace('png', 'html'))
    # fig.write_html(save_path)
    # print(f"Saved plotly pose visualization to {save_path}")
    html = fig.to_html(
        include_plotlyjs='cdn',
        full_html=True,
        post_script="""
        const gd = document.querySelector('.js-plotly-plot');
        // restore saved camera on load
        try {
        const saved = localStorage.getItem('pose_camera');
        if (saved) {
            Plotly.relayout(gd, {'scene.camera': JSON.parse(saved)});
            console.log('restored camera:', JSON.parse(saved));
        }
        } catch(e) { console.warn('camera restore failed', e); }

        // save camera whenever the view changes
        gd.on('plotly_relayout', ev => {
        if (ev['scene.camera']) {
            localStorage.setItem('pose_camera', JSON.stringify(ev['scene.camera']));
            console.log('camera saved:', ev['scene.camera']);
        }
        });
        """,
    )
    with open(save_path, 'w') as f:
      f.write(html)


def plotly_visualize_pose_seq(
    poses,
    video_path=None,
    fps=10,
    vis_depth=1,
    xyz_length=0.2,
    center_size=0.01,
    xyz_width=2,
    mesh_opacity=0.05,
    return_frames=False,
    frame_size=None,
    cam_params=None,
):
  """Visualize a sequence of camera poses as a video using plotly.

  Args:
      poses: Camera poses to visualize [N,3,4], where N is the time dimension.
      video_path: Path to save the output mp4 video. If None and return_frames
        is False, does nothing.
      fps: Frames per second for the video.
      vis_depth: Size of camera frustum visualization.
      xyz_length: Length of coordinate axis indicators.
      center_size: Size of camera center markers.
      xyz_width: Width of coordinate axis lines.
      mesh_opacity: Opacity of camera frustum mesh.
      return_frames: If True, return a numpy array of frames instead of saving
        video.
      frame_size: (width, height) tuple for output frames. If None, use Plotly
        default.

  Returns:
      If return_frames: numpy array of frames [N, H, W, 3] (float32, 0-1)
  """
  N = len(poses)
  centers_cam_all = np.zeros([N, 1, 3])
  centers_world_all = cam2world(centers_cam_all, poses)
  centers_world_all = centers_world_all[:, 0]
  x_min, x_max = centers_world_all[:, 0].min(), centers_world_all[:, 0].max()
  y_min, y_max = centers_world_all[:, 1].min(), centers_world_all[:, 1].max()
  z_min, z_max = centers_world_all[:, 2].min(), centers_world_all[:, 2].max()
  padding = 2.0
  x_min, x_max = x_min - padding, x_max + padding
  y_min, y_max = y_min - padding, y_max + padding
  z_min, z_max = z_min - padding, z_max + padding

  frames = []
  for t in tqdm.tqdm(range(1, N + 1), desc='Rendering plotly frames'):
    poses_t = poses[:t]
    centers_cam = np.zeros([t, 1, 3])
    centers_world = cam2world(centers_cam, poses_t)
    centers_world = centers_world[:, 0]
    vertices, faces, wireframe = get_camera_mesh(poses_t, depth=vis_depth)
    xyz = get_xyz_indicators(poses_t, length=xyz_length)
    vertices_merged, faces_merged = merge_meshes(vertices, faces)
    wireframe_merged = merge_wireframes_plotly(wireframe)
    xyz_merged = merge_xyz_indicators_plotly(xyz)
    wireframe_x, wireframe_y, wireframe_z = unbind_np(wireframe_merged, axis=-1)
    xyz_x, xyz_y, xyz_z = unbind_np(xyz_merged, axis=-1)
    centers_x, centers_y, centers_z = unbind_np(centers_world, axis=-1)
    vertices_x, vertices_y, vertices_z = unbind_np(vertices_merged, axis=-1)
    color_map = plt.get_cmap('cool')
    center_color = []
    faces_merged_color = []
    wireframe_color = []
    xyz_color = []
    x_color, y_color, z_color = (*np.eye(3).T,)
    for i in range(t):
      r, g, b, _ = color_map(i / max(1, t - 1))
      rgb = np.array([r, g, b]) * 0.8
      wireframe_color += [rgb] * 11
      center_color += [rgb]
      faces_merged_color += [rgb] * 6
      xyz_color += [x_color] * 3 + [y_color] * 3 + [z_color] * 3
    plotly_traces = [
        go.Scatter3d(
            x=wireframe_x,
            y=wireframe_y,
            z=wireframe_z,
            mode='lines',
            line=dict(color=wireframe_color, width=1),
        ),
        go.Scatter3d(
            x=xyz_x,
            y=xyz_y,
            z=xyz_z,
            mode='lines',
            line=dict(color=xyz_color, width=xyz_width),
        ),
        go.Scatter3d(
            x=centers_x,
            y=centers_y,
            z=centers_z,
            mode='markers',
            marker=dict(color=center_color, size=center_size, opacity=1),
        ),
        go.Mesh3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            i=[f[0] for f in faces_merged],
            j=[f[1] for f in faces_merged],
            k=[f[2] for f in faces_merged],
            facecolor=faces_merged_color,
            opacity=mesh_opacity,
        ),
    ]
    width, height = frame_size if frame_size is not None else (30, 20)
    layout2 = go.Layout(
        scene=dict(
            xaxis=dict(visible=False, range=[x_min, x_max]),
            yaxis=dict(visible=False, range=[y_min, y_max]),
            zaxis=dict(visible=False, range=[z_min, z_max]),
            dragmode='orbit',
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='cube',
            camera=cam_params,
        ),
        height=height,
        width=width,
        showlegend=False,
    )
    fig = go.Figure(data=plotly_traces, layout=layout2)
    if return_frames:
      # Render to numpy array
      img_bytes = fig.to_image(
          format='png', width=width, height=height, scale=1
      )
      import io
      from PIL import Image

      img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
      frames.append(np.array(img) / 255.0)
    else:
      if video_path is not None:
        if t == 1:
          import tempfile

          temp_dir = tempfile.mkdtemp()
          frame_paths = []
        frame_path = os.path.join(temp_dir, f'frame_{t:04d}.png')
        fig.write_image(frame_path)
        frame_paths.append(frame_path)
  if return_frames:
    return np.stack(frames, axis=0)
  else:
    # Read frames and write video
    frames = [np.array(Image.open(fp).convert('RGB')) for fp in frame_paths]
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    mediapy.write_video(video_path, frames, fps=fps)
    print(f'Saved plotly pose sequence video to {video_path}')
    shutil.rmtree(temp_dir)


def camera_to_world_coordinates(
    camera_to_world: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
  return coordinates @ camera_to_world[:, :3].T + camera_to_world[:, 3]


def texture_to_camera_coordinates(
    intrinsics: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
  inverse_focal_length = 1.0 / intrinsics[0:2]
  principal_point = intrinsics[2:4]
  xy = (coordinates - principal_point) * inverse_focal_length
  z = np.ones_like(xy[..., 0:1])
  return np.concatenate((xy, z), axis=-1)


def draw_frustum(
    camera_to_world: np.ndarray,
    axes: matplotlib.axes.Axes,
    x_limits,
    y_limits,
    z_limits,
    fov_deg: float = 40.0,
    depth_rel: float = 0.04,
    color: tuple = (0.0, 0.0, 0.7),
    linewidth: float = 1.0,
    alpha: float = 1.0,
    aspect: float = 1.0,  # < 1.0 makes it narrower in x, >1.0 wider
):
  """Draw a camera frustum whose size is tied to plot limits so it looks

  consistent across scales. Assumes camera looks along +Z.
  """
  # Scene scale from current limits
  span = np.array([
      x_limits[1] - x_limits[0],
      y_limits[1] - y_limits[0],
      z_limits[1] - z_limits[0],
  ])
  scene_scale = float(np.max(span))
  depth = depth_rel * scene_scale

  half_x = aspect * depth * np.tan(np.radians(fov_deg) / 2.0)
  half_y = depth * np.tan(np.radians(fov_deg) / 2.0)

  corners_cam = np.array([
      [-half_x, half_y, depth],
      [half_x, half_y, depth],
      [half_x, -half_y, depth],
      [-half_x, -half_y, depth],
  ])

  origin = np.array([0.0, 0.0, 0.0], dtype=float)

  # 4 rays + 4 edges
  lines_cam = np.array(
      [[origin, corners_cam[i]] for i in range(4)]
      + [[corners_cam[i], corners_cam[(i + 1) % 4]] for i in range(4)],
      dtype=float,
  )

  # Transform to world coordinates
  lines_world = camera_to_world_coordinates(camera_to_world, lines_cam)

  coll = mpl_toolkits.mplot3d.art3d.Line3DCollection(
      lines_world, colors=[color], linewidths=linewidth
  )
  coll.set_alpha(alpha)
  axes.add_collection3d(coll)


def hide_3d_axes(axes, fig=None, transparent_fig=True):
  # turn off axis artists (ticks, labels, spine/frame)
  axes.set_axis_off()  # works on 3D axes too (sets _axis3don=False)
  axes.grid(False)

  # fully remove pane fills and edges
  for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
    # pane background
    try:
      axis.pane.fill = False
      axis.pane.set_edgecolor((1, 1, 1, 0))
      axis.pane.set_linewidth(0.0)
    except Exception:
      pass
    # axis line (for older/newer mpl variants)
    try:
      axis.line.set_visible(False)
    except Exception:
      pass

  # legacy attributes (harmless if absent)
  for attr in ('w_xaxis', 'w_yaxis', 'w_zaxis'):
    a = getattr(axes, attr, None)
    if a is not None:
      try:
        a.line.set_visible(False)
        a.pane.set_edgecolor((1, 1, 1, 0))
        a.pane.set_linewidth(0.0)
      except Exception:
        pass

  # newer “spines” dict (no-op on some 3D backends)
  for spine in getattr(axes, 'spines', {}).values():
    spine.set_visible(False)

  if fig is not None and transparent_fig:
    fig.patch.set_alpha(0.0)


def hide_3d_axes_new(axes, cord):
  axes.grid(False)
  axes.set_xticks([])
  axes.set_yticks([])
  axes.set_zticks([])

  if cord == '':
    xcolor = 0.98
    ycolor = 0.94
    zcolor = 0.96
  else:
    xcolor = 0.94
    ycolor = 0.96
    zcolor = 0.98

  # axes.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
  # axes.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
  # axes.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

  axes.xaxis.pane.set_facecolor((xcolor, xcolor, xcolor, 0.5))
  axes.yaxis.pane.set_facecolor((ycolor, ycolor, ycolor, 0.5))
  axes.zaxis.pane.set_facecolor((zcolor, zcolor, zcolor, 0.5))

  axes.xaxis.pane.set_edgecolor('none')
  axes.yaxis.pane.set_edgecolor('none')
  axes.zaxis.pane.set_edgecolor('none')


def plot_camera(
    cameras_to_world: np.ndarray,
    intrinsics: np.ndarray,
    fig_size: tuple = (4.48, 4.48),
    max_frames: int = 1_000_000,
    cord: str = '',
    no_background=False,
    resolution=100,
) -> np.ndarray:
  """Produce video showing camera frustums with a light→dark history colormap.

  The frustum size adapts to axis limits, so it looks consistent across scales.
  intrinsics is kept for API compatibility but not used.

  Requires draw_frustum(camera_to_world, axes, x_limits, y_limits, z_limits,
                       fov_deg=..., depth_rel=..., color=(r,g,b),
                       linewidth=..., alpha=...)
  """
  # ---- settings for frustum history visualization ----
  history_len = 1  # number of previous frustums to retain
  hist_stride = (
      1  # 4,8        # draw every k-th historical frustum to save time
  )
  cmap_name = 'cool'  # "Blues"      # single-hue map, light→dark
  base_alpha = 0.9  # overall opacity cap for frustums
  lw_old, lw_new = 0.6, 1.2
  fov_deg = 45  # compact frustum
  depth_rel = 0.35  # size relative to scene span
  aspect = 1.0
  padding = 0  # 0.05
  min_range = 0  # 0.05
  if cord == '':
    depth_rel = 0.35  # 0.4
    padding = 0.01
    min_range = 0.01

  # prebuild colormap
  cmap = cm.get_cmap(cmap_name)

  def color_at(rank: int, total: int):
    # rank: 0 = oldest, total-1 = newest
    if total <= 1:
      v = 0.9
    else:
      v = 0.25 + 0.7 * (rank / (total - 1))  # avoid extremes -> better contrast
    r, g, b, _ = cmap(v)
    # alpha & linewidth ramp up from old→new
    if total > 1:
      lw = lw_old + (lw_new - lw_old) * (rank / (total - 1))
      a = base_alpha * (0.5 + 0.5 * (rank / (total - 1)))  # 0.5→1.0
    else:
      lw, a = lw_new, base_alpha
    return (r, g, b), a, lw

  # ---- axis limits from full trajectory ----
  camera_positions = cameras_to_world[:, :, 3]  # [T, 3]
  x_min, x_max = (
      camera_positions[:, 0].min() - padding,
      camera_positions[:, 0].max() + padding,
  )
  y_min, y_max = (
      camera_positions[:, 1].min() - padding,
      camera_positions[:, 1].max() + padding,
  )
  z_min, z_max = (
      camera_positions[:, 2].min() - padding,
      camera_positions[:, 2].max() + padding,
  )

  if x_max - x_min < min_range:
    c = (x_max + x_min) / 2
    x_min, x_max = c - min_range / 2, c + min_range / 2
  if y_max - y_min < min_range:
    c = (y_max + y_min) / 2
    y_min, y_max = c - min_range / 2, c + min_range / 2
  if z_max - z_min < min_range:
    c = (z_max + z_min) / 2
    z_min, z_max = c - min_range / 2, c + min_range / 2

  x_limits, y_limits, z_limits = [x_min, x_max], [y_min, y_max], [z_min, z_max]

  c = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
  r = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
  x_limits = [c[0] - r, c[0] + r]
  y_limits = [c[1] - r, c[1] + r]
  z_limits = [c[2] - r, c[2] + r]
  print('x_limits, y_limits, z_limits:', x_limits, y_limits, z_limits)
  # z_limits = [2, 12]
  frames = []
  camera_path_ahead = 0  # 20
  sequence_length = cameras_to_world.shape[0]

  for t in tqdm.tqdm(range(min(sequence_length, max_frames))):
    fig = matplotlib.figure.Figure(figsize=fig_size, dpi=resolution)
    canvas = FigureCanvasAgg(fig)
    axes = fig.add_subplot(111, projection='3d', computed_zorder=False)

    axes.set_xlim(x_limits)
    axes.set_ylim(y_limits)
    axes.set_zlim(z_limits)
    # axes.set_xlabel('X')
    # axes.set_ylabel('Y')
    # axes.set_zlabel('Z')

    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.set_zticklabels([])

    axes.xaxis.pane.set_facecolor('white')
    axes.yaxis.pane.set_facecolor('white')
    axes.zaxis.pane.set_facecolor('white')

    if no_background:
      # hide_3d_axes(axes, fig)
      hide_3d_axes_new(axes, cord)
    # coordinate conventions
    if cord == 'opencv':  # x right, y down, z forward
      axes.invert_zaxis()
      axes.invert_yaxis()
      axes.view_init(azim=45, vertical_axis='y')  # 30
    elif cord == 'aria_rel':
      axes.invert_zaxis()
      axes.invert_xaxis()
      axes.view_init(azim=-60, vertical_axis='x')
    else:  # aria absolute
      axes.view_init(azim=45)

    # camera path (optional)
    path_end = min(t + camera_path_ahead, sequence_length)
    camera_path = cameras_to_world[0:path_end, :, 3]
    axes.plot(
        camera_path[..., 0],
        camera_path[..., 1],
        camera_path[..., 2],
        color='#636363',
        linestyle='dashed',
    )  # '#08306b'

    # draw historical frustums with light→dark color ramp
    start_idx = max(0, t - history_len + 1)
    past_ids = list(range(start_idx, t + 1, hist_stride))
    total = len(past_ids)
    for rank, idx in enumerate(past_ids):
      rgb, a, lw = color_at(rank, total)
      a = 1.0
      draw_frustum(
          cameras_to_world[idx],
          axes,
          x_limits,
          y_limits,
          z_limits,
          fov_deg=fov_deg,
          depth_rel=depth_rel,
          color=rgb,
          linewidth=lw,
          alpha=a,
          aspect=aspect,
      )

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    # blend trick (kept from your original)
    fig.canvas.draw()
    with_background = np.array(canvas.buffer_rgba(), dtype=np.float32) / 255.0
    fig.canvas.draw()
    without_background = (
        np.array(canvas.buffer_rgba(), dtype=np.float32) / 255.0
    )
    k = 0.1
    blend = with_background * k + without_background * (1 - k)
    frames.append(blend)

    plt.close(fig)

  return np.array(frames)[..., :3]


def plot_two_cameras(
    cameras_to_world1: np.ndarray,
    cameras_to_world2: np.ndarray,
    intrinsics: np.ndarray,
    fig_size: tuple = (4.48, 4.48),
    max_frames: int = 1000000,
    cord: str = '',
) -> np.ndarray:
  """Produce video showing two camera frustums and their trajectories.

  Args:
      cameras_to_world1: array [t, 3, 4] of camera-to-world tranforms for camera
        1.
      cameras_to_world2: array [t, 3, 4] of camera-to-world tranforms for camera
        2.
      intrinsics: [fx, fy, px, py] Focal length and principal point in
        normalized image coordinates.
      max_frames: Only generate at most this many frames of video.

  Returns:
      ndarray of t video frames.
  """
  # Get camera positions across entire sequence for both cameras
  camera_positions1 = cameras_to_world1[:, :, 3]
  camera_positions2 = cameras_to_world2[:, :, 3]
  all_camera_positions = np.concatenate(
      [camera_positions1, camera_positions2], axis=0
  )

  # Compute min/max for each axis with some padding
  padding = 0.5
  x_min, x_max = (
      all_camera_positions[:, 0].min() - padding,
      all_camera_positions[:, 0].max() + padding,
  )
  y_min, y_max = (
      all_camera_positions[:, 1].min() - padding,
      all_camera_positions[:, 1].max() + padding,
  )
  z_min, z_max = (
      all_camera_positions[:, 2].min() - padding,
      all_camera_positions[:, 2].max() + padding,
  )

  # Ensure minimum range for very small movements
  min_range = 0.5
  if x_max - x_min < min_range:
    center = (x_max + x_min) / 2
    x_min, x_max = center - min_range / 2, center + min_range / 2
  if y_max - y_min < min_range:
    center = (y_max + y_min) / 2
    y_min, y_max = center - min_range / 2, center + min_range / 2
  if z_max - z_min < min_range:
    center = (z_max + z_min) / 2
    z_min, z_max = center - min_range / 2, center + min_range / 2
  x_limits, y_limits, z_limits = [x_min, x_max], [y_min, y_max], [z_min, z_max]

  frames = []
  camera_path_ahead = 20
  sequence_length = min(cameras_to_world1.shape[0], cameras_to_world2.shape[0])

  for t in tqdm.tqdm(range(min(sequence_length, max_frames))):
    fig = matplotlib.figure.Figure(figsize=(fig_size[0], fig_size[1]), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axes = fig.add_subplot(111, projection='3d', computed_zorder=False)

    axes.set_xlim(x_limits)
    axes.set_ylim(y_limits)
    axes.set_zlim(z_limits)

    if cord == 'opencv':
      axes.invert_zaxis()
      axes.invert_yaxis()

      axes.view_init(azim=30, vertical_axis='y')
    elif cord == 'aria_rel':
      axes.invert_zaxis()
      axes.invert_xaxis()
      axes.view_init(azim=-60, vertical_axis='x')
    else:  # aria absolute
      axes.view_init(azim=45)

    # Camera 1 trajectory (blue)
    camera_path1 = cameras_to_world1[
        0 : min(t + camera_path_ahead, sequence_length), :, 3
    ]
    axes.plot(
        camera_path1[..., 0],
        camera_path1[..., 1],
        camera_path1[..., 2],
        color=(0.0, 0.0, 0.7),
        linestyle='dashed',
    )
    draw_frustum(cameras_to_world1[t], intrinsics, axes, color=(0.0, 0.0, 0.7))

    # Camera 2 trajectory (red)
    camera_path2 = cameras_to_world2[
        0 : min(t + camera_path_ahead, sequence_length), :, 3
    ]
    axes.plot(
        camera_path2[..., 0],
        camera_path2[..., 1],
        camera_path2[..., 2],
        color=(0.7, 0.0, 0.0),
        linestyle='dashed',
    )
    draw_frustum(cameras_to_world2[t], intrinsics, axes, color=(0.7, 0.0, 0.0))

    fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
    fig.canvas.draw()
    frame = np.array(canvas.buffer_rgba(), dtype=np.float32) / 255.0
    frames.append(frame)
    plt.close(fig)

  return np.array(frames)[..., :3]


def visualize_camera_motion(
    data,
    viz_dir,
    fps,
    cord,
    save_name='',
    intrinsics=None,
    no_background=False,
    ratio=1.0,
    camera_only=False,
):
  if intrinsics is None:
    intrinsics = np.array([0.8660254, 0.8660254, 0.5, 0.5])
  frames, cam2world, id, label, label_name = data
  # cam2world[..., -1] = cam2world[..., -1] * np.array([1,20,1])[None, ]
  print(id, label, label_name, frames.shape, cam2world.shape)

  if label != '' and label_name != '':
    save_dir = os.path.join(viz_dir, f"{label}_{label_name.replace(' ', '_')}")
    save_name = os.path.join(save_dir, save_name + str(id) + '.mp4')
  else:
    save_name = f'{viz_dir}/{save_name}{id}.mp4'

  os.makedirs(os.path.dirname(save_name), exist_ok=True)
  # if os.path.exists(save_name):
  #     return
  resolution = 100 if 'egoexo4d' in viz_dir else 200
  resolution = 300 if 'demo' in viz_dir else resolution
  fig_size = (frames.shape[2] / resolution, frames.shape[1] / resolution)
  print('Fig size:', fig_size)
  if camera_only:
    fig_size = [5, 5]
    resolution = 500

  # video_camera = plot_camera(cam2world, intrinsics, fig_size=fig_size, cord=cord, no_background=no_background, resolution=resolution)
  # video_camera_wo_bg = plot_camera(cam2world, intrinsics, fig_size=fig_size, cord=cord, no_background=True, resolution=resolution)
  video_camera_w_bg = plot_camera(
      cam2world,
      intrinsics,
      fig_size=fig_size,
      cord=cord,
      no_background=False,
      resolution=resolution,
  )
  # video_camera = 0.25 * video_camera_w_bg + 0.75 * video_camera_wo_bg
  video_camera = video_camera_w_bg
  T, H, W, C = video_camera.shape
  crop_w = int(W * ratio)
  start = (W - crop_w) // 2
  end = start + crop_w
  video_camera = video_camera[:, :, start:end, :]

  if camera_only:
    mediapy.write_video(
        save_name.replace('.mp4', '_camera.mp4'), video_camera, fps=fps
    )
  else:
    side_by_side = np.concatenate([frames / 255.0, video_camera], axis=2)
    mediapy.write_video(save_name, side_by_side, fps=fps)
  print(f'Saved to {save_name}')


def visualize_camera_motion_b(
    frames, cam2world, save_video_path, fps, sample_rate, cam_params=None
):
  """Like visualize_camera_motion, but uses plotly_visualize_pose_seq (Plotly) for the camera motion plot, in-memory."""

  # Use plotly_visualize_pose_seq to get frames in memory, matching the original video frame size
  frames = frames[::sample_rate]
  cam2world = cam2world[::sample_rate]
  frame_size = (frames.shape[2], frames.shape[1])  # (width, height)
  frames_plotly = plotly_visualize_pose_seq(
      cam2world,
      return_frames=True,
      frame_size=frame_size,
      fps=fps,
      cam_params=cam_params,
  )
  frames_orig = frames / 255.0
  min_len = min(len(frames_orig), len(frames_plotly))
  frames_orig = frames_orig[:min_len]
  frames_plotly = frames_plotly[:min_len]
  side_by_side = np.concatenate([frames_orig, frames_plotly], axis=2)
  os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
  mediapy.write_video(save_video_path, side_by_side, fps=fps)
  print(f'Saved to {save_video_path}')


def viz_egoexo4d_scenario():
  viz_dir = './viz/egoexo4d_scenario/'
  os.makedirs(viz_dir, exist_ok=True)
  dataset = EgoExo4DProficiencyVideoAndCameraPoseLongSeq()
  for i in range(10):
    idx = np.random.randint(len(dataset))
    # idx = 23
    data = dataset[idx]
    visualize_camera_motion(
        data, viz_dir=viz_dir, fps=20, save_name=f'{idx}_', cord=''
    )


def viz_egoexo4d_action():
  viz_dir = './viz/egoexo4d_action/'
  os.makedirs(viz_dir, exist_ok=True)
  for label_id in [100]:  # [0, 15, 30, 36, 145, 251]:
    dataset = EgoExo4DVideoAndCameraPoseSeqForAction(
        label_id, use_relative=False
    )
    for i in range(1):
      idx = np.random.randint(len(dataset))
      idx = 35
      data = dataset[idx]
      # print(i, idx, data[-1])
      visualize_camera_motion(
          data, viz_dir=viz_dir, fps=10, save_name=f'{idx}_', cord=''
      )


def viz_egoexo4d_pretrain(random=False, ego_visible=False):
  # np.random.seed(0)
  good_idx = {
      True: [
          27635,
          28970,
          31965,
          31145,
          4125,
          26040,
          24850,
          34670,
          33050,
          13180,
          1110,
          11555,
          24225,
          15205,
          1720,
          22965,
          20275,
          22960,
          5830,
          25360,
          20030,
          17460,
          23585,
          13215,
          22485,
          15745,
          28080,
          23180,
          23880,
          28690,
          2455,
          35320,
          26025,
          3815,
          22875,
          10315,
          18005,
          33595,
          23420,
          20510,
          32610,
          23925,
          15945,
          5200,
          34915,
          23580,
          33520,
          10825,
          14450,
          27680,
      ],
      False: [
          595,
          1770,
          28475,
          26860,
          8705,
          13075,
          8065,
          25360,
          21970,
          6770,
          1500,
          12830,
          11985,
          16170,
          28465,
          15680,
          9030,
          22315,
          13205,
          20825,
          25105,
          12070,
          3030,
          33990,
          28755,
          33045,
          30715,
          20360,
          23505,
          33060,
          8695,
          27880,
          25725,
          29160,
          28890,
          9855,
          12125,
          34630,
          3615,
          16535,
          27655,
          12635,
          23095,
          28595,
          12470,
          15660,
          33075,
          17930,
          9245,
          18280,
      ],
  }
  dur = 4
  if random:
    viz_dir = f'./viz/egoexo4d_pretrain/success_retrieval/takedur{dur}_egovisible{ego_visible}'
  else:
    viz_dir = (  # v2v_retrieval success_1example embedding
        './viz/egoexo4d_pretrain/1example_new'
    )
  dataset = EgoExo4DVideoAndCameraPoseLongSeq(dur, opencv_cord=False)
  if random:
    idx_list = np.random.choice(len(dataset), size=30, replace=False).tolist()
    idx_list = good_idx[ego_visible]
  else:
    idx_list = [5200]
  if 't2v_retrieval' in viz_dir or 'embedding' in viz_dir:
    viz_dir = os.path.join(viz_dir, str(idx_list[0]))
  os.makedirs(viz_dir, exist_ok=True)
  for idx in idx_list:
    data = dataset[idx]
    # plotly_visualize_pose(data[1][::2], viz_dir=f'{viz_dir}/htmls', filename=f'{idx}_{dur}s.png')
    visualize_camera_motion(
        data, viz_dir=viz_dir, fps=15, save_name=f'{data[3]}_{dur}s_', cord=''
    )  # camera_only=True)fps=10
    # visualize_camera_motion(data, viz_dir=viz_dir, fps=15, save_name=f'opencv_{idx}_{dur}s_', cord='opencv')


def viz_dynpose(random=False):
  # from utils.visualize_pose import plotly_visualize_pose as plotly_visualize_pose_original
  # viz_dir = './viz/dynpose/video_cmp'
  if random:
    viz_dir = f'./viz/dynpose/success_retrieval'
  else:
    viz_dir = (  # success_1exp' v2v_retrieval_baseline/12565
        './viz/dynpose/1example'
    )
  os.makedirs(viz_dir, exist_ok=True)
  dataset = DynPoseVideoAndCamTextPair()
  intrinsics = np.array([0.5, 0.5, 0.5, 0.5])
  if random:
    idx_list = np.random.choice(len(dataset), size=100, replace=False).tolist()
    idx_list = [
        790,
        1120,
        735,
        1960,
        170,
        4100,
        4660,
        1710,
        3070,
        2655,
        2770,
        1410,
        2210,
        3005,
        3715,
        2515,
        110,
        1100,
        1345,
        290,
        2475,
        2555,
        3885,
        2840,
        405,
        1505,
        4250,
        2355,
        2225,
        420,
        4015,
        265,
        3440,
        2365,
        3190,
        1520,
        2405,
        4300,
        3340,
        2285,
        3805,
        1890,
        3020,
        3265,
        2290,
        485,
        1445,
        2050,
        1130,
        2775,
        2960,
        3370,
        4820,
        2810,
        1250,
        35,
        525,
        4805,
        1230,
        2730,
        2080,
        1300,
        2860,
        770,
        4035,
        3745,
        2855,
        4905,
        4985,
        1820,
        2270,
        3160,
        1685,
        1595,
        3595,
        3815,
        1825,
        1255,
        385,
        340,
        1290,
        2870,
        3690,
        990,
        3890,
        55,
        4280,
        670,
        2545,
        3640,
        3960,
        150,
        4720,
        540,
        2490,
        1350,
        2550,
        2575,
        1885,
        3810,
    ]
  else:
    idx_list = [9675]  # [1924, 3029, 6145, 6889, 9405, 2590, 329, 5740, 9489]
  # cam_params = None
  for idx in idx_list:
    data = dataset[idx]
    # visualize_camera_motion(data, viz_dir=viz_dir, fps=12, save_name=f'{idx}_gt_', cord='opencv', ratio=0.6, camera_only=True)
    visualize_camera_motion(
        data,
        viz_dir=viz_dir,
        fps=12,
        save_name=f'{idx}_vipe_',
        cord='opencv',
        ratio=0.6,
    )  # , camera_only=True)
    # frames, cam_gt, cam_vipe, seq = dataset[idx]
    # save_name = os.path.join(viz_dir, f"{idx}_{seq}.mp4")
    # fig_size = (frames.shape[2]/100, frames.shape[1]/100)
    # video_camera = plot_two_cameras(cam_gt, cam_vipe, intrinsics, fig_size=fig_size, cord='opencv')
    # side_by_side = np.concatenate([frames/255.0, video_camera], axis=2)
    # mediapy.write_video(save_name, side_by_side, fps=5)
    # print(f"Saved to {save_name}")

    # plotly_visualize_pose(data[1], viz_dir=os.path.join(viz_dir, 'poses'), filename=f'{idx}.png')
    # plotly_visualize_pose_seq(data[1][::10], video_path=f'{viz_dir}/pose_videos/{i}.mp4', fps=12)
    # save_name = os.path.join(viz_dir, f"{idx}_{data[2]}.mp4")
    # if 'walk' in save_name:
    #     continue
    # if os.path.exists(save_name):
    #     continue
    # visualize_camera_motion_b(data[0], data[1], save_name, 12, 3, cam_params=cam_params)


def viz_nymeria(mode):
  good_idx = {
      'a': [
          3830,
          4225,
          3150,
          580,
          2380,
          1510,
          3200,
          3880,
          2730,
          290,
          220,
          1845,
          1715,
          3510,
          2770,
          4925,
          1660,
          145,
          4835,
          3165,
          3935,
          1120,
          2925,
          1545,
          3005,
          975,
          2820,
          4985,
          1525,
          2280,
          1530,
          3600,
          755,
          3810,
          730,
          1670,
          2890,
          2135,
          570,
          690,
          4220,
          3145,
          90,
          4850,
          390,
          2200,
          1870,
          2105,
          2005,
          840,
          3715,
          3185,
          3620,
          3245,
          4450,
          4065,
          4600,
          3885,
          3595,
          4825,
          750,
          745,
          4820,
          1185,
          565,
          1780,
          3635,
          4720,
          880,
          1110,
          1465,
          2650,
          30,
          4245,
          575,
          495,
          2780,
          3760,
          4005,
          530,
          3590,
          3235,
          4300,
          3320,
          3605,
          2950,
          4830,
          635,
          2530,
          2905,
          3920,
          3405,
          1255,
          2555,
      ],
      'b': [
          830,
          150,
          3995,
          4870,
          875,
          4130,
          4070,
          4415,
          3480,
          3460,
          2740,
          4715,
          290,
          2305,
          4275,
          740,
          4690,
          1975,
          1450,
          2660,
          2255,
          1285,
          4705,
          2205,
          110,
          4615,
          2370,
          40,
          1350,
          825,
          730,
          60,
          2935,
          1760,
          1555,
          2710,
          1855,
          3270,
          3375,
          2095,
          3535,
          4975,
          4880,
          2225,
          4250,
          1590,
          610,
          4885,
          1300,
          1655,
          1030,
          2120,
          3485,
          4495,
          2425,
          1905,
          3195,
          2260,
          3960,
      ],
      'c': [
          4490,
          1370,
          540,
          4925,
          1985,
          3920,
          3520,
          2675,
          1055,
          2840,
          1920,
          3125,
          660,
          680,
          2945,
          470,
          4375,
          2235,
          2405,
          785,
          2250,
          4700,
          810,
          2805,
          650,
          2965,
          3110,
          2690,
          4765,
          1620,
          4305,
          725,
          1210,
          1530,
          4525,
          875,
          3135,
          1910,
          2100,
          2640,
      ],
      'd': [
          4405,
          3915,
          3725,
          295,
          1835,
          3195,
          2225,
          2330,
          4235,
          4450,
          4515,
          2045,
          645,
          3650,
          2040,
          3780,
          4505,
          3840,
          3585,
          1360,
          2755,
          2880,
          3700,
          3465,
          1000,
          4730,
          660,
          1825,
          235,
          2535,
          4850,
          3015,
          520,
          1545,
          3720,
          335,
          4370,
          1715,
          2650,
          680,
          3110,
          3845,
          3250,
          3740,
          1605,
          2975,
          1575,
          1100,
          4330,
          2700,
          945,
          15,
          3380,
          3525,
          580,
          3825,
          3850,
          2360,
          1460,
          2845,
          2870,
          2335,
          1480,
          3300,
          210,
          1125,
          2060,
          180,
          3590,
          2540,
          1680,
          305,
          2625,
          1560,
          930,
          2250,
          3475,
          140,
          3675,
          1905,
          4560,
          560,
          1860,
          3285,
          2140,
          3625,
          4360,
          3745,
          2025,
          2510,
          3685,
          4230,
          4695,
          4660,
          4280,
          2590,
          3950,
          170,
          2410,
          2940,
          1600,
          2355,
          2110,
          3690,
          3400,
          175,
          4855,
          3155,
          1440,
          4130,
          115,
          3735,
          4645,
          2520,
          2885,
          1285,
          3770,
          2970,
          4540,
          2735,
          2030,
          2645,
          770,
          570,
          4260,
          2280,
          2445,
          265,
          4270,
          4600,
          2815,
          1385,
          4390,
          1135,
          4080,
          4630,
          710,
          1245,
          3775,
          3245,
          3710,
          2910,
          1655,
          4710,
          4935,
          1630,
          4395,
          455,
          1335,
          3375,
          2470,
          1765,
          3390,
          1095,
          590,
          1475,
          4210,
          1155,
          60,
          1210,
          2185,
          1570,
          4465,
          2440,
          2275,
          2715,
          400,
          1015,
      ],
  }
  viz_dir = f'./viz/nymeria/val_{mode}'
  os.makedirs(viz_dir, exist_ok=True)
  dataset = NymeriaVideoAndCamMotionTextPair(mode)
  idx_list = [570]
  for idx in good_idx[mode]:  # good_idx[mode]:, idx_list
    data = dataset[idx]
    visualize_camera_motion(
        data, viz_dir=viz_dir, save_name=f'{idx}_', fps=20, cord=''
    )


def viz_hdepic():
  viz_dir = './viz/hd-epic/val_data'
  os.makedirs(viz_dir, exist_ok=True)
  dataset = HdEpicVideoAndCameraPoseSeq()
  visualize_camera_motion(dataset[0], viz_dir=viz_dir)
  # for i in range(10):
  #     idx = np.random.randint(0, len(dataset))
  #     data = dataset[idx]
  #     try:
  #         visualize_camera_motion(data, viz_dir=viz_dir)
  #     except Exception as e:
  #         print(f"Error processing {i}, {idx}: {e}")


def viz_megasam(scale_transform):
  intrinsics = np.array([0.8660254, 0.8660254, 0.5, 0.5])
  np.random.seed(0)
  dataset1 = VizVideoAndMegaSAMPose(scale_transform, True)
  viz_dir = f'./viz/megasam_cmp_new/'
  os.makedirs(viz_dir, exist_ok=True)

  # idx_list = [4, 6, 7, 19, 21, 25]
  for i in range(1):
    # idx = np.random.randint(0, len(dataset1))
    idx = 3415
    frames, cam_gt, cam_est, seq = dataset1[idx]
    save_name = os.path.join(
        viz_dir, f'{idx}_{seq}_scale{scale_transform}_rel.mp4'
    )

    fig_size = (frames.shape[2] / 100, frames.shape[1] / 100)
    video_camera = plot_two_cameras(
        cam_gt, cam_est, intrinsics, fig_size=fig_size, cord='opencv'
    )
    side_by_side = np.concatenate([frames / 255.0, video_camera], axis=2)
    mediapy.write_video(save_name, side_by_side, fps=5)
    print(f'Saved to {save_name}')


def viz_estimation(method):
  intrinsics = np.array([0.8660254, 0.8660254, 0.5, 0.5])
  if method == 'vipe':
    dataset = VizVideoAndVipePose()
  else:
    dataset = VizVideoAndPi3Pose()
  viz_dir = f'./viz/{method}/'
  os.makedirs(viz_dir, exist_ok=True)
  for i in range(5):
    idx = np.random.randint(0, len(dataset))
    frames, cam_gt, cam_est, seq = dataset[idx]
    save_name = os.path.join(viz_dir, f'{idx}_{seq}.mp4')

    fig_size = (frames.shape[2] / 100, frames.shape[1] / 100)
    video_camera = plot_two_cameras(
        cam_gt, cam_est, intrinsics, fig_size=fig_size, cord='opencv'
    )
    side_by_side = np.concatenate([frames / 255.0, video_camera], axis=2)
    mediapy.write_video(save_name, side_by_side, fps=5)
    print(f'Saved to {save_name}')


def viz_vipe_dir():
  intrinsics = np.array([0.8660254, 0.8660254, 0.5, 0.5])
  dataset = VizVipeResultDir()
  viz_dir = './viz/vipe_dir/'
  os.makedirs(viz_dir, exist_ok=True)
  for i in range(30):
    idx = np.random.randint(0, len(dataset))
    frames, cam2world, seq = dataset[idx]
    save_name = os.path.join(viz_dir, f'{idx}_{seq}.mp4')
    fig_size = (frames.shape[2] / 100, frames.shape[1] / 100)
    video_camera = plot_camera(
        cam2world, intrinsics, fig_size=fig_size, cord='opencv'
    )
    side_by_side = np.concatenate([frames / 255.0, video_camera], axis=2)
    mediapy.write_video(save_name, side_by_side, fps=30)


def viz_ucf101():
  viz_dir = './viz/ucf101'
  os.makedirs(viz_dir, exist_ok=True)
  dataset = UCF101VideoAndCamPoseSeq('01', method='megasam')
  intrinsics = np.array([0.5, 0.5, 0.5, 0.5])
  for i in range(10):
    idx = np.random.randint(0, len(dataset))
    frames, pi3_poses, vipe_poses, label = dataset[idx]
    save_name = os.path.join(viz_dir, f'{idx}_{label}_vipe_vs_megasam.mp4')
    fig_size = (frames.shape[2] / 100, frames.shape[1] / 100)
    video_camera = plot_two_cameras(
        pi3_poses, vipe_poses, intrinsics, fig_size=fig_size, cord='opencv'
    )
    side_by_side = np.concatenate([frames / 255.0, video_camera], axis=2)
    mediapy.write_video(save_name, side_by_side, fps=5)
    print(f'Saved to {save_name}')

    # visualize_camera_motion(data, viz_dir=viz_dir, fps=30, save_name=f'{i}_new', cord='opencv')


def viz_demo():
  method = 'vipe'  # 'vipe' or 'pi3'
  viz_dir = f'./viz/demo/{method}pose'
  os.makedirs(viz_dir, exist_ok=True)
  dataset = DemoVideoAndCamTextPair(method)
  fps = 5 if method == 'pi3' else 30
  for idx in [75, 92]:
    data = dataset[idx]
    visualize_camera_motion(
        data, viz_dir=viz_dir, fps=fps, cord='opencv', save_name='rel_new_'
    )


def save_egoexo4d_demo_data():
  save_dir = './tmp/hamer_demo_data'
  os.makedirs(save_dir, exist_ok=True)
  dataset = EgoExo4DVideoAndCameraPoseSeqForPretraining(True)
  for i in range(10):
    idx = np.random.randint(0, len(dataset))
    data = dataset[idx]
    frames, cam2world, id, label, label_name = data
    mid_frame = frames[frames.shape[0] // 2]
    Image.fromarray(mid_frame).save(f'{save_dir}/{id}.png')
    print(i)


def print_text():
  # df = pd.read_csv('data/egoexo4d/annotations/pretraining/test_alltasks_mcqv0.csv')
  # rows = df.iloc[12070:12075]   #[17195:17200]
  # print(rows['description_text'].tolist())

  # df = pd.read_csv('data/nymeria/eval1000/split_by_Describe_my_body_posture_new.csv')
  # rows = df.iloc[390:395]
  # print(rows['Describe my body posture'].tolist())

  # df = pd.read_csv('data/nymeria/eval1000/split_by_Describe_my_legs_feet_motion.csv')
  # rows = df.iloc[3920:3925]
  # print(rows['Describe my legs/feet motion'].tolist())

  # df = pd.read_csv('data/nymeria/eval1000/split_by_Describe_my_hands_arms_motion.csv')
  # rows = df.iloc[3995:4000]
  # print(rows['Describe my hands/arms motion'].tolist())

  df = pd.read_csv(
      'data/nymeria/eval1000/split_by_Describe_my_focus_attention.csv'
  )
  rows = df.iloc[570:575]
  print(rows['Describe my focus attention'].tolist())

  # df = pd.read_csv('data/dynpose-100k/dynpose_100k/metadata_samevideo_test5000.csv')
  # rows = df.iloc[2810:2815] #[540:545], [2365:2370] [3815:3820]
  # print(rows['description_text'].tolist())


if __name__ == '__main__':
  # print_text()
  # viz_hdepic()
  # viz_nymeria('d')
  # viz_egoexo4d_scenario()
  # viz_egoexo4d_pretrain()
  # viz_egoexo4d_action()
  # for ego_visible in [False]:
  #     viz_egoexo4d_pretrain(True, ego_visible)
  viz_dynpose()
  # viz_demo()
  # viz_ucf101()
  # save_egoexo4d_demo_data()
  # for mode in ['a']:
  # viz_nymeria(mode)
  # viz_megasam(True)
  # viz_megasam(False)
  # viz_estimation('pi3')
  # viz_vipe_dir()
  # print_text()
