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

from collections import defaultdict
import glob
import json
import os


def get_html_header(title, video_width, extra_css=''):
  return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    background-color: #f0f0f0;
                    margin: 20px;
                    font-family: Arial, sans-serif;
                }}
                .video-grid {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: flex-start;
                }}
                .video-container {{
                    width: {video_width}%;
                    margin-bottom: 20px;
                }}
                .video-container video {{
                    width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .video-label {{
                    margin-top: 8px;
                    font-size: 14px;
                    color: #333;
                    word-wrap: break-word;
                }}
                {extra_css}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="video-grid">
    """


def get_html_footer():
  return """
            </div>
        </body>
    </html>
    """


def get_default_extra_css():
  return ''  # No extra CSS for the default gallery


def get_camerabench_extra_css():
  return """
                .video-labels {
                    margin-top: 8px;
                    font-size: 13px;
                    color: #555;
                    word-wrap: break-word;
                }
                .video-caption {
                    margin-top: 6px;
                    font-size: 14px;
                    color: #222;
                    background: #fff;
                    border-radius: 6px;
                    padding: 8px;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
                }
    """


def create_video_gallery(viz_dir, videos_per_row=2):
  """Create an HTML file displaying MP4 videos from the visualization directory in a grid layout.

  Args:
      viz_dir (str): Directory containing MP4 files
      videos_per_row (int): Number of videos to display in each row
  """
  # Find all MP4 files recursively
  mp4_files = glob.glob(os.path.join(viz_dir, '**/*.mp4'), recursive=True)

  if not mp4_files:
    print(f'No MP4 files found in {viz_dir}')
    return

  # Calculate width percentage based on videos per row
  video_width = 100 // videos_per_row - 2  # 2% margin

  html_content = get_html_header(
      'Video Gallery', video_width, get_default_extra_css()
  )

  # Add video elements
  for video_path in mp4_files:
    relative_path = os.path.relpath(video_path, viz_dir)
    text_display = relative_path.split('/')[-2].replace('_', ' ')
    html_content += f"""
                <div class="video-container">
                    <video controls>
                        <source src="{relative_path}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <div class="video-label">{text_display}</div>
                </div>
    """

  html_content += get_html_footer()

  # Write the HTML file
  output_path = os.path.join(viz_dir, 'video_gallery.html')
  with open(output_path, 'w') as f:
    f.write(html_content)

  print(f'Created video gallery at: {output_path}')


def create_video_gallery_camerabench(json_path, output_html, videos_per_row=2):
  """Create an HTML file displaying CameraBench videos with labels and gt_caption from a JSON file.

  Args:
      json_path (str): Path to the CameraBench JSON file
      output_html (str, optional): Path to output HTML file. Defaults to same
        dir as json_path.
      videos_per_row (int): Number of videos to display in each row
  """
  with open(json_path, 'r') as f:
    data = [json.loads(line) for line in f]

  caption_dict = {}
  with open(
      json_path.replace(
          'test.jsonl', 'output/test_vlm_b/gemini-2.5-pro_8frames.jsonl'
      ),
      'r',
  ) as f:
    for line in f:
      line = json.loads(line)
      caption_dict[line['idx']] = line['response']

  response_dir = json_path.replace('test.jsonl', 'output/test_vlm')
  response_files = glob.glob(os.path.join(response_dir, '*.jsonl'))
  response_dict = defaultdict(dict)
  for response_file in response_files:
    with open(response_file, 'r') as f:
      base_name = os.path.basename(response_file).replace('.jsonl', '')
      for line in f:
        line = json.loads(line)
        response_dict[line['idx']][base_name] = line['response']

  # Calculate width percentage based on videos per row
  video_width = 100 // videos_per_row - 2  # 2% margin

  html_content = get_html_header(
      'CameraBench Video Gallery', video_width, get_camerabench_extra_css()
  )

  for i, entry in enumerate(data):
    video_path = os.path.join('data/CameraBench', entry['path'])
    labels = entry.get('labels', [])
    cam_caption = entry.get('caption', '')
    # Use only the filename for src, assuming videos are in the same dir or adjust as needed
    video_src = os.path.relpath(video_path, os.path.dirname(output_html))
    labels_str = ', '.join(labels)
    response_items = [
        f'<div class="video-caption"><b>Responses ({k}):</b> {v}</div>'
        for k, v in response_dict[f'{i:04d}'].items()
    ]
    response_str = '\n'.join(response_items)
    caption = caption_dict[f'{i:04d}'] if f'{i:04d}' in caption_dict else ''
    html_content += f"""
                <div class="video-container">
                    <video controls>
                        <source src="{video_src}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <div class="video-labels"><b>Labels:</b> {labels_str}</div>
                    <div class="video-caption"><b>GT Camera Caption:</b> {cam_caption}</div>
                    {response_str}
                    <div class="video-caption"><b>Gemini Caption:</b> {caption}</div>
                </div>
        """
    if i > 100:
      break

  html_content += get_html_footer()

  os.makedirs(os.path.dirname(output_html), exist_ok=True)
  with open(output_html, 'w') as f:
    f.write(html_content)
  print(f'Created CameraBench video gallery at: {output_html}')


def create_video_gallery_dynpose(json_path, output_html, videos_per_row=2):
  with open(json_path, 'r') as f:
    data = json.load(f)
  response_dir = json_path.replace('input', 'output').replace('.json', '')
  response_files = glob.glob(os.path.join(response_dir, '*.jsonl'))
  response_dict = defaultdict(dict)
  for response_file in response_files:
    with open(response_file, 'r') as f:
      base_name = os.path.basename(response_file).replace('.jsonl', '')
      for line in f:
        line = json.loads(line)
        response_dict[line['idx']][base_name] = line['response']

  # Calculate width percentage based on videos per row
  video_width = 100 // videos_per_row - 2  # 2% margin

  html_content = get_html_header(
      'Dynpose Video Gallery', video_width, get_camerabench_extra_css()
  )

  for i, entry in enumerate(data):
    video_path = entry['video_path']
    # Use only the filename for src, assuming videos are in the same dir or adjust as needed
    video_src = os.path.relpath(video_path, os.path.dirname(output_html))
    if 'vtalign' in json_path:
      if len(response_dict[f'{i:04d}']) == 0:
        continue
    response_items = [
        f'<div class="video-caption"><b>Responses ({k}):</b> {v}</div>'
        for k, v in response_dict[f'{i:04d}'].items()
    ]
    response_str = '\n'.join(response_items)
    if 'vtalign' in json_path:
      caption = (
          entry['query']
          .replace('You are given a video and the caption:\n', '')
          .split('Rate how well the caption is reflected in the video')[0]
      )
      response_str = (
          '<div class="video-caption"><b>Caption:</b>'
          f' {caption.strip()}</div>\n'
          + response_str
      )
    html_content += f"""
                <div class="video-container">
                    <video controls>
                        <source src="{video_src}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    {response_str}
                </div>
        """
  html_content += get_html_footer()
  os.makedirs(os.path.dirname(output_html), exist_ok=True)
  with open(output_html, 'w') as f:
    f.write(html_content)
  print(f'Created Dynpose video gallery at: {output_html}')


if __name__ == '__main__':
  # Example usage
  # viz_dir = "./viz/egoexo4d_pretrain/val"  # Update this path as needed
  # create_video_gallery_camerabench('data/CameraBench/test.jsonl', './viz/camerabench/test.html', videos_per_row=2)
  # create_video_gallery_dynpose('local_data/dynpose-100k/dynpose_100k/input/cmlabel_query0.json', './viz/dynpose/cmlabel_query0.html', videos_per_row=2)
  create_video_gallery_dynpose(
      'data/dynpose-100k/dynpose_100k/vlm_queries/input/vtalign.json',
      './viz/dynpose/vt_align.html',
  )
