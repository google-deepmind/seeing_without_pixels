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

from collections import Counter, defaultdict
import glob
import json
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import spacy
from tqdm import tqdm
from utils.read_result import load_mapping


def load_camera_name_mapping(data_file):
  camera_name_mapping = {}
  with open(data_file, 'r') as f:
    data_list = json.load(f)
  for data in data_list:
    keys = list(data['frame_aligned_videos'].keys())
    camera_name_mapping[data['take_name']] = data['frame_aligned_videos'][
        keys[0]
    ]['rgb']['relative_path']
  return camera_name_mapping


def transform_input_file():
  data_file = 'data/egoexo4d/annotations/pretraining/vlm_baseline/input/test_alltasks_mcqv0_sampled_new.json'
  output_file = data_file.replace('_new.json', '_byscenariotype.json')
  take_mapping = load_mapping()
  with open(data_file, 'r') as f:
    data_list = json.load(f)

  save_list = []
  for data in data_list:
    take_name = data['video_path'].split('/')[3]
    scenario = take_mapping[take_name]
    data['q_type'] = (
        f"{data['q_type']}_procedural"
        if scenario in ['Cooking', 'Health', 'Bike Repair']
        else f"{data['q_type']}_physical"
    )
    save_list.append(data)
  with open(output_file, 'w') as f:
    json.dump(save_list, f, indent=4)
  print(f'Saved {len(save_list)} items to {output_file}')


def transform_input_file2():
  data_file = 'data/egoexo4d/annotations/pretraining/vlm_baseline/input/test_alltasks_mcqv0_new.json'
  output_file = data_file.replace('_new.json', '_byscenariotype.json')
  take_mapping = load_mapping()
  with open(data_file, 'r') as f:
    data_list = json.load(f)

  save_list = []
  for data in data_list:
    visible = data['q_type'].split('_')[0]
    scenario = data['q_type'].split('_')[-1]
    scenario_type = (
        'procedural'
        if scenario in ['Cooking', 'Health', 'Bike Repair']
        else 'physical'
    )
    data['q_type'] = f'{visible}_{scenario_type}'
    save_list.append(data)
  with open(output_file, 'w') as f:
    json.dump(save_list, f, indent=4)
  print(f'Saved {len(save_list)} items to {output_file}')


def preprocess_vlm_mcq_csv_to_json_egoexo4d():
  """Convert MCQ CSV to input JSON for Gemini inference.

  Each entry contains video path, options, ground truth, etc.
  """
  csv_file = (  #'data/egoexo4d/annotations/pretraining/test_alltasks_mcqv0_sampled.csv'
      'data/egoexo4d/annotations/pretraining/test_alltasks_mcqv0.csv'
  )
  camera_name_mapping = load_camera_name_mapping('data/egoexo4d/takes.json')
  take_name_mapping = load_mapping()
  df = pd.read_csv(csv_file)
  input_list = []
  for i in range(0, len(df), 5):
    row = df.iloc[i]
    batch = df.iloc[i : i + 5]
    text_list = batch['description_text'].tolist()
    ground_truth = text_list[0]
    options = text_list.copy()
    random.shuffle(options)
    correct_answer_idx = options.index(ground_truth)
    query = 'Which of the following descriptions best matches the video?\n'
    for j, option in enumerate(options):
      query += f'{chr(65+j)}. {option}\n'
    query += f'Please answer with just the letter (A, B, C, D, or E).'
    take_id = '_'.join(row['save_id'].split('_')[:-2])
    video_path = os.path.join(
        'data/egoexo4d/takes',
        take_id,
        camera_name_mapping[take_id].replace(
            'frame_aligned_videos', 'frame_aligned_videos/downscaled/448/'
        ),
    )
    input_list.append({
        'qa_idx': f'{i//5:04d}',
        'q_type': (
            f"egovisible{row['ego_visible']}_{take_name_mapping[take_id]}"
        ),
        'video_path': video_path,
        'options': options,
        'ground_truth': ground_truth,
        'answer': str(chr(65 + correct_answer_idx)),
        'query': query,
        'start_time': row['t0'],  # row['start_time'],
        'end_time': row['t1'],  # row['end_time'],
    })
  output_json_file = csv_file.replace(
      'pretraining', 'pretraining/vlm_baseline/input'
  ).replace('.csv', '_new.json')
  os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
  with open(output_json_file, 'w') as f:
    json.dump(input_list, f, indent=4)
  print(f'Saved {len(input_list)} MCQ input items to {output_json_file}')


def preprocess_vlm_mcq_csv_to_json_nymeria():
  data_dir = os.path.expanduser('~/data/nymeria')
  csv_files = glob.glob(f'{data_dir}/eval1000/*.csv')
  cnt = 0
  input_list = []
  for csv_file in csv_files:
    fn = os.path.basename(csv_file).replace('split_by_', '').replace('.csv', '')
    key = (
        fn.replace('_', ' ')
        .replace('hands arms', 'hands/arms')
        .replace('legs feet', 'legs/feet')
    )
    print('processing', fn)
    df = pd.read_csv(csv_file)
    for i in range(0, len(df), 5):
      row = df.iloc[i]
      batch = df.iloc[i : i + 5]
      text_list = batch[key].tolist()
      ground_truth = text_list[0]
      options = text_list.copy()
      random.shuffle(options)
      correct_answer_idx = options.index(ground_truth)
      query = 'Which of the following descriptions best matches the video?\n'
      for j, option in enumerate(options):
        query += f'{chr(65+j)}. {option}\n'
      query += f'Please answer with just the letter (A, B, C, D, or E).'
      video_path = (
          row['motion_file']
          .replace('cam_motion_cache/presr50', 'video_cache')
          .replace('.npz', '/')
          + str(row['row_idx'])
          + '.mp4'
      )
      input_list.append({
          'qa_idx': f'{cnt:04d}',
          'q_type': fn,
          'video_path': video_path,
          'options': options,
          'ground_truth': ground_truth,
          'answer': str(chr(65 + correct_answer_idx)),
          'query': query,
      })
      cnt += 1
  output_json_file = f'{data_dir}/eval1000/vlm_baseline/input/test_all.json'
  os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
  with open(output_json_file, 'w') as f:
    json.dump(input_list, f, indent=4)
  print(f'Saved {len(input_list)} MCQ input items to {output_json_file}')


def preprocess_camerabench_to_json(mode='a'):
  data_dir = os.path.expanduser('~/data/CameraBench')
  assert mode in ['a', 'b']
  query = (
      'Describe the camera motion in this video.'
      if mode == 'a'
      else (
          'Provide a one-sentence description of the video (less than 100'
          ' words). Output in plain text (no code formatting, no backticks).'
      )
  )
  with open(f'{data_dir}/test.jsonl', 'r') as f:
    data_list = [json.loads(line) for line in f]
  save_list = []
  for i, data in enumerate(data_list):
    video_path = os.path.join(data_dir, data['path'])
    assert os.path.exists(video_path)
    save_list.append({
        'qa_idx': f'{i:04d}',
        'video_path': video_path,
        'query': query,
        'labels': data['labels'],
        'gt_caption': data['caption'],
    })
  save_path = f'{data_dir}/input/test_vlm_{mode}.json'
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  with open(save_path, 'w') as f:
    json.dump(save_list, f, indent=4)
  print(f'Saved {len(save_list)} items to {save_path}')


def check_camerabench_textoverlap():
  nlp = spacy.load('en_core_web_sm')

  def get_noun_lemmas(sentence):
    doc = nlp(sentence)
    return [token.lemma_ for token in doc if token.pos_ == 'NOUN']

  def get_verb_lemmas(sentence):
    doc = nlp(sentence)
    return [token.lemma_ for token in doc if token.pos_ == 'VERB']

  data_dir = os.path.expanduser('~/data/CameraBench')
  # Load camera caption
  with open(f'{data_dir}/test.jsonl', 'r') as f:
    data_list = [json.loads(line) for line in f]

  # Load caption mapping
  caption_dict = defaultdict(str)
  with open(
      f'{data_dir}/output/test_vlm_b/gemini-2.5-pro_8frames.jsonl', 'r'
  ) as f:
    for line in f:
      line = json.loads(line)
      caption_dict[line['idx']] = line['response']

  cnt = 0
  data_idx = []
  for i, data in enumerate(data_list):
    cam_caption = data['caption']
    sem_caption = caption_dict[f'{i:04d}']
    cam_lemmas = get_noun_lemmas(cam_caption)
    sem_lemmas = get_noun_lemmas(sem_caption)
    cam_verbs = get_verb_lemmas(cam_caption)
    sem_verbs = get_verb_lemmas(sem_caption)
    noun_overlap = set(cam_lemmas) & set(sem_lemmas)
    verb_overlap = set(cam_verbs) & set(sem_verbs)
    if verb_overlap and not noun_overlap:
      print('-' * 20, i, '-' * 20)
      print(f'Verb overlap found for idx {i}: {verb_overlap}')
      print(f'Camera caption: {cam_caption}')
      print(f'Semantic caption: {sem_caption}')
    if noun_overlap or verb_overlap:
      continue
    cnt += 1
    data_idx.append(i)
  # save data_idx to csv
  df = pd.DataFrame(data_idx, columns=['idx'])
  df.to_csv(f'{data_dir}/test_filtered_idx2.csv', index=False)
  print(f'Found {cnt} non-overlapping pairs from {len(data_list)}')


def preprocess_camerabench_mcq(num=4):
  data_dir = os.path.expanduser('~/data/CameraBench')
  with open(f'{data_dir}/test.jsonl', 'r') as f:
    data_list = [json.loads(line) for line in f]

  filtered_idx_file = f'{data_dir}/test_filtered_idx2.csv'
  idx_list = pd.read_csv(filtered_idx_file)['idx'].tolist()

  # Load caption mapping
  caption_dict = defaultdict(str)
  with open(
      f'{data_dir}/output/test_vlm_b/gemini-2.5-pro_8frames.jsonl', 'r'
  ) as f:
    for line in f:
      line = json.loads(line)
      caption_dict[line['idx']] = line['response']

  filtered_caption_dict = {}
  filtered_data_list = []
  for i, data in enumerate(data_list):
    if i not in idx_list:
      continue
    filtered_caption_dict[len(filtered_data_list)] = caption_dict[f'{i:04d}']
    filtered_data_list.append(data)

  # First, collect all unique labels
  label_list = []
  for data in filtered_data_list:
    for label in data['labels']:
      if label not in label_list:
        label_list.append(label)
  print(f'Found {len(label_list)} unique labels')
  print(f'Labels: {label_list}')

  # Create one-hot encoding for each data point
  one_hot_encodings = []
  for data in filtered_data_list:
    # Initialize zero vector for all labels
    encoding = [0] * len(label_list)
    # Set 1 for labels that this data point has
    for label in data['labels']:
      label_idx = label_list.index(label)
      encoding[label_idx] = 1
    one_hot_encodings.append(encoding)

  # Convert to numpy array for easier computation
  one_hot_matrix = np.array(one_hot_encodings)
  print(f'One-hot matrix shape: {one_hot_matrix.shape}')

  # Calculate cosine distances between all pairs
  distances = cosine_distances(one_hot_matrix)

  save_list = []
  # For each data point, find diverse data points that are far from it
  for i, data in tqdm(
      enumerate(filtered_data_list), total=len(filtered_data_list)
  ):
    # Get distances from current point to all other points
    point_distances = distances[i]

    # Find indices of points that are farthest from current point
    # Exclude the point itself (distance = 0)
    other_indices = np.arange(len(point_distances)) != i
    other_distances = point_distances[other_indices]
    other_indices_full = np.arange(len(point_distances))[other_indices]

    # Greedy selection: start with the farthest point, then add points that are
    # both far from current point AND different from already selected points
    selected_indices = []
    remaining_indices = other_indices_full.copy()
    remaining_distances = other_distances.copy()

    # Start with the farthest point
    farthest_idx = remaining_indices[np.argmax(remaining_distances)]
    selected_indices.append(farthest_idx)

    # Remove the selected point from remaining candidates
    mask = remaining_indices != farthest_idx
    remaining_indices = remaining_indices[mask]
    remaining_distances = remaining_distances[mask]

    # Select remaining points greedily
    for _ in range(min(num - 1, len(remaining_indices))):
      if len(remaining_indices) == 0:
        break

      # Calculate diversity score for each remaining point
      # Score = distance from current point + average distance from selected points
      diversity_scores = []
      for idx in remaining_indices:
        # Distance from current point
        dist_from_current = point_distances[idx]

        # Average distance from already selected points
        if len(selected_indices) > 0:
          dists_from_selected = [
              distances[idx][sel_idx] for sel_idx in selected_indices
          ]
          avg_dist_from_selected = np.mean(dists_from_selected)
        else:
          avg_dist_from_selected = 0

        # Combined score (weighted sum)
        diversity_score = dist_from_current + 0.5 * avg_dist_from_selected
        diversity_scores.append(diversity_score)

      # Select the point with highest diversity score
      best_idx_pos = np.argmax(diversity_scores)
      best_idx = remaining_indices[best_idx_pos]
      selected_indices.append(best_idx)

      # Remove the selected point from remaining candidates
      mask = remaining_indices != best_idx
      remaining_indices = remaining_indices[mask]
      remaining_distances = remaining_distances[mask]

    caption_list = [filtered_caption_dict[i]] + [
        filtered_caption_dict[i] for i in selected_indices
    ]
    cam_caption_list = [data['caption']] + [
        filtered_data_list[i]['caption'] for i in selected_indices
    ]
    if '' in caption_list or '' in cam_caption_list:
      continue
    # === Prompt 1: Video → Camera ===
    query_video_caption = caption_list[0]
    cam_options = cam_caption_list.copy()
    correct_answer_cam = cam_options[0]
    random.shuffle(cam_options)
    answer_cam_letter = chr(
        65 + cam_options.index(correct_answer_cam)
    )  # 'A' = 65

    prompt1 = f"""The following describes the content of a video:"{query_video_caption}"
        Which of the following camera movement descriptions most likely corresponds to this video?\n"""
    for j, option in enumerate(cam_options):
      prompt1 += f'{chr(65 + j)}. {option}\n'
    prompt1 += 'Reply with just the letter (A, B, C, D, or E).'

    save_list.append({
        'qa_idx': f'sem2cam_{i:04d}',
        'video_path': data['path'],
        'query': prompt1,
        'answer': answer_cam_letter,
    })

    # === Prompt 2: Camera → Video ===
    query_cam_caption = cam_caption_list[0]
    vid_options = caption_list.copy()
    correct_answer_vid = vid_options[0]
    random.shuffle(vid_options)
    answer_vid_letter = chr(65 + vid_options.index(correct_answer_vid))

    prompt2 = f"""The following describes the motion and focus of a camera while filming a scene:"{query_cam_caption}"
        Which of the following events or scene descriptions is most likely being filmed with this camera movement?\n"""
    for j, option in enumerate(vid_options):
      prompt2 += f'{chr(65 + j)}. {option}\n'
    prompt2 += 'Reply with just the letter (A, B, C, D, or E).'

    save_list.append({
        'qa_idx': f'cam2sem_{i:04d}',
        'video_path': data['path'],
        'query': prompt2,
        'answer': answer_vid_letter,
    })

    if i < 10:
      print(f'Prompt 1: {prompt1}')
      print(f'Answer 1: {answer_cam_letter}')
      print('-' * 40)
      print(f'Prompt 2: {prompt2}')
      print(f'Answer 2: {answer_vid_letter}')
      print('-' * 60)

  save_path = f'{data_dir}/input/text_mcq_filtered2.json'
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  with open(save_path, 'w') as f:
    json.dump(save_list, f, indent=4)
  print(f'Saved {len(save_list)} items to {save_path}')


def preprocess_dynpose_verb():
  nlp = spacy.load('en_core_web_sm')

  def get_verb_lemmas(sentence):
    doc = nlp(sentence)
    return [token.lemma_ for token in doc if token.pos_ == 'VERB']

  data_dir = os.path.expanduser('~/data/dynpose-100k/dynpose_100k')
  df = pd.read_csv(f'{data_dir}/metadata_val_v1.csv')
  df['verb_lemmas'] = df['description_text'].apply(get_verb_lemmas)
  df.to_csv(f'{data_dir}/metadata_val_v1_with_verbs.csv', index=False)


def preprocess_dynpose_to_json():
  data_dir = os.path.expanduser('~/local_data/dynpose-100k/dynpose_100k')
  video_files = glob.glob(os.path.join(data_dir, 'video/00000', '*.mp4'))
  print(f'Found {len(video_files)} video files')
  save_list = []
  for i, video_file in enumerate(video_files):
    save_list.append({
        'qa_idx': f'{i:04d}',
        'video_path': video_file,
        'query': 'Describe the camera motion in this video.',
    })
  save_path = f'{data_dir}/input/part0.json'
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  with open(save_path, 'w') as f:
    json.dump(save_list, f, indent=4)
  print(f'Saved {len(save_list)} items to {save_path}')


camera_motion_prompt = {
    '0': (
        'Describe the camera motion in the video.\nA. no motion. The camera'
        ' remains stationary with no intentional movement.\nB. simple-motion.'
        ' The camera moves in a straightforward and easily classifiable manner,'
        ' such as a steady pan, tilt, arc, or simple tracking shot.\nC.'
        ' complex-motion. The camera exhibits complex movements that are'
        ' difficult to classify. This includes:\n(1) Conflicting Motion:'
        ' Opposing movements occur (e.g., panning left then right), often seen'
        ' in drone shots, video game footage, or action scenes.\n(2) Sequential'
        ' Motion: Multiple distinct movements happen one after another (e.g.,'
        ' tracking forward, then shifting laterally).\n(3) Simultaneous Motions'
        ' at Different Speeds: Combined movements occur at differing speeds'
        ' (e.g., slow zoom with fast pan).\n(4) Unclear Motion or Missing'
        ' Background Cues: Motion is hard to interpret due to motion blur or'
        ' lack of visual references.\nReply with one single letter (A, B,'
        ' or C).'
    ),
    '1': (
        'Describe the camera steadiness in the video.\nA. static. The camera'
        ' remains completely stationary with no visible movement or'
        ' vibration.\nB. slight-shaking. The camera exhibits minor shaking or'
        ' jitter, either while stationary or moving, but the overall shot'
        ' remains mostly stable.\nC. unsteady. The camera shows moderate to'
        ' strong shaking, whether stationary or in motion, introducing'
        ' noticeable and potentially distracting instability.\nReply with one'
        ' single letter (A, B, or C).'
    ),
}


def preprocess_dynpose_to_json2(mode):
  data_dir = os.path.expanduser('~/local_data/dynpose-100k/dynpose_100k')
  df = pd.read_csv(f'{data_dir}/metadata_val_withvideo_sampled1000.csv')
  save_list = []
  for i, row in df.iterrows():
    video_path = os.path.join(data_dir, row['video_path'])
    assert os.path.exists(video_path)
    save_list.append({
        'qa_idx': f'{i:04d}',
        'video_path': video_path,
        'query': camera_motion_prompt[mode],
    })
  save_path = f'{data_dir}/input/cmlabel_query{mode}.json'
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  with open(save_path, 'w') as f:
    json.dump(save_list, f, indent=4)
  print(f'Saved {len(save_list)} items to {save_path}')


def preprocess_dynpose_to_json3():
  PROMPT_TEMPLATE = (
      'You are given a video and the caption:\n'
      '"{caption}"\n\n'
      'Rate how well the caption is reflected in the video on a 0–4 scale.\n'
      'Scale: 0 none; 1 weak; 2 partial; 3 mostly; 4 exact.\n'
      'Return exactly one integer (0,1,2,3,4) and nothing else.'
  )
  data_dir = os.path.expanduser('~/data/dynpose-100k/dynpose_100k')
  df = pd.read_csv(f'{data_dir}/metadata_val_withvideo.csv')
  save_list = []
  for i, row in df.iterrows():
    video_path = os.path.join(data_dir, row['video_path'])
    assert os.path.exists(video_path)
    save_list.append({
        'qa_idx': f'{i:04d}',
        'video_path': video_path,
        'query': PROMPT_TEMPLATE.format(caption=row['description_text']),
    })
  save_path = f'{data_dir}/vlm_queries/input/vtalign.json'
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  with open(save_path, 'w') as f:
    json.dump(save_list, f, indent=4)
  print(f'Saved {len(save_list)} items to {save_path}')


def read_dynpose_similartext():
  data_dir = os.path.expanduser('~/data/dynpose-100k/dynpose_100k')
  df = pd.read_csv(f'{data_dir}/metadata_val_withvideo.csv')
  feature_dir = (
      'data/misc/retrieval_features/internvideofeatures/dynpose/data3796'
  )
  feats = torch.load(f'{feature_dir}/text.pt', map_location='cpu').float()

  x2 = (feats**2).sum(dim=1, keepdim=True)
  dist2 = x2 + x2.t() - 2 * feats @ feats.t()
  dist2.fill_diagonal_(float('inf'))  # exclude self
  idxs = dist2.topk(10, dim=1, largest=False).indices
  rows = []
  sampled_row_idx = random.sample(range(len(df)), 1000)
  for i in sampled_row_idx:
    row = df.loc[i].copy()
    sub_df = df.loc[idxs[i]].copy()
    sub_df = sub_df.drop_duplicates(subset=['description_text'])
    sub_df = sub_df.sample(n=4, random_state=42)
    rows.append(pd.concat([row.to_frame().T, sub_df], axis=0))
    print(f"Row {i}, {df.loc[i, 'description_text']}")
    for j, r in sub_df.iterrows():
      print('  ->', j, r['description_text'])
    print('-' * 30)
  save_df = pd.concat(rows, axis=0, ignore_index=True)
  save_df.to_csv(
      f'{data_dir}/metadata_val_withvideo_featdist5000.csv', index=False
  )


def read_dynpose_vtalign():
  data_dir = 'data/dynpose-100k/dynpose_100k/'
  input_file = f'{data_dir}/vlm_queries/input/vtalign.json'
  response_file = f'{data_dir}/vlm_queries/output/vtalign/gemini-2.5-pro_8frames_nothinking.jsonl'

  with open(input_file, 'r') as f:
    data_list = json.load(f)
  print(f'{len(data_list)} input items')

  response_dict = defaultdict(str)
  with open(response_file, 'r') as f:
    for line in f:
      data_dict = json.loads(line)
      response_dict[data_dict['idx']] = data_dict['response']
  print(Counter(response_dict.values()))

  sub_data_list = [
      d for d in data_list if response_dict[d['qa_idx']] not in ['', '3', '4']
  ]
  sub_data_list = random.sample(sub_data_list, 1000)
  print(len(sub_data_list))

  df = pd.read_csv(f'{data_dir}/metadata_val_withvideo.csv')
  rows = []
  for data in sub_data_list:
    row = df.iloc[int(data['qa_idx'])]
    sub_df = df[df['description_text'] != row['description_text']]
    sub_df = sub_df.sample(n=4, random_state=42)
    rows.append(pd.concat([row.to_frame().T, sub_df], axis=0))
  save_df = pd.concat(rows, axis=0, ignore_index=True)
  save_df.to_csv(
      f'{data_dir}/metadata_val_withvideo_vtscore5000.csv', index=False
  )


def preprocess_camerabench_to_vqa():
  data_dir = os.path.expanduser('~/data/CameraBench')
  with open(f'{data_dir}/test.jsonl', 'r') as f:
    data_list = [json.loads(line) for line in f]
  save_list = []
  for i, data in enumerate(data_list):
    if 'no-motion' in data['labels']:
      gt = 'A'
    elif 'minor-motion' in data['labels']:
      gt = 'B'
    elif 'complex-motion' in data['labels']:
      gt = 'C'
    else:
      continue
    save_list.append({
        'qa_idx': f'{i:04d}',
        'video_path': os.path.join(data_dir, data['path']),
        'query': camera_motion_prompt['0'],
        'answer': gt,
    })
  save_path = f'{data_dir}/input/vqa_prompt0.json'
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  with open(save_path, 'w') as f:
    json.dump(save_list, f, indent=4)
  print(f'Saved {len(save_list)} items to {save_path}')


def preprocess_dynpose_val_to_mcq():
  data_dir = os.path.expanduser('~/local_data/dynpose-100k/dynpose_100k')
  df = pd.read_csv(f'{data_dir}/metadata.csv')
  valid_indices = np.load(f'{data_dir}/val_indices.npy')
  cnt = -1
  caption_list = []
  for i, row in df.iterrows():
    caption_list = eval(row['caption'])
    for caption in caption_list:
      cnt += 1
      if cnt not in valid_indices:
        continue
      caption_list.append(caption)
      if len(caption_list) == 5:
        print(caption_list)
        print('-' * 40)
        caption_list = []


def preprocess_dynpose_query(mode):
  def build_prompt(query, mode):
    if mode == 'a':
      return f"""Decide if the video is filmed indoors, outdoors, or if it is unclear.\nVideo caption: "{query}"\nLabels: A = Indoor, B = Outdoor, C = Unsure, Answer with only one letter: A, B, or C."""
    elif mode == 'b':
      return f"""How many people are visible in the video?\nVideo caption: "{query}"\nLabels: A = None, B = One, C = Two, D = Three or more, E = Unsure, Answer with only one letter: A, B, C, D, or E."""
    elif mode == 'c':
      return f"""Does the video contain any animals?\nVideo caption: "{query}"\nLabels: A = Yes, B = No, C = Unsure, Answer with only one letter: A, B, or C."""
    elif mode == 'd':
      return f"""Does the video feature any food?\nVideo caption: "{query}"\nLabels: A = Yes, B = No, C = Unsure, Answer with only one letter: A, B, or C."""
    elif mode == 'e':
      return f"""Is the video filmed during the day or at night?\nVideo caption: "{query}"\nLabels: A = Day, B = Night, C = Unsure, Answer with only one letter: A, B, or C."""
    elif mode == 'f':
      return f"""Is the scene in the video urban or rural?\nVideo caption: "{query}"\nLabels: A = Urban, B = Rural, C = Unsure, Answer with only one letter: A, B, or C."""
    elif mode == 'g':
      return f"""Is the camera stable or shaking while filming the video?\nVideo caption: "{query}"\nLabels: A = Stable, B = Shaking, C = Unsure, Answer with only one letter: A, B, or C."""
    elif mode == 'h':
      return f"""What is the age group of the people in the video?\nVideo caption: "{query}"\nLabels: A = Adults, B = Children, C = Both, D = None visible, E = Unsure, Answer with only one letter: A, B, C, D, or E."""
    elif mode == 'i':
      return f"""What is the gender of the people in the video?\nVideo caption: "{query}"\nLabels: A = Female, B = Male, C = Both, D = None visible, E = Unsure, Answer with only one letter: A, B, C, D, or E."""
    elif mode == 'j':
      return """Does the video contain any visible written text?\nLabels: A = Yes, B = No, C = Unsure, Answer with only one letter: A, B, or C."""
    elif mode == 'k':
      return f"""Does the video show people walking?\nVideo caption: "{query}"\nLabels: A = Yes, B = No, C = Unsure, Answer with only one letter: A, B, or C."""
    elif mode == 'l':
      return f"""Is the video related to sports activities?\nVideo caption: "{query}"\nLabels: A = Yes, B = No, C = Unsure, Answer with only one letter: A, B, or C."""

  df = pd.read_csv('data/dynpose-100k/dynpose_100k/metadata_val_v1.csv')
  save_list = []
  for i, row in df.iterrows():
    assert os.path.exists(row['video_path'])
    query = build_prompt(row['description_text'], mode)
    save_list.append({
        'qa_idx': f'{i:04d}',
        'video_path': row['video_path'],
        'query': query,
    })
    # print(i, query)

  if mode == 'a':
    fn = 'indooroutdoor_v1'
  elif mode == 'b':
    fn = 'peoplecount'
  elif mode == 'c':
    fn = 'animal'
  elif mode == 'd':
    fn = 'food'
  elif mode == 'e':
    fn = 'daynight'
  elif mode == 'f':
    fn = 'urbanrural'
  elif mode == 'g':
    fn = 'camerashake'
  elif mode == 'h':
    fn = 'agegroup'
  elif mode == 'i':
    fn = 'gender'
  elif mode == 'j':
    fn = 'textvisible'
  elif mode == 'k':
    fn = 'walking'
  elif mode == 'l':
    fn = 'sports'

  save_path = f'data/dynpose-100k/dynpose_100k/vlm_queries/input/{fn}.json'
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  with open(save_path, 'w') as f:
    json.dump(save_list, f, indent=4)
  print(f'Saved {len(save_list)} items to {save_path}')


if __name__ == '__main__':
  transform_input_file2()
  # preprocess_vlm_mcq_csv_to_json_egoexo4d()
  # preprocess_dynpose_verb()
  # for mode in ['k', 'l']:
  #     preprocess_dynpose_query(mode)
