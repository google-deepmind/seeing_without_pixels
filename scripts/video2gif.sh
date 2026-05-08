#!/bin/bash
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

# Usage: ./video2gif.sh /path/to/input_videos /path/to/output_gifs

input_dir="viz/dynpose/1example"
output_dir="viz/dynpose/1example_gifs"
FPS="${3:-15}"          # default 15 fps
MAX_W="${4:-0}"         # 0 = keep original width

if [[ -z "$input_dir" || -z "$output_dir" ]]; then
  echo "Usage: $0 INPUT_DIR OUTPUT_DIR [FPS] [MAX_WIDTH]"
  exit 1
fi

mkdir -p "$output_dir"

# Build the scale expression
if [[ "$MAX_W" -gt 0 ]]; then
  SCALE="scale='if(gt(iw,$MAX_W),$MAX_W,iw)':-1:flags=lanczos"
else
  SCALE="scale=iw:-1:flags=lanczos"
fi

shopt -s nullglob
for video in "$input_dir"/*.{mp4,mov,avi,mkv,webm,MP4,MOV,AVI,MKV,WEBM}; do
  filename=$(basename "$video")
  name="${filename%.*}"
  out_gif="$output_dir/$name.gif"
  palette="$output_dir/.palette_$name.png"

  echo "Converting: $video → $out_gif (fps=$FPS)"

  # Skip if GIF already exists
  if [[ -f "$out_gif" ]]; then
    echo "Skipping $video — GIF already exists."
    continue
  fi

  # 1. Generate palette
  ffmpeg -y -v error -i "$video" -vf "$SCALE,fps=$FPS,palettegen=max_colors=256" "$palette"

  # 2. Use palette to create high-quality GIF
  ffmpeg -y -v error -i "$video" -i "$palette" \
    -filter_complex "$SCALE,fps=$FPS[p];[p][1:v]paletteuse=dither=bayer:bayer_scale=5" \
    -loop 0 "$out_gif"

  rm -f "$palette"
done

echo "✅ Done. GIFs saved in: $output_dir"
