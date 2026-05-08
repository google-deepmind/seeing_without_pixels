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


# Usage: bash run_pt.sh <config_index>

configs=()
for num_layers in 4 8 12; do
    for d_model in 128 256; do
        nhead=$((d_model / 64))
        for dim_feedforward in 512; do
            configs+=("$num_layers $d_model $nhead $dim_feedforward")
        done
    done
done

if [ -z "$1" ]; then
    echo "Please provide a config index (0-${#configs[@]})."
    exit 1
fi

idx=$1
if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#configs[@]}" ]; then
    echo "Config index out of range. Valid range: 0 to $(( ${#configs[@]} - 1 ))"
    exit 1
fi

read num_layers d_model nhead dim_feedforward <<< "${configs[$idx]}"
job_name="nl${num_layers}_dm${d_model}_df${dim_feedforward}"

cmd="python train.py --dataset egoexo4d_pretrain_longseq --take_duration 4 --encode_pose 2 \
    --num_layers ${num_layers} \
    --d_model ${d_model} \
    --nhead ${nhead} \
    --dim_feedforward ${dim_feedforward} \
    --job_name dur4_pose2/${job_name}"

echo "$cmd"

$cmd

