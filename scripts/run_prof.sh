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


# for lr in 5e-5 1e-4 5e-4 1e-3; do
#     for wd in 5e-2 1e-2 5e-3 1e-3; do
#         python train_ar.py --dataset egoexo4d_action --batch_size 32 --epochs 200 --lr ${lr} --job_name 2gpu_bs32/lr${lr}_wd${wd}
#     done
# done

# for dur in 5 10 15 20 25; do
#     python train_ar.py --dataset egoexo4d_proficiency --sample_rate 50 --lr 1e-5 --job_name bouldering_2label/dur${dur}/lr1e-5 --take_duration ${dur} --epochs 50
# done

# --- Configuration ---
# The base python command
# BASE_CMD="python train_ar.py --dataset egoexo4d_prof --num_gpus 1 --batch_size 128 --take_duration 16 --scenario Dance"
BASE_CMD="python train_ar.py --dataset egoexo4d_prof --num_gpus 1 --batch_size 128 --take_duration 16"

# Set your starting GPU ID here (e.g., 5)
START_GPU=0

# Arrays of parameters to sweep through
LEARNING_RATES=(1e-5 1e-4 1e-3)
WEIGHT_DECAYS=(1e-2 1e-3 1e-4)


# --- Script Logic ---
echo "Starting parameter sweep..."

# Loop through the learning rates and assign a GPU to each
for i in "${!LEARNING_RATES[@]}"; do
  LR=${LEARNING_RATES[$i]}
  GPU_ID=$(($START_GPU + i))

  # Group commands for each GPU to run them sequentially in the background
  (
    echo "--- Starting job group on GPU $GPU_ID with LR $LR ---"

    # Inner loop for weight decays
    for WD in "${WEIGHT_DECAYS[@]}"; do
      JOB_NAME="dur16/lr${LR}_wd${WD}/base"
      echo "Running job on GPU $GPU_ID: LR=$LR, WD=$WD, Name=$JOB_NAME"

      # The actual command to run
      CUDA_VISIBLE_DEVICES=$GPU_ID $BASE_CMD --lr $LR --weight_decay $WD --job_name "$JOB_NAME"
    done

    echo "--- Finished all jobs for GPU $GPU_ID ---"
  ) & # The '&' runs the entire group of commands for one GPU in the background

done

# The 'wait' command will cause the script to wait until all background jobs have finished.
echo "All job groups launched. Waiting for completion..."
wait

echo "All training jobs have completed."
