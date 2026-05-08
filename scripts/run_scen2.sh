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


METHOD=init
START_GPU=1

# --- Configuration ---
BASE_CMD="python train_ar.py --dataset egoexo4d_scenario_subset --num_gpus 1 --batch_size 128 --method $METHOD --umeyama_transform --init_ckpt /home/sherryxue_google_com/data/logs/egoexo4d_pretrain_longseq/bs1024_dur4_pose2_sr4_new/checkpoints/best-epoch=469-val_loss=5.6075.ckpt"
BASE_CMD="python train_ar.py --dataset ucf101 --method pi3 --batch_size 128 --init_ckpt /home/sherryxue_google_com/data/logs/dynpose_pretrain/v1_vipe/checkpoints/best-epoch=315-val_loss=6.0542.ckpt"
BASE_CMD="python train_ar.py --dataset finegym --batch_size 512 --init_ckpt /home/sherryxue_google_com/data/logs/dynpose_pretrain/v1_vipe/checkpoints/best-epoch=315-val_loss=6.0542.ckpt"
BASE_CMD="python train_ar.py --dataset egoexo4d_scenario --num_gpus 1 --batch_size 128 --take_duration 16 --test_take_duration 16 --sample_dur --encode_pose 5 --sample_rate 4 --init_ckpt /home/sherryxue_google_com/data/logs/egoexo4d_pretrain_longseq/bs1024_sampledur8_pose5_sr4/checkpoints/best-epoch=492-val_loss=5.5480.ckpt"

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
      JOB_NAME="pose5_sr4/bs128_lr${LR}_wd${WD}/init16s"
      # JOB_NAME="split01_filtered_v4/bs128_lr${LR}_wd${WD}/pi3_$METHOD"
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
