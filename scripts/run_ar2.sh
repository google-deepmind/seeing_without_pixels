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


# The base python command
BASE_CMD="python train_ar.py --dataset egoexo4d_action --num_gpus 2 --batch_size 128 --init_ckpt /home/sherryxue_google_com/data/logs/egoexo4d_pretrain_longseq/bs1024_dur4_pose2/checkpoints/best-epoch=490-val_loss=5.6580.ckpt"

# Arrays of parameters to sweep through
LEARNING_RATES=(1e-5 1e-4 1e-3)
WEIGHT_DECAYS=(1e-2 1e-3 1e-4)

for LR in "${LEARNING_RATES[@]}"
do
  # Loop through each weight decay
  for WD in "${WEIGHT_DECAYS[@]}"
  do
    JOB_NAME="bs128/lr${LR}_wd${WD}/init_4s"
    # Execute the command with the current LR and WD
    echo "Running with lr=$LR and wd=$WD"
    $BASE_CMD --lr $LR --weight_decay $WD --job_name "$JOB_NAME"
  done
done
