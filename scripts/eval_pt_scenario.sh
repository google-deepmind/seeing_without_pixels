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


SCENARIOS=("Cooking" "Health" "Dance" "Rock_Climbing" "Bike_Repair" "Basketball" "Soccer" "Music")

for scenario in "${SCENARIOS[@]}"; do
    # With --ego_visible
    python train.py --encode_pose 2 --dataset egoexo4d_pretrain_longseq --take_duration 4 --test --batch_size 1000 --ckpt ~/data/logs/egoexo4d_pretrain_longseq/bs1024_dur4_pose2/checkpoints/best-epoch=490-val_loss=5.6580.ckpt --scenario "$scenario" --num_gpus 1 --ego_visible

    # Without --ego_visible
    python train.py --encode_pose 2 --dataset egoexo4d_pretrain_longseq --take_duration 4 --test --batch_size 1000 --ckpt ~/data/logs/egoexo4d_pretrain_longseq/bs1024_dur4_pose2/checkpoints/best-epoch=490-val_loss=5.6580.ckpt --scenario "$scenario" --num_gpus 1
done
