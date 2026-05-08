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

for num_layers in 8; do
    for d_model in 256; do
        nhead=$((d_model / 64))
        for dim_feedforward in 512; do
            job_name="nl${num_layers}_dm${d_model}_df${dim_feedforward}"
            ckpt=~/data/logs/egoexo4d_pretrain_longseq/dur4_pose2/${job_name}/checkpoints/best-*
            for ckpt_file in $ckpt; do
                if [ -f "$ckpt_file" ]; then
                    echo "Testing checkpoint: $ckpt_file"
                    python train.py --dataset egoexo4d_pretrain_longseq --take_duration 4 --encode_pose 2 \
                        --num_layers ${num_layers} --d_model ${d_model} --nhead ${nhead} --dim_feedforward ${dim_feedforward} \
                        --ckpt "$ckpt_file" \
                        --test --num_gpus 1 --batch_size 1000
                    python train.py --dataset egoexo4d_pretrain_longseq --take_duration 4 --encode_pose 2 \
                        --num_layers ${num_layers} --d_model ${d_model} --nhead ${nhead} --dim_feedforward ${dim_feedforward} \
                        --ckpt "$ckpt_file" \
                        --test --num_gpus 1 --batch_size 1000 --ego_visible
                fi
            done
        done
    done
done
