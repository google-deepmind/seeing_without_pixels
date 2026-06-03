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

import argparse
import json
import os
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from tasks.pretrain import CameraPoseTextPretraining
from tasks.pretrain_longseq import CameraPoseTextPretrainingLongSeq


def get_args():
  parser = argparse.ArgumentParser(
      description='CamFormer Camera Trajectory Pretraining'
  )
  # Training setup
  parser.add_argument('--num_gpus', type=int, default=8)
  parser.add_argument('--epochs', type=int, default=500)
  parser.add_argument('--batch_size', type=int, default=1024)
  parser.add_argument('--num_workers', type=int, default=8)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--weight_decay', type=float, default=1e-3)

  # Model architecture
  parser.add_argument('--d_model', type=int, default=128)
  parser.add_argument('--nhead', type=int, default=4)
  parser.add_argument('--num_layers', type=int, default=4)
  parser.add_argument('--dim_feedforward', type=int, default=256)
  parser.add_argument('--dropout', type=float, default=0.1)

  # Dataset
  parser.add_argument(
      '--dataset',
      type=str,
      default='egoexo4d_pretrain_longseq',
      choices=[
          'egoexo4d_pretrain_longseq',
          'dynpose_pretrain',
          'nymeria_pretrain',
      ],
      help=(
          'egoexo4d_pretrain_longseq = ego (contextualized longseq task); '
          'dynpose_pretrain = exo; nymeria_pretrain = ego zero-shot eval. '
          'Datasets without "longseq" use the base (mean-pooled) task.'
      ),
  )
  parser.add_argument('--scenario', type=str, default='')
  parser.add_argument(
      '--pose_source',
      type=str,
      default='original',
      choices=['original', 'vipe'],
      help=(
          "DynPose pose source: the dataset's own poses ('original') or"
          " ViPE-estimated ('vipe')."
      ),
  )
  parser.add_argument(
      '--text_column',
      type=str,
      default='a',
      choices=['a', 'b', 'c', 'd'],
      help=(
          'Nymeria narration type (a=body, b=hands/arms, c=legs/feet, d=focus).'
      ),
  )
  parser.add_argument(
      '--pose_encoding',
      type=str,
      default='rel9d_grav',
      choices=['abs7d', 'rel9d', 'rel9d_grav'],
      help=(
          'Pose representation fed to CamFormer. abs7d = raw absolute pose;'
          ' rel9d = pose relative to the clip center (6D rotation +'
          ' translation); rel9d_grav = rel9d plus the gravity-in-camera vector'
          ' (egocentric default).'
      ),
  )
  parser.add_argument('--pre_sample_rate', type=int, default=50)
  parser.add_argument('--sample_rate', type=int, default=1)
  parser.add_argument('--take_duration', type=int, default=8)
  parser.add_argument('--test_take_duration', type=int, default=4)
  parser.add_argument('--start_ratio', type=float, default=0.2)
  parser.add_argument('--test_time_ratio', type=float, default=0.5)
  parser.add_argument('--ego_visible', action='store_true')
  parser.add_argument('--hier_text', action='store_true')
  parser.add_argument('--sample_dur', action='store_true')
  parser.add_argument('--use_scenario_label', action='store_true')
  parser.add_argument('--use_learnable_gravity', action='store_true')
  parser.add_argument(
      '--use_pi3_pose',
      action='store_true',
      help=(
          'Use Pi3 predicted poses instead of Aria GT; requires '
          'EGOEXO4D_PI3_TRAJ_DIR to point at the per-take .npy files.'
      ),
  )
  parser.add_argument('--finetune_clip', action='store_true')
  parser.add_argument(
      '--cls_loss_weight',
      type=float,
      default=0.0,
      help='Weight of auxiliary scenario classification loss.',
  )

  # IO
  parser.add_argument('--test', action='store_true')
  parser.add_argument('--log_dir', type=str, default='logs')
  parser.add_argument('--job_name', type=str, default='debug')
  parser.add_argument('--ckpt', type=str, default='')
  parser.add_argument('--init_ckpt', type=str, default='')

  args = parser.parse_args()
  # Internal encoder/checkpoints use an integer code; map the named flag to it.
  args.encode_pose = {'abs7d': 0, 'rel9d': 2, 'rel9d_grav': 11}[
      args.pose_encoding
  ]
  return args


def train(args):
  start_time = time.time()
  # Evaluation dumps features per process to a shared path with no cross-rank
  # gather, and the MCQ scoring relies on dataset order, so test must be
  # single-process. Enforce it rather than silently corrupting the dump.
  if args.test and args.num_gpus != 1:
    print(
        f'[test] forcing --num_gpus 1 (was {args.num_gpus}) for deterministic'
        ' feature dumping.'
    )
    args.num_gpus = 1

  # Contextualized long-sequence task for Ego-Exo4D; base mean-pooled task
  # for the flat (trajectory, text) datasets (DynPose, Nymeria).
  if 'longseq' in args.dataset:
    task = CameraPoseTextPretrainingLongSeq(args)
  else:
    task = CameraPoseTextPretraining(args)

  args.log_dir = os.path.join(os.path.expanduser('~/data'), args.log_dir)
  log_dir = os.path.join(args.log_dir, args.dataset, args.job_name)
  os.makedirs(log_dir, exist_ok=True)
  with open(os.path.join(log_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

  checkpoint_callback_best = ModelCheckpoint(
      dirpath=os.path.join(log_dir, 'checkpoints'),
      filename='best-{epoch:02d}-{val_loss:.4f}',
      monitor='val_loss',
      mode='min',
      save_top_k=1,
      save_last=True,
  )

  loggers = []
  if not args.test:
    loggers = [
        TensorBoardLogger(
            save_dir=log_dir, name='tensorboard', default_hp_metric=False
        ),
        WandbLogger(
            project='camera-motion-pretraining',
            name=args.job_name,
            save_dir=log_dir,
            log_model=True,
        ),
    ]
  else:
    print('Test mode: TensorBoard and WandB loggers disabled')

  trainer = pl.Trainer(
      accelerator='gpu',
      devices=args.num_gpus,
      max_epochs=args.epochs,
      default_root_dir=log_dir,
      callbacks=[checkpoint_callback_best],
      logger=loggers,
      strategy='ddp_find_unused_parameters_true',
  )

  # `--ckpt` restores a full Lightning checkpoint. The released slim encoder
  # weights instead load via `--init_ckpt` (in __init__), and are evaluated
  # with `--test` and no `--ckpt` (ckpt_path=None uses the in-memory weights).
  if args.test:
    if args.ckpt and args.cls_loss_weight > 0:
      trainer.validate(task, ckpt_path=args.ckpt)
    trainer.test(task, ckpt_path=(args.ckpt or None))
  elif args.ckpt:
    trainer.validate(task, ckpt_path=args.ckpt)
  else:
    trainer.fit(task)

  end_time = time.time()
  print(f'Total time taken: {end_time - start_time} seconds')


if __name__ == '__main__':
  args = get_args()
  train(args)
