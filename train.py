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
      description='Camera Motion Pretraining Arguments'
  )
  # Training setup
  parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs')
  parser.add_argument(
      '--epochs', type=int, default=500, help='Number of epochs'
  )
  parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
  parser.add_argument(
      '--num_workers', type=int, default=8, help='Number of workers'
  )
  parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
  parser.add_argument(
      '--weight_decay', type=float, default=1e-3, help='Weight decay'
  )

  # Model architecture
  parser.add_argument(
      '--d_model', type=int, default=128, help='Model dimension'
  )
  parser.add_argument(
      '--nhead', type=int, default=4, help='Number of attention heads'
  )
  parser.add_argument(
      '--num_layers', type=int, default=4, help='Number of transformer layers'
  )
  parser.add_argument(
      '--dim_feedforward', type=int, default=256, help='Feedforward dimension'
  )
  parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

  # Dataset and logging
  parser.add_argument(
      '--dataset', type=str, default='egoexo4d_pretrain', help='Dataset name'
  )
  parser.add_argument(
      '--eval_data', type=str, default='mcqv0', help='Eval data split'
  )
  parser.add_argument('--scenario', type=str, default='', help='Scenario')
  parser.add_argument(
      '--text_column', type=str, default='a', help='Text column'
  )
  parser.add_argument(
      '--encode_pose', type=int, default=2, help='Encode pose mode'
  )
  parser.add_argument(
      '--pre_sample_rate', type=int, default=50, help='Pre-sample rate'
  )
  parser.add_argument('--sample_rate', type=int, default=1, help='Sample rate')
  parser.add_argument(
      '--take_duration', type=int, default=4, help='Take duration'
  )
  parser.add_argument(
      '--test_take_duration', type=int, default=4, help='Take duration'
  )
  parser.add_argument(
      '--start_ratio', type=float, default=0.2, help='Ratio range'
  )
  parser.add_argument(
      '--test_time_ratio', type=float, default=0.5, help='Test time ratio'
  )
  parser.add_argument(
      '--num_negatives', type=int, default=0, help='Number of negative samples'
  )
  parser.add_argument(
      '--method',
      type=str,
      default='gt',
      choices=['gt', 'megasam', 'vipe', 'pi3'],
      help='Camera pose estimation method',
  )
  parser.add_argument(
      '--use_label_id', type=str, default='', help='Use label id'
  )
  parser.add_argument(
      '--action_mode', type=str, default='a', help='Action mode'
  )
  parser.add_argument(
      '--ego_visible', action='store_true', help='Use ego-visible data'
  )
  parser.add_argument(
      '--hier_text', action='store_true', help='Use hierarchical text'
  )
  parser.add_argument(
      '--sample_dur', action='store_true', help='Whether to sample duration'
  )
  parser.add_argument(
      '--train_concat_data',
      action='store_true',
      help='Train on concatenated data',
  )
  parser.add_argument(
      '--use_text_embeds', action='store_true', help='Use text embeds'
  )
  parser.add_argument(
      '--use_scenario_label', action='store_true', help='Use scenario label'
  )
  parser.add_argument(
      '--use_learnable_gravity',
      action='store_true',
      help='Use learnable gravity vector',
  )
  parser.add_argument(
      '--use_pi3_pose', action='store_true', help='Use PI3 pose'
  )
  parser.add_argument(
      '--finetune_clip', action='store_true', help='Finetune CLIP model'
  )
  parser.add_argument(
      '--cls_loss_weight',
      type=float,
      default=0.0,
      help='Weight for auxiliary classification loss',
  )
  parser.add_argument('--test', action='store_true', help='Test mode')
  parser.add_argument(
      '--log_dir', type=str, default='logs', help='Path to the log directory'
  )
  parser.add_argument('--job_name', type=str, default='debug', help='Job name')
  parser.add_argument(
      '--ckpt', type=str, default='', help='Path to checkpoint for evaluation'
  )
  parser.add_argument(
      '--init_ckpt',
      type=str,
      default='',
      help='Path to checkpoint for model initialization',
  )

  args = parser.parse_args()
  return args


def train(args):
  start_time = time.time()
  # Initialize model
  task = (
      CameraPoseTextPretraining(args)
      if 'longseq' not in args.dataset
      else CameraPoseTextPretrainingLongSeq(args)
  )

  args.log_dir = os.path.join(os.path.expanduser('~/data'), args.log_dir)
  log_dir = os.path.join(args.log_dir, args.dataset, args.job_name)

  # Save args to json file
  os.makedirs(log_dir, exist_ok=True)
  with open(os.path.join(log_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

  # Define checkpoint callbacks
  checkpoint_callback_best = ModelCheckpoint(
      dirpath=os.path.join(log_dir, 'checkpoints'),
      filename='best-{epoch:02d}-{val_loss:.4f}',
      monitor='val_loss',
      mode='min',
      save_top_k=1,
      save_last=True,
  )

  # Initialize loggers
  loggers = []

  if not args.test:
    # Only initialize loggers if not in test mode
    tensorboard_logger = TensorBoardLogger(
        save_dir=log_dir, name='tensorboard', default_hp_metric=False
    )

    wandb_logger = WandbLogger(
        project='camera-motion-pretraining',
        name=args.job_name,
        save_dir=log_dir,
        log_model=True,
    )

    loggers = [tensorboard_logger, wandb_logger]
  else:
    print('Test mode: TensorBoard and WandB loggers disabled')

  # Initialize trainer
  trainer = pl.Trainer(
      accelerator='gpu',
      devices=args.num_gpus,
      max_epochs=args.epochs,
      default_root_dir=log_dir,
      callbacks=[checkpoint_callback_best],
      logger=loggers,
      strategy='ddp_find_unused_parameters_true',
  )

  if args.ckpt:
    # Load checkpoint and evaluate
    print(f'Loading checkpoint from {args.ckpt} for evaluation')
    if args.test:
      if args.cls_loss_weight > 0:
        trainer.validate(task, ckpt_path=args.ckpt)
      trainer.test(task, ckpt_path=args.ckpt)
    else:
      trainer.validate(task, ckpt_path=args.ckpt)
  else:
    trainer.fit(task)
  end_time = time.time()
  print(f'Total time taken: {end_time - start_time} seconds')


if __name__ == '__main__':
  args = get_args()
  train(args)
