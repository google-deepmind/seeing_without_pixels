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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from tasks.action_recognition import CameraPoseSeqCls


def get_args():
  parser = argparse.ArgumentParser(
      description='Camera Motion Training Arguments'
  )
  parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs')
  parser.add_argument(
      '--epochs', type=int, default=200, help='Number of epochs'
  )
  parser.add_argument(
      '--dataset',
      type=str,
      default='hdepic',
      choices=[
          'hdepic',
          'egoexo4d_scenario',
          'egoexo4d_scenario_subset',
          'egoexo4d_action',
          'egoexo4d_prof',
          'egoexo4d_proficiency',
          'ucf101',
          'finegym',
      ],
      help='Dataset',
  )
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=1,
      help='Camera motion sample rate (after pre-sampling 100Hz)',
  )
  parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
  parser.add_argument(
      '--num_workers', type=int, default=4, help='Number of workers'
  )
  parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
  parser.add_argument(
      '--weight_decay', type=float, default=1e-3, help='Weight decay'
  )
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
      default=None,
      help='Path to checkpoint for initialization',
  )
  parser.add_argument(
      '--absolute_pose',
      action='store_true',
      help='Use absolute pose instead of relative pose',
  )
  parser.add_argument(
      '--use_6d_rotation',
      action='store_true',
      help='Use 6D rotation instead of 7D rotation',
  )
  parser.add_argument(
      '--method',
      type=str,
      default='gt',
      choices=['gt', 'megasam', 'vipe', 'pi3', 'd4rt'],
      help='Camera pose estimation method',
  )
  parser.add_argument('--split', type=str, default='01', help='split')
  parser.add_argument(
      '--ucf_version', type=int, default=4, help='UCF101 filtered version'
  )
  parser.add_argument(
      '--umeyama_transform',
      action='store_true',
      help='Whether to scale the translation component of the camera pose',
  )
  parser.add_argument(
      '--by_participant', action='store_true', help='Use by-participant data'
  )
  parser.add_argument(
      '--use_label_id', type=str, default='', help='Use label id'
  )
  parser.add_argument(
      '--encode_pose', type=int, default=2, help='Encode pose type'
  )
  parser.add_argument('--test', action='store_true', help='Test mode')
  parser.add_argument(
      '--save_pred', action='store_true', help='Save the predictions'
  )
  parser.add_argument(
      '--per_cls_acc',
      action='store_true',
      help='Calculate per-class accuracy during testing',
  )
  parser.add_argument(
      '--scenario_name',
      type=str,
      default='Rock Climbing',
      help='EgoExo4d proficiency scenario name',
  )
  parser.add_argument(
      '--action_mode', type=str, default='a', help='EgoExo4d action mode'
  )
  parser.add_argument(
      '--save_middle_h', action='store_true', help='Save middle hidden states'
  )
  parser.add_argument(
      '--use_learnable_gravity',
      action='store_true',
      help='Use learnable gravity vector',
  )
  parser.add_argument(
      '--sample_dur', action='store_true', help='Whether to sample duration'
  )
  parser.add_argument(
      '--test_ratio', type=float, default=1.0, help='Test ratio'
  )
  parser.add_argument(
      '--take_duration',
      type=float,
      default=4,
      help='EgoExo4d proficiency take duration',
  )
  parser.add_argument(
      '--test_take_duration',
      type=float,
      default=4,
      help='EgoExo4d proficiency test take duration',
  )
  parser.add_argument(
      '--num_test_clips',
      type=int,
      default=1,
      help=(
          'Number of uniformly sampled test clips per take for proficiency task'
      ),
  )
  args = parser.parse_args()
  return args


num_classes_mapping = {
    'hdepic': 105,
    'egoexo4d_action': 278,  # 92, #74,
    'egoexo4d_proficiency': 4,
    'egoexo4d_prof': 4,  # 2,
    'egoexo4d_scenario': 8,
    'egoexo4d_scenario_subset': 8,
    'finegym': 4,
}

ucf_class_mapping = {0: 101, 1: 101, 2: 10, 3: 10, 4: 8}


def train(args):
  args.num_classes = (
      ucf_class_mapping[args.ucf_version]
      if args.dataset == 'ucf101'
      else num_classes_mapping[args.dataset]
  )
  task = CameraPoseSeqCls(args)
  args.logdir = os.path.join(
      os.path.expanduser('~/data'),
      args.log_dir,
      f'{args.dataset}_{args.num_classes}label',
  )
  if args.dataset == 'egoexo4d_prof':
    args.job_name = f'{args.scenario_name}/{args.job_name}'
  log_dir = os.path.join(args.logdir, args.job_name)

  # Save args to json file
  os.makedirs(log_dir, exist_ok=True)
  with open(os.path.join(log_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

  # Setup tensorboard logger
  loggers = []
  if not args.test and not args.ckpt:
    logger_tb = TensorBoardLogger(log_dir, name='')
    wandb_logger = WandbLogger(
        project=args.dataset,
        name=args.job_name,
        save_dir=log_dir,
        log_model=True,
    )
    loggers = [logger_tb, wandb_logger]
  else:
    print('Test mode: TensorBoard and WandB loggers disabled')

  # Define checkpoint callbacks
  checkpoint_callback_best = ModelCheckpoint(
      dirpath=os.path.join(log_dir, 'checkpoints'),
      filename='best-{epoch:02d}-{val_acc:.4f}',
      monitor='val_acc',
      mode='max',
      save_top_k=1,
      save_last=False,
  )

  trainer = pl.Trainer(
      accelerator='gpu',
      devices=args.num_gpus,
      max_epochs=args.epochs,
      default_root_dir=log_dir,
      callbacks=[checkpoint_callback_best],
      logger=loggers,
  )

  if args.ckpt:
    # Load the checkpoint and evaluate
    print(f'Loading checkpoint from {args.ckpt} for evaluation')
    if args.save_pred or args.test:
      trainer.test(task, ckpt_path=args.ckpt)
    else:
      trainer.validate(
          task, ckpt_path=args.ckpt
      )  # Run test after loading checkpoint
  else:
    # Regular training
    trainer.fit(task)
    # Run test with best checkpoint after training
    trainer.validate(ckpt_path=checkpoint_callback_best.best_model_path)
    # trainer.test(ckpt_path=checkpoint_callback_best.best_model_path)


if __name__ == '__main__':
  args = get_args()
  train(args)
