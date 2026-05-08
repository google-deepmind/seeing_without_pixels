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

from collections import defaultdict
import json
import os

from loader import create_loader

# from models.cm_classifier import CameraPoseSeqClassifier
from models.cm_encoder import CameraPoseSeqClassifier
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics


class CameraPoseSeqCls(pl.LightningModule):

  def __init__(self, args):
    super().__init__()
    self.args = args
    # buid a model
    self.model = CameraPoseSeqClassifier(
        args.num_classes,
        args.encode_pose,
        init_ckpt=args.init_ckpt,
        use_learnable_gravity=args.use_learnable_gravity,
    )

    # metrics
    self.train_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=args.num_classes
    )
    self.val_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=args.num_classes
    )
    top_k = 5 if args.num_classes > 5 else 1
    self.val_top5_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=args.num_classes, top_k=top_k
    )

    # Test metrics
    self.test_predictions = {}
    self.test_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=args.num_classes
    )
    # For proficiency test accuracy
    self.test_correct = 0
    self.test_total = 0

    # For per-class accuracy
    if hasattr(self.args, "per_cls_acc") and self.args.per_cls_acc:
      self.per_class_correct = defaultdict(int)
      self.per_class_total = defaultdict(int)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(
        self.model.parameters(),
        lr=self.args.lr,
        weight_decay=self.args.weight_decay,
    )
    return optimizer

  def training_step(self, batch, batch_idx):
    poses, pad_mask, labels, lengths, _ = batch
    logits = self.model(poses, pad_mask, lengths)
    loss = F.cross_entropy(logits, labels)

    # update / log running accuracy
    preds = logits.argmax(dim=1)
    self.train_acc.update(preds, labels)

    self.log(
        "train_loss",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )
    self.log(
        "train_acc",
        self.train_acc,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )
    return loss

  def validation_step(self, batch, batch_idx):
    poses, pad_mask, labels, lengths, _ = batch
    logits = self.model(poses, pad_mask, lengths)
    loss = F.cross_entropy(logits, labels)
    preds = logits.argmax(dim=1)

    # Update overall metrics
    self.val_acc.update(preds, labels)
    self.val_top5_acc.update(logits, labels)

    self.log(
        "val_loss",
        loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )

  def on_validation_epoch_end(self):
    # Log overall metrics
    self.log("val_acc", self.val_acc.compute(), prog_bar=True, logger=True)
    self.log(
        "val_top5_acc", self.val_top5_acc.compute(), prog_bar=True, logger=True
    )

    # Reset all metrics
    self.val_acc.reset()
    self.val_top5_acc.reset()

  def test_step(self, batch, batch_idx):
    poses, pad_mask, labels, lengths, ids = batch
    if self.args.save_middle_h:
      logits, middle_h = self.model(
          poses, pad_mask, lengths, return_middle_h=True
      )
      save_path = f"data/misc/{self.args.dataset}/idx{batch_idx}_{self.args.batch_size}_{middle_h.shape[0]}.pt"
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      torch.save(middle_h.cpu(), save_path)
      print(f"Saved middle hidden states to {save_path}")
    else:
      logits = self.model(poses, pad_mask, lengths)
    preds = logits.argmax(dim=1)
    self.test_acc.update(preds, labels)

    if hasattr(self.args, "per_cls_acc") and self.args.per_cls_acc:
      correct_preds = preds == labels
      for i in range(len(labels)):
        label = labels[i].item()
        self.per_class_total[label] += 1
        if correct_preds[i]:
          self.per_class_correct[label] += 1

    if self.args.dataset == "egoexo4d_prof":
      if self.args.num_test_clips > 1:
        id_to_logits = defaultdict(list)
        id_to_labels = {}
        for i, id_ in enumerate(ids):
          id_to_logits[id_].append(logits[i].detach().cpu().numpy())
          id_to_labels[id_] = labels[i].item()
        for id_, logit_list in id_to_logits.items():
          avg_logits = (
              torch.tensor(np.stack(logit_list))
              .float()
              .mean(dim=0, keepdim=True)
          )
          gt_label = id_to_labels[id_]
          pred = avg_logits.argmax(dim=1).item()
          self.test_correct += int(pred == gt_label)
          self.test_total += 1
      else:
        preds = logits.argmax(dim=1)
        self.test_correct += (preds.cpu() == labels.cpu()).sum().item()
        self.test_total += len(labels)

    if self.args.save_pred:
      probs = F.softmax(logits, dim=1)
      k = 5 if self.args.dataset == "egoexo4d_action" else 1
      top5_probs, top5_indices = torch.topk(probs, k=k, dim=1)
      top5_probs = top5_probs.cpu().numpy()
      top5_indices = top5_indices.cpu().numpy()
      gt_labels = labels.cpu().numpy()
      for i, (gt_label, sample_top5_probs, sample_top5_indices) in enumerate(
          zip(gt_labels, top5_probs, top5_indices)
      ):
        top1_correct = sample_top5_indices[0] == gt_label
        top5_correct = gt_label in sample_top5_indices
        self.test_predictions[ids[i]] = {
            "gt_label": int(gt_label),
            "gt_label_name": (
                ""
                if self.label_mapping is None
                else self.label_mapping[gt_label]
            ),
            "top1_correct": int(top1_correct),
            "top5_correct": int(top5_correct),
            "top1_pred": {
                "label": int(sample_top5_indices[0]),
                "label_name": (
                    ""
                    if self.label_mapping is None
                    else self.label_mapping[sample_top5_indices[0]]
                ),
                "prob": float(sample_top5_probs[0]),
            },
            "top5_preds": [
                {
                    "label": int(idx),
                    "label_name": (
                        ""
                        if self.label_mapping is None
                        else self.label_mapping[idx]
                    ),
                    "prob": float(prob),
                }
                for prob, idx in zip(sample_top5_probs, sample_top5_indices)
            ],
        }

  def on_test_epoch_end(self):
    self.log("test_acc", self.test_acc.compute(), prog_bar=True, logger=True)
    self.test_acc.reset()

    if self.args.dataset == "egoexo4d_prof":
      acc = self.test_correct / max(1, self.test_total)
      print(
          f"\nEgoExo4D Proficiency Test Accuracy: {acc:.4f}"
          f" ({self.test_correct}/{self.test_total})\n"
      )

    if self.args.per_cls_acc:
      print("\nPer-class Accuracy:")
      sorted_labels = sorted(self.per_class_total.keys())
      for label in sorted_labels:
        total = self.per_class_total[label]
        correct = self.per_class_correct[label]
        accuracy = correct / total if total > 0 else 0
        label_name = self.label_mapping.get(label, "Unknown")
        print(
            f"  Class {label} ({label_name}): {accuracy:.4f}"
            f" ({correct}/{total})"
        )

    if self.args.save_pred:
      ckpt_name = (
          "_".join(self.args.ckpt.split("/")[-3:])
          .replace(".ckpt", ".json")
          .replace("checkpoints_", "")
      )
      output_file = os.path.join(self.args.log_dir, "predictions", ckpt_name)
      os.makedirs(os.path.dirname(output_file), exist_ok=True)
      with open(output_file, "w") as f:
        json.dump(self.test_predictions, f, indent=2)
      print(
          f"\nSaved {len(self.test_predictions)} predictions to {output_file}"
      )

  def train_dataloader(self):
    return create_loader(self.args, "train")

  def val_dataloader(self):
    return create_loader(self.args, "val")

  def test_dataloader(self):
    # test_loader = create_loader(self.args, "test")
    test_loader, self.label_mapping = create_loader(self.args, "test")
    return test_loader
