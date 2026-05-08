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

#!/usr/bin/env python3
"""Test script to verify text embedding functionality works correctly."""

from datasets.egoexo4d import EgoExo4DCameraPoseSeqForPretraining
from loader import pose_text_collate
import numpy as np
import torch


class MockArgs:

  def __init__(
      self,
      use_text_embeds=True,
      num_negatives=0,
      encode_pose=0,
      train_concat_data=False,
      ego_visible="ego",
      eval_data="mcqv0",
  ):
    self.use_text_embeds = use_text_embeds
    self.num_negatives = num_negatives
    self.encode_pose = encode_pose
    self.train_concat_data = train_concat_data
    self.ego_visible = ego_visible
    self.eval_data = eval_data


def test_text_embeds_collate():
  """Test the collate function with both text embeddings and raw text."""

  # Test 1: Raw text (strings)
  print("Testing with raw text strings...")
  mock_args = MockArgs(use_text_embeds=False)

  # Create mock batch with raw text
  mock_batch = [
      (np.random.randn(10, 7), "camera moving forward"),  # (trajectory, text)
      (np.random.randn(8, 7), "camera rotating left"),
      (np.random.randn(12, 7), "camera panning right"),
  ]

  try:
    poses, pad_mask, texts, lengths = pose_text_collate(mock_batch)
    print(f"✓ Raw text collate successful")
    print(f"  - poses shape: {poses.shape}")
    print(f"  - pad_mask shape: {pad_mask.shape}")
    print(f"  - texts type: {type(texts)}, length: {len(texts)}")
    print(f"  - lengths shape: {lengths.shape}")
    assert isinstance(texts, list), "Texts should be a list of strings"
    assert all(isinstance(t, str) for t in texts), "All texts should be strings"
  except Exception as e:
    print(f"✗ Raw text collate failed: {e}")
    return False

  # Test 2: Text embeddings (numpy arrays)
  print("\nTesting with text embeddings...")
  mock_args = MockArgs(use_text_embeds=True)

  # Create mock batch with text embeddings
  mock_batch = [
      (
          np.random.randn(10, 7),
          np.random.randn(512),
      ),  # (trajectory, text_embedding)
      (np.random.randn(8, 7), np.random.randn(512)),
      (np.random.randn(12, 7), np.random.randn(512)),
  ]

  try:
    poses, pad_mask, texts, lengths = pose_text_collate(mock_batch)
    print(f"✓ Text embeddings collate successful")
    print(f"  - poses shape: {poses.shape}")
    print(f"  - pad_mask shape: {pad_mask.shape}")
    print(f"  - texts type: {type(texts)}, shape: {texts.shape}")
    print(f"  - lengths shape: {lengths.shape}")
    assert isinstance(texts, torch.Tensor), "Texts should be a torch tensor"
    assert texts.shape[1] == 512, "Text embeddings should have dimension 512"
  except Exception as e:
    print(f"✗ Text embeddings collate failed: {e}")
    return False

  # Test 3: Mixed types (should fail gracefully)
  print("\nTesting with mixed types (should fail)...")
  mock_batch = [
      (np.random.randn(10, 7), np.random.randn(512)),  # embedding
      (np.random.randn(8, 7), "raw text"),  # string
  ]

  try:
    poses, pad_mask, texts, lengths = pose_text_collate(mock_batch)
    print(f"✗ Mixed types should have failed but didn't")
    return False
  except Exception as e:
    print(f"✓ Mixed types correctly failed: {e}")

  print("\n✓ All tests passed!")
  return True


if __name__ == "__main__":
  test_text_embeds_collate()
