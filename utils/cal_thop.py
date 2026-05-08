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

from thop import clever_format, profile
import torch
import torch.nn as nn
from torchvision.models import resnet50


def run_clip():
  import clip

  class CLIPWrapper(nn.Module):

    def __init__(self):
      super().__init__()
      self.clip_model, _ = clip.load("ViT-B/32", device="cuda")

    def forward(self, x):
      return self.clip_model.encode_image(x.cuda())

  x = torch.randn(1, 3, 224, 224)
  wrapper = CLIPWrapper()
  macs, params = profile(wrapper, inputs=(x,))
  print(*clever_format((macs, params), "%.6f"), sep=" | ")


def run_clap():
  # modified /opt/conda/lib/python3.10/site-packages/msclap/CLAPWrapper.py
  from msclap import CLAP

  model = CLAP(version="2023", use_cuda=True)
  x = "data/egoexo4d/audio_cache/test/cmu_soccer06_3_6.98_7.35.wav"
  x = model.preprocess_audio([x], resample=True)
  macs, params = profile(model, inputs=(x,))
  print(*clever_format((macs, params), "%.6f"), sep=" | ")


def run_camerapose_encoder():
  from datasets.egoexo4d import EgoExo4DCameraPoseLongSeqForPretraining
  from models.cm_encoder import CameraPoseSeqEncoder
  from loader import longseq_pose_text_collate

  model = CameraPoseSeqEncoder(encode_pose=2)
  dataset = EgoExo4DCameraPoseLongSeqForPretraining(None, "val")
  loader = torch.utils.data.DataLoader(
      dataset, batch_size=1, shuffle=False, collate_fn=longseq_pose_text_collate
  )
  for i, batch in enumerate(loader):
    poses, pad_mask, text_list, lengths, scenario_labels, full_text_list = batch
    # motion_features = model(poses, pad_mask, lengths, scenario_labels)
    macs, params = profile(
        model, inputs=(poses, pad_mask, lengths, scenario_labels)
    )
    print(*clever_format((macs, params), "%.2f"), sep=" | ")
    break


def run_qwen25():
  from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
  from qwen_vl_utils import process_vision_info

  class QwenWrapper(nn.Module):

    def __init__(self):
      super().__init__()
      self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
          "Qwen/Qwen2.5-VL-7B-Instruct",
          torch_dtype=torch.bfloat16,
          attn_implementation="flash_attention_2",
          device_map="auto",
      )
      self.processor = AutoProcessor.from_pretrained(
          "Qwen/Qwen2.5-VL-7B-Instruct"
      )

    def forward(self, video_path, query, bound):
      video_info = {
          "type": "video",
          "video": video_path,
          "fps": 1.0,
          "video_start": bound[0],
          "video_end": bound[1],
      }
      messages = [{
          "role": "user",
          "content": [
              video_info,
              {"type": "text", "text": query},
          ],
      }]
      text = self.processor.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True
      )
      image_inputs, video_inputs, video_kwargs = process_vision_info(
          messages, return_video_kwargs=True
      )
      inputs = self.processor(
          text=[text],
          images=image_inputs,
          videos=video_inputs,
          padding=True,
          return_tensors="pt",
          **video_kwargs,
      )
      inputs = inputs.to("cuda")
      inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(
          torch.bfloat16
      )
      outputs = self.model.generate(**inputs, max_new_tokens=50)
      return outputs

  model = QwenWrapper()
  video_path = "data/egoexo4d/takes/iiith_cooking_125_2/frame_aligned_videos/downscaled/448//aria01_214-1.mp4"
  query = (
      "Which of the following descriptions best matches the video?\nA. C wipes"
      " the skillet with the kitchen towel in his right hand.\nB. C drains the"
      " water on the skillet with his left hand.\nC. C places the jar of salt"
      " in the kitchen cabinet with his right hand.\nD. C opens the kitchen"
      " drawer with his right hand.\nE. C picks a skillet from the kitchen"
      " trolley with his left hand.\nPlease answer with just the letter (A, B,"
      " C, D, or E)."
  )
  bound = [252.54558280291883, 253.4079171970812]
  macs, params = profile(model, inputs=(video_path, query, bound))
  print(*clever_format((macs, params), "%.8f"), sep=" | ")


if __name__ == "__main__":
  # run_clip()
  # run_clap()
  # run_camerapose_encoder()
  run_qwen25()
