# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The main FACT model and related functions."""
from .base_model_util import *
from .base_models import *
from .multi_modal_model_util import *

from .config import audio_config, motion_config, multi_model_config

from math import floor

import torch
import torch.nn as nn


class FACTModel(nn.Module):
  """Audio Motion Multi-Modal model."""

  def __init__(self, audio_config, motion_config, cross_modal_config, out_dim: int = 225, pred_length: int = 20):
    """Initializer for FACTModel.

    Args:
      config: `FACTConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
    """
    super().__init__()

    self.audio_config = audio_config
    self.motion_config = motion_config
    self.cross_modal_config = cross_modal_config

    self.cross_modal_layer = CrossModalLayer(cross_modal_config, out_dim=out_dim)
    self.pred_length = pred_length

    self.motion_transformer = Transformer(
        in_features=motion_config.transformer.hidden_size,
        hidden_size=motion_config.transformer.hidden_size,
        num_hidden_layers=motion_config.transformer.num_hidden_layers,
        num_attention_heads=motion_config.transformer.num_attention_heads,
        intermediate_size=motion_config.transformer.intermediate_size,
        )

    self.motion_pos_embedding = PositionEmbedding(motion_config.sequence_length, motion_config.transformer.hidden_size)
    self.motion_linear_embedding = LinearEmbedding(motion_config.feature_dim, motion_config.transformer.hidden_size)
    
    self.audio_transformer = Transformer(
        in_features=motion_config.transformer.hidden_size,
        hidden_size=audio_config.transformer.hidden_size,
        num_hidden_layers=audio_config.transformer.num_hidden_layers,
        num_attention_heads=audio_config.transformer.num_attention_heads,
        intermediate_size=audio_config.transformer.intermediate_size,
        )

    self.audio_pos_embedding = PositionEmbedding(audio_config.sequence_length, audio_config.transformer.hidden_size)
    self.audio_linear_embedding = LinearEmbedding(audio_config.feature_dim, audio_config.transformer.hidden_size)

    # Linear Embedding has correct number of params
    # Positional Embedding has correct number of params
    # MLP has the correct number of params
    # Everything should now be correct

  def forward(self, inputs):
    """Predict sequences from inputs. 

    This is a single forward pass that been used during training. 

    Args:
      inputs: Input dict of tensors. The dict should contains 
        `motion_input` ([batch_size, motion_seq_length, motion_feature_dimension]) and
        `audio_input` ([batch_size, audio_seq_length, audio_feature_dimension]).

    Returns:
      Final output after the cross modal transformer. A tensor with shape 
      [batch_size, motion_seq_length + audio_seq_length, motion_feature_dimension]
      will be return. **Be aware only the first N-frames are supervised during training**
    """
    # Computes motion features.
    motion_features = self.motion_linear_embedding(inputs["motion_input"])
    motion_features = self.motion_pos_embedding(motion_features)
    motion_features = self.motion_transformer(motion_features)

    # Computes audio features.
    audio_features = self.audio_linear_embedding(inputs["audio_input"])
    audio_features = self.audio_pos_embedding(audio_features)
    audio_features = self.audio_transformer(audio_features)

    # Computes cross modal output.
    output = self.cross_modal_layer(motion_features, audio_features)

    return output[:, :self.pred_length]

  def infer_auto_regressive(self, inputs, steps: int = 1200, step_size: int = 1):
    """Predict sequences from inputs in an auto-regressive manner. 

    This function should be used only during inference. During each forward step, 
    only the first frame was kept. Inputs are shifted by 1 frame after each forward.


    Args:
      inputs: Input dict of tensors. The dict should contains 
        `motion_input` ([batch_size, motion_seq_length, motion_feature_dimension]) and
        `audio_input` ([batch_size, audio_seq_length, audio_feature_dimension]).

    Returns:
      Final output after the auto-regressive inference. A tensor with shape 
      [batch_size, steps, motion_feature_dimension]
      will be return.
    """
    audio_seq_length = self.audio_config.sequence_length
    outputs = []
    motion_input = inputs["motion_input"]

    for i in range(0, steps, step_size):
        audio_input = inputs["audio_input"][:, i: i + audio_seq_length]
        if audio_input.shape[1] < audio_seq_length:
          break

        output = self.forward({"motion_input": motion_input, "audio_input": audio_input})
        output = output[:, 0:step_size, :]  # only keep the first step_size frames
        
        outputs.append(output)
        # update motion input
        motion_input = torch.cat([motion_input[:, step_size:, :], output], dim=1)
    return torch.cat(outputs, dim=1)

if __name__ == '__main__':
    import time
    model = FACTModel(audio_config, motion_config, multi_model_config)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('The number of params in Million: ', params/1e6)

    model = model.to('cuda:0')

    features = {
        "motion_input": torch.ones([2, 120, 225], dtype=torch.float32).to("cuda:0"),
        "audio_input": torch.ones([2, 240, 35], dtype=torch.float32).to("cuda:0"),
    }
    out = model(features)