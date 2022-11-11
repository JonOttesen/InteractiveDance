import torch
import time

import numpy as np

from dance.models.fact.fact import FACTModel
from dance.models.fact.config import audio_config, fact_model, motion_config, multi_model_config



metrics = {
    'MSE': torch.nn.MSELoss(),
    'L1': torch.nn.L1Loss(),
    }

audio_config.sequence_length = 120
motion_config.sequence_length = 60

audio_config.transformer.intermediate_size = 1536
motion_config.transformer.intermediate_size = 1536
multi_model_config.transformer.intermediate_size = 1536
multi_model_config.transformer.num_hidden_layers =  6

model = FACTModel(audio_config, motion_config, multi_model_config, pred_length=20).to("cuda:0")
model.eval()

motion = torch.zeros((1, motion_config.sequence_length, 225)).to("cuda:0")
audio = torch.zeros((1, 2120, 35)).to("cuda:0")

inp = {"motion_input": motion, "audio_input": audio}

start = time.time()
with torch.no_grad():
    model.infer_auto_regressive(inp, steps=1000)
print(time.time() - start)

