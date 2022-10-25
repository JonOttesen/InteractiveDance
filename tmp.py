import torch

import numpy as np

from dance.models.fact.fact import FACTModel
from dance.models.fact.config import audio_config, fact_model, motion_config, multi_model_config


import warnings
warnings.filterwarnings("ignore")



# audio_config.transformer.intermediate_size = 1536
# motion_config.transformer.intermediate_size = 1536
# multi_model_config.transformer.intermediate_size = 1536
# multi_model_config.transformer.num_hidden_layers =  6

model = FACTModel(audio_config, motion_config, multi_model_config, pred_length=20)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)


