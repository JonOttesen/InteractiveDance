import torch

import numpy as np

from dance.trainer.trainer import Trainer
from dance.loaders.dataloader import Dataloader

from dance.loaders.loader import AISTDataset

import warnings
warnings.filterwarnings("ignore")


dataset = AISTDataset("/home/jon/Documents/dance/data")

x, y, z = dataset.load_motion(dataset.motion_dir, "gBR_sBM_cAll_d04_mBR0_ch01")
print(x.shape, y.shape, z.shape)
print(np.min(x), np.max(x))
print(np.min(y), np.max(y))
print(np.min(z), np.max(z))