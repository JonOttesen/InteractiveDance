import torch

import numpy as np

from dance.trainer.trainer import Trainer
from dance.loaders.dataloader import Dataloader

from dance.loaders.loader import AISTDataset

import warnings
warnings.filterwarnings("ignore")


dataset = AISTDataset("/home/jon/Documents/dance/data")

keypoints2d, bboxes, timestamps = dataset.load_keypoint2d(dataset.keypoint2d_dir, "gBR_sBM_cAll_d04_mBR0_ch06")
print(keypoints2d.shape)