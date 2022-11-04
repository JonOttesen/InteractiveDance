import torch

import numpy as np

from dance.trainer.trainer import Trainer
from dance.loaders.dataloader import Dataloader

from dance.loaders.loader import AISTDataset

import warnings
warnings.filterwarnings("ignore")


dataset = AISTDataset("/home/jon/Documents/dance/data")

loader = Dataloader(
    dataset, 
    "/home/jon/Documents/dance/data/wav", 
    config={"audio_length": 240, "sequence_length": 120, "target_length": 20}, 
    split="train",
    method="2d"
    )
keypoints2d, _, _ = dataset.load_keypoint2d(dataset.keypoint2d_dir, "gBR_sBM_cAll_d04_mBR0_ch01")

print(np.moveaxis(keypoints2d, 0, 1).shape)
print(np.max(keypoints2d[:, :, 0]), np.min(keypoints2d[:, :, 0]))
print(np.max(keypoints2d[:, :, 1]), np.min(keypoints2d[:, :, 1]))