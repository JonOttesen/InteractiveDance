import os
import time
from librosa import beat
import torch
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import scipy.signal as scisignal

from dance.loaders.loader import AISTDataset
from dance.loaders.dataloader import Dataloader

from dance.models.fact.fact import FACTModel
from dance.models.fact.config import audio_config, fact_model, motion_config, multi_model_config


def main():

    audio_config.transformer.intermediate_size = 1024
    motion_config.transformer.intermediate_size = 1024
    multi_model_config.transformer.intermediate_size = 1024
    multi_model_config.transformer.num_hidden_layers =  4

    model = FACTModel(audio_config, motion_config, multi_model_config, pred_length=20)
    model = model.to("cuda:0")
    model.eval()

    model_path = model_path = "/home/jon/Documents/dance/checkpoint-best.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])

    path = "/home/jon/Documents/dance/data/"

    # calculate score on real data
    dataset = AISTDataset(path)

    loader = Dataloader(
        dataset, 
        "/home/jon/Documents/dance/data/wav", 
        None, 
        config={"audio_length": 240, "sequence_length": 120, "target_length": 20}, 
        keypoint_dir="motions",
        no_preprocessed=False,
        return_smpl=False,
        split="train",
        )

    test_loader = torch.utils.data.DataLoader(
        dataset=loader,
        num_workers=8,
        batch_size=4,
        shuffle=True,
        )

    for i, (motion, audio, target) in enumerate(test_loader):
        print(i)

    for i, (motion, audio, target) in enumerate(loader):
        # get real data motion beats
        continue
        inp = {"motion_input": motion.unsqueeze(0).to("cuda:0"), "audio_input": audio.unsqueeze(0).to("cuda:0")}
        with torch.no_grad():
            pred = model(inp).cpu().numpy()[0]
        print(np.mean(np.abs(pred - target.numpy()), axis=-1))
        
        with torch.no_grad():
            pred = model.infer_auto_regressive(inp, steps=20).cpu().numpy()[0]
        print(np.mean(np.abs(pred - target.numpy()), axis=-1))
        print("-------------------------------------")

main()