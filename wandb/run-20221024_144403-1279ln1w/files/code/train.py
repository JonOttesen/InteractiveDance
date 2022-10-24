import torch

import numpy as np

from dance.trainer.trainer import Trainer
from dance.loaders.dataloader import Dataloader
from dance.models.fact.fact import FACTModel
from dance.models.fact.config import audio_config, motion_config, multi_model_config

from aist_plusplus.loader import AISTDataset

import warnings
warnings.filterwarnings("ignore")



dataset = AISTDataset("/home/jon/Documents/dance/data")

train_loader = Dataloader(
    dataset, 
    "/home/jon/Documents/dance/data/wav", 
    None, 
    config={"audio_length": 240, "sequence_length": 120, "target_length": 20}, 
    keypoint_dir="motions",
    split="train"
    )

val_loader = Dataloader(
    dataset, 
    "/home/jon/Documents/dance/data/wav", 
    None, 
    config={"audio_length": 240, "sequence_length": 120, "target_length": 20}, 
    keypoint_dir="motions",
    split="val"
    )

metrics = {
    'MSE': torch.nn.MSELoss(),
    'L1': torch.nn.L1Loss(),
    }


model = FACTModel(audio_config, motion_config, multi_model_config, pred_length=20)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)

config = {
    "name": "dance_gen",
    "epochs": 100,
    "iterative": True,
    "inputs_pr_iteration": 10000,
    "val_inputs_pr_iteration": 2000,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "optimizer": "Adam",
    "lr_scheduler": "CosineAnnealingLR",
    "save_dir": "/home/jon/Documents/dance",
    "save_period": 10,
}

train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                           num_workers=8,
                                           batch_size=config["batch_size"],
                                           shuffle=True)


valid_loader = torch.utils.data.DataLoader(dataset=val_loader,
                                           num_workers=8,
                                           batch_size=1,
                                           shuffle=True)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config['learning_rate'],  
    weight_decay=0,
    )


lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, \
    T_max=config['epochs'], eta_min=0, last_epoch=-1, verbose=False)


loss = torch.nn.L1Loss()

trainer = Trainer(
    model=model,
    loss_function=loss,
    metric_ftns=metrics,
    config=config,
    data_loader=train_loader,
    valid_data_loader=valid_loader,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    seed=None,
    # log_step=2500,
    device='cuda:0',
    project="dance_gen",
    tags=["test"],
    # resume_id="elf7qts1"
    )

# trainer.resume_checkpoint("/itf-fi-ml/home/jonakri/daim/2022-09-17/best_validation/checkpoint-best.pth")


trainer.train()