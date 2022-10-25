from email.mime import audio
import torch

import numpy as np

from dance.trainer.trainer import Trainer
from dance.loaders.dataloader import Dataloader
from dance.models.fact.fact import FACTModel
from dance.models.fact.config import audio_config, fact_model, motion_config, multi_model_config

from dance.loaders.loader import AISTDataset

import warnings
warnings.filterwarnings("ignore")


dataset = AISTDataset("/mnt/CRAI-NAS/all/jona/dance_data")

train_loader = Dataloader(
    dataset, 
    "/mnt/CRAI-NAS/all/jona/dance_data/wav", 
    None, 
    config={"audio_length": 240, "sequence_length": 120, "target_length": 20}, 
    keypoint_dir="motions",
    split="train"
    )

val_loader = Dataloader(
    dataset, 
    "/mnt/CRAI-NAS/all/jona/dance_data/wav", 
    None, 
    config={"audio_length": 240, "sequence_length": 120, "target_length": 20}, 
    keypoint_dir="motions",
    split="val"
    )

metrics = {
    'MSE': torch.nn.MSELoss(),
    'L1': torch.nn.L1Loss(),
    }

audio_config.transformer.intermediate_size = 1536
motion_config.transformer.intermediate_size = 1536
multi_model_config.transformer.intermediate_size = 1536
multi_model_config.transformer.num_hidden_layers =  6

model = FACTModel(audio_config, motion_config, multi_model_config, pred_length=20)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)

config = {
    "name": "dance_gen",
    "epochs": 100,
    "num_hidden_layers": multi_model_config.transformer.num_hidden_layers,
    "intermediate_size": multi_model_config.transformer.intermediate_size,
    "iterative": True,
    "inputs_pr_iteration": 10000,
    "val_inputs_pr_iteration": 2000,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "optimizer": "Adam",
    "lr_scheduler": "CosineAnnealingLR",
    "save_dir": "/mnt/CRAI-NAS/all/jona/dance_models/small",
    "save_period": 10,
}

train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                           num_workers=8,
                                           batch_size=config["batch_size"],
                                           shuffle=True)


valid_loader = torch.utils.data.DataLoader(dataset=val_loader,
                                           num_workers=8,
                                           batch_size=config["batch_size"],
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
    device='cuda:1',
    project="dance_gen",
    tags=["small"],
    # resume_id="elf7qts1"
    )

# trainer.resume_checkpoint("/itf-fi-ml/home/jonakri/daim/2022-09-17/best_validation/checkpoint-best.pth")


trainer.train()