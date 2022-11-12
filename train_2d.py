import torch

import numpy as np

from dance.trainer.trainer import Trainer
from dance.loaders.dataloader import Dataloader
from dance.models.fact.fact import FACTModel
from dance.models.fact.config import audio_config, fact_model, motion_config, multi_model_config

from dance.loaders.loader import AISTDataset

import warnings
warnings.filterwarnings("ignore")


dataset = AISTDataset("/itf-fi-ml/home/jonakri/dance/data")

train_loader = Dataloader(
    dataset, 
    "/itf-fi-ml/home/jonakri/dance/data/wav", 
    config={"audio_length": 180, "sequence_length": 90, "target_length": 20}, 
    split="train",
    method="2d",
    fps=30,
    )

# val_loader = Dataloader(
    # dataset, 
    # "/home/jon/Documents/dance/data/wav", 
    # config={"audio_length": 240, "sequence_length": 120, "target_length": 20}, 
    # split="val",
    # method="2d",
    # )



audio_config.transformer.intermediate_size = 1024
motion_config.transformer.intermediate_size = 1024
multi_model_config.transformer.intermediate_size = 1024
multi_model_config.transformer.num_hidden_layers =  4

motion_config.feature_dim = 34

audio_config.sequence_length = 180
motion_config.sequence_length = 90
multi_model_config.sequence_length = 180

model = FACTModel(audio_config, motion_config, multi_model_config, out_dim=34, pred_length=20)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)

config = {
    "name": "dance_gen",
    "epochs": 200,
    "num_hidden_layers": multi_model_config.transformer.num_hidden_layers,
    "intermediate_size": multi_model_config.transformer.intermediate_size,
    "iterative": True,
    "inputs_pr_iteration": 10000,
    "val_inputs_pr_iteration": 1000,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "optimizer": "Adam",
    "weight_decay": 0,
    "warmup_steps": 10,
    "lr_scheduler": "CosineAnnealingLR",
    "save_dir": "/itf-fi-ml/home/jonakri/dance/2D/30fps",
    "save_period": 10,
}

train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                           num_workers=8,
                                           batch_size=config["batch_size"],
                                           shuffle=True)


# valid_loader = torch.utils.data.DataLoader(dataset=val_loader,
                                        #    num_workers=8,
                                        #    batch_size=config["batch_size"],
                                        #    shuffle=True)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config['learning_rate'],  
    weight_decay=config["weight_decay"],
    )


class LRPolicy(object):
    def __init__(self, initial, warmup_steps=10):
        self.warmup_steps = warmup_steps
        self.initial = initial

    def __call__(self, step):
        return self.initial + step/self.warmup_steps*(1 - self.initial)

warmup_steps = config["warmup_steps"]

scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer,  LRPolicy(initial=1e-2, warmup_steps=warmup_steps))
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config['epochs'] - warmup_steps), eta_min=0, last_epoch=-1, verbose=False)

lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])
# lr_scheduler = config.lr_scheduler(optimizer=optimizer)


# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, \
    # T_max=config['epochs'] - config["warmup_steps"], eta_min=0, last_epoch=-1, verbose=False)


loss = torch.nn.L1Loss()
# loss = torch.nn.MSELoss()

trainer = Trainer(
    model=model,
    loss_function=loss,
    metric_ftns=None,
    config=config,
    data_loader=train_loader,
    valid_data_loader=None,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    seed=None,
    # log_step=2500,
    device='cuda:0',
    project="dance_gen_2d",
    tags=["tiny_30fps"],
    smpl_model="",
    # resume_id="elf7qts1"
    )

# trainer.resume_checkpoint("/itf-fi-ml/home/jonakri/daim/2022-09-17/best_validation/checkpoint-best.pth")


trainer.train()