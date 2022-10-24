from typing import Callable, Dict, Union, Optional, List
from collections import defaultdict
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

# from torchvision.utils import make_grid
# from base import BaseTrainer
# from utils import inf_loop, MetricTracker

from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: Callable,
                 metric_ftns: Dict[str, Callable],
                 optimizer,
                 lr_scheduler,
                 config: dict,
                 project: str,
                 data_loader: torch.utils.data.dataloader,
                 valid_data_loader: torch.utils.data.dataloader = None,
                 seed: int = None,
                 device: str = None,
                 tags: Optional[List[str]] = None,
                 log_step: int = None,
                 resume_id: str = None,
                 ):

        super().__init__(model=model,
                         loss_function=loss_function,
                         metric_ftns=metric_ftns,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         config=config,
                         seed=seed,
                         device=device,
                         tags=tags,
                         project=project,
                         resume_id=resume_id,
                         )

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

        self.inputs_pr_iteration = int(config['inputs_pr_iteration'])
        self.val_inputs_pr_iteration = int(config['val_inputs_pr_iteration'])

        self.len_epoch = len(data_loader) if not self.iterative else self.inputs_pr_iteration
        self.batch_size = data_loader.batch_size
        self.log_step = int(self.len_epoch/(4)) if not isinstance(log_step, int) else log_step

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        losses = defaultdict(list)

        for batch_idx, (data, music, target) in enumerate(self.data_loader):
            data, music, target = data.to(self.device), music.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()  # Clear before inputting to model

            inp = {
                "motion_input": data,
                "audio_input": music,
                }
            output = self.model(inp)
            
            loss = self.loss_function(output, target)

            loss.backward()
            self.optimizer.step()

            loss = loss.item()  # Detach loss from comp graph and moves it to the cpu
            losses['loss'].append(loss)

            if batch_idx % self.log_step == 0:
                self.logger.info('Train {}: {} {} Loss: {:.6f}'.format(
                    'Epoch' if not self.iterative else 'Iteration',
                    epoch,
                    self._progress(batch_idx),
                    loss))

            if batch_idx >= self.inputs_pr_iteration and self.iterative:
                break

        losses['loss'] = np.mean(losses['loss'])

        return losses

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.valid_data_loader is None:
            return None

        self.model.eval()
        metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, (data, music, target) in enumerate(self.valid_data_loader):
                data, music, target = data.to(self.device), music.to(self.device), target.to(self.device)
                inp = {
                    "motion_input": data,
                    "audio_input": music,
                    }

                output = self.model(inp)

                metrics['val_loss'].append(self.loss_function(output, target).item())

                for key, metric in self.metric_ftns.items():
                    metrics[key].append(metric(output, target).item())

                if batch_idx >= self.val_inputs_pr_iteration and self.iterative:
                    break
        metric_dict = dict()
        for key, item in metrics.items():
            metric_dict[key] = np.mean(metrics[key])

        return metric_dict

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx
            total = self.data_loader.n_samples
        elif hasattr(self.data_loader, 'batch_size'):
            current = batch_idx
            total = self.len_epoch
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
