from typing import Callable, Dict, Union, Optional, List
from collections import defaultdict
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from smplx import SMPL
except:
    pass
# from torchvision.utils import make_grid
# from base import BaseTrainer
# from utils import inf_loop, MetricTracker

from .base_trainer import BaseTrainer

from .scores import Scores


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: Callable,
                 optimizer,
                 lr_scheduler,
                 config: dict,
                 project: str,
                 smpl_model: str,
                 data_loader: torch.utils.data.dataloader,
                 valid_data_loader: torch.utils.data.dataloader = None,
                 seed: int = None,
                 device: str = None,
                 tags: Optional[List[str]] = None,
                 log_step: int = None,
                 resume_id: str = None,
                 dont_val: int = 0,
                 ):

        super().__init__(model=model,
                         loss_function=loss_function,
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
        self.dont_val = dont_val

        self.inputs_pr_iteration = int(config['inputs_pr_iteration'])
        self.val_inputs_pr_iteration = int(config['val_inputs_pr_iteration'])

        self.len_epoch = len(data_loader) if not self.iterative else self.inputs_pr_iteration
        self.batch_size = data_loader.batch_size
        self.log_step = int(self.len_epoch/(4)) if not isinstance(log_step, int) else log_step

        if self.valid_data_loader is not None:
            smpl = SMPL(model_path=smpl_model, gender='MALE', batch_size=1).to(self.device)
            self.scores = Scores(smpl)

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
            return dict()
        
        if epoch < self.dont_val:
            return dict() 

        self.model.eval()
        metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, (data, music) in enumerate(self.valid_data_loader):
                data, music= data.to(self.device), music.to(self.device)

                inp = {
                    "motion_input": data[:, :120],
                    "audio_input": music,
                    }

                output = self.model.infer_auto_regressive(inp, steps=1200, step_size=1).cpu().numpy()[0]

                result_motion = np.expand_dims(np.concatenate([
                    data[:, :120].cpu().numpy()[0],
                    output
                    ], axis=0), axis=0)  # [1, 120 + 1200, 225]

                pred_keypoints = self.scores.recover_motion_to_keypoints(result_motion, self.device)
                real_keypoints = self.scores.recover_motion_to_keypoints(data.cpu().numpy(), self.device)

                metrics["beat_align"].append(self.scores.beat_align(pred_keypoints, music.cpu().numpy()[0]))

                # fid_k, dist_k = self.scores.kinetic_fid(pred_keypoints, real_keypoints)
                # fid_g, dist_g = self.scores.manual_fid(pred_keypoints, real_keypoints)

                self.scores.accumulate_fid(pred_keypoints, real_keypoints)

                # metrics['fid_k'].append(fid_k)
                # metrics['dist_k'].append(dist_k)
                # metrics['fid_g'].append(fid_g)
                # metrics['dist_g'].append(dist_g)
                
                # metrics['val_loss'].append(fid_k + fid_g)

                if batch_idx >= self.val_inputs_pr_iteration and self.iterative:
                    break

        metric_dict = dict()
        for key, item in metrics.items():
            metric_dict[key] = np.mean(metrics[key])
        
        FID_k, FID_g, Dist_k, Dist_g = self.scores.fid()
        metric_dict["val_loss"] = FID_g + FID_k

        metric_dict['fid_k'] = FID_k
        metric_dict['dist_k'] = Dist_k
        metric_dict['fid_g'] = FID_g
        metric_dict['dist_g'] = Dist_g

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
