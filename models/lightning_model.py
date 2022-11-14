import pytorch_lightning as pl
import torch
import torchmetrics
import wandb

from torch import nn
from thop import profile

import models


class Criterion(nn.Module):
    def __init__(self, criterion_dict):
        super().__init__()
        criteria = []
        weights = []

        for criterion in criterion_dict:
            criteria.append(eval(criterion_dict[criterion]["module_name"]))
            weights.append(criterion_dict[criterion]["weight"])
        self.criteria = nn.ModuleList(criteria)
        self.weights = weights

    def forward(self, predicted, target):
        loss = 0
        for criterion, weight in zip(self.criteria, self.weights):
            loss += criterion(predicted, target) * weight
        return loss


class LitDehazeformer(pl.LightningModule):
    def __init__(self, network_module, network_params, criterion, optimizer_module, optimizer_params,
                 scheduler_module, scheduler_params, metrics):
        super().__init__()
        self.save_hyperparameters()
        network_module = getattr(models, network_module)
        self.network = network_module(**network_params)
        self.criterion = Criterion(criterion_dict=criterion)
        optimizer_module = eval(optimizer_module)
        self.optimizer = optimizer_module(self.network.parameters(), **optimizer_params)
        scheduler_module = eval(scheduler_module)
        self.scheduler = scheduler_module(self.optimizer, **scheduler_params)
        for metric in metrics:
            metrics[metric] = eval(metrics[metric]["module"])(**metrics[metric]["params"])
        metrics = nn.ModuleDict(metrics)
        self.metrics = metrics
        self.best_metrics = {key: 0 for key in metrics}
        input = torch.randn(1, network_params["in_chans"], 224, 224)
        macs, params = profile(self.network, inputs=(input, ))
        logs = {'Computational complexity': macs, 'Number of parameters': params}
        self.epoch_log = logs

    def training_step(self, batch, batch_idx):
        source_img = batch["source"]
        target_img = batch["target"]
        output = self.network(source_img)
        loss = self.criterion(output, target_img)
        return loss

    def training_epoch_end(self, training_step_outs):
        losses = [i["loss"] for i in training_step_outs]
        loss = torch.mean(torch.stack(losses))
        self.epoch_log["train_loss"] = loss

    def validation_step(self, batch, batch_idx):
        source_img = batch["source"]
        target_img = batch["target"]
        # validation only on rgb channels
        # convert [-1, 1] to [0, 1]
        output = (self.network(source_img)[:, :3] + 1) * 0.5
        target_img = (target_img[:, :3] + 1) * 0.5
        if batch_idx == 0:
            max_img = 6
            hazed_images = wandb.Image(source_img[:max_img, :3], caption="Hazed Images")
            dehazed_images = wandb.Image(output[:max_img, :3], caption="DeHazed Images")
            target_images = wandb.Image(target_img[:max_img, :3], caption="DeHazed Images")
            log_images = {"hazed": hazed_images, "dehazed": dehazed_images, "GT": target_images}
            self.epoch_log = {**self.epoch_log, **log_images}

        result = {}
        # todo: understand how the precision works in pytorch lightning
        target_img = target_img.to(output.dtype)
        for metric_name in self.metrics:
            result[metric_name] = self.metrics[metric_name](output, target_img)
        return result

    def validation_epoch_end(self, validation_step_outs):
        result = {}

        for val_out in validation_step_outs:
            for key in val_out:
                if key in result:
                    result[key].append(val_out[key])
                else:
                    result[key] = [val_out[key], ]
        for key in result:
            result[key] = torch.mean(torch.cat(result[key]))

        for key in self.best_metrics:
            if result[key] > self.best_metrics[key]:
                self.best_metrics[key] = result[key]
            # todo: redo all of the staff bellow
            metric_name = key.split("_")[-1]
            result[f"best_{metric_name}"] = self.best_metrics[key]

        log = {**self.epoch_log, **result}
        wandb.log(log)
        self.epoch_log = {}

    def test_step(self, batch, batahc_idx):
        return self.validation_step(batch, batahc_idx)

    def forward(self, batch):
        return self.network(batch)

    def configure_optimizers(self):
        return [self.optimizer, ], [self.scheduler, ]
