import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from torch import nn

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

    def training_step(self, batch, batch_idx):
        source_img = batch["source"]
        target_img = batch["target"]

        output = self.network(source_img)
        loss = self.criterion(output, target_img)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        source_img = batch["source"]
        target_img = batch["target"]
        # validation only on rgb channels
        output = self.network(source_img)[:, :3]
        target_img = target_img[:, :3]
        if batch_idx == 0:
            max_img = 6
            hazed_images = wandb.Image(source_img[:max_img, :3], caption="Hazed Images")
            dehazed_images = wandb.Image(output[:max_img, :3], caption="DeHazed Images")
            target_images = wandb.Image(target_img[:max_img, :3], caption="DeHazed Images")
            wandb.log({"hazed": hazed_images, "dehazed": dehazed_images, "GT": target_images})

        result = {}
        for metric_name in self.metrics:
            result[metric_name] = self.metrics[metric_name](output, target_img)
        self.log_dict(result, on_step=False, on_epoch=True)
        return result

    def test_step(self, batch, batahc_idx):
        return self.validation_step(batch, batahc_idx)

    def forward(self, batch):
        return self.network(batch)

    def configure_optimizers(self):
        return [self.optimizer, ], [self.scheduler, ]
