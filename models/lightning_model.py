import pytorch_lightning as pl
import torch
import torchmetrics

from torch import nn

import models


class Criterion:
    def __init__(self, criterion_list):
        criteria = []
        weights = []

        for criterion in criterion_list:
            criteria.append(eval(criterion["name"]))
            weights.append(criterion["weight"])
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
        network_module = gettatr(models, network_module)
        self.network = network_module(**network_params)
        self.criterion = self.build_criterion(criterion)
        optimizer_module = eval(optimizer_module)
        self.optimizer = optimizer_module(self.network.parameters(), **optimizer_params)
        scheduler_module = eval(scheduler_module)
        self.scheduler = scheduler_module(self.optimizer, scheduler_params)
        for metric in metrics:
            metrics[metric] = eval(metrics[metric])()
        self.metrics = metrics

    def training_step(self, batch, batch_idx):
        source_img, target_img = batch
        output = self.network(source_img)
        loss = criterion(output, target_img)
        self.log("train_loss", batch_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        source_img, target_img = batch
        # validation only on rgb channels
        output = self.network(source_img)[:, :3]
        result = {}
        for metric_name in self.metrics:
            result[metric_name] = self.metrics[metric_name](output, target_img)
        self.log(result)
        return result["PSNR"]

    def test_step(self, batch, batahc_idx):
        return self.validation_step(batch, batahc_idx)

    def forward(self, batch):
        return self.network(batch)

    def configure_optimizers(self):
        scheduler = {
            'scheduler': self.scheduler,
            'interval': 'epoch',
            'frequency': 1
        }
        return self.optimizer, scheduler
