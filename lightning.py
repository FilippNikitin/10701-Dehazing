import yaml

from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
from models.lightning_model import LitDehazeformer


def main(hparams):
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    wandb_logger = WandbLogger()
    trainer = Trainer(logger=wandb_logger)
    network_module = config_dict["model"]["module_name"]
    network_params = config_dict["model"]["params"]
    optimizer_module = config_dict["optimization"]["optimizer_module_name"]
    optimizer_params = config_dict["optimization"]["optimizer_params"]
    scheduler_module = config_dict["optimization"]["scheduler_module_name"]
    scheduler_params = config_dict["optimization"]["scheduler_params"]
    criterion = config_dict["criterion"]
    metrics = config_dict["metrics"]

    model = LitDehazeformer(network_module, network_params, criterion, optimizer_module,
                            optimizer_params, scheduler_module, scheduler_params, metrics)

    train_dataset = PairLoader(**config_dict["datasets"]["train_dataset_params"])
    train_loader = DataLoader(train_dataset)
    val_dataset = PairLoader(**config_dict["datasets"]["train_dataset_params"])
    val_loader = DataLoader(val_dataset)

    trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices, **config_dict["trainer"]["params"])
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", default=None)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)


