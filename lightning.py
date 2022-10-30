import yaml
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer


from models.lightning_model import LitDehazeformer
from datasets.loader import PairLoader
from torch.utils.data import DataLoader


def main(args):
    config_dict = yaml.load(open(args.config), Loader=yaml.FullLoader)
    print(config_dict)
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
    train_loader = DataLoader(train_dataset,  batch_size=config_dict["trainer"]["batch_size"])
    val_dataset = PairLoader(**config_dict["datasets"]["train_dataset_params"])
    val_loader = DataLoader(val_dataset, batch_size=config_dict["trainer"]["batch_size"])
    del config_dict["trainer"]["batch_size"]
    wandb_logger = WandbLogger(**config_dict["wandb"])
    wandb_logger.watch(model, log="all", log_graph=True)
    trainer = Trainer(accelerator=args.accelerator, devices=args.devices, logger=wandb_logger,
                      **config_dict["trainer"])
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", default=None)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)
