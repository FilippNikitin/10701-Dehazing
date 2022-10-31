import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_msssim import ssim

from datasets.loader import PairLoader
from models import *
from utils import AverageMeter

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='GPUs used for training')
parser.add_argument('--wandb', action='store_true', default=False, help='')
parser.add_argument('--run_name', type=str, default='test_confidence', help='')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.wandb:
    wandb.init(
        entity='10701',
        settings=wandb.Settings(start_method="fork"),
        project='Image Dehazing',
        name=args.run_name,
        config=args
    )


def train(train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        with autocast(args.no_autocast):
            output = network(source_img)
            loss = criterion(output, target_img)

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def valid(val_loader, network):
    PSNR = AverageMeter()
    SSIM = []

    torch.cuda.empty_cache()

    network.eval()
    first_batch = True
    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        with torch.no_grad():  # torch.no_grad() may cause warning
            output = network(source_img).clamp_(-1, 1)

        if first_batch:
            max_img = 6
            hazed_images = wandb.Image(source_img[:max_img, :3], caption="Hazed Images")
            dehazed_images = wandb.Image(output[:max_img, :3], caption="DeHazed Images")
            target_images = wandb.Image(target_img[:max_img, :3], caption="DeHazed Images")
            wandb.log({"hazed": hazed_images, "dehazed": dehazed_images, "GT": target_images})
            first_batch = False

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()

        _, _, H, W = output.size()
        down_ratio = max(1, round(min(H, W) / 256))  # Zhou Wang
        output = output * 0.5 + 0.5
        target = target_img * 0.5 + 0.5
        ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                        F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                        data_range=1, size_average=False)
        SSIM.append(ssim_val)
        PSNR.update(psnr.item(), source_img.size(0))
    SSIM_avg = torch.mean(torch.cat(SSIM))
    return PSNR.avg, SSIM_avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)
    wandb.config = setting
    network = eval(args.model.replace('-', '_'))()
    wandb.watch(network, log="all", log_freq=10000, log_graph=True)
    network = nn.DataParallel(network).cuda()

    criterion = nn.L1Loss()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'], setting['edge_decay'],
                               setting['only_h_flip'], setting['quadruple_color_space'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                             setting['patch_size'], quadruple_color_space=setting['quadruple_color_space'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, args.model + '.pth')):
        print('==> Start training, current model name: ' + args.model)
        # print(network)

        best_psnr = 0
        best_ssim = 0
        for epoch in tqdm(range(setting['epochs'] + 1)):
            loss = train(train_loader, network, criterion, optimizer, scaler)
            loggs = {"train_loss": loss}

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr, avg_ssim = valid(val_loader, network)
                loggs["valid_psnr"] = avg_psnr
                loggs["valid_ssim"] = avg_ssim
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + '.pth'))
                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                loggs["best_psnr"] = best_psnr
                loggs["best_ssim"] = best_ssim
            wandb.log(loggs)

    else:
        print("Adding color space")
        print('==> Existing trained model')
        exit(1)
