import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

# Related to DDP (Distributed Data Parallelism)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import lpips

import argparse
from tqdm.auto import tqdm
import os, re
import glob
import numpy as np
from PIL import Image
import json

import random
import network_training_utils
from models.DA_loss_functions import DA_loss
from pytorch_msssim import ms_ssim
from FDL_pytorch import FDL_loss
from models.localnet_positional_encoding import ParamLocal
from utils.utils import *

# -------------------------------------------------------------------------
# DDP Helper Functions
# -------------------------------------------------------------------------
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
# -------------------------------------------------------------------------

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=5e-3, training=True):
        self.mean = mean
        self.std = std
        self.training = training

    def __call__(self, tensor):
        if not self.training:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std + self.mean

class PairedDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, paired_transform=None):
        self.root = root_dir
        self.transform = transform
        self.paired_transform = paired_transform

        def nkey(s):  # Natural sorting (simple)
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

        def rel_list(root):
            return sorted(
                [os.path.relpath(os.path.join(dp, f), root).replace("\\", "/")
                 for dp, _, fs in os.walk(root) for f in fs if f.lower().endswith(".png")],
                key=nkey
            )

        meta_root = os.path.join(root_dir, "Meta_camera")
        gt_root   = os.path.join(root_dir, "GT_camera")

        meta_rel = rel_list(meta_root)
        gt_rel   = rel_list(gt_root)

        # Assume same structure: use only common paths
        common = sorted(set(meta_rel) & set(gt_rel), key=nkey)

        self.meta_root, self.gt_root = meta_root, gt_root
        self.relpaths = common

    def __len__(self):
        return len(self.relpaths)

    def __getitem__(self, idx):
        rel = self.relpaths[idx]
        inp = Image.open(os.path.join(self.meta_root, rel))
        tgt = Image.open(os.path.join(self.gt_root,   rel))

        if self.transform:
            inp = self.transform(inp)[:3, ...]
            tgt = self.transform(tgt)[:3, ...]
        if self.paired_transform:
            inp, tgt = self.paired_transform(inp, tgt)
        return inp, tgt
    
class PairedRandomTransformWithPositionalEncoding:
    def __init__(self, crop_size, full_size=(1848, 1848), eval=False):
        self.crop_size = crop_size  # (H_crop, W_crop)
        self.full_size = full_size  # (H_full, W_full)
        self.jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
        self.eval = eval

    def __call__(self, img1, img2):
        # Make full positional encoding
        H_full, W_full = self.full_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H_full),
            torch.linspace(-1, 1, W_full),
            indexing='ij'
        )
        rr = torch.sqrt(xx**2 + yy**2)

        k_list = [2**i for i in range(8)]
        
        pos = [torch.sin(k*np.pi*rr) for k in k_list] + \
            [torch.cos(k*np.pi*rr) for k in k_list] + [yy/np.sqrt(2), xx/np.sqrt(2), rr/np.sqrt(2)]

        pos = torch.stack(pos, dim=0)  # (2, H_full, W_full)

        # Sample transform params
        flip_h = random.random() > 0.5
        flip_v = random.random() > 0.5
        angle = random.choice([0, 90, 180, 270])

        # Random crop coords
        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=self.crop_size)

        # Define helper to apply transforms to all
        def apply_transforms(x):
            if flip_h: x = TF.hflip(x)
            if flip_v: x = TF.vflip(x)
            x = torch.rot90(x, angle // 90, dims=(-2, -1))
            x = TF.crop(x, i, j, h, w)
            return x

        if not self.eval:
            img1 = apply_transforms(img1)
            img2 = apply_transforms(img2)
            pos = apply_transforms(pos)

            # Color jitter (input only, and must be PIL input)
            img1_pil_aug = transforms.ToPILImage()(img1)
            img2_pil_aug = transforms.ToPILImage()(img2)
            img1_pil_aug = self.jitter(img1_pil_aug)
            img2_pil_aug = self.jitter(img2_pil_aug)
            img1 = TF.to_tensor(img1_pil_aug)
            img2 = TF.to_tensor(img2_pil_aug)

        # Add positional encoding to input
        img1_position_encoded = torch.cat([img1, pos], dim=-3)  # (C+2, H, W)

        return img1_position_encoded, img2
        
def train_step(args, epoch, step, net, net_optimizer, batch_data, eval=False, writer=None):
    input, target = batch_data
    input = input.to(args.device)
    target = target.to(args.device)

    if eval:
        net = net.eval()

    loss_items = {}

    pred = net(input.to(torch.float32))
    l1_loss = args.l1_criterion(pred, target)
    loss = l1_loss
    loss_items['l1_loss'] = l1_loss.item()

    if args.use_perc_loss:
        perc_loss = torch.mean(args.perceptual_loss_weight * 
                            args.perceptual_criterion(2 * pred.to(torch.float32) - 1, 2 * target.to(torch.float32) - 1))
        loss = loss + perc_loss * args.perceptual_loss_weight
        loss_items['perc_loss'] = perc_loss.item() * args.perceptual_loss_weight

    if args.use_da_loss:
        da_loss = DA_loss(pred, target, args, feature='segmentation')
        loss = loss + da_loss * args.da_loss_weight
        loss_items['da_loss'] = da_loss.item() * args.da_loss_weight

    if args.use_ssim_loss:
        ssim_loss = 1 - ms_ssim(pred, target, data_range=1.0)
        loss = loss + ssim_loss * args.ssim_loss_weight
        loss_items['ssim_loss'] = ssim_loss.item() * args.ssim_loss_weight

    if args.use_fdl_loss:
        fdl_loss = args.fdl_criterion(pred, target)
        loss = loss + fdl_loss * args.fdl_loss_weight
        loss_items['fdl_loss'] = fdl_loss.item() * args.fdl_loss_weight

    if args.use_cobi_loss:
        cobi_loss = network_training_utils.contextual_bilateral_loss(pred, target, args.cobi_loss_weight_sp, args.cobi_loss_weight_band_width)
        loss = loss + cobi_loss * args.cobi_loss_weight
        loss_items['cobi_loss'] = cobi_loss.item() * args.cobi_loss_weight

    if eval and (writer is not None):
        if step==1:
            if input.shape[1] > 3:
                input = input[:, :3, :, :]
            writer.add_image('pred', pred[0].cpu().detach(), epoch)
            writer.add_image('input', input[0].cpu().detach(), epoch)
            writer.add_image('target', target[0].cpu().detach(), epoch)
    elif not eval:
        loss.backward() 
        net_optimizer.step()
        net_optimizer.zero_grad()
        if step%30==0:
            # Log loss for TensorBoard
            if writer is not None:
                writer.add_scalar('L1 Loss step/train', l1_loss, step)
                if args.use_da_loss:
                    writer.add_scalar('DA_loss step/train', da_loss, step)
                if args.use_perc_loss:
                    writer.add_scalar('perc_loss step/train', perc_loss, step)
                if args.use_ssim_loss:
                    writer.add_scalar('ssim_loss step/train', ssim_loss, step)
                if args.use_fdl_loss:
                    writer.add_scalar('fdl_loss step/train', fdl_loss, step)
                if args.use_cobi_loss:
                    writer.add_scalar('cobi_loss step/train', cobi_loss, step)
                psnr = compute_psnr(target, pred, data_range=1.0)[1]
                ssim = compute_ssim(target, pred, data_range=1.0)[1]
                writer.add_scalar('PSNR step/train', psnr, step)
                writer.add_scalar('SSIM step/train', ssim, step)
    # Image metric
    psnr = compute_psnr(target, pred, data_range=1.0)[1]
    ssim = compute_ssim(target, pred, data_range=1.0)[1]
    loss_items['psnr'] = psnr
    loss_items['ssim'] = ssim

    loss_items['total_loss'] = loss.item()
    return loss_items


def train_network(args, continue_training):
    # DDP: Model Wrap
    net = ParamLocal(args).to(args.device)
    
    # Activate the following line if model contains BatchNorm
    # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    # Wrap with DDP
    # device_ids are GPU IDs used by the current process
    net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    net_optimizer = optim.AdamW(params=net.parameters(), lr=args.network_lr)
    if args.use_warm_up_stage:
        warmup_scheduler = optim.lr_scheduler.LinearLR(net_optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(net_optimizer, T_max=args.T_max, eta_min=args.network_lr/100)
        net_scheduler = optim.lr_scheduler.SequentialLR(net_optimizer, 
                                                        schedulers=[warmup_scheduler, main_scheduler], 
                                                        milestones=[args.warmup_epochs]) 
    else:
        net_scheduler = optim.lr_scheduler.CosineAnnealingLR(net_optimizer, T_max=args.T_max, eta_min=args.network_lr/100)

    # Training resumption
    if continue_training:
        newest_training_state = torch.load(network_training_utils.find_latest_checkpoint(args.result_path), map_location=f'cuda:{args.local_rank}') 
        net.load_state_dict(newest_training_state['net_state_dict'])
        net_optimizer.load_state_dict(newest_training_state['net_optimizer_state_dict'])
        net_scheduler.load_state_dict(newest_training_state['net_scheduler_state_dict'])
        total_step = newest_training_state['total_step']
        start_epoch = newest_training_state['epoch']
        eval_minimum_loss = newest_training_state['eval_minimum_loss']
    else:
        if not (args.network_dir is None):
            print(f'Loading from {args.network_dir}')
            net.module.load_state_dict(torch.load(args.network_dir, map_location=args.device))
        total_step = 0
        start_epoch = 0
        eval_minimum_loss = np.inf

    net = net.train()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    if is_main_process():
        writer = SummaryWriter(log_dir=args.result_path + '/runs')
    else:
        writer = None

    transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(),
        ])
    
    transform_test = transforms.Compose([
            transforms.Resize((args.crop_patch_size, args.crop_patch_size)) if args.batch_size==1 else transforms.Resize((args.full_res, args.full_res)),
            transforms.ToTensor(),
        ])

    trainset = PairedDataset(args.training_data_directory, transform=transform_train, 
                                    paired_transform=PairedRandomTransformWithPositionalEncoding((args.crop_patch_size, args.crop_patch_size),
                                                                                                (args.full_res, args.full_res), eval=False))
    testset = PairedDataset(args.test_data_directory, transform=transform_test,
                                    paired_transform=PairedRandomTransformWithPositionalEncoding((args.crop_patch_size, args.crop_patch_size),
                                                                                                   (args.full_res, args.full_res), eval=True))

    num_workers = 4

    # DDP: set DistributedSampler (Devide data with number of GPUs)
    train_sampler = DistributedSampler(trainset, shuffle=True)
    test_sampler = DistributedSampler(testset, shuffle=False) # Device validation (metric is computed by each node)

    # DDP: shuffle=False at DataLoader (Sampler deals with), inject sampler
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=False, # Must be False
                                              sampler=train_sampler,
                                              num_workers=num_workers, 
                                              pin_memory=True,
                                              persistent_workers=True,
                                              prefetch_factor=4,
                                              drop_last=True)
                                              
    testloader = torch.utils.data.DataLoader(testset, batch_size=max(args.batch_size//4, 1), 
                                             shuffle=False,
                                             sampler=test_sampler,
                                             num_workers=num_workers//4, 
                                             pin_memory=True,
                                             persistent_workers=True,
                                             prefetch_factor=4)
    
    args.l1_criterion = nn.L1Loss().to(args.device)
    if args.use_perc_loss:
        args.perceptual_criterion = lpips.LPIPS(net='vgg').to(args.device)
    if args.use_fdl_loss:
        args.fdl_criterion = FDL_loss(phase_weight=args.fdl_loss_phase_weight).to(args.device)
    
    # Print tqdm in only the main process
    iterator = range(start_epoch, args.n_epochs)
    if is_main_process():
        iterator = tqdm(iterator, position=0, leave=False)

    for epoch in iterator:
        # DDP: set epoch to Sampler at every epoch (Change shuffle seed)
        train_sampler.set_epoch(epoch)

        # tqdm for loader
        train_loader_iter = trainloader
        if is_main_process():
            train_loader_iter = tqdm(trainloader)
        step_loss_accum = 0
        for step, batch_data in enumerate(tqdm(train_loader_iter)):
            step_losses = train_step(args, epoch, total_step, net, net_optimizer, batch_data, eval=False, writer=writer)
            step_loss_accum = step_loss_accum + step_losses["total_loss"]
            total_step += 1
            if args.verbose:
                if is_main_process():
                    print(f'\rEpoch {epoch} step {step} total step {total_step} Loss: {step_losses["total_loss"]:.4f}, lr: {net_optimizer.param_groups[0]["lr"]}', flush=True)
        if is_main_process():
            print(f'\rEpoch {epoch} total step {total_step} step loss: {step_loss_accum/len(train_loader_iter):.4f}, lr: {net_optimizer.param_groups[0]["lr"]}', flush=True)
        # end
        if epoch%args.log_freq==0 and epoch>0:
            with torch.no_grad():
                eval_loss = 0
                eval_psnr = 0
                eval_ssim = 0

                eval_l1_loss = 0
                eval_da_loss = 0
                eval_perc_loss = 0
                eval_ssim_loss = 0
                eval_fdl_loss = 0
                eval_cobi_loss = 0

                test_sampler.set_epoch(epoch)
                test_loader_iter = testloader
                if is_main_process():
                    test_loader_iter = tqdm(testloader)

                for step, batch_data in enumerate(test_loader_iter):
                    eval_step_losses = train_step(args, epoch, step, net, net_optimizer, batch_data, eval=True, writer=writer)
                    eval_loss = eval_loss + eval_step_losses['total_loss']
                    eval_psnr = eval_psnr + eval_step_losses['psnr']
                    eval_ssim = eval_ssim + eval_step_losses['ssim']
                    if 'l1_loss' in step_losses:
                        eval_l1_loss = eval_l1_loss + eval_step_losses['l1_loss']
                    if 'da_loss' in step_losses:
                        eval_da_loss = eval_da_loss + eval_step_losses['da_loss']
                    if 'perc_loss' in step_losses:
                        eval_perc_loss = eval_perc_loss + eval_step_losses['perc_loss']
                    if 'ssim_loss' in step_losses:
                        eval_ssim_loss = eval_ssim_loss + eval_step_losses['ssim_loss']
                    if 'fdl_loss' in step_losses:
                        eval_fdl_loss = eval_fdl_loss + eval_step_losses['fdl_loss']
                    if 'cobi_loss' in step_losses:
                        eval_cobi_loss = eval_cobi_loss + eval_step_losses['cobi_loss']

                # DDP: Average evaluation losses and metrics (args.device is already set to current processes' cuda number)
                eval_loss = eval_loss.clone().to(args.device)           if isinstance(eval_loss, torch.Tensor) else torch.tensor(eval_loss, device=args.device)
                eval_l1_loss = eval_l1_loss.clone().to(args.device)     if isinstance(eval_l1_loss, torch.Tensor) else torch.tensor(eval_l1_loss, device=args.device)
                eval_da_loss = eval_da_loss.clone().to(args.device)     if isinstance(eval_da_loss, torch.Tensor) else torch.tensor(eval_da_loss, device=args.device)
                eval_perc_loss = eval_perc_loss.clone().to(args.device) if isinstance(eval_perc_loss, torch.Tensor) else torch.tensor(eval_perc_loss, device=args.device)
                eval_ssim_loss = eval_ssim_loss.clone().to(args.device) if isinstance(eval_ssim_loss, torch.Tensor) else torch.tensor(eval_ssim_loss, device=args.device)
                eval_fdl_loss = eval_fdl_loss.clone().to(args.device)   if isinstance(eval_fdl_loss, torch.Tensor) else torch.tensor(eval_fdl_loss, device=args.device)
                eval_cobi_loss = eval_cobi_loss.clone().to(args.device) if isinstance(eval_cobi_loss, torch.Tensor) else torch.tensor(eval_cobi_loss, device=args.device)
                eval_psnr = eval_psnr.clone().to(args.device)           if isinstance(eval_psnr, torch.Tensor) else torch.tensor(eval_psnr, device=args.device)
                eval_ssim = eval_ssim.clone().to(args.device)           if isinstance(eval_ssim, torch.Tensor) else torch.tensor(eval_ssim, device=args.device)
                dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_l1_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_da_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_perc_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_ssim_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_fdl_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_cobi_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_psnr, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_ssim, op=dist.ReduceOp.SUM)
                eval_loss = eval_loss / dist.get_world_size()
                eval_l1_loss = eval_l1_loss / dist.get_world_size()
                eval_da_loss = eval_da_loss / dist.get_world_size()
                eval_perc_loss = eval_perc_loss / dist.get_world_size()
                eval_ssim_loss = eval_ssim_loss / dist.get_world_size()
                eval_fdl_loss = eval_fdl_loss / dist.get_world_size()
                eval_cobi_loss = eval_cobi_loss / dist.get_world_size()
                eval_psnr = eval_psnr / dist.get_world_size()
                eval_ssim = eval_ssim / dist.get_world_size()

                # Save the best model & Logging - ONLY MAIN PROCESS
                if is_main_process():
                    if 'l1_loss' in step_losses:
                        writer.add_scalar('L1 Loss epoch/eval', eval_l1_loss/len(testloader), epoch)
                    if 'da_loss' in step_losses:
                        writer.add_scalar('DA_loss epoch/eval', eval_da_loss/len(testloader), epoch)
                    if 'perc_loss' in step_losses:
                        writer.add_scalar('perc_loss epoch/eval', eval_perc_loss/len(testloader), epoch)
                    if 'ssim_loss' in step_losses:
                        writer.add_scalar('ssim_loss epoch/eval', eval_ssim_loss/len(testloader), epoch)
                    if 'fdl_loss' in step_losses:
                        writer.add_scalar('fdl_loss epoch/eval', eval_fdl_loss/len(testloader), epoch)
                    if 'cobi_loss' in step_losses:
                        writer.add_scalar('cobi_loss epoch/eval', eval_cobi_loss/len(testloader), epoch)
                    writer.add_scalar('Eval Loss epoch/eval', eval_loss/len(testloader), epoch)
                    writer.add_scalar('Eval PSNR epoch/eval', eval_psnr/len(testloader), epoch)
                    writer.add_scalar('Eval SSIM epoch/eval', eval_ssim/len(testloader), epoch)

                    if eval_loss < eval_minimum_loss:
                        eval_minimum_loss = eval_loss
                        # When saving with DDP: use net.module.state_dict() (remove DDP wrapper)
                        checkpoint_state = {
                            'epoch': epoch + 1,
                            'total_step': total_step,
                            'eval_minimum_loss': eval_minimum_loss,
                            'net_state_dict': net.module.state_dict(), # .module is important
                            'net_optimizer_state_dict': net_optimizer.state_dict(),
                            'net_scheduler_state_dict': net_scheduler.state_dict(),
                        }
                        torch.save(checkpoint_state, os.path.join(args.result_path,f'training_state_minimum_eval_loss.pt'))

                    # Save the best model
                    if eval_loss < eval_minimum_loss:
                        eval_minimum_loss = eval_loss
                        checkpoint_state = {
                            'epoch': epoch + 1,  # Next epoch to start
                            'total_step': total_step,
                            'eval_minimum_loss': eval_minimum_loss,
                            'net_state_dict': net.state_dict(),
                            'net_optimizer_state_dict': net_optimizer.state_dict(),
                            'net_scheduler_state_dict': net_scheduler.state_dict(),
                        }
                        torch.save(checkpoint_state, os.path.join(args.result_path,f'training_state_minimum_eval_loss.pt'))
        # end
        net_scheduler.step()

        # save training states
        if epoch%args.save_freq==0 and epoch>0 and is_main_process():
            checkpoint_state = {
                'epoch': epoch + 1,  # Next epoch to start
                'total_step': total_step,
                'eval_minimum_loss': eval_minimum_loss,
                'net_state_dict': net.state_dict(),
                'net_optimizer_state_dict': net_optimizer.state_dict(),
                'net_scheduler_state_dict': net_scheduler.state_dict(),
            }
            torch.save(checkpoint_state, os.path.join(args.result_path,'training_state_%03d.pt' % (epoch)))
    
def main():
    parser = argparse.ArgumentParser(
        description='paramISP finetuning',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    def none_or_str(value):
        if value == 'None':
            return None
        return value
        
    parser.add_argument('--verbose', action="store_true", help='If True, print losses at every step')

    parser.add_argument('--training_data_directory',  default=None, type=none_or_str, help='Directory where training data exists')
    parser.add_argument('--test_data_directory',  default=None, type=none_or_str, help='Directory where test data exists')
    parser.add_argument('--many_to_one_dataset', action="store_true", help = 'If True, use many-to-one dataset structure (e.g. Meta_camera/2025-08-21/0/*.png -> GT_camera/2025-08-21/0.png)')
    parser.add_argument('--result_path', default = './example/asset/ckpt/metasurface', type=str, help='dir to save models and checkpoints')
    parser.add_argument('--network_dir',  default=None, type=none_or_str, help='Directory where network processed imgs will be saved')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--crop_patch_size', default=512, type=int, help='crop patch size used for random crop in data augmentation')
    parser.add_argument('--full_res', default=1164, type=int, help='crop patch size used for random crop in data augmentation')

    # loss usage and loss weight
    parser.add_argument('--use_perc_loss', action="store_true", help = 'use lpips perceptual loss')
    parser.add_argument('--use_da_loss', action="store_true", help = 'use domain adaptation loss')
    parser.add_argument('--use_ssim_loss', action="store_true", help = 'use ssim loss')
    parser.add_argument('--use_fdl_loss', action="store_true", help = 'use fdl loss')
    parser.add_argument('--use_cobi_loss', action="store_true", help = 'use cobi loss')
    parser.add_argument('--da_loss_weight', default = 1.0, type = float, help = 'weight for domain adaptation loss')
    parser.add_argument('--l1_loss_weight', default = 1, type = float, help = 'weight for L1 loss')
    parser.add_argument('--ssim_loss_weight', default = 1.0, type = float, help = 'weight for ssim loss')
    parser.add_argument('--perceptual_loss_weight', default = 1, type = float, help = 'weight for perceptual loss')
    parser.add_argument('--fdl_loss_weight', default = 1, type = float, help = 'weight for fdl loss (Ni et al., "Misalignment-Robust Frequency Distribution Loss for Image Transformation", CVPR(2024))')
    parser.add_argument('--fdl_loss_phase_weight', default = 1, type = float, help = 'phase weight for fdl loss. default 1.0, sometimes use 0.01. (Ni et al., "Misalignment-Robust Frequency Distribution Loss for Image Transformation", CVPR(2024))')
    parser.add_argument('--cobi_loss_weight', default = 1, type = float, help = 'weight for cobi (contextual bilateral) loss.')
    parser.add_argument('--cobi_loss_weight_sp', default = 0.1, type = float, help = 'weight_sp for cobi (contextual bilateral) loss.')
    parser.add_argument('--cobi_loss_weight_band_width', default = 0.1, type = float, help = 'band_width for cobi (contextual bilateral) loss.')

    # Network training parameter
    parser.add_argument('--n_epochs', default = 100000, type = int, help = 'max num of training epoch')
    parser.add_argument('--network_lr', default=1e-4, type=float, help='reconstruction network learning rate')
    parser.add_argument('--log_freq', default=5, type=int, help = 'frequency (num_steps) of logging')
    parser.add_argument('--save_freq', default=20, type=int, help = 'frequency (num_steps) of saving checkpoint and visual performance')
    parser.add_argument('--use_warm_up_stage', action="store_true", help = 'use warmup stage for training. Usually used when training from scratch, not finetuning.')
    parser.add_argument('--T_max', default = 1000, type = int, help = 'cosine annealing scheudler T_max')
    parser.add_argument('--warmup_epochs', default = 10, type = int, help = 'warmup epochs')

    # Related to reconstruction network usage (LocalNet)
    parser.add_argument("--inverse",   action="store_true",  help="Inverse the tone curve")
    parser.add_argument("--no-grad",   action="store_false", dest="use_grad",     help="Disable gradients")
    parser.add_argument("--no-hist",   action="store_false", dest="use_hist",     help="Disable soft histograms")
    parser.add_argument("--no-satmask", action="store_false", dest="use_satmask", help="Disable saturation mask")

    # DDP: Handle LOCAL_RANK environment variable (automatically set when using torchrun)
    # If not set, default to 0 (for single GPU debugging)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args = parser.parse_args()
    
    # DDP initialization
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    # Device setup
    torch.cuda.set_device(local_rank)
    args.local_rank = local_rank
    args.device = torch.device(f"cuda:{local_rank}")

    # --- Race Condition & Logic Fix ---
    continue_training_tensor = torch.tensor([0], dtype=torch.uint8, device=args.device)

    # 1. Main process checks file system and decides
    if is_main_process():
        if os.path.exists(args.result_path):
            print(f'The path {args.result_path} already exists!!!!!!!!!!!')
            print(f'Continue training on the path {args.result_path}!!!!!!!!!!!')
            continue_training_tensor[0] = 1
        else:
            continue_training_tensor[0] = 0
            
    # 2. Broadcast the decision (Barrier function)
    dist.broadcast(continue_training_tensor, src=0)
    
    # 3. Apply decision
    continue_training = (continue_training_tensor.item() == 1)

    # 4. Handle Directory Creation (Safe on Main Process)
    if is_main_process():
        os.makedirs(args.result_path, exist_ok=True)
        
        args_dict = vars(args).copy()
        if 'device' in args_dict:
            args_dict['device'] = str(args_dict['device'])
        with open(os.path.join(args.result_path,'args.json'), "w") as f:
            json.dump(args_dict, f, indent=4, sort_keys=False)

    # 5. (Optional) Wait for all processes until file system is ready
    # Rank 0 creates folder and writes args.json while other processes wait
    dist.barrier() 

    train_network(args, continue_training)

if __name__ == '__main__':
    main()
