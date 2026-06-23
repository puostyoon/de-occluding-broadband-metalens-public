import os
import argparse
import random
from tqdm.auto import tqdm

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from importlib.machinery import SourceFileLoader
from pytorch_msssim import ms_ssim

from utils.utils import *
from utils.Dirt import *
from models.DA_loss_functions import DA_loss

def backward_image_loss_3channel_at_once_grad_accum_weighted_wvl_sampling(args, wvl_sample_weight, batch_data, DOE_phase, wvls):
    param = args.param
    accum_step = 10
    wvl_batch_size = 1
    for _ in range(accum_step):
        wvls = list()
        for _ in range(wvl_batch_size):
            wvls.append(random.choices(population=param.full_broadband_wvls, weights=wvl_sample_weight['R'], k=1)[0])
            wvls.append(random.choices(population=param.full_broadband_wvls, weights=wvl_sample_weight['G'], k=1)[0])
            wvls.append(random.choices(population=param.full_broadband_wvls, weights=wvl_sample_weight['B'], k=1)[0])
        print(wvls)
        step_loss = 0
        complex_field = torch.exp(1j*DOE_phase.repeat((1, len(wvls), 1, 1)))
        image_far_color, _ = batch_data
        image_far_color = image_far_color.to(args.device)

        # Since channel_idx is None, we convolve single channel psf with 3-channel color image
        image_far, _, mask, _, _, _, img_conv, _, psf_far, psf_far_sensor_sum, _ = image_formation_essential_only(args, 
                                                                                                                    image_far_color, 
                                                                                                                    complex_field, 
                                                                                                                    wvls, 
                                                                                                                    wvl_batch_size,
                                                                                                                    args.compute_obstruction)
    

        incident_light_intensity_sum = np.pi*((param.R/2)**2)*((param.DOE_pitch/param.camera_pitch)**2)
        brightness_regularizer =  (incident_light_intensity_sum*psf_far.shape[-3]/psf_far_sensor_sum-1)

        B, C, H, W = psf_far.shape
        srpcw = 5 # sharpness regularizer psf crop width
        h0 = H // 2 - srpcw
        h1 = H // 2 + srpcw
        w0 = W // 2 - srpcw
        w1 = W // 2 + srpcw
        patch = psf_far[:, :, h0:h1, w0:w1]
        # -> Sum all dimensions in batch/spatial direction and keep only channel dimension => (C,)
        per_channel_sum = patch.sum(dim=(0, 2, 3))
        target_val = 1.0 / 3 # We want center sum to be at least 0.333
        # Apply clamp per channel and sum all channels -> final scalar
        sharpness_regularizer = torch.clamp(
            target_val - per_channel_sum,
            min=0.0,
            max=100.0
        ).sum() / wvl_batch_size

        img_conv_mask = img_conv * mask
        image_far_mask = image_far * mask
        l1_loss = args.l1_loss_weight * args.l1_criterion(img_conv, image_far)
        print('l1 loss: ', l1_loss, 'brightness regularizer: ', brightness_regularizer,'sharpness regularizer: ', sharpness_regularizer)
        masked_loss = args.masked_loss_weight * args.l1_loss_weight * args.l1_criterion(img_conv_mask, image_far_mask)
        loss = l1_loss + masked_loss + brightness_regularizer * args.brightness_regularizer_coeff + sharpness_regularizer * args.sharpness_regularizer_coeff

        if args.use_perc_loss:
            perc_loss = torch.mean(args.perceptual_loss_weight * 
                                args.perceptual_criterion(2 * img_conv.to(torch.float32) - 1, 2 * image_far.to(torch.float32) - 1))
            loss = loss + perc_loss * args.da_loss_weight

        if args.use_da_loss:
            da_loss = DA_loss(img_conv, image_far, args, feature='segmentation')
            loss = loss + da_loss * args.da_loss_weight

        if args.use_ssim_loss:
            ssim_loss = 1 - ms_ssim(img_conv, image_far, data_range=1.0)
            loss = loss + ssim_loss * args.ssim_loss_weight

        loss = loss * args.image_loss_weight
        loss = loss / accum_step
        
        try: 
            loss.backward() 
        except: 
            import pdb 
            pdb.set_trace()
        step_loss = step_loss + loss.item()

    return step_loss

def train(args):

    writer = SummaryWriter(log_dir=args.result_path + '/runs')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    param = args.param

    # Create dataloaders and get compute_obstruction function
    trainloader, testloader, compute_obstruction = create_dataloaders(args, param)
    args.compute_obstruction = compute_obstruction

    # Build models and loss functions
    models = build_models_and_loss_functions(args, param)
    DOE_phase = models['DOE_phase']
    optics_optimizer = models['optics_optimizer']
    optics_scheduler = models['optics_scheduler']
    args.l1_criterion = models['l1_criterion']
    args.perceptual_criterion = models['perceptual_criterion']

    total_step = 0
    eval_minimum_loss = np.inf

    # Setup wavelength parameters
    wvl_params = setup_wavelength_parameters(param, split_spectrum=args.split_spectrum)
    wvls = wvl_params['wvls']
    wvl_sample_weight = wvl_params['wvl_sample_weight']

    for epoch in tqdm(range(args.n_epochs), position=0, leave=False):
        for step, batch_data in enumerate(tqdm(trainloader)):
            image_loss = backward_image_loss_3channel_at_once_grad_accum_weighted_wvl_sampling(args, wvl_sample_weight, batch_data, DOE_phase, wvls)
            psf_loss = 0
            optics_optimizer.step()
            optics_optimizer.zero_grad()
            optics_scheduler.step()

            with torch.no_grad():
                if total_step%args.log_freq==0:
                    eval_loss = 0
                    eval_loss = evaluate_image_loss_essential_only(args, DOE_phase, testloader, writer, total_step)
                    # Save the best model
                    if eval_loss < eval_minimum_loss:
                        eval_minimum_loss = eval_loss
                        save_checkpoint(args, DOE_phase, total_step, best=True)

            
            total_step += 1
            os.system('clear')
            print(f'\n Epoch {epoch} step {step} Loss (image, psf): {image_loss, psf_loss}')
            print('\n DOE phase shape: ', DOE_phase.shape)

            # save model
            if ((param.save_steps is not None) and (total_step in param.save_steps)) or (total_step % args.save_freq == 0):
                save_checkpoint(args, DOE_phase, total_step, best=False)

            print("optics optimizer lr: ", optics_optimizer.param_groups[0]['lr'])

    writer.close()  


def main():
    parser = argparse.ArgumentParser(
        description='PSF-based de-occluding broadband metalens training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--device', default='cpu', type=str, help='torch_device')
    parser.add_argument('--debug', action="store_true", help='debug mode, train on validation data to speed up the process')
    parser.add_argument('--result_path', default = './example/asset/ckpt/metasurface', type=str, help='dir to save models and checkpoints')
    parser.add_argument('--eval_path', default="dataset/eval", type=str, help='path where an image for evaluation is saved')
    parser.add_argument('--param_file', default= './example/asset/config/param_MV_1600_metasurface.py', type=str, help='path to param file')
    parser.add_argument('--pretrained_DOE', default = None, type=none_or_str, help = 'Directory of pretrained DOE or None')

    # Related to dataset usage (used for image restoration loss)
    parser.add_argument('--fence_dataset_dir', help='Directory of dataset used for fence obstruction')
    parser.add_argument('--dirt_raindrop_dataset_train_dir', help='Directory of dataset used for raindrop, dirt, and raindrop_dirt obstructions')
    parser.add_argument('--dirt_raindrop_dataset_val_dir', help='Directory of dataset used for raindrop, dirt, and raindrop_dirt obstructions')

    parser.add_argument('--obstruction', default = 'dirt_raindrop', type = str, help = 'obsturction type')
    parser.add_argument('--propagator', default = 'Fraunhofer', type = str, help = 'propagator used to compute the psf')
    parser.add_argument('--use_lens', action="store_true", help = 'Additional lens usage. Look at compute_psf_arbitrary_prop. Note that Fresnel+Lens is Fraunhofer')
    parser.add_argument('--n_epochs', default = 10000, type = int, help = 'max num of training epoch')
    parser.add_argument('--log_freq', default=30, type=int, help = 'frequency (num_steps) of logging')
    parser.add_argument('--save_freq', default=400, type=int, help = 'frequency (num_steps) of saving checkpoint and visual performance')
    parser.add_argument('--optics_lr', default=0.1, type=float, help='optical element learning rate')
    parser.add_argument('--T_max', default=1000, type=float, help='Cosine annealing T_max')

    # Related to loss
    parser.add_argument('--use_perc_loss', action="store_true", help = 'use lpips perceptual loss')
    parser.add_argument('--use_da_loss', action="store_true", help = 'use domain adaptation loss')
    parser.add_argument('--use_ssim_loss', action="store_true", help = 'use ssim loss')
    parser.add_argument('--use_psf_near_guide_loss', action="store_true", help = 'use psf near guide loss. it guides the near psf shape to be ring')
    parser.add_argument('--da_loss_weight', default = 1.0, type = float, help = 'weight for domain adaptation loss')
    parser.add_argument('--l1_loss_weight', default = 1, type = float, help = 'weight for L1 loss')
    parser.add_argument('--ssim_loss_weight', default = 1.0, type = float, help = 'weight for ssim loss')
    parser.add_argument('--masked_loss_weight', default = 1, type = float, help = 'weight for masked loss (focus on obstructed scene)')
    parser.add_argument('--perceptual_loss_weight', default = 1, type = float, help = 'weight for perceptual loss')
    parser.add_argument('--image_loss_weight', default=1.0, type=float, help='weight for image reconstruction loss') 
    parser.add_argument('--psf_loss_weight', default=1.0, type=float, help='weight for psf reconstruction loss') 
    parser.add_argument('--brightness_regularizer_coeff', default=0.001, type=float, help='Brightness regularizer coefficient')
    parser.add_argument('--sharpness_regularizer_coeff', default=0.1, type=float, help='Sharpness regularizer coefficient')

    # Related to training method
    parser.add_argument('--phase_init', default='random', type=str, help='Phase map initialization method. random, or Fresnel')
    parser.add_argument('--resizing_method', default='area', type=str, help='PSF resizing method. original or area. look at compute_psf function')
    parser.add_argument('--train_broadband', action="store_true", help='If True, train optics for broadband wvls')
    parser.add_argument('--split_spectrum', action="store_true", help='If True, train optics for split spectrum imaging with bandpass filters')
    parser.add_argument('--constant_wvl_phase', action="store_true", help='If True, do not call DOE.change_wvl() ever. i.e., use the same phase map for all wvls (geometric phase)')

    args = parser.parse_args()

    param = SourceFileLoader("param", args.param_file).load_module()
    param = convert_resolution(param,args)

    if args.pretrained_DOE is not None:
        param.DOE_phase_init = torch.load(args.pretrained_DOE, map_location=args.device).detach()
    else:
        if args.phase_init=='random':
            param.DOE_phase_init = torch.rand(param.DOE_phase_init.shape, device=args.device) * 10
        elif args.phase_init=='fresnel':
            param.DOE_phase_init = RefractiveLens(param.DOE_phase_init.shape, param.DOE_pitch, param.focal_length, param.DOE_wvl, args.device).get_phase_change()
        else:
            # zero initialization
            param.DOE_phase_init = torch.zeros(param.DOE_phase_init.shape, device=args.device)
            print('ze initialized DOE phase')
    
    save_settings(args, param)

    train(args)

if __name__ == '__main__':
    main()
