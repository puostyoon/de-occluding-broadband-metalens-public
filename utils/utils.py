import os
import shutil
import json
from glob import glob
import skimage
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import pado
from pado.light import * 
from pado.optical_element import *
from pado.propagator import *
from utils.Dirt import compute_dirt


@torch.no_grad()
def compute_psnr(GT: torch.Tensor, image: torch.Tensor, data_range: float | None = None, reduction: str = "mean"):
    """
    PSNR for batched tensors (B, C, H, W).
    Returns: psnr_per_sample (B,), psnr_mean (scalar)
    """
    assert GT.shape == image.shape and GT.ndim == 4, "GT and image must be (B,C,H,W) with same shape."

    if data_range is None:
        data_range = _infer_data_range(GT)

    # Torch-native PSNR (no CPU sync per sample)
    mse = F.mse_loss(image, GT, reduction="none")  # (B,C,H,W)
    mse = mse.flatten(1).mean(dim=1)               # (B,)
    # Avoid log of zero
    eps = 1e-12
    psnr_per_sample = 10.0 * torch.log10((data_range ** 2) / (mse + eps))  # (B,)
    psnr_mean = psnr_per_sample.mean()

    return psnr_per_sample, psnr_mean

@torch.no_grad()
def compute_ssim(GT: torch.Tensor, image: torch.Tensor, data_range: float | None = None, reduction: str = "mean"):
    """
    SSIM for batched tensors (B, C, H, W) using skimage.
    Returns: ssim_per_sample (B,), ssim_mean (scalar)
    """
    assert GT.shape == image.shape and GT.ndim == 4, "GT and image must be (B,C,H,W) with same shape."

    B, C, H, W = GT.shape
    if data_range is None:
        data_range = _infer_data_range(GT)

    ssim_list = []
    # Convert each sample to HWC numpy and compute SSIM
    # Use channel_axis=-1 when C>1, else grayscale (2D) for C==1.
    for b in range(B):
        gt_np = GT[b].permute(1, 2, 0).cpu().numpy()   # (H,W,C)
        im_np = image[b].permute(1, 2, 0).cpu().numpy()
        if C == 1:
            # Drop channel dim for skimage grayscale path
            gt_np = gt_np[..., 0]
            im_np = im_np[..., 0]
            ssim_val = skimage.metrics.structural_similarity(gt_np, im_np, data_range=data_range)
        else:
            ssim_val = skimage.metrics.structural_similarity(gt_np, im_np, channel_axis=-1, data_range=data_range)
        ssim_list.append(ssim_val)

    ssim_per_sample = torch.tensor(ssim_list, dtype=torch.float32)
    ssim_mean = ssim_per_sample.mean()

    return ssim_per_sample, ssim_mean

# Image and PSF Manipulation

def sample_psf(psf, sample_ratio):
    if sample_ratio == 1:
        return psf
    else:
        return torch.nn.AvgPool2d(sample_ratio, stride=sample_ratio)(psf)
    

def compute_psf_arbitrary_prop(wvl, depth, doe, args, propagator='Fraunhofer', use_lens=True, offset=(0, 0), variable_offset_indices=None, theta=(0, 0), pad=True, full_psf=False, normalize=True):
    '''simulate depth based psf.
    Args:
        depth: propagation distance
        propagator: propagator defined in pado library. Fresenl, Fraunhofer, Rayleigh-Sommerfeld (RS), ASM, or BL_ASM
        use_lens: If True, DOE is placed in front of the lens.
        offset: (offset_y, offset_x) tuple. metric is meter. coordinate system follows matrix indexing ((0,0) is the top left corner.)
        theta: incident angle (theta_y, theta_x) of the point light source
        pad: If True, pad or unpad the psf size to be same as image size
        full_psf: If True, build full PSF which is impossible during training

    Note that Fresenl propagation with propagation distance focal length light after refractive lens is
    equal to the Fraunhofer propagation with propagation distance focal length.
    '''
    param = args.param
    prop = Propagator(propagator)
    dim = (1,1,param.R, param.C)

    # compute compute (dy, dx) for set_spherical_light function
    dy = np.tan(np.deg2rad(theta[0])) * (depth.numpy())
    dx = np.tan(np.deg2rad(theta[1])) * (depth.numpy())
    
    # set incident point light source
    light = Light(dim, param.DOE_pitch, wvl, device=args.device)
    light.set_spherical_light(depth.numpy(), dx=dx, dy=dy)

    if args.constant_wvl_phase:
        doe.wvl = wvl
    else:
        doe.change_wvl(wvl)
    light = doe.forward(light)

    if use_lens:
        lens = pado.optical_element.RefractiveLens(dim, param.DOE_pitch, param.focal_length, wvl, args.device)
        light = lens.forward(light.clone())

    aperture = Aperture(dim, param.DOE_pitch, param.aperture_diamter, param.aperture_shape, wvl, args.device)
    light = aperture.forward(light.clone())

    if full_psf:
        scale = (param.full_psf_scale[0], param.full_psf_scale[1])
    else:
        scale = (param.target_plane_scale[0], param.target_plane_scale[1])

    # Compute PSF where peak resides (similar to shifting PSF so that max value to be at center)
    if args.spatially_varying_PSF:
        ft = args.SVPSF_fitting_table
        shift_y, shift_x = ft[round(theta[0])+len(ft[-2])//2, round(theta[1])+len(ft[-1])//2]
        offset_y = -shift_y*param.camera_pitch
        offset_x = -shift_x*param.camera_pitch
        offset = (offset_y+offset[0], offset_x+offset[1])
    # debugging
    # light_prop = prop.forward(light, param.sensor_dist, offset=offset, variable_offset_indices=variable_offset_indices, linear=False, scale=scale)
    light_prop = prop.forward(light, param.sensor_dist, offset=offset, variable_offset_indices=variable_offset_indices, linear=False, scale=scale)
    # debugging
    psf = light_prop.get_intensity()

    # resize 
    if propagator!='RS':
        if args.resizing_method == 'original':
            psf = F.interpolate(psf, scale_factor=light_prop.pitch / param.DOE_pitch)
            psf = sample_psf(psf, param.DOE_sample_ratio)
        else:
            psf = F.interpolate(psf, scale_factor=light_prop.pitch/(param.camera_pitch* param.image_sample_ratio), 
                                mode=args.resizing_method)
    psf_size = psf.shape

    # Make psf dimensions of RGB to be same. 
    # In Fraunhofer's propagation, output field size depends on wavelength.
    if (pad is True) and (propagator=='Fraunhofer'):
        # compute dimensions of the Fraunhofer propagation result
        bw_r = light.get_bandwidth()[0]
        bw_c = light.get_bandwidth()[1]
        if bw_r != bw_c:
            raise NotImplementedError("Need to implement for the case when image shape is not square.")
        pitch_after_propagation = (np.array(param.broadband_wvls)) * np.array(param.sensor_dist / bw_r)
        scale_factors=pitch_after_propagation/(param.camera_pitch* param.image_sample_ratio)
        scale_factors_max = max(scale_factors)
        max_R = int(scale_factors_max * param.R) # round down
        wl, wr = compute_pad_size(psf_size[-1], max_R)
        hl, hr = compute_pad_size(psf_size[-2], max_R)
        psf = F.pad(psf, (wl, wr, hl, hr), "constant", 0)
        psf_size = psf.shape

    # debugging
    if propagator=='Fraunhofer':
        print('psf size after padding: ', psf.shape)
    # debugging

    cutoff = np.tan(np.arcsin(wvl/(2*param.DOE_pitch)))*param.focal_length / param.equiv_camera_pitch

    # debugging
    if propagator=='Fraunhofer':
        print('cutoff value: ', cutoff)
        DOE_mask = edge_mask(max(psf_size)//2, cutoff, args.device)
        psf = psf * DOE_mask 
    # debugging 
     
    if normalize:
        psf = psf / torch.sum(psf)

    return psf

def compute_psf_intensity_sum(wvl, depth, doe, args, use_lens=True, offset=(0, 0), target_plane_sample_trajectory=None, theta=(0, 0)):
    '''compute sum of intensities of the PSF. To remove light from some parts of the PSF plane.
    Args:
        depth: propagation distance
        use_lens: If True, DOE is placed in front of the lens.
        offset: (offset_y, offset_x) tuple. metric is meter. coordinate system follows matrix indexing ((0,0) is the top left corner.)
        theta: incident angle (theta_y, theta_x) of the point light source
    '''

    param = args.param
    prop = Propagator('SBL_ASM_intensity_sum')
    dim = (1,1,param.R, param.C)

    # compute compute (dy, dx) for set_spherical_light function
    dy = np.tan(np.deg2rad(theta[0])) * (param.focal_length + depth.numpy())
    dx = np.tan(np.deg2rad(theta[1])) * (param.focal_length + depth.numpy())
    
    # set incident point light source
    light = Light(dim, param.DOE_pitch, wvl, device=args.device)
    light.set_spherical_light(depth.numpy(), dx=dx, dy=dy)

    if args.constant_wvl_phase:
        doe.wvl = wvl
    else:
        doe.change_wvl(wvl)
    light = doe.forward(light)

    if use_lens:
        lens = pado.optical_element.RefractiveLens(dim, param.DOE_pitch, param.focal_length, wvl, args.device)
        light = lens.forward(light.clone())

    aperture = Aperture(dim, param.DOE_pitch, param.aperture_diamter, param.aperture_shape, wvl, args.device)
    light = aperture.forward(light.clone())

    # Compute PSF where peak resides (similar to shifting PSF so that max value to be at center)
    if args.spatially_varying_PSF:
        ft = args.SVPSF_fitting_table
        shift_y, shift_x = ft[round(theta[0])+len(ft[-2])//2, round(theta[1])+len(ft[-1])//2]
        offset_y = -shift_y*param.camera_pitch
        offset_x = -shift_x*param.camera_pitch
        offset = (offset_y, offset_x)

    intensity_sum = prop.forward(light, param.sensor_dist, offset=offset, target_plane_sample_trajectory=target_plane_sample_trajectory)
    return intensity_sum


def plot_depth_based_psf(doe, args, depths, wvls = 'RGB', merge_channel = False, pad=True, use_lens=None, propagator=None, eval=False, offset=(0, 0), theta=(0, 0), normalize=True, wvl_batching=False):
    param = args.param
    psfs = []
    if use_lens is None:
        use_lens = args.use_lens 
    if propagator is None:
        propagator = args.propagator
    if wvls == 'RGB':
        for i in range(len(param.wvls)):
            wvl = param.wvls[i]
            psf_depth = []
            for z in depths:
                psf = compute_psf_arbitrary_prop(wvl, 
                                                torch.tensor(z) if not isinstance(z, torch.Tensor) else z.clone().detach(), 
                                                doe, 
                                                args, 
                                                propagator=propagator,
                                                use_lens=use_lens,
                                                offset=offset,
                                                theta=theta,
                                                pad=pad,
                                                normalize=normalize)
                psf_depth.append(psf.detach() if eval else psf)
            psfs.append(torch.cat(psf_depth, -1))
        if merge_channel:
            psfs = torch.cat(psfs, 1)
        else:
            psfs = torch.cat(psfs, -2)
    elif isinstance(wvls, list) and len(wvls) == 3:
        if wvl_batching:
            psf_depth = []
            for z in depths:
                psf = compute_psf_arbitrary_prop(wvls, 
                                                torch.tensor(z) if not isinstance(z, torch.Tensor) else z.clone().detach(), 
                                                doe, 
                                                args, 
                                                propagator=propagator,
                                                use_lens=use_lens,
                                                offset=offset,
                                                theta=theta,
                                                pad=pad,
                                                normalize=normalize)
                psf_depth.append(psf.detach() if eval else psf)
            psfs.append(torch.cat(psf_depth, -1))
            if merge_channel:
                psfs = torch.cat(psfs, 1)
            else:
                psfs = torch.cat(psfs, -2) 
        else:
            for wvl in wvls:
                psf_depth = []
                for z in depths:
                    psf = compute_psf_arbitrary_prop(wvl, 
                                                    torch.tensor(z) if not isinstance(z, torch.Tensor) else z.clone().detach(), 
                                                    doe, 
                                                    args, 
                                                    propagator=propagator,
                                                    use_lens=use_lens,
                                                    offset=offset,
                                                    theta=theta,
                                                    pad=pad,
                                                    normalize=normalize)
                    psf_depth.append(psf.detach() if eval else psf)
                psfs.append(torch.cat(psf_depth, -1))
            if merge_channel:
                psfs = torch.cat(psfs, 1)
            else:
                psfs = torch.cat(psfs, -2)
    else:
        psf_depth = []
        for z in depths:
            # debugging
            # print('@@@@@@@wvls: ', wvls)
            # debugging
            psf = compute_psf_arbitrary_prop(wvls, 
                                            torch.tensor(z) if not isinstance(z, torch.Tensor) else z.clone().detach(), 
                                            doe, 
                                            args, 
                                            propagator=propagator,
                                            use_lens=use_lens,
                                            offset=offset,
                                            theta=theta,
                                            pad=pad,
                                            normalize=normalize)
            psf_depth.append(psf.detach() if eval else psf)
        psfs = torch.cat(psf_depth, -1)

    log_psfs = torch.log(psfs+1e-9)
    log_psfs = log_psfs - torch.min(log_psfs)
    log_psfs = log_psfs / torch.max(log_psfs)
    return psfs, log_psfs

# Parameter and Configuration Management

def convert_resolution(param, args):
    # dataset
    if args.obstruction == 'fence':
        param.dataset_dir = args.fence_dataset_dir
        param.data_resolution = [512,768]
    elif args.obstruction == 'raindrop' or 'dirt' or 'dirt_raindrop':
        param.training_dir = args.dirt_raindrop_dataset_train_dir
        param.val_dir = args.dirt_raindrop_dataset_val_dir
        param.data_resolution = [1024, 2048]
    else:
        assert False, "undefined obstruction"

    # convert resolution and pitch size
    param.equiv_image_size = param.img_res * param.image_sample_ratio # image resolution before downsampling in camera pixel pitch
    param.equiv_crop_size = int(param.equiv_image_size * param.camera_pitch / param.background_pitch)  # convert to background pixel pitch 
    return param

def save_settings(args, param, exist_ok=False):
    os.makedirs(args.result_path, exist_ok=exist_ok)
    args_dict = vars(args)
    with open(os.path.join(args.result_path,'args.json'), "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
    shutil.copy(args.param_file, args.result_path)
    shutil.copy('utils/utils.py', args.result_path)
    shutil.copy('./train_de_occluding_broadband_metalens.py', args.result_path)
    if args.pretrained_DOE is not None:
        shutil.copy(args.pretrained_DOE, os.path.join(args.result_path, 'init'))
    args.param = param

def last_save(ckpt_path, file_format):
    return sorted(glob(os.path.join(ckpt_path, file_format)))[-1]

# Image Processing and Fourier Transforms

def fft_convolve2d(image, kernel, linear=True):
    # Ensure the kernel is centered
    b, c, ih, iw = image.shape

    if kernel.dim() == 2:  # If the kernel has shape [W, H]
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)
    elif kernel.dim() == 3:  # If the kernel has shape [C, W, H]
        kernel = kernel.unsqueeze(0).expand(b, -1, -1, -1)

    # # crop only meaningful area. We don't need area exceeds 2X size of the image
    # if kernel.shape[-2] > ih*2 and kernel.shape[-1] > iw*2: 
    #     kernel = kernel.clone()
    #     kernel = kernel[..., kernel.shape[-2]//2-ih:kernel.shape[-2]//2+ih, kernel.shape[-1]//2-iw:kernel.shape[-1]//2+iw]
    # elif kernel.shape[-2] > ih*2: 
    #     kernel = kernel.clone()
    #     kernel = kernel[..., kernel.shape[-2]//2-ih:kernel.shape[-2]//2+ih, :]
    # elif kernel.shape[-1] > iw*2: 
    #     kernel = kernel.clone()
    #     kernel = kernel[..., :, kernel.shape[-1]//2-iw:kernel.shape[-1]//2+iw]
    _, _, kh, kw = kernel.shape

    # Compute the size of the convolution result. This includes zero padding to avoid circular convolution and ensure linear convolution
    if linear:
        conv_shape_image = (b, c, ih + kh - 1, iw + kw - 1)
        conv_shape_kernel = (b, kernel.shape[-3], ih + kh - 1, iw + kw - 1)
    else:
        raise NotImplementedError('circular convolution is not implemented in fft_convolved2d function!!!')
        
    # Pad the image
    padded_image = torch.zeros(conv_shape_image, dtype=image.dtype, device=image.device)
    padded_image[:, :, :ih, :iw] = image
    
    # Pad the kernel
    padded_kernel = torch.zeros(conv_shape_kernel, dtype=kernel.dtype, device=kernel.device)
    padded_kernel[:, :, :kh, :kw] = kernel
    # Center the kernel (PSF)
    padded_kernel = torch.roll(padded_kernel, shifts=(-kh//2, -kw//2), dims=(2, 3))
    
    image_fft = fft(padded_image, shift=False)
    kernel_fft = fft(padded_kernel, shift=False)
    convolved = ifft(image_fft*kernel_fft, shift=False)
    
    # Extract the original image size
    convolved = convolved[:, :, :ih, :iw].real
    return convolved

def fft_convolve2d_image_reflection(image, kernel):
    # Use this function to avoid zero-padded linear convolution making image darker.

    # Ensure the kernel is centered
    b, c, ih, iw = image.shape

    if kernel.dim() == 2:  # If the kernel has shape [W, H]
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)
    elif kernel.dim() == 3:  # If the kernel has shape [C, W, H]
        kernel = kernel.unsqueeze(0).expand(b, -1, -1, -1)

    _, _, kh, kw = kernel.shape

    # Compute the size of the convolution result. 
    # This includes zero padding to avoid circular convolution and ensure linear convolution.
    # Note that we only zero padd the PSF kernel, and will do the reflection padding for the input image.
    conv_shape_kernel = (b, kernel.shape[-3], ih + kh - 1, iw + kw - 1)
        
    # Pad the image
    reflection_operation = nn.ReplicationPad2d((0, kh-1, 0, kw-1))
    padded_image = reflection_operation(image)
    
    # Pad the kernel (PSF)
    padded_kernel = torch.zeros(conv_shape_kernel, dtype=kernel.dtype, device=kernel.device)
    padded_kernel[:, :, :kh, :kw] = kernel
    # Center the kernel (PSF)
    padded_kernel = torch.roll(padded_kernel, shifts=(-kh//2, -kw//2), dims=(2, 3))
    
    image_fft = fft(padded_image, shift=False)
    kernel_fft = fft(padded_kernel, shift=False)
    convolved = ifft(image_fft*kernel_fft, shift=False)
    
    # Extract the original image size
    convolved = convolved[:, :, :ih, :iw].real
    return convolved


def image_formation_essential_only(args, image_far, complex_field, wvls, wvl_batch_size, compute_obstruction, z_near = None):
    param = args.param
    if z_near is None:
        z_near = randuni(param.depth_near_min, param.depth_near_max, 1)[0] # randomly sample the near-point depth from a range

    z_far = randuni(param.depth_far_min, param.depth_far_max, 1)[0] # randomly sample the far-point depth from a range
    image_near, mask = compute_obstruction(image_far.get_intensity() if isinstance(image_far, pado.light.Light) 
                                           else image_far, z_near, args)
    
    image_near = image_near*0.01 # Make almost black
    
    if mask.shape[3] > 1:
        mask = mask[:,0:1,...]
    
    image_far = image_far.to(torch.float32)
    image_near = image_near.to(torch.float32)
    mask = mask.to(torch.float32)

    def _compute_psf(wvls, light, args, propagator, offset=(0, 0), normalize=True):
        param = args.param
        prop = Propagator(propagator)
        dim = light.field.shape
        aperture = Aperture(dim, param.DOE_pitch, param.aperture_diamter, param.aperture_shape, wvls, args.device)
        light = aperture.forward(light.clone())

        light_prop = prop.forward(light, param.sensor_dist, offset=offset, linear=False)
        psf = light_prop.get_intensity()

        # resize 
        psf = F.interpolate(psf, scale_factor=light_prop.pitch/(param.camera_pitch* param.image_sample_ratio), 
                            mode=args.resizing_method)
        if normalize:
            psf = psf / (torch.sum(psf, dim=(-2,-1), keepdim=True) + 1e-8) # normalize the psf for each wavelength
        return psf
    light = pado.Light(complex_field.shape, pitch=param.DOE_pitch, wvl=wvls, device=args.device)
    light.set_spherical_light(z_far)
    light.set_field(light.get_field()*complex_field)
    psf_far = _compute_psf(wvls, light, args, 'SBL_ASM', offset=(0, 0), normalize=False)

    light = pado.Light(complex_field.shape, pitch=param.DOE_pitch, wvl=wvls, device=args.device)
    light.set_spherical_light(z_near)
    light.set_field(light.get_field()*complex_field)
    psf_near = _compute_psf(wvls, light, args, 'SBL_ASM', offset=(0, 0), normalize=False)

    # compute intensity sum of the PSF for the area that sensor respond to
    if (psf_far.shape[-2] > param.camera_resolution[-2] and psf_far.shape[-1] > param.camera_resolution[-1]):
        psf_far_sensor_sum = psf_far[...,psf_far.shape[-2]//2-param.camera_resolution[-2]//2:psf_far.shape[-2]//2+param.camera_resolution[-2]//2,
                            psf_far.shape[-1]//2-param.camera_resolution[-1]//2:psf_far.shape[-1]//2+param.camera_resolution[-1]//2].sum()
    else:
        psf_far_sensor_sum =  psf_far.sum()

    #### Near PSF Normalization ####
    psf_near = psf_near/psf_near.sum(dim=(-2,-1), keepdim=True)

    #### Before far PSF normalization ####
    incident_light_intensity_sum = np.pi*((param.R/2)**2)*((param.DOE_pitch/param.camera_pitch)**2) # Conserve total light intensity
    convolved_far_norm_w_intensity_sum = fft_convolve2d_image_reflection(image_far.repeat(1, wvl_batch_size, 1, 1), psf_far/incident_light_intensity_sum)
    convolved_near = fft_convolve2d(image_near.repeat(1, wvl_batch_size, 1, 1), psf_near)
    convolved_mask = fft_convolve2d(mask, psf_near)
    convolved_mask = torch.clamp(1.5 * convolved_mask, 0, 1)
    img_conv_norm_w_intensity_sum = convolved_far_norm_w_intensity_sum * (1 - convolved_mask) + convolved_near * convolved_mask

    #### Far PSF Normalization ####
    psf_far = psf_far/psf_far.sum(dim=(-2,-1), keepdim=True)

    #### After PSF normalization ####
    convolved_far = fft_convolve2d_image_reflection(image_far.repeat(1, wvl_batch_size, 1, 1), psf_far)# Conserve total light intensity
    img_conv = convolved_far * (1 - convolved_mask) + convolved_near * convolved_mask

    if wvl_batch_size > 1:
        convolved_far = torch.stack(torch.split(convolved_far, 3, dim=-3)).mean(dim=0)
        convolved_near = torch.stack(torch.split(convolved_near, 3, dim=-3)).mean(dim=0)
        img_conv = torch.stack(torch.split(img_conv, 3, dim=-3)).mean(dim=0)
        img_conv_norm_w_intensity_sum = torch.stack(torch.split(img_conv_norm_w_intensity_sum, 3, dim=-3)).mean(dim=0)
        # psf_near = torch.stack(torch.split(psf_near, 3, dim=-3)).mean(dim=0)
        # psf_far = torch.stack(torch.split(psf_far, 3, dim=-3)).mean(dim=0)

    return image_far, image_near, mask, convolved_far, convolved_near, convolved_mask, img_conv, psf_near, psf_far, psf_far_sensor_sum, img_conv_norm_w_intensity_sum

def image_formation_essential_only_lens(args, image_far, complex_field, wvls, wvl_batch_size, compute_obstruction, z_near = None):
    param = args.param
    z_far = randuni(param.depth_far_min, param.depth_far_max, 1)[0] # randomly sample the far-point depth from a range
    image_far = image_far.to(torch.float32)

    def _compute_psf(wvls, light, args, propagator, offset=(0, 0), normalize=True):
        param = args.param
        prop = Propagator(propagator)
        dim = light.field.shape
        aperture = Aperture(dim, param.DOE_pitch, param.aperture_diamter, param.aperture_shape, wvls, args.device)
        light = aperture.forward(light.clone())

        light_prop = prop.forward(light, param.sensor_dist, offset=offset, linear=False)
        psf = light_prop.get_intensity()

        # resize 
        psf = F.interpolate(psf, scale_factor=light_prop.pitch/(param.camera_pitch* param.image_sample_ratio), 
                            mode=args.resizing_method)
        if normalize:
            psf = psf / (torch.sum(psf, dim=(-2,-1), keepdim=True) + 1e-8) # normalize the psf for each wavelength
        return psf
    light = pado.Light(complex_field.shape, pitch=param.DOE_pitch, wvl=wvls, device=args.device)
    light.set_spherical_light(z_far)
    light.set_field(light.get_field()*complex_field)
    psf_far = _compute_psf(wvls, light, args, 'SBL_ASM', offset=(0, 0), normalize=False)

    # compute intensity sum of the PSF for the area that sensor respond to
    if (psf_far.shape[-2] > param.camera_resolution[-2] and psf_far.shape[-1] > param.camera_resolution[-1]):
        psf_far_sensor_sum = psf_far[...,psf_far.shape[-2]//2-param.camera_resolution[-2]//2:psf_far.shape[-2]//2+param.camera_resolution[-2]//2,
                            psf_far.shape[-1]//2-param.camera_resolution[-1]//2:psf_far.shape[-1]//2+param.camera_resolution[-1]//2].sum()
    else:
        psf_far_sensor_sum =  psf_far.sum()

    #### Before far PSF normalization ####
    incident_light_intensity_sum = np.pi*((param.R/2)**2)*((param.DOE_pitch/param.camera_pitch)**2) # Conserve total light intensity
    convolved_far_norm_w_intensity_sum = fft_convolve2d_image_reflection(image_far.repeat(1, wvl_batch_size, 1, 1), psf_far/incident_light_intensity_sum)

    #### Far PSF Normalization ####
    psf_far = psf_far/psf_far.sum(dim=(-2,-1), keepdim=True)

    #### After PSF normalization ####
    convolved_far = fft_convolve2d_image_reflection(image_far.repeat(1, wvl_batch_size, 1, 1), psf_far)# Conserve total light intensity

    if wvl_batch_size > 1:
        convolved_far = torch.stack(torch.split(convolved_far, 3, dim=-3)).mean(dim=0)
        convolved_far_norm_w_intensity_sum = torch.stack(torch.split(convolved_far_norm_w_intensity_sum, 3, dim=-3)).mean(dim=0)
        # psf_far = torch.stack(torch.split(psf_far, 3, dim=-3)).mean(dim=0)

    return image_far,convolved_far,psf_far, psf_far_sensor_sum,convolved_far_norm_w_intensity_sum

def evaluate_image_loss_essential_only(args, DOE_phase, testloader, writer, total_step):
    param = args.param 
    image_far, _ = next(iter(testloader))
    image_far = image_far.to(args.device)
    image_near, mask = compute_dirt(image_far.get_intensity() if isinstance(image_far, pado.light.Light) 
                                        else image_far, (param.depth_near_min+param.depth_near_max)/2, args, predefined=True)
    z_far = (param.depth_far_min+param.depth_far_max)/2
    z_near = (param.depth_near_min+param.depth_near_max)/2

    complex_field = torch.exp(1j*DOE_phase.repeat((1, len(param.wvls), 1, 1)))

    def _compute_psf(wvls, light, args, propagator, offset=(0, 0), normalize=True):
        param = args.param
        prop = Propagator(propagator)
        dim = light.field.shape
        aperture = Aperture(dim, param.DOE_pitch, param.aperture_diamter, param.aperture_shape, wvls, args.device)
        light = aperture.forward(light.clone())

        light_prop = prop.forward(light, param.sensor_dist, offset=offset, linear=False)
        psf = light_prop.get_intensity()

        # resize 
        psf = F.interpolate(psf, scale_factor=light_prop.pitch/(param.camera_pitch* param.image_sample_ratio), 
                            mode=args.resizing_method)
        if normalize:
            psf = psf / (torch.sum(psf, dim=(-2,-1), keepdim=True) + 1e-8) # normalize the psf for each wavelength
        return psf
    light = pado.Light(complex_field.shape, pitch=param.DOE_pitch, wvl=param.wvls, device=args.device)
    light.set_spherical_light(z_far)
    light.set_field(light.get_field()*complex_field)
    psf_far = _compute_psf(param.wvls, light, args, 'SBL_ASM', offset=(0, 0), normalize=True)

    light = pado.Light(complex_field.shape, pitch=param.DOE_pitch, wvl=param.wvls, device=args.device)
    light.set_spherical_light(z_near)
    light.set_field(light.get_field()*complex_field)
    psf_near = _compute_psf(param.wvls, light, args, 'SBL_ASM', offset=(0, 0), normalize=True)

    for step, batch_data in enumerate(testloader):
        image_far_color, _ = batch_data
        image_far_color = image_far_color.to(args.device)

        # Conserve total light intensity
        convolved_far = fft_convolve2d_image_reflection(image_far_color, psf_far)
        convolved_near = fft_convolve2d(image_near, psf_near)
        convolved_mask = fft_convolve2d(mask, psf_near)

        # To simulate realistic blending ratio between near and far PSF
        psf_center_sum = psf_far[:, :, psf_far.shape[-2]//2-30:psf_far.shape[-2]//2+30, 
                                psf_far.shape[-1]//2-30:psf_far.shape[-1]//2+30].sum()/3
        convolved_mask = torch.clamp(0.5*0.333/psf_center_sum*fft_convolve2d(mask, psf_near), 0, 1) 
        
        img_conv = convolved_far * (1 - convolved_mask) + convolved_near * convolved_mask
                    
        if step==0:
            step_loss = 0
            step_psnr = 0
            os.makedirs(os.path.join(args.result_path, 'logged_images'), exist_ok=True)
            os.makedirs(os.path.join(args.result_path, 'logged_psf_near'), exist_ok=True)
            os.makedirs(os.path.join(args.result_path, 'logged_psf_far'), exist_ok=True)
            os.makedirs(os.path.join(args.result_path, 'logged_phase_map'), exist_ok=True)

            plt.figure(); plt.imshow(np.clip(psf_near.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
            plt.savefig(os.path.join(args.result_path, 'logged_psf_near', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
            
            plt.figure(); plt.imshow(np.clip(psf_far.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
            plt.savefig(os.path.join(args.result_path, 'logged_psf_far', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
            plt.imsave(os.path.join(args.result_path, 'logged_psf_far', f'raw_step_{total_step}.png'), np.clip(psf_far.cpu().detach().numpy()[0, 0, :, :], 0, 1))
            plt.imsave(os.path.join(args.result_path, 'logged_psf_far', f'raw_center_step_{total_step}.png'), np.clip(psf_far.cpu().detach().numpy()[0, 0, psf_far.shape[-2]//2-50:psf_far.shape[-2]//2+50, psf_far.shape[-1]//2-50:psf_far.shape[-1]//2+50], 0, 1))

            plt.figure(); plt.imshow(np.clip(psf_near.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
            plt.savefig(os.path.join(args.result_path, 'logged_psf_near', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
            plt.figure(); plt.imshow(np.clip(psf_far.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
            plt.savefig(os.path.join(args.result_path, 'logged_psf_far', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()

            plt.imsave(os.path.join(args.result_path, 'logged_images', f'raw_step_{total_step}.png'), np.clip(img_conv.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0), 0, 1))

            DOE_train = DOE((1,1,param.R, param.R), param.DOE_pitch, param.material, wvl=param.DOE_wvl, device=args.device)
            DOE_train.set_phase_change(DOE_phase)  
            DOE_train.visualize()
            plt.savefig(os.path.join(args.result_path, 'logged_phase_map', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()


        step_loss = step_loss + args.l1_criterion(img_conv, image_far).item()
        step_psnr = step_psnr +  skimage.metrics.peak_signal_noise_ratio(
                                image_far[0].permute(1, 2, 0).cpu().detach().numpy(), img_conv[0].permute(1, 2, 0).cpu().detach().numpy())

    l1_loss = step_loss/len(testloader)
    psnr = step_psnr/len(testloader)

    if writer is not None:       
        writer.add_scalar('Loss/train', l1_loss, total_step)  # Log loss for TensorBoard
        writer.add_scalar('PSNR/train', psnr, total_step)
        if args.use_da_loss:
            from models.DA_loss_functions import DA_loss
            da_loss = DA_loss(img_conv, image_far, args).detach().item()
            writer.add_scalar('DA_loss/train', da_loss, total_step)
        if args.use_perc_loss:
            perc_loss = torch.mean(args.perceptual_loss_weight * 
                    args.perceptual_criterion(2 * img_conv.to(torch.float32) - 1, 2 * image_far.to(torch.float32) - 1))
            writer.add_scalar('perc_loss/train', perc_loss, total_step)
        writer.add_image('image near', image_near[0].cpu().detach(), total_step)
        writer.add_image('image_far', image_far[0].cpu().detach(), total_step)
        writer.add_image('convolved far only', convolved_far[0].cpu().detach(), total_step)
        writer.add_image('convolved image', img_conv[0].cpu().detach(), total_step)
        writer.add_image('psf near (1st channel)', psf_near.cpu().detach()[0, 0, :, :], total_step, dataformats='HW')
        writer.add_image('psf far (1st channel)', psf_far.cpu().detach()[0, 0, :, :], total_step, dataformats='HW')

    return l1_loss
    

# Utility and Miscellaneous Functions

def randuni(low, high, size):
    '''uniformly sample from [low, high)'''
    return (torch.rand(size)*(high - low) + low)

def compute_pad_size(current_size, target_size):
    if current_size == target_size:
        return (0, 0)
    assert current_size < target_size
    gap = target_size - current_size
    left = int(gap/2)
    right = gap - left
    return int(left), int(right)

def edge_mask(R,cutoff, device):
    [x, y] = np.mgrid[-int(R):int(R),-int(R):int(R)]
    dist = np.sqrt(x**2 +y**2).astype(np.int32)
    mask = torch.tensor(1.0*(dist < cutoff)).to(torch.float32)[None, None, ...]
    return mask.to(device)


def add_poisson_noise(img: torch.Tensor, peak: float = 1.0) -> torch.Tensor:
    """
    Add Poisson noise to an image.

    Args:
        img (torch.Tensor): Input image tensor of shape (..., H, W) or (..., C, H, W),
                            with non-negative values.
        peak (float):       Scaling factor that represents the maximum expected photon count.
                            If your image values are in [0, 1], setting peak > 1 simulates
                            brighter images (higher SNR). Default: 1.0.

    Returns:
        torch.Tensor: Noisy image tensor, same shape and dtype as input.
    """
    # Ensure non-negative
    img = torch.clamp(img, min=0)

    # Scale image to photon counts
    vals = img * peak

    # Sample Poisson noise
    noisy_vals = torch.poisson(vals)

    # Scale back to original range
    noisy_img = noisy_vals.to(img.dtype) / peak

    return noisy_img

def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

def metric2pixel(metric, depth, args):
    return int(metric * args.param.focal_length / (depth * args.param.equiv_camera_pitch))

# Data Structure Management

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

# ============================================================================
# Shared Training Utilities for Modularization
# ============================================================================

def none_or_str(value):
    """Helper for argparse to convert 'None' string to None."""
    if value == 'None':
        return None
    return value

def create_dataloaders(args, param):
    """Create train and test dataloaders based on obstruction type.
    
    Returns:
        tuple: (trainloader, testloader, compute_obstruction_fn)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(param.data_resolution, pad_if_needed=True),
        transforms.RandomCrop([param.equiv_crop_size, param.equiv_crop_size], pad_if_needed=True),
        transforms.Resize([param.img_res, param.img_res]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([param.img_res, param.img_res]),
        transforms.ToTensor(),
    ])
    
    
    if args.obstruction == 'dirt':
        trainset = torchvision.datasets.ImageFolder(param.training_dir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(param.val_dir, transform=transform_test)
        compute_obstruction = compute_dirt
    else:
        raise ValueError(f"Unknown obstruction type: {args.obstruction}")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    return trainloader, testloader, compute_obstruction

def build_models_and_loss_functions(args, param):
    """Build DOE phase, optimizers, and loss functions.
    
    Returns:
        dict: Contains DOE_phase, optics_optimizer, optics_scheduler, l1_criterion, perceptual_criterion
    """
    # Loss functions
    l1_criterion = nn.L1Loss().to(args.device)
    perceptual_criterion = lpips.LPIPS(net='vgg').to(args.device) if args.use_perc_loss else None
    
    # DOE phase initialization
    if args.pretrained_DOE is not None:
        DOE_phase = torch.load(args.pretrained_DOE, map_location=args.device).detach()
    else:
        if args.phase_init == 'random':
            DOE_phase = torch.rand(param.DOE_phase_init.shape, device=args.device) * 10
        elif args.phase_init == 'fresnel':
            DOE_phase = RefractiveLens(param.DOE_phase_init.shape, param.DOE_pitch, 
                                      param.focal_length, param.DOE_wvl, args.device).get_phase_change()
        else:
            # zero initialization
            DOE_phase = torch.zeros(param.DOE_phase_init.shape, device=args.device)
    
    DOE_phase = torch.tensor(DOE_phase.to(args.device), requires_grad=True)
    
    # Optics optimizer and scheduler
    optics_optimizer = optim.AdamW([DOE_phase], lr=args.optics_lr)
    optics_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optics_optimizer, T_max=args.T_max, eta_min=args.optics_lr/100) if hasattr(args, 'T_max') else None
    
    return {
        'DOE_phase': DOE_phase,
        'optics_optimizer': optics_optimizer,
        'optics_scheduler': optics_scheduler,
        'l1_criterion': l1_criterion,
        'perceptual_criterion': perceptual_criterion,
    }

def setup_wavelength_parameters(param, split_spectrum=True):
    """Setup wavelength parameters and sampling weights.
    
    Returns:
        dict: Contains wvls, wvl_sample_weight
    """
    wvls = param.full_broadband_wvls
    wvl_sample_weight = {'R': [], 'G': [], 'B': []}
    if split_spectrum: # with bandpass filter
        for wvl in param.full_broadband_wvls:
            for color_ in ['R', 'G', 'B']:
                wvl_sample_weight[color_].append(param.full_cam_response_func[wvl][color_] * param.full_DOE_eff[wvl] * param.full_bandpass_filter_transmission[wvl])
    else: # without bandpass filter
        for wvl in param.full_broadband_wvls:
            for color_ in ['R', 'G', 'B']:
                wvl_sample_weight[color_].append(param.full_cam_response_func[wvl][color_] * param.full_DOE_eff[wvl])
    return {
        'wvls': wvls,
        'wvl_sample_weight': wvl_sample_weight
    }

def save_checkpoint(args, DOE_phase, total_step, best=False):
    """Save DOE phase checkpoint.
    
    Args:
        args: Training arguments
        DOE_phase: DOE phase tensor to save
        total_step: Current training step
        best: If True, save as best model
    """
    if best:
        torch.save(DOE_phase, os.path.join(args.result_path, 'DOE_phase_minimum_eval_loss.pt'))
    else:
        torch.save(DOE_phase, os.path.join(args.result_path, f'DOE_phase_{total_step:05d}.pt'))
