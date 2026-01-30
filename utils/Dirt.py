import numpy as np
import torch
import torchvision.transforms as transforms
from utils.PerlinBlob import *

def circle_grad(img_res_y, img_res_x):
    center_x, center_y = img_res_x // 2, img_res_y // 2
    circle_grad = np.zeros([img_res_y, img_res_x])

    for y in range(img_res_y):
        for x in range(img_res_x):
            distx = abs(x - center_x)
            disty = abs(y - center_y)
            dist = np.sqrt(distx*distx + disty*disty)
            circle_grad[y][x] = dist
    max_grad = np.max(circle_grad)
    circle_grad = circle_grad / max_grad
    circle_grad -= 0.5
    circle_grad *= 2.0
    circle_grad = -circle_grad

    circle_grad -= np.min(circle_grad)
    max_grad = np.max(circle_grad)
    circle_grad = circle_grad / max_grad
    return circle_grad

def compute_dirt(image_far, depth, args, predefined=False):
    param = args.param
    
    if not hasattr(args, 'broadband') or not args.broadband:
        n_channels = len(param.wvls)
    elif args.broadband:
        n_channels = len(param.broadband_wvls)

    if predefined:
        import os
        import torch.nn.functional as F
        if n_channels==3:
            mask = torch.tensor(np.load(os.path.join(args.eval_path, 'mask_'+str(864)+'_0022.npy')), device=args.device).detach().to(torch.float32)
            image_near = torch.tensor(np.load(os.path.join(args.eval_path, 'img_near_'+str(864)+'_0022.npy')), device=args.device).detach().to(torch.float32)
            mask = F.interpolate(mask, (param.img_res, param.img_res))
            image_near = F.interpolate(image_near, (param.img_res, param.img_res))
            image_near = image_near * mask + image_far.to(args.device) * (1-mask)
            return image_near.to(args.device), mask.to(args.device)
        else:
            mask = torch.tensor(np.load(os.path.join(args.eval_path, 'mask_'+str(864)+'_0022.npy')), device=args.device).detach().to(torch.float32)
            mask = F.interpolate(mask, (param.img_res, param.img_res))
            if mask.shape[-3] > 1: # make mask 1-channel
                mask = mask[:,0:1,...]
            color_spectral = np.linspace(0, 1, n_channels).astype(np.float32)
            color_spectral = color_spectral * np.ones([param.img_res, param.img_res, n_channels])
            color_spectral = color_spectral.transpose(2, 0, 1)[np.newaxis, ...]
            image_near = torch.tensor(torch.from_numpy(color_spectral).to(args.device) * mask)
            image_near = image_near.to(args.device) * mask.to(args.device) + image_far * (1-mask.to(args.device))
            return image_near.to(args.device), mask.to(args.device)
    
    color_spectral = np.random.rand(n_channels)
    color_spectral = (color_spectral * np.random.uniform(0, 1.5) + 
                np.random.uniform(-0.02, 0.02, n_channels) )* np.ones([image_far.shape[-2], image_far.shape[-1], n_channels])
    perlin_noise = generate_fractal_noise_2d([image_far.shape[-2], image_far.shape[-1]], [param.perlin_res,param.perlin_res], tileable=(True,True), interpolant=interpolant)
    depth_adj = depth / param.depth_near_max
    T = transforms.Compose([transforms.ToTensor(),
                            transforms.RandomCrop(int(param.img_res * depth_adj)), 
                            transforms.Resize([image_far.shape[-2], image_far.shape[-1]])
                            ])
    perlin_noise = T(perlin_noise).squeeze().numpy()  
    alpha_map = perlin_noise * (perlin_noise > param.perlin_cutoff) * circle_grad(image_far.shape[-2], image_far.shape[-1])
    alpha_map /= np.max(alpha_map)
    image_near = torch.tensor(color_spectral* alpha_map[...,None]).permute(2,0,1)[None,...]
    mask = torch.tile(torch.tensor(1.0*(alpha_map[...,None] > 0.3)).permute(2,0,1)[None,...],(1,1,1,1)) 
    image_near = image_near.to(args.device) * mask.to(args.device) + image_far * (1-mask.to(args.device))
    return image_near.to(args.device), mask.to(args.device)