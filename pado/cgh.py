import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from .math import fft, ifft, conv_fft, wrap_phase, calculate_psnr
from .math import nm, um, mm, cm, m
from .light import Light
from .propagator import Propagator
import math

class Iterative_CGH:
    def __init__(self, mode):
        """
        Algorithm for the iterative CGH implementations
        Args:
            mode: type of the algorithm, e.g. "GS", "SGD", "ADAM", "DPAC"
        """
        self.mode = mode

        
    def optimize(self, light, target=None, z=None, n_iter=100, prop_model='ASM'):
        """
        Optimize the phase of the light field to match the target intensity
        Args:
            light: instance of the Light class
            target: target image light
            z: propagation distance in meters
            n_iter: number of iterations
            lr: learning rate
        Returns:
            light: optimized phase pattern at the SLM plane
        """
        if z == 0:
            raise ValueError("Propagation distance should be greater than 0")
        
        if self.mode == "GS":
            return self.gerchberg_saxton(light, target, z, n_iter, prop_model)
        elif self.mode == "SGD":
            return self.stochastic_gradient_descent(light, target, z, n_iter, prop_model)
        elif self.mode == "ADAM":
            return self.adaptive_moment_estimation(light, target, z, n_iter, prop_model)
        else:
            raise NotImplementedError("%s algorithm is not implemented" % self.mode)
    
    def gerchberg_saxton(self, init_light, target_light, z, n_iter=100, prop_model='ASM'):

        """
        Gerchberg-Saxton algorithm for iterative phase retrieval
        """
        slm_field = init_light.clone()
        prop = Propagator(prop_model)

        for i in range(n_iter):

            if i % 10 == 0:
                psnr = calculate_psnr(prop.forward(slm_field, -z), target_light)
                print(f"Iteration {i}, PSNR: {psnr:.2f} dB")

            slm_field.set_amplitude_ones()
            recon_field = prop.forward(slm_field, -z)
            recon_field.set_amplitude(target_light.get_amplitude())
            slm_field = prop.forward(recon_field, z)

        return slm_field

    def stochastic_gradient_descent(self, init_light, target_light, z, n_iter=100, prop_model='ASM', lr=500000):
        """
        Stochastic Gradient Descent algorithm for iterative phase retrieval.
        """
        slm_field = init_light.clone()
        prop = Propagator(prop_model)
        target_amp = target_light.get_amplitude().detach()

        optimizer = torch.optim.SGD([slm_field.field.requires_grad_()], lr=lr)
        loss_fn = nn.MSELoss()

        for i in range(n_iter):
            optimizer.zero_grad()
            recon_field = prop.forward(slm_field, -z)
            recon_amp = recon_field.get_amplitude()
            loss = loss_fn(recon_amp, target_amp)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")

        return slm_field

    def adaptive_moment_estimation(self, init_light, target_light, z, n_iter=100, prop_model='ASM', lr=1e-1, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Adaptive Moment Estimation (Adam) algorithm for iterative phase retrieval.
        """
        slm_field = init_light.clone()
        prop = Propagator(prop_model)
        target_amp = target_light.get_amplitude().detach()

        optimizer = torch.optim.Adam([slm_field.field.requires_grad_()], lr=lr, betas=(beta1, beta2), eps=eps)
        loss_fn = nn.MSELoss()

        for i in range(n_iter):
            optimizer.zero_grad()
            recon_field = prop.forward(slm_field, -z)
            recon_amp = recon_field.get_amplitude()

            loss = loss_fn(recon_amp, target_amp)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")

        return slm_field