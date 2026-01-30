import torch
import numpy as np
import torch.nn.functional as Func
from .light import Light
from scipy.special import j1 # J1: First-order Bessel function of the first kind 

class Dun_propagator_1D:
    # This function must be carefully used to satisfy input_pitch <= wavelength/(2*NA) gurantee the accuracy, where NA is the numerical aperture.
    def __init__(self, input_dim, output_dim, input_pitch, output_pitch, wvl, focal_length, input_field_vector=None, device='cpu'):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_pitch = input_pitch
        self.output_pitch = output_pitch
        self.device = device
        self.wvl = wvl
        self.focal_length = focal_length
    
        if input_field_vector is None:
            input_field_vector = torch.ones(input_dim, device=device, dtype=torch.cfloat)
        self.input_field_vector = input_field_vector

        """ Compute H(r_m, rho) in the paper Dun et al.,
         "Learned Rotationally Symmetric Diffractive Achromat for Full-Spectrum Computational Imaging: Supplementary Information", Optical (2020)"""
        self.rr = (np.arange(0, self.input_dim) * self.input_pitch).astype(np.float32)
        self.rhorho = (np.arange(0, self.output_dim) * self.output_pitch / self.wvl / self.focal_length).astype(np.float32)
        self.H = (np.zeros((len(self.rr), len(self.rhorho)))).astype(np.float32)
        
        # Further optimization needed (use dynamic programming)
        for m in range(len(self.rr)):
            if m==0:
                self.H[m] = 1/2/np.pi/(self.rhorho+1e-9) * self.rr[m] * j1(2*np.pi*self.rhorho*self.rr[m])
            else:
                self.H[m] = 1/2/np.pi/(self.rhorho+1e-9)*(self.rr[m]*j1(2*np.pi*self.rhorho*self.rr[m]) - self.rr[m-1]*j1(2*np.pi*self.rhorho*self.rr[m-1]))
        self.rr = torch.from_numpy(self.rr).to(self.device)
        self.rhorho = torch.from_numpy(self.rhorho).to(self.device)
        self.H = torch.from_numpy(self.H).to(self.device)
        
    def compute_psf(self):
        k = 2*np.pi/self.wvl
        coe = (k/self.focal_length)*torch.exp(1j*k/2/self.focal_length*((self.wvl*self.focal_length*self.rhorho)**2))
        acc = 0
        for m in range(len(self.rr)):
            acc = acc + self.input_field_vector[m] * torch.exp(1j*k/2/self.focal_length*((self.rr[m])**2)) * self.H[m]
        return abs(coe*acc)**2
    

class Yoon_propagator_1D:
    # Unlike Dun et al.'s propagator, I directly use Bessel function as the solution of the propagation of the circular aperture
    # This function must be carefully used to satisfy input_pitch <= wavelength/(2*NA) gurantee the accuracy, where NA is the numerical aperture.
    def __init__(self, input_dim, output_dim, input_pitch, output_pitch, wvl, focal_length, input_field_vector=None, device='cpu'):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_pitch = input_pitch
        self.output_pitch = output_pitch
        self.device = device
        self.wvl = wvl
        self.focal_length = focal_length
    
        if input_field_vector is None:
            input_field_vector = torch.ones(input_dim, device=device, dtype=torch.cfloat)
        self.input_field_vector = input_field_vector

        """ Compute H(r_m, rho) in the paper Dun et al.,
         "Learned Rotationally Symmetric Diffractive Achromat for Full-Spectrum Computational Imaging: Supplementary Information", Optical (2020)"""
        self.rr = (np.arange(0, self.input_dim) * self.input_pitch).astype(np.float32)
        self.rhorho = (np.arange(0, self.output_dim) * self.output_pitch / self.wvl / self.focal_length).astype(np.float32)
        self.H = (np.zeros((len(self.rr), len(self.rhorho)))).astype(np.float32)
        
        # Further optimization needed (use dynamic programming)
        for m in range(len(self.rr)):
            if m==0:
                self.H[m] = 1/2/np.pi/(self.rhorho+1e-9) * self.rr[m] * j1(2*np.pi*self.rhorho*self.rr[m])
            else:
                self.H[m] = 1/2/np.pi/(self.rhorho+1e-9)*(self.rr[m]*j1(2*np.pi*self.rhorho*self.rr[m]) - self.rr[m-1]*j1(2*np.pi*self.rhorho*self.rr[m-1]))
        self.rr = torch.from_numpy(self.rr).to(self.device)
        self.rhorho = torch.from_numpy(self.rhorho).to(self.device)
        self.H = torch.from_numpy(self.H).to(self.device)
        
    def compute_psf(self):
        k = 2*np.pi/self.wvl
        coe = (k/self.focal_length)*torch.exp(1j*k/2/self.focal_length*((self.wvl*self.focal_length*self.rhorho)**2))
        acc = 0
        for m in range(len(self.rr)):
            acc = acc + self.input_field_vector[m] * torch.exp(1j*k/2/self.focal_length*((self.rr[m])**2)) * self.H[m]
        return abs(coe*acc)**2