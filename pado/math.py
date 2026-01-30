import torch
import torch.nn.functional as F
import numpy as np
from math import log10
"""
At Pado, all measurements adhere to the International System of Units (SI).
"""
nm = 1e-9
um = 1e-6
mm = 1e-3
cm = 1e-2
m = 1

s = 1
ms = 1e-3
us = ms * 1e-3
ns = us * 1e-3

def wrap_phase(phase_u, stay_positive=False):
    """
    Return the wrapped phase in the range [-pi, pi] or [0, 2pi].
    Args:
        phase_u: tensor of unwrapped phase
        stay_positive: keep the wrapped phase to be positive. Default is False
    Retruns:
        phase_w: wrapped phase
    """
    phase = phase_u%(2*np.pi)
    if stay_positive == False:
        phase[phase>torch.pi] -= (2*np.pi)
    return phase



def fft(arr_c, normalized="backward", pad_width=None, padval=0, shift=True):
    """
    Compute the Fast Fourier transform of a complex tensor 
    Args:
        arr_c: [B,Ch,R,C] complex tensor 
        normalized: Normalize the FFT output. default: "backward" See torch.fft.fft2
        pad_width: (tensor) pad width for the last spatial dimensions. 
        shift: flag for shifting the input data to make the zero-frequency located at the center of the arrc
    Returns:
        arr_c_fft: [B,Ch,R,C] FFT of the input complex tensor
    """

    if pad_width is not None:
        if padval == 0:
            arr_c = torch.nn.functional.pad(arr_c, pad_width)
        else:
            raise NotImplementedError('zero padding is only implemented for now')

    arr_c_shifted = torch.fft.ifftshift(arr_c, dim=(-2,-1)) if shift else arr_c
    arr_c_shifted_fft_c = torch.fft.fft2(arr_c_shifted , norm=normalized)
    arr_c_fft = torch.fft.fftshift(arr_c_shifted_fft_c, dim=(-2,-1)) if shift else arr_c_shifted_fft_c

    return arr_c_fft



def ifft(arr_c, normalized="backward", pad_width=None, shift=True):
    """
    Compute the inverse Fast Fourier transform of a complex tensor 
    Args:
        arr_c: [B,Ch,R,C] complex tensor 
        normalized: Normalize the iFFT output. default: "backward" See torch.fft.ifft2
        pad_width: (tensor) pad width for the last spatial dimensions. 
        shift: flag for inversely shifting the input data 
    Returns:
        arr_c_fft: [B,Ch,R,C] inverse FFT of the input complex tensor
    """

    arr_c_shifted = torch.fft.ifftshift(arr_c, dim=(-2,-1)) if shift else arr_c
    arr_c_shifted_fft_c = torch.fft.ifft2(arr_c_shifted, norm=normalized)
    arr_c_fft = torch.fft.fftshift(arr_c_shifted_fft_c, dim=(-2,-1)) if shift else arr_c_shifted_fft_c

    if pad_width is not None:
        if pad_width[2] != 0 and pad_width[3] != 0:
            arr_c_fft = arr_c_fft[..., pad_width[2]:-pad_width[3], :]
        if pad_width[0] != 0 and pad_width[1] != 0:
            arr_c_fft = arr_c_fft[..., :, pad_width[0]:-pad_width[1]]

    return arr_c_fft


def conv_fft(img_c, kernel_c, pad_width=None):
    """
    Compute the convolution of an image with a convolution kernel using FFT
    Args:
        img_c: [B,Ch,R,C] image as a complex tensor 
        kernel_c: [B,Ch,R,C] convolution kernel as a complex tensor
        pad_width: (tensor) pad width for the last spatial dimensions. should be (0,0,0,0) for circular convolution. for linear convolution, pad zero by the size of the original image
    Returns:
        im_conv: [B,Ch,R,C] blurred image
    """

    img_fft = fft(img_c, pad_width=pad_width)
    kernel_fft = fft(kernel_c, pad_width=pad_width)
    return ifft( img_fft * kernel_fft, pad_width=pad_width)

def calculate_psnr(light1, light2):
    """
    Calculate the PSNR between two images represented by two Light class instances.
    
    Args:
        light1 (Light): First light wave instance.
        light2 (Light): Second light wave instance.

    Returns:
        float: The PSNR value between two light wavefronts.
    """
    img1 = light1.get_amplitude()
    img2 = light2.get_amplitude()
    
    # Ensure the tensors are on the same device and dtype
    if light1.get_device() != light2.get_device():
        img2 = img2.to(img1.device)

    mse = F.mse_loss(img1, img2, reduction='mean')
    
    if mse == 0:
        return float('inf')
    max_pixel = 1.0 
    psnr = 20 * log10(max_pixel / torch.sqrt(mse))

    # print(f"PSNR: {psnr} dB")
    return psnr

def calculate_ssim(light1, light2):
    """
    Calculate the SSIM between two images represented by two Light class instances.

    Args:
        light1 (Light): First light wave instance.
        light2 (Light): Second light wave instance.

    Returns:
        float: The SSIM value between two light wavefronts.
    """
    img1 = light1.get_amplitude()
    img2 = light2.get_amplitude()
    
    # Ensure the tensors are on the same device and dtype
    if light1.get_device() != light2.get_device():
        img2 = img2.to(img1.device)

    ssim_value = ssim_index(img1, img2, window_size=11, sigma=1.5, data_range=1.0)
    
    # print(f"SSIM: {ssim_value}")
    return ssim_value


def gaussian_window(size, sigma):
    """
    Create a 2D Gaussian window.
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    grid = torch.meshgrid(coords, coords)
    window = torch.exp(-(grid[0] ** 2 + grid[1] ** 2) / (2 * sigma ** 2))
    return window / window.sum()

def ssim_index(img1, img2, window_size=11, sigma=1.5, data_range=1.0):
    """
    Calculate the SSIM index for two images.
    """
    window = gaussian_window(window_size, sigma).to(img1.device)
    window = window.expand(img1.shape[1], 1, window_size, window_size)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

##########################
# Additional Helper Functions for Sc-ASM
##########################

def sc_dft_1d(g, M, delta_x, delta_fx):
    """
    Compute the 1D scaled inverse DFT of G(fx).
    """
    device = g.device
    beta = np.pi * delta_fx * delta_x

    M2 = 2*M

    g_padded = torch.zeros(M2, dtype=torch.complex64, device=device)
    g_padded[M//2:M//2+M] = g

    m_big = torch.arange(-M, M, dtype=torch.float32, device=device)  # length = M2

    q1 = g_padded * torch.exp(-1j * beta * (m_big**2))
    q2 = torch.exp(1j * beta * (m_big**2))

    Q1 = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(q1)))
    Q2 = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(q2)))
    conv = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(Q1*Q2)))
    conv = conv[M//2:M//2+M]

    p = torch.arange(-M//2, M//2, dtype=torch.float32, device=device)
    G = delta_x * torch.exp(-1j * beta * (p**2)) * conv

    return G


def sc_idft_1d(G, M, delta_fx, delta_x):
    """
    Compute the 1D scaled inverse DFT of G(fx).
    Similar approach as sc_dft_1d but for inverse transform.
    """
    device = G.device
    beta = np.pi * delta_fx * delta_x
    M2 = 2*M

    G_padded = torch.zeros(M2, dtype=torch.complex64, device=device)
    G_padded[M//2:M//2+M] = G

    m_big = torch.arange(-M, M, dtype=torch.float32, device=device)

    q1_inv = G_padded * torch.exp(-1j * beta * (m_big**2))
    q2_inv = torch.exp(1j * beta * (m_big**2))

    Q1 = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(q1_inv)))
    Q2 = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(q2_inv)))
    conv = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(Q1*Q2)))
    conv = conv[M//2:M//2+M]

    p = torch.arange(-M//2, M//2, dtype=torch.float32, device=device)

    xdomain = delta_fx * torch.exp(-1j*beta*(p**2)) * conv

    return xdomain



def sc_dft_2d(u, Mx, My, delta_x, delta_y, delta_fx, delta_fy):
    """
    2D scaled DFT based on performing Sc-DFT along x and y independently.
    u: 2D field [R, C]
    """
    U_intermediate = torch.zeros_like(u, dtype=torch.complex64)
    for iy in range(My):
        U_intermediate[iy, :] = sc_dft_1d(u[iy, :], Mx, delta_x, delta_fx)

    U_final = torch.zeros_like(U_intermediate, dtype=torch.complex64)
    for ix in range(Mx):
        U_final[:, ix] = sc_dft_1d(U_intermediate[:, ix], My, delta_y, delta_fy)

    return U_final

def sc_idft_2d(U, Mx, My, delta_x, delta_y, delta_fx, delta_fy):
    """
    2D scaled inverse DFT by applying sc_idft_1d on y then x (or x then y).
    """
    u_intermediate = torch.zeros_like(U, dtype=torch.complex64)
    for ix in range(Mx):
        u_intermediate[:, ix] = sc_idft_1d(U[:, ix], My, delta_fy, delta_y)

    u_final = torch.zeros_like(u_intermediate, dtype=torch.complex64)
    for iy in range(My):
        u_final[iy, :] = sc_idft_1d(u_intermediate[iy, :], Mx, delta_fx, delta_x)

    return u_final

def compute_scasm_transfer_function(Mx, My, delta_fx, delta_fy, λ, z):
    """
    Compute the transfer function for Sc-ASM given Mx, My, delta_fx, delta_fy, wavelength λ, and propagation distance z.
    """
    fx = (torch.arange(-Mx//2, Mx//2, dtype=torch.float32)*delta_fx)
    fy = (torch.arange(-My//2, My//2, dtype=torch.float32)*delta_fy)
    fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')
    fxx = fxx
    fyy = fyy
    k = 2*np.pi/λ

    gamma = torch.sqrt(1 - (λ*fxx)**2 - (λ*fyy)**2 + 0j)
    H = torch.exp(1j*k*z*gamma)
    return H