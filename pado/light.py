import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from .math import nm, um, mm, cm, m

class Light:
    def __init__(self, dim, pitch, wvl, field=None, device='cpu'):
        """
        Light wave that has a complex field as a wavefront

        Args:
            dim: (B, Ch, R, C) batch_size, channel, row, and column of the field 
            pitch: pixel pitch in meter
            wvl: wavelength of light in meter
            field: [batch_size, # of channels, row, column] tensor of wavefront, default is None
            device: device to store the wavefront of light. 'cpu', 'cuda:0', ...
        """

        self.dim = dim
        self.pitch = pitch
        self.device = device
        self.wvl = wvl
    
        if field is None:

            field = torch.ones(dim, device=device, dtype=torch.cfloat)
        self.field = field

    def crop(self, crop_width):
        """
        Crop the light wavefront by crop_width
        Args:
            crop_width: (tuple) crop width of the tensor following torch functional pad 
        """

        self.field = self.field[...,crop_width[2]:None if crop_width[3]==0 else -crop_width[3], crop_width[0]:None if crop_width[1]==0 else -crop_width[1]]
        self.dim = self.field.shape

    def clone(self):
        """
        Clone the light and return it
        """

        return Light(self.dim, self.pitch, self.wvl, self.field.clone(), device=self.device)


    def pad(self, pad_width, padval=0):
        """
        Pad the light wavefront with a constant value by pad_width
        Args:
            pad_width: (tuple) pad width of the tensor following torch functional pad 
            padval: value to pad. default is zero
        """

        if padval == 0:
            self.field = torch.nn.functional.pad(self.field, pad_width)
        else:
            raise NotImplementedError('only zero padding supported')
        # self.dim[2], self.dim[3] = self.dim[2]+pad_width[0]+pad_width[1], self.dim[3]+pad_width[2]+pad_width[3]
        self.dim = (self.dim[0], self.dim[1], 
                    self.dim[2]+pad_width[0]+pad_width[1], self.dim[3]+pad_width[2]+pad_width[3])

    def set_real(self, real, c=None):
        """
        Set the real part of the light wavefront
        Args:
            real: real part in the rect representation of the complex number 
        """
        if c is not None:
            self.field.real[:,c,...] = real
        else:
            self.field.real = real 

    def set_imag(self, imag, c=None):
        """
        Set the imaginary part of the light wavefront
        Args:
            imag: imaginary part in the rect representation of the complex number 
        """
        if c is not None:
            self.field.imag[:,c,...] = imag
        else:
            self.field.imag = imag
        
    def set_amplitude(self, amplitude, c=None):
        """
        Set the amplitude of the light wavefront
        Args:
            amplitude: amplitude in the polar representation of the complex number 
        """
        if c is not None:
            self.field[:,c,...] = amplitude*torch.exp(self.field[:,c,...].angle()*1j)
        else:
            phase = self.field.angle()
            self.field = amplitude*torch.exp(phase*1j)

    def set_phase(self, phase, c=None):
        """
        Set the phase of the complex tensor
        Args:
            phase: phase in the polar representation of the complex number 
        """
        if c is not None:
            self.field[:,c,...] = self.field[:,c,...].abs()*torch.exp(phase*1j)
        else:
            amplitude = self.field.abs()
            self.field = amplitude*torch.exp(phase*1j)

    def set_field(self, field, c=None):
        """
        Set the wavefront modulation of the complex tensor
        Args:
            field: wavefront as a complex number
        """
        if c is not None:
            self.field[:,c,...] = field
        else:
            self.field = field

    def set_pitch(self, pitch):
        """
        Set the pixel pitch of the complex tensor
        Args:
            pitch: pixel pitch in meter
        """
        self.pitch = pitch


    def get_channel(self):
        """
        Return the number of channels of the light wavefront
        Returns:
            channel: number of channels
        """

        return self.dim[1]
        

    def get_amplitude(self, c=None):
        """
        Return the amplitude of the wavefront
        Returns:
            mag: magnitude in the polar representation of the complex number 
        """
        if c is not None:
            return self.field[:,c,...].abs()
        else:
            return self.field.abs()

    def get_phase(self, c=None):
        """
        Return the phase of the wavefront
        Returns:
            ang: angle in the polar representation of the complex number
        """
        if c is not None:
            return self.field[:,c,...].angle()
        else:
            return self.field.angle()
        
    def get_intensity(self, c=None):
        """
        Return the intensity of light wavefront
        Returns:
            intensity: intensity of light
        """
        if c is not None:
            return (self.field[:,c,...] * torch.conj(self.field[:,c,...])).real
        else:
            return (self.field * torch.conj(self.field)).real 


    def get_field(self, c=None):
        """
        Return the complex wavefront
        Returns:
            field: complex wavefront
        """
        if c is not None:
            return self.field[:,c,...]
        else:
            return self.field
        
    def get_device(self):
        """
        Return the device of the light wavefront
        Returns:
            device: device of the light wavefront
        """
        return self.device

    def get_intensity(self, c=None):
        """
        Return the intensity of light wavefront
        Returns:
            intensity: intensity of light
        """
        if c is not None:
            return (self.field[:,c,...] * torch.conj(self.field[:,c,...])).real
        else:
            return (self.field * torch.conj(self.field)).real

    def get_bandwidth(self):
        """
        Return the bandwidth of light wavefront
        Returns:
            R_m: spatial height of the wavefront 
            C_m: spatial width of the wavefront 
        """

        return self.pitch*self.dim[2], self.pitch*self.dim[3]
    
    def get_ideal_angle_limit(self):
        """
        Return the ideal angle limit of light wavefront based on the spatial freqeuncy.
        Use the following relation between diffraction angle and spatial freqeuncy: sin(theta)/wvl = spatial_frequency,
        which would be derived from angular spectrum
        Automatically selects the longest wavelength for calculations.
        
        Returns:
            Ideal_angle_limit: ideal angle limit of light wavefront (deg)
        """

        # Check if wvl is a list or a single float, and select the longest wavelength
        if hasattr(self.wvl, "__iter__") and not isinstance(self.wvl, str):
            min_wvl = min(self.wvl)
        else:
            min_wvl = self.wvl

        sin_val = (min_wvl / self.pitch) * 0.5
        Ideal_angle_limit = np.arcsin(sin_val) * 180 / np.pi
        
        return Ideal_angle_limit - 0.0001


    def magnify(self, scale_factor, interp_mode='nearest', c=None):
        '''
        Change the wavefront resolution without changing the pixel pitch
        Args:
            scale_factor: scale factor for interpolation used in tensor.nn.functional.interpolate
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        '''
        if c is not None:
            self.field[:,c,...] = F.interpolate(self.field.real[:,c,...], scale_factor=scale_factor, mode=interp_mode) +\
                                  F.interpolate(self.field.imag[:,c,...], scale_factor=scale_factor, mode=interp_mode) * 1j
        else:
            self.field = F.interpolate(self.field.real, scale_factor=scale_factor, mode=interp_mode) +\
                         F.interpolate(self.field.imag, scale_factor=scale_factor, mode=interp_mode) * 1j
        self.dim = (self.dim[0], self.dim[1], self.field.shape[2], self.field.shape[3])



    def resize(self, target_pitch, interp_mode='nearest'):
        '''
        Resize the wavefront by changing the pixel pitch. 
        Args:
            target_pitch: new pixel pitch to use
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        '''
        scale_factor = self.pitch / target_pitch
        self.magnify(scale_factor, interp_mode)
        self.set_pitch(target_pitch)

    def set_spherical_light(self, z, dx=0., dy=0.):
        '''
        Set the wavefront as spherical one coming from the position of (dx,dy,z). 
        Args:
            z: z distance of the spherical light source from the current light position
            dx: x distance of the spherical light source from the current light position
            dy: y distance of the spherical light source from the current light position
        '''
        x = torch.arange(-self.dim[3]//2, self.dim[3]//2, device=self.device, dtype=torch.float64) * self.pitch
        y = torch.arange(-self.dim[2]//2, self.dim[2]//2, device=self.device, dtype=torch.float64) * self.pitch
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        r = torch.sqrt((xx - dx) ** 2 + (yy - dy) ** 2 + z ** 2)  # this is computed in double precision
        if hasattr(self.wvl, "__iter__") and (not isinstance(self.wvl, str)):
            wvl_tensor = torch.tensor(self.wvl, device=self.device, dtype=r.dtype).view(1, len(self.wvl), 1, 1)
            theta = ((2 * torch.pi * r / wvl_tensor) % (2*torch.pi)).to(torch.float32)
        else:
            theta = ((2 * torch.pi * r / self.wvl) % (2*torch.pi)).to(torch.float32)
            theta = torch.tensor(theta, device=self.device)[None, None, :, :]
        mag = torch.ones_like(theta)
        self.set_field(mag*torch.exp(theta*1j))

    def set_plane_light(self, theta=0):
        '''
        Set the wavefront as a plane wave with zero phase and amptliude of one
        '''
        R, C = self.dim[-2], self.dim[-1]
        amplitude = torch.ones((1, 1, self.dim[-2], self.dim[-1]), device=self.device)
        phase = torch.zeros((1, 1, self.dim[-2], self.dim[-1]), device=self.device)

        dx = torch.linspace(-C * self.pitch / 2,C *self.pitch / 2 , R).to(self.device)
        dy = torch.linspace(-R * self.pitch / 2, R * self.pitch/ 2 , C).to(self.device)
        dx, dy = torch.meshgrid(dx, dy) # R x C 
        dx = dx[None, None, :, :]
        dy = dy[None, None, :, :]

        term = -2 * torch.pi * dx * np.sin(np.deg2rad(theta)) / self.wvl
        phase = phase - term.to(torch.float32) #- 2 * torch.pi * dx * torch.sin(theta_) / self.wvl #r * torch.sin(theta_)

        self.set_field(amplitude*torch.exp(phase*1j))

    def set_amplitude_ones(self):
        '''
        Set the amplitude of the wavefront to one
        '''
        self.set_amplitude(torch.ones_like(self.get_amplitude()))

    def set_amplitude_zeros(self):
        '''
        Set the amplitude of the wavefront to zero
        '''
        self.set_amplitude(torch.zeros_like(self.get_amplitude()))

    def set_phase_pi(self):
        '''
        Set the phase of the wavefront to pi
        '''
        self.set_phase(torch.ones_like(self.get_phase())*np.pi)

    def set_phase_zeros(self):
        '''
        Set the phase of the wavefront to zero
        '''
        self.set_phase(torch.zeros_like(self.get_phase()))

    def set_phase_random(self):
        '''
        Set the phase of the wavefront to random values between 0 and pi
        '''
        self.set_phase((torch.rand(self.dim[2], self.dim[3], device=self.device) * 3.14))

    def save(self, fn):
        '''
        Save the amplitude and phase of the light wavefront as a file
        Args:
            fn: filename to save. the format should be either "npy" or "mat"

        '''
        field_np = self.get_field().data.cpu().numpy()
        if fn[-3:] == 'npy':
            np.save(fn, field_np)
        elif fn[-3:] == 'mat':
            savemat(fn, {'field':field_np})
        else:
            print('extension in %s is unknown'%fn)
        print('light saved to %s\n'%fn)
    
    def adjust_amplitude_to_other_light(self, other_light):
        """
        Adjusts the amplitude of the current light instance to match the average amplitude
        of another light instance.

        Args:
            other_light (Light): Another light instance to compare with.
        """
        other_amplitude = other_light.get_amplitude()
        other_average_amplitude = torch.mean(other_amplitude)

        current_amplitude = self.get_amplitude()
        current_average_amplitude = torch.mean(current_amplitude)

        scale_factor = other_average_amplitude / current_average_amplitude

        scaled_amplitude = current_amplitude * scale_factor

        self.set_amplitude(scaled_amplitude)

        print(f"Amplitude scaled by factor: {scale_factor}")

    def load_image(self, image_path, random_phase=False, grayscale=False):
        """
        Loads an image, optionally applies a random phase to it, and sets it as the amplitude
        (and phase, if random_phase=True) of the light's field. Can load the image directly as grayscale.

        Args:
            image_path: Path to the image file.
            random_phase: If True, applies a random phase to the image. If False, uses the image as is.
            grayscale: If True, loads the image as grayscale. If False, loads as RGB.
        """
        if grayscale:
            img = plt.imread(image_path)
            if len(img.shape) == 3:  # If the image is in RGB format
                img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
        else:
            img = plt.imread(image_path)

        img_tensor = torch.tensor(img, device=self.device, dtype=torch.float32)

        if img_tensor.max() > 1:
            img_tensor /= 255.0

        if not grayscale:
            if img_tensor.shape[-1] == 4: 
                img_tensor = img_tensor[..., :3]

            if self.dim[1] == 1: 
                img_tensor = img_tensor.mean(dim=-1, keepdim=True)
        else:
            img_tensor = img_tensor.unsqueeze(-1)  # Add channel dimension for grayscale

        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Rearrange dimensions to [B, C, H, W]

        img_tensor_resized = F.interpolate(img_tensor, size=(self.dim[2], self.dim[3]), mode='bilinear', align_corners=False)

        amplitude = torch.sqrt(img_tensor_resized)

        if random_phase:
            phase = (torch.rand(self.dim[2], self.dim[3], device=self.device) * 2 - 1) * np.pi
            field = (amplitude * torch.exp(1j * phase)).to(self.device)
        else:
            field = amplitude.to(self.device) + 0j

        self.field = field


    def visualize(self, b=0, c=None, uniform_scale=False, vmin=None, vmax=None):
        """
        Visualize the light wave for the specified batch and channel.
        If the channel is not specified, visualizes all channels separately.
        Adds an option to unify the scale across all channels based on the first channel's scale.
        Args:
            b: Batch index to visualize (default is 0).
            c: Channel index to visualize. If None, visualizes all channels.
            uniform_scale: If True, uses the first channel's scale for all channels (default is True).
        """
        # Determine the bandwidth based on pitch and dimensions.
        bw = self.get_bandwidth()
        
        # Check if a specific channel is requested or visualize all channels.
        channels = [c] if c is not None else range(self.get_channel())

        # Initialize variables to store the first channel's scale for uniform scaling.
        vmin_amplitude, vmax_amplitude = None, None

        for chan in channels:
            plt.figure(figsize=(20, 5))

            # Amplitude
            plt.subplot(131)
            amplitude = self.get_amplitude(c=chan).cpu().numpy().squeeze()
            if uniform_scale and vmin_amplitude is not None and vmax_amplitude is not None:
                plt.imshow(amplitude, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno', vmin=vmin_amplitude, vmax=vmax_amplitude)
            else:
                vmin_amplitude, vmax_amplitude = amplitude.min(), amplitude.max()
                plt.imshow(amplitude, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno', vmin=vmin_amplitude, vmax=vmax_amplitude)
            plt.title(f'Amplitude (Channel {chan})')
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.colorbar()

            # Phase
            plt.subplot(132)
            phase = self.get_phase(c=chan).cpu().numpy().squeeze()
            plt.imshow(phase, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='hsv', vmin=-np.pi, vmax=np.pi)
            plt.title(f'Phase (Channel {chan})')
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.colorbar()

            # Intensity
            plt.subplot(133)
            intensity = self.get_intensity(c=chan).cpu().numpy().squeeze()
            if uniform_scale and vmin_amplitude is not None and vmax_amplitude is not None:
                plt.imshow(intensity, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno', vmin=vmin if vmin is not None else vmin_amplitude, vmax=vmax if vmax is not None else vmax_amplitude)
            else:
                plt.imshow(intensity, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno', vmin=vmin if vmin is not None else None, vmax=vmax if vmax is not None else None)
            plt.title(f'Intensity (Channel {chan})')
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.colorbar()

            wvl_text = f'{self.wvl[chan]*1e9:.2f} [nm]' if isinstance(self.wvl, list) else f'{self.wvl*1e9:.2f} [nm]'
            plt.suptitle(f'({self.dim[2]},{self.dim[3]}), pitch:{self.pitch*1e6:.2f} [um], wvl:{wvl_text}, device:{self.device}')
            plt.show()

    def visualize_image(self, b=0):
        """
        Visualize the amplitude, phase, and intensity as RGB images for a specified batch index with colorbars,
        including detailed title with all wavelengths, pixel pitch, and device information.

        Args:
            b: Batch index to visualize.
        """
        bw = self.get_bandwidth()

        fig, axes = plt.subplots(1, 3, figsize=(16, 8))

        # Amplitude as RGB, clamped to [0, 1]
        amplitude = self.get_amplitude().data.cpu()[b, ...].permute(1, 2, 0).squeeze()
        amplitude_clamped = torch.clamp(amplitude, min=0, max=1)  # Clamp values to [0, 1]
        img0 = axes[0].imshow(amplitude_clamped, extent=[0, bw[1]*1e3, 0, bw[0]*1e3])
        axes[0].set_title('Amplitude as RGB Image')
        axes[0].set_xlabel('mm')
        axes[0].set_ylabel('mm')
        # fig.colorbar(img0, ax=axes[0], orientation='vertical')

        # Phase as RGB, normalized from -π to π
        phase = self.get_phase().data.cpu()[b, ...].permute(1, 2, 0).squeeze()
        phase_normalized = (phase + np.pi) / (2 * np.pi)  # Normalize from 0 to 1 for color mapping
        img1 = axes[1].imshow(phase_normalized, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='hsv')
        axes[1].set_title('Phase as RGB Image')
        axes[1].set_xlabel('mm')
        # Set the colorbar with ticks at normalized -π, 0, π
        # cbar = fig.colorbar(img1, ax=axes[1], orientation='vertical', ticks=[0, 0.5, 1])
        # cbar.ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])  # Set labels to show -π to π

        # Intensity as RGB, clamped to [0, 1]
        intensity = self.get_intensity().data.cpu()[b, ...].permute(1, 2, 0).squeeze()
        intensity_clamped = torch.clamp(intensity, min=0, max=1)  # Clamp values to [0, 1]
        img2 = axes[2].imshow(intensity_clamped, extent=[0, bw[1]*1e3, 0, bw[0]*1e3])
        axes[2].set_title('Intensity as RGB Image')
        axes[2].set_xlabel('mm')
        axes[2].set_ylabel('mm')
        # fig.colorbar(img2, ax=axes[2], orientation='vertical')

        # Format wavelengths for display
        if isinstance(self.wvl, list):
            wvl_text = ', '.join([f'{w/1e-9:.2f}[nm]' for w in self.wvl])
        else:
            wvl_text = f'{self.wvl/1e-9:.2f}[nm]'

        plt.suptitle(f'({self.dim[2]},{self.dim[3]}), pitch: {self.pitch/1e-6:.2f}[um], wvl: {wvl_text}, device: {self.device}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
    def shape(self):
        """
        Returns the shape of light wavefront
        Returns:
            shape
        """
        return self.field.shape()

class PolarizedLight(Light):
    def __init__(self, dim, pitch, wvl, fieldX=None, fieldY=None, device='cuda:0'):
        """
        Light wave that has a polarized complex field as a wavefront

        Args:
            dim: (B, 1, R, C) batch_size, row, and column of the field 
            pitch: pixel pitch in meter
            wvl: wavelength of light in meter
            fieldX: [batch_size, # of channels, row, column] tensor of wavefront of X component, default is None
            fieldY: [batch_size, # of channels, row, column] tensor of wavefront of Y component, default is None
            device: device to store the wavefront of light. 'cpu', 'cuda:0', ...
        """
        self.dim = dim
        self.pitch = pitch
        self.device = device
        self.wvl = wvl
        
        fieldX = torch.ones(dim, device=device, dtype=torch.cfloat) if fieldX is None else fieldX 
        fieldY = torch.ones(dim, device=device, dtype=torch.cfloat) if fieldY is None else fieldY 
        self.lightX = Light(dim, pitch, wvl, fieldX, device)
        self.lightY = Light(dim, pitch, wvl, fieldY, device)

    def get_fieldX(self):
        return self.lightX.get_field()

    def get_fieldY(self):
        return self.lightY.get_field()
     
    def crop(self, crop_width):
        """
        Crop the light wavefront by crop_width
        Args:
            crop_width: (tuple) crop width of the tensor following torch functional pad 
        """
        self.lightX.crop(crop_width)
        self.lightY.crop(crop_width)
        
        self.dim[2], self.dim[3] = self.lightX.dim[2], self.lightX.dim[3]
        
    def clone(self):
        """
        Clone the light and return it
        """
        return PolarizedLight(self.dim, self.pitch, self.wvl, self.get_fieldX().clone(), self.get_fieldX().clone(), device=self.device)
        
    def pad(self, pad_width, padval=0):
        """
        Pad the light wavefront with a constant value by pad_width
        Args:
            pad_width: (tuple) pad width of the tensor following torch functional pad 
            padval: value to pad. default is zero
        """
        if padval == 0:
            self.lightX.pad(pad_width)
            self.lightY.pad(pad_width)
        else:
            raise NotImplementedError('only zero padding supported')

        self.dim[2], self.dim[3] = self.dim[2]+pad_width[0]+pad_width[1], self.dim[3]+pad_width[2]+pad_width[3]

    def set_realX(self, real):
        """
        Set the real part of the light wavefront of X component
        Args:
            real: real part in the rect representation of the complex number 
        """
        self.lightX.set_real(real)
        
    def set_imagX(self, imag):
        """
        Set the imaginary part of the light wavefront of X component
        Args:
            imag: imaginary part in the rect klrepresentation of the complex number
        """
        self.lightX.set_imag(imag)
        
    def set_amplitudeX(self, amplitude):
        """
        Set the amplitude of the light wavefront of X component
        Args:
            amplitude: amplitude in the polar representation of the complex number 
        """
        self.lightX.set_amplitude(amplitude)

    def set_phaseX(self, phase):
        """
        Set the phase of the complex tensor of X component
        Args:
            phase: phase in the polar representation of the complex number 
        """
        self.lightX.set_phase(phase)

    def set_fieldX(self, field):
        """
        Set the wavefront modulation of the complex tensor of X component
        Args:
            field: wavefront as a complex number
        """
        self.lightX.set_field(field)

    def set_realY(self, real):
        """
        Set the real part of the light wavefront of Y component
        Args:
            real: real part in the rect representation of the complex number 
        """
        self.lightY.set_real(real)
        
    def set_imagY(self, imag):
        """
        Set the imaginary part of the light wavefront of Y component
        Args:
            imag: imaginary part in the rect klrepresentation of the complex number
        """
        self.lightY.set_imag(imag)
        
    def set_amplitudeY(self, amplitude):
        """
        Set the amplitude of the light wavefront of Y component
        Args:
            amplitude: amplitude in the polar representation of the complex number 
        """
        self.lightY.set_amplitude(amplitude)

    def set_phaseY(self, phase):
        """
        Set the phase of the complex tensor of Y component
        Args:
            phase: phase in the polar representation of the complex number 
        """
        self.lightY.set_phase(phase)

    def set_fieldY(self, field):
        """
        Set the wavefront modulation of the complex tensor of Y component
        Args:
            field: wavefront as a complex number
        """
        self.lightY.set_field(field)

    def set_pitch(self, pitch):
        """
        Set the pixel pitch of the complex tensor
        Args:
            pitch: pixel pitch in meter
        """
        self.pitch = pitch

    def get_amplitude(self):
        """
        Return the amplitude of the wavefront
        Returns:
            mag: magnitude in the polar representation of the complex number 
        """
        return self.get_intensity() ** 0.5
    
    def get_phase(self):
        """
        Return the phase of the wavefront
        Returns:
            ang: angle in the polar representation of the complex number
        """
        x = self.lightX.get_phase()
        y = self.lightY.get_phase()
        return torch.stack((x, y), -1).unsqueeze(-1)
    
    def get_field(self):
        """
        Return the complex wavefront
        Returns:
            field: complex wavefront
        """
        x = self.lightX.field()
        y = self.lightY.field()
        return torch.stack((x, y), -1).unsqueeze(-1)

    def get_intensity(self):
        """
        Return the intensity of light wavefront
        Returns:
            intensity: intensity of light
        """
        return self.lightX.get_intensity() + self.lightY.get_intensity()
        
    def get_bandwidth(self):
        """
        Return the bandwidth of light wavefront
        Returns:
            R_m: spatial height of the wavefront 
            C_m: spatial width of the wavefront 
        """
        return self.pitch*self.dim[2], self.pitch*self.dim[3]
        

    def get_amplitudeX(self):
        """
        Return the amplitude of the wavefront
        Returns:
            mag: magnitude in the polar representation of the complex number 
        """
        return self.lightX.get_amplitude()

    def get_phaseX(self):
        """
        Return the phase of the wavefront
        Returns:
            ang: angle in the polar representation of the complex number
        """
        return self.lightX.get_phase()

    def get_fieldX(self):
        """
        Return the complex wavefront
        Returns:
            field: complex wavefront
        """
        return self.lightX.get_field()

    def get_intensityX(self):
        """
        Return the intensity of light wavefront
        Returns:
            intensity: intensity of light
        """
        return self.lightX.get_intensity()
    
    def get_amplitudeY(self):
        """
        Return the amplitude of the wavefront
        Returns:
            mag: magnitude in the polar representation of the complex number 
        """
        return self.lightY.get_amplitude()

    def get_phaseY(self):
        """
        Return the phase of the wavefront
        Returns:
            ang: angle in the polar representation of the complex number
        """
        return self.lightY.get_phase()

    def get_fieldY(self):
        """
        Return the complex wavefront
        Returns:
            field: complex wavefront
        """
        return self.lightY.get_field()

    def get_intensityY(self):
        """
        Return the intensity of light wavefront
        Returns:
            intensity: intensity of light
        """
        return self.lightY.get_intensity()
    
    def get_lightX(self):
        """
        Return the light of X component
        Returns:
            light: light of X component
        """
        return self.lightX
    
    def get_lightY(self):
        """
        Return the light of Y component
        Returns:
            light: light of Y component
        """
        return self.lightY
    
    def set_lightX(self, light):
        """
        Set the light of X component
        Args:
            light: light of X component
        """
        self.lightX = light
    
    def set_lightY(self, light):
        """
        Set the light of Y component
        Args:
            light: light of Y component
        """
        self.lightY = light
    
    def set_field(self, field):
        """
        Set the field of light
        Args:
            field: field of light with 2 components
        """
        self.lightX.set_field(field[0])
        self.lightY.set_field(field[1])
    
    def magnify(self, scale_factor, interp_mode='nearest'):
        '''
        Change the wavefront resolution without changing the pixel pitch
        Args:
            scale_factor: scale factor for interpolation used in tensor.nn.functional.interpolate
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        '''
        self.lightX.magnify(scale_factor, interp_mode)
        self.lightY.magnify(scale_factor, interp_mode)
        self.dim[2], self.dim[3] = self.lightX.dim[2], self.lightY.dim[3]

    def resize(self, target_pitch, interp_mode='nearest'):
        '''
        Resize the wavefront by changing the pixel pitch. 
        Args:
            target_pitch: new pixel pitch to use
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        '''
        scale_factor = self.pitch / target_pitch
        self.magnify(scale_factor, interp_mode)
        self.set_pitch(target_pitch)

    def set_spherical_light(self, z, dx=0, dy=0):
        '''
        Set the wavefront as spherical one coming from the position of (dx,dy,z). 
        Args:
            z: z distance of the spherical light source from the current light position
            dx: x distance of the spherical light source from the current light position
            dy: y distance of the spherical light source from the current light position
        '''
        self.lightX.set_spherical_light(z, dx, dy)
        self.lightY.set_spherical_light(z, dx, dy)

    def set_plane_light(self):
        '''
        Set the wavefront as a plane wave with zero phase and amptliude of one
        '''
        self.lightX.set_plane_light()
        self.lightY.set_plane_light()

    def set_incident_plane_light(self, theta):
        '''
        Set the wavefront as spherical one coming from the position of (dx,dy,z). 
        Args:
            theta: incident angle     
        '''
        R, C = self.dim[2], self.dim[3]
        amplitude = torch.ones((1, 1, R, C), device=self.device)
        phase = torch.zeros((1, 1, R, C), device=self.device)
        #self.set_field(amplitude*torch.exp(phase*1j))

        dx = torch.linspace(-C/2, C/2, C).to(self.device) * self.pitch
        dy = torch.linspace(-R/2, R/2, R).to(self.device) * self.pitch

        dx, dy = torch.meshgrid(dx, dy, indexing='xy')

        term = -2*torch.pi * dx * np.sin(np.deg2rad(theta)) / self.wvl 
        phase_change = phase + term

        self.set_field(amplitude*torch.exp(phase_change*1j))
        
    def visualize(self,b=0,c=0):
        """
        Visualize the light wave 
        Args:
            b: batch index to visualize default is 0
            c: channel index to visualize. default is 0
        """
        bw = self.get_bandwidth()
        std = 3
        
        plt.figure(figsize=(15, 11))
        plt.subplot(331)
        amplitude_x = self.get_amplitudeX().data.cpu()[b,c,...].squeeze()
        plt.imshow(amplitude_x, extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno', 
                   vmin=0, vmax=amplitude_x.mean() + amplitude_x.std() * std)
        plt.title('amplitude X')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(332)
        phase_x = self.get_phaseX().data.cpu()[b,c,...].squeeze()
        plt.imshow(phase_x.squeeze(),
                   extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='hsv', vmin=-np.pi, vmax=np.pi)  # cyclic colormap
        plt.title('phase X')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(333)
        intensity_x = self.get_intensityX().data.cpu()[b,c,...].squeeze()
        plt.imshow(intensity_x, extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=intensity_x.mean() + intensity_x.std() * std)
        plt.title('intensity X')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()
        
        plt.subplot(334)
        amplitude_y = self.get_amplitudeY().data.cpu()[b,c,...].squeeze()
        plt.imshow(amplitude_y, extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=amplitude_y.mean() + amplitude_y.std() * std)
        plt.title('amplitude Y')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(335)
        phase_y = self.get_phaseY().data.cpu()[b,c,...].squeeze()
        plt.imshow(phase_y.squeeze(),
                   extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='hsv', vmin=-np.pi, vmax=np.pi)  # cyclic colormap
        plt.title('phase Y')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(336)
        intensity_y = self.get_intensityY().data.cpu()[b,c,...].squeeze()
        plt.imshow(intensity_y, extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=intensity_y.mean() + intensity_y.std() * std)
        plt.title('intensity Y')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()
        
        plt.subplot(337)
        amplitude = self.get_amplitude().data.cpu()[b,c,...].squeeze()
        plt.imshow(amplitude, extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=amplitude.mean() + amplitude.std() * std)
        plt.title('amplitude')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(338)
        ratio = amplitude_x / amplitude_y
        plt.imshow(ratio.squeeze(),
                   extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='gray',
                   vmin=ratio.mean()-ratio.std()*5, vmax=ratio.mean()+ratio.std()*5)  # cyclic colormap
        plt.title('Ratio (X / Y)')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(339)
        intensity = self.get_intensity().data.cpu()[b,c,...].squeeze()
        plt.imshow(intensity, extent=[0,bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=intensity.mean() + intensity.std() * std)
        plt.title('intensity')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()
        
        plt.suptitle('(%d,%d), pitch:%.2f[um], wvl:%.2f[nm], device:%s'%(self.dim[2], self.dim[3],
                                                                         self.pitch/1e-6, self.wvl/1e-9, self.device))
        plt.show()

    def shape(self):
        """
        Returns the shape of light wavefront
        Returns:
            shape
        """
        return self.lightX.get_field().shape()
