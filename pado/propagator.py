import torch
import numpy as np
from .math import fft, ifft, conv_fft, wrap_phase, sc_dft_2d, sc_idft_2d, compute_scasm_transfer_function
import torch.nn.functional as Func
from .light import Light


def compute_pad_width(field, linear):
    """
    Compute the pad width of an array for FFT-based convolution
    Args:
        field: (B,Ch,R,C) complex tensor
        linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
    Returns:
        pad_width: pad-width tensor
    """

    if linear:
        R,C = field.shape[-2:]
        pad_width = (C//2, C//2, R//2, R//2)
    else:
        pad_width = (0,0,0,0)
    return pad_width 

def unpad(field_padded, pad_width):
    """
    Unpad the already-padded complex tensor 
    Args:
        field_padded: (B,Ch,R,C) padded complex tensor 
        pad_width: pad-width tensor
    Returns:
        field: unpadded complex tensor
    """

    field = field_padded[...,pad_width[2]:-pad_width[3],pad_width[0]:-pad_width[1]]
    return field

class Propagator:
    def __init__(self, mode, polar='non'):
        """
        Free-space propagator of light waves
        One can simulate the propagation of light waves on free space (no medium change at all).
        Args:
            mode: type of propagator. Currently, we support "Fraunhofer", "Fresnel", "ASM" propagation methods.
        """
        self.mode = mode
        self.polar = polar

    def forward(self, light, z, offset=None, variable_offset_indices=None, target_plane_sample_trajectory=None, linear=True, scale=None, target_plane=None, sampling_ratio=1, steps=100, b=None):
        """
        Forward the incident light with the propagator. 
        Args:
            light: incident light 
            z: propagation distance in meter
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
            offset: offset (y, x) between source and target plane in meters for off-axis propagation
            variable_offsets: list of offsets (y, x) that only for that offsets are differentiable and others are considered constant (detached from the computation graph).
            target_plane_sample_trajectory: list of (idx_y, idx_x). offset will be (idx_y*R*pitch, idx_x*C*pitch).
                                            used for propagators that use multiple ASMs (SBL_ASM_sparse, SBL_ASM_intensity_sum)
            scale: scale (scale_y, scale_x) only used for forward_SBL_ASM_sparse 
            target_plane: cartesian coordinate of target plane for Ryaleigh-sommerfeld calculation
            steps: Used in forward_RayleighSommerfeld_vectorized. 
                How many steps will the total computation be dividied into. If memory limitation issue happens, increase
                the number of steps to reduce the memory consumption per each computation step.
        Returns:
            light: light after propagation
        """
        
        if z == 0:
            # No propagation
            return light
    
        if self.polar=='non':
            return self.forwardNonPolar(light, z, offset, variable_offset_indices, target_plane_sample_trajectory, linear, scale, target_plane, sampling_ratio, steps, b)
        elif self.polar=='polar':
            x = self.forwardNonPolar(light.get_lightX(), z, linear)
            y = self.forwardNonPolar(light.get_lightY(), z, linear)
            light.set_lightX(x)
            light.set_lightY(y)
            return light
        else:
            raise NotImplementedError('Polar is not set.')

    def forwardNonPolar(self, light, z, offset=None, variable_offset_indices=None, target_plane_sample_trajectory=None, linear=True, scale=None, target_plane=None, sampling_ratio=1, steps=100, b=None):
        """
        Forward the incident light with the propagator. 
        Args:
            light: incident light 
            z: propagation distance in meter
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
            offset: offset (y, x) between source and target plane in meters for off-axis propagation
            target_plane: cartesian coordinate of target plane for Ryaleigh-sommerfeld calculation
            steps: Used in forward_RayleighSommerfeld_vectorized. 
                How many steps will the total computation be dividied into. If memory limitation issue happens, increase
                the number of steps to reduce the memory consumption per each computation step.
        Returns:
            light: light after propagation
        """

        if self.mode == 'Fraunhofer':
            return self.forward_Fraunhofer(light, z, linear)
        if self.mode == 'Fraunhofer_SW':
            return self.forward_Fraunhofer_SW(light, z, linear)
        if self.mode == 'FFT':
            return self.forward_FFT(light, z)
        if self.mode == 'Fresnel':
            return self.forward_Fresnel(light, z, linear)
        if self.mode == 'ASM':
            return self.forward_ASM(light, z, offset, linear)
        if self.mode == 'pq_ASM':
            return self.forward_periodic_qudrant_ASM(light, z, offset, linear)
        if self.mode == 'SBL_ASM':
            return self.forward_shifted_BL_ASM(light, z, offset, linear)
        if self.mode == 'BL_ASM':
            return self.forward_BL_ASM(light, z, linear)
        if self.mode == 'SBL_ASM_sparse':
            return self.forward_SBL_ASM_sparse(light, z, offset, linear, scale)
        if self.mode == 'SBL_ASM_sparse_constant_variable':
            return self.forward_SBL_ASM_sparse_constant_variable(light, z, offset, variable_offset_indices, linear, scale)
        if self.mode == 'SBL_ASM_intensity_sum':
            return self.forward_SBL_ASM_intensity_sum(light, z, target_plane_sample_trajectory, offset, linear)
        if self.mode == 'Sc_ASM':
            if b is None:
                raise ValueError('b is not given')
            return (self.forward_ScASM(light, z, b) if b > 1
                    else self.forward_ASM(light, z, offset, linear) if b == 1
                    else self.forward_ScASM_focusing(light, z, b))
        if self.mode == 'RS':
            return self.forward_RayleighSommerfeld(light, z, target_plane, sampling_ratio)
        if self.mode == 'RS_vector':
            return self.forward_RayleighSommerfeld_vectorized(light, z, target_plane, steps)
        if self.mode == 'RS_dh':
                return self.forward_RayleighSommerfeld_dh(light, z, linear=linear, target_plane=target_plane, sampling_ratio=sampling_ratio)
        else:
            raise NotImplementedError('%s propagator is not implemented'%self.mode)


    def forward_Fraunhofer_ori(self, light, z, linear=True):
        """
        Forward the incident light with the Fraunhofer propagator for multiple wavelength channels.
        Args:
            light: incident light 
            z: propagation distance in meter.
                The propagated wavefront is independent w.r.t. the travel distance z.
                The distance z only affects the size of the "pixel", effectively adjusting the entire image size.
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """

        light_propagated = light.clone()  # Assuming this method properly duplicates the Light object
        
        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * light.field.shape[1]  # Replicate the wavelength for each channel

        for chan in range(light.field.shape[1]):
            pad_width = compute_pad_width(light.field[:, chan, :, :], linear)
            field_input_ch = light.field[:, chan, :, :]
            field_propagated_ch = fft(field_input_ch, pad_width=pad_width)
            field_propagated_ch = unpad(field_propagated_ch, pad_width)

            # Adjust the computation of pitch after propagation per channel
            wvl = wavelengths[chan]
            bw_r, bw_c = light.get_bandwidth()
            pitch_r_after_propagation = wvl * z / bw_r
            pitch_c_after_propagation = wvl * z / bw_c

            if pitch_r_after_propagation >= pitch_c_after_propagation:
                scale_c = 1
                scale_r = int(pitch_r_after_propagation / pitch_c_after_propagation)
                pitch_after_propagation = pitch_c_after_propagation
            else:
                scale_r = 1
                scale_c = int(pitch_c_after_propagation / pitch_r_after_propagation)
                pitch_after_propagation = pitch_r_after_propagation

            # Set the propagated field for the current channel
            light_propagated.set_field(field_propagated_ch, c=chan)
            light_propagated.magnify((scale_r, scale_c))
            light_propagated.set_pitch(pitch_after_propagation)

        return light_propagated



    def forward_Fraunhofer(self, light, z, linear=True):
        """
        Forward the incident light with the Fraunhofer propagator for multiple wavelength channels.
        Args:
            light: incident light 
            z: propagation distance in meter.
                The propagated wavefront is independent w.r.t. the travel distance z.
                The distance z only affects the size of the "pixel", effectively adjusting the entire image size.
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """

        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * light.field.shape[1]  # Replicate the wavelength for each channel
        wavelengths = np.array(wavelengths)

        field_propagated = fft(light.field)

        # If there is only one color channel
        if light.dim[1] == 1:
            wavelengths = wavelengths[0]

        bw_r = light.get_bandwidth()[0]
        bw_c = light.get_bandwidth()[1]
        pitch_r_after_propagation = wavelengths * z / bw_r
        pitch_c_after_propagation = wavelengths * z / bw_c

        light_propagated = light.clone()

        if pitch_r_after_propagation >= pitch_c_after_propagation:
            scale_c = 1
            scale_r = pitch_r_after_propagation / pitch_c_after_propagation
            pitch_after_propagation = pitch_c_after_propagation
        elif pitch_r_after_propagation < pitch_c_after_propagation:
            scale_r = 1
            scale_c = pitch_c_after_propagation / pitch_r_after_propagation
            pitch_after_propagation = pitch_r_after_propagation
        
        scale_r = float(scale_r)
        scale_c = float(scale_c)
        
        light_propagated.set_field(field_propagated)
        light_propagated.magnify((scale_r, scale_c))
        light_propagated.set_pitch(pitch_after_propagation)

        return light_propagated

    def forward_Fraunhofer_SW(self, light, z, linear=True):
        """
        Forward the incident light with the Fraunhofer propagator for multiple wavelength channels.
        Multiplied by pixelwise term
        Args:
            light: incident light 
            z: propagation distance in meter.
                The propagated wavefront is independent w.r.t. the travel distance z.
                The distance z only affects the size of the "pixel", effectively adjusting the entire image size.
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """

        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * light.field.shape[1]  # Replicate the wavelength for each channel

        # If there is only one color channel
        if light.dim[1] == 1:
            wavelengths = wavelengths[0]
        bw_r = light.get_bandwidth()[0]
        bw_c = light.get_bandwidth()[1]
        pitch_r_after_propagation = wavelengths * z / bw_r
        pitch_c_after_propagation = wavelengths * z / bw_c

        k = 2*torch.pi/wavelengths
        X = torch.linspace(-bw_c/2, bw_c/2, light.dim[3], dtype=torch.float32)
        Y = torch.linspace(-bw_r/2, bw_r/2, light.dim[2], dtype=torch.float32)
        XX, YY = torch.meshgrid(X, Y, indexing='ij')
        c = 1./(1j*wavelengths*z)*torch.exp(1j*k*0.5/z*(torch.pow(XX, 2) + torch.pow(YY, 2)))
        c = c.to(light.field.device)
        field_propagated = c*fft(light.field)

        light_propagated = light.clone()

        if pitch_r_after_propagation >= pitch_c_after_propagation:
            scale_c = 1
            scale_r = pitch_r_after_propagation / pitch_c_after_propagation
            pitch_after_propagation = pitch_c_after_propagation
        elif pitch_r_after_propagation < pitch_c_after_propagation:
            scale_r = 1
            scale_c = pitch_c_after_propagation / pitch_r_after_propagation
            pitch_after_propagation = pitch_r_after_propagation
        
        scale_r = float(scale_r)
        scale_c = float(scale_c)
        
        light_propagated.set_field(field_propagated)
        light_propagated.magnify((scale_r, scale_c))
        light_propagated.set_pitch(pitch_after_propagation)

        return light_propagated
    
    def forward_FFT(self, light, z=None):
        """
        Forward the incident light using simple FFT-based propagation.
        This method applies exp(1j * phase) before FFT and doesn't consider the distance z or padding.
        
        Args:
            light: incident light 
            z: propagation distance in meter (not used in this method)
        
        Returns:
            light: light after propagation
        """
        field_input = light.field
        light_propagated = light.clone()

        for chan in range(field_input.shape[1]):
            field_input_ch = field_input[:, chan, :, :]

            # Apply exp(1j * phase)
            phase = field_input_ch.angle()
            field_exp = torch.exp(1j * phase)

            # Perform forward FFT
            field_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field_exp)))

            # Set the propagated field for the current channel
            light_propagated.set_field(field_fft, c=chan)

        return light_propagated

    def forward_Fresnel(self, light, z, linear):
        field_input = light.field
        light_propagated = light.clone()  # Assuming this method properly duplicates the Light object

        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]  # Replicate the wavelength for each channel

        for chan in range(field_input.shape[1]):
            field_input_ch = field_input[:, chan, :, :]
            pad_width = compute_pad_width(field_input_ch, linear) 

            # Adjust spatial domain calculations based on 'linear'
            if linear:
                # 2x zero padding for the transfer function
                sx = light.dim[3]
                sy = light.dim[2]
                x = torch.arange(-sx, sx, 1, device=light.device)
                y = torch.arange(-sy, sy, 1, device=light.device)
            else:
                sx = light.dim[3] / 2
                sy = light.dim[2] / 2
                x = torch.arange(-sx, sx, 1, device=light.device)
                y = torch.arange(-sy, sy, 1, device=light.device)

            xx, yy = torch.meshgrid(x,y)
            xx = (xx*light.pitch).to(light.device)
            yy = (yy*light.pitch).to(light.device)
            wvl = wavelengths[chan]

            k = 2*np.pi/wvl # wavenumber
            phase_u = (k*(xx**2 + yy**2)/(2*z))
            phase_u = phase_u.unsqueeze(0).unsqueeze(0)
            phase_w = wrap_phase(phase_u, stay_positive=False)
            amplitude = torch.ones_like(phase_w) / z / wvl
            conv_kernel = amplitude * torch.exp(phase_w*1j)
            conv_kernel /= conv_kernel.abs().sum()

            H = fft(conv_kernel)
            F = fft(field_input_ch, pad_width=pad_width)
            G = F * H
            field_propagated = ifft(G, pad_width=pad_width)

            # return the propagated light
            light_propagated.set_field(field_propagated, c=chan)

        return light_propagated

    def forward_ASM(self, light, z, offset, linear):
        '''
        Forward the incident light with the angular spectrum method, supporting multi-wavelength channels.
        Args:
            light: incident light object, which contains multi-wavelength channels.
            z: propagation distance in meters.
            offset: offset (y, x) between source and target plane in meters for off-axis propagation
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding).
        Returns:
            light_propagated: light after propagation, with each channel processed according to its wavelength.
        '''
        field_input = light.field
        light_propagated = light.clone()  # Clone the light object for the propagated light

        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]  # Replicate the wavelength for each channel

        for chan in range(field_input.shape[1]):
            field_input_ch = field_input[:, chan, :, :]
            pad_width = compute_pad_width(field_input_ch, linear)

            # Adjust frequency calculations based on 'linear'
            if linear:
                # 2x zero padding for the transfer function
                fx = torch.arange(-light.dim[3], light.dim[3], device=light.device) / (2*light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2], light.dim[2], device=light.device) / (2*light.pitch * light.dim[2])
            else:
                fx = torch.arange(-light.dim[3]//2, light.dim[3]//2, device=light.device) / (light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2]//2, light.dim[2]//2, device=light.device) / (light.pitch * light.dim[2])
            fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')

            wvl = wavelengths[chan]
            k = 2 * torch.pi / wvl
            gamma = torch.sqrt(torch.abs(1. - (wvl*fxx)**2 - (wvl*fyy)**2))
            if offset is not None:
                gamma_offset = wvl*fxx*offset[1] + wvl*fyy*offset[0]
                H = (torch.exp(1j*k*(z*gamma+gamma_offset))).to(light.device)
            else:
                H = (torch.exp(1j*k*z*gamma)).to(light.device)

            F = fft(field_input_ch, pad_width=pad_width)
            G = F * H
            field_propagated = ifft(G, pad_width=pad_width)

            # Set the propagated field for each channel
            light_propagated.set_field(field_propagated, c=chan)

        return light_propagated
    
    def forward_periodic_qudrant_ASM(self, light, z, offset, linear):
        '''
        Forward the incident light with the angular spectrum method, supporting multi-wavelength channels.
        Args:
            light: incident light object, which contains multi-wavelength channels.
            z: propagation distance in meters.
            offset: In this case, offset is a bandwidth of one period
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding).
        Returns:
            light_propagated: light after propagation, with each channel processed according to its wavelength.
        '''       

        def compute_uv_filter(offset_x, offset_y):
            fx_max = np.sin(np.arctan((offset_x + light.dim[-1]*light.pitch) / abs(z))) / wvl
            fx_min = np.sin(np.arctan((offset_x - light.dim[-1]*light.pitch) / abs(z))) / wvl
            fy_max = np.sin(np.arctan((offset_y + light.dim[-2]*light.pitch) / abs(z))) / wvl
            fy_min = np.sin(np.arctan((offset_y - light.dim[-2]*light.pitch) / abs(z))) / wvl
            uv_filter = (fx_min < fxx) & (fx_max > fxx) & (fy_min < fyy) & (fy_max > fyy)
            return uv_filter

        field_input = light.field
        light_propagated = light.clone()  # Clone the light object for the propagated light

        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]  # Replicate the wavelength for each channel

        for chan in range(field_input.shape[1]):
            field_input_ch = field_input[:, chan, :, :]
            pad_width = compute_pad_width(field_input_ch, linear)

            # Adjust frequency calculations based on 'linear'
            if linear:
                # 2x zero padding for the transfer function
                fx = torch.arange(-light.dim[3], light.dim[3], device=light.device) / (2*light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2], light.dim[2], device=light.device) / (2*light.pitch * light.dim[2])
            else:
                fx = torch.arange(-light.dim[3]//2, light.dim[3]//2, device=light.device) / (light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2]//2, light.dim[2]//2, device=light.device) / (light.pitch * light.dim[2])
            fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')

            wvl = wavelengths[chan]
            k = 2 * torch.pi / wvl
            gamma = torch.sqrt(torch.abs(1. - (wvl*fxx)**2 - (wvl*fyy)**2))

            H = (torch.exp(1j*k*(z*gamma))).to(light.device)

            F0 = fft(field_input_ch, pad_width=pad_width)
            F1 = fft(torch.flip(field_input_ch, dims=[-1]), pad_width=pad_width)*torch.exp(1j*torch.pi*2*(fxx*offset)*compute_uv_filter(offset_x=offset, offset_y=0)).to(light.device)
            F2 = fft(torch.flip(field_input_ch, dims=[-2]), pad_width=pad_width)*torch.exp(1j*torch.pi*2*(fyy*-offset)*compute_uv_filter(offset_x=0, offset_y=-offset)).to(light.device) 
            F3 = fft(torch.flip(field_input_ch, dims=[-2, -1]), pad_width=pad_width)*torch.exp(1j*torch.pi*2*(fxx*offset+fyy*-offset)*compute_uv_filter(offset_x=offset, offset_y=-offset)).to(light.device)

            G = H * (F0+F1+F2+F3)
            field_propagated = ifft(G, pad_width=pad_width)

            # Set the propagated field for each channel
            light_propagated.set_field(field_propagated, c=chan)

        return light_propagated

    def forward_BL_ASM(self, light, z, linear):
        '''
        Forward the incident light with the bandlimited angular spectrum method,
        supporting multi-wavelength channels.

        Args:
            light: incident light 
            z: propagation distance in meter. 
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        '''
        field_input = light.field

        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]  # Replicate the wavelength for each channel

        light_propagated = light.clone()  # Assuming this method properly duplicates the Light object

        for chan in range(field_input.shape[1]):
            field_input_ch = field_input[:, chan, :, :]
            pad_width = compute_pad_width(field_input_ch, linear)

            # Adjust frequency calculations based on 'linear'
            if linear:
                # 2x zero padding for the transfer function
                fx = torch.arange(-light.dim[3], light.dim[3], device=light.device) / (2*light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2], light.dim[2], device=light.device) / (2*light.pitch * light.dim[2])
            else:
                fx = torch.arange(-light.dim[3]//2, light.dim[3]//2, device=light.device) / (light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2]//2, light.dim[2]//2, device=light.device) / (light.pitch * light.dim[2])

            fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')
            wvl = wavelengths[chan]
            k = 2*torch.pi / wvl
            gamma = torch.sqrt(torch.abs(1. - (wvl*fxx)**2 - (wvl*fyy)**2))

            Ideal_angle_limit = light.get_ideal_angle_limit()
            """
            use the following diffraction angle and spatial frequency relationship: sin(theta_x)/wvl = f_x
            Below theta_limit is derived from geometry. When Bandwidth = 1/2/pitch, freq limit from Nyquist limit and geometry is same, which is the case
            when we use linear convolution to avoid circular convolution.
            Refer to Matsushima and Shimobaba, "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields"
            """
            theta_limit = np.arctan(light.dim[3]*light.pitch/abs(z)) * 180 / np.pi

            # When theta_limit >= Ideal_angle_limit, freq domain of input field is smaller than filter size, so don't need to filter
            if theta_limit < Ideal_angle_limit:
                print('BL_ASM: theta_limit is smaller than Ideal_angle_limit. Activating the frequency filter!')
                uv_filter = (torch.abs(fxx) < np.sin(np.deg2rad(theta_limit)) / wvl) & \
                            (torch.abs(fyy) < np.sin(np.deg2rad(theta_limit)) / wvl)
                H = (torch.exp(1j*k*z*gamma) * uv_filter).to(light.device)
            else:
                H = torch.exp(1j*k*z*gamma).to(light.device)

            F = fft(field_input_ch, pad_width=pad_width)
            G = F * H
            field_propagated = ifft(G, pad_width=pad_width)

            # Set the propagated field for the current channel
            light_propagated.set_field(field_propagated, c=chan)

        return light_propagated
    
    def forward_shifted_BL_ASM(self, light, z, offset, linear):
        '''
        Forward the incident light with the angular spectrum method, supporting multi-wavelength channels.
        Args:
            light: incident light object, which contains multi-wavelength channels.
            z: propagation distance in meters.
            offset: offset (y, x) between source and target plane in meters for off-axis propagation
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding).
        Returns:
            light_propagated: light after propagation, with each channel processed according to its wavelength.
        '''
        field_input = light.field
        light_propagated = light.clone()  # Clone the light object for the propagated light

        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]  # Replicate the wavelength for each channel

        for chan in range(field_input.shape[1]):
            field_input_ch = field_input[:, chan, :, :]
            pad_width = compute_pad_width(field_input_ch, linear)

            # Adjust frequency calculations based on 'linear'
            if linear:
                # 2x zero padding for the transfer function
                fx = torch.arange(-light.dim[3], light.dim[3], device=light.device) / (2*light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2], light.dim[2], device=light.device) / (2*light.pitch * light.dim[2])
            else:
                fx = torch.arange(-light.dim[3]//2, light.dim[3]//2, device=light.device) / (light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2]//2, light.dim[2]//2, device=light.device) / (light.pitch * light.dim[2])
            fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')

            wvl = wavelengths[chan]
            k = 2 * torch.pi / wvl
            # debugging
            # gamma = torch.sqrt(torch.abs(1. - (wvl*fxx)**2 - (wvl*fyy)**2))
            # debugging

            # debugging
            gamma = (1. - (wvl*fxx)**2 - (wvl*fyy)**2)
            gamma = gamma * (gamma>=0)
            gamma = torch.sqrt(gamma)
            # debugging

            """
            use the following diffraction angle and spatial frequency relationship: sin(theta_x)/wvl = f_x
            Below theta_limit is derived from geometry. When Bandwidth = 1/2/pitch, freq limit from Nyquist limit and geometry is same, which is the case
            when we use linear convolution to avoid circular convolution.
            Refer to Matsushima, "Shifted angular spectrum method for off-axis numerical propagation"
            """
            # define constants to compute bandlimit
            bw_x = light.dim[-1]*light.pitch
            bw_y = light.dim[-2]*light.pitch
            u_limit_plus = 1/np.sqrt(1+np.power(z/(offset[-1]+bw_x+1e-7), 2))/wvl
            u_limit_minus = 1/np.sqrt(1+np.power(z/(offset[-1]-bw_x+1e-7), 2))/wvl 
            v_limit_plus = 1/np.sqrt(1+np.power(z/(offset[-2]+bw_y+1e-7), 2))/wvl 
            v_limit_minus = 1/np.sqrt(1+np.power(z/(offset[-2]-bw_y+1e-7), 2))/wvl  
            # fxx bound
            if offset[-1] > bw_x:
                fxx_upper_bound = fxx <= torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_plus,2))
                fxx_lower_bound = fxx >= torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_minus,2))
            elif offset[-1] <= bw_x and offset[-1] >= -bw_x:
                fxx_upper_bound = fxx <= torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_plus,2)) - (np.sin(np.radians(0.1))/wvl if offset[-1]==-bw_x else 0)
                fxx_lower_bound = fxx >= -torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_minus,2)) + (np.sin(np.radians(0.1))/wvl if offset[-1]==bw_x else 0)
            else:
                fxx_upper_bound = fxx <= -torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_plus,2))
                fxx_lower_bound = fxx >= -torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_minus,2))
            # fyy bound
            if offset[-2] > bw_y:
                fyy_upper_bound = fyy <= torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_plus,2))
                fyy_lower_bound = fyy >= torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_minus,2))
            elif offset[-2] <= bw_y and offset[-2] >= -bw_y:
                fyy_upper_bound = fyy <= torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_plus,2)) - (np.sin(np.radians(0.1))/wvl if offset[-2]==-bw_y else 0)
                fyy_lower_bound = fyy >= -torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_minus,2)) + (np.sin(np.radians(0.1))/wvl if offset[-2]==bw_y else 0)
            else:
                fyy_upper_bound = fyy <= -torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_plus,2))
                fyy_lower_bound = fyy >= -torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_minus,2))
            uv_filter =  fxx_upper_bound & fxx_lower_bound & fyy_upper_bound & fyy_lower_bound

            gamma_offset = wvl*fxx*offset[1] + wvl*fyy*offset[0]
            H = (torch.exp(1j*k*(z*gamma+gamma_offset)) * uv_filter).to(light.device)
            F = fft(field_input_ch, pad_width=pad_width)
            G = F * H
            field_propagated = ifft(G, pad_width=pad_width)

            # Set the propagated field for each channel
            light_propagated.set_field(field_propagated, c=chan)

        return light_propagated
    
    def forward_SBL_ASM_sparse(self, light, z, offset, linear, scale=(1, 1)):
        '''
        Forward the incident light with the shifted angular spectrum method for multiple times for sparse larger target plane
        supporting multi-wavelength channels. If input field shape is (1, 1, R, C), the output field shape is (1, 1, R, 5*C)
        This function is based on forward_shifted_BL_ASM.
        Args:
            light: incident light object, which contains multi-wavelength channels.
            z: propagation distance in meters.
            offset: global offset (y, x). (m)
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding).
            scale: tuple (y_scale, x_scale). The indicator of the target plane size. The target plane size will be 
                   (light.shape[0]*scale[0], light.shape[1]*scale[1])
        Returns:
            light_propagated: light after propagation, with each channel processed according to its wavelength.
        '''

        if scale is None:
            scale = (1, 1)

        def compute_uv_filter(offset_x, offset_y):
            # define constants to compute bandlimit
            bw_x = light.dim[-1]*light.pitch
            bw_y = light.dim[-2]*light.pitch
            u_limit_plus = 1/np.sqrt(1+np.power(z/(offset_x+bw_x+1e-7), 2))/wvl
            u_limit_minus = 1/np.sqrt(1+np.power(z/(offset_x-bw_x+1e-7), 2))/wvl 
            v_limit_plus = 1/np.sqrt(1+np.power(z/(offset_y+bw_y+1e-7), 2))/wvl 
            v_limit_minus = 1/np.sqrt(1+np.power(z/(offset_y-bw_y+1e-7), 2))/wvl  
            # fxx bound
            if offset_x > bw_x:
                fxx_upper_bound = fxx <= torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_plus,2))
                fxx_lower_bound = fxx >= torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_minus,2))
            elif offset_x <= bw_x and offset_x >= -bw_x:
                fxx_upper_bound = fxx <= torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_plus,2)) - (np.sin(np.radians(0.1))/wvl if offset_x==-bw_x else 0)
                fxx_lower_bound = fxx >= -torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_minus,2)) + (np.sin(np.radians(0.1))/wvl if offset_x==bw_x else 0)
            else:
                fxx_upper_bound = fxx <= -torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_plus,2))
                fxx_lower_bound = fxx >= -torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_minus,2))
            # fyy bound
            if offset_y > bw_y:
                fyy_upper_bound = fyy <= torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_plus,2))
                fyy_lower_bound = fyy >= torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_minus,2))
            elif offset_y <= bw_y and offset_y >= -bw_y:
                fyy_upper_bound = fyy <= torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_plus,2)) - (np.sin(np.radians(0.1))/wvl if offset_y==-bw_y else 0)
                fyy_lower_bound = fyy >= -torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_minus,2)) + (np.sin(np.radians(0.1))/wvl if offset_y==bw_y else 0)
            else:
                fyy_upper_bound = fyy <= -torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_plus,2))
                fyy_lower_bound = fyy >= -torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_minus,2))
            uv_filter =  fxx_upper_bound & fxx_lower_bound & fyy_upper_bound & fyy_lower_bound
            return uv_filter       
        
        field_input = light.field
        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]  # Replicate the wavelength for each channel

        R, C = field_input.shape[-2], field_input.shape[-1]
        light_propagated=Light((1, field_input.shape[-3], R*scale[0], C*scale[1]), light.pitch, wavelengths, device=light.device)
        light_propagated_field = torch.zeros((1, field_input.shape[-3], R*scale[0], C*scale[1]), device=light.device, dtype=torch.complex64)

        for chan in range(field_input.shape[-3]):
            field_input_ch = field_input[:, chan, :, :]
            pad_width = compute_pad_width(field_input_ch, linear)

            # Adjust frequency calculations based on 'linear'
            if linear:
                # 2x zero padding for the transfer function
                fx = torch.arange(-light.dim[3], light.dim[3], device=light.device) / (2*light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2], light.dim[2], device=light.device) / (2*light.pitch * light.dim[2])
            else:
                fx = torch.arange(-light.dim[3]//2, light.dim[3]//2, device=light.device) / (light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2]//2, light.dim[2]//2, device=light.device) / (light.pitch * light.dim[2])
            fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')

            wvl = wavelengths[chan]
            k = 2 * torch.pi / wvl
            gamma = torch.sqrt(torch.abs(1. - (wvl*fxx)**2 - (wvl*fyy)**2))

            """
            use the following diffraction angle and spatial frequency relationship: sin(theta_x)/wvl = f_x
            Below theta_limit is derived from geometry. When Bandwidth = 1/2/pitch, freq limit from Nyquist limit and geometry is same, which is the case
            when we use linear convolution to avoid circular convolution.
            Refer to Matsushima, "Shifted angular spectrum method for off-axis numerical propagation"
            """
            F = fft(field_input_ch, pad_width=pad_width)

            offset_list_y, offset_list_x = np.meshgrid(np.linspace(-(scale[0]-1)/2*R*light.pitch, (scale[0]-1)/2*R*light.pitch, scale[0]), 
                                                       np.linspace(-(scale[1]-1)/2*C*light.pitch, (scale[1]-1)/2*C*light.pitch, scale[1]), indexing='ij')
            idx_list_y, idx_list_x = np.meshgrid(np.arange(0, scale[0], 1), np.arange(0, scale[1], 1), indexing='ij')

            for (offset_y, offset_x), (idx_y, idx_x) in zip(zip(offset_list_y.ravel(), offset_list_x.ravel()), zip(idx_list_y.ravel(), idx_list_x.ravel())):
                gamma_offset = wvl*fxx*(offset_x+offset[1]) + wvl*fyy*(offset_y+offset[0])
                H = (torch.exp(1j*k*(z*gamma+gamma_offset)) * compute_uv_filter(offset_x+offset[1], offset_y+offset[0])).to(light.device)
                G = F * H
                light_propagated_field.clone()
                light_propagated_field[:, :, idx_y*R:(idx_y+1)*R, idx_x*C:(idx_x+1)*C] = ifft(G, pad_width=pad_width)
            # Set the propagated field for each channel
            light_propagated.set_field(light_propagated_field)

        return light_propagated

    def forward_SBL_ASM_intensity_sum(self, light, z, offset_indicies, offset, linear):
        '''
        Forward the incident light with the shifted angular spectrum method for multiple times and compute
        the sum of the intensities for the whole propagation. The main purpose of this function is to compute
        the sum of intensities for loss and remove lights from unnecessary target planes.
        Args:
            light: incident light object, which contains multi-wavelength channels.
            z: propagation distance in meters.
            offset_indicies: list of offset index (y, x) between source and target plane for off-axis propagation. 
                     The real offset value will be in meter and computed by (y*light.pitch*R, x*light.pitch*C). 
                     For these offset values, the target plane will be computed and will have values. 
                     For other parts of the target plane will have zero values
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding).
        Returns:
            light_propagated: light after propagation, with each channel processed according to its wavelength.
        '''
        if offset_indicies is None:
            offset_indicies = [(0, 0)]
    
    def forward_SBL_ASM_sparse_constant_variable(self, light, z, offset, variable_offset_indices, linear, scale=(1, 1)):
        '''
        Forward the incident light with the shifted angular spectrum method for multiple times for sparse larger target plane
        supporting multi-wavelength channels. If input field shape is (1, 1, R, C), the output field shape is (1, 1, R, 5*C)
        This function is based on forward_shifted_BL_ASM.
        Args:
            light: incident light object, which contains multi-wavelength channels.
            z: propagation distance in meters.
            offset: global offset (y, x). (m)
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding).
            scale: tuple (y_scale, x_scale). The indicator of the target plane size. The target plane size will be 
                   (light.shape[0]*scale[0], light.shape[1]*scale[1])
        Returns:
            light_propagated: light after propagation, with each channel processed according to its wavelength.
        '''

        if scale is None:
            scale = (1, 1)

        def compute_uv_filter(offset_x, offset_y):
            # define constants to compute bandlimit
            bw_x = light.dim[-1]*light.pitch
            bw_y = light.dim[-2]*light.pitch
            u_limit_plus = 1/np.sqrt(1+np.power(z/(offset_x+bw_x+1e-7), 2))/wvl
            u_limit_minus = 1/np.sqrt(1+np.power(z/(offset_x-bw_x+1e-7), 2))/wvl 
            v_limit_plus = 1/np.sqrt(1+np.power(z/(offset_y+bw_y+1e-7), 2))/wvl 
            v_limit_minus = 1/np.sqrt(1+np.power(z/(offset_y-bw_y+1e-7), 2))/wvl  
            # fxx bound
            if offset_x > bw_x:
                fxx_upper_bound = fxx <= torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_plus,2))
                fxx_lower_bound = fxx >= torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_minus,2))
            elif offset_x <= bw_x and offset_x >= -bw_x:
                fxx_upper_bound = fxx <= torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_plus,2)) - (np.sin(np.radians(0.1))/wvl if offset_x==-bw_x else 0)
                fxx_lower_bound = fxx >= -torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_minus,2)) + (np.sin(np.radians(0.1))/wvl if offset_x==bw_x else 0)
            else:
                fxx_upper_bound = fxx <= -torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_plus,2))
                fxx_lower_bound = fxx >= -torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_minus,2))
            # fyy bound
            if offset_y > bw_y:
                fyy_upper_bound = fyy <= torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_plus,2))
                fyy_lower_bound = fyy >= torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_minus,2))
            elif offset_y <= bw_y and offset_y >= -bw_y:
                fyy_upper_bound = fyy <= torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_plus,2)) - (np.sin(np.radians(0.1))/wvl if offset_y==-bw_y else 0)
                fyy_lower_bound = fyy >= -torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_minus,2)) + (np.sin(np.radians(0.1))/wvl if offset_y==bw_y else 0)
            else:
                fyy_upper_bound = fyy <= -torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_plus,2))
                fyy_lower_bound = fyy >= -torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_minus,2))
            uv_filter =  fxx_upper_bound & fxx_lower_bound & fyy_upper_bound & fyy_lower_bound
            return uv_filter       
        
        field_input = light.field
        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]  # Replicate the wavelength for each channel

        R, C = field_input.shape[-2], field_input.shape[-1]
        light_propagated=Light((1, field_input.shape[-3], R*scale[0], C*scale[1]), light.pitch, wavelengths, device=light.device)
        light_propagated_field = torch.zeros((1, field_input.shape[-3], R*scale[0], C*scale[1]), device=light.device, dtype=torch.complex64)

        for chan in range(field_input.shape[-3]):
            field_input_ch = field_input[:, chan, :, :]
            pad_width = compute_pad_width(field_input_ch, linear)

            # Adjust frequency calculations based on 'linear'
            if linear:
                # 2x zero padding for the transfer function
                fx = torch.arange(-light.dim[3], light.dim[3], device=light.device) / (2*light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2], light.dim[2], device=light.device) / (2*light.pitch * light.dim[2])
            else:
                fx = torch.arange(-light.dim[3]//2, light.dim[3]//2, device=light.device) / (light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2]//2, light.dim[2]//2, device=light.device) / (light.pitch * light.dim[2])
            fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')

            wvl = wavelengths[chan]
            k = 2 * torch.pi / wvl
            gamma = torch.sqrt(torch.abs(1. - (wvl*fxx)**2 - (wvl*fyy)**2))

            """
            use the following diffraction angle and spatial frequency relationship: sin(theta_x)/wvl = f_x
            Below theta_limit is derived from geometry. When Bandwidth = 1/2/pitch, freq limit from Nyquist limit and geometry is same, which is the case
            when we use linear convolution to avoid circular convolution.
            Refer to Matsushima, "Shifted angular spectrum method for off-axis numerical propagation"
            """
            offset_list_y, offset_list_x = np.meshgrid(np.linspace(-(scale[0]-1)/2*R*light.pitch, (scale[0]-1)/2*R*light.pitch, scale[0]), 
                                                       np.linspace(-(scale[1]-1)/2*C*light.pitch, (scale[1]-1)/2*C*light.pitch, scale[1]), indexing='ij')
            idx_list_y, idx_list_x = np.meshgrid(np.arange(0, scale[0], 1), np.arange(0, scale[1], 1), indexing='ij')

            with torch.no_grad():
                F = fft(field_input_ch, pad_width=pad_width)
                for (offset_y, offset_x), (idx_y, idx_x) in zip(zip(offset_list_y.ravel(), offset_list_x.ravel()), zip(idx_list_y.ravel(), idx_list_x.ravel())):
                    if (idx_y, idx_x) in variable_offset_indices:
                        continue
                    gamma_offset = wvl*fxx*(offset_x+offset[1]) + wvl*fyy*(offset_y+offset[1])
                    H = (torch.exp(1j*k*(z*gamma+gamma_offset)) * compute_uv_filter(offset_x, offset_y)).to(light.device)
                    G = F * H
                    light_propagated_field[:, :, idx_y*R:(idx_y+1)*R, idx_x*C:(idx_x+1)*C] = ifft(G, pad_width=pad_width)
            del(F)

            F = fft(field_input_ch, pad_width=pad_width)
            for (offset_y, offset_x), (idx_y, idx_x) in zip(zip(offset_list_y.ravel(), offset_list_x.ravel()), zip(idx_list_y.ravel(), idx_list_x.ravel())):
                if (idx_y, idx_x) not in variable_offset_indices:
                    continue
                gamma_offset = wvl*fxx*(offset_x+offset[1]) + wvl*fyy*(offset_y+offset[1])
                H = (torch.exp(1j*k*(z*gamma+gamma_offset)) * compute_uv_filter(offset_x, offset_y)).to(light.device)
                G = F * H
                light_propagated_field.clone()
                light_propagated_field[:, :, idx_y*R:(idx_y+1)*R, idx_x*C:(idx_x+1)*C] = ifft(G, pad_width=pad_width)

            # Set the propagated field for each channel
            light_propagated.set_field(light_propagated_field)

        return light_propagated

    def forward_SBL_ASM_intensity_sum(self, light, z, offset_indicies, offset, linear):
        '''
        Forward the incident light with the shifted angular spectrum method for multiple times and compute
        the sum of the intensities for the whole propagation. The main purpose of this function is to compute
        the sum of intensities for loss and remove lights from unnecessary target planes.
        Args:
            light: incident light object, which contains multi-wavelength channels.
            z: propagation distance in meters.
            offset_indicies: list of offset index (y, x) between source and target plane for off-axis propagation. 
                     The real offset value will be in meter and computed by (y*light.pitch*R, x*light.pitch*C). 
                     For these offset values, the target plane will be computed and will have values. 
                     For other parts of the target plane will have zero values
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding).
        Returns:
            light_propagated: light after propagation, with each channel processed according to its wavelength.
        '''
        if offset_indicies is None:
            offset_indicies = [(0, 0)]

        def compute_uv_filter(offset_x, offset_y):
            # define constants to compute bandlimit
            bw_x = light.dim[-1]*light.pitch
            bw_y = light.dim[-2]*light.pitch
            u_limit_plus = 1/np.sqrt(1+np.power(z/(offset_x+bw_x+1e-7), 2))/wvl
            u_limit_minus = 1/np.sqrt(1+np.power(z/(offset_x-bw_x+1e-7), 2))/wvl 
            v_limit_plus = 1/np.sqrt(1+np.power(z/(offset_y+bw_y+1e-7), 2))/wvl 
            v_limit_minus = 1/np.sqrt(1+np.power(z/(offset_y-bw_y+1e-7), 2))/wvl  
            # fxx bound
            if offset_x > bw_x:
                fxx_upper_bound = fxx <= torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_plus,2))
                fxx_lower_bound = fxx >= torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_minus,2))
            elif offset_x <= bw_x and offset_x >= -bw_x:
                fxx_upper_bound = fxx <= torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_plus,2)) - (np.sin(np.radians(0.1))/wvl if offset_x==-bw_x else 0)
                fxx_lower_bound = fxx >= -torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_minus,2)) + (np.sin(np.radians(0.1))/wvl if offset_x==bw_x else 0)
            else:
                fxx_upper_bound = fxx <= -torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_plus,2))
                fxx_lower_bound = fxx >= -torch.sqrt((1-torch.pow(fyy*wvl,2))*np.power(u_limit_minus,2))
            # fyy bound
            if offset_y > bw_y:
                fyy_upper_bound = fyy <= torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_plus,2))
                fyy_lower_bound = fyy >= torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_minus,2))
            elif offset_y <= bw_y and offset_y >= -bw_y:
                fyy_upper_bound = fyy <= torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_plus,2)) - (np.sin(np.radians(0.1))/wvl if offset_y==-bw_y else 0)
                fyy_lower_bound = fyy >= -torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_minus,2)) + (np.sin(np.radians(0.1))/wvl if offset_y==bw_y else 0)
            else:
                fyy_upper_bound = fyy <= -torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_plus,2))
                fyy_lower_bound = fyy >= -torch.sqrt((1-torch.pow(fxx*wvl,2))*np.power(v_limit_minus,2))
            uv_filter =  fxx_upper_bound & fxx_lower_bound & fyy_upper_bound & fyy_lower_bound
            return uv_filter
        
        field_input = light.field
        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]  # Replicate the wavelength for each channel

        R, C = field_input.shape[-2], field_input.shape[-1]

        for chan in range(field_input.shape[-3]):
            field_input_ch = field_input[:, chan, :, :]
            pad_width = compute_pad_width(field_input_ch, linear)

            # Adjust frequency calculations based on 'linear'
            if linear:
                # 2x zero padding for the transfer function
                fx = torch.arange(-light.dim[3], light.dim[3], device=light.device) / (2*light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2], light.dim[2], device=light.device) / (2*light.pitch * light.dim[2])
            else:
                fx = torch.arange(-light.dim[3]//2, light.dim[3]//2, device=light.device) / (light.pitch * light.dim[3])
                fy = torch.arange(-light.dim[2]//2, light.dim[2]//2, device=light.device) / (light.pitch * light.dim[2])
            fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')

            wvl = wavelengths[chan]
            k = 2 * torch.pi / wvl
            gamma = torch.sqrt(torch.abs(1. - (wvl*fxx)**2 - (wvl*fyy)**2))

            field_list = list()

            F = fft(field_input_ch, pad_width=pad_width)
            for offset_y_index, offset_x_index in offset_indicies:
                offset_x, offset_y = light.pitch * C * offset_x_index + offset[1], light.pitch * R * offset_y_index + offset[0]
                gamma_offset = wvl*fxx*offset_x + wvl*fyy*offset_y
                H = (torch.exp(1j*k*(z*gamma+gamma_offset)) * compute_uv_filter(offset_x, offset_y)).to(light.device)
                G = F * H
                partial_propagated_field = ifft(G, pad_width=pad_width)
                field_list.append(partial_propagated_field)
            intensity_sum = (abs(torch.cat(field_list, axis=-1))**2).sum()
        return intensity_sum    
    
    def forward_ScASM(self, light, z, b, linear=True):
        """
        Propagates the input field using the scaled Angular Spectrum Method (Sc-ASM).

        This function performs a scaled forward angular spectrum propagation. It takes an
        input optical field 'light', propagates it over a distance 'z', and scales the observation
        plane by a factor 'b' relative to the source plane. If 'linear' is True, zero-padding is
        applied to avoid wrap-around effects from FFT-based convolutions.

        Args:
            light: A Light object containing the input field and wavelength information.
            z (float): The propagation distance.
            b (float): Scaling factor for the observation plane size relative to the source plane.
            linear (bool): If True, applies zero-padding for linear convolution.

        Returns:
            A Light object with the propagated field. The spatial sampling pitch of the returned field
            is automatically adjusted according to 'b' and the propagation geometry.
        """
        field_input = light.field
        B, Ch, R, C = field_input.shape
        assert B == 1, "Batch dimension B should be 1 for simplicity."

        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * Ch

        Lsrc = light.pitch * C
        delta_xsrc = Lsrc / C
        delta_ysrc = Lsrc / R
        Lobs = b * Lsrc
        delta_xobs = Lobs / C
        delta_yobs = Lobs / R
        delta_fx = 1 / Lsrc
        delta_fy = 1 / Lsrc

        field_output_all_channels = torch.zeros((B, Ch, R, C), dtype=torch.complex64, device=field_input.device)

        if linear:
            pad_width = (C//2, C//2, R//2, R//2)
            Rp = 2 * R
            Cp = 2 * C
        else:
            pad_width = (0,0,0,0)
            Rp = R
            Cp = C

        for chan in range(Ch):
            λ = wavelengths[chan]
            field_input_ch = field_input[0, chan, :, :].to(torch.complex64)

            if linear:
                field_input_ch = torch.nn.functional.pad(field_input_ch.unsqueeze(0).unsqueeze(0), pad=pad_width)
                field_input_ch = field_input_ch[0,0]

            Lsrc_padded = 2 * Lsrc if linear else Lsrc
            delta_xsrc_p = Lsrc_padded / Cp
            delta_ysrc_p = Lsrc_padded / Rp
            delta_fx_p = 1 / Lsrc_padded
            delta_fy_p = 1 / Lsrc_padded

            U = sc_dft_2d(field_input_ch, Cp, Rp, delta_xsrc_p, delta_ysrc_p, delta_fx_p, delta_fy_p)
            H = compute_scasm_transfer_function(Cp, Rp, delta_fx_p, delta_fy_p, λ, z).to(field_input.device)
            U_prop = U * H
            field_output_ch = sc_idft_2d(U_prop, Cp, Rp, delta_xobs, delta_yobs, delta_fx_p, delta_fy_p)

            if linear:
                field_output_ch = field_output_ch[R//2:R//2+R, C//2:C//2+C]

            field_output_all_channels[0, chan, :, :] = field_output_ch

        light_propagated = light.clone()
        light_propagated.field = field_output_all_channels
        light_propagated.set_pitch(delta_xobs)

        return light_propagated


    def forward_ScASM_farfield(self, light, z, b, linear=True):
        """
        Propagates the input field to the far field (large b) using Sc-ASM.

        This method implements a far-field propagation scenario by scaling the observation plane.
        It applies a scaled DFT, multiplies by the transfer function, and then performs a standard
        IFFT to obtain the far-field distribution. If 'linear' is True, zero-padding is used to
        avoid circular convolution artifacts.

        Args:
            light: A Light object containing the input field and wavelength information.
            z (float): The propagation distance.
            b (float): Scaling factor for the observation plane size (b > 1 for far field).
            linear (bool): If True, applies zero-padding for linear convolution.

        Returns:
            A Light object with the far-field field distribution.
        """
        field_input = light.field
        B, Ch, R, C = field_input.shape
        assert B == 1, "Batch dimension B should be 1 for simplicity."

        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * Ch

        Lsrc = light.pitch * C
        Lobs = b * Lsrc
        delta_xsrc = Lsrc / C
        delta_ysrc = Lsrc / R

        if linear:
            Rp = 2 * R
            Cp = 2 * C
            pad_width = (C//2, C//2, R//2, R//2)
        else:
            Rp = R
            Cp = C
            pad_width = (0,0,0,0)

        Lsrc_padded = 2 * Lsrc if linear else Lsrc
        delta_xsrc_p = Lsrc_padded / Cp
        delta_ysrc_p = Lsrc_padded / Rp
        delta_fx_p = 1 / Lsrc_padded
        delta_fy_p = 1 / Lsrc_padded

        field_output_all_channels = torch.zeros((B, Ch, R, C), dtype=torch.complex64, device=field_input.device)

        for chan in range(Ch):
            λ = wavelengths[chan]
            field_ch = field_input[0, chan, :, :]

            if linear:
                field_ch = torch.nn.functional.pad(field_ch.unsqueeze(0).unsqueeze(0), pad=pad_width)
                field_ch = field_ch[0,0]

            U_scaled = sc_dft_2d(field_ch, Cp, Rp, delta_xsrc_p, delta_ysrc_p, delta_fx_p, delta_fy_p)
            H = compute_scasm_transfer_function(Cp, Rp, delta_fx_p, delta_fy_p, λ, z).to(field_input.device)
            U_prop = U_scaled * H
            U_ifft = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(U_prop)))

            if linear:
                U_ifft = U_ifft[R//2:R//2+R, C//2:C//2+C]

            field_output_all_channels[0, chan, :, :] = U_ifft

        light_farfield = light.clone()
        light_farfield.set_field(field_output_all_channels)
        light_farfield.set_pitch(Lobs / C)
        return light_farfield


    def forward_ScASM_focusing(self, light, z, b, linear=True):
        """
        Propagates the input field to a focusing plane using Sc-ASM.

        This method simulates focusing by propagating the field to a plane closer than the source,
        resulting in a scaled observation plane smaller than the source plane. It first applies a
        standard FFT to get the field in the frequency domain, multiplies by the transfer function,
        and then uses the Sc-IDFT to resample the field at the smaller observation plane. If 'linear'
        is True, zero-padding is applied to ensure a linear convolution scenario.

        Args:
            light: A Light object containing the input field and wavelength information.
            z (float): The propagation distance to the focusing plane.
            b (float): Scaling factor (b < 1 for focusing to a smaller observation plane).
            linear (bool): If True, applies zero-padding to avoid circular convolution.

        Returns:
            A Light object representing the field at the focusing plane.
        """
        field_input = light.field
        B, Ch, R, C = field_input.shape
        assert B == 1, "Batch dimension B should be 1."

        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * Ch

        Lsrc = light.pitch * C
        Lobs = b * Lsrc
        delta_xsrc = Lsrc / C
        delta_ysrc = Lsrc / R

        if linear:
            Rp = 2 * R
            Cp = 2 * C
            pad_width = (C//2, C//2, R//2, R//2)
        else:
            Rp = R
            Cp = C
            pad_width = (0,0,0,0)

        field_output_all_channels = torch.zeros((B, Ch, R, C), dtype=torch.complex64, device=field_input.device)

        for chan in range(Ch):
            λ = wavelengths[chan]
            field_ch = field_input[0, chan, :, :]

            if linear:
                field_ch = torch.nn.functional.pad(field_ch.unsqueeze(0).unsqueeze(0), pad=pad_width)
                field_ch = field_ch[0,0]

            U = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field_ch)))

            Lsrc_padded = 2 * Lsrc if linear else Lsrc
            delta_fx_p = 1 / Lsrc_padded
            delta_fy_p = 1 / Lsrc_padded
            delta_xobs = Lobs / C
            delta_yobs = Lobs / R

            H = compute_scasm_transfer_function(Cp, Rp, delta_fx_p, delta_fy_p, λ, z).to(field_input.device)
            G = U * H
            u_obs = sc_idft_2d(G, Cp, Rp, delta_xobs, delta_yobs, delta_fx_p, delta_fy_p)

            if linear:
                u_obs = u_obs[R//2:R//2+R, C//2:C//2+C]

            field_output_all_channels[0, chan, :, :] = u_obs

        light_focus = light.clone()
        light_focus.set_field(field_output_all_channels)
        light_focus.set_pitch(Lobs / C)

        return light_focus

    
    def forward_RayleighSommerfeld(self, light, z, target_plane=None, sampling_ratio=1):
        # RayleighSommerfeld is not FFT-based calculation, so window size of target_plane is not fixed. 
        
        field_input = light.field
        
        k = 2 * np.pi / light.wvl

        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            k = k.view(1, len(light.wvl), 1, 1)

        R, C = light.dim[2], light.dim[3]
        x = torch.linspace(-C//2, C//2, C, dtype=torch.float64) * light.pitch
        y = torch.linspace(-R//2, R//2, R, dtype=torch.float64) * light.pitch
        xx, yy = torch.meshgrid(x, y)
        xx = xx.to(light.device)
        yy = yy.to(light.device)
        
        if target_plane is not None:
            xx_t, yy_t, zz_t = target_plane 
            R, C = xx_t.shape[0], xx_t.shape[1]
        else: 
            xx_t, yy_t, zz_t = xx.clone(), yy.clone(), torch.full(xx.shape, z, dtype=torch.float64, device=light.device) 
        
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            light_propagated = Light((1, len(light.wvl), R, C), pitch=(abs(xx_t[0, 0]-xx_t[-1, 0])/len(xx_t[0])).item(), 
                                    wvl=light.wvl, device=light.device)
            field_output = torch.zeros((1, len(light.wvl), R//sampling_ratio, C//sampling_ratio), dtype=torch.complex128, device=light.device)
        else:
            light_propagated = Light((1, 1, R, C), pitch=(abs(xx_t[0, 0]-xx_t[-1, 0])/len(xx_t[0])).item(), 
                                    wvl=light.wvl, device=light.device)
            field_output = torch.zeros((1, 1, R//sampling_ratio, C//sampling_ratio), dtype=torch.complex128, device=light.device)
        # Iterate over each pixel in the output image
        for idx_y in range(0, R, sampling_ratio):
            for idx_x in range(0, C, sampling_ratio):
                X, Y, Z = xx_t[idx_y, idx_x], yy_t[idx_y, idx_x], zz_t[idx_y, idx_x]
                r = torch.sqrt((X-xx)**2 + (Y-yy)**2 + Z**2)

                contribution = (Z/r) * (1-1./(1j*k*r)) * torch.exp(1j*k*r) / r * field_input 
                field_output[:, :, idx_y//sampling_ratio, idx_x//sampling_ratio] = torch.sum(contribution, dim=(-2, -1))
                
        
        field_output = k / (2*torch.pi*1j) * light.pitch**2 * field_output

        light_propagated.field.real = Func.interpolate(field_output.real, scale_factor=sampling_ratio)
        light_propagated.field.imag = Func.interpolate(field_output.imag, scale_factor=sampling_ratio)
        return light_propagated
    

    def forward_RayleighSommerfeld_vectorized(self, light, z, target_plane=None, steps=100):
        """
        More parallelized implementation of RayleighSommerfeld propagation using vectorization.
        Args:
            light: incident light 
            z: propagation distance in meter. 
            target_plane: RayleighSommerfeld is not FFT-based calculation, so window size of target_plane is not fixed. 
                            composed of (target_XX, target_YY, target_ZZ) each target_OO is a torch.meshgrid.
            steps: How many steps will the total computation be dividied into. If memory limitation issue happens, increase
                    the number of steps to reduce the memory consumption per each computation step.
        Returns:
            light_propagated: light after propagation
        """
        field_input = light.field
        light_propagated = light.clone()

        k = 2 * np.pi / light.wvl
        R, C = light.dim[2], light.dim[3]
        x = torch.linspace(-C//2, C//2, C, dtype=torch.float64, device=light.device) * light.pitch
        y = torch.linspace(-R//2, R//2, R, dtype=torch.float64, device=light.device) * light.pitch
        xx, yy = torch.meshgrid(x, y)
        xx = xx.to(light.device)
        yy = yy.to(light.device)

        if target_plane:
            xx_t, yy_t, zz_t = target_plane 
            R, C = xx_t.shape[0], xx_t.shape[1]
        else: 
            xx_t, yy_t, zz_t = xx.clone(), yy.clone(), torch.full(xx.shape, z, dtype=torch.float64, device=light.device)

        # Flatten the target plane coordinates
        X = xx_t.flatten().to(light.device)
        Y = yy_t.flatten().to(light.device)
        Z = zz_t.flatten().to(light.device)

        # Expand the coordinate grids for broadcasting
        xx = xx.unsqueeze(0) # xx.shape == (1, R, C)
        yy = yy.unsqueeze(0) # yy.shape == (1, R, C)
        X = X.unsqueeze(1).unsqueeze(1) # X.shape = (R*C, 1, 1)
        Y = Y.unsqueeze(1).unsqueeze(1) # Y.shape = (R*C, 1, 1)
        Z = Z.unsqueeze(1).unsqueeze(1) # Z.shape = (R*C, 1, 1)

        """Original implementation (with inifinte memory):
        # r.shape == (R*C, R, C). r[i*C+j, :, :] is a (i,j) r value of the target plane
        r = torch.sqrt((X - xx)**2 + (Y - yy)**2 + Z**2)
        contribution = (Z / r) * (1 - 1./(1j * k * r)) * torch.exp(1j * k * r) / r * field_input 
        contribution = contribution.sum(dim=(-2, -1))
        field_output = contribution.view(field_output.shape)
        field_output = k / (2 * torch.pi * 1j) * light.pitch**2 * field_output
        """
        field_output = torch.zeros((R*C,), dtype=torch.complex128, device=light.device)
        step_length = R*C//steps # length of vector used in a computation step
        for step in range(steps):
            print('step: ', step)
            start = step * step_length
            end = (step + 1) * step_length if step < steps - 1 else R * C
            r_partial = (torch.sqrt((X[start: end] - xx)**2
                                    + (Y[start: end] - yy)**2
                                    + Z[start: end]**2)) # r_partial.shape = (step_length, R, C)
            contribution_partial = ((Z[start: end] / r_partial) *
                                    (1 - 1./(1j * k * r_partial)) *
                                    torch.exp(1j * k * r_partial) / r_partial * field_input)
            contribution_partial = contribution_partial.sum(dim=(-2, -1)).view((len(r_partial),))
            field_output[start:end] = contribution_partial

        field_output = field_output.view(field_input.shape)
        field_output = k / (2 * torch.pi * 1j) * light.pitch**2 * field_output

        light_propagated.field = field_output
        light_propagated.set_pitch((abs(xx_t[0, 0]-xx_t[-1, 0])/len(xx_t[0])).item())
        return light_propagated
    
    def forward_RayleighSommerfeld_dh(self, light, z, linear=True, target_plane=None, sampling_ratio=1):
        # Rayleigh-Sommerfeld is not FFT-based calculation, so window size of target_plane is not fixed. 
        
        field_input = light.field
        light_propagated = light.clone()
        
        k = 2 * np.pi / light.wvl
        R_s, C_s = light.dim[2], light.dim[3]  
        x_s = (torch.arange(C_s) - (C_s - 1) / 2) * light.pitch
        y_s = (torch.arange(R_s) - (R_s - 1) / 2) * light.pitch
        xx_s, yy_s = torch.meshgrid(x_s, y_s, indexing='ij')
        xx_s = xx_s.to(light.device)
        yy_s = yy_s.to(light.device)
        
        if target_plane:
            xx_t, yy_t, zz_t = target_plane 
            R_t, C_t = xx_t.shape[0], xx_t.shape[1] 
        else: 
            xx_t, yy_t, zz_t = xx_s.clone(), yy_s.clone(), torch.full(xx_s.shape, z, dtype=torch.float64, device=light.device) 
            R_t, C_t = R_s, C_s
        
        field_output = torch.zeros((field_input.shape[0], field_input.shape[1], R_t, C_t), dtype=torch.complex128, device=light.device)

        # Iterate over each pixel in the output image
        for idx_x in range(C_t):
            for idx_y in range(R_t):
                X, Y, Z = xx_t[idx_y, idx_x], yy_t[idx_y, idx_x], zz_t[idx_y, idx_x]
                r = torch.sqrt((X - xx_s)**2 + (Y - yy_s)**2 + Z**2)

                r = torch.clamp(r, min=1e-9)

                contribution = (Z / r) * (1 - 1./(1j * k * r)) * torch.exp(1j * k * r) / r * field_input
    
                contribution = contribution.sum(dim=(-2, -1))
                field_output[:, :, idx_y, idx_x] = contribution
                
        field_output = k / (2 * torch.pi * 1j) * light.pitch**2 * field_output

        light_propagated.field = field_output

        return light_propagated
