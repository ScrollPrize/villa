import torch
import numpy as np
from typing import Union, Optional, Tuple, List
import random
import torch.nn.functional as F
import math
from ..base import BaseAugmentation, ArrayLike

class GaussianNoise(BaseAugmentation):
    """
    Applies Gaussian noise to 3D volumes.
    
    This augmentation adds random Gaussian noise to the input data,
    which is useful for simulating sensor noise and improving model 
    robustness to noisy inputs common in CT and other imaging modalities.
    
    The noise is characterized by its mean and standard deviation, with
    the possibility to apply it selectively to certain intensity ranges.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        mean: float = 0.0,
        std: float = 0.05,
        clip: bool = True,
        per_channel: bool = False,
        apply_to_label: bool = False,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the Gaussian noise augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        mean : float, default=0.0
            Mean of the Gaussian noise distribution
        std : float, default=0.05
            Standard deviation of the Gaussian noise distribution
            Controls the noise intensity (higher = more noise)
        clip : bool, default=True
            Whether to clip values to the original data range after adding noise
        per_channel : bool, default=False
            Whether to generate different noise for each channel
        apply_to_label : bool, default=False
            Whether to apply noise to label data (usually False for segmentation)
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.mean = mean
        self.std = std
        self.clip = clip
        self.per_channel = per_channel
        self.apply_to_label = apply_to_label
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _add_gaussian_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Data with added Gaussian noise
        """
        # Check if the data is a label and if we should apply noise to it
        is_label = (self.label is not None and data is self.label)
        if is_label and not self.apply_to_label:
            return data
        
        # Get original data range for clipping if needed
        if self.clip:
            min_val = torch.min(data)
            max_val = torch.max(data)
        
        # Generate noise based on data shape
        if self.per_channel and data.dim() == 4:  # Multi-channel data
            # Generate different noise for each channel
            channels = data.shape[0]
            shape = (channels,) + tuple(data.shape[1:])
            noise = torch.randn(shape, device=self.device) * self.std + self.mean
        else:
            # Generate same noise pattern across all channels
            noise = torch.randn_like(data, device=self.device) * self.std + self.mean
        
        # Add noise to data
        noisy_data = data + noise
        
        # Clip values if requested to maintain the original range
        if self.clip:
            noisy_data = torch.clamp(noisy_data, min_val, max_val)
        
        return noisy_data
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the Gaussian noise to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with added noise
        """
        return self._add_gaussian_noise(data)


class RicianNoise(BaseAugmentation):
    """
    Applies Rician noise to 3D volumes.
    
    Rician noise is common in magnitude MRI images and other modalities where 
    the signal is computed as the magnitude of complex measurements. It follows
    a Rice distribution rather than a Gaussian distribution.
    
    This augmentation is particularly relevant for medical imaging applications
    where the noise characteristics differ from typical Gaussian assumptions.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        sigma: float = 0.05,
        clip: bool = True,
        per_channel: bool = False,
        apply_to_label: bool = False,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the Rician noise augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        sigma : float, default=0.05
            Standard deviation of the underlying Gaussian distributions
            Controls the noise intensity (higher = more noise)
        clip : bool, default=True
            Whether to clip values to the original data range after adding noise
        per_channel : bool, default=False
            Whether to generate different noise for each channel
        apply_to_label : bool, default=False
            Whether to apply noise to label data (usually False for segmentation)
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.sigma = sigma
        self.clip = clip
        self.per_channel = per_channel
        self.apply_to_label = apply_to_label
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _add_rician_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Apply Rician noise to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Data with added Rician noise
        """
        # Check if the data is a label and if we should apply noise to it
        is_label = (self.label is not None and data is self.label)
        if is_label and not self.apply_to_label:
            return data
        
        # Get original data range for clipping if needed
        if self.clip:
            min_val = torch.min(data)
            max_val = torch.max(data)
        
        # Generate shape for noise
        if self.per_channel and data.dim() == 4:  # Multi-channel data
            # Generate different noise for each channel
            channels = data.shape[0]
            shape = (channels,) + tuple(data.shape[1:])
        else:
            shape = data.shape
        
        # Generate two independent Gaussian noise components
        # For Rician noise: sqrt((v + n1)^2 + n2^2) where n1, n2 ~ N(0, sigma^2)
        noise_real = torch.randn(shape, device=self.device) * self.sigma
        noise_imag = torch.randn(shape, device=self.device) * self.sigma
        
        # Add real component to data and combine with imaginary component
        # This creates the characteristic Rician distribution
        noisy_data = torch.sqrt((data + noise_real)**2 + noise_imag**2)
        
        # Clip values if requested to maintain the original range
        if self.clip:
            noisy_data = torch.clamp(noisy_data, min_val, max_val)
        
        return noisy_data
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the Rician noise to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with added noise
        """
        return self._add_rician_noise(data)


class SaltAndPepperNoise(BaseAugmentation):
    """
    Applies salt and pepper noise (impulse noise) to 3D volumes.
    
    This augmentation randomly sets pixels to minimum or maximum values,
    creating a "salt and pepper" effect. It's useful for simulating 
    detector defects, transmission errors, or other impulse-like noise.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        prob: float = 0.01,
        salt_vs_pepper: float = 0.5,
        per_channel: bool = False,
        apply_to_label: bool = False,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the salt and pepper noise augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        prob : float, default=0.01
            Probability of a pixel being replaced with salt or pepper noise
        salt_vs_pepper : float, default=0.5
            Ratio of salt (high value) vs pepper (low value) noise
            0.0 = all pepper, 1.0 = all salt
        per_channel : bool, default=False
            Whether to generate different noise for each channel
        apply_to_label : bool, default=False
            Whether to apply noise to label data (usually False for segmentation)
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.prob = prob
        self.salt_vs_pepper = salt_vs_pepper
        self.per_channel = per_channel
        self.apply_to_label = apply_to_label
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _add_salt_and_pepper_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Apply salt and pepper noise to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Data with added salt and pepper noise
        """
        # Check if the data is a label and if we should apply noise to it
        is_label = (self.label is not None and data is self.label)
        if is_label and not self.apply_to_label:
            return data
        
        # Get min and max values for salt and pepper
        min_val = torch.min(data)
        max_val = torch.max(data)
        
        # Create a copy of the data
        noisy_data = data.clone()
        
        if self.per_channel and data.dim() == 4:
            # Apply noise per channel
            channels = data.shape[0]
            for c in range(channels):
                # Create a mask of where to apply noise
                mask = torch.rand(data[c].shape, device=self.device) < self.prob
                
                # Create a mask for salt noise (high values)
                salt_mask = torch.rand(data[c].shape, device=self.device) < self.salt_vs_pepper
                salt_mask = mask & salt_mask
                
                # Create a mask for pepper noise (low values)
                pepper_mask = mask & ~salt_mask
                
                # Apply the noise
                noisy_data[c][salt_mask] = max_val
                noisy_data[c][pepper_mask] = min_val
        else:
            # Apply the same noise pattern to all channels or single-channel data
            # Create a mask of where to apply noise
            mask = torch.rand(data.shape, device=self.device) < self.prob
            
            # Create a mask for salt noise (high values)
            salt_mask = torch.rand(data.shape, device=self.device) < self.salt_vs_pepper
            salt_mask = mask & salt_mask
            
            # Create a mask for pepper noise (low values)
            pepper_mask = mask & ~salt_mask
            
            # Apply the noise
            noisy_data[salt_mask] = max_val
            noisy_data[pepper_mask] = min_val
        
        return noisy_data
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the salt and pepper noise to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with added noise
        """
        return self._add_salt_and_pepper_noise(data)


class ScipyDownsampling(BaseAugmentation):
    """
    Applies downsampling and then upsampling to 3D volumes using scipy.ndimage.
    
    This implementation is more reliable across platforms, including those
    without CUDA support. It simulates lower resolution data by first downsampling
    the volume to a lower resolution and then upsampling it back to the
    original size, removing high-frequency details in the process.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        scale_factor: Union[float, List[float]] = 0.5,
        order: int = 1,            # Interpolation order for volume data
        order_label: int = 0,      # Interpolation order for label data
        apply_to_label: bool = False,
        device: Optional[Union[str, torch.device]] = 'cpu',  # Ignored but kept for API
        **kwargs
    ):
        """Initialize the scipy-based downsampling augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        scale_factor : Union[float, List[float]], default=0.5
            Scale factor for downsampling. Can be a single float or a list of 3 floats [z, y, x]
            Values must be between 0 and 1 (smaller values = more downsampling)
        order : int, default=1
            Interpolation order for volume data (0=nearest, 1=linear, etc.)
        order_label : int, default=0
            Interpolation order for label data (default: nearest)
        apply_to_label : bool, default=False
            Whether to apply downsampling to label data
        device : Optional[Union[str, torch.device]], default='cpu'
            Device is ignored but kept for API compatibility
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        # Validate scale factor
        if isinstance(scale_factor, (int, float)):
            if not 0 < scale_factor < 1:
                raise ValueError("Scale factor must be between 0 and 1")
            self.scale_factor = [scale_factor] * 3
        elif isinstance(scale_factor, list):
            if len(scale_factor) != 3:
                raise ValueError("Scale factor list must have 3 elements for 3D data [z, y, x]")
            if not all(0 < sf < 1 for sf in scale_factor):
                raise ValueError("All scale factors must be between 0 and 1")
            self.scale_factor = scale_factor
        else:
            raise TypeError("Scale factor must be a float or a list of 3 floats")
        
        self.order = order
        self.order_label = order_label
        self.apply_to_label = apply_to_label
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the downsampling to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with resolution reduction
        """
        from scipy.ndimage import zoom
        import numpy as np
        
        # Convert to numpy array for scipy operations
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            orig_device = data.device
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data
        
        # Ensure the data is contiguous
        data_np = np.ascontiguousarray(data_np)
        
        # Determine if we're dealing with a volume or label
        is_label = False
        if self.label is not None and data is self.label:
            is_label = True
            order = self.order_label
        else:
            order = self.order
            
        # Skip if not supposed to apply to labels
        if is_label and not self.apply_to_label:
            # Convert back to tensor if input was tensor
            if is_tensor:
                return torch.from_numpy(data_np).to(orig_device)
            else:
                return data_np
        
        # Apply downsampling and upsampling
        result = self._apply_downsampling(data_np, order)
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            # Ensure the result is contiguous
            result = np.ascontiguousarray(result)
            result = torch.from_numpy(result).to(orig_device)
        
        return result
    
    def _apply_downsampling(self, data_np: np.ndarray, order: int) -> np.ndarray:
        """Apply downsampling and upsampling to numpy array.
        
        Parameters
        ----------
        data_np : np.ndarray
            Input data as numpy array
        order : int
            Interpolation order
            
        Returns
        -------
        np.ndarray
            Downsampled and upsampled data
        """
        from scipy.ndimage import zoom
        import numpy as np
        
        # Handle different dimensionality
        if data_np.ndim == 4:  # (C, D, H, W)
            channels = data_np.shape[0]
            result = np.zeros_like(data_np)
            
            # Process each channel independently
            for c in range(channels):
                # Downsample (smaller factors = more downsampling)
                downsampled = zoom(data_np[c], self.scale_factor, order=order)
                
                # Upsample back to original size (larger factors = more upsampling)
                upsampling_factors = [1/sf for sf in self.scale_factor]
                result[c] = zoom(downsampled, upsampling_factors, order=order)
            
            return result
        
        else:  # 3D volume (D, H, W)
            # Downsample
            downsampled = zoom(data_np, self.scale_factor, order=order)
            
            # Upsample back to original size
            upsampling_factors = [1/sf for sf in self.scale_factor]
            result = zoom(downsampled, upsampling_factors, order=order)
            
            return result


class Downsampling(BaseAugmentation):
    """
    Applies downsampling and then upsampling to 3D volumes.
    
    This augmentation simulates lower resolution data by first downsampling
    the volume to a lower resolution and then upsampling it back to the
    original size. This process removes high-frequency details and can
    simulate variations in resolution common in imaging data, including 
    ancient scroll CT scans where resolution might vary between scans.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        scale_factor: Union[float, List[float]] = 0.5,
        mode: str = 'trilinear',
        label_mode: str = 'nearest',
        align_corners: Optional[bool] = True, 
        apply_to_label: bool = False,
        z_consistency: bool = True,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the downsampling augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        scale_factor : Union[float, List[float]], default=0.5
            Scale factor for downsampling. Can be a single float or a list of 3 floats [z, y, x]
            Values must be between 0 and 1 (smaller values = more downsampling)
        mode : str, default='trilinear'
            Interpolation mode for volume data ('nearest', 'linear', 'bilinear', 'trilinear')
        label_mode : str, default='nearest'
            Interpolation mode for label data (usually 'nearest' to preserve label values)
        align_corners : Optional[bool], default=True
            Geometrically, determines whether the extreme values (-1 and 1) are considered
            as referring to the centers or corners of the input's corner pixels
        apply_to_label : bool, default=False
            Whether to apply downsampling to label data
        z_consistency : bool, default=True
            Whether to apply the same transformation to all z-slices
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        # Validate scale factor
        if isinstance(scale_factor, (int, float)):
            if not 0 < scale_factor < 1:
                raise ValueError("Scale factor must be between 0 and 1")
            self.scale_factor = [scale_factor] * 3
        elif isinstance(scale_factor, list):
            if len(scale_factor) != 3:
                raise ValueError("Scale factor list must have 3 elements for 3D data [z, y, x]")
            if not all(0 < sf < 1 for sf in scale_factor):
                raise ValueError("All scale factors must be between 0 and 1")
            self.scale_factor = scale_factor
        else:
            raise TypeError("Scale factor must be a float or a list of 3 floats")
        
        self.mode = mode
        self.label_mode = label_mode
        self.align_corners = align_corners
        self.apply_to_label = apply_to_label
        self.z_consistency = z_consistency
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _apply_downsampling(self, data: torch.Tensor, is_label: bool = False) -> torch.Tensor:
        """Apply downsampling and upsampling to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
        is_label : bool, default=False
            Whether the data is a label (affects interpolation mode)
            
        Returns
        -------
        torch.Tensor
            Downsampled and upsampled data
        """
        # Check if the data is a label and if we should apply transformation to it
        if is_label and not self.apply_to_label:
            return data
        
        # Handle channel dimension
        has_channel_dim = data.dim() == 4
        
        # If no channel dimension, add one for compatibility with F.interpolate
        if not has_channel_dim:
            data = data.unsqueeze(0)
        
        # Get current size
        current_size = list(data.shape[-3:])  # [z, y, x]
        
        # Compute downsampled size
        downsampled_size = [int(s * sf) for s, sf in zip(current_size, self.scale_factor)]
        
        # Choose interpolation mode based on whether it's a label
        mode = self.label_mode if is_label else self.mode
        
        # Downsample
        downsampled = F.interpolate(
            data, 
            size=downsampled_size, 
            mode=mode,
            align_corners=self.align_corners if mode != 'nearest' else None
        )
        
        # Upsample back to original size
        upsampled = F.interpolate(
            downsampled, 
            size=current_size, 
            mode=mode,
            align_corners=self.align_corners if mode != 'nearest' else None
        )
        
        # Remove channel dimension if it was added
        if not has_channel_dim:
            upsampled = upsampled.squeeze(0)
        
        return upsampled
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the downsampling to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with resolution reduction
        """
        # Check if the data is a label to use appropriate parameters
        is_label = (self.label is not None and data is self.label)
        
        return self._apply_downsampling(data, is_label=is_label)


class MotionBlur(BaseAugmentation):
    """
    Applies motion blur to 3D volumes to simulate motion artifacts.
    
    This augmentation simulates the blurring that occurs when there is
    relative motion between the imaging device and the subject during
    acquisition. For vesuvius scrolls, this can simulate micro-movements
    during the CT scanning process that lead to motion artifacts.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        kernel_size: int = 7,
        angle: Optional[float] = None,
        direction: Optional[List[float]] = None,
        per_slice: bool = True,
        apply_to_label: bool = False,
        z_consistency: bool = True,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the motion blur augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        kernel_size : int, default=7
            Size of the blur kernel (must be odd, larger = more blur)
        angle : Optional[float], default=None
            Angle of motion blur in degrees. If None, a random angle is chosen.
            Only used if direction is None. Range: [0, 180]
        direction : Optional[List[float]], default=None
            3D direction vector [z, y, x] for the blur. If provided, overrides angle.
            Will be normalized internally
        per_slice : bool, default=True
            If True, applies 2D motion blur to each slice independently.
            If False, applies 3D motion blur to the whole volume.
        apply_to_label : bool, default=False
            Whether to apply blur to label data (usually False for segmentation)
        z_consistency : bool, default=True
            Whether to use the same blur direction for all z-slices
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        
        self.angle = angle
        self.direction = direction
        self.per_slice = per_slice
        self.apply_to_label = apply_to_label
        self.z_consistency = z_consistency
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _create_motion_blur_kernel(self, angle: float = None, direction: List[float] = None, dims: int = 2) -> torch.Tensor:
        """Create a motion blur kernel.
        
        Parameters
        ----------
        angle : float, default=None
            Angle of motion blur in degrees (for 2D)
        direction : List[float], default=None
            Direction vector for the blur (for 3D)
        dims : int, default=2
            Dimensions of the kernel (2 for 2D, 3 for 3D)
            
        Returns
        -------
        torch.Tensor
            Motion blur kernel
        """
        if dims == 2:
            # Generate a 2D motion blur kernel
            kernel_size = self.kernel_size
            
            # If angle is None, generate a random angle
            if angle is None:
                angle = random.uniform(0, 180)
            
            # Convert angle to radians
            angle_rad = torch.tensor(angle * torch.pi / 180.0, device=self.device)
            
            # Create a line across the kernel in the specified direction
            kernel = torch.zeros((kernel_size, kernel_size), device=self.device)
            center = kernel_size // 2
            
            for i in range(-center, center + 1):
                # Calculate point on the line
                x = center + round(float(i * torch.cos(angle_rad)))
                y = center + round(float(i * torch.sin(angle_rad)))
                
                # Check if the point is inside the kernel
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1.0
            
            # Normalize the kernel
            kernel = kernel / kernel.sum()
            
            return kernel
        
        elif dims == 3:
            # Generate a 3D motion blur kernel
            kernel_size = self.kernel_size
            
            # If direction is None, generate a random direction
            if direction is None:
                # Generate a random vector
                direction = [random.uniform(-1, 1) for _ in range(3)]
            
            # Normalize the direction vector using torch
            direction_tensor = torch.tensor(direction, device=self.device)
            direction_tensor = direction_tensor / torch.norm(direction_tensor)
            direction = direction_tensor.tolist()
            
            # Create a 3D kernel
            kernel = torch.zeros((kernel_size, kernel_size, kernel_size), device=self.device)
            center = kernel_size // 2
            
            for i in range(-center, center + 1):
                # Calculate point on the line
                z = center + round(i * direction[0])
                y = center + round(i * direction[1])
                x = center + round(i * direction[2])
                
                # Check if the point is inside the kernel
                if (0 <= z < kernel_size and 
                    0 <= y < kernel_size and 
                    0 <= x < kernel_size):
                    kernel[z, y, x] = 1.0
            
            # Normalize the kernel
            if kernel.sum() > 0:
                kernel = kernel / kernel.sum()
            else:
                # Fallback if no points were set
                kernel[center, center, center] = 1.0
            
            return kernel
        
        else:
            raise ValueError("Dimensions must be 2 or 3")
    
    def _apply_motion_blur(self, data: torch.Tensor) -> torch.Tensor:
        """Apply motion blur to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Data with motion blur applied
        """
        # Check if the data is a label and if we should apply blur to it
        is_label = (self.label is not None and data is self.label)
        if is_label and not self.apply_to_label:
            return data
        
        # Handle channel dimension
        has_channel_dim = data.dim() == 4
        
        # If no channel dimension, add one for compatibility with convolution
        if not has_channel_dim:
            data = data.unsqueeze(0)
        
        # Get number of channels and depth
        channels = data.shape[0]
        depth = data.shape[1] if data.dim() >= 4 else 1
        
        # Create output tensor
        result = torch.zeros_like(data)
        
        if self.per_slice:
            # Apply 2D motion blur to each slice independently
            for c in range(channels):
                z_angle = None if not self.z_consistency else random.uniform(0, 180)
                
                for z in range(depth):
                    # Get the current slice
                    slice_data = data[c, z]
                    
                    # Create a 2D kernel for this slice
                    if self.z_consistency:
                        angle = z_angle
                    else:
                        angle = self.angle if self.angle is not None else random.uniform(0, 180)
                    
                    kernel = self._create_motion_blur_kernel(angle=angle, dims=2)
                    
                    # Reshape kernel for Conv2d
                    kernel = kernel.unsqueeze(0).unsqueeze(0)
                    
                    # Apply convolution
                    padded_slice = F.pad(slice_data.unsqueeze(0).unsqueeze(0), 
                                        [self.kernel_size//2] * 4, 
                                        mode='reflect')
                    blurred_slice = F.conv2d(padded_slice, kernel)
                    
                    # Store the result
                    result[c, z] = blurred_slice[0, 0]
        else:
            # Apply 3D motion blur to the whole volume
            direction = self.direction
            if direction is None:
                direction = [random.uniform(-1, 1) for _ in range(3)]
            
            # Create a 3D kernel
            kernel = self._create_motion_blur_kernel(direction=direction, dims=3)
            
            # Reshape kernel for Conv3d and apply separately to each channel
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            
            for c in range(channels):
                channel_data = data[c].unsqueeze(0).unsqueeze(0)
                
                # Pad the volume
                padded_volume = F.pad(channel_data, 
                                     [self.kernel_size//2] * 6, 
                                     mode='reflect')
                
                # Apply convolution
                blurred_volume = F.conv3d(padded_volume, kernel)
                
                # Store the result
                result[c] = blurred_volume[0, 0]
        
        # Remove channel dimension if it was added
        if not has_channel_dim:
            result = result.squeeze(0)
        
        return result
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply motion blur to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with motion blur
        """
        return self._apply_motion_blur(data)


class RingArtifact(BaseAugmentation):
    """
    Applies ring artifacts to 3D volumes to simulate detector malfunctions.
    
    This augmentation simulates the ring artifacts that commonly occur in CT 
    scanners due to miscalibrated or defective detector elements. It creates 
    concentric rings centered at specified locations, which is particularly 
    relevant for vesuvius scroll CT data where detector artifacts might be present.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        num_rings: Union[int, Tuple[int, int]] = (2, 5),
        intensity_range: Tuple[float, float] = (0.1, 0.3),
        thickness_range: Tuple[int, int] = (1, 3),
        center: Optional[Tuple[int, int]] = None,
        apply_to_label: bool = False,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the ring artifact augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        num_rings : Union[int, Tuple[int, int]], default=(2, 5)
            Number of rings to add or a range (min, max) for random selection
        intensity_range : Tuple[float, float], default=(0.1, 0.3)
            Range of intensity values for rings (relative to data range)
            Positive values create bright rings, negative values create dark rings
        thickness_range : Tuple[int, int], default=(1, 3)
            Range of ring thickness in pixels
        center : Optional[Tuple[int, int]], default=None
            Center coordinates (y, x) for rings. If None, uses image center
            or a random position.
        apply_to_label : bool, default=False
            Whether to apply artifacts to label data (usually False)
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        # Set up ring parameters
        if isinstance(num_rings, int):
            self.num_rings = num_rings
        else:
            self.num_rings_range = num_rings
            self.num_rings = None
            
        self.intensity_range = intensity_range
        self.thickness_range = thickness_range
        self.center = center
        self.apply_to_label = apply_to_label
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
    def _generate_rings(self, shape: Tuple[int, int]) -> torch.Tensor:
        """Generate ring artifact mask.
        
        Parameters
        ----------
        shape : Tuple[int, int]
            Shape of the 2D slice to generate rings for (height, width)
            
        Returns
        -------
        torch.Tensor
            Mask with ring artifacts
        """
        height, width = shape
        
        # Determine number of rings
        if self.num_rings is None:
            num_rings = random.randint(self.num_rings_range[0], self.num_rings_range[1])
        else:
            num_rings = self.num_rings
            
        # Determine center
        if self.center is None:
            # Use the center of the image or a random position
            if random.random() < 0.7:  # 70% chance to use image center
                center_y, center_x = height // 2, width // 2
            else:
                center_y = random.randint(height // 4, 3 * height // 4)
                center_x = random.randint(width // 4, 3 * width // 4)
        else:
            center_y, center_x = self.center
            
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing='ij'
        )
        
        # Calculate distance from center
        distance = torch.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        
        # Create mask to hold all rings
        ring_mask = torch.zeros((height, width), device=self.device)
        
        # Maximum radius (distance to the corners)
        max_radius = math.sqrt(height**2 + width**2) / 2
        
        # Generate rings
        for _ in range(num_rings):
            # Random radius between 10% and 90% of max distance
            radius = random.uniform(0.1 * max_radius, 0.9 * max_radius)
            
            # Random thickness
            thickness = random.randint(self.thickness_range[0], self.thickness_range[1])
            
            # Random intensity
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
            # Randomly choose bright or dark ring
            if random.random() < 0.5:
                intensity = -intensity
                
            # Create ring
            ring = torch.abs(distance - radius) < thickness
            
            # Add ring to mask
            ring_mask[ring] = intensity
            
        return ring_mask
            
    def _apply_ring_artifacts(self, data: torch.Tensor) -> torch.Tensor:
        """Apply ring artifacts to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Data with added ring artifacts
        """
        # Check if the data is a label and if we should apply artifacts to it
        is_label = (self.label is not None and data is self.label)
        if is_label and not self.apply_to_label:
            return data
        
        # Get data range for scaling artifacts
        data_min = torch.min(data)
        data_range = torch.max(data) - data_min
        
        # Handle channel dimension
        has_channel_dim = data.dim() == 4
        
        # Create output tensor (copy of input)
        result = data.clone()
        
        # If no channel dimension, add one temporarily
        if not has_channel_dim:
            result = result.unsqueeze(0)
        
        # Get number of channels and depth
        channels = result.shape[0]
        depth = result.shape[1]
        height = result.shape[2]
        width = result.shape[3]
        
        # Create a single ring pattern for all z-slices
        ring_mask = self._generate_rings((height, width))
        
        # Scale the ring mask by data range
        scaled_mask = ring_mask * data_range
        
        # Apply rings to each channel and slice
        for c in range(channels):
            for z in range(depth):
                # Add ring artifacts
                result[c, z] = result[c, z] + scaled_mask
        
        # Ensure values stay in original range
        result = torch.clamp(result, data_min, data_min + data_range)
        
        # Remove channel dimension if it was added
        if not has_channel_dim:
            result = result.squeeze(0)
            
        return result
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply ring artifacts to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with ring artifacts
        """
        return self._apply_ring_artifacts(data)