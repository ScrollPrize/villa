import torch
import numpy as np
from typing import Union, Optional, Tuple, List
import random
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