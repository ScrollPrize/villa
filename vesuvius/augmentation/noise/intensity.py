import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Tuple, List, Callable
import random
from ..base import BaseAugmentation, ArrayLike

class InhomogeneousLighting(BaseAugmentation):
    """
    Applies inhomogeneous lighting variations across z-slices of 3D volumes.
    
    This augmentation simulates intensity variations that can occur in 3D imaging
    due to illumination inconsistencies, sensor drift, or other acquisition artifacts.
    It applies smooth, spatially varying intensity adjustments that change gradually
    across z-slices.
    
    The variation can be sinusoidal, polynomial, or based on random fields,
    providing a realistic simulation of scan-specific inconsistencies.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        mode: str = 'sinusoidal',
        strength: float = 0.3,
        z_frequency: float = 0.5,
        spatial_frequency: float = 2.0,
        apply_to_label: bool = False,
        per_channel: bool = True,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the inhomogeneous lighting augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        mode : str, default='sinusoidal'
            Type of lighting variation:
            - 'sinusoidal': Smooth sinusoidal variations
            - 'polynomial': Polynomial intensity drift
            - 'random_field': Perlin-like random field variations
        strength : float, default=0.3
            Maximum intensity adjustment magnitude (0.0-1.0)
        z_frequency : float, default=0.5
            Frequency of variation along z-axis (0.0-10.0)
        spatial_frequency : float, default=2.0
            Frequency of variation in spatial (y,x) dimensions (0.0-10.0)
        apply_to_label : bool, default=False
            Whether to apply intensity changes to label data (usually False)
        per_channel : bool, default=True
            Whether to generate different patterns for each channel
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.mode = mode
        self.strength = strength
        self.z_frequency = z_frequency
        self.spatial_frequency = spatial_frequency
        self.apply_to_label = apply_to_label
        self.per_channel = per_channel
        
        # Validate parameters
        if mode not in ['sinusoidal', 'polynomial', 'random_field']:
            raise ValueError("Mode must be one of: 'sinusoidal', 'polynomial', 'random_field'")
        if strength < 0 or strength > 1:
            raise ValueError("Strength must be between 0 and 1")
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _generate_sinusoidal_pattern(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate sinusoidal lighting pattern.
        
        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the volume to generate pattern for
            
        Returns
        -------
        torch.Tensor
            Pattern tensor with values in range [-1, 1]
        """
        # Get spatial dimensions
        if len(shape) == 4:  # (C, D, H, W)
            depth, height, width = shape[1:]
        else:  # (D, H, W)
            depth, height, width = shape
        
        # Create coordinate grids
        z = torch.linspace(0, self.z_frequency * np.pi, depth, device=self.device)
        y = torch.linspace(0, self.spatial_frequency * np.pi, height, device=self.device)
        x = torch.linspace(0, self.spatial_frequency * np.pi, width, device=self.device)
        
        # Create meshgrid
        z_grid, y_grid, x_grid = torch.meshgrid(z, y, x, indexing='ij')
        
        # Generate sinusoidal pattern
        pattern = torch.sin(z_grid) * torch.sin(y_grid) * torch.sin(x_grid)
        
        # Normalize to range [-1, 1]
        pattern = pattern / pattern.abs().max()
        
        return pattern
    
    def _generate_polynomial_pattern(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate polynomial intensity drift pattern.
        
        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the volume to generate pattern for
            
        Returns
        -------
        torch.Tensor
            Pattern tensor with values in range [-1, 1]
        """
        # Get spatial dimensions
        if len(shape) == 4:  # (C, D, H, W)
            depth, height, width = shape[1:]
        else:  # (D, H, W)
            depth, height, width = shape
        
        # Create normalized coordinate grids in range [-1, 1]
        z = torch.linspace(-1, 1, depth, device=self.device)
        y = torch.linspace(-1, 1, height, device=self.device)
        x = torch.linspace(-1, 1, width, device=self.device)
        
        # Create meshgrid
        z_grid, y_grid, x_grid = torch.meshgrid(z, y, x, indexing='ij')
        
        # Generate polynomial with random coefficients
        coeffs = torch.randn(10, device=self.device) * 0.1
        
        pattern = (coeffs[0] * z_grid + 
                  coeffs[1] * z_grid**2 + 
                  coeffs[2] * y_grid + 
                  coeffs[3] * y_grid**2 + 
                  coeffs[4] * x_grid + 
                  coeffs[5] * x_grid**2 + 
                  coeffs[6] * z_grid * y_grid + 
                  coeffs[7] * y_grid * x_grid +
                  coeffs[8] * z_grid * x_grid +
                  coeffs[9] * z_grid * y_grid * x_grid)
        
        # Normalize to range [-1, 1]
        pattern = pattern / pattern.abs().max()
        
        return pattern
    
    def _generate_random_field_pattern(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate Perlin-like random field pattern.
        
        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the volume to generate pattern for
            
        Returns
        -------
        torch.Tensor
            Pattern tensor with values in range [-1, 1]
        """
        # Get spatial dimensions
        if len(shape) == 4:  # (C, D, H, W)
            depth, height, width = shape[1:]
        else:  # (D, H, W)
            depth, height, width = shape
        
        # We'll simulate a Perlin-like noise by generating a low-resolution
        # random field and then upsampling it with trilinear interpolation
        
        # Low resolution grid size (proportional to spatial frequency)
        grid_depth = max(2, int(depth / (10 / self.z_frequency)))
        grid_height = max(2, int(height / (10 / self.spatial_frequency)))
        grid_width = max(2, int(width / (10 / self.spatial_frequency)))
        
        # Generate random low-resolution field
        low_res_field = torch.randn(1, 1, grid_depth, grid_height, grid_width, device=self.device)
        
        # Upsample to target resolution (trilinear interpolation)
        # Note: F.interpolate expects 5D input: [N, C, D, H, W]
        pattern = F.interpolate(
            low_res_field, 
            size=(depth, height, width), 
            mode='trilinear', 
            align_corners=True
        ).squeeze(0).squeeze(0)
        
        # Normalize to range [-1, 1]
        pattern = pattern / pattern.abs().max()
        
        return pattern
    
    def _apply_lighting_variation(self, data: torch.Tensor) -> torch.Tensor:
        """Apply inhomogeneous lighting variation to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Data with lighting variation applied
        """
        # Check if the data is a label and if we should apply transformation to it
        is_label = (self.label is not None and data is self.label)
        if is_label and not self.apply_to_label:
            return data
        
        # Create a copy of the data
        transformed_data = data.clone()
        
        # Generate appropriate pattern based on mode
        if self.mode == 'sinusoidal':
            pattern_generator = self._generate_sinusoidal_pattern
        elif self.mode == 'polynomial':
            pattern_generator = self._generate_polynomial_pattern
        elif self.mode == 'random_field':
            pattern_generator = self._generate_random_field_pattern
        
        if self.per_channel and data.dim() == 4:
            # Apply different patterns to each channel
            channels = data.shape[0]
            for c in range(channels):
                # Generate pattern for this channel
                pattern = pattern_generator(data[c].shape)
                
                # Scale pattern by strength and add to data
                transformed_data[c] = data[c] * (1 + self.strength * pattern)
        else:
            # Apply the same pattern to all channels
            pattern = pattern_generator(data.shape)
            
            # Scale pattern by strength and apply to data
            if data.dim() == 4:  # Multi-channel data
                # Expand pattern to include channel dimension
                pattern = pattern.unsqueeze(0)
                
            # Apply pattern
            transformed_data = data * (1 + self.strength * pattern)
        
        return transformed_data
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the lighting variation to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with lighting variation
        """
        return self._apply_lighting_variation(data)


class GammaTransform(BaseAugmentation):
    """
    Applies gamma correction to 3D volumes.
    
    Gamma correction adjusts the intensity of voxels using a non-linear
    power-law expression (I_out = I_in^gamma). Values of gamma < 1 make 
    dark regions lighter, while gamma > 1 make them darker.
    
    This is useful for adjusting contrast in a way that mimics the non-linear
    response of human perception or various imaging systems.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        invert_image: float = 0.0,  # Probability of applying to inverted image
        per_channel: bool = False,
        retain_stats: bool = False,
        apply_to_label: bool = False,
        epsilon: float = 1e-7,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the gamma transform augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        gamma_range : Tuple[float, float], default=(0.7, 1.5)
            Range of gamma values to sample from
        invert_image : float, default=0.0
            Probability of applying gamma to inverted image (1-I)
            This inverts the effect (gamma<1 darkens, gamma>1 brightens)
        per_channel : bool, default=False
            Whether to use different gamma for each channel
        retain_stats : bool, default=False
            Whether to retain mean and std of the original image
        apply_to_label : bool, default=False
            Whether to apply to label data (usually False)
        epsilon : float, default=1e-7
            Small constant to avoid numerical issues
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.per_channel = per_channel
        self.retain_stats = retain_stats
        self.apply_to_label = apply_to_label
        self.epsilon = epsilon
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _apply_gamma(self, data: torch.Tensor) -> torch.Tensor:
        """Apply gamma correction to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Gamma-corrected data
        """
        # Check if the data is a label and if we should apply to it
        is_label = (self.label is not None and data is self.label)
        if is_label and not self.apply_to_label:
            return data
        
        # Create a copy of the data
        transformed_data = data.clone()
        
        # Get min/max for normalization
        min_val = data.min()
        max_val = data.max()
        
        # Normalize data to [0, 1] range for gamma correction
        # Add epsilon to avoid issues with zero values
        data_normalized = (data - min_val) / (max_val - min_val + self.epsilon)
        
        # Store original mean and std if needed
        if self.retain_stats:
            original_mean = data.mean()
            original_std = data.std()
        
        # Apply gamma correction to each channel (or to all channels together)
        if self.per_channel and data.dim() == 4:
            # Apply different gamma to each channel
            channels = data.shape[0]
            for c in range(channels):
                # Randomly select gamma
                gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
                
                # Decide whether to invert before applying gamma
                if random.random() < self.invert_image:
                    transformed_data[c] = ((1 - data_normalized[c]) ** gamma)
                    # Re-invert after gamma correction
                    transformed_data[c] = 1 - transformed_data[c]
                else:
                    transformed_data[c] = data_normalized[c] ** gamma
                
                # Scale back to original range
                transformed_data[c] = transformed_data[c] * (max_val - min_val) + min_val
                
                # Adjust mean and std if required
                if self.retain_stats:
                    current_mean = transformed_data[c].mean()
                    current_std = transformed_data[c].std()
                    if current_std > 0:
                        transformed_data[c] = ((transformed_data[c] - current_mean) / current_std) * original_std + original_mean
        else:
            # Apply same gamma to all channels
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            
            # Decide whether to invert before applying gamma
            if random.random() < self.invert_image:
                transformed_data = ((1 - data_normalized) ** gamma)
                # Re-invert after gamma correction
                transformed_data = 1 - transformed_data
            else:
                transformed_data = data_normalized ** gamma
            
            # Scale back to original range
            transformed_data = transformed_data * (max_val - min_val) + min_val
            
            # Adjust mean and std if required
            if self.retain_stats:
                current_mean = transformed_data.mean()
                current_std = transformed_data.std()
                if current_std > 0:
                    transformed_data = ((transformed_data - current_mean) / current_std) * original_std + original_mean
        
        return transformed_data
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the gamma correction to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with gamma correction
        """
        return self._apply_gamma(data)


class ContrastAdjustment(BaseAugmentation):
    """
    Adjusts contrast in 3D volumes.
    
    This augmentation enhances or reduces the contrast of the image by
    scaling intensities around the mean value. It can be applied globally
    or with different settings for each channel.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        contrast_range: Tuple[float, float] = (0.75, 1.25),
        preserve_range: bool = True,
        per_channel: bool = False,
        apply_to_label: bool = False,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the contrast adjustment augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        contrast_range : Tuple[float, float], default=(0.75, 1.25)
            Range of contrast factors to sample from
            < 1 decreases contrast, > 1 increases contrast
        preserve_range : bool, default=True
            Whether to preserve original intensity range after adjustment
        per_channel : bool, default=False
            Whether to use different contrast for each channel
        apply_to_label : bool, default=False
            Whether to apply to label data (usually False)
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.apply_to_label = apply_to_label
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _adjust_contrast(self, data: torch.Tensor) -> torch.Tensor:
        """Adjust contrast of the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Contrast-adjusted data
        """
        # Check if the data is a label and if we should apply to it
        is_label = (self.label is not None and data is self.label)
        if is_label and not self.apply_to_label:
            return data
        
        # Get original min/max if preserving range
        if self.preserve_range:
            min_val = data.min()
            max_val = data.max()
        
        # Apply contrast adjustment to each channel (or to all channels together)
        if self.per_channel and data.dim() == 4:
            # Apply different contrast to each channel
            channels = data.shape[0]
            adjusted_data = data.clone()
            
            for c in range(channels):
                # Randomly select contrast factor
                contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
                
                # Get mean intensity for this channel
                mean = data[c].mean()
                
                # Adjust contrast: I_new = (I - mean) * contrast + mean
                adjusted_data[c] = (data[c] - mean) * contrast + mean
                
                # Preserve original range if requested
                if self.preserve_range:
                    channel_min = adjusted_data[c].min()
                    channel_max = adjusted_data[c].max()
                    if channel_max > channel_min:
                        adjusted_data[c] = (adjusted_data[c] - channel_min) / (channel_max - channel_min) * (max_val - min_val) + min_val
        else:
            # Apply same contrast to all channels
            contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
            
            # Get mean intensity
            mean = data.mean()
            
            # Adjust contrast: I_new = (I - mean) * contrast + mean
            adjusted_data = (data - mean) * contrast + mean
            
            # Preserve original range if requested
            if self.preserve_range:
                adj_min = adjusted_data.min()
                adj_max = adjusted_data.max()
                if adj_max > adj_min:
                    adjusted_data = (adjusted_data - adj_min) / (adj_max - adj_min) * (max_val - min_val) + min_val
        
        return adjusted_data
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the contrast adjustment to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with adjusted contrast
        """
        return self._adjust_contrast(data)


class Sharpen(BaseAugmentation):
    """
    Applies sharpening to 3D volumes.
    
    This augmentation enhances edges and fine details by applying a
    sharpening kernel, which is essentially a combination of the original
    image and a high-pass filtered version of it.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        alpha_range: Tuple[float, float] = (0.2, 0.8),
        kernel_size: int = 3,
        per_channel: bool = False,
        apply_to_label: bool = False,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the sharpening augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        alpha_range : Tuple[float, float], default=(0.2, 0.8)
            Range of sharpening strength to sample from
            Higher values create stronger sharpening effect
        kernel_size : int, default=3
            Size of the sharpening kernel (3, 5, 7, etc.)
        per_channel : bool, default=False
            Whether to use different sharpening for each channel
        apply_to_label : bool, default=False
            Whether to apply to label data (usually False)
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.alpha_range = alpha_range
        self.kernel_size = kernel_size
        self.per_channel = per_channel
        self.apply_to_label = apply_to_label
        
        # Validate kernel size
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Create Laplacian kernel for sharpening
        self.laplacian_kernel = self._create_3d_laplacian_kernel()
    
    def _create_3d_laplacian_kernel(self) -> torch.Tensor:
        """Create a 3D Laplacian kernel for sharpening.
        
        Returns
        -------
        torch.Tensor
            3D Laplacian kernel of shape [1, 1, kernel_size, kernel_size, kernel_size]
        """
        # Create a kernel with negative weights and a positive center
        kernel_size = self.kernel_size
        kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=self.device) * -1
        center = kernel_size // 2
        kernel[0, 0, center, center, center] = kernel_size**3 - 1
        
        # Normalize kernel to preserve image brightness
        kernel = kernel / kernel_size**3
        
        return kernel
    
    def _apply_sharpening(self, data: torch.Tensor) -> torch.Tensor:
        """Apply sharpening to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Sharpened data
        """
        # Check if the data is a label and if we should apply to it
        is_label = (self.label is not None and data is self.label)
        if is_label and not self.apply_to_label:
            return data
        
        if self.per_channel and data.dim() == 4:
            # Apply different sharpening to each channel
            channels = data.shape[0]
            sharpened_data = data.clone()
            
            for c in range(channels):
                # Random sharpening strength
                alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])
                
                # Add batch and channel dimensions for conv3d
                channel_data = data[c:c+1].unsqueeze(0)  # [1, 1, D, H, W]
                
                # Apply Laplacian filter to get high-frequency components
                high_freq = F.conv3d(
                    channel_data,
                    self.laplacian_kernel,
                    padding=self.kernel_size//2
                )
                
                # Combine original and high-frequency components
                sharpened_channel = channel_data - alpha * high_freq
                
                # Remove batch dimension and update result
                sharpened_data[c] = sharpened_channel.squeeze(0).squeeze(0)
        else:
            # Apply same sharpening to all channels
            alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])
            
            # Prepare data for conv3d (add batch dimension if needed)
            if data.dim() == 3:
                # For 3D data add both batch and channel dims
                input_data = data.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            else:
                # For 4D data [C, D, H, W] add only batch dim
                input_data = data.unsqueeze(0)  # [1, C, D, H, W]
            
            # Apply Laplacian filter separately for each channel
            high_freq = F.conv3d(
                input_data,
                self.laplacian_kernel,
                padding=self.kernel_size//2,
                groups=input_data.shape[1]  # Apply to each channel separately
            )
            
            # Combine original and high-frequency components
            sharpened_data = input_data - alpha * high_freq
            
            # Remove batch dimension
            sharpened_data = sharpened_data.squeeze(0)
            
            # If original was 3D, remove channel dimension too
            if data.dim() == 3:
                sharpened_data = sharpened_data.squeeze(0)
        
        return sharpened_data
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the sharpening to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with sharpening applied
        """
        return self._apply_sharpening(data)


class BeamHardening(BaseAugmentation):
    """
    Simulates beam hardening artifacts in 3D volumes.
    
    Beam hardening is a common CT artifact that occurs when lower energy X-rays
    are attenuated more than higher energy ones, creating characteristic
    crescent-shaped bright and dark regions in the image. This augmentation
    creates realistic beam hardening patterns to help models learn robustness
    to this type of imaging artifact.
    
    The implementation generates random crescent shapes with bright regions 
    (upper 20% of intensity range) and dark regions (lower 30% of intensity range).
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        num_crescents: Tuple[int, int] = (1, 3),
        bright_intensity_range: Tuple[float, float] = (0.8, 1.0),
        dark_intensity_range: Tuple[float, float] = (0.0, 0.3),
        crescent_width_range: Tuple[float, float] = (0.1, 0.3),
        crescent_angle_range: Tuple[float, float] = (20, 120),
        strength: float = 0.7,
        apply_to_label: bool = False,
        per_channel: bool = False,
        z_consistency: float = 0.8,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the beam hardening artifact augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        num_crescents : Tuple[int, int], default=(1, 3)
            Range for the number of crescent artifacts to generate (min, max)
        bright_intensity_range : Tuple[float, float], default=(0.8, 1.0)
            Intensity range for bright crescents, as proportion of the full range
        dark_intensity_range : Tuple[float, float], default=(0.0, 0.3)
            Intensity range for dark crescents, as proportion of the full range
        crescent_width_range : Tuple[float, float], default=(0.1, 0.3)
            Width of crescents as proportion of the volume size
        crescent_angle_range : Tuple[float, float], default=(20, 120)
            Angular range of crescents in degrees
        strength : float, default=0.7
            Strength of the effect (0.0-1.0)
        apply_to_label : bool, default=False
            Whether to apply to label data (usually False)
        per_channel : bool, default=False
            Whether to generate different artifacts for each channel
        z_consistency : float, default=0.8
            How consistent the artifact is across z-slices (0.0-1.0)
            1.0 means identical across all slices, 0.0 means independent
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.num_crescents = num_crescents
        self.bright_intensity_range = bright_intensity_range
        self.dark_intensity_range = dark_intensity_range
        self.crescent_width_range = crescent_width_range
        self.crescent_angle_range = crescent_angle_range
        self.strength = strength
        self.apply_to_label = apply_to_label
        self.per_channel = per_channel
        self.z_consistency = z_consistency
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _generate_crescent_mask(self, shape: Tuple[int, ...], is_bright: bool) -> torch.Tensor:
        """Generate a crescent-shaped mask.
        
        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the target volume (D, H, W) or (C, D, H, W)
        is_bright : bool
            Whether to generate a bright (True) or dark (False) crescent
            
        Returns
        -------
        torch.Tensor
            Mask with crescent pattern
        """
        # Get spatial dimensions
        if len(shape) == 4:  # (C, D, H, W)
            depth, height, width = shape[1:]
        else:  # (D, H, W)
            depth, height, width = shape
        
        # Sample crescent parameters
        center_y = random.uniform(0.3 * height, 0.7 * height)
        center_x = random.uniform(0.3 * width, 0.7 * width)
        radius = random.uniform(0.3 * min(height, width), 0.5 * min(height, width))
        
        # Crescent width as proportion of radius
        crescent_width = random.uniform(self.crescent_width_range[0], self.crescent_width_range[1]) * radius
        
        # Crescent angle in radians
        angle_degrees = random.uniform(self.crescent_angle_range[0], self.crescent_angle_range[1])
        angle_radians = angle_degrees * np.pi / 180.0
        
        # Random rotation of the entire crescent
        rotation = random.uniform(0, 2 * np.pi)
        
        # Generate coordinates
        y_indices, x_indices = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing='ij'
        )
        
        # Calculate distance from center
        distance = torch.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
        
        # Calculate angles of each point relative to center
        angles = torch.atan2(y_indices - center_y, x_indices - center_x)
        
        # Adjust angles based on random rotation
        angles = (angles - rotation) % (2 * np.pi)
        
        # Create circular mask
        circle_mask = distance <= radius
        
        # Create angular mask for the crescent
        angular_mask = angles <= angle_radians
        
        # Create inner circle mask for crescent shape
        if is_bright:
            # For bright crescents, use a smaller inner circle
            inner_radius = radius - crescent_width
            inner_circle_mask = distance <= inner_radius
        else:
            # For dark crescents, use a slightly shifted inner circle
            # to create a different crescent shape
            shift_y = random.uniform(-0.15, 0.15) * radius
            shift_x = random.uniform(-0.15, 0.15) * radius
            inner_distance = torch.sqrt(
                (y_indices - (center_y + shift_y))**2 + 
                (x_indices - (center_x + shift_x))**2
            )
            inner_radius = radius - crescent_width
            inner_circle_mask = inner_distance <= inner_radius
        
        # Combine masks for final crescent shape
        if is_bright:
            crescent_mask = circle_mask & angular_mask & ~inner_circle_mask
        else:
            crescent_mask = circle_mask & angular_mask & inner_circle_mask
        
        # Generate z-dimension masks with consistency along z
        z_masks = []
        prev_mask = crescent_mask.clone()
        
        for z in range(depth):
            if z == 0 or random.random() > self.z_consistency:
                # Generate a new pattern for this z-slice
                if z > 0:
                    # Generate new parameters while keeping some consistency
                    center_y += random.uniform(-radius * 0.05, radius * 0.05)
                    center_x += random.uniform(-radius * 0.05, radius * 0.05)
                    radius = max(0.3 * min(height, width), min(radius * random.uniform(0.95, 1.05), 0.5 * min(height, width)))
                    angle_radians = max(self.crescent_angle_range[0] * np.pi / 180.0, 
                                         min(angle_radians * random.uniform(0.95, 1.05),
                                             self.crescent_angle_range[1] * np.pi / 180.0))
                    
                    # Recalculate distance and angles
                    distance = torch.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
                    angles = torch.atan2(y_indices - center_y, x_indices - center_x)
                    angles = (angles - rotation) % (2 * np.pi)
                    
                    # Recreate masks
                    circle_mask = distance <= radius
                    angular_mask = angles <= angle_radians
                    
                    if is_bright:
                        inner_radius = radius - crescent_width
                        inner_circle_mask = distance <= inner_radius
                    else:
                        shift_y += random.uniform(-0.02, 0.02) * radius
                        shift_x += random.uniform(-0.02, 0.02) * radius
                        inner_distance = torch.sqrt(
                            (y_indices - (center_y + shift_y))**2 + 
                            (x_indices - (center_x + shift_x))**2
                        )
                        inner_radius = radius - crescent_width
                        inner_circle_mask = inner_distance <= inner_radius
                    
                    if is_bright:
                        crescent_mask = circle_mask & angular_mask & ~inner_circle_mask
                    else:
                        crescent_mask = circle_mask & angular_mask & inner_circle_mask
                
                prev_mask = crescent_mask.clone()
            else:
                # Use previous mask with slight modifications
                noise = torch.rand_like(prev_mask.float()) * 0.05
                crescent_mask = (prev_mask.float() + noise).round().bool()
            
            z_masks.append(crescent_mask)
        
        # Stack z-masks to create 3D mask
        stacked_mask = torch.stack(z_masks, dim=0)
        
        return stacked_mask
    
    def _apply_beam_hardening(self, data: torch.Tensor) -> torch.Tensor:
        """Apply beam hardening artifacts to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Data with beam hardening artifacts
        """
        # Check if the data is a label and if we should apply to it
        is_label = (self.label is not None and data is self.label)
        if is_label and not self.apply_to_label:
            return data
        
        # Create a copy of the data
        result = data.clone()
        
        # Get data range for intensity scaling
        min_val = data.min()
        max_val = data.max()
        intensity_range = max_val - min_val
        
        # Calculate intensity values for bright and dark crescents
        bright_min = min_val + intensity_range * self.bright_intensity_range[0]
        bright_max = min_val + intensity_range * self.bright_intensity_range[1]
        
        dark_min = min_val + intensity_range * self.dark_intensity_range[0]
        dark_max = min_val + intensity_range * self.dark_intensity_range[1]
        
        # Sample number of crescents to generate
        num_bright = random.randint(self.num_crescents[0], self.num_crescents[1])
        num_dark = random.randint(self.num_crescents[0], self.num_crescents[1])
        
        if self.per_channel and data.dim() == 4:
            # Apply different beam hardening to each channel
            channels = data.shape[0]
            
            for c in range(channels):
                channel_data = data[c]
                
                # Apply bright crescents
                for _ in range(num_bright):
                    # Generate bright crescent mask
                    mask = self._generate_crescent_mask(channel_data.shape, is_bright=True)
                    
                    # Sample intensity for this crescent
                    intensity = random.uniform(bright_min, bright_max)
                    
                    # Apply with strength
                    result[c] = result[c] * (~mask) + mask * (
                        (1 - self.strength) * result[c] + 
                        self.strength * intensity
                    )
                
                # Apply dark crescents
                for _ in range(num_dark):
                    # Generate dark crescent mask
                    mask = self._generate_crescent_mask(channel_data.shape, is_bright=False)
                    
                    # Sample intensity for this crescent
                    intensity = random.uniform(dark_min, dark_max)
                    
                    # Apply with strength
                    result[c] = result[c] * (~mask) + mask * (
                        (1 - self.strength) * result[c] + 
                        self.strength * intensity
                    )
        else:
            # Apply same beam hardening to all channels
            
            # Apply bright crescents
            for _ in range(num_bright):
                # Generate bright crescent mask
                mask = self._generate_crescent_mask(data.shape, is_bright=True)
                
                # Sample intensity for this crescent
                intensity = random.uniform(bright_min, bright_max)
                
                # Apply with strength based on dimensionality
                if data.dim() == 4:  # Multi-channel data
                    # Expand mask for channel dimension
                    expanded_mask = mask.unsqueeze(0).expand(data.shape[0], -1, -1, -1)
                    
                    # Apply to all channels
                    result = result * (~expanded_mask) + expanded_mask * (
                        (1 - self.strength) * result + 
                        self.strength * intensity
                    )
                else:  # Single-channel data
                    result = result * (~mask) + mask * (
                        (1 - self.strength) * result + 
                        self.strength * intensity
                    )
            
            # Apply dark crescents
            for _ in range(num_dark):
                # Generate dark crescent mask
                mask = self._generate_crescent_mask(data.shape, is_bright=False)
                
                # Sample intensity for this crescent
                intensity = random.uniform(dark_min, dark_max)
                
                # Apply with strength based on dimensionality
                if data.dim() == 4:  # Multi-channel data
                    # Expand mask for channel dimension
                    expanded_mask = mask.unsqueeze(0).expand(data.shape[0], -1, -1, -1)
                    
                    # Apply to all channels
                    result = result * (~expanded_mask) + expanded_mask * (
                        (1 - self.strength) * result + 
                        self.strength * intensity
                    )
                else:  # Single-channel data
                    result = result * (~mask) + mask * (
                        (1 - self.strength) * result + 
                        self.strength * intensity
                    )
        
        return result
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the beam hardening artifacts to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with beam hardening artifacts
        """
        return self._apply_beam_hardening(data)