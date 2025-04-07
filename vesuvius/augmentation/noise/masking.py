import torch
import numpy as np
from typing import Union, Optional, Tuple, List, Dict
import random
from ..base import BaseAugmentation, ArrayLike

class BlankRectangle(BaseAugmentation):
    """
    Applies random rectangular masks to 3D volumes.
    
    This augmentation replaces random rectangular regions with the mean intensity
    of the image. It's useful for forcing models to learn from contextual information
    and simulating occlusions or missing data in medical volumes.
    
    The size of the rectangles is specified as a percentage of the input volume
    dimensions, making it adaptable to different input sizes.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        num_rectangles: int = 3,
        min_size: float = 0.05,  # As fraction of volume dimensions
        max_size: float = 0.2,   # As fraction of volume dimensions
        apply_to_label: bool = False,
        fill_mode: str = 'mean',  # 'mean', 'zero', or 'random'
        per_channel: bool = False,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the blank rectangle augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        num_rectangles : int, default=3
            Number of blank rectangles to apply
        min_size : float, default=0.05
            Minimum size of rectangles as a fraction of volume dimensions (0.0-1.0)
        max_size : float, default=0.2
            Maximum size of rectangles as a fraction of volume dimensions (0.0-1.0)
        apply_to_label : bool, default=False
            Whether to apply masking to label data (usually False for segmentation)
        fill_mode : str, default='mean'
            How to fill the masked regions:
            - 'mean': Fill with mean intensity of the image
            - 'zero': Fill with zeros
            - 'random': Fill with random values from the image intensity range
        per_channel : bool, default=False
            Whether to apply different masks for each channel
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.num_rectangles = num_rectangles
        self.min_size = min_size
        self.max_size = max_size
        self.apply_to_label = apply_to_label
        self.fill_mode = fill_mode
        self.per_channel = per_channel
        
        # Validate parameters
        if min_size < 0 or min_size > 1:
            raise ValueError("min_size must be between 0 and 1")
        if max_size < 0 or max_size > 1:
            raise ValueError("max_size must be between 0 and 1")
        if min_size > max_size:
            raise ValueError("min_size must be less than or equal to max_size")
        if fill_mode not in ['mean', 'zero', 'random']:
            raise ValueError("fill_mode must be one of: 'mean', 'zero', 'random'")
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _generate_random_rectangles(self, shape: Tuple[int, ...]) -> List[Dict[str, Tuple[int, int]]]:
        """Generate random rectangle coordinates.
        
        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the input data (D, H, W) or (C, D, H, W)
            
        Returns
        -------
        List[Dict[str, Tuple[int, int]]]
            List of dictionaries with rectangle coordinates
            Each dict contains 'z', 'y', 'x' keys with (start, end) tuples
        """
        # Get spatial dimensions (D, H, W)
        if len(shape) == 4:  # (C, D, H, W)
            depth, height, width = shape[1:]
        else:  # (D, H, W)
            depth, height, width = shape
        
        rectangles = []
        for _ in range(self.num_rectangles):
            # Calculate random rectangle dimensions
            z_size = int(depth * random.uniform(self.min_size, self.max_size))
            y_size = int(height * random.uniform(self.min_size, self.max_size))
            x_size = int(width * random.uniform(self.min_size, self.max_size))
            
            # Ensure rectangle sizes are at least 1
            z_size = max(1, z_size)
            y_size = max(1, y_size)
            x_size = max(1, x_size)
            
            # Calculate random starting positions
            z_start = random.randint(0, max(0, depth - z_size))
            y_start = random.randint(0, max(0, height - y_size))
            x_start = random.randint(0, max(0, width - x_size))
            
            # Calculate end positions
            z_end = min(depth, z_start + z_size)
            y_end = min(height, y_start + y_size)
            x_end = min(width, x_start + x_size)
            
            # Store rectangle coordinates
            rectangles.append({
                'z': (z_start, z_end),
                'y': (y_start, y_end),
                'x': (x_start, x_end)
            })
        
        return rectangles
    
    def _apply_blank_rectangles(self, data: torch.Tensor) -> torch.Tensor:
        """Apply blank rectangles to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Data with blank rectangles
        """
        # Check if the data is a label and if we should apply masking to it
        is_label = (self.label is not None and data is self.label)
        if is_label and not self.apply_to_label:
            return data
        
        # Create a copy of the data
        masked_data = data.clone()
        
        # Determine fill value based on mode
        if self.fill_mode == 'zero':
            fill_value = 0.0
        elif self.fill_mode == 'mean':
            if data.dim() == 4:  # Multi-channel data
                # Compute mean per channel
                fill_value = data.mean(dim=(1, 2, 3)).reshape(-1, 1, 1, 1)
            else:  # Single-channel data
                fill_value = data.mean()
        elif self.fill_mode == 'random':
            # Will be set within the rectangle application loop
            min_val = data.min()
            max_val = data.max()
            fill_value = None
        
        if self.per_channel and data.dim() == 4:
            # Apply different masks to each channel
            channels = data.shape[0]
            for c in range(channels):
                rectangles = self._generate_random_rectangles(data[c].shape)
                
                for rect in rectangles:
                    z_start, z_end = rect['z']
                    y_start, y_end = rect['y']
                    x_start, x_end = rect['x']
                    
                    # For random fill mode, generate new value for each rectangle
                    if self.fill_mode == 'random':
                        fill_value = torch.rand(1, device=self.device) * (max_val - min_val) + min_val
                    
                    # Apply mask
                    masked_data[c, z_start:z_end, y_start:y_end, x_start:x_end] = fill_value
        else:
            # Apply the same masks to all channels
            rectangles = self._generate_random_rectangles(data.shape)
            
            for rect in rectangles:
                z_start, z_end = rect['z']
                y_start, y_end = rect['y']
                x_start, x_end = rect['x']
                
                # For random fill mode, generate new value for each rectangle
                if self.fill_mode == 'random':
                    fill_value = torch.rand(1, device=self.device) * (max_val - min_val) + min_val
                
                # Apply mask
                if data.dim() == 4:  # Multi-channel data
                    masked_data[:, z_start:z_end, y_start:y_end, x_start:x_end] = fill_value
                else:  # Single-channel data
                    masked_data[z_start:z_end, y_start:y_end, x_start:x_end] = fill_value
        
        return masked_data
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the blank rectangle masking to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with masked regions
        """
        return self._apply_blank_rectangles(data)


class Cutout3D(BaseAugmentation):
    """
    Applies 3D Cutout augmentation to volumes.
    
    Cutout is a simple regularization technique that masks out random cubic
    sections of the input volume, encouraging the model to focus on a broader
    range of features. Unlike BlankRectangle, Cutout typically uses zero-filling
    and is applied with a more structured approach.
    
    This is the 3D extension of the 2D Cutout technique commonly used in
    computer vision tasks.
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        n_holes: int = 1,
        cutout_size: Union[int, Tuple[int, int, int]] = 16,
        fill_value: float = 0.0,
        apply_to_label: bool = False,
        per_channel: bool = False,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the 3D Cutout augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        n_holes : int, default=1
            Number of cutout regions to apply
        cutout_size : Union[int, Tuple[int, int, int]], default=16
            Size of the cutout regions. If int, creates cubic regions.
            If tuple, specifies (z_size, y_size, x_size).
        fill_value : float, default=0.0
            Value to fill the cutout regions with
        apply_to_label : bool, default=False
            Whether to apply cutout to label data (usually False for segmentation)
        per_channel : bool, default=False
            Whether to apply different cutouts for each channel
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.n_holes = n_holes
        self.apply_to_label = apply_to_label
        self.per_channel = per_channel
        self.fill_value = fill_value
        
        # Handle cutout_size parameter
        if isinstance(cutout_size, int):
            self.cutout_size = (cutout_size, cutout_size, cutout_size)
        else:
            self.cutout_size = cutout_size
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _apply_cutout(self, data: torch.Tensor) -> torch.Tensor:
        """Apply cutout to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Data with cutout applied
        """
        # Check if the data is a label and if we should apply cutout to it
        is_label = (self.label is not None and data is self.label)
        if is_label and not self.apply_to_label:
            return data
        
        # Create a copy of the data
        masked_data = data.clone()
        
        if self.per_channel and data.dim() == 4:
            # Apply different cutouts to each channel
            channels = data.shape[0]
            for c in range(channels):
                if data.dim() == 4:  # Multi-channel data
                    depth, height, width = data.shape[1:]
                else:  # Single-channel data
                    depth, height, width = data.shape
                
                z_size, y_size, x_size = self.cutout_size
                
                # Ensure cutout size doesn't exceed volume dimensions
                z_size = min(z_size, depth)
                y_size = min(y_size, height)
                x_size = min(x_size, width)
                
                for _ in range(self.n_holes):
                    # Random position of cutout center
                    z = random.randint(0, depth - 1)
                    y = random.randint(0, height - 1)
                    x = random.randint(0, width - 1)
                    
                    # Calculate cutout bounds
                    z1 = max(0, z - z_size // 2)
                    z2 = min(depth, z + z_size // 2)
                    y1 = max(0, y - y_size // 2)
                    y2 = min(height, y + y_size // 2)
                    x1 = max(0, x - x_size // 2)
                    x2 = min(width, x + x_size // 2)
                    
                    # Apply cutout mask
                    masked_data[c, z1:z2, y1:y2, x1:x2] = self.fill_value
        else:
            # Apply the same cutout pattern to all channels
            if data.dim() == 4:  # Multi-channel data
                depth, height, width = data.shape[1:]
            else:  # Single-channel data
                depth, height, width = data.shape
            
            z_size, y_size, x_size = self.cutout_size
            
            # Ensure cutout size doesn't exceed volume dimensions
            z_size = min(z_size, depth)
            y_size = min(y_size, height)
            x_size = min(x_size, width)
            
            for _ in range(self.n_holes):
                # Random position of cutout center
                z = random.randint(0, depth - 1)
                y = random.randint(0, height - 1)
                x = random.randint(0, width - 1)
                
                # Calculate cutout bounds
                z1 = max(0, z - z_size // 2)
                z2 = min(depth, z + z_size // 2)
                y1 = max(0, y - y_size // 2)
                y2 = min(height, y + y_size // 2)
                x1 = max(0, x - x_size // 2)
                x2 = min(width, x + x_size // 2)
                
                # Apply cutout mask
                if data.dim() == 4:  # Multi-channel data
                    masked_data[:, z1:z2, y1:y2, x1:x2] = self.fill_value
                else:  # Single-channel data
                    masked_data[z1:z2, y1:y2, x1:x2] = self.fill_value
        
        return masked_data
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the cutout transformation to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with cutout regions
        """
        return self._apply_cutout(data)