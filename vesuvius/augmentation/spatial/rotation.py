import torch
import numpy as np
from typing import Union, Optional, List, Tuple, Dict, Any
import random
from ..base import BaseAugmentation, ArrayLike
import torch.nn.functional as F


class RandomRotation(BaseAugmentation):
    """
    Applies random rotation to 3D volumes using PyTorch operations.

    This is a GPU-accelerated implementation that works efficiently on CUDA-enabled devices.
    It supports both arbitrary angle rotations and 90-degree rotations.

    Features:
    1. Fully GPU-accelerated using PyTorch tensors and operations
    2. Handles both arbitrary and 90-degree rotations in a single class
    3. Has consistent behavior across different device types
    """

    def __init__(
            self,
            volume: Optional[ArrayLike] = None,
            label: Optional[ArrayLike] = None,
            use_90_rotation: bool = False,  # If True, uses 90-degree rotations
            axes: Optional[List[int]] = None,
            k_values: Optional[List[int]] = None,  # For 90-degree rotations: [1, 2, 3] (90°, 180°, 270°)
            angles: Optional[List[float]] = None,  # For arbitrary rotations
            angle_range: Optional[Tuple[float, float]] = None,  # For arbitrary rotations
            p_per_axis: Optional[List[float]] = None,
            mode: str = 'bilinear',  # Interpolation mode for volume
            mode_label: str = 'nearest',  # Interpolation mode for labels
            device: Optional[Union[str, torch.device]] = 'cuda',
            **kwargs
    ):
        """Initialize the PyTorch-based random rotation augmentation.

        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array or torch tensor
        label : Optional[ArrayLike], default=None
            Input label data as numpy array or torch tensor
        use_90_rotation : bool, default=False
            If True, uses 90-degree rotations (k=1, 2, 3 for 90°, 180°, 270°)
            If False, uses arbitrary angle rotations from angle_range or angles
        axes : Optional[List[int]], default=None
            Axes to rotate around. For 3D data, this can be [0,1,2] for [z,y,x].
            If None, all axes will be considered.
        k_values : Optional[List[int]], default=None
            Values of k for 90-degree rotations to choose from. Default is [1, 2, 3].
            Only used when use_90_rotation=True.
        angles : Optional[List[float]], default=None
            Specific angles in degrees to randomly select from.
            Only used when use_90_rotation=False. Overrides angle_range if provided.
        angle_range : Optional[Tuple[float, float]], default=None
            Range of angles in degrees for random selection.
            Only used when use_90_rotation=False and angles=None.
            Default is (-30, 30) if not provided.
        p_per_axis : Optional[List[float]], default=None
            Probability of rotation for each axis [p_z, p_y, p_x].
            Values should be between 0 and 1. If an axis has probability 0,
            no rotation will be applied around that axis.
            Default is None, which gives equal probability to all axes.
        mode : str, default='bilinear'
            Interpolation mode for volume data in grid_sample ('bilinear', 'nearest', 'bicubic')
        mode_label : str, default='nearest'
            Interpolation mode for label data in grid_sample (default: 'nearest')
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)

        self.use_90_rotation = use_90_rotation

        # For rotations
        if axes is None:
            # Default to all three axes for 3D data
            self.axes = [0, 1, 2]  # z, y, x axes
        else:
            self.axes = axes

        # For 90-degree rotations
        if k_values is None:
            self.k_values = [1, 2, 3]  # 90°, 180°, 270°
        else:
            self.k_values = k_values

        # For arbitrary angle rotations
        self.angles = angles
        if angle_range is None:
            self.angle_range = (-30, 30)  # Default range in degrees
        else:
            self.angle_range = angle_range

        # Store probabilities per axis
        self.p_per_axis = p_per_axis
        if p_per_axis is None:
            self.p_per_axis = [1.0, 1.0, 1.0]  # Default: equal probability for all axes

        # Interpolation modes
        self.mode = mode
        self.mode_label = mode_label

    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the rotation transformation to the data.

        Parameters
        ----------
        data : torch.Tensor
            Input data to transform

        Returns
        -------
        torch.Tensor
            Transformed data
        """
        # Ensure data is a torch tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, device=self.device)

        # Move data to correct device if needed
        if data.device != self.device:
            data = data.to(self.device)

        # Determine interpolation mode (volume vs label)
        is_label = False
        if self.label is not None and id(data) == id(self.label):
            is_label = True
            mode = self.mode_label
        else:
            mode = self.mode

        # Use appropriate rotation method
        if self.use_90_rotation:
            result = self._apply_90_degree_rotation(data)
        else:
            result = self._apply_arbitrary_rotation(data, mode)

        return result

    def _apply_90_degree_rotation(self, data: torch.Tensor) -> torch.Tensor:
        """Apply 90-degree rotation to tensor data.

        Parameters
        ----------
        data : torch.Tensor
            Input data as torch tensor

        Returns
        -------
        torch.Tensor
            Rotated data
        """
        # Handle axis probabilities
        available_axes = []
        for i, axis in enumerate(self.axes):
            if self.p_per_axis[i] > 0 and random.random() < self.p_per_axis[i]:
                available_axes.append(axis)

        # If no rotation should be applied, return original data
        if not available_axes:
            return data

        # Select a random axis from available axes
        axis = random.choice(available_axes)

        # Select a random k value (number of 90-degree rotations)
        k = random.choice(self.k_values)

        # Get data dimensionality
        ndim = len(data.shape)

        # For 3D volumes (assuming order is [channels, z, y, x] or [z, y, x])
        if ndim == 4:
            # For [channels, z, y, x] format, skip channel dimension
            rot_dims = [(2, 3), (1, 3), (1, 2)][axis]  # [(y,x), (z,x), (z,y)]
        elif ndim == 3:
            # For [z, y, x] format
            rot_dims = [(1, 2), (0, 2), (0, 1)][axis]  # [(y,x), (z,x), (z,y)]
        else:
            raise ValueError(f"Unsupported data dimensionality: {ndim}. Expected 3 or 4.")

        # Apply the rotation using torch.rot90
        result = torch.rot90(data, k=k, dims=rot_dims)

        return result

    def _apply_arbitrary_rotation(self, data: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply arbitrary angle rotation to tensor data using PyTorch's grid_sample.

        Parameters
        ----------
        data : torch.Tensor
            Input data as torch tensor
        mode : str
            Interpolation mode for grid_sample

        Returns
        -------
        torch.Tensor
            Rotated data
        """
        # Get data dimensionality and shape
        ndim = len(data.shape)
        original_shape = data.shape

        # Handle different input shapes
        if ndim == 3:  # [D, H, W]
            # Add channel dimension for processing
            data = data.unsqueeze(0)  # [1, D, H, W]
            has_channel_dim = False
        elif ndim == 4:  # [C, D, H, W]
            has_channel_dim = True
        else:
            raise ValueError(f"Unsupported data dimensionality: {ndim}. Expected 3 or 4.")

        # Get the shape after adding channel dimension if needed
        _, depth, height, width = data.shape

        # Determine the rotation angles for each axis
        angles_rad = []
        for i, axis in enumerate(self.axes):
            # Check if this axis should be rotated based on probability
            if random.random() >= self.p_per_axis[i]:
                angles_rad.append(0.0)  # No rotation
                continue

            # Determine the angle
            if self.angles is not None:
                # Choose from the provided angles
                angle = random.choice(self.angles)
            else:
                # Generate random angle within the specified range
                angle = random.uniform(*self.angle_range)

            # Convert to radians
            angles_rad.append(angle * np.pi / 180.0)

        # Pad with zeros for any missing axes
        while len(angles_rad) < 3:
            angles_rad.append(0.0)

        # Create rotation matrices for each axis
        # Rotation around X axis
        R_x = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(torch.tensor(angles_rad[0], device=self.device)),
             -torch.sin(torch.tensor(angles_rad[0], device=self.device))],
            [0, torch.sin(torch.tensor(angles_rad[0], device=self.device)),
             torch.cos(torch.tensor(angles_rad[0], device=self.device))]
        ], device=self.device)

        # Rotation around Y axis
        R_y = torch.tensor([
            [torch.cos(torch.tensor(angles_rad[1], device=self.device)), 0,
             torch.sin(torch.tensor(angles_rad[1], device=self.device))],
            [0, 1, 0],
            [-torch.sin(torch.tensor(angles_rad[1], device=self.device)), 0,
             torch.cos(torch.tensor(angles_rad[1], device=self.device))]
        ], device=self.device)

        # Rotation around Z axis
        R_z = torch.tensor([
            [torch.cos(torch.tensor(angles_rad[2], device=self.device)),
             -torch.sin(torch.tensor(angles_rad[2], device=self.device)), 0],
            [torch.sin(torch.tensor(angles_rad[2], device=self.device)),
             torch.cos(torch.tensor(angles_rad[2], device=self.device)), 0],
            [0, 0, 1]
        ], device=self.device)

        # Combined rotation matrix (Z * Y * X order)
        R = torch.mm(torch.mm(R_z, R_y), R_x)

        # Create a grid of normalized coordinates [-1, 1]
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, depth, device=self.device),
            torch.linspace(-1, 1, height, device=self.device),
            torch.linspace(-1, 1, width, device=self.device),
            indexing='ij'
        )

        # Stack coordinates and reshape for matrix multiplication
        # Note: x, y, z are already in the right order for grid_sample
        grid = torch.stack([x.flatten(), y.flatten(), z.flatten()])

        # Apply rotation to coordinates
        rotated_grid = torch.mm(R, grid)

        # Reshape back to proper dimensions
        x_rotated = rotated_grid[0].reshape(depth, height, width)
        y_rotated = rotated_grid[1].reshape(depth, height, width)
        z_rotated = rotated_grid[2].reshape(depth, height, width)

        # Stack in the proper order for grid_sample (x, y, z)
        grid = torch.stack([x_rotated, y_rotated, z_rotated], dim=3)

        # Add batch dimension for grid_sample
        grid = grid.unsqueeze(0)  # [1, D, H, W, 3]
        data = data.unsqueeze(0)  # [1, C, D, H, W] or [1, 1, D, H, W]

        # Apply grid_sample
        rotated_data = F.grid_sample(
            data,
            grid,
            mode=mode,
            align_corners=True,
            padding_mode='zeros'
        )

        # Remove the batch dimension
        rotated_data = rotated_data.squeeze(0)

        # Remove the channel dimension if the original data didn't have it
        if not has_channel_dim:
            rotated_data = rotated_data.squeeze(0)

        return rotated_data

class ScipyRandomRotation(BaseAugmentation):
    """
    Applies random rotation to 3D volumes using scipy.ndimage.rotate.
    
    This is a more reliable implementation that works on all platforms,
    including those without CUDA support. It supports both arbitrary angle
    rotations and 90-degree rotations.
    
    Unlike the PyTorch-based RandomRotation class, this implementation:
    1. Works reliably on CPU-only systems
    2. Handles both arbitrary and 90-degree rotations in a single class
    3. Has consistent behavior across numpy arrays and torch tensors
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        use_90_rotation: bool = False,  # If True, uses 90-degree rotations
        axes: Optional[List[int]] = None,
        k_values: Optional[List[int]] = None,  # For 90-degree rotations: [1, 2, 3] (90°, 180°, 270°)
        angles: Optional[List[float]] = None,  # For arbitrary rotations
        angle_range: Optional[Tuple[float, float]] = None,  # For arbitrary rotations
        p_per_axis: Optional[List[float]] = None,
        order: int = 1,  # Interpolation order (0=nearest, 1=linear, 3=cubic)
        order_label: int = 0,  # Interpolation order for labels (default: nearest)
        device: Optional[Union[str, torch.device]] = 'cpu',  # Device is ignored but kept for API compatibility
        **kwargs
    ):
        """Initialize the scipy-based random rotation augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        use_90_rotation : bool, default=False
            If True, uses 90-degree rotations (k=1, 2, 3 for 90°, 180°, 270°)
            If False, uses arbitrary angle rotations from angle_range or angles
        axes : Optional[List[int]], default=None
            Axes to rotate around. For 3D data, this can be [0,1,2] for [z,y,x].
            If None, all axes will be considered.
        k_values : Optional[List[int]], default=None
            Values of k for 90-degree rotations to choose from. Default is [1, 2, 3].
            Only used when use_90_rotation=True.
        angles : Optional[List[float]], default=None
            Specific angles in degrees to randomly select from.
            Only used when use_90_rotation=False. Overrides angle_range if provided.
        angle_range : Optional[Tuple[float, float]], default=None
            Range of angles in degrees for random selection.
            Only used when use_90_rotation=False and angles=None.
            Default is (-30, 30) if not provided.
        p_per_axis : Optional[List[float]], default=None
            Probability of rotation for each axis [p_z, p_y, p_x].
            Values should be between 0 and 1. If an axis has probability 0,
            no rotation will be applied around that axis.
            Default is None, which gives equal probability to all axes.
        order : int, default=1
            Interpolation order for volume data (0=nearest, 1=linear, etc.)
        order_label : int, default=0
            Interpolation order for label data (default: nearest neighbor)
        device : Optional[Union[str, torch.device]], default='cpu'
            Device parameter is ignored but kept for API compatibility
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.use_90_rotation = use_90_rotation
        
        # For rotations
        if axes is None:
            # Default to all three axes for 3D data
            self.axes = [0, 1, 2]  # z, y, x axes
        else:
            self.axes = axes
        
        # For 90-degree rotations    
        if k_values is None:
            self.k_values = [1, 2, 3]  # 90°, 180°, 270°
        else:
            self.k_values = k_values
            
        # For arbitrary angle rotations
        self.angles = angles
        if angle_range is None:
            self.angle_range = (-30, 30)  # Default range in degrees
        else:
            self.angle_range = angle_range
            
        # Store probabilities per axis
        self.p_per_axis = p_per_axis
        if p_per_axis is None:
            self.p_per_axis = [1.0, 1.0, 1.0]  # Default: equal probability for all axes
            
        # Interpolation order
        self.order = order
        self.order_label = order_label
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the rotation transformation to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data
        """
        # Convert to numpy array for scipy operations
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            orig_device = data.device
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data
            
        # Ensure the input array is contiguous before any operations
        # This prevents issues with negative strides in subsequent operations
        data_np = np.ascontiguousarray(data_np)
            
        # Determine if we're dealing with a volume or label (for interpolation order)
        is_label = False
        if self.label is not None and data is self.label:
            is_label = True
            interpolation_order = self.order_label
        else:
            interpolation_order = self.order
        
        # Use appropriate rotation method
        if self.use_90_rotation:
            result_np = self._apply_90_degree_rotation(data_np)
        else:
            result_np = self._apply_arbitrary_rotation(data_np, interpolation_order)
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            # Make a contiguous copy to ensure no negative strides
            # This avoids "tensors with negative strides are not currently supported" error
            result_np = np.ascontiguousarray(result_np)
            result = torch.from_numpy(result_np).to(orig_device)
        else:
            result = result_np
            
        return result
    
    def _apply_90_degree_rotation(self, data_np: np.ndarray) -> np.ndarray:
        """Apply 90-degree rotation to numpy array.
        
        Parameters
        ----------
        data_np : np.ndarray
            Input data as numpy array
            
        Returns
        -------
        np.ndarray
            Rotated data
        """
        import random
        
        # Handle axis probabilities
        available_axes = []
        for i, axis in enumerate(self.axes):
            if self.p_per_axis[i] > 0 and random.random() < self.p_per_axis[i]:
                available_axes.append(axis)
        
        # If no rotation should be applied, return original data
        if not available_axes:
            return data_np
            
        # Select a random axis from available axes
        axis = random.choice(available_axes)
        
        # Select a random k value (number of 90-degree rotations)
        k = random.choice(self.k_values)
        
        # Convert axis number to axis pair for np.rot90
        axes_pairs = [(1, 2), (0, 2), (0, 1)]  # [(y,x), (z,x), (z,y)]
        rot_axes = axes_pairs[axis]
        
        # Apply the rotation
        result = np.rot90(data_np, k=k, axes=rot_axes)
        
        # Ensure the result has positive strides (avoid negative stride error)
        return np.ascontiguousarray(result)
    
    def _apply_arbitrary_rotation(self, data_np: np.ndarray, order: int) -> np.ndarray:
        """Apply arbitrary angle rotation to numpy array.
        
        Parameters
        ----------
        data_np : np.ndarray
            Input data as numpy array
        order : int
            Interpolation order
            
        Returns
        -------
        np.ndarray
            Rotated data
        """
        import random
        from scipy.ndimage import rotate
        
        # Make a copy to avoid modifying the original
        result = data_np.copy()
        
        # Process each axis based on probability
        axes_pairs = [(1, 2), (0, 2), (0, 1)]  # [(y,x), (z,x), (z,y)]
        
        # Apply rotations one by one
        for axis, (axes, prob) in enumerate(zip(axes_pairs, self.p_per_axis)):
            # Check if this axis is in the allowed axes
            if axis not in self.axes:
                continue
                
            # Check probability
            if random.random() >= prob:
                continue
                
            # Determine the angle
            if self.angles is not None:
                # Choose from the provided angles
                angle = random.choice(self.angles)
            else:
                # Generate random angle within the specified range
                angle = random.uniform(*self.angle_range)
                
            # Apply rotation (without reshaping to maintain dimensions)
            result = rotate(
                result, 
                angle=angle, 
                axes=axes,
                reshape=False, 
                order=order,
                mode='constant', 
                cval=0
            )
            
            # Ensure result is contiguous after each rotation
            result = np.ascontiguousarray(result)
        
        return result
