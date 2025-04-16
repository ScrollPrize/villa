import torch
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import map_coordinates
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

class SineWaveDeformation(BasicTransform):
    """
    Applies sine wave deformations to 3D images and corresponding labels.
    Creates random sine waves with varying numbers of peaks and magnitudes.
    """

    def __init__(self, min_peaks=1, max_peaks=5, min_magnitude=0.0, max_magnitude=1.0,
                 random_state=None):
        """
        Initialize the sine wave deformation transform.

        Args:
            min_peaks (int): Minimum number of peaks in the sine waves
            max_peaks (int): Maximum number of peaks in the sine waves
            min_magnitude (float): Minimum magnitude of deformation (0.0 = no deformation)
            max_magnitude (float): Maximum magnitude of deformation (1.0 = 100% of the maximum possible)
            random_state (int, optional): Seed for random number generator
        """
        super().__init__()
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.random_state = random_state

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def get_parameters(self, **data_dict) -> dict:
        """
        Generate random parameters for the sine wave deformation.
        Parameters depend on configuration (single_axis, fixed_axis).

        Returns:
            dict: Parameters for the transformation
        """
        # Determine input dimensions from image if available, otherwise from segmentation
        if data_dict.get('image') is not None:
            tensor = data_dict['image']
        elif data_dict.get('segmentation') is not None:
            tensor = data_dict['segmentation']
        else:
            raise ValueError("Input must contain either 'image' or 'segmentation'")

        # Extract shape information
        shape = tensor.shape
        if len(shape) == 3:  # (C, X, Y)
            ndim = 2
            spatial_dims = shape[1:]
        elif len(shape) == 4:  # (C, X, Y, Z)
            ndim = 3
            spatial_dims = shape[1:]
        else:
            raise ValueError(f"Unsupported tensor shape: {shape}")
            
        # Determine which axis to use
        random_axis = None
        if self.single_axis:
            if self.fixed_axis is not None:
                if self.fixed_axis >= ndim:
                    raise ValueError(f"Fixed axis {self.fixed_axis} is out of bounds for {ndim}D data")
                random_axis = self.fixed_axis
            else:
                # Select one random axis for all waves
                random_axis = torch.randint(0, ndim, (1,)).item()
        
        # Generate number of peaks for each dimension
        num_peaks = {}
        if self.single_axis:
            # Use the same value for all dimensions
            num_peaks_value = torch.randint(self.min_peaks, self.max_peaks + 1, (1,)).item()
            for dim in range(ndim):
                num_peaks[dim] = num_peaks_value
        else:
            # Generate independent values for each dimension
            for dim in range(ndim):
                num_peaks[dim] = torch.randint(self.min_peaks, self.max_peaks + 1, (1,)).item()
            
        # Generate magnitude for each dimension
        magnitude = {}
        if self.single_axis:
            # Use the same value for all dimensions
            magnitude_value = torch.rand(1).item() * (self.max_magnitude - self.min_magnitude) + self.min_magnitude
            for dim in range(ndim):
                magnitude[dim] = magnitude_value
        else:
            # Generate independent values for each dimension
            for dim in range(ndim):
                magnitude[dim] = torch.rand(1).item() * (self.max_magnitude - self.min_magnitude) + self.min_magnitude
            
        # Generate phase for each dimension
        phase = {}
        if self.single_axis:
            # Use the same value for all dimensions
            phase_value = torch.rand(1).item() * 2 * np.pi
            for dim in range(ndim):
                phase[dim] = phase_value
        else:
            # Generate independent values for each dimension
            for dim in range(ndim):
                phase[dim] = torch.rand(1).item() * 2 * np.pi

        return {
            'num_peaks': num_peaks,
            'magnitude': magnitude,
            'phase': phase,
            'spatial_dims': spatial_dims,
            'ndim': ndim,
            'random_axis': random_axis,
            'single_axis': self.single_axis
        }

    def _generate_deformation_field(self, shape, ndim, num_peaks, magnitude, phase, random_axis=None, single_axis=True):
        """
        Generate deformation field based on sine waves.
        Supports both single-axis and multi-axis deformation.

        Args:
            shape (tuple): Spatial dimensions of the input tensor
            ndim (int): Number of spatial dimensions
            num_peaks (dict): Number of peaks for each dimension
            magnitude (dict): Magnitude of deformation for each dimension
            phase (dict): Phase of the sine waves for each dimension
            random_axis (int, optional): The random axis to use for all waves. If None, uses independent axes.
            single_axis (bool): Whether to use single axis mode or original behavior.

        Returns:
            torch.Tensor: Deformation field
        """
        # Create coordinate grids
        coordinates = []
        for dim in range(ndim):
            size = shape[dim]
            coords = torch.linspace(0, 2 * torch.pi, size)  # Using torch.pi instead of np.pi
            coordinates.append(coords)

        # Calculate scaling factors to prevent stretching image beyond its original size
        # We'll use a percentage of the shape to prevent going beyond boundaries
        # The maximum displacement should not exceed 15% of the dimension size
        max_displacements = [0.15 * s for s in shape]

        # Create meshgrid from coordinates
        if ndim == 2:
            x_grid, y_grid = torch.meshgrid(coordinates[0], coordinates[1], indexing='ij')
            grids = [x_grid, y_grid]

            # Create deformation fields for each dimension
            dx = torch.zeros_like(x_grid)
            dy = torch.zeros_like(y_grid)
            deformations = [dx, dy]

            if single_axis and random_axis is not None:
                # Use the specified axis for all deformations
                axis_grid = grids[random_axis]
                
                # Apply waves along the chosen axis
                for dim in range(ndim):
                    if dim != random_axis:  # Only deform non-axis dimensions
                        # Scale the magnitude to prevent exceeding boundaries
                        scaled_magnitude = magnitude[dim] * max_displacements[dim]
                        deformations[dim] += scaled_magnitude * torch.sin(num_peaks[dim] * axis_grid + phase[dim])
            else:
                # Original behavior with different axes
                # Use scaled magnitudes
                scaled_magnitude_x = magnitude[0] * max_displacements[0]
                scaled_magnitude_y = magnitude[1] * max_displacements[1]
                
                dx += scaled_magnitude_x * torch.sin(num_peaks[0] * y_grid + phase[0])
                dy += scaled_magnitude_y * torch.sin(num_peaks[1] * x_grid + phase[1])

            deformation_field = torch.stack(deformations)

        elif ndim == 3:
            x_grid, y_grid, z_grid = torch.meshgrid(coordinates[0], coordinates[1], coordinates[2], indexing='ij')
            grids = [x_grid, y_grid, z_grid]

            # Create deformation fields for each dimension
            dx = torch.zeros_like(x_grid)
            dy = torch.zeros_like(y_grid)
            dz = torch.zeros_like(z_grid)
            deformations = [dx, dy, dz]

            if single_axis and random_axis is not None:
                # Use the specified axis for all deformations
                axis_grid = grids[random_axis]
                
                # Apply waves along the chosen axis
                for dim in range(ndim):
                    if dim != random_axis:  # Only deform non-axis dimensions
                        # Scale the magnitude to prevent exceeding boundaries
                        scaled_magnitude = magnitude[dim] * max_displacements[dim]
                        deformations[dim] += scaled_magnitude * torch.sin(num_peaks[dim] * axis_grid + phase[dim])
            else:
                # Original behavior with different axes
                # Use scaled magnitudes
                scaled_magnitude_x = magnitude[0] * max_displacements[0]
                scaled_magnitude_y = magnitude[1] * max_displacements[1]
                scaled_magnitude_z = magnitude[2] * max_displacements[2]
                
                # Add sine waves along x dimension - with half intensity per contribution
                dx += 0.5 * scaled_magnitude_x * torch.sin(num_peaks[0] * y_grid + phase[0])
                dx += 0.5 * scaled_magnitude_x * torch.sin(num_peaks[0] * z_grid + phase[0])

                # Add sine waves along y dimension
                dy += 0.5 * scaled_magnitude_y * torch.sin(num_peaks[1] * x_grid + phase[1])
                dy += 0.5 * scaled_magnitude_y * torch.sin(num_peaks[1] * z_grid + phase[1])

                # Add sine waves along z dimension
                dz += 0.5 * scaled_magnitude_z * torch.sin(num_peaks[2] * x_grid + phase[2])
                dz += 0.5 * scaled_magnitude_z * torch.sin(num_peaks[2] * y_grid + phase[2])

            deformation_field = torch.stack(deformations)

        return deformation_field

    def __init__(self, min_peaks=1, max_peaks=5, min_magnitude=0.0, max_magnitude=1.0,
                 random_state=None, boundary_mode='constant', constant_value=0.0, 
                 single_axis=True, fixed_axis=None):
        """
        Initialize the sine wave deformation transform.

        Args:
            min_peaks (int): Minimum number of peaks in the sine waves
            max_peaks (int): Maximum number of peaks in the sine waves
            min_magnitude (float): Minimum magnitude of deformation (0.0 = no deformation)
            max_magnitude (float): Maximum magnitude of deformation (1.0 = 100% of the maximum possible)
            random_state (int, optional): Seed for random number generator
            boundary_mode (str): Boundary handling mode.
                Options: 'constant', 'nearest', 'reflect', 'mirror'. Default: 'constant'
            constant_value (float): Fill value when boundary_mode is 'constant'. Default: 0.0
            single_axis (bool): If True, use the same random axis for all waves. If False, use independent axes.
            fixed_axis (int, optional): If provided, always use this specific axis instead of a random one.
                Only used when single_axis=True. Axis is 0-indexed (0=X, 1=Y, 2=Z).
        """
        super().__init__()
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.random_state = random_state
        self.boundary_mode = boundary_mode
        self.constant_value = constant_value
        self.single_axis = single_axis
        self.fixed_axis = fixed_axis

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def _apply_deformation(self, tensor, deformation_field):
        """
        Apply deformation field to the input tensor using PyTorch grid_sample.
        Supports both 2D and 3D data entirely with PyTorch operations.
        Ensures output has exactly the same shape as input.

        Args:
            tensor (torch.Tensor): Input tensor
            deformation_field (torch.Tensor): Deformation field

        Returns:
            torch.Tensor: Deformed tensor with exactly the same shape as input
        """
        # Get shape
        shape = tensor.shape[1:] if len(tensor.shape) > 1 else tensor.shape
        ndim = len(shape)
        
        # Map boundary mode to PyTorch padding mode
        pad_mode = 'zeros'  # Default for 'constant'
        if self.boundary_mode == 'reflect':
            pad_mode = 'reflection'
        elif self.boundary_mode == 'mirror':
            pad_mode = 'reflection'
        elif self.boundary_mode == 'nearest':
            pad_mode = 'border'
        
        # Check if we need to handle non-zero constant values
        use_custom_constant = self.boundary_mode == 'constant' and abs(self.constant_value) > 1e-6
        
        # Create sampling grid based on dimensions
        if ndim == 2:
            # For 2D images (C, H, W)
            h, w = shape
            
            # Create base grid
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=tensor.device),
                torch.arange(w, device=tensor.device),
                indexing='ij'
            )
            
            # Apply deformation field - ensure field doesn't exceed boundaries
            # Clamp values to prevent exceeding shape limits
            deformation_field_y = torch.clamp(deformation_field[1], -h//4, h//4)
            deformation_field_x = torch.clamp(deformation_field[0], -w//4, w//4)
            
            y_coords = y_coords + deformation_field_y
            x_coords = x_coords + deformation_field_x
            
            # Normalize coordinates to [-1, 1]
            y_coords = 2.0 * y_coords / (h - 1) - 1.0
            x_coords = 2.0 * x_coords / (w - 1) - 1.0
            
            # Stack for grid_sample (x, y order for PyTorch)
            grid = torch.stack([x_coords, y_coords], dim=-1)
            
            # Add batch dimension [B, H, W, 2]
            grid = grid.unsqueeze(0)
            
            # Add batch dimension to tensor [B, C, H, W]
            tensor_with_batch = tensor.unsqueeze(0)
            
            # Apply grid_sample
            deformed_tensor = F.grid_sample(
                tensor_with_batch,
                grid,
                mode='bilinear',
                padding_mode=pad_mode,
                align_corners=True
            )
            
            # Handle non-zero constant value (add constant after sampling)
            if use_custom_constant:
                # Create a mask where grid points are outside the input image
                # Values outside [-1, 1] will be padded
                outside_mask = ((grid.abs() > 1).any(dim=-1)).float()
                outside_mask = outside_mask.unsqueeze(1)  # Add channel dimension
                
                # Apply constant value to areas outside the input image
                constant_tensor = torch.ones_like(deformed_tensor) * self.constant_value
                deformed_tensor = deformed_tensor * (1 - outside_mask) + constant_tensor * outside_mask
            
            # Squeeze batch dimension
            deformed_tensor = deformed_tensor.squeeze(0)
            
        elif ndim == 3:
            # For 3D volumes (C, D, H, W)
            d, h, w = shape
            
            # Create base grid
            z_coords, y_coords, x_coords = torch.meshgrid(
                torch.arange(d, device=tensor.device),
                torch.arange(h, device=tensor.device),
                torch.arange(w, device=tensor.device),
                indexing='ij'
            )
            
            # Apply deformation field - ensure field doesn't exceed boundaries
            # Clamp values to prevent exceeding shape limits and causing size mismatches
            deformation_field_z = torch.clamp(deformation_field[2], -d//4, d//4)
            deformation_field_y = torch.clamp(deformation_field[1], -h//4, h//4)
            deformation_field_x = torch.clamp(deformation_field[0], -w//4, w//4)
            
            z_coords = z_coords + deformation_field_z
            y_coords = y_coords + deformation_field_y
            x_coords = x_coords + deformation_field_x
            
            # Normalize coordinates to [-1, 1]
            z_coords = 2.0 * z_coords / (d - 1) - 1.0
            y_coords = 2.0 * y_coords / (h - 1) - 1.0
            x_coords = 2.0 * x_coords / (w - 1) - 1.0
            
            # Stack for grid_sample (x, y, z order for PyTorch)
            grid = torch.stack([x_coords, y_coords, z_coords], dim=-1)
            
            # Add batch dimension [B, D, H, W, 3]
            grid = grid.unsqueeze(0)
            
            # Add batch dimension to tensor [B, C, D, H, W]
            tensor_with_batch = tensor.unsqueeze(0)
            
            # Apply 3D grid_sample
            deformed_tensor = F.grid_sample(
                tensor_with_batch,
                grid,
                mode='bilinear',
                padding_mode=pad_mode,
                align_corners=True
            )
            
            # Handle non-zero constant value (add constant after sampling)
            if use_custom_constant:
                # Create a mask where grid points are outside the input image
                # Values outside [-1, 1] will be padded
                outside_mask = ((grid.abs() > 1).any(dim=-1)).float()
                outside_mask = outside_mask.unsqueeze(1)  # Add channel dimension
                
                # Apply constant value to areas outside the input image
                constant_tensor = torch.ones_like(deformed_tensor) * self.constant_value
                deformed_tensor = deformed_tensor * (1 - outside_mask) + constant_tensor * outside_mask
            
            # Squeeze batch dimension
            deformed_tensor = deformed_tensor.squeeze(0)
        
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")
            
        # Check for shape mismatch and fix if needed
        if deformed_tensor.shape != tensor.shape:
            # Create a new tensor with the original shape
            result = torch.zeros_like(tensor)
            
            # For each dimension, determine how to handle the copied data
            src_slices = []
            dst_slices = []
            for i in range(len(tensor.shape)):
                if deformed_tensor.shape[i] > tensor.shape[i]:
                    # If deformed is larger, we need to crop it
                    src_slices.append(slice(0, tensor.shape[i]))
                    dst_slices.append(slice(0, tensor.shape[i]))
                else:
                    # If deformed is smaller, we copy all of it and leave the rest as zeros
                    src_slices.append(slice(0, deformed_tensor.shape[i]))
                    dst_slices.append(slice(0, deformed_tensor.shape[i]))
            
            # Copy the data that will fit
            result[dst_slices] = deformed_tensor[src_slices]
            return result
            
        return deformed_tensor

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        """
        Apply sine wave deformation to the image.

        Args:
            img (torch.Tensor): Input image
            **params: Parameters from get_parameters

        Returns:
            torch.Tensor: Deformed image with exactly the same shape as input
        """
        # Store original shape
        original_shape = img.shape
        
        # Generate deformation field
        deformation_field = self._generate_deformation_field(
            params['spatial_dims'],
            params['ndim'],
            params['num_peaks'],
            params['magnitude'],
            params['phase'],
            params.get('random_axis'),
            params.get('single_axis', True)  # Pass single_axis parameter
        )

        # Apply deformation
        deformed_img = self._apply_deformation(img, deformation_field)
        
        # Ensure output has the same shape as input
        if deformed_img.shape != original_shape:
            # Create a new tensor with the original shape
            result = torch.zeros_like(img)
            
            # For each dimension, determine how to handle the copied data
            src_slices = []
            dst_slices = []
            for i in range(len(original_shape)):
                if deformed_img.shape[i] > original_shape[i]:
                    # If deformed is larger, we need to crop it
                    src_slices.append(slice(0, original_shape[i]))
                    dst_slices.append(slice(0, original_shape[i]))
                else:
                    # If deformed is smaller, we copy all of it and leave the rest as zeros
                    src_slices.append(slice(0, deformed_img.shape[i]))
                    dst_slices.append(slice(0, deformed_img.shape[i]))
            
            # Copy the data that will fit
            result[dst_slices] = deformed_img[src_slices]
            
            return result
        
        return deformed_img

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        """
        Apply sine wave deformation to the segmentation.

        Args:
            segmentation (torch.Tensor): Input segmentation
            **params: Parameters from get_parameters

        Returns:
            torch.Tensor: Deformed segmentation with exactly the same shape as input
        """
        # Store original shape
        original_shape = segmentation.shape
        
        # Generate deformation field - use the same as for the image to ensure consistency
        deformation_field = self._generate_deformation_field(
            params['spatial_dims'],
            params['ndim'],
            params['num_peaks'],
            params['magnitude'],
            params['phase'],
            params.get('random_axis'),
            params.get('single_axis', True)  # Pass single_axis parameter
        )

        # Apply deformation - ensure segmentation is float type for grid_sample
        # Convert to float for grid_sample (which doesn't support Short/Int types)
        orig_dtype = segmentation.dtype
        seg_float = segmentation.float()
        deformed_seg = self._apply_deformation(seg_float, deformation_field)
        # Convert back to original dtype
        deformed_seg = deformed_seg.to(orig_dtype)
        
        # Ensure output has the same shape as input
        if deformed_seg.shape != original_shape:
            # Create a new tensor with the original shape
            result = torch.zeros_like(segmentation)
            
            # For each dimension, determine how to handle the copied data
            src_slices = []
            dst_slices = []
            for i in range(len(original_shape)):
                if deformed_seg.shape[i] > original_shape[i]:
                    # If deformed is larger, we need to crop it
                    src_slices.append(slice(0, original_shape[i]))
                    dst_slices.append(slice(0, original_shape[i]))
                else:
                    # If deformed is smaller, we copy all of it and leave the rest as zeros
                    src_slices.append(slice(0, deformed_seg.shape[i]))
                    dst_slices.append(slice(0, deformed_seg.shape[i]))
            
            # Copy the data that will fit
            result[dst_slices] = deformed_seg[src_slices]
            
            return result
        
        return deformed_seg

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> torch.Tensor:
        """
        Apply sine wave deformation to the distance map.

        Args:
            dist_map (torch.Tensor): Input distance map
            **params: Parameters from get_parameters

        Returns:
            torch.Tensor: Deformed distance map with exactly the same shape as input
        """
        # Store original shape
        original_shape = dist_map.shape
        
        # Generate deformation field - use the same as for the other components
        deformation_field = self._generate_deformation_field(
            params['spatial_dims'],
            params['ndim'],
            params['num_peaks'],
            params['magnitude'],
            params['phase'],
            params.get('random_axis'),
            params.get('single_axis', True)  # Pass single_axis parameter
        )

        # Apply deformation - ensure dist_map is float type for grid_sample
        # Convert to float for grid_sample (which doesn't support Short/Int types)
        orig_dtype = dist_map.dtype
        dist_map_float = dist_map.float()
        deformed_dist_map = self._apply_deformation(dist_map_float, deformation_field)
        # Convert back to original dtype
        deformed_dist_map = deformed_dist_map.to(orig_dtype)
        
        # Ensure output has the same shape as input
        if deformed_dist_map.shape != original_shape:
            # Create a new tensor with the original shape
            result = torch.zeros_like(dist_map)
            
            # For each dimension, determine how to handle the copied data
            src_slices = []
            dst_slices = []
            for i in range(len(original_shape)):
                if deformed_dist_map.shape[i] > original_shape[i]:
                    # If deformed is larger, we need to crop it
                    src_slices.append(slice(0, original_shape[i]))
                    dst_slices.append(slice(0, original_shape[i]))
                else:
                    # If deformed is smaller, we copy all of it and leave the rest as zeros
                    src_slices.append(slice(0, deformed_dist_map.shape[i]))
                    dst_slices.append(slice(0, deformed_dist_map.shape[i]))
            
            # Copy the data that will fit
            result[dst_slices] = deformed_dist_map[src_slices]
            
            return result
        
        return deformed_dist_map

    def _apply_to_keypoints(self, keypoints, **params):
        """
        Apply sine wave deformation to keypoints.

        Args:
            keypoints: Input keypoints
            **params: Parameters from get_parameters

        Returns:
            Deformed keypoints
        """
        # If keypoints are needed, implement the deformation
        # For now, return unmodified
        return keypoints

    def _apply_to_bbox(self, bbox, **params):
        """
        Apply sine wave deformation to bounding boxes.

        Args:
            bbox: Input bounding boxes
            **params: Parameters from get_parameters

        Returns:
            Deformed bounding boxes
        """
        # If bounding boxes are needed, implement the deformation
        # For now, return unmodified
        return bbox