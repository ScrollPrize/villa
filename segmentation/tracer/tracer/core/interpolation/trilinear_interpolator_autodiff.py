"""PyTorch-native trilinear interpolator using grid_sample."""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn.functional as F
import numpy as np


class TrilinearInterpolatorAutoDiff:
    """
    PyTorch-native trilinear interpolator using grid_sample.
    Fully supports autograd and maintains gradient chains.
    
    This implementation uses PyTorch's built-in grid_sample function
    which is optimized for GPU execution and properly preserves gradient chains.
    
    Coordinate Convention:
    - 3D coordinates are in ZYX order [z, y, x] with 0=z, 1=y, 2=x
    - Volume is indexed as volume[batch, channel, z, y, x]
    - Input coordinates to interpolate should be in ZYX order
    - Gradient outputs are in ZYX order [dz, dy, dx]
    """
    def __init__(
        self, 
        volume: Union[torch.Tensor, np.ndarray, 'Any'],  # 'Any' to include zarr.Array
        batch_size: Optional[int] = None,
        use_cache: bool = False,
        cache_size: int = 1000
    ):
        """
        Initialize the interpolator with a 3D volume.
        
        Args:
            volume: A 3D or 4D tensor/array/zarr-array
            batch_size: Override batch size (ignored, as we use the volume's batch size)
            use_cache: Whether to use caching (ignored in this implementation)
            cache_size: Maximum cache size (ignored in this implementation)
        """
        # Handle zarr arrays - store reference without converting to numpy
        if hasattr(volume, 'ndim') and not isinstance(volume, (np.ndarray, torch.Tensor)):
            # This is likely a zarr array or similar
            self.is_zarr = True
            self.zarr_volume = volume
            # Create a small sample tensor for shape and device information
            sample_shape = (min(2, volume.shape[0]), min(2, volume.shape[1]), min(2, volume.shape[2]))
            volume = torch.from_numpy(np.array(volume[0:sample_shape[0], 0:sample_shape[1], 0:sample_shape[2]], dtype=np.float32))
        # Handle numpy arrays
        elif isinstance(volume, np.ndarray):
            self.is_zarr = False
            volume = torch.from_numpy(volume.astype(np.float32))
        else:
            self.is_zarr = False
            
        # Handle dimensionality
        # For torch tensors, use dim()
        if isinstance(volume, torch.Tensor):
            if volume.dim() == 3:
                self.volume = volume.unsqueeze(0)  # [1, depth, height, width]
            elif volume.dim() == 4:
                self.volume = volume
            else:
                raise ValueError(f"Volume must be 3D or 4D, got shape {volume.shape}")
        # For other array-like objects, use ndim
        else:
            if len(volume.shape) == 3:
                self.volume = torch.tensor(volume).unsqueeze(0)
            elif len(volume.shape) == 4:
                self.volume = torch.tensor(volume)
            else:
                raise ValueError(f"Volume must be 3D or 4D, got shape {volume.shape}")
            
        # Add channel dimension expected by grid_sample
        if self.volume.dim() == 4:  # [batch, depth, height, width]
            self.volume = self.volume.unsqueeze(1)  # [batch, 1, depth, height, width]
            
        self.device = self.volume.device
        self.dtype = self.volume.dtype
        self.batch_size = self.volume.shape[0]
        self.shape = self.volume.shape[2:]  # [depth, height, width]
        
        # Log volume information
        print(f"DEBUG: PyTorch grid_sample interpolator initialized with batch_size={self.batch_size}")
        print(f"DEBUG: Volume shape: {self.volume.shape}")
        
    def evaluate(self, z: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate volume at points (z, y, x) using trilinear interpolation.
        
        Args:
            z: Z coordinates tensor of shape [batch_size, ...] 
            y: Y coordinates tensor of shape [batch_size, ...]
            x: X coordinates tensor of shape [batch_size, ...]
            
        Returns:
            Interpolated values tensor of shape [batch_size, ...]
        """
        # Store original shape
        original_shape = z.shape
        batch_size = original_shape[0]
        
        # For zarr arrays, use direct point sampling with vectorized operations where possible
        if hasattr(self, 'is_zarr') and self.is_zarr:
            # Convert to numpy for indexing
            z_np = z.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            x_np = x.detach().cpu().numpy()
            
            # Prepare result array
            result_np = np.zeros_like(z_np, dtype=np.float32)
            
            # Get volume shape
            z_max, y_max, x_max = self.zarr_volume.shape
            
            # Flatten arrays for vectorized operations
            orig_shape = z_np.shape
            z_flat = z_np.flatten()
            y_flat = y_np.flatten()
            x_flat = x_np.flatten()
            result_flat = np.zeros_like(z_flat, dtype=np.float32)
            
            # Ensure coordinates are within bounds
            z_flat = np.clip(z_flat, 0, z_max - 1.001)
            y_flat = np.clip(y_flat, 0, y_max - 1.001)
            x_flat = np.clip(x_flat, 0, x_max - 1.001)
            
            # Calculate floor indices
            z0 = np.floor(z_flat).astype(np.int32)
            y0 = np.floor(y_flat).astype(np.int32)
            x0 = np.floor(x_flat).astype(np.int32)
            
            # Calculate ceiling indices (ensuring they're within bounds)
            z1 = np.minimum(z0 + 1, z_max - 1)
            y1 = np.minimum(y0 + 1, y_max - 1)
            x1 = np.minimum(x0 + 1, x_max - 1)
            
            # Calculate interpolation weights
            wz = z_flat - z0
            wy = y_flat - y0
            wx = x_flat - x0
            
            # Use chunk-based processing to avoid memory issues
            # Process in chunks of 1000 points at a time
            chunk_size = 1000
            for i in range(0, len(z_flat), chunk_size):
                end_idx = min(i + chunk_size, len(z_flat))
                chunk_indices = slice(i, end_idx)
                
                # Get chunk data
                z0_chunk = z0[chunk_indices]
                y0_chunk = y0[chunk_indices]
                x0_chunk = x0[chunk_indices]
                z1_chunk = z1[chunk_indices]
                y1_chunk = y1[chunk_indices]
                x1_chunk = x1[chunk_indices]
                wz_chunk = wz[chunk_indices]
                wy_chunk = wy[chunk_indices]
                wx_chunk = wx[chunk_indices]
                
                # Process each point in the chunk
                for j in range(end_idx - i):
                    # Get the eight neighboring voxel values
                    v000 = self.zarr_volume[z0_chunk[j], y0_chunk[j], x0_chunk[j]]
                    v001 = self.zarr_volume[z0_chunk[j], y0_chunk[j], x1_chunk[j]]
                    v010 = self.zarr_volume[z0_chunk[j], y1_chunk[j], x0_chunk[j]]
                    v011 = self.zarr_volume[z0_chunk[j], y1_chunk[j], x1_chunk[j]]
                    v100 = self.zarr_volume[z1_chunk[j], y0_chunk[j], x0_chunk[j]]
                    v101 = self.zarr_volume[z1_chunk[j], y0_chunk[j], x1_chunk[j]]
                    v110 = self.zarr_volume[z1_chunk[j], y1_chunk[j], x0_chunk[j]]
                    v111 = self.zarr_volume[z1_chunk[j], y1_chunk[j], x1_chunk[j]]
                    
                    # Perform trilinear interpolation
                    c00 = v000 * (1 - wx_chunk[j]) + v001 * wx_chunk[j]
                    c01 = v010 * (1 - wx_chunk[j]) + v011 * wx_chunk[j]
                    c10 = v100 * (1 - wx_chunk[j]) + v101 * wx_chunk[j]
                    c11 = v110 * (1 - wx_chunk[j]) + v111 * wx_chunk[j]
                    
                    c0 = c00 * (1 - wy_chunk[j]) + c01 * wy_chunk[j]
                    c1 = c10 * (1 - wy_chunk[j]) + c11 * wy_chunk[j]
                    
                    result_flat[i + j] = c0 * (1 - wz_chunk[j]) + c1 * wz_chunk[j]
            
            # Reshape result back to original shape
            result_np = result_flat.reshape(orig_shape)
            
            # Convert back to tensor
            return torch.from_numpy(result_np).to(z.device)
            
        # For torch tensors, use properly differentiable implementation
        else:
            # FIXED: Instead of processing point-by-point, do proper batched processing
            # with autograd-compatible operations to ensure gradient flow
            
            # Find proper range to clamp coordinates
            z_size, y_size, x_size = self.shape
            
            # Reshape inputs to [batch_size, num_points] if needed
            z_flat = z.reshape(batch_size, -1)
            y_flat = y.reshape(batch_size, -1)
            x_flat = x.reshape(batch_size, -1)
            num_points = z_flat.shape[1]
            
            # Clamp coordinates to valid range with a small margin
            z_clamped = torch.clamp(z_flat, 0.001, z_size - 1.001)
            y_clamped = torch.clamp(y_flat, 0.001, y_size - 1.001)
            x_clamped = torch.clamp(x_flat, 0.001, x_size - 1.001)
            
            # Convert to normalized coordinates for grid_sample
            z_norm = 2.0 * (z_clamped / (z_size - 1)) - 1.0
            y_norm = 2.0 * (y_clamped / (y_size - 1)) - 1.0
            x_norm = 2.0 * (x_clamped / (x_size - 1)) - 1.0
            
            # Stack coordinates into a grid tensor - important for gradient tracking
            # Keep operations as tensor ops, no itemization
            grid = torch.stack([z_norm, y_norm, x_norm], dim=-1)
            
            # Reshape grid for grid_sample: [batch, points] -> [batch, 1, 1, points, 3]
            grid = grid.unsqueeze(1).unsqueeze(1) 
            
            # Ensure volume has batch and channel dimensions
            vol = self.volume
            if vol.dim() == 4:  # [batch, d, h, w]
                vol = vol.unsqueeze(1)  # [batch, 1, d, h, w]
                
            # Use grid_sample for the whole batch at once to maintain gradient chain
            try:
                # This preserves gradients through the whole operation
                sampled = F.grid_sample(
                    vol,                     # [batch, 1, d, h, w]
                    grid,                    # [batch, 1, 1, points, 3]
                    mode='bilinear',         # 'bilinear' in 3D is trilinear
                    padding_mode='border',   # Use border value for out-of-bounds
                    align_corners=True       # Scale coordinates properly
                )
                
                # Extract values: [batch, 1, 1, 1, points] -> [batch, points]
                result = sampled.squeeze(1).squeeze(1).squeeze(1)
                
                # Reshape back to match input shape
                return result.reshape(original_shape)
                
            except Exception as e:
                # Log error but don't raise to allow fallback
                print(f"ERROR in grid_sample batch processing: {e}")
                
                # Fallback to older point-by-point approach but maintain gradient
                result = torch.zeros_like(z_flat)
                
                # Handle batch dimensions with modulo
                vol_batch_size = self.volume.shape[0]
                batch_indices = torch.arange(batch_size, device=self.device) % vol_batch_size
                
                # Process each batch separately
                for b in range(batch_size):
                    vol_b = batch_indices[b]  # Volume batch index
                    
                    # Make a copy of the volume for this batch
                    vol = self.volume[vol_b:vol_b+1]
                    
                    # If the volume is exactly 4D [batch, depth, height, width], add channel dim
                    if vol.dim() == 4:
                        vol = vol.unsqueeze(1)  # Make it [batch, channel, depth, height, width]
                    
                    # Use single grid_sample call for all points in batch
                    # Create a grid of all points for this batch
                    batch_grid = torch.stack([
                        z_norm[b], y_norm[b], x_norm[b]
                    ], dim=-1).unsqueeze(0).unsqueeze(0)
                    
                    try:
                        # Sample all points at once
                        sampled = F.grid_sample(
                            vol,                # [1, 1, depth, height, width]
                            batch_grid,         # [1, 1, 1, num_points, 3]
                            mode='bilinear',    # 'bilinear' in 3D is actually trilinear
                            padding_mode='border', # Use border value for out-of-bounds
                            align_corners=True  # Scale coordinates properly
                        )
                        
                        # Extract the values
                        result[b] = sampled.squeeze()
                    except Exception as e:
                        print(f"ERROR in fallback batch sampling: {e}")
                        
                        # Second fallback: point-by-point sampling
                        for i in range(num_points):
                            # Create grid for one point with gradient tracking
                            point_grid = torch.stack([
                                z_norm[b, i:i+1], 
                                y_norm[b, i:i+1], 
                                x_norm[b, i:i+1]
                            ], dim=-1).unsqueeze(0).unsqueeze(0)
                            
                            try:
                                point_sampled = F.grid_sample(
                                    vol, 
                                    point_grid,
                                    mode='bilinear',
                                    padding_mode='border',
                                    align_corners=True
                                )
                                result[b, i] = point_sampled.squeeze()
                            except Exception as e2:
                                print(f"ERROR in point-by-point sampling: {e2}")
                                # Ultimate fallback - use a default value
                                result[b, i] = 0.0
                
                # Reshape to match original input
                return result.reshape(original_shape)
        
    def evaluate_with_gradient(self, z: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate volume at points and compute gradients.
        This implementation preserves the gradient chain completely.
        
        Args:
            z: Z coordinates tensor of shape [batch_size, ...]
            y: Y coordinates tensor of shape [batch_size, ...]
            x: X coordinates tensor of shape [batch_size, ...]
            
        Returns:
            Tuple of (values, gradients):
                values: Interpolated values tensor of shape [batch_size, ...]
                gradients: Gradient tensor of shape [batch_size, ..., 3] in ZYX order
        """
        # For zarr arrays, always use numerical gradient since we can't use autograd
        if hasattr(self, 'is_zarr') and self.is_zarr:
            # Use a small epsilon for numerical stability
            epsilon = 1e-4
            
            # Debug print
            print(f"DEBUG: evaluate_with_gradient for zarr array with shape: {self.zarr_volume.shape}")
            print(f"DEBUG: Input coordinate shapes: z={z.shape}, y={y.shape}, x={x.shape}")
            
            # Forward pass through evaluate to get values
            values = self.evaluate(z, y, x)
            
            # For large batches, compute gradients in smaller chunks to avoid memory issues
            if z.numel() > 10000:  # If total number of elements is large
                # Get original shape
                orig_shape = z.shape
                
                # Flatten tensors
                z_flat = z.reshape(-1)
                y_flat = y.reshape(-1)
                x_flat = x.reshape(-1)
                
                # Initialize gradient tensors
                dz_flat = torch.zeros_like(z_flat)
                dy_flat = torch.zeros_like(y_flat)
                dx_flat = torch.zeros_like(x_flat)
                
                # Process in chunks
                chunk_size = 5000
                for i in range(0, z_flat.numel(), chunk_size):
                    end_idx = min(i + chunk_size, z_flat.numel())
                    idx = slice(i, end_idx)
                    
                    # Get chunk
                    z_chunk = z_flat[idx].reshape(1, -1)
                    y_chunk = y_flat[idx].reshape(1, -1)
                    x_chunk = x_flat[idx].reshape(1, -1)
                    
                    # Compute gradients for chunk
                    z_plus = self.evaluate(z_chunk + epsilon, y_chunk, x_chunk)
                    z_minus = self.evaluate(z_chunk - epsilon, y_chunk, x_chunk)
                    dz_chunk = (z_plus - z_minus) / (2 * epsilon)
                    
                    y_plus = self.evaluate(z_chunk, y_chunk + epsilon, x_chunk)
                    y_minus = self.evaluate(z_chunk, y_chunk - epsilon, x_chunk)
                    dy_chunk = (y_plus - y_minus) / (2 * epsilon)
                    
                    x_plus = self.evaluate(z_chunk, y_chunk, x_chunk + epsilon)
                    x_minus = self.evaluate(z_chunk, y_chunk, x_chunk - epsilon)
                    dx_chunk = (x_plus - x_minus) / (2 * epsilon)
                    
                    # Store results
                    dz_flat[idx] = dz_chunk.reshape(-1)
                    dy_flat[idx] = dy_chunk.reshape(-1)
                    dx_flat[idx] = dx_chunk.reshape(-1)
                
                # Reshape back to original shape
                dz = dz_flat.reshape(orig_shape)
                dy = dy_flat.reshape(orig_shape)
                dx = dx_flat.reshape(orig_shape)
            else:
                # For smaller batches, compute gradients all at once
                # First convert input values to ensure they're tensors
                z_tensor = z if isinstance(z, torch.Tensor) else torch.tensor(z, dtype=torch.float32)
                y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
                x_tensor = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
                
                print(f"DEBUG: Computing Z gradient component")
                z_plus = self.evaluate(z_tensor + epsilon, y_tensor, x_tensor)
                z_minus = self.evaluate(z_tensor - epsilon, y_tensor, x_tensor)
                dz = (z_plus - z_minus) / (2 * epsilon)
                
                print(f"DEBUG: Computing Y gradient component")
                y_plus = self.evaluate(z_tensor, y_tensor + epsilon, x_tensor)
                y_minus = self.evaluate(z_tensor, y_tensor - epsilon, x_tensor)
                dy = (y_plus - y_minus) / (2 * epsilon)
                
                print(f"DEBUG: Computing X gradient component")
                x_plus = self.evaluate(z_tensor, y_tensor, x_tensor + epsilon)
                x_minus = self.evaluate(z_tensor, y_tensor, x_tensor - epsilon)
                dx = (x_plus - x_minus) / (2 * epsilon)
                
                print(f"DEBUG: Gradient components computed successfully")
            
            # Debug print - check gradient components before stacking
            print(f"DEBUG: Gradient components before stacking: dz type={type(dz)}, dy type={type(dy)}, dx type={type(dx)}")
            if isinstance(dz, torch.Tensor):
                print(f"DEBUG: dz shape={dz.shape}")
            if isinstance(dy, torch.Tensor):
                print(f"DEBUG: dy shape={dy.shape}")
            if isinstance(dx, torch.Tensor):
                print(f"DEBUG: dx shape={dx.shape}")
            
            # Stack gradients in ZYX order
            gradients = torch.stack([dz, dy, dx], dim=-1)
            print(f"DEBUG: Gradients after stacking, type={type(gradients)}, shape={gradients.shape}")
            return values, gradients
        
        # For torch tensors, use autograd with properly maintained gradient chain
        else:
            # Store original shapes for reshaping at the end
            original_shape = z.shape

            # We need input tensors that require gradients
            # If inputs already have gradients enabled, use them directly
            # Otherwise, create new tensors with gradients enabled
            if z.requires_grad and y.requires_grad and x.requires_grad:
                z_grad, y_grad, x_grad = z, y, x
            else:
                # Create new tensors with gradients enabled
                z_grad = z.detach().clone().requires_grad_(True)
                y_grad = y.detach().clone().requires_grad_(True)
                x_grad = x.detach().clone().requires_grad_(True)
            
            # Forward pass - this should use our improved grid_sample approach
            # which maintains gradients through the whole computation graph
            values = self.evaluate(z_grad, y_grad, x_grad)
            
            # Prepare for backward pass - we need a grad_output tensor of the same shape
            grad_outputs = torch.ones_like(values)
            
            # Compute gradients - properly maintain gradient chain with create_graph=True
            try:
                # This should work now with our fixed evaluate method
                dz, dy, dx = torch.autograd.grad(
                    outputs=values,
                    inputs=[z_grad, y_grad, x_grad],
                    grad_outputs=grad_outputs,
                    retain_graph=True,  # Keep the graph for potential future backprop
                    create_graph=True,   # Allow higher-order gradients
                    allow_unused=False      # Require all inputs to have gradients
                )
                
                # Stack gradients in ZYX order to match our convention
                gradients = torch.stack([dz, dy, dx], dim=-1)
                
                # Note: Higher-order gradients (gradients of gradients) may not work
                # due to PyTorch limitation with grid_sampler_3d_backward
                
            except Exception as e:
                # There's still a problem with autograd, fall back to numerical approximation
                print(f"WARNING: Autograd gradient calculation failed: {e}")
                print(f"WARNING: Input tensors require_grad: z={z_grad.requires_grad}, y={y_grad.requires_grad}, x={x_grad.requires_grad}")
                print(f"WARNING: Values tensor requires_grad: {values.requires_grad}")
                
                if hasattr(values, 'grad_fn'):
                    print(f"WARNING: Values has grad_fn: {values.grad_fn}")
                else:
                    print(f"WARNING: Values has no grad_fn")
                
                # Special warning for higher-order gradient issue
                if 'grid_sampler_3d_backward' in str(e):
                    print(f"WARNING: PyTorch does not support higher-order gradients for 3D grid sampling.")
                    print(f"WARNING: First-order gradients will work, but gradients of gradients won't.")
                
                print(f"WARNING: Falling back to numerical gradient approximation")
                
                # Use a small epsilon for numerical stability
                epsilon = 1e-4
                
                # Compute numerical gradients
                z_plus = self.evaluate(z_grad + epsilon, y_grad, x_grad)
                z_minus = self.evaluate(z_grad - epsilon, y_grad, x_grad)
                dz = (z_plus - z_minus) / (2 * epsilon)
                
                y_plus = self.evaluate(z_grad, y_grad + epsilon, x_grad)
                y_minus = self.evaluate(z_grad, y_grad - epsilon, x_grad)
                dy = (y_plus - y_minus) / (2 * epsilon)
                
                x_plus = self.evaluate(z_grad, y_grad, x_grad + epsilon)
                x_minus = self.evaluate(z_grad, y_grad, x_grad - epsilon)
                dx = (x_plus - x_minus) / (2 * epsilon)
                
                # Stack gradients in ZYX order
                gradients = torch.stack([dz, dy, dx], dim=-1)
            
            return values, gradients
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the volume at the specified points using trilinear interpolation.
        
        Args:
            points: Points of shape (batch_size, ..., 3) in order (z, y, x)
            
        Returns:
            Interpolated values at the specified points (batch_size, ...)
        """
        # Extract z, y, x coordinates
        z = points[..., 0]
        y = points[..., 1]
        x = points[..., 2]
        
        return self.evaluate(z, y, x)
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return dummy cache stats for backward compatibility."""
        return {
            "cache_enabled": False,
            "size": 0,
            "max_size": 0,
            "hits": 0,
            "misses": 0,
            "hit_ratio": 0.0
        }
        
    def clear_cache(self):
        """Dummy method for backward compatibility."""
        pass