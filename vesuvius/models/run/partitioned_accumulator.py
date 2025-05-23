import numpy as np
import zarr
import os
import threading
from data.utils import open_zarr




class PartitionedAccumulator:
    """
    Direct accumulator for partitioned inference that eliminates intermediate patch storage.
    Accumulates weighted logits directly during inference using Hann weighting.
    """
    
    def __init__(self, dataset, num_classes, hann_map, output_path, patch_size):
        """
        Initialize accumulator for a partition.
        
        Args:
            dataset: VCDataset instance with partition bounds
            num_classes: Number of output classes
            hann_map: Hann weighting map (C, Z, Y, X) or (Z, Y, X)
            output_path: Path for accumulator zarr stores
            patch_size: Model patch size tuple (pZ, pY, pX) for chunking
        """
        # Get partition bounds from modified VCDataset
        bounds = dataset.get_partition_bounds()
        volume_shape = dataset.input_shape[1:] if len(dataset.input_shape) == 4 else dataset.input_shape
        
        # Store partition information
        self.expanded_z_start = bounds['expanded_z_start']
        self.expanded_z_end = bounds['expanded_z_end']
        self.core_z_start = bounds['core_z_start']
        self.core_z_end = bounds['core_z_end']
        self.overlap_size = bounds['overlap_size']
        
        # Store patch size for chunking
        self.patch_size = patch_size
        
        # Hann windows with 50% overlap form exact partition of unity - no normalization needed
        self.normalization_factor = 1.0
        
        # Accumulator covers EXPANDED region (includes overlaps)
        expanded_z_size = self.expanded_z_end - self.expanded_z_start
        self.accumulator_shape = (num_classes, expanded_z_size, volume_shape[1], volume_shape[2])
        
        # Store Hann map
        if len(hann_map.shape) == 4:  # (1, Z, Y, X) - remove batch dimension
            self.hann_map = hann_map[0]
        else:  # (Z, Y, X)
            self.hann_map = hann_map
            
        # Create zarr stores for logits and weights
        os.makedirs(output_path, exist_ok=True)
        
        self.logits_store = open_zarr(
            path=f"{output_path}/logits",
            mode='w',
            shape=self.accumulator_shape,
            chunks=(1, self.patch_size[0], self.patch_size[1], self.patch_size[2]),
            dtype=np.float32,
            fill_value=0.0
        )
        
        # Store partition bounds in logits zarr metadata
        self.logits_store.attrs['core_z_start'] = self.core_z_start
        self.logits_store.attrs['core_z_end'] = self.core_z_end
        self.logits_store.attrs['expanded_z_start'] = self.expanded_z_start
        self.logits_store.attrs['expanded_z_end'] = self.expanded_z_end
        self.logits_store.attrs['overlap_size'] = self.overlap_size
        self.logits_store.attrs['weighting_scheme'] = 'hann_window'
        
    def accumulate_patch(self, patch_logits, global_coords):
        """
        Accumulate a single patch with Hann weighting.
        
        Args:
            patch_logits: Patch output from model (C, Z, Y, X)
            global_coords: Global coordinates (z, y, x) in original volume
        """
        z_global, y_global, x_global = global_coords
        pZ, pY, pX = patch_logits.shape[1:]  # Skip class dimension
        
        # Convert to local coordinates
        z_local = z_global - self.expanded_z_start
        
        # Check bounds - patch must be within expanded region
        if z_local < 0 or z_local + pZ > (self.expanded_z_end - self.expanded_z_start):
            return  # Patch outside this partition
            
        if y_global < 0 or y_global + pY > self.accumulator_shape[2]:
            return  # Patch outside Y bounds
            
        if x_global < 0 or x_global + pX > self.accumulator_shape[3]:
            return  # Patch outside X bounds
        
        # Apply Hann weighting and accumulate logits
        weighted_logits = patch_logits * self.hann_map[np.newaxis, :, :, :]
        self.logits_store[:, z_local:z_local+pZ, y_global:y_global+pY, x_global:x_global+pX] += weighted_logits
        
    def get_core_region_data(self):
        """
        Extract only the CORE region data (non-overlapping) for final merging.
        
        Returns:
            tuple: (core_logits, core_bounds)
        """
        # Calculate core region in local coordinates
        core_z_local_start = self.core_z_start - self.expanded_z_start
        core_z_local_end = self.core_z_end - self.expanded_z_start
        
        # Extract core region data
        core_logits = self.logits_store[:, core_z_local_start:core_z_local_end, :, :]
        
        # Core bounds in global coordinates
        core_bounds = {
            'z_start': self.core_z_start,
            'z_end': self.core_z_end,
            'y_start': 0,
            'y_end': self.accumulator_shape[2],
            'x_start': 0,
            'x_end': self.accumulator_shape[3]
        }
        
        return core_logits, core_bounds
        
    def finalize(self, mode='binary', threshold=False):
        """
        Apply finalization to accumulated logits (no normalization needed with Hann windows).
        
        Args:
            mode: Output mode ('binary', 'multiclass', 'raw')
            threshold: Whether to apply thresholding for binary mode
            
        Returns:
            Finalized output
        """
        # Read accumulated data - no normalization needed with Hann windows
        logits_data = self.logits_store[:]
        
        # Apply finalization logic directly
        return self._apply_finalization(logits_data, mode, threshold)
        
    def _apply_finalization(self, logits_data, mode, threshold):
        """
        Apply finalization logic (softmax, argmax, etc.) to accumulated logits.
        
        Args:
            logits_data: Accumulated logits array
            mode: Output mode ('binary', 'multiclass', 'raw')
            threshold: Whether to apply thresholding
            
        Returns:
            Finalized output array
        """
        import torch
        import torch.nn.functional as F
        
        if mode == 'raw':
            return logits_data
            
        # Convert to torch for processing
        logits_tensor = torch.from_numpy(logits_data)
        
        if mode == 'binary':
            softmax = F.softmax(logits_tensor, dim=0)
            if threshold:
                output = (softmax[1] > softmax[0]).float().unsqueeze(0)
            else:
                output = softmax[1].unsqueeze(0)  # Foreground probability
        elif mode == 'multiclass':
            if threshold:
                output = torch.argmax(logits_tensor, dim=0, keepdim=True).float()
            else:
                output = F.softmax(logits_tensor, dim=0)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Convert back to numpy
        return output.numpy().astype(np.float32)