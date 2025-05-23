import numpy as np
import os
from typing import List, Dict, Tuple
from data.utils import open_zarr


def find_partition_boundaries(accumulator_paths: List[str], patch_size: Tuple[int, int, int]) -> List[Dict]:
    """
    Find regions where partitions need to be merged based on core/expanded regions.
    
    Args:
        accumulator_paths: List of paths to accumulator directories
        patch_size: Tuple (pZ, pY, pX) - not used directly but kept for compatibility
        
    Returns:
        List of boundary dictionaries with overlap information
    """
    boundaries = []
    
    # Sort paths to ensure correct ordering (part_0, part_1, etc.)
    def extract_part_id(path):
        import re
        match = re.search(r'part_(\d+)', path)
        return int(match.group(1)) if match else 0
    
    sorted_paths = sorted(accumulator_paths, key=extract_part_id)
    
    for i in range(len(sorted_paths) - 1):
        current_path = sorted_paths[i]
        next_path = sorted_paths[i + 1]
        
        # Extract partition bounds from accumulator metadata or reconstruct
        current_bounds = get_accumulator_bounds(current_path, i)
        next_bounds = get_accumulator_bounds(next_path, i + 1)
        
        # Check if there's an overlap between partitions
        if current_bounds['expanded_z_end'] > next_bounds['core_z_start']:
            boundaries.append({
                'partition_a': i,
                'partition_b': i + 1,
                'partition_a_path': current_path,
                'partition_b_path': next_path,
                'overlap_region': calculate_overlap_region(current_bounds, next_bounds)
            })
    
    return boundaries


def get_accumulator_bounds(accumulator_path: str, part_id: int) -> Dict:
    """
    Get partition bounds from accumulator metadata or logits zarr attributes.
    
    Args:
        accumulator_path: Path to accumulator directory
        part_id: Partition ID for fallback calculation
        
    Returns:
        Dictionary with partition bounds
    """
    try:
        # Try to read bounds from logits zarr metadata
        logits_path = os.path.join(accumulator_path, 'logits')
        logits_store = open_zarr(logits_path, mode='r')
        
        if hasattr(logits_store, 'attrs'):
            attrs = logits_store.attrs
            return {
                'core_z_start': attrs.get('core_z_start'),
                'core_z_end': attrs.get('core_z_end'),
                'expanded_z_start': attrs.get('expanded_z_start'),
                'expanded_z_end': attrs.get('expanded_z_end'),
                'overlap_size': attrs.get('overlap_size', 0),
                'normalization_factor': attrs.get('normalization_factor', 1.0)
            }
            
    except Exception as e:
        print(f"Warning: Could not read bounds from {accumulator_path}: {e}")
    
    # Fallback: reconstruct bounds from zarr shape and part_id
    # This assumes we can infer the original partitioning scheme
    logits_path = os.path.join(accumulator_path, 'logits')
    logits_store = open_zarr(logits_path, mode='r')
    
    # Get Z dimension from accumulator shape
    expanded_z_size = logits_store.shape[1]  # (classes, Z, Y, X)
    
    # This is a simplified fallback - ideally bounds should be stored in metadata
    return {
        'core_z_start': part_id * expanded_z_size,  # Simplified assumption
        'core_z_end': (part_id + 1) * expanded_z_size,
        'expanded_z_start': part_id * expanded_z_size,
        'expanded_z_end': (part_id + 1) * expanded_z_size,
        'overlap_size': 0
    }


def calculate_overlap_region(bounds_a: Dict, bounds_b: Dict) -> Dict:
    """
    Calculate the overlapping region between two partitions.
    
    Args:
        bounds_a: Bounds for partition A
        bounds_b: Bounds for partition B
        
    Returns:
        Dictionary describing the overlap region
    """
    # Find the overlapping Z region
    overlap_z_start = max(bounds_a['core_z_end'], bounds_b['core_z_start'])
    overlap_z_end = min(bounds_a['expanded_z_end'], bounds_b['expanded_z_end'])
    
    if overlap_z_start >= overlap_z_end:
        # No overlap
        return None
    
    return {
        'z_start': overlap_z_start,
        'z_end': overlap_z_end,
        'z_size': overlap_z_end - overlap_z_start,
        'partition_a_local_z_start': overlap_z_start - bounds_a['expanded_z_start'],
        'partition_a_local_z_end': overlap_z_end - bounds_a['expanded_z_start'],
        'partition_b_local_z_start': overlap_z_start - bounds_b['expanded_z_start'],
        'partition_b_local_z_end': overlap_z_end - bounds_b['expanded_z_start']
    }


class BoundaryMerger:
    """
    Handles merging of overlapping regions between partitioned accumulators.
    """
    
    def __init__(self, volume_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize boundary merger.
        
        Args:
            volume_shape: Original volume shape (Z, Y, X)
            num_classes: Number of output classes
        """
        self.volume_shape = volume_shape
        self.num_classes = num_classes
        
    def merge_partitions(self, accumulator_paths: List[str], final_output_path: str, 
                        patch_size: Tuple[int, int, int], mode: str = 'binary', 
                        threshold: bool = False, verbose: bool = False) -> str:
        """
        Merge partitioned accumulators into final output with finalization.
        
        Args:
            accumulator_paths: List of paths to accumulator directories
            final_output_path: Path for final merged output
            patch_size: Patch size tuple (pZ, pY, pX)
            mode: Output mode ('binary', 'multiclass', 'raw')
            threshold: Whether to apply thresholding
            verbose: Enable verbose output
            
        Returns:
            Path to final merged output
        """
        if verbose:
            print(f"Merging {len(accumulator_paths)} partitioned accumulators...")
        
        # Find boundaries between partitions
        boundaries = find_partition_boundaries(accumulator_paths, patch_size)
        
        if verbose:
            print(f"Found {len(boundaries)} partition boundaries to merge")
        
        # Create final output zarr
        final_shape = (self.num_classes, *self.volume_shape)
        final_store = open_zarr(
            path=final_output_path,
            mode='w',
            shape=final_shape,
            chunks=(1, patch_size[0], patch_size[1], patch_size[2]),
            dtype=np.float32,
            fill_value=0.0
        )
        
        # Copy core regions from each partition (no overlaps)
        for i, acc_path in enumerate(accumulator_paths):
            self._copy_core_region(acc_path, final_store, i, verbose)
        
        # Merge overlapping boundary regions
        for boundary in boundaries:
            if boundary['overlap_region'] is not None:
                self._merge_boundary_region(boundary, final_store, verbose)
        
        # Apply finalization (softmax/argmax) and convert to final format
        if verbose:
            print(f"Applying finalization (mode: {mode}, threshold: {threshold})...")
        
        finalized_output_path = self._finalize_output(final_store, final_output_path, mode, threshold, patch_size, verbose)
        
        if verbose:
            print(f"Processing complete. Final output saved to: {finalized_output_path}")
        
        return finalized_output_path
    
    def _copy_core_region(self, accumulator_path: str, final_store, part_id: int, verbose: bool = False):
        """
        Copy core (non-overlapping) region from partition to final output.
        
        Args:
            accumulator_path: Path to accumulator directory
            final_store: Final output zarr store
            part_id: Partition ID
            verbose: Enable verbose output
        """
        try:
            # Load accumulator core region data
            logits_path = os.path.join(accumulator_path, 'logits')
            logits_store = open_zarr(logits_path, mode='r')
            
            # Get bounds and normalization factor from metadata
            bounds = get_accumulator_bounds(accumulator_path, part_id)
            normalization_factor = bounds.get('normalization_factor', 1.0)
            
            # Extract only the core region (non-overlapping part)
            core_z_local_start = bounds['core_z_start'] - bounds['expanded_z_start']
            core_z_local_end = bounds['core_z_end'] - bounds['expanded_z_start']
            
            core_logits = logits_store[:, core_z_local_start:core_z_local_end, :, :]
            
            # Normalize core region using precomputed factor
            normalized_logits = core_logits / normalization_factor
            
            # Calculate global position for this core region
            z_start = bounds['core_z_start']
            z_end = bounds['core_z_end']
            
            # Copy to final output
            final_store[:, z_start:z_end, :, :] = normalized_logits
            
            if verbose:
                print(f"  Copied core region for partition {part_id}: Z=[{z_start}:{z_end}]")
                
        except Exception as e:
            print(f"Error copying core region for partition {part_id}: {e}")
    
    def _merge_boundary_region(self, boundary: Dict, final_store, verbose: bool = False):
        """
        Merge overlapping boundary region between two partitions.
        
        Args:
            boundary: Boundary information dictionary
            final_store: Final output zarr store
            verbose: Enable verbose output
        """
        try:
            overlap = boundary['overlap_region']
            if overlap is None:
                return
            
            # Load data from both partitions
            acc_a_path = boundary['partition_a_path']
            acc_b_path = boundary['partition_b_path']
            
            logits_a = open_zarr(os.path.join(acc_a_path, 'logits'), mode='r')
            logits_b = open_zarr(os.path.join(acc_b_path, 'logits'), mode='r')
            
            # Get normalization factors from metadata
            bounds_a = get_accumulator_bounds(acc_a_path, boundary['partition_a'])
            bounds_b = get_accumulator_bounds(acc_b_path, boundary['partition_b'])
            norm_factor_a = bounds_a.get('normalization_factor', 1.0)
            norm_factor_b = bounds_b.get('normalization_factor', 1.0)
            
            # Extract overlapping regions
            a_z_start, a_z_end = overlap['partition_a_local_z_start'], overlap['partition_a_local_z_end']
            b_z_start, b_z_end = overlap['partition_b_local_z_start'], overlap['partition_b_local_z_end']
            
            overlap_logits_a = logits_a[:, a_z_start:a_z_end, :, :]
            overlap_logits_b = logits_b[:, b_z_start:b_z_end, :, :]
            
            # Normalize each partition's contribution, then average
            normalized_a = overlap_logits_a / norm_factor_a
            normalized_b = overlap_logits_b / norm_factor_b
            normalized_logits = (normalized_a + normalized_b) / 2.0
            
            # Write to final output
            global_z_start = overlap['z_start']
            global_z_end = overlap['z_end']
            final_store[:, global_z_start:global_z_end, :, :] = normalized_logits
            
            if verbose:
                print(f"  Merged boundary region: Z=[{global_z_start}:{global_z_end}] "
                      f"(partitions {boundary['partition_a']} & {boundary['partition_b']})")
                
        except Exception as e:
            print(f"Error merging boundary region: {e}")
    
    def _finalize_output(self, merged_store, output_path: str, mode: str, threshold: bool, patch_size: tuple, verbose: bool) -> str:
        """
        Apply finalization processing (softmax/argmax) and convert to final uint8 format.
        
        Args:
            merged_store: Merged zarr store with normalized logits
            output_path: Base output path
            mode: Output mode ('binary', 'multiclass', 'raw')
            threshold: Whether to apply thresholding
            verbose: Enable verbose output
            
        Returns:
            Path to finalized output
        """
        import torch
        import torch.nn.functional as F
        
        if mode == 'raw':
            # For raw mode, just return the merged logits as-is
            return output_path
        
        # Process in chunks to avoid memory issues
        chunk_size = patch_size[0]  # Use patch Z size for processing chunks
        total_z = merged_store.shape[1]
        
        # Determine final output shape and dtype
        if mode == 'binary':
            if threshold:
                final_shape = (1, total_z, merged_store.shape[2], merged_store.shape[3])
                final_dtype = np.uint8
            else:
                final_shape = (1, total_z, merged_store.shape[2], merged_store.shape[3])
                final_dtype = np.float32
        elif mode == 'multiclass':
            if threshold:
                final_shape = (1, total_z, merged_store.shape[2], merged_store.shape[3])
                final_dtype = np.uint8
            else:
                final_shape = merged_store.shape  # Keep all class probabilities
                final_dtype = np.float32
        
        # Write directly to the final output path (no intermediate file)
        final_output_path = output_path
        final_store = open_zarr(
            path=final_output_path,
            mode='w',
            shape=final_shape,
            chunks=(1, min(patch_size[0], total_z), patch_size[1], patch_size[2]),
            dtype=final_dtype,
            fill_value=0
        )
        
        # Process in chunks
        for z_start in range(0, total_z, chunk_size):
            z_end = min(z_start + chunk_size, total_z)
            
            if verbose and z_start % (chunk_size * 4) == 0:
                print(f"  Processing slices {z_start}-{z_end}/{total_z}")
            
            # Load chunk of normalized logits
            chunk_logits = merged_store[:, z_start:z_end, :, :]
            
            # Convert to torch for processing
            logits_tensor = torch.from_numpy(chunk_logits)
            
            # Apply finalization based on mode
            if mode == 'binary':
                softmax = F.softmax(logits_tensor, dim=0)
                if threshold:
                    # Binary mask: foreground > background
                    output = (softmax[1] > softmax[0]).unsqueeze(0).numpy().astype(np.uint8)
                else:
                    # Foreground probability
                    output = softmax[1].unsqueeze(0).numpy().astype(np.float32)
                    
            elif mode == 'multiclass':
                if threshold:
                    # Class indices
                    output = torch.argmax(logits_tensor, dim=0, keepdim=True).numpy().astype(np.uint8)
                else:
                    # Class probabilities
                    output = F.softmax(logits_tensor, dim=0).numpy().astype(np.float32)
            
            # Write chunk to final output
            final_store[:, z_start:z_end, :, :] = output
        
        return final_output_path