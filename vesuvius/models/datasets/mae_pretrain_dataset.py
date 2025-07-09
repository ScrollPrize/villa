import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Queue, Process
from functools import partial
import zarr

from .zarr_dataset import ZarrDataset
from utils.io.patch_cache_utils import load_cached_patches, save_computed_patches
from utils.io.zarr_io import _is_ome_zarr


def _validate_patch_batch(args):
    """
    Worker function for validating patches in parallel.
    
    Parameters
    ----------
    args : tuple
        Contains (positions, data_info, patch_size, is_2d, vol_idx, chunk_idx, total_chunks)
        
    Returns
    -------
    tuple
        (valid_positions, chunk_idx, positions_checked, positions_valid)
    """
    positions, data_info, patch_size, is_2d, vol_idx, chunk_idx, total_chunks = args
    
    # Extract data info
    zarr_path = data_info['zarr_path']
    resolution_level = data_info['resolution_level']
    scale_factor = data_info['scale_factor']
    
    # Open the zarr array at the specified resolution
    try:
        if resolution_level > 0:
            # Directly open the path with resolution level appended
            # This works for both S3 and local paths
            resolution_path = f"{str(zarr_path).rstrip('/')}/{resolution_level}"
            data_array = zarr.open_array(resolution_path, mode='r')
        else:
            data_array = zarr.open_array(str(zarr_path), mode='r')
    except Exception as e:
        print(f"Error opening zarr array at resolution {resolution_level}: {e}")
        return [], chunk_idx, len(positions), 0
    
    valid_positions = []
    positions_checked = 0
    
    # Validate each position with internal progress tracking
    for i, pos in enumerate(positions):
        positions_checked += 1
        
        try:
            if is_2d:
                y, x = pos[1], pos[2]  # pos is [dummy_z, y, x]
                dy, dx = patch_size
                
                # Extract patch at downsampled resolution
                patch = data_array[y:y+dy, x:x+dx]
            else:
                z, y, x = pos
                dz, dy, dx = patch_size
                
                # Extract patch at downsampled resolution
                patch = data_array[z:z+dz, y:y+dy, x:x+dx]
            
            # Check if patch contains non-zero data
            if np.any(patch > 0):
                # Scale position back to full resolution
                if is_2d:
                    full_res_pos = [0, y * scale_factor, x * scale_factor]
                else:
                    full_res_pos = [z * scale_factor, y * scale_factor, x * scale_factor]
                
                valid_positions.append({
                    "volume_index": vol_idx,
                    "position": full_res_pos
                })
                
        except Exception as e:
            # Skip patches that can't be extracted (e.g., at boundaries)
            continue
    
    return valid_positions, chunk_idx, positions_checked, len(valid_positions)


class MAEPretrainDataset(ZarrDataset):
    """Dataset for masked autoencoder pretraining on unlabeled data.
    
    This dataset loads only image data (no labels) and applies random masking
    for self-supervised pretraining using masked autoencoding.
    """
    
    
    def __init__(
        self,
        mgr,
        is_training: bool = True,
        mask_ratio: float = 0.75,
        mask_patch_size: Optional[List[int]] = None,
        normalize_targets: bool = False,
        **kwargs
    ):
        """Initialize MAE pretraining dataset.
        
        Args:
            mgr: ConfigManager instance
            is_training: Whether this is training or validation
            mask_ratio: Fraction of patches to mask (0-1)
            mask_patch_size: Size of patches for masking [H, W] or [D, H, W]
            normalize_targets: Whether to normalize reconstruction targets
            **kwargs: Additional arguments
        """
        # Store MAE-specific parameters - get from config if available
        dataset_config = getattr(mgr, 'dataset_config', {})
        self.mask_ratio = dataset_config.get('mask_ratio', mask_ratio)
        self.min_mask_ratio = dataset_config.get('min_mask_ratio', 0.10)  # Default to 10%
        self.normalize_targets = dataset_config.get('normalize_targets', normalize_targets)
        self._mask_patch_size = dataset_config.get('mask_patch_size', mask_patch_size)
        
        # Get skip_bounding_box from mgr
        self.skip_bounding_box = getattr(mgr, 'skip_bounding_box', False)
        
        # For MAE, we don't skip patch validation - we'll use our custom validation
        mgr.skip_patch_validation = False
        
        # Force minmax normalization for MAE (no intensity sampling needed)
        mgr.normalization_scheme = 'minmax'
        if hasattr(mgr, 'dataset_config'):
            mgr.dataset_config['normalization_scheme'] = 'minmax'
        
        # Set a flag to prevent base class from running validation
        self._mae_will_handle_validation = True
        
        # Call parent init - ZarrDataset will handle MAE mode through _initialize_volumes
        super().__init__(mgr, is_training, **kwargs)
        
        # Set mask patch size based on dimensionality after initialization
        # Determine dimensionality from the dataset
        dim = 2 if self.is_2d_dataset else 3
        
        if self._mask_patch_size is None:
            if dim == 2:
                self.mask_patch_size = [16, 16]
            else:
                self.mask_patch_size = [8, 16, 16]
        else:
            # Ensure mask_patch_size is always a list
            if isinstance(self._mask_patch_size, (list, tuple)):
                self.mask_patch_size = list(self._mask_patch_size)
            else:
                # If it's somehow not a list/tuple, try to extract values
                print(f"Warning: mask_patch_size has unexpected type: {type(self._mask_patch_size)}")
                self.mask_patch_size = [32, 32, 32]  # Default fallback for 3D
            
        # Ensure mask patch size matches dimensionality
        assert len(self.mask_patch_size) == dim, \
            f"Mask patch size dimensionality {len(self.mask_patch_size)} doesn't match data dim {dim}"
        
        # Store dimension for later use
        self.dim = dim
        
        # Initialize counter for skipped zero patches
        self._zero_patch_count = 0
        self._last_reported_count = 0
        
        # Now run our custom validation since parent class skipped it
        self._get_valid_patches()
    
    def _get_config_params(self):
        """Get configuration parameters for caching, including MAE-specific params."""
        base_params = super()._get_config_params()
        base_params['dataset_type'] = 'mae_pretrain'
        base_params['normalization_scheme'] = 'minmax'  # Always use minmax for MAE
        base_params['skip_bounding_box'] = self.skip_bounding_box  # Include skip_bounding_box flag
        base_params['nonzero_validated'] = True  # Flag that patches are pre-validated for non-zero data
        return base_params
    
    
    def _get_label_patch(self, *args, **kwargs):
        """Override to return None since we don't use labels."""
        return None
    
    
    def _compute_data_bounding_box(self, data_array, is_2d):
        """Compute bounding box containing all non-zero data by sampling slices."""
        # For OME-Zarr, use downsampled resolution for faster bounding box computation
        downsampled_array = data_array
        scale_factor = 1
        
        # Check if this is an OME-Zarr with multiple resolutions
        # For OME-Zarr arrays opened with zarr.open_group()[resolution], we need to go up two levels
        if hasattr(data_array, 'store'):
            if hasattr(data_array.store, 'path'):
                # This is a regular zarr array
                store_path = Path(data_array.store.path)
                # If the path ends with a resolution number (0, 1, 2, etc), go up one level
                if store_path.name in ['0', '1', '2', '3', '4', '5']:
                    zarr_path = store_path.parent
                else:
                    zarr_path = store_path
            elif hasattr(data_array, 'path'):
                # Alternative path attribute
                zarr_path = Path(data_array.path)
            else:
                # Try to find the zarr path from the store
                zarr_path = None
                if hasattr(data_array.store, 'dir_path'):
                    zarr_path = Path(data_array.store.dir_path)
                    
            if zarr_path:
                print(f"Checking if {zarr_path} is OME-Zarr...")
                if _is_ome_zarr(zarr_path):
                    print(f"Detected OME-Zarr format")
                    # Try to open resolution level 2 for faster computation
                    try:
                        root = zarr.open_group(str(zarr_path), mode='r')
                        print(f"Available groups in OME-Zarr: {list(root.keys())}")
                        if '2' in root:
                            downsampled_array = root['2']
                            scale_factor = 4  # Resolution 2 is typically 4x downsampled
                            print(f"Using resolution level 2 (4x downsampled) for bounding box computation")
                            print(f"Downsampled shape: {downsampled_array.shape}")
                        elif '1' in root:
                            downsampled_array = root['1']
                            scale_factor = 2  # Resolution 1 is typically 2x downsampled
                            print(f"Using resolution level 1 (2x downsampled) for bounding box computation")
                            print(f"Downsampled shape: {downsampled_array.shape}")
                        else:
                            print(f"No downsampled resolutions found, using full resolution")
                    except Exception as e:
                        # Fall back to original array if we can't access downsampled versions
                        print(f"Error accessing downsampled versions: {e}")
                        pass
                else:
                    print(f"Not an OME-Zarr, using full resolution")
            else:
                print(f"Could not determine zarr path, using full resolution")
        
        # Try to access the downsampled array to verify it works
        if scale_factor > 1:
            try:
                # Test access to the downsampled array
                test_slice = downsampled_array[0]
                print(f"Successfully accessed downsampled array")
            except Exception as e:
                print(f"Error accessing downsampled array: {e}")
                print(f"Falling back to full resolution for bounding box computation")
                downsampled_array = data_array
                scale_factor = 1
        
        shape = downsampled_array.shape
        original_shape = data_array.shape
        
        if is_2d:
            # For 2D: just check where data exists
            print(f"Computing 2D bounding box for array with shape {shape}...")
            
            # Check if there's any data
            nonzero_coords = np.nonzero(downsampled_array)
            if len(nonzero_coords[0]) == 0:
                # No data found
                return {
                    'y_min': 0, 'y_max': shape[0]-1,
                    'x_min': 0, 'x_max': shape[1]-1
                }
            
            y_min = int(nonzero_coords[0].min()) * scale_factor
            y_max = int(nonzero_coords[0].max()) * scale_factor + (scale_factor - 1)
            x_min = int(nonzero_coords[1].min()) * scale_factor
            x_max = int(nonzero_coords[1].max()) * scale_factor + (scale_factor - 1)
            
            # Ensure bounds don't exceed original dimensions
            y_max = min(y_max, original_shape[0] - 1)
            x_max = min(x_max, original_shape[1] - 1)
            
            print(f"Bounding box: Y[{y_min}:{y_max+1}], X[{x_min}:{x_max+1}]")
            return {
                'y_min': y_min, 'y_max': y_max,
                'x_min': x_min, 'x_max': x_max
            }
        else:
            # For 3D: sample every 25 slices
            print(f"Computing 3D bounding box for array with shape {shape}...")
            
            # Initialize bounds - use None to track if we've found any data yet
            z_min, z_max = None, None
            y_min, y_max = None, None
            x_min, x_max = None, None
            
            # Sample every 25 slices
            sample_step = 25
            found_data = False
            
            # Skip first and last 20% of volume
            skip_percent = 0.05
            start_slice = int(shape[0] * skip_percent)
            end_slice = int(shape[0] * (1 - skip_percent))
            
            print(f"Sampling every {sample_step} slices to find bounding box...")
            print(f"Skipping first {skip_percent*100:.0f}% (0-{start_slice}) and last {skip_percent*100:.0f}% ({end_slice}-{shape[0]}) of volume")
            
            # Calculate total slices to sample within the middle 60%
            slices_to_sample = list(range(start_slice, end_slice, sample_step))
            if end_slice - 1 not in slices_to_sample and end_slice - 1 > start_slice:
                slices_to_sample.append(end_slice - 1)
            
            for idx, z in enumerate(slices_to_sample):
                if idx % 10 == 0:  # Print progress every 10 slices
                    print(f"  Checking slice {z}/{shape[0]-1} ({idx+1}/{len(slices_to_sample)} samples)...")
                
                try:
                    slice_data = downsampled_array[z]
                except Exception as e:
                    # Skip this slice if we can't access it
                    continue
                
                # Check if this slice has any data
                if np.any(slice_data > 0):
                    if not found_data:
                        print(f"    First data found at slice {z}")
                    found_data = True
                    
                    # Update Z bounds
                    if z_min is None:
                        z_min = z
                        z_max = z
                    else:
                        z_min = min(z_min, z)
                        z_max = max(z_max, z)
                    
                    # Find Y and X bounds in this slice
                    nonzero_coords = np.nonzero(slice_data)
                    if len(nonzero_coords[0]) > 0:
                        slice_y_min = int(nonzero_coords[0].min()) * scale_factor
                        slice_y_max = int(nonzero_coords[0].max()) * scale_factor + (scale_factor - 1)
                        slice_x_min = int(nonzero_coords[1].min()) * scale_factor
                        slice_x_max = int(nonzero_coords[1].max()) * scale_factor + (scale_factor - 1)
                        
                        # Ensure we don't exceed bounds when scaling
                        slice_y_max = min(slice_y_max, original_shape[1] - 1)
                        slice_x_max = min(slice_x_max, original_shape[2] - 1)
                        
                        if y_min is None:
                            y_min = slice_y_min
                            y_max = slice_y_max
                            x_min = slice_x_min
                            x_max = slice_x_max
                        else:
                            y_min = min(y_min, slice_y_min)
                            y_max = max(y_max, slice_y_max)
                            x_min = min(x_min, slice_x_min)
                            x_max = max(x_max, slice_x_max)
            
            
            if not found_data:
                print("Warning: No data found in sampled slices!")
                return {
                    'z_min': 0, 'z_max': original_shape[0]-1,
                    'y_min': 0, 'y_max': original_shape[1]-1,
                    'x_min': 0, 'x_max': original_shape[2]-1
                }
            
            print(f"\nBounds in downsampled space: Z[{z_min}:{z_max+1}], Y[{y_min}:{y_max+1}], X[{x_min}:{x_max+1}]")
            
            # Scale Z bounds back to original resolution
            z_min = z_min * scale_factor
            z_max = z_max * scale_factor + (scale_factor - 1)
            
            # Ensure bounds don't exceed original dimensions
            z_max = min(z_max, original_shape[0] - 1)
            y_max = min(y_max, original_shape[1] - 1)
            x_max = min(x_max, original_shape[2] - 1)
            
            # Refine Z bounds by checking around the found bounds in the original array
            print(f"\nRefining Z bounds (currently Z[{z_min}:{z_max+1}])...")
            
            # Check backwards from z_min
            refine_start = max(0, z_min - sample_step * scale_factor)
            if refine_start < z_min:
                print(f"  Checking backwards from {z_min} to {refine_start}...")
                for z in range(refine_start, z_min):
                    try:
                        if np.any(data_array[z] > 0):
                            z_min = z
                            print(f"    Found data at Z={z}, updating z_min")
                            break
                    except Exception:
                        # Skip if we can't access this slice
                        continue
            
            # Check forwards from z_max
            refine_end = min(original_shape[0] - 1, z_max + sample_step * scale_factor)
            if refine_end > z_max:
                print(f"  Checking forwards from {z_max} to {refine_end}...")
                for z in range(refine_end, z_max, -1):
                    try:
                        if np.any(data_array[z] > 0):
                            z_max = z
                            print(f"    Found data at Z={z}, updating z_max")
                            break
                    except Exception:
                        # Skip if we can't access this slice
                        continue
            
            print(f"\nBounding box found: Z[{z_min}:{z_max+1}], Y[{y_min}:{y_max+1}], X[{x_min}:{x_max+1}]")
            print(f"Data spans {z_max-z_min+1} slices in Z, {y_max-y_min+1} pixels in Y, {x_max-x_min+1} pixels in X")
            return {
                'z_min': z_min, 'z_max': z_max,
                'y_min': y_min, 'y_max': y_max,
                'x_min': x_min, 'x_max': x_max
            }
    
    def _random_masking(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random masking to image patches.
        
        Args:
            image: Input image array of shape (*spatial_dims) without channel dim
            
        Returns:
            masked_image: Image with masked patches
            mask: Binary mask indicating masked regions (1 = masked, 0 = visible)
        """
        spatial_shape = image.shape
        
        # Store mask_patch_size locally to avoid attribute access issues in worker processes
        # Ensure mask_patch_size is a list, not a dict
        if isinstance(self.mask_patch_size, dict):
            # Handle case where mask_patch_size might be corrupted as a dict
            mask_patch_size = list(self.mask_patch_size.values()) if hasattr(self.mask_patch_size, 'values') else [32, 32, 32]
        else:
            mask_patch_size = list(self.mask_patch_size)  # Ensure it's a list
        mask_ratio = self.mask_ratio
        dim = self.dim
        
        # Calculate number of patches in each dimension
        num_patches_per_dim = []
        for i, (img_size, patch_size) in enumerate(zip(spatial_shape, mask_patch_size)):
            assert img_size % patch_size == 0, \
                f"Image size {img_size} not divisible by patch size {patch_size} in dim {i}"
            num_patches_per_dim.append(img_size // patch_size)
        
        # Total number of patches
        num_patches = np.prod(num_patches_per_dim)
        num_masked = int(num_patches * mask_ratio)
        
        # Random masking pattern
        mask_flat = np.zeros(num_patches, dtype=np.float32)
        masked_indices = np.random.choice(num_patches, num_masked, replace=False)
        mask_flat[masked_indices] = 1.0
        
        # Reshape mask to patch grid
        mask_patches = mask_flat.reshape(tuple(num_patches_per_dim))
        
        # Upsample mask to image resolution
        mask = np.zeros(spatial_shape, dtype=np.float32)
        if dim == 2:
            for i in range(num_patches_per_dim[0]):
                for j in range(num_patches_per_dim[1]):
                    h_start = i * mask_patch_size[0]
                    h_end = (i + 1) * mask_patch_size[0]
                    w_start = j * mask_patch_size[1]
                    w_end = (j + 1) * mask_patch_size[1]
                    mask[h_start:h_end, w_start:w_end] = mask_patches[i, j]
        else:  # 3D
            for i in range(num_patches_per_dim[0]):
                for j in range(num_patches_per_dim[1]):
                    for k in range(num_patches_per_dim[2]):
                        d_start = i * mask_patch_size[0]
                        d_end = (i + 1) * mask_patch_size[0]
                        h_start = j * mask_patch_size[1]
                        h_end = (j + 1) * mask_patch_size[1]
                        w_start = k * mask_patch_size[2]
                        w_end = (k + 1) * mask_patch_size[2]
                        mask[d_start:d_end, h_start:h_end, w_start:w_end] = mask_patches[i, j, k]
        
        # Apply mask to image
        masked_image = image.copy()
        masked_image = masked_image * (1 - mask)  # Zero out masked regions
        
        return masked_image, mask
    
    def _patchwise_normalize(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Normalize image patchwise for better reconstruction targets.
        
        Args:
            image: Input image array
            mask: Binary mask indicating masked regions
            
        Returns:
            Normalized image
        """
        if not self.normalize_targets:
            return image
            
        # Compute statistics only on visible (non-masked) regions
        visible_mask = 1 - mask
        if visible_mask.sum() > 0:
            visible_pixels = image * visible_mask
            mean = visible_pixels.sum() / visible_mask.sum()
            std = np.sqrt(((visible_pixels - mean * visible_mask) ** 2).sum() / visible_mask.sum())
            
            # Normalize
            normalized = (image - mean) / (std + 1e-6)
        else:
            # If everything is masked, just use global normalization
            normalized = (image - image.mean()) / (image.std() + 1e-6)
            
        return normalized
    
    def _get_valid_patches(self):
        """Override to generate and validate patches using multiprocessing."""
        # Check if patches have already been loaded (e.g., by parent class)
        if self.valid_patches:
            print(f"MAE patches already loaded: {len(self.valid_patches)} patches available")
            return
            
        # Try to load from cache first
        if self.cache_enabled and self.cache_dir is not None and self.data_path is not None:
            print("\nAttempting to load MAE patches from cache...")
            config_params = self._get_config_params()
            print(f"MAE cache configuration: {config_params}")
            
            # Generate cache key for debugging
            from utils.io.patch_cache_utils import compute_cache_key
            cache_key = compute_cache_key(config_params)
            print(f"MAE cache key: {cache_key}")
            
            cache_result = load_cached_patches(
                self.cache_dir,
                config_params,
                self.data_path
            )
            if cache_result is not None:
                cached_patches, cached_intensity_properties, cached_normalization_scheme = cache_result
                self.valid_patches = cached_patches
                print(f"Successfully loaded {len(self.valid_patches)} pre-validated MAE patches from cache\n")
                return
            else:
                print("No valid MAE cache found, will compute patches...")
        
        # Generate and validate patches using multiprocessing
        print("Generating and validating MAE patches using multiprocessing...")
        ref_target = list(self.target_volumes.keys())[0]
        total_volumes = len(self.target_volumes[ref_target])
        
        # Determine number of workers
        num_workers = min(multiprocessing.cpu_count(), 8)
        print(f"Using {num_workers} workers for parallel validation")
        
        for vol_idx, volume_info in enumerate(tqdm(self.target_volumes[ref_target], 
                                                    desc="Processing volumes for MAE", 
                                                    total=total_volumes)):
            vdata = volume_info['data']
            data_array = vdata['data']  # This is the image data (zarr array)
            shape = data_array.shape
            is_2d = len(shape) == 2
            
            # Determine zarr path and check if it's OME-Zarr
            zarr_path = None
            resolution_level = 0
            scale_factor = 1
            
            # Get the zarr path from the volume info (stored by zarr_dataset.py)
            zarr_path = volume_info.get('zarr_path', None)
            
            # If not found in volume_info, try to extract from store
            if not zarr_path and hasattr(data_array, 'store'):
                if hasattr(data_array.store, 'path'):
                    store_path_str = str(data_array.store.path)
                    # For remote paths, use the string directly
                    if any(prefix in store_path_str for prefix in ['s3://', 'http://', 'https://']):
                        zarr_path = store_path_str
                        # Remove resolution suffix if present
                        if store_path_str.endswith(('/0', '/1', '/2', '/3', '/4', '/5')):
                            zarr_path = store_path_str.rsplit('/', 1)[0]
                    else:
                        # For local paths, use Path object
                        store_path = Path(store_path_str)
                        if store_path.name in ['0', '1', '2', '3', '4', '5']:
                            zarr_path = str(store_path.parent)
                        else:
                            zarr_path = str(store_path)
            
            # Check if we can use downsampled resolution for validation
            use_downsampled = False
            downsampled_shape = shape  # Default to original shape
            
            # Check if this is an S3/HTTP path
            is_remote_path = False
            if zarr_path:
                # Check if zarr_path itself is remote
                is_remote_path = any(prefix in str(zarr_path) for prefix in ['s3://', 'http://', 'https://'])
                print(f"Volume {vol_idx}: zarr_path = {zarr_path}, is_remote = {is_remote_path}")
            elif hasattr(data_array, 'store') and hasattr(data_array.store, 'path'):
                store_path_str = str(data_array.store.path)
                is_remote_path = any(prefix in store_path_str for prefix in ['s3://', 'http://', 'https://', 's3fs'])
                print(f"Volume {vol_idx}: store_path = {store_path_str}, is_remote = {is_remote_path}")
            
            if zarr_path:
                # For remote paths (S3/HTTP), assume multi-resolution exists and try to use it
                if is_remote_path:
                    try:
                        # For 3D data, always try resolution level 2 (4x downsample)
                        if not is_2d:
                            resolution_level = 2
                            scale_factor = 4
                            # Directly open the path with resolution appended
                            resolution_path = f"{str(zarr_path).rstrip('/')}/2"
                            downsampled_array = zarr.open_array(resolution_path, mode='r')
                            use_downsampled = True
                            downsampled_shape = downsampled_array.shape
                            print(f"\nUsing resolution level 2 (4x downsample) for 3D patch validation (remote path)")
                            print(f"Downsampled shape: {downsampled_shape}")
                        else:
                            # For 2D, try resolution level 1
                            resolution_level = 1
                            scale_factor = 2
                            # Directly open the path with resolution appended
                            resolution_path = f"{str(zarr_path).rstrip('/')}/1"
                            downsampled_array = zarr.open_array(resolution_path, mode='r')
                            use_downsampled = True
                            downsampled_shape = downsampled_array.shape
                            print(f"\nUsing resolution level 1 (2x downsample) for 2D patch validation (remote path)")
                            print(f"Downsampled shape: {downsampled_shape}")
                    except Exception as e:
                        print(f"Could not access downsampled resolutions for remote path: {e}")
                        print(f"Falling back to full resolution")
                        use_downsampled = False
                # For non-S3 paths, check if it's OME-Zarr first
                elif _is_ome_zarr(zarr_path):
                    try:
                        root = zarr.open_group(str(zarr_path), mode='r')
                        available_resolutions = list(root.keys())
                        
                        # Choose resolution based on dimensionality
                        if is_2d and '1' in available_resolutions:
                            resolution_level = 1
                            scale_factor = 2
                            use_downsampled = True
                            downsampled_array = root['1']
                            downsampled_shape = downsampled_array.shape
                            print(f"\nUsing resolution level 1 (2x downsample) for 2D patch validation")
                            print(f"Downsampled shape: {downsampled_shape}")
                        elif not is_2d and '2' in available_resolutions:
                            resolution_level = 2
                            scale_factor = 4
                            use_downsampled = True
                            downsampled_array = root['2']
                            downsampled_shape = downsampled_array.shape
                            print(f"\nUsing resolution level 2 (4x downsample) for 3D patch validation")
                            print(f"Downsampled shape: {downsampled_shape}")
                    except Exception as e:
                        print(f"Could not access downsampled resolutions: {e}")
            
            # Compute bounding box for this volume or use full volume if skip_bounding_box is True
            if self.skip_bounding_box:
                print(f"\nSkipping bounding box computation for volume {vol_idx} (using full volume)...")
                # Use downsampled shape if available for displaying dimensions
                display_shape = downsampled_shape if use_downsampled else shape
                if is_2d:
                    bbox = {
                        'y_min': 0, 'y_max': shape[0] - 1,
                        'x_min': 0, 'x_max': shape[1] - 1
                    }
                    if use_downsampled:
                        print(f"Using full volume at downsampled resolution: Y[0:{display_shape[0]}], X[0:{display_shape[1]}]")
                        print(f"Full resolution dimensions: Y[0:{shape[0]}], X[0:{shape[1]}]")
                    else:
                        print(f"Using full volume: Y[0:{shape[0]}], X[0:{shape[1]}]")
                else:
                    bbox = {
                        'z_min': 0, 'z_max': shape[0] - 1,
                        'y_min': 0, 'y_max': shape[1] - 1,
                        'x_min': 0, 'x_max': shape[2] - 1
                    }
                    if use_downsampled:
                        print(f"Using full volume at downsampled resolution: Z[0:{display_shape[0]}], Y[0:{display_shape[1]}], X[0:{display_shape[2]}]")
                        print(f"Full resolution dimensions: Z[0:{shape[0]}], Y[0:{shape[1]}], X[0:{shape[2]}]")
                    else:
                        print(f"Using full volume: Z[0:{shape[0]}], Y[0:{shape[1]}], X[0:{shape[2]}]")
            else:
                print(f"\nComputing bounding box for volume {vol_idx}...")
                bbox = self._compute_data_bounding_box(data_array, is_2d)
                
                if is_2d:
                    print(f"Bounding box: Y[{bbox['y_min']}:{bbox['y_max']+1}], X[{bbox['x_min']}:{bbox['x_max']+1}]")
                else:
                    print(f"Bounding box: Z[{bbox['z_min']}:{bbox['z_max']+1}], Y[{bbox['y_min']}:{bbox['y_max']+1}], X[{bbox['x_min']}:{bbox['x_max']+1}]")
            
            # Generate sliding window positions
            all_positions = []
            
            if is_2d:
                h, w = self.patch_size[0], self.patch_size[1]
                stride = (h*2, w*2)  # 2x stride (step 2 patch sizes)
                
                # Adjust for downsampled resolution
                if use_downsampled:
                    h_down = h // scale_factor
                    w_down = w // scale_factor
                    stride_down = (h_down*2, w_down*2)  # 2x stride at downsampled resolution
                    y_start = bbox['y_min'] // scale_factor
                    y_end = bbox['y_max'] // scale_factor - h_down + 2
                    x_start = bbox['x_min'] // scale_factor
                    x_end = bbox['x_max'] // scale_factor - w_down + 2
                else:
                    h_down, w_down = h, w
                    stride_down = stride
                    y_start = bbox['y_min']
                    y_end = bbox['y_max'] - h + 2
                    x_start = bbox['x_min']
                    x_end = bbox['x_max'] - w + 2
                
                y_end = max(y_end, y_start + 1)
                x_end = max(x_end, x_start + 1)
                
                for y in range(y_start, y_end, stride_down[0]):
                    for x in range(x_start, x_end, stride_down[1]):
                        all_positions.append([0, y, x])
                        
            else:  # 3D
                d, h, w = self.patch_size
                stride = (d*2, h*2, w*2)  # 2x stride (step 2 patch sizes)
                
                # Adjust for downsampled resolution
                if use_downsampled:
                    d_down = d // scale_factor
                    h_down = h // scale_factor
                    w_down = w // scale_factor
                    stride_down = (d_down*2, h_down*2, w_down*2)  # 2x stride at downsampled resolution
                    z_start = bbox['z_min'] // scale_factor
                    z_end = bbox['z_max'] // scale_factor - d_down + 2
                    y_start = bbox['y_min'] // scale_factor
                    y_end = bbox['y_max'] // scale_factor - h_down + 2
                    x_start = bbox['x_min'] // scale_factor
                    x_end = bbox['x_max'] // scale_factor - w_down + 2
                else:
                    d_down, h_down, w_down = d, h, w
                    stride_down = stride
                    z_start = bbox['z_min']
                    z_end = bbox['z_max'] - d + 2
                    y_start = bbox['y_min']
                    y_end = bbox['y_max'] - h + 2
                    x_start = bbox['x_min']
                    x_end = bbox['x_max'] - w + 2
                
                z_end = max(z_end, z_start + 1)
                y_end = max(y_end, y_start + 1)
                x_end = max(x_end, x_start + 1)
                
                for z in range(z_start, z_end, stride_down[0]):
                    for y in range(y_start, y_end, stride_down[1]):
                        for x in range(x_start, x_end, stride_down[2]):
                            all_positions.append([z, y, x])
            
            print(f"Generated {len(all_positions)} candidate positions")
            
            # Validate patches in parallel
            if use_downsampled and zarr_path:
                # Prepare data for multiprocessing
                data_info = {
                    'zarr_path': zarr_path,
                    'resolution_level': resolution_level,
                    'scale_factor': scale_factor
                }
                
                # Adjust patch size for downsampled resolution
                if is_2d:
                    patch_size_down = [h_down, w_down]
                else:
                    patch_size_down = [d_down, h_down, w_down]
                
                # Split positions into chunks for parallel processing
                chunk_size = max(100, len(all_positions) // (num_workers * 4))
                position_chunks = [all_positions[i:i + chunk_size] 
                                 for i in range(0, len(all_positions), chunk_size)]
                
                # Process chunks in parallel with detailed progress tracking
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all tasks with chunk indices
                    futures = []
                    for chunk_idx, chunk in enumerate(position_chunks):
                        args = (chunk, data_info, patch_size_down, is_2d, vol_idx, chunk_idx, len(position_chunks))
                        futures.append(executor.submit(_validate_patch_batch, args))
                    
                    # Track overall progress
                    total_positions_checked = 0
                    total_valid_found = 0
                    
                    # Collect results with detailed progress bar
                    with tqdm(total=len(all_positions), 
                             desc=f"Validating patches for volume {vol_idx}", 
                             unit="patches",
                             leave=False) as pbar:
                        
                        completed_chunks = 0
                        for future in as_completed(futures):
                            try:
                                valid_positions, chunk_idx, positions_checked, positions_valid = future.result()
                                self.valid_patches.extend(valid_positions)
                                
                                # Update progress
                                total_positions_checked += positions_checked
                                total_valid_found += positions_valid
                                completed_chunks += 1
                                
                                # Update progress bar
                                pbar.update(positions_checked)
                                pbar.set_postfix({
                                    'valid': f"{total_valid_found}/{total_positions_checked}",
                                    'chunks': f"{completed_chunks}/{len(position_chunks)}",
                                    'valid_rate': f"{(total_valid_found/total_positions_checked*100):.1f}%" if total_positions_checked > 0 else "0%"
                                })
                                
                            except Exception as e:
                                print(f"Error in patch validation: {e}")
                                # Still update progress for failed chunk
                                pbar.update(len(position_chunks[0]))  # Approximate
            else:
                # Fallback: validate without multiprocessing if not using downsampled
                print("Validating patches without downsampling (slower)...")
                with tqdm(total=len(all_positions), desc=f"Validating patches for volume {vol_idx}", leave=False) as pbar:
                    for pos in all_positions:
                        try:
                            if is_2d:
                                y, x = pos[1], pos[2]
                                patch = data_array[y:y+h, x:x+w]
                            else:
                                z, y, x = pos
                                patch = data_array[z:z+d, y:y+h, x:x+w]
                            
                            if np.any(patch > 0):
                                self.valid_patches.append({
                                    "volume_index": vol_idx,
                                    "position": pos
                                })
                        except Exception:
                            pass
                        pbar.update(1)
            
            valid_count = len([p for p in self.valid_patches if p['volume_index'] == vol_idx])
            print(f"Found {valid_count} valid non-zero patches in volume {vol_idx} (out of {len(all_positions)} candidates)")
        
        print(f"\nTotal valid patches found: {len(self.valid_patches)}")
        
        # Save to cache after computing all patches
        if self.cache_enabled and self.cache_dir is not None and self.data_path is not None:
            print(f"\nAttempting to save {len(self.valid_patches)} validated MAE patches to cache...")
            config_params = self._get_config_params()
            config_params['nonzero_validated'] = True  # Flag for pre-validated patches
            success = save_computed_patches(
                self.valid_patches,
                self.cache_dir,
                config_params,
                self.data_path,
                intensity_properties=self.intensity_properties,
                normalization_scheme=self.normalization_scheme
            )
            if success:
                print(f"Successfully saved validated MAE patches to cache directory: {self.cache_dir}\n")
            else:
                print("Failed to save MAE patches to cache\n")
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get a training sample with masking applied.
        
        Since patches are pre-validated to contain non-zero data, we don't need
        to check for zero patches anymore.
        
        Args:
            index: Index of the sample
            
        Returns:
            Dictionary containing:
                - image: Original image patch
                - masked_image: Image with masked regions
                - mask: Binary mask (1 = masked, 0 = visible)
        """
        # Get patch info and coordinates
        patch_info = self.valid_patches[index]
        vol_idx = patch_info["volume_index"]
        z, y, x, dz, dy, dx, is_2d = self._extract_patch_coords(patch_info)
        
        # Extract image patch using parent's method (includes normalization)
        img_patch = self._extract_image_patch(vol_idx, z, y, x, dz, dy, dx, is_2d)
        
        # Remove the channel dimension that was added by _extract_image_patch
        img_patch = img_patch[0] if img_patch.shape[0] == 1 else img_patch
        
        # Apply random masking
        masked_img, mask = self._random_masking(img_patch)
        
        # Optionally normalize targets
        if self.normalize_targets:
            img_patch = self._patchwise_normalize(img_patch, mask)
        
        # Add channel dimension back
        img_patch = img_patch[np.newaxis, ...]
        masked_img = masked_img[np.newaxis, ...]
        
        # Create output dict compatible with training pipeline
        # For MAE mode, train.py expects:
        # - "masked_image": the masked input
        # - "image": the original image (for reconstruction target)  
        # - "mask": the mask for loss computation
        data_dict = {
            "masked_image": torch.from_numpy(masked_img.copy()).float(),  # Input to model (masked)
            "image": torch.from_numpy(img_patch.copy()).float(),  # Original image (target)
            "mask": torch.from_numpy(mask[np.newaxis, ...].copy()).float(),  # Mask for loss computation
        }
        
        # Note: We don't apply augmentations in MAE mode because:
        # 1. The masking is already a form of augmentation
        # 2. Standard augmentations might interfere with reconstruction task
        
        return data_dict
    
    def set_mask_ratio(self, mask_ratio: float):
        """Update the mask ratio dynamically during training.
        
        Args:
            mask_ratio: New mask ratio to use (0-1)
        """
        assert 0.0 <= mask_ratio <= 1.0, f"Mask ratio must be between 0 and 1, got {mask_ratio}"
        self.mask_ratio = mask_ratio
    
    def __len__(self) -> int:
        """Return the number of valid patches."""
        return len(self.valid_patches)
