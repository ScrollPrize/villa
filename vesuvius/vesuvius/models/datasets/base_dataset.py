from pathlib import Path
import os
import json
import numpy as np
import torch
import fsspec
import zarr
from torch.utils.data import Dataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.ndimage import distance_transform_edt
from skimage.filters import scharr
from vesuvius.models.augmentation.transforms.spatial.transpose import TransposeAxesTransform
# Augmentations will be handled directly in this file
from vesuvius.models.augmentation.transforms.utils.random import RandomTransform
from vesuvius.models.augmentation.helpers.scalar_type import RandomScalar
from vesuvius.models.augmentation.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from vesuvius.models.augmentation.transforms.intensity.contrast import ContrastTransform, BGContrast
from vesuvius.models.augmentation.transforms.intensity.gamma import GammaTransform
from vesuvius.models.augmentation.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from vesuvius.models.augmentation.transforms.noise.gaussian_blur import GaussianBlurTransform
from vesuvius.models.augmentation.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
from vesuvius.models.augmentation.transforms.spatial.spatial import SpatialTransform
from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms
from vesuvius.models.augmentation.transforms.noise.extranoisetransforms import BlankRectangleTransform
from vesuvius.models.augmentation.transforms.intensity.illumination import InhomogeneousSliceIlluminationTransform

from vesuvius.utils.utils import find_mask_patches, find_mask_patches_2d, pad_or_crop_3d, pad_or_crop_2d
from vesuvius.utils.io.patch_cache_utils import (
    get_data_checksums,
    load_cached_patches,
    save_computed_patches,
    save_intensity_properties,
    load_intensity_properties
)
from vesuvius.models.training.normalization import get_normalization
from vesuvius.models.datasets.intensity_sampling import compute_intensity_properties_parallel

class BaseDataset(Dataset):
    """
    A PyTorch Dataset base class for handling both 2D and 3D data from various sources.
    
    Subclasses must implement the _initialize_volumes() method to specify how
    data is loaded from their specific data source.
    """
    def __init__(self,
                 mgr,
                 is_training=True):
        """
        Initialize the dataset with configuration from the manager.
        
        Parameters
        ----------
        mgr : ConfigManager
            Manager containing configuration parameters
        is_training : bool
            Whether this dataset is for training (applies augmentations) or validation

        """
        super().__init__()
        self.mgr = mgr
        self.is_training = is_training

        self.model_name = mgr.model_name
        self.targets = mgr.targets               # e.g. {"ink": {...}, "normals": {...}}
        self.patch_size = mgr.train_patch_size   # Expected to be [z, y, x]
        self.min_labeled_ratio = mgr.min_labeled_ratio
        self.min_bbox_percent = mgr.min_bbox_percent

        # New binarization control parameters
        self.binarize_labels = getattr(mgr, 'binarize_labels', False)
        self.target_value = getattr(mgr, 'target_value', 1)
        self.label_threshold = getattr(mgr, 'label_threshold', None)
        
        # Skip patch validation (defaults to False)
        self.skip_patch_validation = getattr(mgr, 'skip_patch_validation', False)
        
        # Initialize normalization (will be set after computing intensity properties)
        self.normalization_scheme = getattr(mgr, 'normalization_scheme', 'zscore')
        self.intensity_properties = getattr(mgr, 'intensity_properties', {})
        self.normalizer = None  # Will be initialized after volumes are loaded

        self.target_volumes = {}
        self.valid_patches = []
        self.is_2d_dataset = None  
        
        # Store data_path as an attribute for cache handling
        self.data_path = Path(mgr.data_path) if hasattr(mgr, 'data_path') else None
        
        # Cache-related attributes
        self.cache_enabled = getattr(mgr, 'cache_valid_patches', True)
        self.cache_dir = None
        if self.data_path is not None:
            self.cache_dir = self.data_path / '.patches_cache'
            print(f"Cache directory: {self.cache_dir}")
            print(f"Cache enabled: {self.cache_enabled}")
        
        self._initialize_volumes()
        ref_target = list(self.target_volumes.keys())[0]
        ref_volume = self.target_volumes[ref_target][0]['data']['label']
        self.is_2d_dataset = len(ref_volume.shape) == 2
        
        if self.is_2d_dataset:
            print("Detected 2D dataset")
        else:
            print("Detected 3D dataset")
        
        # Try to load intensity properties from JSON file first
        loaded_from_cache = False
        if self.cache_enabled and self.cache_dir is not None and self.normalization_scheme in ['zscore', 'ct'] and not self.intensity_properties:
            # Try to load from separate JSON file
            print("\nChecking for cached intensity properties...")
            intensity_result = load_intensity_properties(self.cache_dir)
            if intensity_result is not None:
                cached_intensity_properties, cached_normalization_scheme = intensity_result
                if cached_normalization_scheme == self.normalization_scheme:
                    self.intensity_properties = cached_intensity_properties
                    self.mgr.intensity_properties = cached_intensity_properties
                    if hasattr(self.mgr, 'dataset_config'):
                        self.mgr.dataset_config['intensity_properties'] = cached_intensity_properties
                    print("\nLoaded intensity properties from JSON cache - skipping computation")
                    print("Cached intensity properties:")
                    for key, value in cached_intensity_properties.items():
                        print(f"  {key}: {value:.4f}")
                    loaded_from_cache = True
                else:
                    print(f"Cached normalization scheme '{cached_normalization_scheme}' doesn't match current '{self.normalization_scheme}'")
            
        # Also check patches cache for backward compatibility
        if self.cache_enabled and self.cache_dir is not None and self.data_path is not None and not loaded_from_cache:
            config_params = self._get_config_params()
            cache_result = load_cached_patches(
                self.cache_dir,
                config_params,
                self.data_path
            )
            if cache_result is not None:
                cached_patches, cached_intensity_properties, cached_normalization_scheme = cache_result
                # If we have cached intensity properties and don't already have them, use the cached ones
                if cached_intensity_properties and not self.intensity_properties:
                    self.intensity_properties = cached_intensity_properties
                    self.mgr.intensity_properties = cached_intensity_properties
                    if hasattr(self.mgr, 'dataset_config'):
                        self.mgr.dataset_config['intensity_properties'] = cached_intensity_properties
                    print("\nLoaded intensity properties from patches cache - skipping computation")
                    print("Cached intensity properties:")
                    for key, value in cached_intensity_properties.items():
                        print(f"  {key}: {value:.4f}")
                    loaded_from_cache = True
                    # Also save to JSON for visibility
                    save_intensity_properties(self.cache_dir, cached_intensity_properties, cached_normalization_scheme or self.normalization_scheme)
        
        # Compute intensity properties if not provided and not loaded from cache
        if self.normalization_scheme in ['zscore', 'ct'] and not self.intensity_properties and not loaded_from_cache:
            print(f"\nComputing intensity properties for {self.normalization_scheme} normalization...")
            self.intensity_properties = compute_intensity_properties_parallel(self.target_volumes, sample_ratio=0.001, max_samples=1000000)
            # Update the config manager with computed properties
            if hasattr(self.mgr, 'intensity_properties'):
                self.mgr.intensity_properties = self.intensity_properties
                self.mgr.dataset_config['intensity_properties'] = self.intensity_properties
            
            # Save to separate JSON file for visibility
            if self.cache_enabled and self.cache_dir is not None:
                save_intensity_properties(self.cache_dir, self.intensity_properties, self.normalization_scheme)
        
        # Now initialize the normalizer with the computed or provided properties
        self.normalizer = get_normalization(self.normalization_scheme, self.intensity_properties)
        
        # Initialize transforms for training
        self.transforms = None
        if self.is_training:
            self.transforms = self._create_training_transforms()
            print("Training transforms initialized")
        
        # Only run validation if not handled by subclass (e.g., MAEPretrainDataset)
        if not getattr(self, '_mae_will_handle_validation', False):
            self._get_valid_patches()

    def _initialize_volumes(self):
        """
        Initialize volumes from the data source.
        
        This method must be implemented by subclasses to specify how
        data is loaded from their specific data source (napari, TIFs, Zarr, etc.).
        
        The implementation should populate self.target_volumes in the format:
        {
            'target_name': [
                {
                    'data': {
                        'data': numpy_array,      # Image data
                        'label': numpy_array,     # Label data  
                        'mask': numpy_array       # Mask data (optional)
                    }
                },
                ...  # Additional volumes for this target
            ],
            ...  # Additional targets
        }
        """
        raise NotImplementedError("Subclasses must implement _initialize_volumes() method")

    def _get_all_sliding_window_positions(self, volume_shape, patch_size, stride=None):
        """
        Generate all possible sliding window positions for a volume.
        
        Parameters
        ----------
        volume_shape : tuple
            Shape of the volume (2D or 3D)
        patch_size : tuple
            Size of patches to extract
        stride : tuple, optional
            Stride for sliding window, defaults to 50% overlap
            
        Returns
        -------
        list
            List of positions as dictionaries with 'start_pos' key
        """
        if len(volume_shape) == 2:
            # 2D case
            H, W = volume_shape
            h, w = patch_size
            
            if stride is None:
                stride = (h // 2, w // 2)  # 50% overlap by default
            
            positions = []
            # Calculate total iterations for progress bar
            y_positions = list(range(0, H - h + 1, stride[0]))
            total_positions = len(y_positions) * len(range(0, W - w + 1, stride[1]))
            
            with tqdm(total=total_positions, desc="Generating 2D sliding window positions", leave=False) as pbar:
                for y in y_positions:
                    for x in range(0, W - w + 1, stride[1]):
                        positions.append({
                            'start_pos': [0, y, x]  # [dummy_z, y, x] for 2D
                        })
                        pbar.update(1)
            
            # Ensure we cover the edges
            # Add patches at the bottom edge if needed
            if H - h > positions[-1]['start_pos'][1]:
                for x in range(0, W - w + 1, stride[1]):
                    positions.append({
                        'start_pos': [0, H - h, x]
                    })
            
            # Add patches at the right edge if needed
            if W - w > positions[-1]['start_pos'][2]:
                for y in range(0, H - h + 1, stride[0]):
                    positions.append({
                        'start_pos': [0, y, W - w]
                    })
            
            # Add the bottom-right corner if needed
            if H - h > positions[-1]['start_pos'][1] or W - w > positions[-1]['start_pos'][2]:
                positions.append({
                    'start_pos': [0, H - h, W - w]
                })
                
        else:
            # 3D case
            D, H, W = volume_shape
            d, h, w = patch_size
            
            if stride is None:
                stride = (d // 2, h // 2, w // 2)  # 50% overlap by default
            
            positions = []
            # Calculate total iterations for progress bar
            z_positions = list(range(0, D - d + 1, stride[0]))
            y_positions = list(range(0, H - h + 1, stride[1]))
            x_positions = list(range(0, W - w + 1, stride[2]))
            total_positions = len(z_positions) * len(y_positions) * len(x_positions)
            
            with tqdm(total=total_positions, desc="Generating 3D sliding window positions", leave=False) as pbar:
                for z in z_positions:
                    for y in y_positions:
                        for x in x_positions:
                            positions.append({
                                'start_pos': [z, y, x]
                            })
                            pbar.update(1)
            
            # Ensure we cover the edges in 3D
            # This is more complex but follows the same principle
            # Add patches to cover the edges if the stride doesn't naturally cover them
            if D - d > 0 and (D - d) % stride[0] != 0:
                for y in range(0, H - h + 1, stride[1]):
                    for x in range(0, W - w + 1, stride[2]):
                        positions.append({
                            'start_pos': [D - d, y, x]
                        })
            
            if H - h > 0 and (H - h) % stride[1] != 0:
                for z in range(0, D - d + 1, stride[0]):
                    for x in range(0, W - w + 1, stride[2]):
                        positions.append({
                            'start_pos': [z, H - h, x]
                        })
            
            if W - w > 0 and (W - w) % stride[2] != 0:
                for z in range(0, D - d + 1, stride[0]):
                    for y in range(0, H - h + 1, stride[1]):
                        positions.append({
                            'start_pos': [z, y, W - w]
                        })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_positions = []
        for pos in positions:
            pos_tuple = tuple(pos['start_pos'])
            if pos_tuple not in seen:
                seen.add(pos_tuple)
                unique_positions.append(pos)
        
        return unique_positions

    def _get_config_params(self):
        """
        Get configuration parameters for caching.
        
        Returns
        -------
        dict
            Configuration parameters
        """
        return {
            'patch_size': self.patch_size,
            'min_labeled_ratio': self.min_labeled_ratio,
            'min_bbox_percent': self.min_bbox_percent,
            'skip_patch_validation': self.skip_patch_validation,
            'targets': sorted(self.targets.keys()),
            'is_2d_dataset': self.is_2d_dataset
        }
    
    def _load_approved_patches(self):
        """
        Load approved patches from vc_proofreader export file.
        
        Returns
        -------
        bool
            True if patches were successfully loaded, False otherwise
        """
        approved_patches_file = Path(self.mgr.approved_patches_file)
        
        if not approved_patches_file.exists():
            print(f"Approved patches file not found: {approved_patches_file}")
            return False
        
        try:
            with open(approved_patches_file, 'r') as f:
                data = json.load(f)
            
            # Validate the file format
            if 'metadata' not in data or 'approved_patches' not in data:
                print(f"Invalid approved patches file format: missing required keys")
                return False
            
            metadata = data['metadata']
            approved_patches = data['approved_patches']
            
            # Validate patch size compatibility
            if metadata.get('patch_size') != self.patch_size:
                print(f"Warning: Patch size mismatch. Expected {self.patch_size}, got {metadata.get('patch_size')}")
                # Could choose to continue or return False depending on requirements
            
            # Convert approved patches to the format expected by BaseDataset
            self.valid_patches = []
            for patch_info in approved_patches:
                self.valid_patches.append({
                    "volume_index": patch_info.get("volume_index", 0),
                    "position": patch_info["coords"]
                })
            
            print(f"Successfully loaded {len(self.valid_patches)} approved patches from {approved_patches_file}")
            print(f"Approved patches metadata:")
            print(f"  - Patch size: {metadata.get('patch_size')}")
            print(f"  - Coordinate system: {metadata.get('coordinate_system', 'unknown')}")
            print(f"  - Total approved: {metadata.get('total_approved', len(approved_patches))}")
            print(f"  - Export timestamp: {metadata.get('export_timestamp', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"Error loading approved patches file: {e}")
            return False
    
    def _get_valid_patches(self):
        """Find valid patches based on mask coverage and labeled ratio requirements."""
        # Check if we should load approved patches from vc_proofreader
        if hasattr(self.mgr, 'approved_patches_file') and self.mgr.approved_patches_file:
            if self._load_approved_patches():
                return
        
        # Try to load from cache first
        if self.cache_enabled and self.cache_dir is not None and self.data_path is not None:
            print("\nAttempting to load patches from cache...")
            config_params = self._get_config_params()
            print(f"Cache configuration: {config_params}")
            cache_result = load_cached_patches(
                self.cache_dir,
                config_params,
                self.data_path
            )
            if cache_result is not None:
                cached_patches, cached_intensity_properties, cached_normalization_scheme = cache_result
                self.valid_patches = cached_patches
                print(f"Successfully loaded {len(self.valid_patches)} patches from cache\n")
                
                # Load cached intensity properties if available and not already set
                if cached_intensity_properties and not self.intensity_properties:
                    self.intensity_properties = cached_intensity_properties
                    self.mgr.intensity_properties = cached_intensity_properties
                    if hasattr(self.mgr, 'dataset_config'):
                        self.mgr.dataset_config['intensity_properties'] = cached_intensity_properties
                    print("Loaded intensity properties from cache - skipping computation")
                    print("Cached intensity properties:")
                    for key, value in cached_intensity_properties.items():
                        print(f"  {key}: {value:.4f}")
                    
                return
            else:
                print("No valid cache found, will compute patches...")
            
        # If no valid cache, compute patches
        print("Computing valid patches...")
        ref_target = list(self.target_volumes.keys())[0]
        total_volumes = len(self.target_volumes[ref_target])

        for vol_idx, volume_info in enumerate(tqdm(self.target_volumes[ref_target], 
                                                    desc="Processing volumes", 
                                                    total=total_volumes)):
            vdata = volume_info['data']
            is_2d = len(vdata['label'].shape) == 2
            
            if self.skip_patch_validation:
                # Skip validation - generate all sliding window positions
                print(f"Skipping patch validation for volume {vol_idx} - using all sliding window positions")
                
                volume_shape = vdata['label'].shape
                
                # Handle multi-channel labels (e.g., eigenvalues with shape [C, D, H, W])
                if len(volume_shape) == 4 and not is_2d:
                    # Skip first dimension (channels) for 3D multi-channel data
                    volume_shape = volume_shape[1:]  # [D, H, W]
                elif len(volume_shape) == 3 and is_2d:
                    # Skip first dimension (channels) for 2D multi-channel data
                    volume_shape = volume_shape[1:]  # [H, W]
                
                if is_2d:
                    patch_size = self.patch_size[:2]  # [h, w]
                else:
                    patch_size = self.patch_size  # [d, h, w]
                
                patches = self._get_all_sliding_window_positions(volume_shape, patch_size)
                print(f"Generated {len(patches)} patches from {'2D' if is_2d else '3D'} data using sliding window")
            else:
                # Original validation logic
                label_data = vdata['label']  # Get the label data explicitly
                
                # Handle multi-channel labels for patch finding
                if len(label_data.shape) == 4 and not is_2d:
                    # 4D data: [C, D, H, W] - use first channel for patch finding to avoid loading entire array
                    print(f"Detected 4D label data with shape {label_data.shape} for volume {vol_idx}")
                    print(f"Using first channel (of {label_data.shape[0]}) for patch validation to avoid memory issues")
                    label_data_for_patches = label_data[0]  # [D, H, W] - only loads first channel
                elif len(label_data.shape) == 3 and is_2d:
                    # 3D data for 2D: [C, H, W] - use first channel
                    print(f"Detected multi-channel 2D label data with shape {label_data.shape} for volume {vol_idx}")
                    print(f"Using first channel (of {label_data.shape[0]}) for patch validation to avoid memory issues")
                    label_data_for_patches = label_data[0]  # [H, W] - only loads first channel
                else:
                    # Normal 3D or 2D data
                    label_data_for_patches = label_data
                
                # Check if mask is available
                has_mask = 'mask' in vdata and vdata['mask'] is not None
                
                if has_mask:
                    mask_data = vdata['mask']
                    # Handle multi-channel masks similarly
                    if len(mask_data.shape) == 4 and not is_2d:
                        mask_data_for_patches = mask_data[0]  # Use first channel only
                    elif len(mask_data.shape) == 3 and is_2d:
                        mask_data_for_patches = mask_data[0]  # Use first channel only
                    else:
                        mask_data_for_patches = mask_data
                    print(f"Using mask for patch extraction in volume {vol_idx}")
                else:
                    # When no mask is available, we'll skip mask checks entirely
                    # The mask_data shape is still needed for the patch finding functions
                    mask_data_for_patches = label_data_for_patches  # Just for shape, won't be used with skip_mask_check=True
                    print(f"No mask found for volume {vol_idx}, skipping mask coverage checks")
                
                if is_2d:
                    h, w = self.patch_size[0], self.patch_size[1]  # y, x
                    patches = find_mask_patches_2d(
                        mask_data_for_patches,
                        label_data_for_patches,
                        patch_size=[h, w], 
                        min_mask_coverage=1.0,
                        min_labeled_ratio=self.min_labeled_ratio,
                        skip_mask_check=not has_mask  # Skip mask check if no mask provided
                    )
                    print(f"Found {len(patches)} patches from 2D data with min labeled ratio {self.min_labeled_ratio}")
                else:
                    patches = find_mask_patches(
                        mask_data_for_patches,
                        label_data_for_patches,
                        patch_size=self.patch_size, 
                        min_mask_coverage=1.0,
                        min_labeled_ratio=self.min_labeled_ratio,
                        skip_mask_check=not has_mask  # Skip mask check if no mask provided
                    )
                    print(f"Found {len(patches)} patches from 3D data with min labeled ratio {self.min_labeled_ratio}")

            for p in patches:
                self.valid_patches.append({
                    "volume_index": vol_idx,
                    "position": p["start_pos"]  # (z,y,x)
                })
        
        # Save to cache after computing all patches
        if self.cache_enabled and self.cache_dir is not None and self.data_path is not None:
            print(f"\nAttempting to save {len(self.valid_patches)} patches to cache...")
            config_params = self._get_config_params()
            success = save_computed_patches(
                self.valid_patches,
                self.cache_dir,
                config_params,
                self.data_path,
                intensity_properties=self.intensity_properties,
                normalization_scheme=self.normalization_scheme
            )
            if success:
                print(f"Successfully saved patches to cache directory: {self.cache_dir}\n")
            else:
                print("Failed to save patches to cache\n")
        else:
            if not self.cache_enabled:
                print("Patch caching is disabled")
            elif self.cache_dir is None:
                print("Cache directory is not set")
            elif self.data_path is None:
                print("Data path is not set")

    def __len__(self):
        return len(self.valid_patches)

    def _validate_dimensionality(self, data_item, ref_item=None):
        """
        Validate and ensure consistent dimensionality between different data samples.
        
        Parameters
        ----------
        data_item : numpy.ndarray
            The data item to validate
        ref_item : numpy.ndarray, optional
            A reference item to compare against
            
        Returns
        -------
        bool
            True if the data is 2D, False if 3D
        """
        is_2d = len(data_item.shape) == 2
        
        if ref_item is not None:
            ref_is_2d = len(ref_item.shape) == 2
            if is_2d != ref_is_2d:
                raise ValueError(
                    f"Dimensionality mismatch: Data item is {'2D' if is_2d else '3D'} "
                    f"but reference item is {'2D' if ref_is_2d else '3D'}"
                )
        
        return is_2d
            
    def _extract_patch_coords(self, patch_info):
        """
        Extract patch coordinates and sizes based on dataset dimensionality.
        
        Parameters
        ----------
        patch_info : dict
            Dictionary containing patch position information
        
        Returns
        -------
        tuple
            (z, y, x, dz, dy, dx, is_2d) coordinates and dimensions
        """
        if self.is_2d_dataset:
            # For 2D, position is [dummy_z, y, x] and patch_size should be [h, w]
            _, y, x = patch_info["position"]  # Unpack properly ignoring dummy z value
            
            # Handle patch_size dimensionality - take last 2 dimensions for 2D
            if len(self.patch_size) >= 2:
                dy, dx = self.patch_size[-2:]  # Take last 2 elements (height, width)
            else:
                raise ValueError(f"patch_size {self.patch_size} insufficient for 2D data")
                
            z, dz = 0, 0  # Not used for 2D
            is_2d = True
        else:
            # For 3D, position is (z, y, x) and patch_size is (d, h, w)
            z, y, x = patch_info["position"]
            
            # Handle patch_size dimensionality
            if len(self.patch_size) >= 3:
                dz, dy, dx = self.patch_size[:3]  # Take first 3 elements
            elif len(self.patch_size) == 2:
                # 2D patch_size for 3D data - assume depth of 1
                dy, dx = self.patch_size
                dz = 1
            else:
                raise ValueError(f"patch_size {self.patch_size} insufficient for 3D data")
                
            is_2d = False
            
        return z, y, x, dz, dy, dx, is_2d
    
    def _extract_image_patch(self, vol_idx, z, y, x, dz, dy, dx, is_2d):
        """
        Extract and normalize an image patch from the volume.
        
        Parameters
        ----------
        vol_idx : int
            Volume index
        z, y, x : int
            Starting coordinates
        dz, dy, dx : int
            Patch dimensions
        is_2d : bool
            Whether the data is 2D
            
        Returns
        -------
        numpy.ndarray
            Normalized image patch with channel dimension [C, H, W] or [C, D, H, W]
        """
        # Get the image from the first target (all targets share the same image)
        first_target_name = list(self.target_volumes.keys())[0]
        img_arr = self.target_volumes[first_target_name][vol_idx]['data']['data']
        
        try:
            # Extract image patch with appropriate dimensionality
            if is_2d:
                img_patch = img_arr[y:y+dy, x:x+dx]
                # Check if we got valid data
                if img_patch.size == 0:
                    raise ValueError(f"Empty patch extracted at position y={y}, x={x}")
                img_patch = pad_or_crop_2d(img_patch, (dy, dx))
            else:
                img_patch = img_arr[z:z+dz, y:y+dy, x:x+dx]
                # Check if we got valid data
                if img_patch.size == 0:
                    raise ValueError(f"Empty patch extracted at position z={z}, y={y}, x={x}")
                img_patch = pad_or_crop_3d(img_patch, (dz, dy, dx))
        except ValueError as e:
            # Handle corrupt or missing chunks by creating a zero patch
            print(f"Warning: Failed to extract image patch at vol={vol_idx}, z={z}, y={y}, x={x}: {str(e)}")
            print(f"Creating zero patch of size {'('+str(dy)+','+str(dx)+')' if is_2d else '('+str(dz)+','+str(dy)+','+str(dx)+')'}")
            
            if is_2d:
                img_patch = np.zeros((dy, dx), dtype=np.float32)
            else:
                img_patch = np.zeros((dz, dy, dx), dtype=np.float32)
        
        # Apply normalization
        if self.normalizer is not None:
            img_patch = self.normalizer.run(img_patch)
        else:
            img_patch = img_patch.astype(np.float32)
        
        # Add channel dimension and ensure contiguous
        img_patch = np.ascontiguousarray(img_patch[np.newaxis, ...])     
        return img_patch
    
    def _extract_label_patches(self, vol_idx, z, y, x, dz, dy, dx, is_2d):
        """
        Extract all label patches for all targets.
        
        Parameters
        ----------
        vol_idx : int
            Volume index
        z, y, x : int
            Starting coordinates
        dz, dy, dx : int
            Patch dimensions
        is_2d : bool
            Whether the data is 2D
            
        Returns
        -------
        dict
            Dictionary of label patches for each target
        """
        label_patches = {}
        
        for t_name, volumes_list in self.target_volumes.items():
            volume_info = volumes_list[vol_idx]
            label_arr = volume_info['data']['label']
            
            # Check if label has channel dimension
            has_channels = False
            if is_2d and len(label_arr.shape) == 3:
                # 2D multi-channel: [C, H, W]
                has_channels = True
                n_channels = label_arr.shape[0]
            elif not is_2d and len(label_arr.shape) == 4:
                # 3D multi-channel: [C, D, H, W]
                has_channels = True
                n_channels = label_arr.shape[0]
            
            if is_2d:
                if has_channels:
                    # Multi-channel 2D: extract channels one at a time to avoid memory issues
                    channel_patches = []
                    for c in range(n_channels):
                        channel_patches.append(label_arr[c, y:y+dy, x:x+dx])
                    label_patch = np.stack(channel_patches, axis=0)
                else:
                    # Single-channel 2D
                    label_patch = label_arr[y:y+dy, x:x+dx]
                
                # Apply threshold if configured (before binarization)
                if self.label_threshold is not None and not self.targets.get(t_name, {}).get('auxiliary_task', False):
                    # Apply threshold: values below threshold become 0
                    label_patch = np.where(label_patch < self.label_threshold, 0, label_patch)
                
                # Apply binarization only if configured to do so and not an auxiliary task
                if self.binarize_labels and not self.targets.get(t_name, {}).get('auxiliary_task', False):
                    target_value = self._get_target_value(t_name)
                    
                    if isinstance(target_value, dict):
                        # Check if this is a multi-class config with regions
                        if 'mapping' in target_value:
                            # New format with mapping and optional regions
                            mapping = target_value['mapping']
                            regions = target_value.get('regions', {})
                            
                            # First apply standard mapping
                            new_label_patch = np.zeros_like(label_patch)
                            for original_val, new_val in mapping.items():
                                mask = (label_patch == original_val)
                                new_label_patch[mask] = new_val
                            
                            # Then apply regions (which override)
                            for region_id, source_classes in regions.items():
                                # Create mask for any pixel that belongs to source classes
                                region_mask = np.zeros_like(new_label_patch, dtype=bool)
                                for source_class in source_classes:
                                    region_mask |= (new_label_patch == source_class)
                                # Override those pixels with the region ID
                                new_label_patch[region_mask] = region_id
                            
                            label_patch = new_label_patch
                        else:
                            # Old format: direct mapping
                            new_label_patch = np.zeros_like(label_patch)
                            for original_val, new_val in target_value.items():
                                mask = (label_patch == original_val)
                                new_label_patch[mask] = new_val
                            label_patch = new_label_patch
                    else:
                        # Single class binarization: keep zeros as zero, set non-zeros to target_value
                        binary_mask = (label_patch > 0)
                        new_label_patch = np.zeros_like(label_patch)
                        new_label_patch[binary_mask] = target_value
                        label_patch = new_label_patch
                
                if has_channels:
                    # Pad each channel separately
                    padded_channels = []
                    for c in range(n_channels):
                        padded_channels.append(pad_or_crop_2d(label_patch[c], (dy, dx)))
                    label_patch = np.stack(padded_channels, axis=0)
                else:
                    label_patch = pad_or_crop_2d(label_patch, (dy, dx))
            else:
                if has_channels:
                    # Multi-channel 3D: extract channels one at a time to avoid memory issues
                    channel_patches = []
                    for c in range(n_channels):
                        channel_patches.append(label_arr[c, z:z+dz, y:y+dy, x:x+dx])
                    label_patch = np.stack(channel_patches, axis=0)
                else:
                    # Single-channel 3D
                    label_patch = label_arr[z:z+dz, y:y+dy, x:x+dx]
                
                # Apply threshold if configured (before binarization)
                if self.label_threshold is not None and not self.targets.get(t_name, {}).get('auxiliary_task', False):
                    # Apply threshold: values below threshold become 0
                    label_patch = np.where(label_patch < self.label_threshold, 0, label_patch)
                
                # Apply binarization only if configured to do so
                if self.binarize_labels:
                    target_value = self._get_target_value(t_name)
                    
                    if isinstance(target_value, dict):
                        # Check if this is a multi-class config with regions
                        if 'mapping' in target_value:
                            # New format with mapping and optional regions
                            mapping = target_value['mapping']
                            regions = target_value.get('regions', {})
                            
                            # First apply standard mapping
                            new_label_patch = np.zeros_like(label_patch)
                            for original_val, new_val in mapping.items():
                                mask = (label_patch == original_val)
                                new_label_patch[mask] = new_val
                            
                            # Then apply regions (which override)
                            for region_id, source_classes in regions.items():
                                # Create mask for any pixel that belongs to source classes
                                region_mask = np.zeros_like(new_label_patch, dtype=bool)
                                for source_class in source_classes:
                                    region_mask |= (new_label_patch == source_class)
                                # Override those pixels with the region ID
                                new_label_patch[region_mask] = region_id
                            
                            label_patch = new_label_patch
                        else:
                            # Old format: direct mapping
                            new_label_patch = np.zeros_like(label_patch)
                            for original_val, new_val in target_value.items():
                                mask = (label_patch == original_val)
                                new_label_patch[mask] = new_val
                            label_patch = new_label_patch
                    else:
                        # Single class binarization: keep zeros as zero, set non-zeros to target_value
                        binary_mask = (label_patch > 0)
                        new_label_patch = np.zeros_like(label_patch)
                        new_label_patch[binary_mask] = target_value
                        label_patch = new_label_patch
                        
                if has_channels:
                    # Pad each channel separately
                    padded_channels = []
                    for c in range(n_channels):
                        padded_channels.append(pad_or_crop_3d(label_patch[c], (dz, dy, dx)))
                    label_patch = np.stack(padded_channels, axis=0)
                else:
                    label_patch = pad_or_crop_3d(label_patch, (dz, dy, dx))
            
            # Add channel dimension if not already present
            if not has_channels:
                label_patch = label_patch[np.newaxis, ...]
            # Ensure contiguous
            label_patch = np.ascontiguousarray(label_patch, dtype=np.float32)
            label_patches[t_name] = label_patch
            
        # Process auxiliary tasks - generate distance transforms and other auxiliary targets
        self._process_auxiliary_tasks(label_patches, is_2d)
            
        return label_patches

    def _get_target_value(self, t_name):
        """
        Extract target value from configuration.
        
        Parameters
        ----------
        t_name : str
            Target name
            
        Returns
        -------
        int or dict
            Target value(s) - can be int for single class or dict for multi-class
        """
        # Check if this is an auxiliary task - they don't need target values
        if t_name in self.targets and self.targets[t_name].get('auxiliary_task', False):
            return None  # Auxiliary tasks don't have target values
        
        # Don't warn about missing target value if binarization is disabled
        if not self.binarize_labels:
            return 1  # Return default without warning
        
        # Check if target_value is configured
        if isinstance(self.target_value, dict) and t_name in self.target_value:
            return self.target_value[t_name]
        elif isinstance(self.target_value, int):
            return self.target_value
        else:
            # Default to 1 if not configured
            print(f"Warning: No target value configured for '{t_name}', defaulting to 1")
            return 1

    def _get_volume_id(self, vol_idx):
        """
        Get the volume identifier for a given volume index.
        
        Parameters
        ----------
        vol_idx : int
            Volume index
            
        Returns
        -------
        str or None
            Volume identifier if available, None otherwise
        """
        # Get the first target to access volume info
        first_target = list(self.target_volumes.keys())[0]
        
        # Check if volume_id was stored during initialization
        if vol_idx < len(self.target_volumes[first_target]):
            volume_info = self.target_volumes[first_target][vol_idx]
            if 'volume_id' in volume_info:
                return volume_info['volume_id']
        
        # Fallback to None if not available
        return None

    
    def _extract_ignore_mask(self, vol_idx, z, y, x, dz, dy, dx, is_2d):
        """
        Extract ignore mask if available.
        
        Parameters
        ----------
        vol_idx : int
            Volume index
        z, y, x : int
            Starting coordinates
        dz, dy, dx : int
            Patch dimensions
        is_2d : bool
            Whether the data is 2D
            
        Returns
        -------
        dict or None
            Dictionary of ignore masks per target if available, None otherwise
        """
        # Initialize ignore masks dictionary
        ignore_masks = {}
        
        # Check for volume-task loss configuration
        if hasattr(self.mgr, 'volume_task_loss_config') and self.mgr.volume_task_loss_config:
            # Get the volume ID for this patch
            volume_id = self._get_volume_id(vol_idx)
            
            if volume_id and volume_id in self.mgr.volume_task_loss_config:
                # Get enabled tasks for this volume
                enabled_tasks = self.mgr.volume_task_loss_config[volume_id]
                
                # Create ignore masks for disabled tasks
                for t_name in self.target_volumes.keys():
                    if t_name not in enabled_tasks:
                        # Create a full ignore mask (all ones) for this disabled task
                        if is_2d:
                            mask_shape = (1, dy, dx)
                        else:
                            mask_shape = (1, dz, dy, dx)
                        
                        ignore_masks[t_name] = np.ones(mask_shape, dtype=np.float32)
                        
                        # Log only once per volume-task combination
                        if not hasattr(self, '_logged_disabled_tasks'):
                            self._logged_disabled_tasks = set()
                        
                        log_key = (volume_id, t_name)
                        if log_key not in self._logged_disabled_tasks:
                            print(f"Task '{t_name}' disabled for volume '{volume_id}' - creating ignore mask")
                            self._logged_disabled_tasks.add(log_key)
        
        # Check for auxiliary task source masks
        if hasattr(self, '_aux_source_masks'):
            for aux_task_name, source_mask in self._aux_source_masks.items():
                if aux_task_name in self.targets:
                    # Invert the mask: source_mask has 1 where source > 0 (compute), 
                    # but ignore_mask needs 1 where we ignore (source == 0)
                    ignore_mask = 1.0 - source_mask
                    ignore_masks[aux_task_name] = ignore_mask
        
        # Check only the first target for the mask
        first_target_name = list(self.target_volumes.keys())[0]
        volume_info = self.target_volumes[first_target_name][vol_idx]
        vdata = volume_info['data']
        
        if 'mask' in vdata:
            # Use explicit mask if available - same mask for all targets
            mask_arr = vdata['mask']
            
            # Check if mask has channel dimension
            has_channels = False
            if is_2d and len(mask_arr.shape) == 3:
                has_channels = True
                n_channels = mask_arr.shape[0]
            elif not is_2d and len(mask_arr.shape) == 4:
                has_channels = True
                n_channels = mask_arr.shape[0]
            
            if is_2d:
                if has_channels:
                    # Extract channels one at a time to avoid memory issues
                    channel_patches = []
                    for c in range(n_channels):
                        channel_patches.append(mask_arr[c, y:y+dy, x:x+dx])
                    mask_patch = np.stack(channel_patches, axis=0)
                    # Pad each channel separately
                    padded_channels = []
                    for c in range(n_channels):
                        padded_channels.append(pad_or_crop_2d(mask_patch[c], (dy, dx)))
                    mask_patch = np.stack(padded_channels, axis=0)
                else:
                    mask_patch = mask_arr[y:y+dy, x:x+dx]
                    mask_patch = pad_or_crop_2d(mask_patch, (dy, dx))
            else:
                if has_channels:
                    # Extract channels one at a time to avoid memory issues
                    channel_patches = []
                    for c in range(n_channels):
                        channel_patches.append(mask_arr[c, z:z+dz, y:y+dy, x:x+dx])
                    mask_patch = np.stack(channel_patches, axis=0)
                    # Pad each channel separately
                    padded_channels = []
                    for c in range(n_channels):
                        padded_channels.append(pad_or_crop_3d(mask_patch[c], (dz, dy, dx)))
                    mask_patch = np.stack(padded_channels, axis=0)
                else:
                    mask_patch = mask_arr[z:z+dz, y:y+dy, x:x+dx]
                    mask_patch = pad_or_crop_3d(mask_patch, (dz, dy, dx))
            
            # Convert to binary mask and invert (1 = ignore, 0 = compute)
            single_mask = (mask_patch == 0).astype(np.float32)
            
            # Add channel dimension if not already present
            if not has_channels:
                single_mask = single_mask[np.newaxis, ...]  # [1, H, W] or [1, D, H, W]
            
            # Return same mask for all targets
            return {t_name: single_mask for t_name in self.target_volumes.keys()}
            
        elif hasattr(self.mgr, 'compute_loss_on_labeled_only') and self.mgr.compute_loss_on_labeled_only:
            # Create separate masks from each target's labels
            ignore_masks = {}
            
            for t_name, volumes_list in self.target_volumes.items():
                label_arr = volumes_list[vol_idx]['data']['label']
                
                # Check if label has channel dimension
                has_channels = False
                if is_2d and len(label_arr.shape) == 3:
                    has_channels = True
                    n_channels = label_arr.shape[0]
                elif not is_2d and len(label_arr.shape) == 4:
                    has_channels = True
                    n_channels = label_arr.shape[0]
                
                # Extract label patch
                if is_2d:
                    if has_channels:
                        # Extract channels one at a time to avoid memory issues
                        channel_patches = []
                        for c in range(n_channels):
                            channel_patches.append(label_arr[c, y:y+dy, x:x+dx])
                        label_patch = np.stack(channel_patches, axis=0)
                        # Pad each channel separately
                        padded_channels = []
                        for c in range(n_channels):
                            padded_channels.append(pad_or_crop_2d(label_patch[c], (dy, dx)))
                        label_patch = np.stack(padded_channels, axis=0)
                    else:
                        label_patch = label_arr[y:y+dy, x:x+dx]
                        label_patch = pad_or_crop_2d(label_patch, (dy, dx))
                else:
                    if has_channels:
                        # Extract channels one at a time to avoid memory issues
                        channel_patches = []
                        for c in range(n_channels):
                            channel_patches.append(label_arr[c, z:z+dz, y:y+dy, x:x+dx])
                        label_patch = np.stack(channel_patches, axis=0)
                        # Pad each channel separately
                        padded_channels = []
                        for c in range(n_channels):
                            padded_channels.append(pad_or_crop_3d(label_patch[c], (dz, dy, dx)))
                        label_patch = np.stack(padded_channels, axis=0)
                    else:
                        label_patch = label_arr[z:z+dz, y:y+dy, x:x+dx]
                        label_patch = pad_or_crop_3d(label_patch, (dz, dy, dx))
                
                # Create binary mask from this target's label (1 = ignore unlabeled, 0 = compute on labeled)
                # For multi-channel, we might want to consider all channels
                if has_channels:
                    # Take the max across channels - if any channel has a value, consider it labeled
                    mask = (label_patch.max(axis=0) == 0).astype(np.float32)
                else:
                    mask = (label_patch == 0).astype(np.float32)
                
                # Add channel dimension
                mask = mask[np.newaxis, ...]  # Shape: [1, H, W] or [1, D, H, W]
                ignore_masks[t_name] = mask
            
            return ignore_masks
        
        return None

    
    def _prepare_tensors(self, img_patch, label_patches, ignore_mask, vol_idx, is_2d):
        """
        Convert numpy arrays to PyTorch tensors.
        
        All input arrays already have channel dimensions from extraction methods.
        
        Parameters
        ----------
        img_patch : numpy.ndarray
            Image patch with shape [C, H, W] or [C, D, H, W]
        label_patches : dict
            Dictionary of label patches for each target with shape [C, H, W] or [C, D, H, W]
        ignore_mask : dict or None
            Dictionary of ignore masks per target if available
        vol_idx : int
            Volume index
        is_2d : bool
            Whether the data is 2D
            
        Returns
        -------
        dict
            Dictionary of tensors for training
        """
        data_dict = {}
        
        # Simply convert image to tensor (already has channel dimension)
        data_dict["image"] = torch.from_numpy(img_patch)
        
        # Add ignore masks if available
        if ignore_mask is not None:
            data_dict["ignore_masks"] = {
                t_name: torch.from_numpy(mask) 
                for t_name, mask in ignore_mask.items()
            }
        
        # Process all labels based on target configuration
        for t_name, label_patch in label_patches.items():
            # Skip target value processing for auxiliary tasks
            if t_name in self.targets and self.targets[t_name].get('auxiliary_task', False):
                # Auxiliary tasks are regression targets, just convert to tensor
                data_dict[t_name] = torch.from_numpy(label_patch)
                continue
                
            # Check if this is a multi-class target
            target_value = self._get_target_value(t_name)
            is_multiclass = (
                isinstance(target_value, dict) and 
                'mapping' in target_value and 
                ('regions' in target_value or len(target_value['mapping']) > 2)
            )
            
            # Get the number of output channels for this target
            if t_name not in self.targets:
                print(f"Warning: Target '{t_name}' not found in self.targets. Available targets: {list(self.targets.keys())}")
                # Create a default entry
                self.targets[t_name] = {"out_channels": 2}
            out_channels = self.targets[t_name].get("out_channels", 2)
            
            if is_multiclass or out_channels > 2:
                # Multi-class segmentation - update metadata
                unique_vals = np.unique(label_patch)
                num_classes = int(unique_vals.max()) + 1
                self.targets[t_name]["out_channels"] = max(num_classes, out_channels)
            
            # Check if we need to convert to one-hot for binary segmentation
            if out_channels == 2 and label_patch.shape[0] == 1:
                # Binary segmentation with 2 output channels - convert to one-hot
                # label_patch has shape [1, H, W] or [1, D, H, W]
                label_tensor = torch.from_numpy(label_patch).long()
                
                # Create one-hot encoding
                if is_2d:
                    # Shape: [1, H, W] -> [2, H, W]
                    one_hot = torch.zeros(2, label_tensor.shape[1], label_tensor.shape[2])
                    one_hot[0] = (label_tensor[0] == 0).float()  # Background channel
                    one_hot[1] = (label_tensor[0] == 1).float()  # Foreground channel
                else:
                    # Shape: [1, D, H, W] -> [2, D, H, W]
                    one_hot = torch.zeros(2, label_tensor.shape[1], label_tensor.shape[2], label_tensor.shape[3])
                    one_hot[0] = (label_tensor[0] == 0).float()  # Background channel
                    one_hot[1] = (label_tensor[0] == 1).float()  # Foreground channel
                
                data_dict[t_name] = one_hot
            else:
                # Simply convert to tensor (already has proper shape and channel dimension)
                data_dict[t_name] = torch.from_numpy(label_patch)
        
        return data_dict
    

    def _create_training_transforms(self):
        """
        Create training transforms using custom batchgeneratorsv2.
        Returns None for validation (no augmentations).
        """
        # Check if spatial transformations are disabled
        if getattr(self.mgr, 'no_spatial', False):
            print("Spatial transformations disabled (no_spatial=True)")
            return None
            
        dimension = len(self.mgr.train_patch_size)
        
        # Handle both 2D and 3D patch sizes
        if dimension == 2:
            patch_h, patch_w = self.mgr.train_patch_size
            patch_d = None  # Not used for 2D
        elif dimension == 3:
            patch_d, patch_h, patch_w = self.mgr.train_patch_size
        else:
            raise ValueError(f"Invalid patch size dimension: {dimension}. Expected 2 or 3.")
        if dimension == 2:
            if max(self.mgr.train_patch_size) / min(self.mgr.train_patch_size) > 1.5:
                rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            mirror_axes = (0, 1)
        elif dimension == 3:
            rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)

        transforms = []
        
        if dimension == 2:
            # 2D transforms (no transpose transform)
            
            transforms.append(RandomTransform(
                MirrorTransform(allowed_axes=(0, 1)), 
                apply_probability=0.5
            )),
            transforms.append(RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.5),
                    synchronize_channels=True,
                    synchronize_axes=False,
                    p_per_channel=1.0
                ),
                apply_probability=0.2
            ))
        else:
            # 3D transforms
            transforms.append(
            SpatialTransform(
                self.mgr.train_patch_size, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.0,
                rotation=rotation_for_DA, p_scaling=0.0, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )),
            transforms.append(RandomTransform(
                    MirrorTransform(allowed_axes=mirror_axes), 
                    apply_probability=0.0
                )),
            transforms.append(RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.5, 1.0),
                        synchronize_channels=True,
                        synchronize_axes=False,
                        p_per_channel=1.0
                    ),
                    apply_probability=0.2
                )),
            transforms.append(RandomTransform(
                TransposeAxesTransform(
                    allowed_axes={0, 1, 2},
                    ),
                    apply_probability=0.0
            )),
            transforms.append(RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.1),
                    p_per_channel=1,
                    synchronize_channels=True
                ), apply_probability=0.1
            ))
            transforms.append(RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5, benchmark=True
                ), apply_probability=0.2
            ))
            transforms.append(RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.75, 1.25)),
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.15
            ))
            transforms.append(RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.75, 1.25)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.15
            ))
            transforms.append(RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=None,
                    allowed_channels=None,
                    p_per_channel=0.5
                ), apply_probability=0.25
            ))
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.1
            ))
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.3
            ))
            transforms.append(RandomTransform(
                BlankRectangleTransform(
                    rectangle_size=((max(1, self.mgr.train_patch_size[0] // 10), self.mgr.train_patch_size[0] // 3),
                                    (max(1, self.mgr.train_patch_size[1] // 10), self.mgr.train_patch_size[1] // 3),
                                    (max(1, self.mgr.train_patch_size[2] // 10), self.mgr.train_patch_size[2] // 3)),
                    rectangle_value=np.mean,  # keeping the mean value
                    num_rectangles=(1, 5),  # same as original
                    force_square=False,  # same as original
                    p_per_sample=0.4,  # same as original
                    p_per_channel=0.5  # same as original
                ), apply_probability=0.5
            ))
            transforms.append(RandomTransform(
                InhomogeneousSliceIlluminationTransform(
                    num_defects=(2, 5),  # Range for number of defects
                    defect_width=(5, 20),  # Range for defect width
                    mult_brightness_reduction_at_defect=(0.3, 0.7),  # Range for brightness reduction
                    base_p=(0.2, 0.4),  # Base probability range
                    base_red=(0.5, 0.9),  # Base reduction range
                    p_per_sample=1.0,  # Probability per sample
                    per_channel=True,  # Apply per channel
                    p_per_channel=0.5  # Probability per channel
                ), apply_probability=0.25
            ))

            # Only add transpose transform if all three dimensions (z, y, x) are equal
            if patch_d == patch_h == patch_w:
                transforms.insert(1, RandomTransform(
                    TransposeAxesTransform(allowed_axes={0, 1, 2}), 
                    apply_probability=0.5
                ))
                print(f"Added transpose transform for 3D (equal dimensions: {patch_d}x{patch_h}x{patch_w})")
            else:
                print(f"Skipped transpose transform for 3D (unequal dimensions: {patch_d}x{patch_h}x{patch_w})")

        return ComposeTransforms(transforms)
    
    def __getitem__(self, index):
        """
        Get a patch from the dataset.
        
        Parameters
        ----------
        index : int
            Index of the patch
            
        Returns
        -------
        dict
            Dictionary of tensors for training
        """
        # 1. Get patch info and coordinates
        patch_info = self.valid_patches[index]
        vol_idx = patch_info["volume_index"]
        z, y, x, dz, dy, dx, is_2d = self._extract_patch_coords(patch_info)
        
        # 2. Extract and normalize image patch
        img_patch = self._extract_image_patch(vol_idx, z, y, x, dz, dy, dx, is_2d)
        
        # 3. Extract label patches for all targets
        label_patches = self._extract_label_patches(vol_idx, z, y, x, dz, dy, dx, is_2d)
        
        # 4. Extract ignore mask if available
        ignore_mask = self._extract_ignore_mask(vol_idx, z, y, x, dz, dy, dx, is_2d)
    
        # 5. Convert to tensors and format for the model
        data_dict = self._prepare_tensors(img_patch, label_patches, ignore_mask, vol_idx, is_2d)
        
        # Clean up intermediate numpy arrays
        del img_patch, label_patches
        if ignore_mask is not None:
            del ignore_mask
        
        # 6. Apply augmentations if in training mode
        if self.is_training and self.transforms is not None:
            # Apply transforms directly to the torch tensors
            data_dict = self.transforms(**data_dict)
        
        return data_dict
    
    def clear_transform_cache(self):
        """
        Clear any cached state in transforms to prevent memory accumulation.
        This should be called periodically during training.
        """
        if hasattr(self, 'transforms') and self.transforms is not None:
            # Clear any potential cached state in individual transforms
            if hasattr(self.transforms, 'transforms'):
                for transform in self.transforms.transforms:
                    # Clear any cached random state or parameters
                    if hasattr(transform, '_cached_params'):
                        delattr(transform, '_cached_params')
                    # For RandomTransform, clear the wrapped transform too
                    if hasattr(transform, 'transform') and hasattr(transform.transform, '_cached_params'):
                        delattr(transform.transform, '_cached_params')
            
            # Force garbage collection of any lingering references
            import gc
            gc.collect()

    def _process_auxiliary_tasks(self, label_patches, is_2d):
        """
        Process auxiliary tasks by computing additional targets like distance transforms and surface normals.
        
        Parameters
        ----------
        label_patches : dict
            Dictionary of label patches for each target
        is_2d : bool
            Whether the data is 2D
        """
        # Check if manager has auxiliary tasks
        if not hasattr(self.mgr, 'auxiliary_tasks') or not self.mgr.auxiliary_tasks:
            return
            
        # Store source masks for auxiliary tasks that need them
        if not hasattr(self, '_aux_source_masks'):
            self._aux_source_masks = {}
            
        # Cache for signed distance transforms to avoid recomputation
        sdt_cache = {}
        
        # First pass: check if we need surface normals (which compute SDT)
        needs_sdt = {}
        for aux_task_name, aux_config in self.mgr.auxiliary_tasks.items():
            if aux_config["type"] == "surface_normals":
                source_target = aux_config["source_target"]
                if source_target in label_patches:
                    needs_sdt[source_target] = True
                    
        # Process all auxiliary tasks
        for aux_task_name, aux_config in self.mgr.auxiliary_tasks.items():
            task_type = aux_config["type"]
            
            if task_type == "distance_transform":
                source_target = aux_config["source_target"]
                
                # Check if source target exists in label_patches
                if source_target not in label_patches:
                    continue
                    
                source_patch = label_patches[source_target]
                
                # Check if we have cached SDT from surface normals computation
                cache_key = f"{source_target}_sdt"
                
                # Compute distance transform for each channel
                distance_transforms = []
                for c in range(source_patch.shape[0]):  # Iterate over channels
                    channel_key = f"{cache_key}_{c}"
                    
                    if channel_key in sdt_cache:
                        # Reuse cached signed distance transform
                        sdt = sdt_cache[channel_key]
                        # For distance transform, we want the absolute distance inside the object
                        # SDT is negative inside, positive outside
                        distance_map = np.where(sdt < 0, -sdt, 0)
                    else:
                        # Get binary mask from source patch
                        channel_patch = source_patch[c]
                        binary_mask = (channel_patch > 0).astype(np.uint8)
                        
                        # If this source will be used for surface normals later, compute SDT
                        if source_target in needs_sdt and np.any(binary_mask):
                            inside_dist = distance_transform_edt(binary_mask)
                            outside_dist = distance_transform_edt(1 - binary_mask)
                            sdt = outside_dist - inside_dist
                            sdt_cache[channel_key] = sdt
                            # For distance transform, use the inside distance
                            distance_map = inside_dist
                        else:
                            # Standard distance transform computation
                            if np.any(binary_mask):
                                # Compute distance transform from foreground pixels
                                distance_map = distance_transform_edt(binary_mask)
                            else:
                                # If no foreground pixels, distance map is zero everywhere
                                distance_map = np.zeros_like(channel_patch)
                    
                    distance_transforms.append(distance_map)
                
                # Stack distance transforms and ensure proper format
                if len(distance_transforms) == 1:
                    # Single channel output
                    distance_patch = distance_transforms[0][np.newaxis, ...]
                else:
                    # Multi-channel output
                    distance_patch = np.stack(distance_transforms, axis=0)
                
                # Ensure contiguous and proper dtype
                distance_patch = np.ascontiguousarray(distance_patch, dtype=np.float32)
                
                # Add to label_patches with the auxiliary task name
                aux_target_name = f"{aux_task_name}" 
                label_patches[aux_target_name] = distance_patch
                
            elif task_type == "surface_normals":
                source_target = aux_config["source_target"]
                
                # Check if source target exists in label_patches
                if source_target not in label_patches:
                    continue
                    
                source_patch = label_patches[source_target]
                
                # Compute surface normals for each channel
                all_normals = []
                for c in range(source_patch.shape[0]):  # Iterate over channels
                    # Get binary mask from source patch
                    channel_patch = source_patch[c]
                    binary_mask = (channel_patch > 0).astype(np.uint8)
                    
                    # Check if we need to cache SDT
                    cache_key = f"{source_target}_sdt_{c}"
                    
                    # Compute surface normals from signed distance transform
                    normals, sdt = self._compute_surface_normals_from_sdt(binary_mask, is_2d, return_sdt=True)
                    
                    # Cache the SDT if it might be needed by distance transform
                    if cache_key not in sdt_cache:
                        sdt_cache[cache_key] = sdt
                    
                    all_normals.append(normals)
                
                # For multi-channel input, we average the normals
                if len(all_normals) > 1:
                    # Stack and average
                    stacked_normals = np.stack(all_normals, axis=0)
                    normals_patch = np.mean(stacked_normals, axis=0)
                    # Re-normalize after averaging
                    magnitude = np.sqrt(np.sum(normals_patch**2, axis=0, keepdims=True))
                    magnitude = np.maximum(magnitude, 1e-6)
                    normals_patch = normals_patch / magnitude
                else:
                    normals_patch = all_normals[0]
                
                # Ensure contiguous and proper dtype
                normals_patch = np.ascontiguousarray(normals_patch, dtype=np.float32)
                
                # Add to label_patches with the auxiliary task name
                aux_target_name = f"{aux_task_name}"
                label_patches[aux_target_name] = normals_patch
                
                # Store the source mask for this auxiliary task
                # Create mask where source > 0 (compute loss) vs source == 0 (ignore)
                source_mask = (source_patch > 0).astype(np.float32)
                if source_mask.ndim > 1:
                    # Take max across channels if multi-channel
                    source_mask = source_mask.max(axis=0, keepdims=True)
                else:
                    source_mask = source_mask[np.newaxis, ...]
                self._aux_source_masks[aux_target_name] = source_mask
                
                # Update the target's out_channels based on dimensionality
                if aux_target_name in self.targets:
                    self.targets[aux_task_name]["out_channels"] = 2 if is_2d else 3
                    
    def _compute_surface_normals_from_sdt(self, binary_mask, is_2d, epsilon=1e-6, return_sdt=False):
        """
        Compute surface normals from a binary mask using signed distance transform.
        
        Parameters
        ----------
        binary_mask : numpy.ndarray
            Binary segmentation mask
        is_2d : bool
            Whether the data is 2D or 3D
        epsilon : float
            Small value for numerical stability
        return_sdt : bool
            If True, also return the signed distance transform
        
        Returns
        -------
        numpy.ndarray or tuple
            If return_sdt is False: Normalized surface normals with shape:
            - 2D: (2, H, W) containing (nx, ny)
            - 3D: (3, D, H, W) containing (nx, ny, nz)
            If return_sdt is True: (normals, sdt)
        """
        # 1. Compute signed distance transform
        inside_dist = distance_transform_edt(binary_mask)
        outside_dist = distance_transform_edt(1 - binary_mask)
        sdt = outside_dist - inside_dist
        
        # 2. Compute gradients using Scharr filter
        if is_2d:
            # For 2D: gradients along x and y
            grad_x = scharr(sdt, axis=1)  # Gradient along x (width)
            grad_y = scharr(sdt, axis=0)  # Gradient along y (height)
            gradients = np.stack([grad_x, grad_y], axis=0)
        else:
            # For 3D: gradients along x, y, and z
            grad_x = scharr(sdt, axis=2)  # Gradient along x (width)
            grad_y = scharr(sdt, axis=1)  # Gradient along y (height)
            grad_z = scharr(sdt, axis=0)  # Gradient along z (depth)
            gradients = np.stack([grad_x, grad_y, grad_z], axis=0)
        
        # 3. Normalize gradients to unit vectors
        magnitude = np.sqrt(np.sum(gradients**2, axis=0, keepdims=True))
        magnitude = np.maximum(magnitude, epsilon)
        normals = gradients / magnitude
        
        # 4. Handle undefined regions (optional)
        # Where gradient magnitude is very small, normal direction is undefined
        undefined_mask = magnitude < epsilon
        normals = normals * (1 - undefined_mask)
        
        if return_sdt:
            return normals.astype(np.float32), sdt.astype(np.float32)
        else:
            return normals.astype(np.float32)
