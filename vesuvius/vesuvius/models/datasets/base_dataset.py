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
from vesuvius.models.training.auxiliary_tasks import (
    compute_distance_transform,
    compute_surface_normals_from_sdt
)
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

from vesuvius.utils.utils import pad_or_crop_3d, pad_or_crop_2d
from ..training.normalization import get_normalization
from .intensity_properties import initialize_intensity_properties
from .find_valid_patches import find_valid_patches
from .save_valid_patches import save_valid_patches, load_cached_patches

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
        
        # if you are certain your data contains dense labels (everything is labeled), you can choose
        # to skip the valid patch finding
        self.skip_patch_validation = getattr(mgr, 'skip_patch_validation', False)

        # for semi-supervised workflows, unlabeled data is obviously needed,
        # we want a flag for this so in fully supervised workflows we can assert that all images have
        # corresponding labels (so we catch it early)
        self.allow_unlabeled_data = getattr(mgr, 'allow_unlabeled_data', False)
        
        # Initialize normalization (will be set after computing intensity properties)
        self.normalization_scheme = getattr(mgr, 'normalization_scheme', 'zscore')
        self.intensity_properties = getattr(mgr, 'intensity_properties', {})
        self.normalizer = None  # Will be initialized after volumes are loaded

        self.target_volumes = {}
        self.valid_patches = []
        self.is_2d_dataset = None
        self.data_path = Path(mgr.data_path) if hasattr(mgr, 'data_path') else None
        self.zarr_arrays = []
        self.zarr_names = []
        self.data_paths = []
        
        self.cache_enabled = getattr(mgr, 'cache_valid_patches', True)
        self.cache_dir = None
        if self.data_path is not None:
            self.cache_dir = self.data_path / '.patches_cache'
            print(f"Cache directory: {self.cache_dir}")
            print(f"Cache enabled: {self.cache_enabled}")
        
        self._initialize_volumes()
        ref_target = list(self.target_volumes.keys())[0]
        ref_volume_data = self.target_volumes[ref_target][0]['data']
        
        # Determine dimensionality from label if available, otherwise from image data
        if 'label' in ref_volume_data and ref_volume_data['label'] is not None:
            ref_shape = ref_volume_data['label'].shape
        else:
            ref_shape = ref_volume_data['data'].shape
        
        self.is_2d_dataset = len(ref_shape) == 2 or (len(ref_shape) == 3 and ref_shape[0] <= 20)
        
        if self.is_2d_dataset:
            print("Detected 2D dataset")
        else:
            print("Detected 3D dataset")

        self.intensity_properties = initialize_intensity_properties(
            target_volumes=self.target_volumes,
            normalization_scheme=self.normalization_scheme,
            existing_properties=self.intensity_properties,
            cache_enabled=self.cache_enabled,
            cache_dir=self.cache_dir,
            mgr=self.mgr,
            sample_ratio=0.001,
            max_samples=1000000
        )

        self.normalizer = get_normalization(self.normalization_scheme, self.intensity_properties)

        self.transforms = None
        if self.is_training:
            self.transforms = self._create_training_transforms()
            print("Training transforms initialized")

        self._get_valid_patches()

    def _initialize_volumes(self):
        """
        Initialize volumes from the data source.
        
        This method must be implemented by subclasses to specify how
        data is loaded from their specific data source (napari, TIFs, Zarr, etc.).
        
        The implementation should:
        1. Populate self.target_volumes in the format:
           {
               'target_name': [
                   {
                       'data': {
                           'data': numpy_array,      # Image data
                           'label': numpy_array      # Label data  
                       }
                   },
                   ...  # Additional volumes for this target
               ],
               ...  # Additional targets
           }
        
        2. Populate zarr arrays for patch finding:
           - self.zarr_arrays: List of zarr arrays (label volumes)
           - self.zarr_names: List of names for each volume
           - self.data_paths: List of data paths for each volume
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
        
        seen = set()
        unique_positions = []
        for pos in positions:
            pos_tuple = tuple(pos['start_pos'])
            if pos_tuple not in seen:
                seen.add(pos_tuple)
                unique_positions.append(pos)
        
        return unique_positions
    
    def _get_valid_patches(self):
        """Find valid patches based on labeled ratio requirements."""
        # Check if we should load approved patches from vc_proofreader
        if hasattr(self.mgr, 'approved_patches_file') and self.mgr.approved_patches_file:
            if self._load_approved_patches():
                return
        
        # Get configuration parameters
        bbox_threshold = getattr(self.mgr, 'bbox_threshold', 0.97)
        downsample_level = getattr(self.mgr, 'downsample_level', 1)
        num_workers = getattr(self.mgr, 'num_workers', 4)
        
        # Try to load from cache first
        if self.cache_enabled and len(self.zarr_arrays) > 0:
            cached_patches = load_cached_patches(
                train_data_paths=self.data_paths,
                label_paths=self.data_paths,  # Using same paths for labels
                patch_size=tuple(self.patch_size),
                min_labeled_ratio=self.min_labeled_ratio,
                bbox_threshold=bbox_threshold,
                downsample_level=downsample_level,
                cache_path=str(self.cache_dir) if self.cache_dir else None
            )
            
            if cached_patches is not None:
                self.valid_patches = cached_patches
                print(f"Successfully loaded {len(self.valid_patches)} patches from cache\n")
                return
        
        # Compute patches using zarr arrays
        if len(self.zarr_arrays) == 0:
            raise ValueError("No zarr arrays available for patch finding. Subclasses must populate self.zarr_arrays")
            
        print("Computing valid patches using zarr arrays...")
        valid_patches = find_valid_patches(
            label_arrays=self.zarr_arrays,
            label_names=self.zarr_names,
            patch_size=tuple(self.patch_size),
            bbox_threshold=bbox_threshold,
            label_threshold=self.min_labeled_ratio,
            num_workers=num_workers,
            downsample_level=downsample_level
        )
        
        # Convert to the expected format
        self.valid_patches = []
        for patch in valid_patches:
            self.valid_patches.append({
                "volume_index": patch["volume_idx"],
                "position": patch["start_pos"]  # (z,y,x)
            })
        
        # Save to cache after computing
        if self.cache_enabled and self.cache_dir is not None and len(self.zarr_arrays) > 0:
            cache_path = save_valid_patches(
                valid_patches=valid_patches,
                train_data_paths=self.data_paths,
                label_paths=self.data_paths,
                patch_size=tuple(self.patch_size),
                min_labeled_ratio=self.min_labeled_ratio,
                bbox_threshold=bbox_threshold,
                downsample_level=downsample_level,
                cache_path=str(self.cache_dir) if self.cache_dir else None
            )
            print(f"Saved patches to cache: {cache_path}")

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
            label_arr = volume_info['data'].get('label')
            
            # If no label exists and unlabeled data is allowed, create zero array
            if label_arr is None:
                if self.allow_unlabeled_data:
                    # Create zero array with same shape as patch
                    if is_2d:
                        label_patch = np.zeros((dy, dx), dtype=np.float32)
                    else:
                        label_patch = np.zeros((dz, dy, dx), dtype=np.float32)
                    label_patches[t_name] = label_patch
                    continue
                else:
                    raise ValueError(f"No label found for target '{t_name}' and allow_unlabeled_data=False")
            
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

    
    def _prepare_tensors(self, img_patch, label_patches, vol_idx, is_2d):
        """
        Convert numpy arrays to PyTorch tensors.
        
        All input arrays already have channel dimensions from extraction methods.
        
        Parameters
        ----------
        img_patch : numpy.ndarray
            Image patch with shape [C, H, W] or [C, D, H, W]
        label_patches : dict
            Dictionary of label patches for each target with shape [C, H, W] or [C, D, H, W]
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
        
        # Process all labels - just convert to tensors
        for t_name, label_patch in label_patches.items():
            # Convert to tensor (already has proper shape and channel dimension)
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
                    apply_probability=0.2
                )),
            transforms.append(RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.5, 1.0),
                        synchronize_channels=True,
                        synchronize_axes=False,
                        p_per_channel=1.0
                    ),
                    apply_probability=0.3
                )),
            transforms.append(RandomTransform(
                TransposeAxesTransform(
                    allowed_axes={0, 1, 2},
                    ),
                    apply_probability=0.3
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
                ), apply_probability=0.20
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
                ), apply_probability=0.3
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
    
        # 4. Convert to tensors and format for the model
        data_dict = self._prepare_tensors(img_patch, label_patches, vol_idx, is_2d)
        
        # Clean up intermediate numpy arrays
        del img_patch, label_patches
        
        # 5. Apply augmentations if in training mode
        if self.is_training and self.transforms is not None:
            # Apply transforms directly to the torch tensors
            data_dict = self.transforms(**data_dict)
        
        return data_dict

