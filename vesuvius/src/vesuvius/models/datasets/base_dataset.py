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
from vesuvius.models.augmentation.transforms.utils.oneoftransform import OneOfTransform
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
from vesuvius.models.augmentation.transforms.noise.extranoisetransforms import BlankRectangleTransform, RicianNoiseTransform, SmearTransform
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

        if 'label' in ref_volume_data and ref_volume_data['label'] is not None:
            ref_shape = ref_volume_data['label'].shape
        else:
            ref_shape = ref_volume_data['data'].shape

        self.is_2d_dataset = len(ref_shape) == 2 or (len(ref_shape) == 3 and ref_shape[0] <= 20)
        
        if self.is_2d_dataset:
            print("Detected 2D dataset")
            if len(self.patch_size) == 3:
                self.patch_size = list(self.patch_size[-2:])
                print(f"Adjusted patch size for 2D data: {self.patch_size}")
        else:
            print("Detected 3D dataset")

        # Check if we should skip intensity sampling
        skip_intensity_sampling = getattr(mgr, 'skip_intensity_sampling', False)

        if skip_intensity_sampling:
            print("Skipping intensity sampling as requested")
            # Use default values if intensity properties not provided
            if not self.intensity_properties:
                self.intensity_properties = {
                    'mean': 0.0,
                    'std': 1.0,
                    'min': 0.0,
                    'max': 1.0,
                    'percentile_00_5': 0.0,
                    'percentile_99_5': 1.0
                }
        else:
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
        else:
            # For validation, we might still need skeleton transform
            self.transforms = self._create_validation_transforms()
            if self.transforms is not None:
                print("Validation transforms initialized")

        if not self.skip_patch_validation:
            self._get_valid_patches()
        else:
            print("Skipping patch validation as requested")
            # Generate all possible patches without validation
            self.valid_patches = []

            # Get the first target to access all volumes (since all targets share the same volumes)
            first_target = list(self.target_volumes.keys())[0]
            all_volumes = self.target_volumes[first_target]

            for vol_idx, volume_info in enumerate(all_volumes):
                # Get the image array (data) for this volume
                img_array = volume_info['data']['data']
                vol_name = volume_info.get('volume_id', f"volume_{vol_idx}")

                positions = self._get_all_sliding_window_positions(
                    volume_shape=img_array.shape,
                    patch_size=self.patch_size
                )
                for pos_dict in positions:
                    self.valid_patches.append({
                        "volume_index": vol_idx,
                        "volume_name": vol_name,
                        "position": pos_dict['start_pos']
                    })
            print(f"Generated {len(self.valid_patches)} patches without validation")

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

    def _needs_skeleton_transform(self):
        """
        Check if any configured loss requires skeleton data.
        
        Returns
        -------
        bool
            True if skeleton transform should be added to the pipeline
        """
        skeleton_losses = ['MedialSurfaceRecall', 'DC_SkelREC_and_CE_loss', 'SoftSkeletonRecallLoss']
        
        for target_name, target_info in self.targets.items():
            if "losses" in target_info:
                for loss_cfg in target_info["losses"]:
                    if loss_cfg["name"] in skeleton_losses:
                        return True
        return False

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

        bbox_threshold = getattr(self.mgr, 'min_bbox_percent', 0.97)
        downsample_level = getattr(self.mgr, 'downsample_level', 1)
        num_workers = getattr(self.mgr, 'num_workers', 4)

        if self.cache_enabled and len(self.zarr_arrays) > 0:
            cached_patches = load_cached_patches(
                train_data_paths=self.data_paths,
                label_paths=self.data_paths,
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
                "volume_name": patch["volume_name"],
                "position": patch["start_pos"]  # (z,y,x) for 3D or (y,x) for 2D
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
            
    def _extract_patch(self, patch_info):
        """
        Extract both image and label patches from the volume and return as tensors.
        
        Parameters
        ----------
        patch_info : dict
            Dictionary containing patch position and volume index information

        Returns
        -------
        dict
            Dictionary containing:
            - 'image': torch tensor of the image patch [C, H, W] or [C, D, H, W]
            - target labels: torch tensors for each target
            - 'is_unlabeled': bool indicating if patch has no valid labels
        """
        vol_idx = patch_info["volume_index"]

        # Extract coordinates based on dimensionality
        if self.is_2d_dataset:
            y, x = patch_info["position"]
            dy, dx = self.patch_size[-2:]  # Last 2 dimensions
            z, dz = 0, 0
            target_shape = (dy, dx)
        else:
            z, y, x = patch_info["position"]
            if len(self.patch_size) >= 3:
                dz, dy, dx = self.patch_size[:3]
            else:
                dy, dx = self.patch_size
                dz = 1
            target_shape = (dz, dy, dx)
        
        # Get the image data
        first_target_name = list(self.target_volumes.keys())[0]
        img_arr = self.target_volumes[first_target_name][vol_idx]['data']['data']
        
        # Extract image patch
        try:
            if self.is_2d_dataset:
                img_patch = img_arr[y:y+dy, x:x+dx]
                if img_patch.size == 0:
                    raise ValueError("Empty patch")
                img_patch = pad_or_crop_2d(img_patch, (dy, dx))
            else:
                img_patch = img_arr[z:z+dz, y:y+dy, x:x+dx]
                if img_patch.size == 0:
                    raise ValueError("Empty patch")
                img_patch = pad_or_crop_3d(img_patch, (dz, dy, dx))
        except Exception as e:
            print(f"Warning: Failed to extract image patch at vol={vol_idx}, z={z}, y={y}, x={x}: {str(e)}")
            img_patch = np.zeros(target_shape, dtype=np.float32)
        
        # Apply normalization
        if self.normalizer is not None:
            img_patch = self.normalizer.run(img_patch)
        else:
            img_patch = img_patch.astype(np.float32)
        
        # Add channel dimension
        img_patch = np.ascontiguousarray(img_patch[np.newaxis, ...])
        
        # Extract label patches and check if unlabeled
        label_patches = {}
        is_unlabeled = True  # Assume unlabeled until we find valid labels

        for t_name, volumes_list in self.target_volumes.items():
            label_arr = volumes_list[vol_idx]['data'].get('label')
            
            if label_arr is None:
                # No label exists - create zero array
                label_patch = np.zeros(target_shape, dtype=np.float32)
                # Add channel dimension for consistency with labeled data
                label_patch = label_patch[np.newaxis, ...]
            else:
                # Extract label patch
                has_channels = (self.is_2d_dataset and len(label_arr.shape) == 3) or \
                              (not self.is_2d_dataset and len(label_arr.shape) == 4)

                try:
                    if self.is_2d_dataset:
                        if has_channels:
                            label_patch = label_arr[:, y:y+dy, x:x+dx]
                        else:
                            label_patch = label_arr[y:y+dy, x:x+dx]

                        # Pad/crop
                        if has_channels:
                            n_channels = label_patch.shape[0]
                            padded = [pad_or_crop_2d(label_patch[c], (dy, dx)) for c in range(n_channels)]
                            label_patch = np.stack(padded, axis=0)
                        else:
                            label_patch = pad_or_crop_2d(label_patch, (dy, dx))
                    else:
                        if has_channels:
                            label_patch = label_arr[:, z:z+dz, y:y+dy, x:x+dx]
                        else:
                            label_patch = label_arr[z:z+dz, y:y+dy, x:x+dx]

                        # Pad/crop
                        if has_channels:
                            n_channels = label_patch.shape[0]
                            padded = [pad_or_crop_3d(label_patch[c], (dz, dy, dx)) for c in range(n_channels)]
                            label_patch = np.stack(padded, axis=0)
                        else:
                            label_patch = pad_or_crop_3d(label_patch, (dz, dy, dx))

                    # Check if patch has any non-zero labels
                    if np.any(label_patch > 0):
                        is_unlabeled = False

                except Exception as e:
                    print(f"Warning: Failed to extract label patch for {t_name}: {str(e)}")
                    label_patch = np.zeros(target_shape, dtype=np.float32)
                    # Mark that we need to add channel dimension
                    has_channels = False

                # Add channel dimension if needed
                if not has_channels:
                    label_patch = label_patch[np.newaxis, ...]

            label_patch = np.ascontiguousarray(label_patch, dtype=np.float32)
            label_patches[t_name] = label_patch


        # Build result dictionary with torch tensors
        result = {
            'image': torch.from_numpy(img_patch),
            'is_unlabeled': is_unlabeled
        }
        
        # Convert all label patches to tensors
        for t_name, label_patch in label_patches.items():
            result[t_name] = torch.from_numpy(label_patch)
        
        return result

    def _create_training_transforms(self):
        """
        Create training transforms using custom batchgeneratorsv2.
        Returns None for validation (no augmentations).
        """
        no_spatial = getattr(self.mgr, 'no_spatial', False)
            
        dimension = len(self.mgr.train_patch_size)

        if dimension == 2:
            patch_h, patch_w = self.mgr.train_patch_size
            patch_d = None  # Not used for 2D
        elif dimension == 3:
            patch_d, patch_h, patch_w = self.mgr.train_patch_size
        else:
            raise ValueError(f"Invalid patch size dimension: {dimension}. Expected 2 or 3.")

        transforms = []

        if not no_spatial:
            # Configure rotation based on patch aspect ratio
            if dimension == 2:
                if max(self.mgr.train_patch_size) / min(self.mgr.train_patch_size) > 1.5:
                    rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
                else:
                    rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
                mirror_axes = (0, 1)
            else:  # 3D
                rotation_for_DA = (-np.pi, np.pi)
                mirror_axes = (0, 1, 2)

            # Add SpatialTransform for rotations, scaling, elastic deformations
            transforms.append(
                SpatialTransform(
                    self.mgr.train_patch_size,
                    patch_center_dist_from_border=0,
                    random_crop=False,
                    p_elastic_deform=0,
                    p_rotation=0.5,
                    rotation=rotation_for_DA,
                    p_scaling=0.2,
                    scaling=(0.7, 1.4),
                    p_synchronize_scaling_across_axes=1,
                    bg_style_seg_sampling=False,  # =, mode_seg='nearest'
                    elastic_deform_magnitude=(10, 50)
                )
            )
            
            # Add morphological closing after spatial transform if needed for skeleton
            if self._needs_skeleton_transform():
                from vesuvius.models.augmentation.transforms.utils.morphological_closing import MorphologicalClosingTransform
                transforms.append(MorphologicalClosingTransform(structure_size=2))
                print("Added MorphologicalClosingTransform after spatial transforms")
            
            # # Add mirroring
            # transforms.append(RandomTransform(
            #     MirrorTransform(allowed_axes=mirror_axes),
            #     apply_probability=0.5
            # ))

        if dimension == 2:
            transforms.append(RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.5),
                    synchronize_channels=True,
                    synchronize_axes=False,
                    p_per_channel=1.0
                ),
                apply_probability=0.2
            ))
            transforms.append(RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.75, 1.25)),
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.3
            ))

            transforms.append(RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.5, 1.5)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.15
            ))
            transforms.append(RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.15),
                    p_per_channel=1,
                    synchronize_channels=True
                ), apply_probability=0.15
            ))
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.2
            ))

            rectangle_sizes_2d = tuple(
                (max(1, size // 10), size // 3) for size in self.mgr.train_patch_size
            )
            transforms.append(RandomTransform(
                BlankRectangleTransform(
                    rectangle_size=rectangle_sizes_2d,
                    rectangle_value=np.mean,
                    num_rectangles=(1, 5),
                    force_square=False,
                    p_per_sample=0.3,
                    p_per_channel=0.5
                ), apply_probability=0.2
            ))
        else:
            if not no_spatial:
                # Only add transpose transform if all three dimensions are equal
                if patch_d == patch_h == patch_w:
                    transforms.append(RandomTransform(
                        TransposeAxesTransform(allowed_axes={0, 1, 2}),
                        apply_probability=0.2
                    ))

            one_of_noise = OneOfTransform([
                GaussianNoiseTransform(
                    noise_variance=(0, 0.20),
                    p_per_channel=1,
                    synchronize_channels=True
                ),
                RicianNoiseTransform(
                    noise_variance=(0, 0.1),
                ),
                SmearTransform(
                    shift=(10, 0),
                    alpha=0.5,
                    num_prev_slices=1,
                    smear_axis=1
                )
            ])
            one_of_blur = OneOfTransform([
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.0),
                    synchronize_channels=True,
                    synchronize_axes=False,
                    p_per_channel=1.0
                ),
                SimulateLowResolutionTransform(
                    scale=(0.3, 1.5),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=None,
                    allowed_channels=None,
                    p_per_channel=0.5
                )
            ])
            one_of_intensity = OneOfTransform([
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.75, 1.25)),
                    synchronize_channels=False,
                    p_per_channel=1
                ),
                ContrastTransform(
                    contrast_range=BGContrast((0.50, 1.50)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1
                ),
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ),
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ),
                InhomogeneousSliceIlluminationTransform(
                    num_defects=(2, 5),
                    defect_width=(5, 20),
                    mult_brightness_reduction_at_defect=(0.3, 0.7),
                    base_p=(0.2, 0.4),
                    base_red=(0.5, 0.9),
                    p_per_sample=1.0,
                    per_channel=True,
                    p_per_channel=0.5
                )
            ])

            transforms.append(RandomTransform(
                one_of_noise,
                apply_probability=0.2
            ))
            transforms.append(RandomTransform(
                one_of_intensity,
                apply_probability=0.2
            ))
            transforms.append(RandomTransform(
                one_of_blur,
                apply_probability=0.2
            ))

            rectangle_sizes_3d = tuple(
                (max(1, size // 10), size // 3) for size in self.mgr.train_patch_size
            )
            transforms.append(RandomTransform(
                BlankRectangleTransform(
                    rectangle_size=rectangle_sizes_3d,
                    rectangle_value=np.mean,
                    num_rectangles=(1, 3),
                    force_square=False,
                    p_per_sample=0.4,
                    p_per_channel=0.5
                ), apply_probability=0.3
            ))

        if no_spatial:
            print("Spatial transformations disabled (no_spatial=True)")

        # Check if we need skeleton transform
        if self._needs_skeleton_transform():
            # Import here to avoid circular dependencies
            from vesuvius.models.augmentation.transforms.utils.skeleton_transform import MedialSurfaceTransform
            transforms.append(MedialSurfaceTransform(do_tube=False))
            print("Added MedialSurfaceTransform to training pipeline")

        return ComposeTransforms(transforms)
    
    def _create_validation_transforms(self):
        """
        Create validation transforms.
        For validation, we only apply skeleton transform if needed (no augmentations).
        """
        if not self._needs_skeleton_transform():
            return None
            
        # Import here to avoid circular dependencies
        from vesuvius.models.augmentation.transforms.utils.skeleton_transform import MedialSurfaceTransform
        
        transforms = []
        transforms.append(MedialSurfaceTransform(do_tube=False))
        print("Added MedialSurfaceTransform to validation pipeline")
        
        return ComposeTransforms(transforms)
    
    def __getitem__(self, index):
        """
        Returns a dictionary with the following format:
        {
            'image': torch.Tensor,           # Shape: [C, H, W] (2D) or [C, D, H, W] (3D)
            'is_unlabeled': bool,            # True if patch has no valid labels
            'target_name_1': torch.Tensor,   # Shape: [C, H, W] (2D) or [C, D, H, W] (3D)
            'target_name_2': torch.Tensor,   # Additional targets as configured
            ...                              # (e.g., 'ink', 'normals', 'distance_transform', etc.)
        }
        
        Where:
        - C = number of channels (usually 1 for image, can vary for targets)
        - H, W = height and width of patch
        - D = depth of patch (3D only)
        - All tensors are float32
        - Labels are 0 for background, >0 for foreground
        - Unlabeled patches will have all-zero label tensors
        """
        patch_info = self.valid_patches[index]
        data_dict = self._extract_patch(patch_info)

        if self.transforms is not None:
            data_dict = self.transforms(**data_dict)
        
        return data_dict

