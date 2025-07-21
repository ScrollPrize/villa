import numpy as np
import zarr
from pathlib import Path
from collections import defaultdict
from .base_dataset import BaseDataset
from vesuvius.utils.io.zarr_io import _is_ome_zarr

class ZarrDataset(BaseDataset):
    """
    A PyTorch Dataset for handling both 2D and 3D data from Zarr files.
    
    This dataset loads Zarr files which are already lazily loaded by design,
    supporting numpy array slicing without loading all data into memory.
    
    Supports both regular Zarr files and OME-Zarr files with multiple resolution levels.
    For OME-Zarr files, defaults to using resolution level 0 (highest resolution).
    
    Can optionally load approved patches from vc_proofreader instead of computing patches automatically.
    """
    
    def _initialize_volumes(self):
        """
        Initialize volumes from Zarr files.
        
        Expected directory structure:
        
        For multi-task scenarios:
        data_path/
        ├── images/
        │   ├── image1.zarr/      # Single image directory
        │   ├── image2.zarr/      # Single image directory
        │   └── ...
        ├── labels/
        │   ├── image1_ink.zarr/
        │   ├── image1_damage.zarr/
        │   ├── image2_ink.zarr/
        │   ├── image2_damage.zarr/
        │   └── ...
        └── masks/
            ├── image1_ink.zarr/
            ├── image1_damage.zarr/
            ├── image2_ink.zarr/
            ├── image2_damage.zarr/
            └── ...
            
        For single-task scenarios:
        data_path/
        ├── images/
        │   ├── image1_ink.zarr/
        │   ├── image2_ink.zarr/
        │   └── ...
        ├── labels/
        │   ├── image1_ink.zarr/
        │   ├── image2_ink.zarr/
        │   └── ...
        └── masks/
            ├── image1_ink.zarr/
            ├── image2_ink.zarr/
            └── ...
        """
        # Always need data_path
        if not hasattr(self.mgr, 'data_path'):
            raise ValueError("ConfigManager must have 'data_path' attribute for Zarr dataset")
        
        data_path = Path(self.mgr.data_path)
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        images_dir = data_path / "images"
        labels_dir = data_path / "labels"
        masks_dir = data_path / "masks"
        
        # Check required directories exist
        if not images_dir.exists():
            raise ValueError(f"Images directory does not exist: {images_dir}")
        
        # Labels directory is optional when allow_unlabeled_data is True
        has_labels_dir = labels_dir.exists()
        if not has_labels_dir and not self.allow_unlabeled_data:
            raise ValueError(f"Labels directory does not exist: {labels_dir} and allow_unlabeled_data=False")
        elif not has_labels_dir:
            print(f"Labels directory not found, loading unlabeled data only")
        
        # Get the configured targets
        configured_targets = set(self.mgr.targets.keys())
        print(f"Looking for configured targets: {configured_targets}")
        
        # Find all label directories if they exist
        if has_labels_dir:
            label_dirs = [d for d in labels_dir.iterdir() if d.is_dir() and d.suffix == '.zarr']
            if not label_dirs and not self.allow_unlabeled_data:
                raise ValueError(f"No .zarr directories found in {labels_dir} and allow_unlabeled_data=False")
        else:
            label_dirs = []
        
        # Group files by target and image identifier
        targets_data = defaultdict(lambda: defaultdict(dict))
        
        # If no labels but unlabeled data is allowed, discover images directly
        if not label_dirs and self.allow_unlabeled_data:
            print("No label directories found, discovering unlabeled images...")
            image_dirs = [d for d in images_dir.iterdir() if d.is_dir() and d.suffix == '.zarr']
            
            if not image_dirs:
                raise ValueError(f"No .zarr directories found in {images_dir}")
            
            # Process each image directory as unlabeled
            for image_dir in image_dirs:
                stem = image_dir.stem  # Remove .zarr extension
                
                # For unlabeled data, we'll create entries for each configured target
                for target in configured_targets:
                    # Skip auxiliary tasks
                    if self.mgr.targets.get(target, {}).get('auxiliary_task', False):
                        continue
                    
                    # Open zarr arrays
                    try:
                        resolved_image_path = Path(image_dir).resolve()
                        
                        # Open image zarr
                        if _is_ome_zarr(resolved_image_path):
                            root = zarr.open_group(str(resolved_image_path), mode='r')
                            if '0' in root:
                                data_array = root['0']
                            else:
                                data_array = zarr.open(str(resolved_image_path), mode='r')
                        else:
                            data_array = zarr.open(str(resolved_image_path), mode='r')
                        
                        # Store with None for label to indicate unlabeled
                        data_dict = {
                            'data': data_array,
                            'label': None  # None indicates unlabeled data
                        }
                        
                        targets_data[target][stem] = data_dict
                        print(f"Registered unlabeled image {stem} for target {target} with shape {data_array.shape}")
                        
                    except Exception as e:
                        print(f"Warning: Could not open zarr {image_dir}: {e}")
                        continue
        
        # Process each label directory
        for label_dir in label_dirs:
            stem = label_dir.stem  # Remove .zarr extension
            
            # Parse label directory name: image1_ink.zarr -> image_id="image1", target="ink"
            if '_' not in stem:
                print(f"Skipping label directory without underscore: {label_dir.name}")
                continue
            
            # Split on the last underscore to handle cases like "image1_test_ink"
            parts = stem.rsplit('_', 1)
            if len(parts) != 2:
                print(f"Invalid label directory name format: {label_dir.name}")
                continue
            
            image_id, target = parts
            
            # Only process targets that are in the configuration
            if target not in configured_targets:
                print(f"Skipping {image_id}_{target} - not in configured targets")
                continue
            
            # Look for corresponding image directory
            # First try without task suffix (multi-task scenario)
            image_dir = images_dir / f"{image_id}.zarr"
            
            # If not found, try with task suffix (single-task/backward compatibility)
            if not image_dir.exists():
                image_dir = images_dir / f"{image_id}_{target}.zarr"
                if not image_dir.exists():
                    raise ValueError(f"Image directory not found for {image_id} (tried {image_id}.zarr and {image_id}_{target}.zarr)")
            
            # Look for mask directory (always with task suffix)
            mask_dir = masks_dir / f"{image_id}_{target}.zarr"
            
            # Open zarr arrays - these are already lazily loaded
            try:
                # Resolve symlinks if needed
                resolved_image_path = Path(image_dir).resolve()
                resolved_label_path = Path(label_dir).resolve()
                
                # Open zarr directly - handle OME-Zarr structure for images
                if _is_ome_zarr(resolved_image_path):
                    root = zarr.open_group(str(resolved_image_path), mode='r')
                    if '0' in root:
                        data_array = root['0']
                    else:
                        data_array = zarr.open(str(resolved_image_path), mode='r')
                else:
                    data_array = zarr.open(str(resolved_image_path), mode='r')
                
                # Open label zarr
                if _is_ome_zarr(resolved_label_path):
                    root = zarr.open_group(str(resolved_label_path), mode='r')
                    if '0' in root:
                        label_array = root['0']
                    else:
                        label_array = zarr.open(str(resolved_label_path), mode='r')
                else:
                    label_array = zarr.open(str(resolved_label_path), mode='r')
                
                # Store in the nested dictionary - only include mask if it exists
                data_dict = {
                    'data': data_array,
                    'label': label_array
                }
                
                # Load mask if available
                if mask_dir.exists():
                    # Resolve symlinks if needed
                    resolved_mask_path = Path(mask_dir).resolve()
                    
                    # Open mask zarr
                    if _is_ome_zarr(resolved_mask_path):
                        root = zarr.open_group(str(resolved_mask_path), mode='r')
                        if '0' in root:
                            mask_array = root['0']
                        else:
                            mask_array = zarr.open(str(resolved_mask_path), mode='r')
                    else:
                        mask_array = zarr.open(str(resolved_mask_path), mode='r')
                    data_dict['mask'] = mask_array
                    print(f"Found mask for {image_id}_{target}")
                else:
                    print(f"No mask directory found for {image_id}_{target}, will use no mask")
                
                targets_data[target][image_id] = data_dict
                
                # Print information about the loaded arrays
                if _is_ome_zarr(image_dir):
                    resolution = getattr(self.mgr, 'ome_zarr_resolution', 0)
                    print(f"Registered {image_id}_{target} with shape {data_array.shape} (OME-Zarr, resolution level {resolution})")
                else:
                    print(f"Registered {image_id}_{target} with shape {data_array.shape} (regular zarr)")
                
            except Exception as e:
                raise ValueError(f"Error opening zarr directories: {e}")
        
        # Check that all configured targets were found (excluding auxiliary tasks)
        found_targets = set(targets_data.keys())
        # Filter out auxiliary tasks from the check - they are generated dynamically
        non_auxiliary_targets = {t for t in configured_targets 
                                if not self.mgr.targets.get(t, {}).get('auxiliary_task', False)}
        missing_targets = non_auxiliary_targets - found_targets
        if missing_targets and not self.allow_unlabeled_data:
            raise ValueError(f"Configured targets not found in data: {missing_targets}")
        elif missing_targets:
            print(f"Warning: Some configured targets not found in labeled data: {missing_targets}")
            print("These targets will only have unlabeled data.")
        
        # Convert to the expected format for BaseDataset
        self.target_volumes = {}
        
        # Also store volume IDs in order for each target
        self.volume_ids = {}
        
        for target, images_dict in targets_data.items():
            self.target_volumes[target] = []
            self.volume_ids[target] = []
            
            for image_id, data_dict in images_dict.items():
                volume_info = {
                    'data': data_dict,
                    'volume_id': image_id  # Store the volume ID
                }
                self.target_volumes[target].append(volume_info)
                self.volume_ids[target].append(image_id)
            
            print(f"Target '{target}' has {len(self.target_volumes[target])} volumes")
        
        print(f"Total targets loaded: {list(self.target_volumes.keys())}")
