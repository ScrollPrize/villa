import numpy as np
import zarr
from pathlib import Path
from collections import defaultdict
from .base_dataset import BaseDataset
from utils.io.zarr_io import _is_ome_zarr, _get_zarr_path

class ZarrDataset(BaseDataset):
    """
    A PyTorch Dataset for handling both 2D and 3D data from Zarr files.
    
    This dataset loads Zarr files which are already lazily loaded by design,
    supporting numpy array slicing without loading all data into memory.
    
    Supports both regular Zarr files and OME-Zarr files with multiple resolution levels.
    For OME-Zarr files, defaults to using resolution level 0 (highest resolution).
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
        if not labels_dir.exists():
            raise ValueError(f"Labels directory does not exist: {labels_dir}")
        
        # Get the configured targets
        configured_targets = set(self.mgr.targets.keys())
        print(f"Looking for configured targets: {configured_targets}")
        
        # Find all label directories to determine which images and targets we need
        label_dirs = [d for d in labels_dir.iterdir() if d.is_dir() and d.suffix == '.zarr']
        if not label_dirs:
            raise ValueError(f"No .zarr directories found in {labels_dir}")
        
        # Group files by target and image identifier
        targets_data = defaultdict(lambda: defaultdict(dict))
        
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
                # Get appropriate paths for OME-Zarr or regular zarr
                data_path = _get_zarr_path(image_dir)
                label_path = _get_zarr_path(label_dir)
                
                data_array = zarr.open(data_path, mode='r')
                label_array = zarr.open(label_path, mode='r')
                
                # Store in the nested dictionary - only include mask if it exists
                data_dict = {
                    'data': data_array,
                    'label': label_array
                }
                
                # Load mask if available
                if mask_dir.exists():
                    mask_path = _get_zarr_path(mask_dir)
                    mask_array = zarr.open(mask_path, mode='r')
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
        if missing_targets:
            raise ValueError(f"Configured targets not found in data: {missing_targets}")
        
        # Convert to the expected format for BaseDataset
        self.target_volumes = {}
        
        for target, images_dict in targets_data.items():
            self.target_volumes[target] = []
            
            for image_id, data_dict in images_dict.items():
                volume_info = {
                    'data': data_dict
                }
                self.target_volumes[target].append(volume_info)
            
            print(f"Target '{target}' has {len(self.target_volumes[target])} volumes")
        
        print(f"Total targets loaded: {list(self.target_volumes.keys())}")
