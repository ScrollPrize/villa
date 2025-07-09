import numpy as np
import zarr
from pathlib import Path
from collections import defaultdict
from .base_dataset import BaseDataset
from utils.io.zarr_io import _is_ome_zarr

class ZarrDataset(BaseDataset):
    """
    A PyTorch Dataset for handling both 2D and 3D data from Zarr files.
    
    This dataset loads Zarr files which are already lazily loaded by design,
    supporting numpy array slicing without loading all data into memory.
    
    Supports both regular Zarr files and OME-Zarr files with multiple resolution levels.
    For OME-Zarr files, defaults to using resolution level 0 (highest resolution).
    
    Can optionally load approved patches from vc_proofreader instead of computing patches automatically.
    """
    
    
    def _initialize_mae_volumes(self, images_dir):
        """Initialize volumes for MAE pretraining - only loads images."""
        # Check if data_paths is specified in dataset_config
        data_paths = getattr(self.mgr, 'dataset_config', {}).get('data_paths', None)
        
        if data_paths:
            # Use specified paths directly
            print(f"Using {len(data_paths)} specified data paths for MAE pretraining")
            image_paths = data_paths
        else:
            # Fall back to original behavior - scan images directory
            if not images_dir.exists():
                raise ValueError(f"Images directory not found: {images_dir}")
            
            # Find all image zarr directories
            image_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir() and d.suffix == '.zarr'])
            
            if not image_dirs:
                raise ValueError(f"No zarr directories found in {images_dir}")
            
            print(f"Found {len(image_dirs)} image volumes for MAE pretraining")
            image_paths = [str(d) for d in image_dirs]
        
        # For MAE, we create a fake "reconstruction" target
        self.target_volumes = {'reconstruction': []}
        self.volume_ids = {'reconstruction': []}
        
        # Load each image volume
        for idx, image_path in enumerate(image_paths):
            # Determine path type and extract image ID
            if image_path.startswith('s3://') or image_path.startswith('http://') or image_path.startswith('https://'):
                # For remote paths, extract ID from the last part of the path
                path_parts = image_path.rstrip('/').split('/')
                image_id = path_parts[-1].replace('.zarr', '')
                is_remote = True
            else:
                # For local paths, use Path object
                path_obj = Path(image_path)
                image_id = path_obj.stem  # Remove .zarr extension
                is_remote = False
            
            try:
                # Open zarr directly - zarr-python handles S3/HTTP paths natively
                if is_remote:
                    # For remote paths, check if it's OME-Zarr by trying to open as group first
                    try:
                        root = zarr.open_group(image_path, mode='r')
                        # Check if data is in a subgroup (common OME-Zarr structure)
                        if '0' in root:
                            data_array = root['0']
                            is_ome = True
                        else:
                            # If no standard subgroups, use the root array
                            data_array = zarr.open(image_path, mode='r')
                            is_ome = False
                    except:
                        # If opening as group fails, try as array
                        data_array = zarr.open(image_path, mode='r')
                        is_ome = False
                else:
                    # For local paths, use existing logic
                    resolved_path = Path(image_path).resolve()
                    
                    # Open zarr directly - handle OME-Zarr structure
                    if _is_ome_zarr(resolved_path):
                        # For OME-Zarr, try to open the root group first
                        root = zarr.open_group(str(resolved_path), mode='r')
                        # Check if data is in a subgroup (common OME-Zarr structure)
                        if '0' in root:
                            data_array = root['0']
                        else:
                            # If no standard subgroups, use the root array
                            data_array = zarr.open(str(resolved_path), mode='r')
                        is_ome = True
                    else:
                        # Regular zarr
                        data_array = zarr.open(str(resolved_path), mode='r')
                        is_ome = False
                
                # For MAE, we only need the image data
                # Use the image data as the "label" to satisfy BaseDataset's dimensionality check
                volume_info = {
                    'data': {
                        'data': data_array,
                        'label': data_array,  # Use image data to satisfy BaseDataset
                        'mask': None    # No masks for MAE
                    },
                    'volume_id': image_id,
                    'zarr_path': image_path  # Store the original path for later use
                }
                
                self.target_volumes['reconstruction'].append(volume_info)
                self.volume_ids['reconstruction'].append(image_id)
                
                # Print information about the loaded array
                path_type = "remote" if is_remote else "local"
                zarr_type = "OME-Zarr" if is_ome else "regular zarr"
                print(f"Loaded {image_id} for MAE with shape {data_array.shape} ({zarr_type}, {path_type} path)")
                    
            except Exception as e:
                raise ValueError(f"Error opening zarr path {image_path}: {e}")
        
        print(f"Total volumes loaded for MAE pretraining: {len(self.target_volumes['reconstruction'])}")
    
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
        # Check if we're in MAE mode - if so, only load images
        mae_mode = getattr(self.mgr, 'model_config', {}).get('mae_mode', False)
        if mae_mode:
            # For MAE mode, check if data_paths is specified
            data_paths = getattr(self.mgr, 'dataset_config', {}).get('data_paths', None)
            if data_paths:
                # Use data_paths directly, no need for data_path
                self._initialize_mae_volumes(None)
                return
            else:
                # Fall back to traditional approach - need data_path
                if not hasattr(self.mgr, 'data_path'):
                    raise ValueError("ConfigManager must have 'data_path' attribute for Zarr dataset (or use 'data_paths' in dataset_config for MAE mode)")
                data_path = Path(self.mgr.data_path)
                if not data_path.exists():
                    raise ValueError(f"Data path does not exist: {data_path}")
                images_dir = data_path / "images"
                self._initialize_mae_volumes(images_dir)
                return
        
        # For non-MAE mode, we always need data_path
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
        if missing_targets:
            raise ValueError(f"Configured targets not found in data: {missing_targets}")
        
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
