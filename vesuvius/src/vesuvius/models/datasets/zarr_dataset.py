import numpy as np
import zarr
from pathlib import Path
from collections import defaultdict
from .base_dataset import BaseDataset
from vesuvius.utils.io.zarr_io import _is_ome_zarr

class ZarrDataset(BaseDataset):
    """
    A PyTorch Dataset for handling both 2D and 3D data from Zarr files.
    """
    def _initialize_volumes(self):
        """
        Initialize volumes from Zarr files. Expected directory structure:

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

        configured_targets = set(self.mgr.targets.keys())
        print(f"Looking for configured targets: {configured_targets}")
        
        # Find all label directories to determine which images and targets we need
        label_dirs = [d for d in labels_dir.iterdir() if d.is_dir() and d.suffix == '.zarr']
        if not label_dirs:
            raise ValueError(f"No .zarr directories found in {labels_dir}")

        targets_data = defaultdict(lambda: defaultdict(dict))

        for label_dir in label_dirs:
            stem = label_dir.stem

            if '_' not in stem:
                print(f"Skipping label directory without underscore: {label_dir.name}")
                continue

            parts = stem.rsplit('_', 1)
            if len(parts) != 2:
                print(f"Invalid label directory name format: {label_dir.name}")
                continue
            
            image_id, target = parts

            if target not in configured_targets:
                print(f"Skipping {image_id}_{target} - not in configured targets")
                continue

            image_dir = images_dir / f"{image_id}.zarr"

            if not image_dir.exists():
                image_dir = images_dir / f"{image_id}_{target}.zarr"
                if not image_dir.exists():
                    raise ValueError(f"Image directory not found for {image_id} (tried {image_id}.zarr and {image_id}_{target}.zarr)")

            mask_dir = masks_dir / f"{image_id}_{target}.zarr"

            try:
                # Resolve symlinks if needed
                resolved_image_path = Path(image_dir).resolve()
                resolved_label_path = Path(label_dir).resolve()

                if _is_ome_zarr(resolved_image_path):
                    root = zarr.open_group(str(resolved_image_path), mode='r')
                    if '0' in root:
                        data_array = root['0']
                    else:
                        data_array = zarr.open(str(resolved_image_path), mode='r')
                else:
                    data_array = zarr.open(str(resolved_image_path), mode='r')

                if _is_ome_zarr(resolved_label_path):
                    root = zarr.open_group(str(resolved_label_path), mode='r')
                    if '0' in root:
                        label_array = root['0']
                    else:
                        label_array = zarr.open(str(resolved_label_path), mode='r')
                else:
                    label_array = zarr.open(str(resolved_label_path), mode='r')

                data_dict = {
                    'data': data_array,
                    'label': label_array
                }

                if mask_dir.exists():
                    resolved_mask_path = Path(mask_dir).resolve()

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

                if _is_ome_zarr(image_dir):
                    resolution = getattr(self.mgr, 'ome_zarr_resolution', 0)
                    print(f"Registered {image_id}_{target} with shape {data_array.shape} (OME-Zarr, resolution level {resolution})")
                else:
                    print(f"Registered {image_id}_{target} with shape {data_array.shape} (regular zarr)")
                
            except Exception as e:
                raise ValueError(f"Error opening zarr directories: {e}")

        found_targets = set(targets_data.keys())
        # Filter out auxiliary tasks from the check - they are generated dynamically
        non_auxiliary_targets = {t for t in configured_targets 
                                if not self.mgr.targets.get(t, {}).get('auxiliary_task', False)}
        missing_targets = non_auxiliary_targets - found_targets
        if missing_targets:
            raise ValueError(f"Configured targets not found in data: {missing_targets}")

        self.target_volumes = {}

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
