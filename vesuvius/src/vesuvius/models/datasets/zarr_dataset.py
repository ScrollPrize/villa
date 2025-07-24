import zarr
from pathlib import Path
from .base_dataset import BaseDataset
from vesuvius.utils.io.zarr_io import _is_ome_zarr

class ZarrDataset(BaseDataset):
    """
    A PyTorch Dataset for loading 2D and 3D data from Zarr files. Expected directory structure:
    data_path/
    ├── images/
    │   ├── image1.zarr/          # Multi-task: single image for all tasks
    │   ├── image1_task.zarr/     # Single-task: task-specific image
    │   └── ...
    ├── labels/
        ├── image1_task.zarr/     # Always task-specific
        └── ...
    """

    def _open_zarr_array(self, path, resolution=0):
        """Open a zarr array, handling OME-Zarr format with resolution levels."""
        if _is_ome_zarr(path):
            root = zarr.open_group(str(path), mode='r')
            # Try to access the specified resolution level
            resolution_key = str(resolution)
            if resolution_key in root:
                return root[resolution_key]
            elif '0' in root:
                print(f"Warning: Resolution {resolution} not found, falling back to resolution 0")
                return root['0']
            else:
                # Fallback to opening as regular zarr if no resolution levels
                return zarr.open(str(path), mode='r')
        else:
            return zarr.open(str(path), mode='r')

    def _initialize_volumes(self):
        if not hasattr(self.mgr, 'data_path'):
            raise ValueError("ConfigManager must have 'data_path' attribute for Zarr dataset")
        
        data_path = Path(self.mgr.data_path)
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        images_dir = data_path / "images"
        labels_dir = data_path / "labels"

        if not images_dir.exists():
            raise ValueError(f"Images directory does not exist: {images_dir}")

        has_labels = labels_dir.exists()
        if not has_labels and not self.allow_unlabeled_data:
            raise ValueError(f"Labels directory does not exist: {labels_dir} and allow_unlabeled_data=False")

        # Get OME-Zarr resolution level from config (default to 0)
        self.resolution = getattr(self.mgr, 'ome_zarr_resolution', 0)
        if self.resolution != 0:
            print(f"Using OME-Zarr resolution level: {self.resolution}")

        configured_targets = set(self.mgr.targets.keys())
        configured_targets = {t for t in configured_targets
                            if not self.mgr.targets.get(t, {}).get('auxiliary_task', False)}

        print(f"Looking for configured targets: {configured_targets}")

        self.target_volumes = {target: [] for target in configured_targets}
        self.zarr_arrays = []
        self.zarr_names = []
        self.data_paths = []

        image_zarrs = {d.stem: d for d in images_dir.iterdir()
                      if d.is_dir() and d.suffix == '.zarr'}

        if not image_zarrs:
            raise ValueError(f"No .zarr directories found in {images_dir}")

        for target in configured_targets:
            volume_count = 0

            if has_labels:
                label_zarrs = [d for d in labels_dir.iterdir()
                             if d.is_dir() and d.suffix == '.zarr' and d.stem.endswith(f'_{target}')]

                for label_dir in label_zarrs:
                    stem = label_dir.stem
                    image_id = stem.rsplit('_', 1)[0]

                    image_dir = image_zarrs.get(image_id) or image_zarrs.get(f"{image_id}_{target}")
                    if not image_dir:
                        print(f"Warning: No image found for label {stem}")
                        continue

                    data_array = self._open_zarr_array(image_dir, self.resolution)
                    label_array = self._open_zarr_array(label_dir, self.resolution)

                    data_dict = {
                        'data': data_array,
                        'label': label_array
                    }

                    self.target_volumes[target].append({
                        'data': data_dict,
                        'volume_id': image_id
                    })

                    self.zarr_arrays.append(label_array)
                    self.zarr_names.append(f"{image_id}_{target}")
                    self.data_paths.append(str(label_dir))

                    volume_count += 1

            if volume_count == 0 and self.allow_unlabeled_data:
                print(f"No labeled data found for target '{target}', using unlabeled images")

                for image_id, image_dir in image_zarrs.items():
                    if '_' in image_id and not image_id.endswith(f'_{target}'):
                        continue

                    data_array = self._open_zarr_array(image_dir, self.resolution)

                    self.target_volumes[target].append({
                        'data': {
                            'data': data_array,
                            'label': None
                        },
                        'volume_id': image_id.split('_')[0]
                    })
                    volume_count += 1
            
            print(f"Target '{target}' has {volume_count} volumes")

        if not any(len(vols) > 0 for vols in self.target_volumes.values()):
            raise ValueError("No data found for any configured targets")
        
        print(f"Total targets loaded: {list(self.target_volumes.keys())}")
