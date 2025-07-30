import numpy as np
import zarr
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from .base_dataset import BaseDataset
from vesuvius.utils.type_conversion import convert_to_uint8_dtype_range
import cv2
import tifffile

def convert_image_to_zarr_worker(args):
    """
    Worker function to convert a single image file to a Zarr array.
    """
    image_path, zarr_group_path, array_name, patch_size, pre_created = args
    
    try:
        if str(image_path).lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(str(image_path))
        else:
            img = cv2.imread(str(image_path))

        img = convert_to_uint8_dtype_range(img)
        group = zarr.open_group(str(zarr_group_path), mode='r+')
        
        if pre_created:
            group[array_name][:] = img
        else:
            if len(img.shape) == 2:  # 2D
                chunks = tuple(patch_size[:2])  # [h, w]
            else:  # 3D
                chunks = tuple(patch_size)  # [d, h, w]
            
            group.create_dataset(
                array_name,
                data=img,
                shape=img.shape,
                dtype=np.uint8,
                chunks=chunks,
                compressor=None,
                overwrite=True,
                write_empty_chunks=False
            )
        
        return array_name, img.shape, True, None
        
    except Exception as e:
        return array_name, None, False, str(e)

class ImageDataset(BaseDataset):
    def get_labeled_unlabeled_patch_indices(self):
        """Get indices of patches that are labeled vs unlabeled.

        Returns:
            labeled_indices: List of patch indices with labels
            unlabeled_indices: List of patch indices without labels
        """
        labeled_indices = []
        unlabeled_indices = []

        # First, let's understand the actual structure
        # Since all targets share the same volume indexing, check the first target
        first_target = list(self.target_volumes.keys())[0]

        for idx, patch_info in enumerate(self.valid_patches):
            vol_idx = patch_info['volume_index']

            # Get the volume info for this index
            if vol_idx < len(self.target_volumes[first_target]):
                volume_info = self.target_volumes[first_target][vol_idx]
                has_label = volume_info.get('has_label', False)

                if has_label:
                    labeled_indices.append(idx)
                else:
                    unlabeled_indices.append(idx)
            else:
                # This shouldn't happen, but let's be safe
                print(f"Warning: patch {idx} references volume {vol_idx} which doesn't exist")
                unlabeled_indices.append(idx)

        return labeled_indices, unlabeled_indices

    """
    A PyTorch Dataset for handling both 2D and 3D data from image files.
    
    - images.zarr/  (contains image1, image2, etc. as arrays)
    - labels.zarr/  (contains image1_task, image2_task, etc. as arrays)
    
    Expected directory structure:
    data_path/
    ├── images/
    │   ├── image1.tif          # Multi-task: single image for all tasks
    │   ├── image1_task.tif     # Single-task: task-specific image
    │   └── ...
    └── labels/
        ├── image1_task.tif     # Always task-specific
        └── ...
    """
    
    def _get_or_create_zarr_groups(self):

        images_zarr_path = self.data_path / "images.zarr"
        labels_zarr_path = self.data_path / "labels.zarr"

        images_group = zarr.open_group(str(images_zarr_path), mode='a')
        labels_group = zarr.open_group(str(labels_zarr_path), mode='a')
        
        return images_group, labels_group
    
    def _read_image_shape(self, image_path):
        """Read the shape of an image file without loading all data."""
        if str(image_path).lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(str(image_path))
            return img.shape
        else:
            img = cv2.imread(str(image_path))
            return img.shape
    
    def _needs_update(self, image_file, zarr_group, array_name):

        if array_name not in zarr_group:
            return True

        image_mtime = os.path.getmtime(image_file)
        group_store_path = Path(zarr_group.store.path)

        if group_store_path.exists():
            array_meta_path = group_store_path / array_name / ".zarray"
            if array_meta_path.exists():
                zarr_mtime = os.path.getmtime(array_meta_path)
                return image_mtime > zarr_mtime
        
        return True
    
    def _find_image_files(self, directory, extensions):
        """Find all image files with given extensions in a directory."""
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
        return files
    
    def _initialize_volumes(self):
        """Initialize volumes from image files, converting to Zarr format for fast access."""
        if not hasattr(self.mgr, 'data_path'):
            raise ValueError("ConfigManager must have 'data_path' attribute for image dataset")
        
        self.data_path = Path(self.mgr.data_path)
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        images_dir = self.data_path / "images"
        labels_dir = self.data_path / "labels"

        if not images_dir.exists():
            raise ValueError(f"Images directory does not exist: {images_dir}")

        has_labels = labels_dir.exists()
        if not has_labels and not self.allow_unlabeled_data:
            raise ValueError(f"Labels directory does not exist: {labels_dir} and allow_unlabeled_data=False")

        configured_targets = set(self.mgr.targets.keys())
        configured_targets = {t for t in configured_targets
                            if not self.mgr.targets.get(t, {}).get('auxiliary_task', False)}

        print(f"Looking for configured targets: {configured_targets}")

        self.target_volumes = {target: [] for target in configured_targets}
        self.zarr_arrays = []
        self.zarr_names = []
        self.data_paths = []

        images_group, labels_group = self._get_or_create_zarr_groups()

        supported_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        all_image_files = self._find_image_files(images_dir, supported_extensions)

        if not all_image_files:
            raise ValueError(f"No image files found in {images_dir}")

        print(f"Found {len(all_image_files)} image files")

        label_files = []
        if has_labels:
            label_files = self._find_image_files(labels_dir, supported_extensions)
            print(f"Found {len(label_files)} label files")

        conversion_tasks = []
        files_to_process = []

        if label_files:
            for label_file in label_files:
                stem = label_file.stem
                if '_' not in stem:
                    continue

                parts = stem.rsplit('_', 1)
                if len(parts) != 2:
                    continue

                image_id, target = parts

                if target not in configured_targets:
                    continue

                image_file = None
                for ext in supported_extensions:
                    test_file = images_dir / f"{image_id}{ext}"
                    if test_file.exists():
                        image_file = test_file
                        break

                if image_file is None:
                    for ext in supported_extensions:
                        test_file = images_dir / f"{image_id}_{target}{ext}"
                        if test_file.exists():
                            image_file = test_file
                            break

                if image_file is None:
                    print(f"Warning: No image found for label {stem}")
                    continue

                image_array_name = image_id if not image_file.stem.endswith(f"_{target}") else f"{image_id}_{target}"
                label_array_name = f"{image_id}_{target}"

                if self._needs_update(image_file, images_group, image_array_name):
                    shape = self._read_image_shape(image_file)
                    conversion_tasks.append((image_file, self.data_path / "images.zarr", image_array_name, self.patch_size, shape))

                if self._needs_update(label_file, labels_group, label_array_name):
                    shape = self._read_image_shape(label_file)
                    conversion_tasks.append((label_file, self.data_path / "labels.zarr", label_array_name, self.patch_size, shape))

                files_to_process.append((target, image_id, image_array_name, label_array_name))

        if self.allow_unlabeled_data:
            # Collect the image array names that already have labels
            labeled_image_names = {f[2] for f in files_to_process}

            for image_file in all_image_files:
                stem = image_file.stem

                # Skip if this image already has a label
                if stem in labeled_image_names:
                    continue

                # For unlabeled data, we need to include all images regardless of naming pattern
                image_array_name = stem

                if self._needs_update(image_file, images_group, image_array_name):
                    shape = self._read_image_shape(image_file)
                    conversion_tasks.append((image_file, self.data_path / "images.zarr", image_array_name, self.patch_size, shape))

                image_id = stem.split('_')[0] if '_' in stem else stem
                # Add this unlabeled image for each configured target
                for target in configured_targets:
                    files_to_process.append((target, image_id, image_array_name, None))

        if conversion_tasks:
            print(f"\nConverting {len(conversion_tasks)} image files to Zarr format...")
            print("Pre-creating Zarr arrays...")

            for file_path, zarr_path, array_name, patch_size, shape in conversion_tasks:

                group = images_group if "images.zarr" in str(zarr_path) else labels_group
                chunks = tuple(patch_size[:2]) if len(shape) == 2 else tuple(patch_size)

                group.create_dataset(
                    array_name,
                    shape=shape,
                    dtype=np.uint8,
                    chunks=chunks,
                    compressor=None,
                    overwrite=True,
                    write_empty_chunks=False
                )

            worker_tasks = [(f[0], f[1], f[2], f[3], True) for f in conversion_tasks]
            
            with ProcessPoolExecutor(max_workers=max(1, cpu_count() // 4)) as executor:
                futures = {executor.submit(convert_image_to_zarr_worker, task): task for task in worker_tasks}
                
                with tqdm(total=len(futures), desc="Converting to Zarr") as pbar:
                    for future in as_completed(futures):
                        array_name, shape, success, error_msg = future.result()
                        if not success:
                            print(f"ERROR converting {array_name}: {error_msg}")
                        pbar.update(1)
            
            print("✓ Conversion complete!")

        print("\nLoading Zarr arrays...")
        
        # Track which volume indices have labels
        # We need to track per target since each target can have different labeled/unlabeled volumes
        self.volume_has_label = {}  # Maps (target, volume_idx) to bool

        for target, image_id, image_array_name, label_array_name in files_to_process:
            data_array = images_group[image_array_name]
            label_array = labels_group[label_array_name] if label_array_name and label_array_name in labels_group else None

            data_dict = {
                'data': data_array,
                'label': label_array
            }

            volume_idx = len(self.target_volumes[target])
            self.target_volumes[target].append({
                'data': data_dict,
                'volume_id': image_id,
                'has_label': label_array is not None
            })
            
            # Track if this volume has a label for this target
            self.volume_has_label[(target, volume_idx)] = (label_array is not None)

            if label_array is not None:
                self.zarr_arrays.append(label_array)
                self.zarr_names.append(f"{image_id}_{target}")
                self.data_paths.append(str(self.data_path / "labels.zarr" / label_array_name))

        for target, volumes in self.target_volumes.items():
            print(f"Target '{target}' has {len(volumes)} volumes")

        if not any(len(vols) > 0 for vols in self.target_volumes.values()):
            raise ValueError("No data found for any configured targets")

        print(f"Total targets loaded: {list(self.target_volumes.keys())}")

        # Count labeled vs unlabeled
        labeled_count = sum(1 for has_label in self.volume_has_label.values() if has_label)
        unlabeled_count = sum(1 for has_label in self.volume_has_label.values() if not has_label)
        print(f"Labeled volume entries: {labeled_count}, Unlabeled volume entries: {unlabeled_count}")