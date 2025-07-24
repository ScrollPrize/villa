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
    
    def _image_to_zarr_array(self, image_path, zarr_group, array_name):

        if str(image_path).lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(str(image_path))
        else:
            img = cv2.imread(str(image_path))

        img = convert_to_uint8_dtype_range(img)

        if len(img.shape) == 2:  # 2D
            chunks = tuple(self.patch_size[:2])  # [h, w]
        else:  # 3D
            chunks = tuple(self.patch_size)  # [d, h, w]
        
        z_array = zarr_group.create_dataset(
            array_name,
            data=img,
            shape=img.shape,
            dtype=np.uint8,
            chunks=chunks,
            compressor=None,
            overwrite=True,
            write_empty_chunks=False
        )
        
        return z_array
    
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
    
    def _ensure_zarr_array(self, image_file, zarr_group, array_name):

        if array_name in zarr_group:
            # Check if we need to update (image is newer)
            image_mtime = os.path.getmtime(image_file)
            
            # For groups, we check the group's store path modification time
            group_store_path = Path(zarr_group.store.path)
            if group_store_path.exists():
                array_meta_path = group_store_path / array_name / ".zarray"
                if array_meta_path.exists():
                    zarr_mtime = os.path.getmtime(array_meta_path)
                    if zarr_mtime >= image_mtime:
                        return zarr_group[array_name]

        print(f"Converting {image_file.name} to Zarr format...")
        return self._image_to_zarr_array(image_file, zarr_group, array_name)
    
    def _initialize_volumes(self):
        """
        Initialize volumes from image files, converting to Zarr format for fast access.
        
        Expected directory structure:
        
        For multi-task scenarios:
        data_path/
        ├── images/
        │   ├── image1.tif      # Single image file
        │   ├── image2.tif      # Single image file
        │   └── ...
        ├── labels/
            ├── image1_ink.tif
            ├── image1_damage.tif
            ├── image2_ink.tif
            ├── image2_damage.tif
            └── ...
            
        For single-task scenarios:
        data_path/
        ├── images/
        │   ├── image1_ink.tif
        │   ├── image2_ink.tif
        │   └── ...
        ├── labels/
            ├── image1_ink.tif
            ├── image2_ink.tif
            └── ...

        """
        if not hasattr(self.mgr, 'data_path'):
            raise ValueError("ConfigManager must have 'data_path' attribute for image dataset")
        
        self.data_path = Path(self.mgr.data_path)
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        images_dir = self.data_path / "images"
        labels_dir = self.data_path / "labels"

        if not images_dir.exists():
            raise ValueError(f"Images directory does not exist: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Labels directory does not exist: {labels_dir}")

        images_group, labels_group = self._get_or_create_zarr_groups()
        configured_targets = set(self.mgr.targets.keys())
        print(f"Looking for configured targets: {configured_targets}")

        supported_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        label_files = []
        for ext in supported_extensions:
            found_files = list(labels_dir.glob(f"*{ext}"))
            label_files.extend(found_files)

        print(f"Total label files found: {len(label_files)}")
        if not label_files:
            raise ValueError(f"No image files found in {labels_dir} with supported extensions: {supported_extensions}")


        targets_data = defaultdict(lambda: defaultdict(dict))
        files_to_process = []

        for idx, label_file in enumerate(label_files):
            stem = label_file.stem  # Remove image extension

            if '_' not in stem:
                print(f"Skipping label file without underscore: {label_file.name}")
                continue
            
            # Split on the last underscore to handle cases like "image1_test_ink"
            parts = stem.rsplit('_', 1)
            if len(parts) != 2:
                print(f"Invalid label filename format: {label_file.name}")
                continue
            
            image_id, target = parts
            
            # Only process targets that are in the configuration
            if target not in configured_targets:
                print(f"Skipping {image_id}_{target} - not in configured targets")
                continue

            label_ext = label_file.suffix

            image_file = None
            for ext in [label_ext] + supported_extensions:
                test_file = images_dir / f"{image_id}{ext}"
                if test_file.exists():
                    image_file = test_file
                    break

            if image_file is None:
                for ext in [label_ext] + supported_extensions:
                    test_file = images_dir / f"{image_id}_{target}{ext}"
                    if test_file.exists():
                        image_file = test_file
                        break
            
            if image_file is None:
                tried_names = [f"{image_id}{ext}" for ext in supported_extensions]
                tried_names.extend([f"{image_id}_{target}{ext}" for ext in supported_extensions])
                raise ValueError(f"Image file not found for {image_id} (tried {', '.join(tried_names)})")

            
            files_to_process.append((target, image_id, image_file, label_file))

        conversion_tasks = []
        array_info = {}  # Track which arrays go where
        arrays_to_create = []  # Track arrays that need pre-creation
        
        print(f"Found {len(files_to_process)} files to process")
        for idx, (target, image_id, image_file, label_file) in enumerate(files_to_process):
            # Determine array names
            if image_file.stem.endswith(f"_{target}"):
                image_array_name = f"{image_id}_{target}"
            else:
                image_array_name = image_id

            images_zarr_path = self.data_path / "images.zarr"
            labels_zarr_path = self.data_path / "labels.zarr"

            if image_array_name not in images_group or self._needs_update(image_file, images_group, image_array_name):
                try:
                    if str(image_file).lower().endswith(('.tif', '.tiff')):
                        img_shape = tifffile.imread(str(image_file)).shape
                    else:
                        img_shape = cv2.imread(str(image_file)).shape
                except Exception as e:
                    print(f"ERROR [{idx}]: Failed to read {image_file.name}: {str(e)}")
                    raise
                arrays_to_create.append((images_group, image_array_name, img_shape))
                conversion_tasks.append((image_file, images_zarr_path, image_array_name, self.patch_size, True))

            label_array_name = f"{image_id}_{target}"
            if label_array_name not in labels_group or self._needs_update(label_file, labels_group, label_array_name):
                # Read shape for pre-creation
                if str(label_file).lower().endswith(('.tif', '.tiff')):
                    label_shape = tifffile.imread(str(label_file)).shape
                else:
                    label_shape = cv2.imread(str(label_file)).shape
                arrays_to_create.append((labels_group, label_array_name, label_shape))
                conversion_tasks.append((label_file, labels_zarr_path, label_array_name, self.patch_size, True))

            array_info[(target, image_id)] = {
                'image_array_name': image_array_name,
                'label_array_name': label_array_name
            }

        if conversion_tasks:
            print(f"\nConverting {len(conversion_tasks)} image files to Zarr format...")

            print("Pre-creating Zarr array structure...")
            for group, array_name, shape in arrays_to_create:
                if len(shape) == 2:  # 2D
                    chunks = tuple(self.patch_size[:2])
                else:  # 3D
                    chunks = tuple(self.patch_size)
                
                # Create empty array
                group.create_dataset(
                    array_name,
                    shape=shape,
                    dtype=np.uint8,
                    chunks=chunks,
                    compressor=None,
                    overwrite=True,
                    write_empty_chunks=False
                )
            
            print(f"Using {cpu_count()} workers for parallel conversion...")
            
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                # Submit all tasks
                futures = {executor.submit(convert_image_to_zarr_worker, task): task for task in conversion_tasks}
                
                # Process completed tasks with progress bar
                with tqdm(total=len(futures), desc="Converting images to Zarr") as pbar:
                    for future in as_completed(futures):
                        array_name, shape, success, error_msg = future.result()
                        
                        if success:
                            pbar.set_description(f"Converted {array_name}")
                        else:
                            print(f"ERROR converting {array_name}: {error_msg}")
                        
                        pbar.update(1)
            
            print("✓ Conversion complete!")
        else:
            print("✓ All Zarr arrays are up to date!")

        print("\nLoading Zarr arrays...")
        
        for target, image_id, image_file, label_file in files_to_process:
            info = array_info[(target, image_id)]

            data_array = images_group[info['image_array_name']]
            label_array = labels_group[info['label_array_name']]

            data_dict = {
                'data': data_array,
                'label': label_array
            }

            targets_data[target][image_id] = data_dict
            print(f"Loaded {image_id}_{target} with shape {data_array.shape}")

        found_targets = set(targets_data.keys())
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
        
        print(f"\nTotal targets loaded: {list(self.target_volumes.keys())}")
        print("✓ Zarr cache ready")
