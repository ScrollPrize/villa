from typing import Tuple, Union, List
import os
import zarr
import fsspec
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.morphology import dilation, ball
import albumentations as A
from pathlib import Path
from tqdm import tqdm

from helpers import find_valid_patches

class ZarrSegmentationDataset3D(Dataset):
    def __init__(self, mgr):
        self.mgr = mgr
        self.model_name = mgr.model_name
        self.volume_paths = mgr.volume_paths  # array of dicts
        self.targets = mgr.targets
        self.patch_size = mgr.train_patch_size
        self.min_labeled_ratio = mgr.min_labeled_ratio
        self.min_bbox_percent = mgr.min_bbox_percent
        self.dilate_label = mgr.dilate_label
        self.use_cache = mgr.use_cache
        self.cache_folder = mgr.cache_folder

        # We will store only paths (not open Zarr objects) here:
        self.volumes = []
        for vol_idx, vol_info in enumerate(self.volume_paths):
            ref_label_key = vol_info.get("ref_label", "sheet")
            # We'll keep them as strings so we can open them later in __getitem__
            # this is so we can easily use fsspec for http zarrs in a fork-safe manner
            # since a zarr once open in fsspec is not fork-safe and will error
            # minor overhead here, but it is what it is
            vol_dict = {
                "input_path": vol_info["input"],
                "targets_path": {},
                "ref_label_key": ref_label_key
            }
            # For each task, remember its path
            for task_name in self.targets.keys():
                if task_name in vol_info:
                    vol_dict["targets_path"][task_name] = vol_info[task_name]
                else:
                    raise ValueError(f"Volume {vol_idx} missing path for '{task_name}'")

            self.volumes.append(vol_dict)

        # Build or load the patch cache
        self.cache_file = Path(f"{self.cache_folder}/"
                               f"{self.model_name}_"
                               f"{self.patch_size[0]}_{self.patch_size[1]}_{self.patch_size[2]}_cache.json")

        self.all_valid_patches = []
        if self.use_cache and self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                self.all_valid_patches = json.load(f)
            print(f"Loaded {len(self.all_valid_patches)} patches from cache.")
        else:
            print("Computing valid patches from scratch...")
            for vol_idx, vol_dict in enumerate(self.volumes):
                ref_label_key = vol_dict["ref_label_key"]
                # Only open the reference label Zarr for bounding-box scanning:
                ref_label_path = vol_dict["targets_path"][ref_label_key]

                # If it's HTTP, use fsspec.filesystem("http") -> get_mapper()
                if ref_label_path.startswith("http"):
                    http_fs = fsspec.filesystem("http")
                    store = http_fs.get_mapper(ref_label_path)
                    ref_label_zarr = zarr.open(store, mode='r')
                else:
                    ref_label_zarr = zarr.open(ref_label_path, mode='r')

                # Find the valid patches
                vol_patches = find_valid_patches(
                    ref_label_zarr,
                    patch_size=self.patch_size,
                    bbox_threshold=self.min_bbox_percent,
                    label_threshold=self.min_labeled_ratio
                )
                # Done reading, close right away
                ref_label_zarr.store.close()

                # Tag each patch with volume index
                for p in vol_patches:
                    p["volume_idx"] = vol_idx
                self.all_valid_patches.extend(vol_patches)

            if self.use_cache:
                cache_parent = os.path.dirname(str(self.cache_file))
                os.makedirs(cache_parent, exist_ok=True)
                with open(self.cache_file, 'w') as f:
                    json.dump(self.all_valid_patches, f)
                print(f"Saved {len(self.all_valid_patches)} patches to cache.")

    def __len__(self):
        return len(self.all_valid_patches)

    def __getitem__(self, idx):
        patch_info = self.all_valid_patches[idx]
        vol_idx = patch_info["volume_idx"]

        z0, y0, x0 = patch_info["start_pos"]
        dz, dy, dx = self.patch_size
        patch_slice = np.s_[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]

        vol_dict = self.volumes[vol_idx]
        input_path = vol_dict["input_path"]

        # === OPEN THE INPUT ZARR HERE (per worker) ===
        if input_path.startswith("http"):
            http_fs = fsspec.filesystem("http")
            in_store = http_fs.get_mapper(input_path)
            input_zarr = zarr.open(in_store, mode='r')
            input_zarr = input_zarr[0]
        else:
            input_zarr = zarr.open(input_path, mode='r')

        # get the patch
        input_data = input_zarr[patch_slice]
        og_input_dtype = input_data.dtype
        input_data = input_data.astype(np.float32)

        if og_input_dtype == np.uint8:
            input_data /= 255.0
        elif og_input_dtype == np.uint16:
            input_data /= 65535.0

        data_dict = {"image": input_data}

        # === OPEN EACH TARGET TASK ZARR HERE (per worker) ===
        for task_name, task_path in vol_dict["targets_path"].items():
            # open that store
            if task_path.startswith("http"):
                http_fs = fsspec.filesystem("http")
                t_store = http_fs.get_mapper(task_path)
                t_arr = zarr.open(t_store, mode='r')
            else:
                t_arr = zarr.open(task_path, mode='r')

            t_patch = t_arr[patch_slice].astype(np.float32)

            # If normals, special scaling
            if task_name.lower() == "normals":
                if t_arr.dtype == np.uint16:
                    t_patch = (t_patch / 32767.5) - 1.0
                else:
                    t_patch = (t_patch * 2.0) - 1.0
                # If shape is (Z, Y, X, C), transpose
                if t_patch.ndim == 4:
                    t_patch = t_patch.transpose(3, 0, 1, 2).copy()
            else:
                # scale to [0,1]
                if t_arr.dtype == np.uint8:
                    t_patch /= 255.0
                elif t_arr.dtype == np.uint16:
                    t_patch /= 65535.0

                if self.dilate_label:
                    t_patch = (t_patch > 0).astype(np.float32)
                    t_patch = dilation(t_patch, ball(5))

            data_dict[task_name] = t_patch

            # Close target store
            t_arr.store.close()

        # we can now close the input store too
        input_zarr.store.close()

        # --- Augmentations (2D + 3D) ---
        img_transform = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(),
                A.Illumination(),
            ], p=0.3),
            A.OneOf([
                A.MultiplicativeNoise(),
                A.GaussNoise()
            ], p=0.35),
            A.OneOf([
                A.MotionBlur(),
                A.Defocus(),
                A.Downscale(),
                A.AdvancedBlur()
            ], p=0.4),
        ], p=1.0)

        vol_transform = A.Compose([
            A.CoarseDropout3D(
                fill=0.5,
                num_holes_range=(1, 4),
                hole_depth_range=(0.1, 0.4),
                hole_height_range=(0.1, 0.4),
                hole_width_range=(0.1, 0.4)
            )
        ], p=0.5)

        # Apply 2D augs (slice-wise) to the image
        img_augmented = img_transform(image=data_dict["image"])
        image_2d_aug = img_augmented["image"]

        # Apply volumetric augs
        vol_augmented = vol_transform(volume=image_2d_aug)
        data_dict["image"] = vol_augmented["volume"]

        # Convert to torch tensors, ensure shape is [C, Z, Y, X]
        if data_dict["image"].ndim == 3:
            data_dict["image"] = data_dict["image"][None, ...]
        data_dict["image"] = torch.from_numpy(np.ascontiguousarray(data_dict["image"]))

        for task_name in self.targets.keys():
            tgt = data_dict[task_name]
            if tgt.ndim == 3 and task_name.lower() != "normals":
                tgt = tgt[None, ...]
            data_dict[task_name] = torch.from_numpy(np.ascontiguousarray(tgt))

        return data_dict

    def close(self):
        """No-op here since we open/close inside __getitem__.
           (Kept if you need to do something special on dataset shutdown.)"""
        pass