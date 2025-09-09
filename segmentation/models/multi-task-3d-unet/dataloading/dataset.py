from pathlib import Path
import os
import json
import numpy as np
import torch
import fsspec
import zarr
from torch.utils.data import Dataset
from skimage.morphology import dilation, ball, skeletonize


from helpers import find_valid_patches, find_valid_patches_xyz, pad_or_crop_3d, pad_or_crop_4d
from dataloading.wk_helper import read_webk_dataset, slice_and_reorder_volume

class MultiTask3dDataset(Dataset):
    def __init__(self,
                 mgr,
                 image_transforms=None,
                 volume_transforms=None):
        super().__init__()
        self.mgr = mgr

        self.model_name = mgr.model_name
        self.targets = mgr.targets               # e.g. {"ink": {...}, "normals": {...}}
        self.patch_size = mgr.train_patch_size
        self.min_labeled_ratio = mgr.min_labeled_ratio
        self.min_bbox_percent = mgr.min_bbox_percent
        self.dilate_label = mgr.dilate_label

        self.use_cache = mgr.use_cache
        self.cache_folder = mgr.cache_folder
        self.cache_file = Path(f"{self.cache_folder}/"
                               f"{self.model_name}_"
                               f"{self.patch_size[0]}_{self.patch_size[1]}_{self.patch_size[2]}_cache.json")

        self.image_transforms = image_transforms
        self.volume_transforms = volume_transforms

        # This dict looks like:
        # {
        #   "ink": [
        #       {
        #         'data': {'label': <zarr array>, 'data': <zarr array>},
        #         'out_channels': 1,
        #         ...
        #       },
        #       { ... },
        #   ],
        #   "normals": [ ... ],
        # }
        self.target_volumes = {}
        self.valid_patches = []
        self._initialize_volumes()
        self._get_valid_patches()


    def _initialize_volumes(self):
        for target_name, target_config in self.targets.items():
            self.target_volumes[target_name] = []

            for volume in target_config['volumes']:
                volume_type = volume.get('format')
                if not volume_type:
                    raise ValueError(f"Volume type not specified for {target_name}")

                # Example: "zarr_local"
                if volume_type == 'zarr_local':
                    label_zarr_path = volume["label_volume"]
                    data_zarr_path = volume["data_volume"]
                    volume_data = {
                        "label": zarr.open(label_zarr_path, mode='r'),
                        "data": zarr.open(data_zarr_path, mode='r')
                    }

                elif volume_type == 'wk_api':
                    # e.g. read_webk_dataset
                    annotation_id = volume.get('annotation_id', '678d7e580100001956ee6494')
                    volume_data = read_webk_dataset(
                        self.mgr.wk_url,
                        self.mgr.wk_token,
                        annotation_id=annotation_id,
                        annotation_name=volume.get('label_id'),
                        image_name=volume.get('data_id'),
                    )

                else:
                    raise ValueError(f"Unsupported volume type {volume_type}")

                volume_info = {
                    'data': volume_data,
                    'out_channels': target_config.get('out_channels', 1),
                    'shape': volume.get('shape', 'zyx')
                }
                self.target_volumes[target_name].append(volume_info)

    def _get_valid_patches(self):
        if self.use_cache and self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                self.valid_patches = json.load(f)
            print(f"Loaded {len(self.valid_patches)} patches from cache.")
            return

        print("Computing valid patches from scratch...")
        ref_target = list(self.target_volumes.keys())[0]

        for vol_idx, volume_info in enumerate(self.target_volumes[ref_target]):
            label_data = volume_info['data']['label']   # The "label" array
            shape_format = volume_info.get('shape', 'zyx')

            if shape_format.lower() == 'zyx':
                patches = find_valid_patches(
                    label_data,
                    patch_size=self.patch_size,
                    bbox_threshold=self.min_bbox_percent,
                    label_threshold=self.min_labeled_ratio,
                )
            elif shape_format.lower() == 'xyz':
                pz, py, px = self.patch_size
                patch_size_xyz = (px, py, pz)
                patches = find_valid_patches_xyz(
                    label_data,
                    patch_size_xyz=patch_size_xyz,
                    bbox_threshold=self.min_bbox_percent,
                    label_threshold=self.min_labeled_ratio,
                )
            else:
                raise ValueError(f"Unknown shape format {shape_format}.")

            for p in patches:
                self.valid_patches.append({
                    "volume_index": vol_idx,
                    "position": p["start_pos"]  # (z,y,x)
                })

        # Optionally cache it
        if self.use_cache:
            os.makedirs(self.cache_folder, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.valid_patches, f)
            print(f"Saved {len(self.valid_patches)} patches to cache.")

    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, index):
        patch_info = self.valid_patches[index]
        z, y, x = patch_info["position"]
        dz, dy, dx = self.patch_size
        vol_idx = patch_info["volume_index"]

        data_dict = {}

        for t_name, volumes_list in self.target_volumes.items():
            volume_info = volumes_list[vol_idx]
            vdata = volume_info['data']
            shape_format = volume_info['shape']
            out_c = volume_info['out_channels']

            label_arr = vdata['label']
            img_arr = vdata['data']

            # 1) Slice label patch
            label_patch = slice_and_reorder_volume(
                label_arr, shape_format, [z, y, x], [dz, dy, dx]
            ).astype(np.float32)

            # 2) If "normals", do [-1..1] scaling & transpose if 4D
            if t_name.lower() == "normals":
                # scale from [0..1] => [-1..1], or from uint16 => [-1..1]
                if label_arr.dtype == np.uint16:
                    label_patch = (label_patch / 32767.5) - 1.0
                else:
                    label_patch = (label_patch * 2.0) - 1.0

                # if shape is (Z, Y, X, C) => (C, Z, Y, X)
                if label_patch.ndim == 4:
                    label_patch = label_patch.transpose(3, 0, 1, 2).copy()

            elif t_name.lower() == "uv":
                    if label_patch.ndim == 4:
                        label_patch = label_patch.transpose(3, 0, 1, 2)

            else:
                # Typical clamp to [0,1]
                label_patch = np.clip(label_patch, 0, 1)



            # 3) Pad/crop so final shape is (out_c, D, H, W)
            #    (If label_patch is 3D => (D,H,W), if 4D => (C,D,H,W))
            if label_patch.ndim == 3:
                # => (dz, dy, dx)
                label_patch = pad_or_crop_3d(label_patch, (dz, dy, dx))
                # if out_c=1, maybe add channel
                if out_c == 1:
                    label_patch = label_patch[None, ...]  # => (1, D, H, W)
            else:
                # => (C, D, H, W)
                label_patch = pad_or_crop_4d(label_patch, (out_c, dz, dy, dx))

            label_patch = np.ascontiguousarray(label_patch).copy()



            # 4) Slice the image patch once
            if "image" not in data_dict:
                img_patch = slice_and_reorder_volume(
                    img_arr, shape_format, [z, y, x], [dz, dy, dx]
                ).astype(np.float32)

                # pad/crop image
                img_patch = pad_or_crop_3d(img_patch, (dz, dy, dx))  # if single‐channel
                img_patch = np.ascontiguousarray(img_patch).copy()

                patch_mean = img_patch.mean()
                patch_std = img_patch.std()
                if patch_std < 1e-8:
                    patch_std = 1e-8
                img_patch = (img_patch - patch_mean) / patch_std

                # if self.image_transforms:
                #     aug = self.image_transforms(image=img_patch)
                #     img_patch = aug["image"]
                #
                if self.volume_transforms:
                    vol_augmented = self.volume_transforms(volume=img_patch)
                    img_patch = vol_augmented["volume"]

                # ensure shape => (1, D, H, W) if single‐channel

                if img_patch.ndim == 3:
                    img_patch = img_patch[None, ...]

                data_dict["image"] = torch.from_numpy(img_patch)

            data_dict[t_name] = torch.from_numpy(label_patch)

            loss_name = self.targets[t_name]["loss_fn"].lower()  # e.g. "skeleton" or "skel"
            if "skel" in loss_name:
                self.do_tube = True
                # label_patch shape = (1, D, H, W)

                # Binarize
                bin_seg = (label_patch[0] > 0).astype(np.uint8)  # shape => (D, H, W)

                seg_skel = np.zeros_like(bin_seg, dtype=np.int16)  # shape => (D, H, W)

                if np.sum(bin_seg) != 0:
                    for z_id in range(bin_seg.shape[0]):
                        if bin_seg[z_id].sum() > 0:
                            skel_slice = skeletonize(bin_seg[z_id])
                            skel_slice = (skel_slice > 0).astype(np.int16)

                            if self.do_tube:
                                # "double" dilation
                                skel_slice = dilation(skel_slice)
                                skel_slice = dilation(skel_slice)

                            skel_slice *= bin_seg[z_id].astype(np.int16)
                            seg_skel[z_id] = skel_slice

                # Expand dims back to (1, D, H, W), then convert to float32
                seg_skel = seg_skel[None, ...].astype(np.float32)
                data_dict[f"{t_name}_skel"] = torch.from_numpy(seg_skel)

        return data_dict


