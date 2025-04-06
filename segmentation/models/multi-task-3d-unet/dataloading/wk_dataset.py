from torch.utils.data import Dataset
from helpers import find_valid_patches, find_valid_patches_xyz
from dataloading.wk_helper import read_webk_dataset, slice_and_reorder_volume
import webknossos as wk
import numpy as np
import torch
import fsspec
import zarr
import albumentations as A

class MultiTask3dDataset(Dataset):
    def __init__(self, mgr):

        super().__init__()
        self.mgr = mgr
        self.patch_size = mgr.train_patch_size # z , y , x
        self.wk_url = mgr.wk_url
        self.wk_token = mgr.wk_token
        self.targets = mgr.targets
        self.target_volumes = {}

        self._initialize_volumes()
        self._get_valid_patches()

    def _initialize_volumes(self):
        for target_name, target_config in self.targets.items():
            self.target_volumes[target_name] = []

            for volume in target_config['volumes']:
                volume_type = volume['format']
                label_volume_id = volume['label_id']

                if volume_type == 'wk_api':
                    # wk annos grabbed from api always include both data and anno ,
                    # this is kinda fucky and complicates using it within my existing dataset, but for now it works
                    # read_webk_dataset also handles squeezing out the channel dimension (for augs later) and
                    # transposing from wk x,y,z to our z,y,x. we have to add the channel dim back in
                    # before we return our tensors , but this makes handling augs easier as they are not all compatible
                    # with images that include a channel dimension
                    volume_data = read_webk_dataset(
                        self.wk_url,
                        self.wk_token,
                        annotation_id= '678d7e580100001956ee6494',
                        annotation_name=volume['label_id'],
                        image_name=volume['data_id'],
                    )

                elif volume_type == 'zarr_stream':
                    http_fs = fsspec.filesystem("http")
                    label_zarr_path = volume["label_volume"]
                    data_zarr_path = volume["data_volume"]
                    label_in_store = http_fs.get_mapper(label_zarr_path)
                    data_in_store = http_fs.get_mapper(data_zarr_path)
                    volume_data = {
                        "label": zarr.open(label_in_store, mode='r'),
                        "data": zarr.open(data_in_store, mode='r')
                    }

                elif volume_type == 'zarr_local':
                    label_zarr_path = volume["label_volume"]
                    data_zarr_path = volume["data_volume"]
                    volume_data = {
                        "label": zarr.open(label_zarr_path, mode='r'),
                        "data": zarr.open(data_zarr_path, mode='r')
                    }

                else:
                    print(f"volume type not specified. please specify a volume type for {target_name}")

                volume_info = {
                    'data': volume_data,  # Contains both 'image' and 'label' , this is because get anno from wk always gets both
                    'in_channels': target_config['in_channels'],
                    'out_channels': target_config['out_channels'],
                    'spacing': volume.get('spacing', [1, 1, 1]),
                    'label_id': volume['label_id'],
                    'data_id': volume['data_id'],
                    'shape': volume.get('shape', 'zyx')
                }

                self.target_volumes[target_name].append(volume_info)

    def _get_valid_patches(self):
        self.valid_patches = []

        for target_name, volumes in self.target_volumes.items():
            for vol_idx, volume_info in enumerate(volumes):
                label_data = volume_info['data']['label']

                shape_format = volume_info.get('shape', 'zyx')  # or shape_format?

                if shape_format.lower() == 'zyx':
                    patches = find_valid_patches(
                        label_data,
                        patch_size=self.patch_size,
                        bbox_threshold=self.mgr.min_labeled_ratio,
                        label_threshold=self.mgr.min_bbox_percent
                    )

                elif shape_format.lower() == 'xyz':
                    # note that patch_size here is (z, y, x);
                    # if your patch_size is also in zyx, you need to reorder to (x, y, z)!
                    pz, py, px = self.patch_size
                    patch_size_xyz = (px, py, pz)

                    patches = find_valid_patches_xyz(
                        label_data,
                        patch_size_xyz=patch_size_xyz,
                        bbox_threshold=self.mgr.min_labeled_ratio,
                        label_threshold=self.mgr.min_bbox_percent
                    )
                else:
                    raise ValueError(f"Unknown shape format {shape_format}.")

                for patch_dict in patches:
                    self.valid_patches.append({
                        "position": patch_dict["start_pos"],  # always z,y,x
                        "target_name": target_name,
                        "volume_index": vol_idx
                    })

    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, index):
        patch_info = self.valid_patches[index]
        z, y, x = patch_info["position"]
        dz, dy, dx = self.patch_size

        target_name = patch_info['target_name']
        vol_idx = patch_info['volume_index']
        volume_info = self.target_volumes[target_name][vol_idx]

        shape_format = volume_info.get("shape", "zyx")

        volume_data = volume_info['data']
        label_array = volume_data['label']
        image_array = volume_data['data']

        # slice and reorder label if needed . i have to have this because wk stores in cxyz, and i hate cxyz
        label_data = slice_and_reorder_volume(
            array=label_array,
            shape_format=shape_format,
            start_pos=[z, y, x],
            patch_size=[dz, dy, dx]
        )

        # slice and reorder image if needed . i have to have this because wk stores in cxyz, and i hate cxyz
        image_data = slice_and_reorder_volume(
            array=image_array,
            shape_format=shape_format,
            start_pos=[z, y, x],
            patch_size=[dz, dy, dx]
        )

        label_data = label_data.astype(np.float32)
        image_data = image_data.astype(np.float32)

        # Normalize [0..1] for image
        image_min = image_data.min()
        image_max = image_data.max()
        if image_max > image_min:
            epsilon = 1e-8
            image_data = (image_data - image_min) / (image_max - image_min + epsilon)

        # clamp label to [0, 1]
        if target_name != 'normals':
            label_data = np.clip(label_data, 0.0, 1.0)

        # re-add channel dim we removed earlier
        label_data = label_data[None, ...]
        image_data = image_data[None, ...]

        # convert to Tensors
        label_tensor = torch.from_numpy(np.ascontiguousarray(label_data))
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_data))

        data_dict = {
            'image': image_tensor,
            target_name: label_tensor
        }
        return data_dict
