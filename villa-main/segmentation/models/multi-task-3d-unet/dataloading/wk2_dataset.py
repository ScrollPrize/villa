
from torch.utils.data import Dataset
from dataloading.transforms.preprocess import find_valid_patches
from dataloading.wk_helper import read_webk_dataset
import webknossos as wk
import numpy as np
import torch
import albumentations as A

class wkDataset(Dataset):
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
                label_volume_id = volume['label_volume']

                # wk annos grabbed from api always include both data and anno ,
                # this is kinda fucky and complicates using it within my existing dataset, but for now it works
                # read_webk_dataset also handles squeezing out the channel dimension (for augs later) and
                # transposing from wk x,y,z to our z,y,x. we have to add the channel dim back in
                # before we return our tensors , but this makes handling augs easier as they are not all compatible
                # with images that include a channel dimension
                volume_data = read_webk_dataset(
                    self.wk_url,
                    self.wk_token,
                    annotation_id=label_volume_id,
                    annotation_name=volume['label_id'],
                    image_name=volume['data_id'],
                )

                volume_info = {
                    'data': volume_data,  # Contains both 'image' and 'label' , this is because get anno from wk always gets both
                    'in_channels': target_config['in_channels'],
                    'out_channels': target_config['out_channels'],
                    'spacing': volume.get('spacing', [1, 1, 1]),
                    'label_id': volume['label_id'],
                    'data_id': volume['data_id']
                }

                self.target_volumes[target_name].append(volume_info)

    def _get_valid_patches(self):
        self.valid_patches = []

        for target_name, volumes in self.target_volumes.items():
            for vol_idx, volume_info in enumerate(volumes):

                label_data = volume_info['data']['label'] # this is stupid shit as a result of the above dict comment

                patches = find_valid_patches(
                    label_data,
                    patch_size=self.patch_size,
                    bbox_threshold=self.mgr.min_labeled_ratio,
                    label_threshold=self.mgr.min_bbox_percent
                )

                for patch_dict in patches:
                    patch_info = {
                        "position": patch_dict["start_pos"],  # [z, y, x]
                        "target_name": target_name,
                        "volume_index": vol_idx
                    }

                    self.valid_patches.append(patch_info)

    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, index):
        patch_info = self.valid_patches[index]
        z, y, x = patch_info["position"]
        dz, dy, dx = self.patch_size

        target_name = patch_info['target_name']
        vol_idx = patch_info['volume_index']
        volume_info = self.target_volumes[target_name][vol_idx]

        volume_data = volume_info['data']
        label_data = volume_data['label'][z:z + dz, y:y + dy, x:x + dx]
        image_data = volume_data['data'][z:z + dz, y:y + dy, x:x + dx]

        # cast to float32
        label_data = label_data.astype(np.float32)
        image_data = image_data.astype(np.float32)

        # normalize image data to 0 to 1
        image_min = image_data.min()
        image_max = image_data.max()
        if image_max > image_min:
            epsilon = 1e-8
            image_data = (image_data - image_min) / (image_max - image_min + epsilon)

        # Clamp label to [0, 1]
        label_data = np.clip(label_data, 0.0, 1.0)

        # add the channel dim back that we removed earlier
        label_data = label_data[None, ...]
        image_data = image_data[None, ...]

        label_tensor = torch.from_numpy(np.ascontiguousarray(label_data))
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_data))

        target_name = patch_info["target_name"]
        data_dict = {
            'image': image_tensor,
            target_name: label_tensor
        }

        return data_dict