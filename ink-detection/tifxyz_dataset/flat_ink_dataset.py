import os
from pathlib import Path
import json
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset
import zarr
import numpy as np 
import tifffile
from common import flat_patch_cache_path, load_flat_patch_cache, open_zarr, save_flat_patch_cache
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.image_proc.intensity.normalization import normalize_robust

@dataclass
class Patch:
    segment: 'Segment'
    bbox: tuple  # (z0, y0, x0, z1, y1, x1)

    @property
    def image_volume(self):
        return self.segment.image_volume

    @property
    def supervision_mask(self):
        return self.segment.supervision_mask
    
    @property
    def inklabels(self):
        return self.segment.inklabels
    
class Segment:
    def __init__(
            self,
            config,
            image_volume=None,
            supervision_mask=None,
            inklabels=None,
            scale=None,
            ):
        
        self.config = config
        self.scale = scale
        self.image_volume = image_volume
        self.supervision_mask = supervision_mask
        self.inklabels = inklabels
        self.patch_size = config['patch_size']

    def _find_patches(self):
        supervision_mask = open_zarr(self.supervision_mask, resolution=self.scale, auth=self.config['volume_auth_json'])
        inklabels = open_zarr(self.inklabels, resolution=self.scale, auth=self.config['volume_auth_json'])
        surface = supervision_mask.shape[0] // 2
        surface_slice = supervision_mask[surface]
        ys, xs = np.nonzero(surface_slice)
        if len(ys) == 0:
            raise ValueError(f"{self.supervision_mask} contains no nonzero voxels")

        stride = int(self.patch_size[1] * self.config['patch_overlap'])
        patch_corners_top_left = np.unique(
            np.stack([ys // stride * stride, xs // stride * stride], axis=1),
            axis=0
        )

        labeled_patches = []
        for y0, x0 in patch_corners_top_left:
            patch_bbox = inklabels[surface, y0:y0+self.patch_size[1], x0:x0+self.patch_size[2]]
            if patch_bbox.size == 0:
                continue
            labeled_ys, labeled_xs = np.nonzero(patch_bbox)
            if labeled_ys.size == 0:
                continue
            labeled_area = (labeled_ys.max() - labeled_ys.min() + 1) * (labeled_xs.max() - labeled_xs.min() + 1)
            labeled_patch_coverage = labeled_area / patch_bbox.size

            if labeled_patch_coverage >= self.config['patch_min_labeled_coverage']:
                z0 = surface - self.patch_size[0] // 2
                labeled_patches.append(Patch(
                    segment=self,
                    bbox=(z0, int(y0), int(x0),
                          z0 + self.patch_size[0], int(y0) + self.patch_size[1], int(x0) + self.patch_size[2]),
                ))

        if len(labeled_patches) == 0:
            raise ValueError(f"{self.inklabels} produced no valid patches")

        self.patches = labeled_patches
    

class FlatInkDataset(Dataset):
    def __init__(self, config, do_augmentations=True, debug=False, patches=None):
        
        self.debug            = debug
        self.config           = config
        self.patch_size       = config['patch_size']
        self.datasets         = config['datasets']
        self.vol_auth         = config['volume_auth_json']
        self.num_workers      = config.get('dataloader_workers', 8)
        self.do_augmentations = do_augmentations

        self.augmentations = create_training_transforms(self.patch_size) if self.do_augmentations else None

        if patches is None:
            segments = list(self._gather_segments())
            cache_path = flat_patch_cache_path(self.config)
            segments_by_key = {
                (str(seg.image_volume), str(seg.supervision_mask), str(seg.inklabels), seg.scale): seg
                for seg in segments
            }

            if cache_path.exists():
                self.patches = [
                    Patch(
                        segment=segments_by_key[
                            (
                                record['image_volume'],
                                record['supervision_mask'],
                                record['inklabels'],
                                record['scale'],
                            )
                        ],
                        bbox=tuple(record['bbox']),
                    )
                    for record in load_flat_patch_cache(cache_path)
                ]
                return

            def _process_segment(seg):
                seg._find_patches()
                return seg.patches

            self.patches = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                for patches in tqdm(pool.map(_process_segment, segments), total=len(segments), desc='Finding patches'):
                    self.patches.extend(patches)
            save_flat_patch_cache(cache_path, self.patches)
        else:
            self.patches = patches

    def _gather_segments(self):
        for ds in self.datasets:

            seg_path = Path(ds['segments_path'])

            for tifxyz_folder in sorted(seg_path.iterdir()):
                if tifxyz_folder.is_dir() and any(tifxyz_folder.rglob('x.tif')) and tifxyz_folder.name != 'unused':
                    image_volume     = Path(str(tifxyz_folder) + "/" + tifxyz_folder.name + '.zarr')
                    supervision_mask = Path(str(tifxyz_folder) + "/" + tifxyz_folder.name + '_supervision_mask.zarr')
                    inklabels        = Path(str(tifxyz_folder) + "/" + tifxyz_folder.name + '_inklabels.zarr')

                    if self.debug:
                        print(image_volume)
                        print(supervision_mask)
                        print(inklabels)

                    if not (image_volume.exists() and supervision_mask.exists() and inklabels.exists()):
                        raise ValueError(f"{tifxyz_folder.name} is missing required data. make sure the image volume, supervision mask, and labels exist")

                    yield Segment(
                        config           = self.config,
                        image_volume     = image_volume,
                        supervision_mask = supervision_mask,
                        inklabels        = inklabels,
                        scale            = ds['volume_scale']
                    )

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        z0, y0, x0, z1, y1, x1 = patch.bbox

        image_vol = open_zarr(patch.image_volume, resolution=patch.segment.scale, auth=self.vol_auth)
        supervision_mask = open_zarr(patch.supervision_mask, resolution=patch.segment.scale, auth=self.vol_auth)
        inklabels = open_zarr(patch.inklabels, resolution=patch.segment.scale, auth=self.vol_auth)

        image_crop = image_vol[z0:z1, y0:y1, x0:x1]
        supervision_crop = supervision_mask[z0:z1, y0:y1, x0:x1]
        inklabels_crop = inklabels[z0:z1, y0:y1, x0:x1]

        image_crop = normalize_robust(image_crop)

        surface_mask = None
        if self.config.get('show_surface_mask', False):
            middle = self.patch_size[0] // 2
            surface_mask = np.zeros_like(image_crop)
            surface_mask[middle] = True

        image_crop = torch.from_numpy(image_crop).float().unsqueeze(0)              
        inklabels_crop = torch.from_numpy(inklabels_crop).float().unsqueeze(0)      
        supervision_crop = torch.from_numpy(supervision_crop).float().unsqueeze(0)  


        data = {
            'image' : image_crop,
            'inklabels' : inklabels_crop,
            'supervision_mask' : supervision_crop
        }

        if self.config.get('show_surface_mask', False) and surface_mask is not None:
            data['surface_mask'] = torch.from_numpy(surface_mask).float().unsqueeze(0) 

        if self.do_augmentations and self.augmentations is not None:  
            augmented = self.augmentations(**data)
            return augmented
        
        return data

if __name__ == "__main__":
    import napari

    config_path = '/home/sean/villa/ink-detection/tifxyz_dataset/example_config_flat.json'
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ds = FlatInkDataset(config)

    data = ds[0]

    for k, v in data.items():
        print(f'{k:20s} shape={str(list(v.shape)):20s} dtype={str(v.dtype):15s} min={v.min().item():.4f}  max={v.max().item():.4f}')

    viewer = napari.Viewer()
    viewer.add_image(data['image'].squeeze(0).numpy(), name='image')
    viewer.add_labels(data['inklabels'].squeeze(0).numpy().astype(int), name='inklabels')
    viewer.add_labels(data['supervision_mask'].squeeze(0).numpy().astype(int), name='supervision_mask')
    if 'surface_mask' in data:
        viewer.add_labels(data['surface_mask'].squeeze(0).numpy().astype(int), name='surface_mask')

    napari.run()
