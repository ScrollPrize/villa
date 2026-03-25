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
from common import (
    _read_bbox_with_padding,
    flat_patch_cache_path,
    load_flat_patch_cache,
    open_zarr,
    resolve_local_label_paths,
    save_flat_patch_cache,
)
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.image_proc.intensity.normalization import normalize_robust

@dataclass
class Patch:
    segment: 'Segment'
    bbox: tuple  # (z0, y0, x0, z1, y1, x1)
    is_validation: bool = False
    supervision_mask_override: object = None

    @property
    def image_volume(self):
        return self.segment.image_volume

    @property
    def supervision_mask(self):
        if self.supervision_mask_override is not None:
            return self.supervision_mask_override
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
            validation_mask=None,
            inklabels=None,
            scale=None,
            dataset_idx=None,
            segment_relpath=None,
            ):
        
        self.config = config
        self.scale = scale
        self.image_volume = image_volume
        self.supervision_mask = supervision_mask
        self.validation_mask = validation_mask
        self.inklabels = inklabels
        self.dataset_idx = dataset_idx
        self.segment_relpath = segment_relpath
        self.patch_size = config['patch_size']

    @property
    def cache_key(self):
        return (
            int(self.dataset_idx),
            str(self.segment_relpath),
            self.scale,
            str(self.inklabels),
            str(self.supervision_mask),
            "" if self.validation_mask is None else str(self.validation_mask),
        )

    def _find_patches(self):
        volume_auth = self.config.get('volume_auth_json')
        supervision_mask = open_zarr(self.supervision_mask, resolution=self.scale, auth=volume_auth)
        inklabels = open_zarr(self.inklabels, resolution=self.scale, auth=volume_auth)
        validation_mask = None
        if self.validation_mask is not None:
            validation_mask = open_zarr(self.validation_mask, resolution=self.scale, auth=volume_auth)
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

        training_patches = []
        validation_patches = []
        for y0, x0 in patch_corners_top_left:
            z0 = surface - self.patch_size[0] // 2
            patch_bbox_zyx = (
                z0,
                int(y0),
                int(x0),
                z0 + self.patch_size[0],
                int(y0) + self.patch_size[1],
                int(x0) + self.patch_size[2],
            )
            has_validation_supervision = False
            if validation_mask is not None:
                validation_patch = validation_mask[surface, y0:y0+self.patch_size[1], x0:x0+self.patch_size[2]]
                has_validation_supervision = bool(validation_patch.size > 0 and np.any(validation_patch))
            if has_validation_supervision:
                validation_patches.append(Patch(
                    segment=self,
                    bbox=patch_bbox_zyx,
                    is_validation=True,
                    supervision_mask_override=self.validation_mask,
                ))
                continue

            patch_bbox = inklabels[surface, y0:y0+self.patch_size[1], x0:x0+self.patch_size[2]]
            if patch_bbox.size == 0:
                continue
            labeled_ys, labeled_xs = np.nonzero(patch_bbox)
            if labeled_ys.size == 0:
                continue
            labeled_area = (labeled_ys.max() - labeled_ys.min() + 1) * (labeled_xs.max() - labeled_xs.min() + 1)
            labeled_patch_coverage = labeled_area / patch_bbox.size

            if labeled_patch_coverage >= self.config['patch_min_labeled_coverage']:
                training_patches.append(Patch(
                    segment=self,
                    bbox=patch_bbox_zyx,
                ))

        if len(training_patches) == 0 and len(validation_patches) == 0:
            raise ValueError(f"{self.inklabels} produced no valid patches")

        self.training_patches = training_patches
        self.validation_patches = validation_patches
        self.patches = training_patches + validation_patches
    

class FlatInkDataset(Dataset):
    def __init__(self, config, do_augmentations=True, debug=False, patches=None):
        
        self.debug            = debug
        self.config           = config
        self.patch_size       = config['patch_size']
        self.datasets         = config['datasets']
        self.vol_auth         = config.get('volume_auth_json')
        self.num_workers      = config.get('dataloader_workers', 8)
        self.do_augmentations = do_augmentations
        self.training_patches = []
        self.validation_patches = []

        self.augmentations = create_training_transforms(self.patch_size) if self.do_augmentations else None

        if patches is None:
            segments = list(self._gather_segments())
            cache_path = flat_patch_cache_path(self.config)
            segments_by_key = {
                seg.cache_key: seg
                for seg in segments
            }

            if cache_path.exists():
                cached_records = load_flat_patch_cache(cache_path)
                cached_patches = []
                cache_valid = True
                for record in cached_records:
                    cache_key = (
                        int(record['dataset_idx']),
                        str(record['segment_relpath']),
                        record['scale'],
                        str(record.get('inklabels_path', '')),
                        str(record.get('supervision_mask_path', '')),
                        str(record.get('validation_mask_path', '')),
                    )
                    segment = segments_by_key.get(cache_key)
                    if segment is None:
                        cache_valid = False
                        break
                    patch = Patch(
                        segment=segment,
                        bbox=tuple(record['bbox']),
                        is_validation=bool(record.get('is_validation', False)),
                        supervision_mask_override=record.get('active_supervision_mask_path') or None,
                    )
                    cached_patches.append(patch)
                    if patch.is_validation:
                        self.validation_patches.append(patch)
                    else:
                        self.training_patches.append(patch)
                if cache_valid:
                    self.patches = cached_patches
                    return

            def _process_segment(seg):
                seg._find_patches()
                return seg.training_patches, seg.validation_patches

            self.patches = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                for training_patches, validation_patches in tqdm(pool.map(_process_segment, segments), total=len(segments), desc='Finding patches'):
                    self.training_patches.extend(training_patches)
                    self.validation_patches.extend(validation_patches)
            self.patches = self.training_patches + self.validation_patches
            save_flat_patch_cache(cache_path, self.patches)
        else:
            self.patches = list(patches)
            self.training_patches = [patch for patch in self.patches if not getattr(patch, 'is_validation', False)]
            self.validation_patches = [patch for patch in self.patches if getattr(patch, 'is_validation', False)]

    def _gather_segments(self):
        for dataset_idx, ds in enumerate(self.datasets):

            seg_path = Path(ds['segments_path'])

            for tifxyz_folder in sorted(seg_path.iterdir()):
                if tifxyz_folder.is_dir() and any(tifxyz_folder.rglob('x.tif')) and tifxyz_folder.name != 'unused':
                    image_volume     = Path(str(tifxyz_folder) + "/" + tifxyz_folder.name + '.zarr')
                    inklabels, supervision_mask, validation_mask = resolve_local_label_paths(
                        tifxyz_folder,
                        tifxyz_folder.name,
                        label_version=self.config.get('label_version'),
                        extension='.zarr',
                    )

                    if self.debug:
                        print(image_volume)
                        print(supervision_mask)
                        print(inklabels)
                        print(validation_mask)

                    if not (image_volume.exists() and supervision_mask.exists() and inklabels.exists()):
                        raise ValueError(f"{tifxyz_folder.name} is missing required data. make sure the image volume, supervision mask, and labels exist")

                    yield Segment(
                        config           = self.config,
                        image_volume     = image_volume,
                        supervision_mask = supervision_mask,
                        validation_mask  = validation_mask,
                        inklabels        = inklabels,
                        scale            = ds['volume_scale'],
                        dataset_idx      = dataset_idx,
                        segment_relpath  = tifxyz_folder.relative_to(seg_path).as_posix(),
                    )

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        z0, y0, x0, z1, y1, x1 = patch.bbox
        expected_shape = tuple(int(v) for v in self.patch_size)

        image_vol = open_zarr(patch.image_volume, resolution=patch.segment.scale, auth=self.vol_auth)
        supervision_mask = open_zarr(patch.supervision_mask, resolution=patch.segment.scale, auth=self.vol_auth)
        inklabels = open_zarr(patch.inklabels, resolution=patch.segment.scale, auth=self.vol_auth)

        image_crop, image_valid_slices = _read_bbox_with_padding(
            image_vol,
            patch.bbox,
            fill_value=0,
        )
        supervision_crop, _ = _read_bbox_with_padding(
            supervision_mask,
            patch.bbox,
            fill_value=0,
        )
        inklabels_crop, _ = _read_bbox_with_padding(
            inklabels,
            patch.bbox,
            fill_value=0,
        )

        if image_valid_slices is not None:
            image_crop = image_crop.astype(np.float32, copy=False)
            image_crop[image_valid_slices] = normalize_robust(image_crop[image_valid_slices])
        else:
            image_crop = image_crop.astype(np.float32, copy=False)

        for name, array in (
            ("image", image_crop),
            ("supervision_mask", supervision_crop),
            ("inklabels", inklabels_crop),
        ):
            if tuple(int(v) for v in array.shape) != expected_shape:
                raise AssertionError(
                    f"{name} crop shape {tuple(int(v) for v in array.shape)} does not match "
                    f"requested patch size {expected_shape} for bbox {patch.bbox!r}"
                )

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
