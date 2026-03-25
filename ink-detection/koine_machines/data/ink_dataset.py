import json
from pathlib import Path
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset
import numpy as np 
from scipy.ndimage import distance_transform_edt
from koine_machines.augmentation.translation import maybe_translate_normal_pooled_crop_bbox
from koine_machines.common.common import (
    _read_bbox_with_padding,
    flat_patch_cache_path,
    flat_patch_finding_cache_token,
    load_flat_patch_cache,
    open_zarr,
    save_flat_patch_cache,
)
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.image_proc.intensity.normalization import normalize_robust
import vesuvius.tifxyz as tifxyz
from koine_machines.data.patch import Patch
from koine_machines.data.segment import Segment

def _read_flat_surface_patch(volume, *, y0, y1, x0, x1):
    surface = int(volume.shape[0] // 2)
    patch, _ = _read_bbox_with_padding(
        volume,
        (surface, int(y0), int(x0), surface + 1, int(y1), int(x1)),
        fill_value=0,
    )
    return patch[0]


def _select_flat_pixels_for_native_crop(patch_zyxs, valid_mask, crop_bbox):
    patch_zyxs = np.asarray(patch_zyxs)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    crop_start = np.asarray(crop_bbox[:3], dtype=np.int64)
    crop_stop = np.asarray(crop_bbox[3:], dtype=np.int64)

    finite_mask = np.isfinite(patch_zyxs).all(axis=-1)
    within = valid_mask & finite_mask
    within &= patch_zyxs[..., 0] >= crop_start[0]
    within &= patch_zyxs[..., 0] < crop_stop[0]
    within &= patch_zyxs[..., 1] >= crop_start[1]
    within &= patch_zyxs[..., 1] < crop_stop[1]
    within &= patch_zyxs[..., 2] >= crop_start[2]
    within &= patch_zyxs[..., 2] < crop_stop[2]
    if not np.any(within):
        raise ValueError(f"crop_bbox {crop_bbox!r} does not intersect any valid flat tifxyz pixels")

    rows, cols = np.where(within)
    support_y0 = int(rows.min())
    support_y1 = int(rows.max()) + 1
    support_x0 = int(cols.min())
    support_x1 = int(cols.max()) + 1
    return (
        (support_y0, support_y1, support_x0, support_x1),
        patch_zyxs[support_y0:support_y1, support_x0:support_x1],
        within[support_y0:support_y1, support_x0:support_x1],
    )


def _project_flat_patch_to_native_crop(flat_patch, patch_zyxs, valid_mask, crop_bbox):
    z0, y0, x0, z1, y1, x1 = (int(v) for v in crop_bbox)
    output = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.asarray(flat_patch).dtype)

    positive_mask = np.asarray(flat_patch) != 0
    valid_mask = np.asarray(valid_mask, dtype=bool) & positive_mask
    if not np.any(valid_mask):
        return output

    patch_zyxs = np.asarray(patch_zyxs)
    finite_mask = np.isfinite(patch_zyxs).all(axis=-1)
    valid_mask &= finite_mask
    if not np.any(valid_mask):
        return output

    mapped_zyxs = patch_zyxs[valid_mask].astype(np.int64, copy=False)
    local_zyxs = mapped_zyxs - np.asarray((z0, y0, x0), dtype=np.int32)
    within_crop = (
        (local_zyxs[:, 0] >= 0)
        & (local_zyxs[:, 0] < output.shape[0])
        & (local_zyxs[:, 1] >= 0)
        & (local_zyxs[:, 1] < output.shape[1])
        & (local_zyxs[:, 2] >= 0)
        & (local_zyxs[:, 2] < output.shape[2])
    )
    if not np.any(within_crop):
        return output

    local_zyxs = local_zyxs[within_crop]
    values = np.asarray(flat_patch)[valid_mask][within_crop]
    flat_indices = np.ravel_multi_index(local_zyxs.T, output.shape)
    np.maximum.at(output.reshape(-1), flat_indices, values)
    return output


def _project_valid_surface_mask_to_native_crop(patch_zyxs, valid_mask, crop_bbox):
    flat_mask = np.ones(np.asarray(valid_mask).shape, dtype=np.float32)
    surface_occupancy = _project_flat_patch_to_native_crop(flat_mask, patch_zyxs, valid_mask, crop_bbox)
    surface_occupancy = surface_occupancy > 0
    if not np.any(surface_occupancy):
        return surface_occupancy.astype(np.float32)

    max_distance_voxels = 10.0
    distance = distance_transform_edt(~surface_occupancy)
    surface_distance_field = np.clip(1.0 - (distance / max_distance_voxels), 0.0, 1.0)
    return surface_distance_field.astype(np.float32, copy=False)


def _build_normal_pooled_flat_metadata(
    *,
    patch_tifxyz,
    support_bbox,
    support_patch_zyxs,
    support_valid,
    support_inklabels_flat_patch,
    support_supervision_flat_patch,
    crop_bbox,
):
    support_y0, support_y1, support_x0, support_x1 = (int(v) for v in support_bbox)
    nx, ny, nz = patch_tifxyz.get_normals(support_y0, support_y1, support_x0, support_x1)
    normals_local_zyx = np.stack([nz, ny, nx], axis=-1).astype(np.float32, copy=False)

    flat_points_local_zyx = (
        np.asarray(support_patch_zyxs, dtype=np.float32)
        - np.asarray(crop_bbox[:3], dtype=np.float32)
    )

    flat_target = (np.asarray(support_inklabels_flat_patch) > 0).astype(np.float32, copy=False)
    flat_supervision = (np.asarray(support_supervision_flat_patch) > 0).astype(np.float32, copy=False)

    flat_valid = np.asarray(support_valid, dtype=bool)
    flat_valid &= np.isfinite(flat_points_local_zyx).all(axis=-1)
    flat_valid &= np.isfinite(normals_local_zyx).all(axis=-1)

    normal_magnitudes = np.linalg.norm(normals_local_zyx, axis=-1)
    flat_valid &= normal_magnitudes > 1e-6

    safe_normals = np.zeros_like(normals_local_zyx, dtype=np.float32)
    safe_points = np.zeros_like(flat_points_local_zyx, dtype=np.float32)
    if np.any(flat_valid):
        safe_normals[flat_valid] = (
            normals_local_zyx[flat_valid]
            / normal_magnitudes[flat_valid, None]
        ).astype(np.float32, copy=False)
        safe_points[flat_valid] = flat_points_local_zyx[flat_valid].astype(np.float32, copy=False)

    return {
        'flat_target': torch.from_numpy(flat_target).float().unsqueeze(0),
        'flat_supervision': torch.from_numpy(flat_supervision).float().unsqueeze(0),
        'flat_valid': torch.from_numpy(flat_valid.astype(np.float32, copy=False)).float().unsqueeze(0),
        'flat_points_local_zyx': torch.from_numpy(safe_points).float(),
        'flat_normals_local_zyx': torch.from_numpy(safe_normals).float(),
    }


def _pack_normal_pooled_augmentation_data(data):
    flat_valid_mask = data['flat_valid'][0] > 0
    keypoints = data['flat_points_local_zyx'][flat_valid_mask]
    flat_normals = data['flat_normals_local_zyx'][flat_valid_mask]

    augmentation_data = {
        'image': data['image'],
        'surface_mask': data['surface_mask'],
        'regression_keys': ['surface_mask'],
        'keypoints': keypoints,
        'flat_normals': flat_normals,
        'vector_keys': ['flat_normals'],
        'crop_shape': tuple(int(v) for v in data['image'].shape[1:]),
    }
    return augmentation_data, flat_valid_mask


def _restore_normal_pooled_augmentation_data(augmented, original_data, flat_valid_mask):
    restored = {
        'image': augmented['image'],
        'surface_mask': augmented['surface_mask'],
        'flat_target': original_data['flat_target'],
        'flat_supervision': original_data['flat_supervision'],
        'flat_valid': original_data['flat_valid'],
        'flat_points_local_zyx': torch.zeros_like(original_data['flat_points_local_zyx']),
        'flat_normals_local_zyx': torch.zeros_like(original_data['flat_normals_local_zyx']),
    }
    restored['flat_points_local_zyx'][flat_valid_mask] = augmented['keypoints']
    restored['flat_normals_local_zyx'][flat_valid_mask] = augmented['flat_normals']
    return restored


class InkDataset(Dataset):
    def __init__(self, config, do_augmentations=True, debug=False, patches=None, mode="flat"):
        
        self.debug            = debug
        self.config           = config
        self.patch_size       = config['patch_size']
        self.datasets         = config['datasets']
        self.vol_auth         = config.get('volume_auth_json')
        self.num_workers      = config.get('dataloader_workers', 8)
        self.mode             = config.get('mode', 'flat')
        self.do_augmentations = bool(do_augmentations)
        self.input_channels   = 1 + int(self.mode == "normal_pooled_3d")
        self.training_patches = []
        self.validation_patches = []
  

        if self.do_augmentations:
            self.augmentations = create_training_transforms(self.patch_size)
        else:
            self.augmentations = None

        if patches is None:
            segments = list(self._gather_segments())
            cache_path = flat_patch_cache_path(self.config)
            expected_patch_finding_key = flat_patch_finding_cache_token(self.config)
            segments_by_key = {
                seg.cache_key: seg
                for seg in segments
            }

            if cache_path.exists():
                cached_records = load_flat_patch_cache(cache_path)
                cached_patches = []
                cache_valid = True
                for record in cached_records:
                    if record.get('patch_finding_key') != expected_patch_finding_key:
                        cache_valid = False
                        break
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
                if not tifxyz_folder.is_dir() or tifxyz_folder.name == 'unused':
                    continue
                if not any(tifxyz_folder.rglob('x.tif')):
                    continue

                if self.mode == "normal_pooled_3d":
                    image_volume = Path(ds['volume_path'])
                else:
                    image_volume = Path(str(tifxyz_folder) + "/" + tifxyz_folder.name + '.zarr')

                segment = Segment(
                    config=self.config,
                    image_volume=image_volume,
                    scale=ds['volume_scale'],
                    dataset_idx=dataset_idx,
                    segment_relpath=tifxyz_folder.relative_to(seg_path).as_posix(),
                    segment_dir=tifxyz_folder,
                    segment_name=tifxyz_folder.name,
                )
                inklabels, supervision_mask, validation_mask = segment.discover_labels(extension='.zarr')

                if self.debug:
                    print(image_volume)
                    print(supervision_mask)
                    print(inklabels)
                    print(validation_mask)

                if not (image_volume.exists() and supervision_mask.exists() and inklabels.exists()):
                    raise ValueError(f"{tifxyz_folder.name} is missing required data. make sure the image volume, supervision mask, and labels exist")

                yield segment

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        z0, y0, x0, z1, y1, x1 = patch.bbox
        expected_shape = tuple(int(v) for v in self.patch_size)
        crop_bbox = patch.bbox
        use_surface_mask = self.mode == "normal_pooled_3d"
        surface_mask = None
        normal_pooled_metadata = None
    
        if self.mode == "normal_pooled_3d":
            image_vol = open_zarr(patch.image_volume, resolution=patch.segment.scale, auth=self.vol_auth)
            supervision_mask = open_zarr(patch.supervision_mask, resolution=patch.segment.scale, auth=self.vol_auth)
            inklabels = open_zarr(patch.inklabels, resolution=patch.segment.scale, auth=self.vol_auth)

            patch_tifxyz = tifxyz.read_tifxyz(patch.segment_dir)
            patch_tifxyz.use_full_resolution()
            flat_x, flat_y, flat_z, flat_valid = patch_tifxyz[y0:y1, x0:x1]
            patch_zyxs = np.stack([flat_z, flat_y, flat_x], axis=-1)
            valid_pts = patch_zyxs[flat_valid]
            if valid_pts.size == 0:
                raise ValueError(f"No valid tifxyz points found for bbox {patch.bbox!r}")

            mins = valid_pts.min(axis=0).astype(int)
            maxs = valid_pts.max(axis=0).astype(int)
            target_zyx_shape = np.asarray(expected_shape, dtype=int)
            actual_zyx_shape = (maxs - mins + 1).astype(int)
            shape_diff = target_zyx_shape - actual_zyx_shape

            # center before crop when the occupied extent is larger than the target.
            trim_before = np.maximum(-shape_diff, 0) // 2
            trim_after = np.maximum(-shape_diff, 0) - trim_before
            mins = mins + trim_before
            maxs = maxs - trim_after

            adjusted_shape = (maxs - mins + 1).astype(int)
            remaining_diff = target_zyx_shape - adjusted_shape

            pad_before = np.maximum(remaining_diff, 0) // 2
            pad_after = np.maximum(remaining_diff, 0) - pad_before
            mins = mins - pad_before
            maxs = maxs + pad_after

            crop_bbox = (int(mins[0]), int(mins[1]), int(mins[2]), int(maxs[0] + 1), int(maxs[1] + 1), int(maxs[2] + 1))
            supervision_flat_patch = _read_flat_surface_patch(supervision_mask, y0=y0, y1=y1, x0=x0, x1=x1)
            if self.do_augmentations:
                crop_bbox = maybe_translate_normal_pooled_crop_bbox(
                    crop_bbox,
                    patch_zyxs,
                    flat_valid,
                    supervision_flat_patch,
                )
            full_x, full_y, full_z, full_valid = patch_tifxyz[:, :]
            full_patch_zyxs = np.stack([full_z, full_y, full_x], axis=-1)
            (support_y0, support_y1, support_x0, support_x1), support_patch_zyxs, support_valid = _select_flat_pixels_for_native_crop(
                full_patch_zyxs,
                full_valid,
                crop_bbox,
            )
            support_supervision_flat_patch = _read_flat_surface_patch(
                supervision_mask,
                y0=support_y0,
                y1=support_y1,
                x0=support_x0,
                x1=support_x1,
            )
            support_inklabels_flat_patch = _read_flat_surface_patch(
                inklabels,
                y0=support_y0,
                y1=support_y1,
                x0=support_x0,
                x1=support_x1,
            )
            image_crop, image_valid_slices = _read_bbox_with_padding(image_vol, crop_bbox, fill_value=0)
            surface_mask = _project_valid_surface_mask_to_native_crop(
                support_patch_zyxs,
                support_valid,
                crop_bbox,
            )
            normal_pooled_metadata = _build_normal_pooled_flat_metadata(
                patch_tifxyz=patch_tifxyz,
                support_bbox=(support_y0, support_y1, support_x0, support_x1),
                support_patch_zyxs=support_patch_zyxs,
                support_valid=support_valid,
                support_inklabels_flat_patch=support_inklabels_flat_patch,
                support_supervision_flat_patch=support_supervision_flat_patch,
                crop_bbox=crop_bbox,
            )
            
        else:
            image_vol = open_zarr(patch.image_volume, resolution=patch.segment.scale, auth=self.vol_auth)
            supervision_mask = open_zarr(patch.supervision_mask, resolution=patch.segment.scale, auth=self.vol_auth)
            inklabels = open_zarr(patch.inklabels, resolution=patch.segment.scale, auth=self.vol_auth)
            image_crop, image_valid_slices = _read_bbox_with_padding(image_vol, patch.bbox, fill_value=0)
            supervision_crop, _ = _read_bbox_with_padding(supervision_mask,patch.bbox, fill_value=0)
            inklabels_crop, _ = _read_bbox_with_padding(inklabels, patch.bbox, fill_value=0)

        
        image_crop = image_crop.astype(np.float32, copy=False)
        if image_valid_slices is not None:
            image_crop[image_valid_slices] = normalize_robust(image_crop[image_valid_slices])

        arrays_to_validate = [("image", image_crop)]
        if self.mode != "normal_pooled_3d":
            arrays_to_validate.extend(
                [
                    ("supervision_mask", supervision_crop),
                    ("inklabels", inklabels_crop),
                ]
            )
        for name, array in arrays_to_validate:
            if tuple(int(v) for v in array.shape) != expected_shape:
                raise AssertionError(
                    f"{name} crop shape {tuple(int(v) for v in array.shape)} does not match "
                    f"requested patch size {expected_shape} for bbox {crop_bbox!r}"
                )


        image_crop = torch.from_numpy(image_crop).float().unsqueeze(0)
        data = {'image': image_crop}

        if self.mode == "normal_pooled_3d":
            assert normal_pooled_metadata is not None
            data.update(normal_pooled_metadata)
        else:
            inklabels_crop = torch.from_numpy(inklabels_crop).float().unsqueeze(0)
            supervision_crop = torch.from_numpy(supervision_crop).float().unsqueeze(0)
            data.update({
                'inklabels': inklabels_crop,
                'supervision_mask': supervision_crop,
            })

        if use_surface_mask and surface_mask is not None:
            data['surface_mask'] = torch.from_numpy(surface_mask).float().unsqueeze(0)

        if self.do_augmentations and self.augmentations is not None:  
            if self.mode == "normal_pooled_3d":
                augmentation_data, flat_valid_mask = _pack_normal_pooled_augmentation_data(data)
                augmented = self.augmentations(**augmentation_data)
                return _restore_normal_pooled_augmentation_data(augmented, data, flat_valid_mask)
            augmented = self.augmentations(**data)
            return augmented
        
        return data

if __name__ == "__main__":
    import argparse
    import napari
    from qtpy.QtWidgets import QPushButton

    parser = argparse.ArgumentParser(description="Visualize an InkDataset sample in napari.")
    parser.add_argument("config_path", help="Path to the dataset config JSON.")
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ds = InkDataset(config, do_augmentations=False)
    viewer = napari.Viewer()

    state = {"current_index": 0}
    layers = {
        "image": None,
        "target": None,
        "supervision": None,
        "surface_mask": None,
    }

    def load_sample(index):
        data = ds[index]
        print(f"\nSample {index}")
        for k, v in data.items():
            print(f'{k:20s} shape={str(list(v.shape)):20s} dtype={str(v.dtype):15s} min={v.min().item():.4f}  max={v.max().item():.4f}')
        return data

    def render_sample(index):
        data = load_sample(index)
        image = data['image'][0].numpy()
        target_key = 'inklabels' if 'inklabels' in data else 'flat_target'
        supervision_key = 'supervision_mask' if 'supervision_mask' in data else 'flat_supervision'
        target = data[target_key].squeeze(0).numpy().astype(int)
        supervision = data[supervision_key].squeeze(0).numpy().astype(int)

        if layers["image"] is None:
            layers["image"] = viewer.add_image(image, name='image')
            layers["target"] = viewer.add_labels(target, name=target_key)
            layers["supervision"] = viewer.add_labels(supervision, name=supervision_key)
        else:
            layers["image"].data = image
            layers["target"].data = target
            layers["target"].name = target_key
            layers["supervision"].data = supervision
            layers["supervision"].name = supervision_key

        surface_mask = data.get('surface_mask')
        if surface_mask is not None:
            surface_mask_data = surface_mask.squeeze(0).numpy()
            if layers["surface_mask"] is None:
                layers["surface_mask"] = viewer.add_image(
                    surface_mask_data,
                    name='surface_mask',
                    contrast_limits=(0.0, 1.0),
                    colormap='cyan',
                    opacity=0.5,
                )
            else:
                layers["surface_mask"].data = surface_mask_data
        elif layers["surface_mask"] is not None:
            viewer.layers.remove(layers["surface_mask"])
            layers["surface_mask"] = None

        viewer.title = f"InkDataset sample {index}"

    def show_next_sample():
        if len(ds) <= 1:
            render_sample(state["current_index"])
            return

        next_index = state["current_index"]
        while next_index == state["current_index"]:
            next_index = random.randrange(len(ds))
        state["current_index"] = next_index
        render_sample(state["current_index"])

    next_button = QPushButton("Next")
    next_button.clicked.connect(show_next_sample)
    viewer.window.add_dock_widget(next_button, area="right", name="Sample Controls")

    render_sample(state["current_index"])

    napari.run()
