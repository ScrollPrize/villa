import json
from pathlib import Path
import random
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset
import numpy as np 
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
from koine_machines.data.normal_pooled_sample import (
    _build_normal_pooled_flat_metadata,
    _filter_support_components_by_active_supervision,
    _project_flat_patch_to_native_crop,
    _pack_normal_pooled_augmentation_data,
    _project_flat_labels_and_supervision_to_native_crop,
    _project_valid_surface_mask_to_native_crop,
    _select_flat_pixels_for_native_crop,
    _restore_normal_pooled_augmentation_data,
    _select_flat_pixels_for_native_crop_via_stored_resolution,
    _slice_support_halo_for_subwindow,
)
from koine_machines.data.native_crop import compute_native_crop_bbox_from_patch_points
from koine_machines.data.segment import Segment


_NATIVE_3D_MODES = {"normal_pooled_3d", "full_3d"}


def _is_native_3d_mode(mode):
    return str(mode).strip().lower() in _NATIVE_3D_MODES


def _read_flat_surface_patch(volume, *, y0, y1, x0, x1):
    surface = int(volume.shape[0] // 2)
    patch, _ = _read_bbox_with_padding(
        volume,
        (surface, int(y0), int(x0), surface + 1, int(y1), int(x1)),
        fill_value=0,
    )
    return patch[0]


def _exclude_validation_voxels_from_training_supervision(
    supervision_patch,
    validation_patch,
    *,
    is_validation_patch=False,
):
    if is_validation_patch or validation_patch is None:
        return supervision_patch

    supervision_patch = np.asarray(supervision_patch)
    validation_patch = np.asarray(validation_patch)
    if supervision_patch.shape != validation_patch.shape:
        raise ValueError(
            "supervision_patch and validation_patch must have matching shapes, "
            f"got {tuple(supervision_patch.shape)} and {tuple(validation_patch.shape)}"
        )
    if supervision_patch.size == 0 or not np.any(validation_patch):
        return supervision_patch

    masked_supervision = np.array(supervision_patch, copy=True)
    masked_supervision[validation_patch > 0] = 0
    return masked_supervision


class InkDataset(Dataset):
    def __init__(self, config, do_augmentations=True, debug=False, patches=None, mode="flat"):
        
        self.debug            = debug
        self.config           = config
        self.patch_size       = config['patch_size']
        self.discovery_mode   = str(config.get('patch_discovery_mode', 'labeled')).strip().lower()
        if self.discovery_mode == 'unlabeled':
            self.datasets = list(config.get('unlabeled_datasets') or [])
        else:
            self.datasets = list(config['datasets'])
        self.vol_auth         = config.get('volume_auth_json')
        self.num_workers      = config.get('dataloader_workers', 8)
        self.mode             = str(config.get('mode', 'flat')).strip().lower()
        self.do_augmentations = bool(do_augmentations)
        self.input_channels   = 1 + int(_is_native_3d_mode(self.mode))
        self.training_patches = []
        self.validation_patches = []
        self.unlabeled_patches = []
        self._zarr_cache = {}
        self._tifxyz_cache = {}
        self._stored_resolution_zyx_cache = {}
  

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
                        is_unlabeled=bool(record.get('is_unlabeled', False)),
                        supervision_mask_override=record.get('active_supervision_mask_path') or None,
                    )
                    cached_patches.append(patch)
                    if patch.is_unlabeled:
                        self.unlabeled_patches.append(patch)
                        self.training_patches.append(patch)
                    elif patch.is_validation:
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
            self.unlabeled_patches = [patch for patch in self.training_patches if getattr(patch, 'is_unlabeled', False)]
            self.patches = self.training_patches + self.validation_patches
            save_flat_patch_cache(cache_path, self.patches)
        else:
            self.patches = list(patches)
            self.training_patches = [
                patch for patch in self.patches
                if not getattr(patch, 'is_validation', False)
            ]
            self.validation_patches = [patch for patch in self.patches if getattr(patch, 'is_validation', False)]
            self.unlabeled_patches = [patch for patch in self.patches if getattr(patch, 'is_unlabeled', False)]

    def _get_cached_zarr(self, path, *, resolution):
        cache_key = (str(path), str(resolution), str(self.vol_auth))
        volume = self._zarr_cache.get(cache_key)
        if volume is None:
            volume = open_zarr(path, resolution=resolution, auth=self.vol_auth)
            self._zarr_cache[cache_key] = volume
        return volume

    def _get_cached_tifxyz(self, segment_dir):
        cache_key = str(segment_dir)
        patch_tifxyz = self._tifxyz_cache.get(cache_key)
        if patch_tifxyz is None:
            patch_tifxyz = tifxyz.read_tifxyz(segment_dir)
            patch_tifxyz.use_full_resolution()
            self._tifxyz_cache[cache_key] = patch_tifxyz
        return patch_tifxyz

    def _get_cached_stored_resolution_zyxs(self, segment_dir, *, patch_tifxyz=None):
        cache_key = str(segment_dir)
        cached = self._stored_resolution_zyx_cache.get(cache_key)
        if cached is None:
            if patch_tifxyz is None:
                patch_tifxyz = self._get_cached_tifxyz(segment_dir)
            coarse_patch_zyxs = np.asarray(
                patch_tifxyz.get_zyxs(stored_resolution=True),
                dtype=np.float32,
            )
            coarse_valid = np.isfinite(coarse_patch_zyxs).all(axis=-1)
            coarse_valid &= (coarse_patch_zyxs >= 0).all(axis=-1)
            cached = (coarse_patch_zyxs, coarse_valid)
            self._stored_resolution_zyx_cache[cache_key] = cached
        return cached

    def _gather_segments(self):
        for dataset_idx, ds in enumerate(self.datasets):

            seg_path = Path(ds['segments_path'])

            for tifxyz_folder in sorted(seg_path.iterdir()):
                if not tifxyz_folder.is_dir() or tifxyz_folder.name == 'unused':
                    continue
                if not any(tifxyz_folder.rglob('x.tif')):
                    continue

                if _is_native_3d_mode(self.mode):
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
                if self.discovery_mode == 'unlabeled':
                    inklabels, supervision_mask, validation_mask = segment.discover_labels(
                        extension='.zarr',
                        required=False,
                    )
                else:
                    inklabels, supervision_mask, validation_mask = segment.discover_labels(extension='.zarr')

                if self.debug:
                    print(image_volume)
                    print(supervision_mask)
                    print(inklabels)
                    print(validation_mask)

                if not image_volume.exists():
                    raise ValueError(f"{tifxyz_folder.name} is missing its image volume")
                if self.discovery_mode != 'unlabeled' and not (
                    supervision_mask is not None
                    and inklabels is not None
                    and supervision_mask.exists()
                    and inklabels.exists()
                ):
                    raise ValueError(f"{tifxyz_folder.name} is missing required data. make sure the image volume, supervision mask, and labels exist")

                yield segment

    def get_labeled_unlabeled_patch_indices(self):
        labeled_indices = []
        unlabeled_indices = []
        for patch_idx, patch in enumerate(self.patches):
            if getattr(patch, 'is_validation', False):
                continue
            if getattr(patch, 'is_unlabeled', False):
                unlabeled_indices.append(int(patch_idx))
            else:
                labeled_indices.append(int(patch_idx))
        return labeled_indices, unlabeled_indices

    def __len__(self):
        return len(self.patches)

    def _choose_replacement_patch_index(self, *, current_idx):
        if len(self.patches) <= 1:
            raise RuntimeError("Cannot resample an oversized normal pooled patch from a dataset with <= 1 patch")
        seed = int(self.config.get('seed', 0))
        rng = random.Random(seed + (int(current_idx) * 7919))
        while True:
            replacement_idx = rng.randrange(len(self.patches))
            if replacement_idx != int(current_idx):
                return replacement_idx
    
    def __getitem__(self, idx):
        requested_idx = int(idx)
        current_idx = requested_idx

        while True:
            patch = self.patches[current_idx]
            z0, y0, x0, z1, y1, x1 = patch.bbox
            expected_shape = tuple(int(v) for v in self.patch_size)
            crop_bbox = patch.bbox
            use_surface_mask = _is_native_3d_mode(self.mode)
            surface_mask = None
            normal_pooled_metadata = None
            inklabels_crop = None
            supervision_crop = None
            resample_idx = None
            resample_warning_message = None
            
            # this entire if block only applies if you're using a "3d" mode. it samples the supervision in 2d 'flat' space,
            # and extracts the crop using the same patch finding as the 2d patch finding code. the result of this is that sometimes this patch
            # does not occupy the full 3d crop (or more than the full 3d crop). we handle this by either padding (adding adjacent quads until we reach crop size)
            # or by cropping. the supervision mask is built by first doing a 3d connected components on the surface voxels, and then filtering once again to the 2d
            # connected components "in crop". the first may be unnecessary.
            # the conditioning is a edt clipped to a dist of 10
            if _is_native_3d_mode(self.mode):
                image_vol = self._get_cached_zarr(patch.image_volume, resolution=patch.segment.scale)
                supervision_mask = self._get_cached_zarr(patch.supervision_mask, resolution=patch.segment.scale)
                inklabels = self._get_cached_zarr(patch.inklabels, resolution=patch.segment.scale)
                validation_mask = None
                if (not patch.is_validation) and patch.segment.validation_mask is not None:
                    validation_mask = self._get_cached_zarr(
                        patch.segment.validation_mask,
                        resolution=patch.segment.scale,
                    )
                patch_tifxyz = self._get_cached_tifxyz(patch.segment_dir)
                coarse_patch_zyxs, coarse_valid = self._get_cached_stored_resolution_zyxs(patch.segment_dir,patch_tifxyz=patch_tifxyz)

                flat_x, flat_y, flat_z, flat_valid = patch_tifxyz[y0:y1, x0:x1]
                patch_zyxs = np.stack([flat_z, flat_y, flat_x], axis=-1)
                try:
                    crop_bbox = compute_native_crop_bbox_from_patch_points(patch_zyxs,flat_valid,expected_shape)
                except ValueError as exc:
                    if str(exc) != "No valid tifxyz points found for patch":
                        raise
                    resample_idx = self._choose_replacement_patch_index(current_idx=current_idx)
                    resample_warning_message = (
                        f"Normal pooled patch had no valid tifxyz points "
                        f"for requested idx {requested_idx}, patch idx {current_idx}, "
                        f"segment {patch.segment.segment_name}; resampling idx {resample_idx}"
                    )
                if resample_idx is None:
                    supervision_flat_patch = _read_flat_surface_patch(supervision_mask,y0=y0,y1=y1,x0=x0,x1=x1)
                    
                    if validation_mask is not None:
                        validation_flat_patch = _read_flat_surface_patch(validation_mask,y0=y0,y1=y1,x0=x0,x1=x1)
                        supervision_flat_patch = _exclude_validation_voxels_from_training_supervision(
                            supervision_flat_patch,
                            validation_flat_patch,
                            is_validation_patch=patch.is_validation,
                        )
                        
                    if self.do_augmentations:
                        crop_bbox = maybe_translate_normal_pooled_crop_bbox(crop_bbox,patch_zyxs,flat_valid,supervision_flat_patch)
                    
                    (
                        base_support_bbox,
                        support_patch_zyxs,
                        support_valid,
                        support_patch_zyxs_halo,
                        support_valid_halo,
                        trim_slices,
                    ) = _select_flat_pixels_for_native_crop_via_stored_resolution(
                        patch_tifxyz,
                        crop_bbox,
                        coarse_patch_zyxs=coarse_patch_zyxs,
                        coarse_valid=coarse_valid,
                        return_halo=True,
                    )
                    support_y0, support_y1, support_x0, support_x1 = base_support_bbox
                    support_supervision_flat_patch = _read_flat_surface_patch(supervision_mask,y0=support_y0,y1=support_y1,x0=support_x0,x1=support_x1)
                    
                    if validation_mask is not None:
                        support_validation_flat_patch = _read_flat_surface_patch(validation_mask,y0=support_y0,y1=support_y1,x0=support_x0,x1=support_x1)
                        support_supervision_flat_patch = _exclude_validation_voxels_from_training_supervision(
                            support_supervision_flat_patch,
                            support_validation_flat_patch,
                            is_validation_patch=patch.is_validation,
                        )
                        
                    support_inklabels_flat_patch = _read_flat_surface_patch(inklabels,y0=support_y0,y1=support_y1,x0=support_x0,x1=support_x1)
                    pooling_config = self.config.get('normal_pooling') or {}
                    max_support_grid_distance = pooling_config.get('support_grid_max_distance', 64.0)
                    (
                        (support_y0, support_y1, support_x0, support_x1),
                        support_patch_zyxs,
                        support_valid,
                        support_inklabels_flat_patch,
                        support_supervision_flat_patch,
                    ) = _filter_support_components_by_active_supervision(
                        support_bbox=(support_y0, support_y1, support_x0, support_x1),
                        support_patch_zyxs=support_patch_zyxs,
                        support_valid=support_valid,
                        support_inklabels_flat_patch=support_inklabels_flat_patch,
                        support_supervision_flat_patch=support_supervision_flat_patch,
                        crop_bbox=crop_bbox,
                        patch_bbox=patch.bbox,
                        max_supervision_grid_distance=max_support_grid_distance,
                    )
                    (
                        support_patch_zyxs_halo,
                        support_valid_halo,
                        trim_slices,
                    ) = _slice_support_halo_for_subwindow(
                        support_patch_zyxs_halo,
                        support_valid_halo,
                        trim_slices,
                        base_support_bbox,
                        (support_y0, support_y1, support_x0, support_x1),
                    )

                    support_grid_shape = tuple(int(v) for v in support_valid.shape)
                    support_grid_side_limits = (int(expected_shape[1] * 4),int(expected_shape[2] * 4))
                    if support_grid_shape[0] > support_grid_side_limits[0] or support_grid_shape[1] > support_grid_side_limits[1]:
                        resample_idx = self._choose_replacement_patch_index(current_idx=current_idx)
                        resample_warning_message = (
                            f"Oversized normal pooled support grid {support_grid_shape!r} "
                            f"exceeded side limits {support_grid_side_limits!r} "
                            f"for requested idx {requested_idx}, patch idx {current_idx}, "
                            f"segment {patch.segment.segment_name}; resampling idx {resample_idx}"
                        )
                    else:
                        image_crop, image_valid_slices = _read_bbox_with_padding(image_vol, crop_bbox, fill_value=0)
                        surface_mask = _project_valid_surface_mask_to_native_crop(
                            support_patch_zyxs,
                            support_valid,
                            crop_bbox,
                        )
                        if self.mode == "normal_pooled_3d":
                            normal_pooled_metadata = _build_normal_pooled_flat_metadata(
                                support_patch_zyxs=support_patch_zyxs,
                                support_valid=support_valid,
                                support_patch_zyxs_halo=support_patch_zyxs_halo,
                                support_valid_halo=support_valid_halo,
                                trim_slices=trim_slices,
                                support_inklabels_flat_patch=support_inklabels_flat_patch,
                                support_supervision_flat_patch=support_supervision_flat_patch,
                                crop_bbox=crop_bbox,
                            )
                        else:
                            full_3d_config = self.config.get('full_3d') or {}
                            inklabels_crop, supervision_crop = _project_flat_labels_and_supervision_to_native_crop(
                                support_patch_zyxs=support_patch_zyxs,
                                support_valid=support_valid,
                                support_inklabels_flat_patch=support_inklabels_flat_patch,
                                support_supervision_flat_patch=support_supervision_flat_patch,
                                crop_bbox=crop_bbox,
                                label_dilation_distance=float(
                                    full_3d_config.get('label_dilation_distance', 0.0)
                                ),
                                supervision_dilation_distance=float(
                                    full_3d_config.get('supervision_dilation_distance', 0.0)
                                ),
                            )
            
            # for pooled 2d, this is the only block that applies (outside of potential resampling)
            else:
                image_vol = self._get_cached_zarr(patch.image_volume, resolution=patch.segment.scale)
                image_crop, image_valid_slices = _read_bbox_with_padding(image_vol, patch.bbox, fill_value=0)
                if getattr(patch, 'is_unlabeled', False):
                    supervision_crop = np.zeros(expected_shape, dtype=np.uint8)
                    inklabels_crop = np.zeros(expected_shape, dtype=np.uint8)
                else:
                    supervision_mask = self._get_cached_zarr(patch.supervision_mask, resolution=patch.segment.scale)
                    inklabels = self._get_cached_zarr(patch.inklabels, resolution=patch.segment.scale)
                    validation_mask = None
                    if (not patch.is_validation) and patch.segment.validation_mask is not None:
                        validation_mask = self._get_cached_zarr(patch.segment.validation_mask,resolution=patch.segment.scale)
                        
                    supervision_crop, _ = _read_bbox_with_padding(supervision_mask, patch.bbox, fill_value=0)
                    if validation_mask is not None:
                        validation_crop, _ = _read_bbox_with_padding(validation_mask, patch.bbox, fill_value=0)
                        supervision_crop = _exclude_validation_voxels_from_training_supervision(
                            supervision_crop,
                            validation_crop,
                            is_validation_patch=patch.is_validation,
                        )
                    inklabels_crop, _ = _read_bbox_with_padding(inklabels, patch.bbox, fill_value=0)

            if resample_idx is None:
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
                    assert inklabels_crop is not None
                    assert supervision_crop is not None
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
                        result = _restore_normal_pooled_augmentation_data(augmented, data, flat_valid_mask)
                    else:
                        augmentation_data = data
                        if self.mode == "full_3d" and 'surface_mask' in data:
                            augmentation_data = dict(data)
                            augmentation_data['regression_keys'] = ['surface_mask']
                        result = self.augmentations(**augmentation_data)
                else:
                    result = data

            if isinstance(result, dict):
                result['is_unlabeled'] = torch.tensor(bool(getattr(patch, 'is_unlabeled', False)), dtype=torch.bool)

            if resample_idx is not None:
                warnings.warn(
                    resample_warning_message,
                    RuntimeWarning,
                    stacklevel=2,
                )
                current_idx = resample_idx
                continue

            return result

if __name__ == "__main__":
    import argparse
    from qtpy.QtWidgets import QPushButton

    parser = argparse.ArgumentParser(description="Visualize an InkDataset sample in napari.")
    parser.add_argument("config_path", help="Path to the dataset config JSON.")
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ds = InkDataset(config, do_augmentations=False)

    import napari

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
