import json
import time
from collections import defaultdict
from contextlib import contextmanager
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
from koine_machines.data.normal_pooled_sample import (
    _build_normal_pooled_flat_metadata,
    _pack_normal_pooled_augmentation_data,
    _restore_normal_pooled_augmentation_data,
)
from koine_machines.data.native_crop import compute_native_crop_bbox_from_patch_points
from koine_machines.data.segment import Segment


class DatasetSampleProfiler:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.section_totals = defaultdict(float)

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            self.section_totals[name] += time.perf_counter() - start

    def as_dict(self) -> dict[str, float]:
        return dict(self.section_totals)

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

    # We only need the enclosing row/column span, not every matching index.
    row_hits = np.any(within, axis=1)
    col_hits = np.any(within, axis=0)
    row_indices = np.flatnonzero(row_hits)
    col_indices = np.flatnonzero(col_hits)
    support_y0 = int(row_indices[0])
    support_y1 = int(row_indices[-1]) + 1
    support_x0 = int(col_indices[0])
    support_x1 = int(col_indices[-1]) + 1
    return (
        (support_y0, support_y1, support_x0, support_x1),
        patch_zyxs[support_y0:support_y1, support_x0:support_x1],
        within[support_y0:support_y1, support_x0:support_x1],
    )


def _select_flat_pixels_for_native_crop_via_stored_resolution(
    patch_tifxyz,
    crop_bbox,
    *,
    coarse_native_pad=20,
    coarse_patch_zyxs=None,
    coarse_valid=None,
):
    coarse_native_pad = int(coarse_native_pad)
    coarse_crop_bbox = (
        int(crop_bbox[0]) - coarse_native_pad,
        int(crop_bbox[1]) - coarse_native_pad,
        int(crop_bbox[2]) - coarse_native_pad,
        int(crop_bbox[3]) + coarse_native_pad,
        int(crop_bbox[4]) + coarse_native_pad,
        int(crop_bbox[5]) + coarse_native_pad,
    )

    if coarse_patch_zyxs is None:
        coarse_patch_zyxs = np.asarray(
            patch_tifxyz.get_zyxs(stored_resolution=True),
            dtype=np.float32,
        )
    else:
        coarse_patch_zyxs = np.asarray(coarse_patch_zyxs, dtype=np.float32)

    if coarse_valid is None:
        coarse_valid = np.isfinite(coarse_patch_zyxs).all(axis=-1)
        coarse_valid &= (coarse_patch_zyxs >= 0).all(axis=-1)
    else:
        coarse_valid = np.asarray(coarse_valid, dtype=bool)

    (coarse_y0, coarse_y1, coarse_x0, coarse_x1), _, _ = _select_flat_pixels_for_native_crop(
        coarse_patch_zyxs,
        coarse_valid,
        coarse_crop_bbox,
    )

    stored_h, stored_w = (int(v) for v in coarse_patch_zyxs.shape[:2])
    full_h, full_w = (int(v) for v in patch_tifxyz.full_resolution_shape)
    if stored_h <= 0 or stored_w <= 0:
        raise ValueError(f"stored-resolution tifxyz grid must have positive shape, got {(stored_h, stored_w)!r}")

    factor_y = full_h / float(stored_h)
    factor_x = full_w / float(stored_w)

    # Expand by one stored cell before mapping back to full resolution so the
    # exact full-res refinement can't miss intersections near a coarse edge.
    coarse_y0 = max(0, coarse_y0 - 1)
    coarse_y1 = min(stored_h, coarse_y1 + 1)
    coarse_x0 = max(0, coarse_x0 - 1)
    coarse_x1 = min(stored_w, coarse_x1 + 1)

    full_y0 = max(0, int(np.floor(coarse_y0 * factor_y)))
    full_y1 = min(full_h, int(np.ceil(coarse_y1 * factor_y)))
    full_x0 = max(0, int(np.floor(coarse_x0 * factor_x)))
    full_x1 = min(full_w, int(np.ceil(coarse_x1 * factor_x)))

    full_x, full_y, full_z, full_valid = patch_tifxyz[full_y0:full_y1, full_x0:full_x1]
    full_patch_zyxs = np.stack([full_z, full_y, full_x], axis=-1)
    (local_y0, local_y1, local_x0, local_x1), support_patch_zyxs, support_valid = _select_flat_pixels_for_native_crop(
        full_patch_zyxs,
        full_valid,
        crop_bbox,
    )
    return (
        (full_y0 + local_y0, full_y0 + local_y1, full_x0 + local_x0, full_x0 + local_x1),
        support_patch_zyxs,
        support_valid,
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
        self.profile_enabled  = bool(config.get('profile_dataset', False))
        self.profile_section_totals = defaultdict(float)
        self.profile_sample_count = 0
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

    def _record_profile_timings(self, profile_timings: dict[str, float]):
        if not self.profile_enabled:
            return
        self.profile_sample_count += 1
        for name, duration_seconds in profile_timings.items():
            self.profile_section_totals[name] += float(duration_seconds)

    def profile_summary_lines(self) -> list[str]:
        if not self.profile_enabled or self.profile_sample_count == 0:
            return []

        lines = [f"[profile] dataset timings across {self.profile_sample_count} samples"]
        for name, total_seconds in sorted(self.profile_section_totals.items(), key=lambda item: item[1], reverse=True):
            avg_ms = (total_seconds / self.profile_sample_count) * 1000.0
            lines.append(f"[profile] {name}: total={total_seconds:.6f}s avg_sample={avg_ms:.3f}ms")
        return lines

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
        sample_profiler = DatasetSampleProfiler(self.profile_enabled)
        z0, y0, x0, z1, y1, x1 = patch.bbox
        expected_shape = tuple(int(v) for v in self.patch_size)
        crop_bbox = patch.bbox
        use_surface_mask = self.mode == "normal_pooled_3d"
        surface_mask = None
        normal_pooled_metadata = None

        with sample_profiler.section('dataset/total'):
            if self.mode == "normal_pooled_3d":
                with sample_profiler.section('dataset/get_cached_inputs'):
                    image_vol = self._get_cached_zarr(patch.image_volume, resolution=patch.segment.scale)
                    supervision_mask = self._get_cached_zarr(patch.supervision_mask, resolution=patch.segment.scale)
                    inklabels = self._get_cached_zarr(patch.inklabels, resolution=patch.segment.scale)
                    patch_tifxyz = self._get_cached_tifxyz(patch.segment_dir)
                    coarse_patch_zyxs, coarse_valid = self._get_cached_stored_resolution_zyxs(
                        patch.segment_dir,
                        patch_tifxyz=patch_tifxyz,
                    )
                with sample_profiler.section('dataset/read_patch_tifxyz'):
                    flat_x, flat_y, flat_z, flat_valid = patch_tifxyz[y0:y1, x0:x1]
                    patch_zyxs = np.stack([flat_z, flat_y, flat_x], axis=-1)
                with sample_profiler.section('dataset/compute_crop_bbox'):
                    crop_bbox = compute_native_crop_bbox_from_patch_points(
                        patch_zyxs,
                        flat_valid,
                        expected_shape,
                    )
                with sample_profiler.section('dataset/read_seed_supervision'):
                    supervision_flat_patch = _read_flat_surface_patch(
                        supervision_mask,
                        y0=y0,
                        y1=y1,
                        x0=x0,
                        x1=x1,
                    )
                if self.do_augmentations:
                    with sample_profiler.section('dataset/translate_crop_bbox'):
                        crop_bbox = maybe_translate_normal_pooled_crop_bbox(
                            crop_bbox,
                            patch_zyxs,
                            flat_valid,
                            supervision_flat_patch,
                        )
                with sample_profiler.section('dataset/select_support_window'):
                    (support_y0, support_y1, support_x0, support_x1), support_patch_zyxs, support_valid = _select_flat_pixels_for_native_crop_via_stored_resolution(
                        patch_tifxyz,
                        crop_bbox,
                        coarse_patch_zyxs=coarse_patch_zyxs,
                        coarse_valid=coarse_valid,
                    )
                with sample_profiler.section('dataset/read_support_labels'):
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
                with sample_profiler.section('dataset/read_image_crop'):
                    image_crop, image_valid_slices = _read_bbox_with_padding(image_vol, crop_bbox, fill_value=0)
                with sample_profiler.section('dataset/project_surface_mask'):
                    surface_mask = _project_valid_surface_mask_to_native_crop(
                        support_patch_zyxs,
                        support_valid,
                        crop_bbox,
                    )
                with sample_profiler.section('dataset/build_metadata'):
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
                with sample_profiler.section('dataset/get_cached_inputs'):
                    image_vol = self._get_cached_zarr(patch.image_volume, resolution=patch.segment.scale)
                    supervision_mask = self._get_cached_zarr(patch.supervision_mask, resolution=patch.segment.scale)
                    inklabels = self._get_cached_zarr(patch.inklabels, resolution=patch.segment.scale)
                with sample_profiler.section('dataset/read_supervised_crops'):
                    image_crop, image_valid_slices = _read_bbox_with_padding(image_vol, patch.bbox, fill_value=0)
                    supervision_crop, _ = _read_bbox_with_padding(supervision_mask, patch.bbox, fill_value=0)
                    inklabels_crop, _ = _read_bbox_with_padding(inklabels, patch.bbox, fill_value=0)

            with sample_profiler.section('dataset/normalize_image'):
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
            with sample_profiler.section('dataset/validate_shapes'):
                for name, array in arrays_to_validate:
                    if tuple(int(v) for v in array.shape) != expected_shape:
                        raise AssertionError(
                            f"{name} crop shape {tuple(int(v) for v in array.shape)} does not match "
                            f"requested patch size {expected_shape} for bbox {crop_bbox!r}"
                        )

            with sample_profiler.section('dataset/to_torch'):
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
                with sample_profiler.section('dataset/augment'):
                    if self.mode == "normal_pooled_3d":
                        augmentation_data, flat_valid_mask = _pack_normal_pooled_augmentation_data(data)
                        augmented = self.augmentations(**augmentation_data)
                        result = _restore_normal_pooled_augmentation_data(augmented, data, flat_valid_mask)
                    else:
                        result = self.augmentations(**data)
            else:
                result = data

        profile_timings = sample_profiler.as_dict()
        self._record_profile_timings(profile_timings)
        if self.profile_enabled:
            result = dict(result)
            result['profile_timings'] = profile_timings
        return result

if __name__ == "__main__":
    import argparse
    from qtpy.QtWidgets import QPushButton

    parser = argparse.ArgumentParser(description="Visualize an InkDataset sample in napari.")
    parser.add_argument("config_path", help="Path to the dataset config JSON.")
    parser.add_argument(
        "--profile",
        type=int,
        default=None,
        help="Load the given number of dataset samples and print the total elapsed time.",
    )
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config['profile_dataset'] = args.profile is not None

    ds = InkDataset(config, do_augmentations=False)
    if args.profile is not None:
        num_samples = max(0, min(int(args.profile), len(ds)))
        start = time.perf_counter()
        for index in tqdm(range(num_samples), desc="Profiling dataset samples"):
            _ = ds[index]
        elapsed = time.perf_counter() - start
        print(f"Profiled {num_samples} samples in {elapsed:.3f}s")
        for line in ds.profile_summary_lines():
            print(line)
        raise SystemExit(0)

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
