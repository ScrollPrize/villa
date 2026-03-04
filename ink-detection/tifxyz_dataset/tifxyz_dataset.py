import os
import numpy as np
import cv2

from torch.utils.data import Dataset

from common import (
    _build_normal_offset_mask_from_labeled_points,
    _get_segment_normals_zyx as _common_get_segment_normals_zyx,
    _get_segment_positive_points_zyx as _common_get_segment_positive_points_zyx,
    _load_segment_ink_mask as _common_load_segment_ink_mask,
    _normalize_patch_size_zyx,
    _points_to_voxels,
    _points_within_minmax,
    _read_volume_crop_from_patch_dict,
    _voxelize_positive_labels as _common_voxelize_positive_labels,
    _voxelize_surface as _common_voxelize_surface,
)
from patch_finding import _PATCH_CACHE_DEFAULT_FILENAME, find_patches
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms

class TifxyzInkDataset(Dataset):
    def __init__(
        self,
        config,
        apply_augmentation: bool = True,
        apply_perturbation: bool = True,
    ):
        self.apply_augmentation = apply_augmentation
        self.apply_perturbation = bool(apply_perturbation)
        self.patch_size = config["patch_size"]                                          # 3d vol crop / model input patch size
        self.bg_distance = config["bg_distance"]                                        # distance to project along normal (or edt), in "layers"
        self.label_distance = config["label_distance"]                                  # distance to project along normal (or edt), in "layers"
        self.bg_dilate_distance = int(config.get("bg_dilate_distance", 192))            # 2d surface dilation radius (pixels) used to define near-ink background
        self.normal_sample_step = float(config.get("normal_sample_step", 0.5))
        self.normal_trilinear_threshold = float(config.get("normal_trilinear_threshold", 1e-4))
        self.use_numba_for_normal_mask = bool(config.get("use_numba_for_normal_mask", True))
        self.patch_size_zyx = _normalize_patch_size_zyx(self.patch_size)        
        self.overlap_fraction = float(config.get("overlap_fraction", 0.25))             # amount of overlap (stride) in train/val patches, as a percentage of the patch size
        self.min_positive_fraction = float(config.get("min_positive_fraction", 0.01))   # minimum amount of labeled voxels in a candidate bbox to be added to our patches list, as a percentage of the total voxels
        self.min_span_ratio = float(config.get("min_span_ratio", 0.50))                 # the "span" in this instance is how far across the principle "direction" axis the segment should span (bbox local)
        self.patch_finding_workers = int(
            config.get("patch_finding_workers", 4)
        )                                                                               # workers for both z-band generation and bbox filtering
        self.patch_cache_force_recompute = bool(
            config.get("patch_cache_force_recompute", False)
        )
        self.patch_cache_filename = str(
            config.get("patch_cache_filename", _PATCH_CACHE_DEFAULT_FILENAME)
        )

        self.auto_fix_padding_multiples = [64, 256]                                     # if we find these common leftover padding multiples, we'll remove them

        self._segment_grid_cache = {}
        self._segment_ink_mask_cache = {}
        self._segment_surface_supervision_cache = {}
        self._segment_normal_cache = {}
        self._segment_positive_points_cache = {}
        self._segment_positive_samples_cache = {}
        self._segment_background_samples_cache = {}

        if apply_augmentation:                                                          # we'll use the vesuvius augmentation pipeline , see vesuvius/src/vesuvius/models/augmentation/pipelines/training_transforms.py
            self.augmentations = create_training_transforms(                            # for current defaults 
                patch_size=self.patch_size,                                             # TODO: make these configurable 
                no_spatial=False,
            )
        else:
            self.augmentations = None

        self.patches, self.patch_generation_stats = find_patches(                       # greedily add bboxes along the 2d tifxyz grid , adding a new patch any time we meet requirements for: 
            config,                                                                     
            patch_size_zyx=self.patch_size_zyx,                                         # - patch size (in 3d)       
            overlap_fraction=self.overlap_fraction,                                     # - 3d bbox overlap
            min_positive_fraction=self.min_positive_fraction,                           # - label percentage  
            min_span_ratio=self.min_span_ratio,                                         # - axis span
            patch_finding_workers=self.patch_finding_workers,
            patch_cache_force_recompute=self.patch_cache_force_recompute,               # see vesuvius/src/vesuvius/neural_tracing/inference/generate_segment_cover_bboxes.py  
            patch_cache_filename=self.patch_cache_filename,                             # for info on the bbox generation
            auto_fix_padding_multiples=self.auto_fix_padding_multiples,
        )

    def __len__(self):
        return len(self.patches)

    def _get_segment_stored_grid(self, segment):
        segment_uuid = str(segment.uuid)
        cached = self._segment_grid_cache.get(segment_uuid)
        if cached is not None:
            return cached

        segment.use_stored_resolution()
        x_stored, y_stored, z_stored, valid_stored = segment[:, :]

        x_stored = np.asarray(x_stored, dtype=np.float32)
        y_stored = np.asarray(y_stored, dtype=np.float32)
        z_stored = np.asarray(z_stored, dtype=np.float32)
        valid_mask = np.asarray(valid_stored, dtype=bool)
        valid_mask &= np.isfinite(x_stored)
        valid_mask &= np.isfinite(y_stored)
        valid_mask &= np.isfinite(z_stored)

        cached = {
            "x": x_stored,
            "y": y_stored,
            "z": z_stored,
            "valid": valid_mask,
            "shape": (int(x_stored.shape[0]), int(x_stored.shape[1])),
        }
        self._segment_grid_cache[segment_uuid] = cached
        return cached

    def _load_segment_ink_mask(self, segment):
        return _common_load_segment_ink_mask(self, segment)

    def _get_segment_positive_points_zyx(self, segment):
        return _common_get_segment_positive_points_zyx(self, segment)

    def _load_segment_surface_supervision(self, segment):
        segment_uuid = str(segment.uuid)
        cached = self._segment_surface_supervision_cache.get(segment_uuid)
        if cached is not None:
            return cached

        ink_mask = self._load_segment_ink_mask(segment)
        # Surface labels: 1=ink, 0=near-ink background, 100=ignore/far background.
        surface_supervision = np.full(ink_mask.shape, 100, dtype=np.uint8)
        surface_supervision[ink_mask] = 1

        dilate_radius = max(0, int(self.bg_dilate_distance))
        if dilate_radius > 0 and bool(np.any(ink_mask)):
            kernel_size = 2 * dilate_radius + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )
            dilated = cv2.dilate(ink_mask.astype(np.uint8), kernel, iterations=1) > 0
            surface_supervision[dilated & (~ink_mask)] = 0

        self._segment_surface_supervision_cache[segment_uuid] = surface_supervision
        return surface_supervision

    def _get_segment_positive_samples(self, segment):
        segment_uuid = str(segment.uuid)
        cached = self._segment_positive_samples_cache.get(segment_uuid)
        if cached is not None:
            return cached

        grid = self._get_segment_stored_grid(segment)
        ink_mask = self._load_segment_ink_mask(segment)
        if tuple(int(v) for v in ink_mask.shape) != tuple(int(v) for v in grid["shape"]):
            ink_mask = (
                cv2.resize(
                    ink_mask.astype(np.uint8),
                    (int(grid["shape"][1]), int(grid["shape"][0])),
                    interpolation=cv2.INTER_AREA,
                )
                > 0
            )

        positive_mask = np.asarray(grid["valid"] & ink_mask, dtype=bool)
        if not bool(np.any(positive_mask)):
            out = {
                "rows": np.empty((0,), dtype=np.int32),
                "cols": np.empty((0,), dtype=np.int32),
                "points_zyx": np.empty((0, 3), dtype=np.float32),
            }
            self._segment_positive_samples_cache[segment_uuid] = out
            self._segment_positive_points_cache[segment_uuid] = out["points_zyx"]
            return out

        row_idx, col_idx = np.where(positive_mask)
        row_idx = np.asarray(row_idx, dtype=np.int32)
        col_idx = np.asarray(col_idx, dtype=np.int32)
        points_zyx = np.stack(
            [
                grid["z"][row_idx, col_idx],
                grid["y"][row_idx, col_idx],
                grid["x"][row_idx, col_idx],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

        out = {
            "rows": row_idx,
            "cols": col_idx,
            "points_zyx": points_zyx,
        }
        self._segment_positive_samples_cache[segment_uuid] = out
        self._segment_positive_points_cache[segment_uuid] = points_zyx
        return out

    def _get_segment_background_samples(self, segment):
        segment_uuid = str(segment.uuid)
        cached = self._segment_background_samples_cache.get(segment_uuid)
        if cached is not None:
            return cached

        grid = self._get_segment_stored_grid(segment)
        surface_supervision = self._load_segment_surface_supervision(segment)
        if tuple(int(v) for v in surface_supervision.shape) != tuple(int(v) for v in grid["shape"]):
            surface_supervision = cv2.resize(
                surface_supervision.astype(np.uint8),
                (int(grid["shape"][1]), int(grid["shape"][0])),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8, copy=False)

        background_mask = np.asarray(grid["valid"] & (surface_supervision == 0), dtype=bool)
        if not bool(np.any(background_mask)):
            out = {
                "rows": np.empty((0,), dtype=np.int32),
                "cols": np.empty((0,), dtype=np.int32),
                "points_zyx": np.empty((0, 3), dtype=np.float32),
            }
            self._segment_background_samples_cache[segment_uuid] = out
            return out

        row_idx, col_idx = np.where(background_mask)
        row_idx = np.asarray(row_idx, dtype=np.int32)
        col_idx = np.asarray(col_idx, dtype=np.int32)
        points_zyx = np.stack(
            [
                grid["z"][row_idx, col_idx],
                grid["y"][row_idx, col_idx],
                grid["x"][row_idx, col_idx],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

        out = {
            "rows": row_idx,
            "cols": col_idx,
            "points_zyx": points_zyx,
        }
        self._segment_background_samples_cache[segment_uuid] = out
        return out

    def _get_segment_normals_zyx(self, segment):
        return _common_get_segment_normals_zyx(self, segment)

    def _voxelize_positive_labels(self, segment, min_corner, max_corner, crop_size):
        return _common_voxelize_positive_labels(
            self,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
        )

    def _voxelize_surface(self, segment, min_corner, max_corner, crop_size):
        return _common_voxelize_surface(
            self,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
        )

    def _voxelize_label_tolerance_band(self, segment, min_corner, max_corner, crop_size):
        positive_samples = self._get_segment_positive_samples(segment)
        return self._voxelize_normal_offset_band(
            segment=segment,
            samples=positive_samples,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
            label_distance=float(self.label_distance),
        )

    def _voxelize_background_surface_labels(self, segment, min_corner, max_corner, crop_size):
        crop_size_tuple = tuple(int(v) for v in crop_size)
        background_samples = self._get_segment_background_samples(segment)
        points_world = background_samples["points_zyx"]
        if points_world.shape[0] == 0:
            return np.zeros(crop_size_tuple, dtype=np.float32)

        in_bbox = _points_within_minmax(points_world, min_corner, max_corner)
        if not bool(np.any(in_bbox)):
            return np.zeros(crop_size_tuple, dtype=np.float32)

        local_points = points_world[in_bbox] - np.asarray(min_corner, dtype=np.float32)[None, :]
        return _points_to_voxels(local_points, crop_size_tuple)

    def _voxelize_background_tolerance_band(self, segment, min_corner, max_corner, crop_size):
        background_samples = self._get_segment_background_samples(segment)
        return self._voxelize_normal_offset_band(
            segment=segment,
            samples=background_samples,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
            label_distance=float(self.bg_distance),
        )

    def _voxelize_normal_offset_band(
        self,
        segment,
        samples,
        min_corner,
        max_corner,
        crop_size,
        label_distance,
    ):
        crop_size_tuple = tuple(int(v) for v in crop_size)
        points_world = samples["points_zyx"]
        if points_world.shape[0] == 0:
            return np.zeros(crop_size_tuple, dtype=np.float32)

        distance = float(label_distance)
        if distance <= 0.0:
            in_bbox = _points_within_minmax(points_world, min_corner, max_corner)
            if not bool(np.any(in_bbox)):
                return np.zeros(crop_size_tuple, dtype=np.float32)
            local_points = points_world[in_bbox] - np.asarray(min_corner, dtype=np.float32)[None, :]
            return _points_to_voxels(local_points, crop_size_tuple)

        normals_grid = self._get_segment_normals_zyx(segment)
        row_idx = samples["rows"]
        col_idx = samples["cols"]
        normals_zyx = normals_grid[row_idx, col_idx].astype(np.float32, copy=False)

        expand = distance + 1.0
        expanded_min = np.asarray(min_corner, dtype=np.float32) - expand
        expanded_max = np.asarray(max_corner, dtype=np.float32) + expand
        in_expanded = _points_within_minmax(points_world, expanded_min, expanded_max)
        if not bool(np.any(in_expanded)):
            return np.zeros(crop_size_tuple, dtype=np.float32)

        return _build_normal_offset_mask_from_labeled_points(
            points_world[in_expanded],
            normals_zyx[in_expanded],
            min_corner=min_corner,
            crop_size=crop_size_tuple,
            label_distance=distance,
            sample_step=float(self.normal_sample_step),
            trilinear_threshold=float(self.normal_trilinear_threshold),
            use_numba=bool(self.use_numba_for_normal_mask),
        )

    def _build_projected_loss_mask_volume(self, segment, min_corner, max_corner, crop_size):
        crop_size_tuple = tuple(int(v) for v in crop_size)
        out = np.full(crop_size_tuple, 2.0, dtype=np.float32)
        background_tolerance_vox = self._voxelize_background_tolerance_band(
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size_tuple,
        )
        out[background_tolerance_vox > 0.0] = 0.0

        label_tolerance_vox = self._voxelize_label_tolerance_band(
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size_tuple,
        )
        out[label_tolerance_vox > 0.0] = 1.0
        return out

    def _build_surface_label_volume(self, positive_label_vox, background_label_vox, crop_size):
        crop_size_tuple = tuple(int(v) for v in crop_size)
        out = np.full(crop_size_tuple, 2.0, dtype=np.float32)
        out[background_label_vox > 0.0] = 0.0
        out[positive_label_vox > 0.0] = 1.0
        return out

    def __getitem__(self, idx):
        patch = self.patches[idx]

        z0, z1, y0, y1, x0, x1 = patch['world_bbox']
        min_corner = np.array([z0, y0, x0], dtype=np.int32)
        max_corner = np.array([z1 + 1, y1 + 1, x1 + 1], dtype=np.int32)
        crop_size = tuple(int(v) for v in self.patch_size_zyx)

        vol_crop = _read_volume_crop_from_patch_dict(
            patch,
            crop_size=crop_size,
            min_corner=min_corner,
            max_corner=max_corner,
        )
        
        segment = patch["segment"]

        positive_label_vox = self._voxelize_positive_labels(
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
        )
        background_label_vox = self._voxelize_background_surface_labels(
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
        )
        surface_vox = self._voxelize_surface(
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
        )
        surface_label_vox = self._build_surface_label_volume(
            positive_label_vox=positive_label_vox,
            background_label_vox=background_label_vox,
            crop_size=crop_size,
        )
        mil_loss_mask_vox = self._build_projected_loss_mask_volume(
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
        )

        return {
            "vol": vol_crop,
            "positive_label_vox": positive_label_vox,
            "background_label_vox": background_label_vox,
            "surface_label_vox": surface_label_vox,
            "surface_vox": surface_vox,
            "mil_loss_mask_vox": mil_loss_mask_vox,
            "patch": patch,
            "idx": int(idx),
        }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect the TifxyzInkDataset.")
    parser.add_argument(
        "--napari",
        action="store_true",
        help="Iterate the dataset once and visualize outputs in a Napari viewer.",
    )
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "example_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ds = TifxyzInkDataset(
        config,
        apply_augmentation=False,
        apply_perturbation=False,
    )

    print(f"loaded patches: {len(ds)}")
    print(json.dumps(ds.patch_generation_stats, indent=2, sort_keys=True))

    if args.napari:
        try:
            import napari
        except ImportError as exc:
            raise ImportError(
                "napari is required for --napari. Install it and re-run."
            ) from exc

        output_keys = (
            "vol",
            "positive_label_vox",
            "background_label_vox",
            "surface_label_vox",
            "surface_vox",
            "mil_loss_mask_vox",
        )
        stacked_outputs = {k: [] for k in output_keys}

        for sample in ds:
            for key in output_keys:
                stacked_outputs[key].append(np.asarray(sample[key]))

        viewer = napari.Viewer()
        for key in output_keys:
            if not stacked_outputs[key]:
                continue
            viewer.add_image(np.stack(stacked_outputs[key], axis=0), name=key)

        napari.run()
