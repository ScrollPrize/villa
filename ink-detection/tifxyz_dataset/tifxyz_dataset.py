import os
import numpy as np
import cv2

from torch.utils.data import Dataset

from common import (
    _fix_known_bottom_right_padding,
    _normalize_patch_size_zyx,
    _points_to_voxels,
    _points_within_minmax,
    _read_volume_crop_from_patch_dict,
)
from patch_finding import _PATCH_CACHE_DEFAULT_FILENAME, find_patches
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid_masked
from vesuvius.tifxyz import interpolate_at_points

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
        self._segment_positive_points_cache = {}

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
        segment_uuid = str(segment.uuid)
        cached = self._segment_ink_mask_cache.get(segment_uuid)
        if cached is not None:
            return cached

        grid = self._get_segment_stored_grid(segment)
        expected_shape = tuple(int(v) for v in grid["shape"])
        ink_meta = next(
            (label for label in segment.list_labels() if label.get("name") == "inklabels"),
            None,
        )
        if ink_meta is None:
            out = np.zeros(expected_shape, dtype=bool)
            self._segment_ink_mask_cache[segment_uuid] = out
            return out

        ink_label = cv2.imread(str(ink_meta["path"]), cv2.IMREAD_UNCHANGED)
        if ink_label is None:
            out = np.zeros(expected_shape, dtype=bool)
            self._segment_ink_mask_cache[segment_uuid] = out
            return out
        if ink_label.ndim == 3:
            if ink_label.shape[2] == 4:
                ink_label = cv2.cvtColor(ink_label, cv2.COLOR_BGRA2GRAY)
            else:
                ink_label = cv2.cvtColor(ink_label, cv2.COLOR_BGR2GRAY)

        if tuple(int(v) for v in ink_label.shape) != expected_shape:
            fixed_label, _ = _fix_known_bottom_right_padding(
                ink_label,
                expected_shape,
                self.auto_fix_padding_multiples,
            )
            if fixed_label is not None:
                ink_label = fixed_label
            else:
                ink_label = (
                    cv2.resize(
                        (ink_label > 0).astype(np.uint8),
                        (int(expected_shape[1]), int(expected_shape[0])),
                        interpolation=cv2.INTER_AREA,
                    )
                    > 0
                ).astype(np.uint8)

        out = np.asarray(ink_label > 0, dtype=bool)
        self._segment_ink_mask_cache[segment_uuid] = out
        return out

    def _get_segment_positive_points_zyx(self, segment):
        segment_uuid = str(segment.uuid)
        cached = self._segment_positive_points_cache.get(segment_uuid)
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
            out = np.empty((0, 3), dtype=np.float32)
            self._segment_positive_points_cache[segment_uuid] = out
            return out

        row_idx, col_idx = np.where(positive_mask)
        out = np.stack(
            [
                grid["z"][row_idx, col_idx],
                grid["y"][row_idx, col_idx],
                grid["x"][row_idx, col_idx],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        self._segment_positive_points_cache[segment_uuid] = out
        return out

    def _voxelize_positive_labels(self, segment, min_corner, max_corner, crop_size):
        positive_points_world = self._get_segment_positive_points_zyx(segment)
        if positive_points_world.shape[0] == 0:
            return np.zeros(tuple(int(v) for v in crop_size), dtype=np.float32)

        in_bbox = _points_within_minmax(positive_points_world, min_corner, max_corner)
        if not bool(np.any(in_bbox)):
            return np.zeros(tuple(int(v) for v in crop_size), dtype=np.float32)

        local_points = positive_points_world[in_bbox] - np.asarray(min_corner, dtype=np.float32)[None, :]
        return _points_to_voxels(local_points, crop_size)

    def _voxelize_surface(self, segment, min_corner, max_corner, crop_size):
        crop_size_tuple = tuple(int(v) for v in crop_size)
        grid = self._get_segment_stored_grid(segment)
        x_stored = grid["x"]
        y_stored = grid["y"]
        z_stored = grid["z"]
        valid_mask = grid["valid"]

        in_bbox = (
            valid_mask &
            (z_stored >= float(min_corner[0])) & (z_stored < float(max_corner[0])) &
            (y_stored >= float(min_corner[1])) & (y_stored < float(max_corner[1])) &
            (x_stored >= float(min_corner[2])) & (x_stored < float(max_corner[2]))
        )
        if not bool(np.any(in_bbox)):
            return np.zeros(crop_size_tuple, dtype=np.float32)

        rows, cols = np.where(in_bbox)
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())
        query_rows = np.arange(row_min, row_max + 1, dtype=np.float32)
        query_cols = np.arange(col_min, col_max + 1, dtype=np.float32)
        query_y, query_x = np.meshgrid(query_rows, query_cols, indexing="ij")

        x_int, y_int, z_int, int_valid = interpolate_at_points(
            x_stored,
            y_stored,
            z_stored,
            valid_mask,
            query_y,
            query_x,
            scale=(1.0, 1.0),
            method="catmull_rom",
            invalid_value=-1.0,
        )
        zyx_world = np.stack([z_int, y_int, x_int], axis=-1).astype(np.float32, copy=False)
        valid_interp = np.asarray(int_valid, dtype=bool)
        valid_interp &= np.isfinite(zyx_world).all(axis=-1)
        valid_interp &= (
            (zyx_world[..., 0] >= float(min_corner[0])) & (zyx_world[..., 0] < float(max_corner[0])) &
            (zyx_world[..., 1] >= float(min_corner[1])) & (zyx_world[..., 1] < float(max_corner[1])) &
            (zyx_world[..., 2] >= float(min_corner[2])) & (zyx_world[..., 2] < float(max_corner[2]))
        )
        if not bool(np.any(valid_interp)):
            return np.zeros(crop_size_tuple, dtype=np.float32)

        local_grid = zyx_world - np.asarray(min_corner, dtype=np.float32).reshape(1, 1, 3)
        return voxelize_surface_grid_masked(local_grid, crop_size_tuple, valid_interp).astype(
            np.float32,
            copy=False,
        )

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
        surface_vox = self._voxelize_surface(
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
        )

        return {
            "vol": vol_crop,
            "positive_label_vox": positive_label_vox,
            "surface_vox": surface_vox,
            "patch": patch,
            "idx": int(idx),
        }


if __name__ == "__main__":
    import json

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
