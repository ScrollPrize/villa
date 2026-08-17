import os
import numpy as np
import warnings
import torch

from torch.utils.data import Dataset

from .common import (
    _build_normal_offset_mask_from_labeled_points,
    _build_projected_loss_mask_volume,
    _build_surface_label_volume,
    _build_surface_supervision_from_ink_mask,
    _load_segment_ink_mask,
    _normalize_distance_pair,
    _normalize_patch_size_zyx,
    _read_volume_crop_from_patch_dict,
    _sample_patch_supervision_grid,
    _voxelize_background_surface_labels_from_sampled_grid,
    _voxelize_positive_labels_from_sampled_grid,
    _voxelize_surface_from_sampled_grid,
)
from .patch_finding import _PATCH_CACHE_DEFAULT_FILENAME, find_patches
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
        self.bg_distance = config["bg_distance"]                                        # accepts scalar or [positive, negative]
        self.label_distance = config["label_distance"]                                  # accepts scalar or [positive, negative]
        self.bg_distance_pos, self.bg_distance_neg = _normalize_distance_pair(
            self.bg_distance,
            name="bg_distance",
        )
        self.label_distance_pos, self.label_distance_neg = _normalize_distance_pair(
            self.label_distance,
            name="label_distance",
        )
        self.bg_distance_max = max(self.bg_distance_pos, self.bg_distance_neg)
        self.label_distance_max = max(self.label_distance_pos, self.label_distance_neg)
        self.bg_dilate_distance = int(config.get("bg_dilate_distance", 192))            # 2d label EDT radius (pixels) used to define near-ink background
        self.normal_sample_step = float(config.get("normal_sample_step", 0.5))
        self.normal_trilinear_threshold = float(config.get("normal_trilinear_threshold", 1e-4))
        self.use_numba_for_normal_mask = bool(config.get("use_numba_for_normal_mask", True))
        self.surface_bbox_pad = float(config.get("surface_bbox_pad", 2.0))
        if self.surface_bbox_pad < 0.0:
            self.surface_bbox_pad = 0.0
        self.surface_interp_method = str(config.get("surface_interp_method", "catmull_rom")).strip().lower()
        self.wrap_mode = str(config.get("wrap_mode", "single_wrap")).strip().lower()
        if self.wrap_mode not in {"single_wrap", "multi_wrap"}:
            warnings.warn(
                f"Unknown wrap_mode={self.wrap_mode!r}; falling back to 'single_wrap'."
            )
            self.wrap_mode = "single_wrap"
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
        self._segment_world_bounds_cache = {}
        self._segment_positive_points_cache = {}
        self._segment_positive_samples_cache = {}
        self._segment_background_samples_cache = {}
        self._multi_wrap_segments_by_dataset_idx = {}
        self._multi_wrap_candidate_segments_cache = {}

        if apply_augmentation:                                                          # we'll use the vesuvius augmentation pipeline , see vesuvius/src/vesuvius/models/augmentation/pipelines/training_transforms.py
            self.augmentations = create_training_transforms(                            # for current defaults 
                patch_size=tuple(int(v) for v in self.patch_size_zyx),                  # TODO: make these configurable
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
        self._segment_ink_label_path_by_uuid = {}
        for patch in self.patches:
            segment_uuid = str(patch.get("segment_uuid", ""))
            dataset_idx = int(patch.get("dataset_idx", -1))
            segment = patch.get("segment")
            if segment is not None:
                dataset_segments = self._multi_wrap_segments_by_dataset_idx.setdefault(
                    dataset_idx,
                    {},
                )
                dataset_segments.setdefault(segment_uuid, segment)

            ink_label_path = patch.get("ink_label_path")
            if not ink_label_path:
                warnings.warn(
                    f"Unable to load ink labels for segment: {segment_uuid}"
                )
                continue
            self._segment_ink_label_path_by_uuid[segment_uuid] = str(ink_label_path)

        self._multi_wrap_segments_by_dataset_idx = {
            int(dataset_idx): tuple(segment_map.values())
            for dataset_idx, segment_map in self._multi_wrap_segments_by_dataset_idx.items()
        }



    def _apply_sample_augmentation(
        self,
        vol_crop,
        labeled_vox_at_surface,
        surface_vox,
        projected_loss_mask,
    ):
        image = torch.as_tensor(
            np.asarray(vol_crop, dtype=np.float32),
            dtype=torch.float32,
        ).unsqueeze(0)
        segmentation = torch.as_tensor(
            np.stack(
                (
                    np.asarray(labeled_vox_at_surface, dtype=np.float32),
                    np.asarray(surface_vox, dtype=np.float32),
                    np.asarray(projected_loss_mask, dtype=np.float32),
                ),
                axis=0,
            ),
            dtype=torch.float32,
        )

        augmented = self.augmentations(
            image=image,
            segmentation=segmentation,
        )
        image_out = augmented["image"]
        segmentation_out = augmented["segmentation"]

        return (
            image_out[0].to(dtype=torch.float32).cpu(),
            segmentation_out[0].to(dtype=torch.float32).cpu(),
            segmentation_out[1].to(dtype=torch.float32).cpu(),
            segmentation_out[2].to(dtype=torch.float32).cpu(),
        )

    @staticmethod
    def _to_float32_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=torch.float32)
        return torch.as_tensor(
            np.asarray(value, dtype=np.float32),
            dtype=torch.float32,
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

    def _load_segment_surface_supervision(self, segment):
        segment_uuid = str(segment.uuid)
        cached = self._segment_surface_supervision_cache.get(segment_uuid)
        if cached is not None:
            return cached

        ink_mask = _load_segment_ink_mask(self, segment)
        surface_supervision = _build_surface_supervision_from_ink_mask(
            ink_mask,
            bg_dilate_distance=self.bg_dilate_distance,
        )

        self._segment_surface_supervision_cache[segment_uuid] = surface_supervision
        return surface_supervision

    @staticmethod
    def _segment_bounds_intersect_crop(segment_bounds, min_corner, max_corner):
        if segment_bounds is None:
            return False

        min_corner = np.asarray(min_corner, dtype=np.float32).reshape(3)
        max_corner = np.asarray(max_corner, dtype=np.float32).reshape(3)
        z_min, z_max, y_min, y_max, x_min, x_max = [float(v) for v in segment_bounds]
        return not (
            (z_max < float(min_corner[0])) or (z_min >= float(max_corner[0])) or
            (y_max < float(min_corner[1])) or (y_min >= float(max_corner[1])) or
            (x_max < float(min_corner[2])) or (x_min >= float(max_corner[2]))
        )

    def _get_segment_world_bounds(self, segment):
        segment_uuid = str(segment.uuid)
        if segment_uuid in self._segment_world_bounds_cache:
            return self._segment_world_bounds_cache[segment_uuid]

        grid = self._get_segment_stored_grid(segment)
        valid = np.asarray(grid["valid"], dtype=bool)
        if not bool(np.any(valid)):
            self._segment_world_bounds_cache[segment_uuid] = None
            return None

        z_vals = np.asarray(grid["z"], dtype=np.float32)[valid]
        y_vals = np.asarray(grid["y"], dtype=np.float32)[valid]
        x_vals = np.asarray(grid["x"], dtype=np.float32)[valid]
        bounds = (
            float(np.min(z_vals)),
            float(np.max(z_vals)),
            float(np.min(y_vals)),
            float(np.max(y_vals)),
            float(np.min(x_vals)),
            float(np.max(x_vals)),
        )
        self._segment_world_bounds_cache[segment_uuid] = bounds
        return bounds

    @staticmethod
    def _merge_projected_loss_mask(combined_mask, segment_mask):
        segment_bg = np.asarray(segment_mask == 0.0, dtype=bool)
        segment_pos = np.asarray(segment_mask == 1.0, dtype=bool)
        combined_mask[segment_bg & (combined_mask != 1.0)] = 0.0
        combined_mask[segment_pos] = 1.0

    @staticmethod
    def _suppress_projected_mask_overlap_with_foreign_labels(
        projected_mask,
        foreign_labeled_surface_mask,
        own_labeled_surface_mask=None,
    ):
        foreign_mask = np.asarray(foreign_labeled_surface_mask > 0.0, dtype=bool)
        if own_labeled_surface_mask is not None:
            own_mask = np.asarray(own_labeled_surface_mask > 0.0, dtype=bool)
            foreign_mask &= ~own_mask
        if bool(np.any(foreign_mask)):
            projected_mask[foreign_mask] = 2.0

    def _get_multi_wrap_candidate_segments(self, idx, patch, min_corner, max_corner):
        cached = self._multi_wrap_candidate_segments_cache.get(int(idx))
        if cached is not None:
            return cached

        if self.wrap_mode != "multi_wrap":
            out = tuple()
            self._multi_wrap_candidate_segments_cache[int(idx)] = out
            return out

        dataset_idx = int(patch.get("dataset_idx", -1))
        patch_segment_uuid = str(patch.get("segment_uuid", ""))
        patch_segment = patch.get("segment")
        if patch_segment and not patch_segment_uuid:
            patch_segment_uuid = str(patch_segment.uuid)

        candidate_segments = []
        for segment in self._multi_wrap_segments_by_dataset_idx.get(dataset_idx, ()):
            segment_uuid = str(segment.uuid)
            if segment_uuid == patch_segment_uuid:
                continue
            segment_bounds = self._get_segment_world_bounds(segment)
            if not self._segment_bounds_intersect_crop(segment_bounds, min_corner, max_corner):
                continue
            candidate_segments.append(segment)

        out = tuple(candidate_segments)
        self._multi_wrap_candidate_segments_cache[int(idx)] = out
        return out

    def _voxelize_full_surface_background_tolerance_from_sampled_grid(
        self,
        segment,
        min_corner,
        max_corner,
        crop_size,
        sampled_grid,
    ):
        in_patch_with_normals = sampled_grid["in_patch"] & sampled_grid["normals_valid"]
        if bool(np.any(in_patch_with_normals)):
            return _build_normal_offset_mask_from_labeled_points(
                sampled_grid["world_grid"][in_patch_with_normals],
                sampled_grid["normals_zyx"][in_patch_with_normals],
                min_corner=min_corner,
                crop_size=crop_size,
                label_distance=(self.bg_distance_pos, self.bg_distance_neg),
                sample_step=float(self.normal_sample_step),
                trilinear_threshold=float(self.normal_trilinear_threshold),
                use_numba=bool(self.use_numba_for_normal_mask),
            )

        return _voxelize_surface_from_sampled_grid(
            self,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
            sampled_grid=sampled_grid,
        )

    def __getitem__(self, idx):
        patch = self.patches[idx]

        z0, z1, y0, y1, x0, x1 = patch['world_bbox']
        min_corner = np.array([z0, y0, x0], dtype=np.int32)
        max_corner = np.array([z1 + 1, y1 + 1, x1 + 1], dtype=np.int32)
        crop_size = tuple(int(v) for v in self.patch_size_zyx)

        # zscore normalization is applied within this function , DO NOT MINMAX NORMALIZE AFTER WITHOUT CONSIDERING THIS
        vol_crop = _read_volume_crop_from_patch_dict(
            patch,
            crop_size=crop_size,
            min_corner=min_corner,
            max_corner=max_corner,
        )
        
        segment = patch["segment"]
        
        sampled_grid = _sample_patch_supervision_grid(
            self,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            extra_bbox_pad=max(float(self.bg_distance_max), float(self.label_distance_max)) + 1.0,
        )

        positive_label_vox = _voxelize_positive_labels_from_sampled_grid(
            self,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
            sampled_grid=sampled_grid,
        )
        background_label_vox = _voxelize_background_surface_labels_from_sampled_grid(
            self,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
            sampled_grid=sampled_grid,
        )
        surface_vox = _voxelize_surface_from_sampled_grid(
            self,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
            sampled_grid=sampled_grid,
        )
        projected_loss_mask = _build_projected_loss_mask_volume(
            self,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
            sampled_grid=sampled_grid,
        )
        
        if self.wrap_mode == "multi_wrap":
            extra_bbox_pad = max(float(self.bg_distance_max), float(self.label_distance_max)) + 1.0
            labeled_surface_vox_union = np.asarray(
                (positive_label_vox > 0.0) | (background_label_vox > 0.0),
                dtype=bool,
            )
            other_segments = self._get_multi_wrap_candidate_segments(
                idx=idx,
                patch=patch,
                min_corner=min_corner,
                max_corner=max_corner,
            )
            for other_segment in other_segments:
                other_sampled_grid = _sample_patch_supervision_grid(
                    self,
                    other_segment,
                    min_corner=min_corner,
                    max_corner=max_corner,
                    extra_bbox_pad=extra_bbox_pad,
                )
                if other_sampled_grid["local_grid"].size == 0 or not bool(np.any(other_sampled_grid["in_patch"])):
                    continue

                other_surface_vox = _voxelize_surface_from_sampled_grid(
                    self,
                    other_segment,
                    min_corner=min_corner,
                    max_corner=max_corner,
                    crop_size=crop_size,
                    sampled_grid=other_sampled_grid,
                )
                if bool(np.any(other_surface_vox > 0.0)):
                    surface_vox = np.maximum(surface_vox, other_surface_vox)

                has_labels_in_crop = bool(
                    np.any(other_sampled_grid["in_patch"] & (other_sampled_grid["class_codes"] == 1))
                )
                if has_labels_in_crop:
                    other_positive_label_vox = _voxelize_positive_labels_from_sampled_grid(
                        self,
                        other_segment,
                        min_corner=min_corner,
                        max_corner=max_corner,
                        crop_size=crop_size,
                        sampled_grid=other_sampled_grid,
                    )

                    other_background_label_vox = _voxelize_background_surface_labels_from_sampled_grid(
                        self,
                        other_segment,
                        min_corner=min_corner,
                        max_corner=max_corner,
                        crop_size=crop_size,
                        sampled_grid=other_sampled_grid,
                    )
                    other_labeled_surface_vox = np.asarray(
                        (other_positive_label_vox > 0.0) | (other_background_label_vox > 0.0),
                        dtype=bool,
                    )
                    self._suppress_projected_mask_overlap_with_foreign_labels(
                        projected_loss_mask,
                        other_labeled_surface_vox,
                        own_labeled_surface_mask=labeled_surface_vox_union,
                    )

                    if bool(np.any(other_background_label_vox > 0.0)):
                        background_label_vox = np.maximum(
                            background_label_vox,
                            other_background_label_vox,
                        )

                    other_projected_loss_mask = _build_projected_loss_mask_volume(
                        self,
                        other_segment,
                        min_corner=min_corner,
                        max_corner=max_corner,
                        crop_size=crop_size,
                        sampled_grid=other_sampled_grid,
                    )
                    self._suppress_projected_mask_overlap_with_foreign_labels(
                        other_projected_loss_mask,
                        labeled_surface_vox_union,
                        own_labeled_surface_mask=other_labeled_surface_vox,
                    )
                    self._merge_projected_loss_mask(
                        projected_loss_mask,
                        other_projected_loss_mask,
                    )
                    if bool(np.any(other_positive_label_vox > 0.0)):
                        positive_label_vox = np.maximum(
                            positive_label_vox,
                            other_positive_label_vox,
                        )
                    labeled_surface_vox_union |= other_labeled_surface_vox
                else:
                    if bool(np.any(other_surface_vox > 0.0)):
                        background_label_vox = np.maximum(
                            background_label_vox,
                            other_surface_vox,
                        )

                    other_bg_tolerance_vox = self._voxelize_full_surface_background_tolerance_from_sampled_grid(
                        other_segment,
                        min_corner=min_corner,
                        max_corner=max_corner,
                        crop_size=crop_size,
                        sampled_grid=other_sampled_grid,
                    )
                    bg_mask = np.asarray(other_bg_tolerance_vox > 0.0, dtype=bool)
                    projected_loss_mask[bg_mask & (projected_loss_mask != 1.0)] = 0.0

        labeled_vox_at_surface = _build_surface_label_volume(
            positive_label_vox=positive_label_vox,
            background_label_vox=background_label_vox,
            crop_size=crop_size,
        )

        if self.augmentations is not None:
            (
                vol_crop,
                labeled_vox_at_surface,
                surface_vox,
                projected_loss_mask,
            ) = self._apply_sample_augmentation(
                vol_crop=vol_crop,
                labeled_vox_at_surface=labeled_vox_at_surface,
                surface_vox=surface_vox,
                projected_loss_mask=projected_loss_mask,
            )

        vol_crop = self._to_float32_tensor(vol_crop)
        labeled_vox_at_surface = self._to_float32_tensor(labeled_vox_at_surface)
        surface_vox = self._to_float32_tensor(surface_vox)
        projected_loss_mask = self._to_float32_tensor(projected_loss_mask)

        return {
            "vol": vol_crop,
            "labeled_vox_at_surface": labeled_vox_at_surface,
            "surface_vox": surface_vox,
            "projected_loss_mask": projected_loss_mask,
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
    parser.add_argument(
        "--napari-downsample",
        type=int,
        default=10,
        help="Spatial downsample factor for arrays shown in Napari.",
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

        napari_downsample = max(1, int(args.napari_downsample))

        def _downsample_spatial_3d(arr, factor):
            if factor <= 1:
                return arr
            if arr.ndim != 3:
                return arr
            return arr[
                ::factor,
                ::factor,
                ::factor,
            ]

        if len(ds) == 0:
            raise RuntimeError("Dataset produced no samples to visualize.")

        from qtpy.QtWidgets import QPushButton

        print(f"Napari spatial downsample factor: {napari_downsample}")

        def _build_napari_sample_data(sample):
            sample_idx = int(sample.get("idx", -1))

            vol_3d = np.asarray(sample["vol"], dtype=np.float32)
            surface_label_raw = np.asarray(sample["labeled_vox_at_surface"]).astype(np.int16, copy=False)
            projected_loss_mask_raw = np.asarray(sample["projected_loss_mask"]).astype(np.int16, copy=False)
            positive_3d = (surface_label_raw == 1).astype(np.uint8)
            background_3d = (surface_label_raw == 0).astype(np.uint8)
            surface_3d = (np.asarray(sample["surface_vox"]) > 0.0).astype(np.uint8)

            # For visualization, map ignore=2 to 0 (transparent) so only labeled classes remain visible.
            surface_label_vis = np.zeros_like(surface_label_raw, dtype=np.uint8)
            surface_label_vis[surface_label_raw == 1] = 1
            surface_label_vis[surface_label_raw == 0] = 2

            projected_loss_mask_vis = np.zeros_like(projected_loss_mask_raw, dtype=np.uint8)
            projected_loss_mask_vis[projected_loss_mask_raw == 1] = 1
            projected_loss_mask_vis[projected_loss_mask_raw == 0] = 2

            return {
                "sample_idx": sample_idx,
                "positive_total": int(np.count_nonzero(surface_label_raw == 1)),
                "background_total": int(np.count_nonzero(surface_label_raw == 0)),
                "vol_3d": _downsample_spatial_3d(vol_3d, napari_downsample),
                "surface_3d": _downsample_spatial_3d(surface_3d, napari_downsample),
                "positive_3d": _downsample_spatial_3d(positive_3d, napari_downsample),
                "background_3d": _downsample_spatial_3d(background_3d, napari_downsample),
                "surface_label_vis": _downsample_spatial_3d(surface_label_vis, napari_downsample),
                "projected_loss_mask_vis": _downsample_spatial_3d(projected_loss_mask_vis, napari_downsample),
            }

        def _log_sample_stats(sample_data):
            print(f"napari sample idx: {sample_data['sample_idx']}")
            print(f"positive_label_vox sample nonzero: {sample_data['positive_total']}")
            print(f"background_label_vox sample nonzero: {sample_data['background_total']}")

        sample_cursor = {"idx": 0}
        initial_sample_data = _build_napari_sample_data(ds[sample_cursor["idx"]])
        _log_sample_stats(initial_sample_data)

        viewer = napari.Viewer(ndisplay=3)
        vol_layer = viewer.add_image(
            initial_sample_data["vol_3d"],
            name="vol",
            rendering="mip",
            interpolation3d="nearest",
        )
        surface_layer = viewer.add_labels(
            initial_sample_data["surface_3d"],
            name="surface_vox",
            opacity=0.2,
            blending="additive",
        )
        positive_layer = viewer.add_labels(
            initial_sample_data["positive_3d"],
            name="positive_label_vox",
            opacity=0.9,
            blending="additive",
        )
        background_layer = viewer.add_labels(
            initial_sample_data["background_3d"],
            name="background_label_vox",
            opacity=0.7,
            blending="additive",
        )
        surface_label_layer = viewer.add_labels(
            initial_sample_data["surface_label_vis"],
            name="labeled_vox_at_surface",
            opacity=0.5,
            blending="additive",
        )
        projected_loss_layer = viewer.add_labels(
            initial_sample_data["projected_loss_mask_vis"],
            name="projected_loss_mask",
            opacity=0.5,
            blending="additive",
        )
        viewer.title = f"TifxyzInkDataset sample {initial_sample_data['sample_idx']}"

        def _show_sample_at_cursor():
            sample_data = _build_napari_sample_data(ds[sample_cursor["idx"]])
            vol_layer.data = sample_data["vol_3d"]
            surface_layer.data = sample_data["surface_3d"]
            positive_layer.data = sample_data["positive_3d"]
            background_layer.data = sample_data["background_3d"]
            surface_label_layer.data = sample_data["surface_label_vis"]
            projected_loss_layer.data = sample_data["projected_loss_mask_vis"]
            viewer.title = f"TifxyzInkDataset sample {sample_data['sample_idx']}"
            _log_sample_stats(sample_data)

        next_button = QPushButton("next")

        def _on_next_clicked():
            sample_cursor["idx"] = (sample_cursor["idx"] + 1) % len(ds)
            _show_sample_at_cursor()

        next_button.clicked.connect(_on_next_clicked)
        viewer.window.add_dock_widget(next_button, area="right", name="next")

        napari.run()
