import os
import numpy as np
import warnings

from torch.utils.data import Dataset

from common import (
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
        self._segment_ink_label_path_by_uuid = {}
        for patch in self.patches:
            segment_uuid = str(patch.get("segment_uuid", ""))
            ink_label_path = patch.get("ink_label_path")
            if not ink_label_path:
                warnings.warn(
                    f"Unable to load ink labels for segment: {segment_uuid}"
                )
                continue
            self._segment_ink_label_path_by_uuid[segment_uuid] = str(ink_label_path)

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
        surface_label_vox = _build_surface_label_volume(
            positive_label_vox=positive_label_vox,
            background_label_vox=background_label_vox,
            crop_size=crop_size,
        )
        mil_loss_mask_vox = _build_projected_loss_mask_volume(
            self,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
            sampled_grid=sampled_grid,
        )

        return {
            "vol": vol_crop,
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
            "surface_label_vox",
            "surface_vox",
            "mil_loss_mask_vox",
        )
        stacked_outputs = {k: [] for k in output_keys}
        sample_positive_counts = []
        sample_background_counts = []

        for sample in ds:
            for key in output_keys:
                stacked_outputs[key].append(np.asarray(sample[key]))
            surface_label = np.asarray(sample["surface_label_vox"])
            sample_positive_counts.append(int(np.count_nonzero(surface_label == 1.0)))
            sample_background_counts.append(int(np.count_nonzero(surface_label == 0.0)))

        if not stacked_outputs["vol"]:
            raise RuntimeError("Dataset produced no samples to visualize.")

        positive_total = int(np.sum(np.asarray(sample_positive_counts, dtype=np.int64)))
        background_total = int(np.sum(np.asarray(sample_background_counts, dtype=np.int64)))
        print(f"positive_label_vox total nonzero: {positive_total}")
        print(f"background_label_vox total nonzero: {background_total}")
        if positive_total == 0:
            print("warning: positive_label_vox is empty across all samples.")
        if background_total == 0:
            print("warning: background_label_vox is empty across all samples.")

        focus_patch_idx = 0
        combined_counts = np.asarray(sample_positive_counts, dtype=np.int64) + np.asarray(
            sample_background_counts,
            dtype=np.int64,
        )
        if combined_counts.size > 0 and int(np.max(combined_counts)) > 0:
            focus_patch_idx = int(np.argmax(combined_counts))

        vol_4d = np.stack(stacked_outputs["vol"], axis=0)
        surface_label_raw = np.stack(stacked_outputs["surface_label_vox"], axis=0).astype(np.int16, copy=False)
        mil_label_raw = np.stack(stacked_outputs["mil_loss_mask_vox"], axis=0).astype(np.int16, copy=False)
        positive_4d = (surface_label_raw == 1).astype(np.uint8)
        background_4d = (surface_label_raw == 0).astype(np.uint8)
        surface_4d = (np.stack(stacked_outputs["surface_vox"], axis=0) > 0.0).astype(np.uint8)

        # For visualization, map ignore=2 to 0 (transparent) so only labeled classes remain visible.
        surface_label_vis = np.zeros_like(surface_label_raw, dtype=np.uint8)
        surface_label_vis[surface_label_raw == 1] = 1
        surface_label_vis[surface_label_raw == 0] = 2

        mil_label_vis = np.zeros_like(mil_label_raw, dtype=np.uint8)
        mil_label_vis[mil_label_raw == 1] = 1
        mil_label_vis[mil_label_raw == 0] = 2

        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(vol_4d, name="vol", rendering="mip", interpolation3d="nearest")
        viewer.add_labels(
            surface_4d,
            name="surface_vox",
            opacity=0.2,
            blending="additive",
        )
        viewer.add_labels(
            positive_4d,
            name="positive_label_vox",
            opacity=0.9,
            blending="additive",
        )
        viewer.add_labels(
            background_4d,
            name="background_label_vox",
            opacity=0.7,
            blending="additive",
        )
        viewer.add_labels(
            surface_label_vis,
            name="surface_label_vox",
            opacity=0.5,
            blending="additive",
        )
        viewer.add_labels(
            mil_label_vis,
            name="mil_loss_mask_vox",
            opacity=0.5,
            blending="additive",
        )
        viewer.dims.set_current_step(0, focus_patch_idx)
        print(f"Napari focus patch index: {focus_patch_idx}")

        napari.run()
