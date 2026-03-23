import os
import cc3d
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from torch.utils.data import Dataset

from common import (
    _normalize_patch_size_zyx,
    _normalize_vectors_last_axis,
    _points_to_voxels,
    _project_label_from_sampled_grid,
    _read_volume_crop_from_patch_dict,
    _sample_patch_supervision_grid,
)
from .patch_finding import find_patches
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid_masked

class TifxyzInkDataset(Dataset):
    def __init__(
        self,
        config,
        apply_augmentation: bool = True,
        apply_perturbation: bool = True,
        wrap_mode: str | None = None,
        patches=None,
        patch_generation_stats=None,
    ):
        self.apply_augmentation = apply_augmentation
        self.apply_perturbation = bool(apply_perturbation)
        self.mode = str(config.get("mode", "default")).strip().lower()
        self.use_normal_pooled_3d = self.mode == "normal_pooled_3d"
        self.patch_size = config["patch_size"]                                          # 3d vol crop / model input patch size
        self.patch_size_zyx = _normalize_patch_size_zyx(self.patch_size)  
        config_wrap_mode = config.get("wrap_mode", "single")
        if wrap_mode is None:
            wrap_mode = config_wrap_mode
        self.wrap_mode = str(wrap_mode).strip().lower()
        if self.wrap_mode not in {"single", "all"}:
            raise ValueError(
                f"wrap_mode must be 'single' or 'all', got {wrap_mode!r}"
            )
        if self.use_normal_pooled_3d and self.wrap_mode != "single":
            raise ValueError("mode='normal_pooled_3d' currently requires wrap_mode='single'")
        
        fg_distance_value = config.get("fg_distance", config.get("label_distance", 10))
        if fg_distance_value is None:
            raise ValueError("fg_distance must not be None")
        self.fg_distance = (
            (float(fg_distance_value), float(fg_distance_value))
            if np.isscalar(fg_distance_value)
            else tuple(float(v) for v in fg_distance_value)
        )
        if len(self.fg_distance) != 2:
            raise ValueError(f"fg_distance must be a number or a length-2 sequence, got {fg_distance_value!r}")

        bg_distance_value = config.get("bg_distance", self.fg_distance)
        if bg_distance_value is None:
            raise ValueError("bg_distance must not be None")
        self.bg_distance = (
            (float(bg_distance_value), float(bg_distance_value))
            if np.isscalar(bg_distance_value)
            else tuple(float(v) for v in bg_distance_value)
        )
        if len(self.bg_distance) != 2:
            raise ValueError(f"bg_distance must be a number or a length-2 sequence, got {bg_distance_value!r}")

        self.normal_sample_step = float(config.get("normal_sample_step", 0.5))
        self.surface_bbox_pad = float(config.get("surface_bbox_pad", 2.0))
        self.surface_distance_clip = float(config.get("surface_distance_clip", 10.0))
        if self.surface_distance_clip <= 0.0:
            raise ValueError(
                f"surface_distance_clip must be > 0, got {self.surface_distance_clip}"
            )
        self.input_channels = 2 if self.use_normal_pooled_3d else 1
      
        self.overlap_fraction = float(config.get("overlap_fraction", 0.25))             # amount of overlap (stride) in train/val patches, as a percentage of the patch size
        self.patch_finding_workers = int(
            config.get("patch_finding_workers", 4)
        )                                                                               # reserved for patch-finding parallelism
        self.patch_cache_force_recompute = bool(
            config.get("patch_cache_force_recompute", False)
        )
        patch_cache_filename = config.get("patch_cache_filename")
        if patch_cache_filename in (None, ""):
            self.patch_cache_filename = os.path.join(
                str(config.get("out_dir", ".")),
                ".tifxyz_patch_cache.json",
            )
        else:
            self.patch_cache_filename = str(patch_cache_filename)

        self._segment_grid_cache = {}
        self._segment_labels_and_mask_cache = {}
        self._segment_normal_cache = {}
        self.surface_target_size_yx = (
            int(self.patch_size_zyx[1]),
            int(self.patch_size_zyx[2]),
        )

        if apply_augmentation:                                                          # we'll use the vesuvius augmentation pipeline , see vesuvius/src/vesuvius/models/augmentation/pipelines/training_transforms.py
            self.augmentations = create_training_transforms(                            # for current defaults 
                patch_size=tuple(int(v) for v in self.patch_size_zyx),                  # TODO: make these configurable
            )
        else:
            self.augmentations = None

        if patches is None:
            self.patches, self.patch_generation_stats = find_patches(                   # compute shared dataset bboxes, then keep only bboxes with tifxyz points and positive supervision
                config,                                                                 
                patch_size_zyx=self.patch_size_zyx,                                     # - patch size (in 3d)       
                overlap_fraction=self.overlap_fraction,                                 # - 3d bbox overlap
                patch_finding_workers=self.patch_finding_workers,
                patch_cache_force_recompute=self.patch_cache_force_recompute,
                patch_cache_filename=self.patch_cache_filename,                         # for info on the bbox generation
            )
        else:
            self.patches = patches
            self.patch_generation_stats = (
                {} if patch_generation_stats is None else patch_generation_stats
            )

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

    def _apply_normal_pooled_augmentation(self, vol_crop, sampled_grid):
        image = torch.as_tensor(
            np.asarray(vol_crop, dtype=np.float32),
            dtype=torch.float32,
        ).unsqueeze(0)
        local_grid = np.asarray(sampled_grid["local_grid"], dtype=np.float32)
        normals_zyx = np.asarray(sampled_grid["normals_zyx"], dtype=np.float32)

        augmented = self.augmentations(
            image=image,
            keypoints=torch.as_tensor(
                local_grid.reshape(-1, local_grid.shape[-1]),
                dtype=torch.float32,
            ),
            surface_normals=torch.as_tensor(
                normals_zyx.reshape(-1, normals_zyx.shape[-1]),
                dtype=torch.float32,
            ),
            vector_keys=("surface_normals",),
            crop_shape=tuple(int(v) for v in self.patch_size_zyx),
        )

        augmented_grid = dict(sampled_grid)
        augmented_grid["local_grid"] = (
            augmented["keypoints"]
            .reshape(local_grid.shape)
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
        augmented_grid["normals_zyx"] = (
            augmented["surface_normals"]
            .reshape(normals_zyx.shape)
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
        return augmented["image"][0].to(dtype=torch.float32).cpu(), augmented_grid

    @staticmethod
    def _resize_surface_array(array, size_yx, *, mode):
        array_np = np.asarray(array, dtype=np.float32)
        tensor = torch.as_tensor(array_np, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(
                f"Expected 2D or 3D array for surface resize, got shape {tuple(array_np.shape)}"
            )

        kwargs = {}
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        resized = F.interpolate(tensor, size=size_yx, mode=mode, **kwargs)[0]
        if array_np.ndim == 2:
            return resized[0].cpu().numpy()
        return resized.permute(1, 2, 0).cpu().numpy()

    def _build_normal_pooled_surface_sample(self, sampled_grid):
        surface_points_zyx = self._resize_surface_array(
            sampled_grid["local_grid"],
            self.surface_target_size_yx,
            mode="bilinear",
        )
        surface_normals_zyx = self._resize_surface_array(
            sampled_grid["normals_zyx"],
            self.surface_target_size_yx,
            mode="bilinear",
        )
        surface_normals_zyx, normals_nonzero = _normalize_vectors_last_axis(
            surface_normals_zyx
        )

        surface_targets_2d = self._resize_surface_array(
            (sampled_grid["class_codes"] == 1).astype(np.float32, copy=False),
            self.surface_target_size_yx,
            mode="nearest",
        )
        surface_valid_2d = self._resize_surface_array(
            (
                sampled_grid["in_patch"]
                & sampled_grid["valid_interp"]
                & sampled_grid["normals_valid"]
                & (sampled_grid["class_codes"] != 100)
            ).astype(np.float32, copy=False),
            self.surface_target_size_yx,
            mode="nearest",
        )

        surface_targets_2d = (surface_targets_2d > 0.5).astype(np.float32, copy=False)
        surface_valid_2d = (surface_valid_2d > 0.5)
        surface_valid_2d &= normals_nonzero
        assert bool(np.any(surface_valid_2d)), (
            "normal_pooled_3d supervision must contain at least one valid surface sample"
        )
        assert bool(np.any(surface_targets_2d[surface_valid_2d] > 0.0)), (
            "normal_pooled_3d supervision must contain at least one positive surface label"
        )

        surface_points_zyx[~surface_valid_2d] = 0.0
        surface_normals_zyx[~surface_valid_2d] = 0.0

        return {
            "surface_points_zyx": self._to_float32_tensor(surface_points_zyx),
            "surface_normals_zyx": self._to_float32_tensor(surface_normals_zyx),
            "surface_targets_2d": self._to_float32_tensor(surface_targets_2d),
            "surface_valid_2d": self._to_float32_tensor(surface_valid_2d.astype(np.float32)),
        }

    @staticmethod
    def _compute_surface_point_mask(local_grid, crop_size, base_mask):
        local_grid = np.asarray(local_grid, dtype=np.float32)
        surface_point_mask = np.asarray(base_mask, dtype=bool).copy()
        surface_point_mask &= np.isfinite(local_grid).all(axis=-1)
        for axis, axis_size in enumerate(tuple(int(v) for v in crop_size)):
            coords = local_grid[..., axis]
            surface_point_mask &= coords >= 0.0
            surface_point_mask &= coords < float(axis_size)
        return surface_point_mask

    def _build_surface_voxels_and_labels(self, sampled_grid, crop_size):
        local_grid = np.asarray(sampled_grid["local_grid"], dtype=np.float32)
        crop_size = tuple(int(v) for v in crop_size)
        surface_point_mask = self._compute_surface_point_mask(
            local_grid,
            crop_size,
            sampled_grid["in_patch"],
        )
        assert local_grid.size > 0 and bool(np.any(surface_point_mask)), (
            "sampled_grid must contain in-patch surface points"
        )

        positive_point_mask = surface_point_mask & (sampled_grid["class_codes"] == 1)
        assert bool(np.any(positive_point_mask)), (
            "sampled_grid must contain labeled points inside the patch"
        )
        positive_label_vox = _points_to_voxels(
            local_grid[positive_point_mask],
            crop_size,
        )

        background_point_mask = surface_point_mask & (sampled_grid["class_codes"] == 0)
        background_label_vox = _points_to_voxels(
            local_grid[background_point_mask],
            crop_size,
        )

        surface_vox = voxelize_surface_grid_masked(
            local_grid,
            crop_size,
            surface_point_mask,
        ).astype(np.float32, copy=False)

        labeled_vox_at_surface = np.full(crop_size, 2.0, dtype=np.float32)
        labeled_vox_at_surface[background_label_vox > 0.0] = 0.0
        labeled_vox_at_surface[positive_label_vox > 0.0] = 1.0
        surface_component_labels = cc3d.connected_components(
            (surface_vox > 0.0).astype(np.uint8),
            connectivity=26,
        )
        keep_component_ids = np.unique(
            surface_component_labels[
                (labeled_vox_at_surface != 2.0) & (surface_vox > 0.0)
            ]
        )
        keep_component_ids = keep_component_ids[keep_component_ids != 0]
        surface_vox = np.isin(
            surface_component_labels,
            keep_component_ids,
        ).astype(np.float32, copy=False)

        return labeled_vox_at_surface, surface_vox

    def _build_surface_distance_channel(self, surface_vox):
        surface_mask = np.asarray(surface_vox, dtype=np.float32) > 0.0
        if not bool(np.any(surface_mask)):
            return np.zeros(surface_mask.shape, dtype=np.float32)

        distance = distance_transform_edt(~surface_mask).astype(np.float32, copy=False)
        distance = np.minimum(distance, float(self.surface_distance_clip))
        distance /= float(self.surface_distance_clip)
        return (1.0 - distance).astype(np.float32, copy=False)

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

    def _select_patch_segment_indices(self, patch):
        supervised_segment_indices = tuple(
            int(v) for v in patch["supervised_segment_indices"]
        )
        assert supervised_segment_indices, "patch must contain at least one supervised segment"
        if self.wrap_mode == "all":
            return supervised_segment_indices
        chosen_segment_idx = int(
            supervised_segment_indices[np.random.randint(len(supervised_segment_indices))]
        )
        return (chosen_segment_idx,)

    def _build_patch_segment_sample(
        self,
        *,
        patch,
        patch_segment,
        crop_size,
        min_corner,
        max_corner,
        vol_crop,
    ):
        segment = patch_segment["segment"]

        sampled_grid = _sample_patch_supervision_grid(
            self,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            extra_bbox_pad=max(max(self.bg_distance), max(self.fg_distance)) + 1.0,
            stored_rowcol_bounds=patch_segment["stored_rowcol_bounds"],
        )

        if self.use_normal_pooled_3d:
            if self.augmentations is not None:
                vol_crop, sampled_grid = self._apply_normal_pooled_augmentation(
                    vol_crop,
                    sampled_grid,
                )
            _, surface_vox = self._build_surface_voxels_and_labels(
                sampled_grid,
                crop_size,
            )
            surface_distance = self._build_surface_distance_channel(surface_vox)
            vol_input = np.stack(
                (
                    np.asarray(vol_crop, dtype=np.float32),
                    surface_distance,
                ),
                axis=0,
            )
            out = {
                "vol": self._to_float32_tensor(vol_input),
                "patch_segment": patch_segment,
            }
            out.update(self._build_normal_pooled_surface_sample(sampled_grid))
            return out

        labeled_vox_at_surface, surface_vox = self._build_surface_voxels_and_labels(
            sampled_grid,
            crop_size,
        )

        projected_loss_mask = np.full(crop_size, 2.0, dtype=np.float32)
        background_projection_vox = _project_label_from_sampled_grid(
            self,
            sampled_grid,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
            class_value=0,
            label_distance=self.bg_distance,
        )
        projected_loss_mask[background_projection_vox > 0.0] = 0.0
        label_projection_vox = _project_label_from_sampled_grid(
            self,
            sampled_grid,
            min_corner=min_corner,
            max_corner=max_corner,
            crop_size=crop_size,
            class_value=1,
            label_distance=self.fg_distance,
            require_points=True,
        )
        assert bool(np.any(label_projection_vox > 0.0)), "label projection must produce non-empty supervision"
        projected_loss_mask[label_projection_vox > 0.0] = 1.0

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

        return {
            "vol": self._to_float32_tensor(vol_crop),
            "labeled_vox_at_surface": self._to_float32_tensor(labeled_vox_at_surface),
            "surface_vox": self._to_float32_tensor(surface_vox),
            "projected_loss_mask": self._to_float32_tensor(projected_loss_mask),
            "patch_segment": patch_segment,
        }

    def __getitem__(self, idx):
        patch = self.patches[idx]
        selected_segment_indices = self._select_patch_segment_indices(patch)
        selected_patch_segments = [
            patch["segments"][segment_idx] for segment_idx in selected_segment_indices
        ]

        z0, z1, y0, y1, x0, x1 = patch['world_bbox']
        min_corner = np.array([z0, y0, x0], dtype=np.int32)
        max_corner = np.array([z1 + 1, y1 + 1, x1 + 1], dtype=np.int32)
        crop_size = tuple(int(v) for v in self.patch_size_zyx)

        # robust normalization is applied within this function; avoid stacking a second intensity normalization on top.
        vol_crop = _read_volume_crop_from_patch_dict(
            patch,
            crop_size=crop_size,
            min_corner=min_corner,
            max_corner=max_corner,
        )

        segment_samples = [
            self._build_patch_segment_sample(
                patch=patch,
                patch_segment=patch_segment,
                crop_size=crop_size,
                min_corner=min_corner,
                max_corner=max_corner,
                vol_crop=vol_crop,
            )
            for patch_segment in selected_patch_segments
        ]

        if self.wrap_mode == "single":
            sample = segment_samples[0]
            sample.update(
                {
                    "patch": patch,
                    "patch_segments": selected_patch_segments,
                    "selected_segment_indices": selected_segment_indices,
                    "wrap_mode": self.wrap_mode,
                    "idx": int(idx),
                }
            )
            return sample

        return {
            "vol": torch.stack([sample["vol"] for sample in segment_samples], dim=0),
            "labeled_vox_at_surface": torch.stack(
                [sample["labeled_vox_at_surface"] for sample in segment_samples],
                dim=0,
            ),
            "surface_vox": torch.stack(
                [sample["surface_vox"] for sample in segment_samples],
                dim=0,
            ),
            "projected_loss_mask": torch.stack(
                [sample["projected_loss_mask"] for sample in segment_samples],
                dim=0,
            ),
            "patch": patch,
            "patch_segment": None,
            "patch_segments": selected_patch_segments,
            "selected_segment_indices": selected_segment_indices,
            "wrap_mode": self.wrap_mode,
            "idx": int(idx),
        }

if __name__ == "__main__":
    import argparse
    import json

    default_config_path = os.path.join(
        os.path.dirname(__file__),
        "example_config.json",
    )
    parser = argparse.ArgumentParser(description="Inspect the TifxyzInkDataset.")
    parser.add_argument(
        "--config",
        default=default_config_path,
        help="Path to the dataset config JSON.",
    )
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
    parser.add_argument(
        "--wrap-mode",
        choices=("single", "all"),
        default=None,
        help="Override the config wrap_mode for dataset inspection.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    ds = TifxyzInkDataset(
        config,
        apply_augmentation=False,
        apply_perturbation=False,
        wrap_mode=args.wrap_mode,
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

        from napari.qt.threading import thread_worker
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

        print(f"Napari spatial downsample factor: {napari_downsample}")

        def _ensure_wrap_axis(arr):
            arr = np.asarray(arr)
            if arr.ndim == 3:
                return arr[np.newaxis, ...]
            if arr.ndim == 4:
                return arr
            raise ValueError(f"Expected a 3D or 4D array for visualization, got shape {arr.shape!r}")

        def _build_napari_sample_data(sample):
            sample_idx = int(sample.get("idx", -1))
            wrap_mode = str(sample.get("wrap_mode", "single"))
            vol_4d = _ensure_wrap_axis(np.asarray(sample["vol"], dtype=np.float32))
            surface_label_4d = _ensure_wrap_axis(
                np.asarray(sample["labeled_vox_at_surface"]).astype(np.int16, copy=False)
            )
            projected_loss_mask_4d = _ensure_wrap_axis(
                np.asarray(sample["projected_loss_mask"]).astype(np.int16, copy=False)
            )
            surface_vox_4d = _ensure_wrap_axis(np.asarray(sample["surface_vox"], dtype=np.float32))

            wrap_samples = []
            positive_totals = []
            background_totals = []

            for wrap_idx in range(int(vol_4d.shape[0])):
                vol_3d = vol_4d[wrap_idx]
                surface_label_raw = surface_label_4d[wrap_idx]
                projected_loss_mask_raw = projected_loss_mask_4d[wrap_idx]
                positive_3d = (surface_label_raw == 1).astype(np.uint8)
                background_3d = (surface_label_raw == 0).astype(np.uint8)
                surface_3d = (surface_vox_4d[wrap_idx] > 0.0).astype(np.uint8)

                # For visualization, map ignore=2 to 0 (transparent) so only labeled classes remain visible.
                surface_label_vis = np.zeros_like(surface_label_raw, dtype=np.uint8)
                surface_label_vis[surface_label_raw == 1] = 1
                surface_label_vis[surface_label_raw == 0] = 2

                projected_loss_mask_vis = np.zeros_like(projected_loss_mask_raw, dtype=np.uint8)
                projected_loss_mask_vis[projected_loss_mask_raw == 1] = 1
                projected_loss_mask_vis[projected_loss_mask_raw == 0] = 2

                positive_totals.append(int(np.count_nonzero(surface_label_raw == 1)))
                background_totals.append(int(np.count_nonzero(surface_label_raw == 0)))
                wrap_samples.append(
                    {
                        "vol_3d": _downsample_spatial_3d(vol_3d, napari_downsample),
                        "surface_3d": _downsample_spatial_3d(surface_3d, napari_downsample),
                        "positive_3d": _downsample_spatial_3d(positive_3d, napari_downsample),
                        "background_3d": _downsample_spatial_3d(background_3d, napari_downsample),
                        "surface_label_vis": _downsample_spatial_3d(surface_label_vis, napari_downsample),
                        "projected_loss_mask_vis": _downsample_spatial_3d(projected_loss_mask_vis, napari_downsample),
                    }
                )

            return {
                "sample_idx": sample_idx,
                "wrap_mode": wrap_mode,
                "wrap_count": int(len(wrap_samples)),
                "positive_total": int(sum(positive_totals)),
                "background_total": int(sum(background_totals)),
                "positive_totals": positive_totals,
                "background_totals": background_totals,
                "wrap_samples": wrap_samples,
            }

        def _log_sample_stats(sample_data):
            print(f"napari sample idx: {sample_data['sample_idx']}")
            print(f"wrap mode: {sample_data['wrap_mode']} (visualizing all {sample_data['wrap_count']})")
            print(f"positive_label_vox sample nonzero: {sample_data['positive_total']} per-wrap={sample_data['positive_totals']}")
            print(f"background_label_vox sample nonzero: {sample_data['background_total']} per-wrap={sample_data['background_totals']}")

        sample_cursor = {"idx": 0}
        load_state = {"request_id": 0, "worker": None}
        multi_wrap_indices = tuple(
            idx
            for idx, patch in enumerate(ds.patches)
            if len(tuple(int(v) for v in patch["supervised_segment_indices"])) > 1
        )
        empty_image = np.zeros((1, 1, 1), dtype=np.float32)
        empty_labels = np.zeros((1, 1, 1), dtype=np.uint8)

        viewer = napari.Viewer(ndisplay=3)
        wrap_layers = []

        def _create_wrap_layers(wrap_idx):
            wrap_num = int(wrap_idx) + 1
            return {
                "vol": viewer.add_image(
                    empty_image,
                    name=f"vol wrap {wrap_num}",
                    rendering="mip",
                    interpolation3d="nearest",
                ),
                "surface": viewer.add_labels(
                    empty_labels,
                    name=f"surface_vox wrap {wrap_num}",
                    opacity=0.2,
                    blending="additive",
                ),
                "positive": viewer.add_labels(
                    empty_labels,
                    name=f"positive_label_vox wrap {wrap_num}",
                    opacity=0.9,
                    blending="additive",
                ),
                "background": viewer.add_labels(
                    empty_labels,
                    name=f"background_label_vox wrap {wrap_num}",
                    opacity=0.7,
                    blending="additive",
                ),
                "surface_label": viewer.add_labels(
                    empty_labels,
                    name=f"labeled_vox_at_surface wrap {wrap_num}",
                    opacity=0.5,
                    blending="additive",
                ),
                "projected_loss": viewer.add_labels(
                    empty_labels,
                    name=f"projected_loss_mask wrap {wrap_num}",
                    opacity=0.5,
                    blending="additive",
                ),
            }

        def _ensure_wrap_layers(wrap_count):
            while len(wrap_layers) < int(wrap_count):
                wrap_layers.append(_create_wrap_layers(len(wrap_layers)))

        viewer.title = "TifxyzInkDataset loading..."

        def _apply_sample_data(sample_data, request_id):
            if request_id != load_state["request_id"]:
                return
            wrap_samples = sample_data["wrap_samples"]
            _ensure_wrap_layers(len(wrap_samples))
            for wrap_idx, wrap_sample in enumerate(wrap_samples):
                layer_group = wrap_layers[wrap_idx]
                layer_group["vol"].data = wrap_sample["vol_3d"]
                layer_group["surface"].data = wrap_sample["surface_3d"]
                layer_group["positive"].data = wrap_sample["positive_3d"]
                layer_group["background"].data = wrap_sample["background_3d"]
                layer_group["surface_label"].data = wrap_sample["surface_label_vis"]
                layer_group["projected_loss"].data = wrap_sample["projected_loss_mask_vis"]
                for layer in layer_group.values():
                    layer.visible = True

            for wrap_idx in range(len(wrap_samples), len(wrap_layers)):
                layer_group = wrap_layers[wrap_idx]
                layer_group["vol"].data = empty_image
                layer_group["surface"].data = empty_labels
                layer_group["positive"].data = empty_labels
                layer_group["background"].data = empty_labels
                layer_group["surface_label"].data = empty_labels
                layer_group["projected_loss"].data = empty_labels
                for layer in layer_group.values():
                    layer.visible = False

            viewer.title = f"TifxyzInkDataset sample {sample_data['sample_idx']}"
            status_label.setText(
                f"sample {sample_data['sample_idx']} loaded "
                f"(wraps={sample_data['wrap_count']}, pos={sample_data['positive_total']}, "
                f"bg={sample_data['background_total']})"
            )
            next_button.setEnabled(True)
            next_multi_wrap_button.setEnabled(bool(multi_wrap_indices))
            _log_sample_stats(sample_data)

        def _handle_sample_error(exc, request_id):
            if request_id != load_state["request_id"]:
                return
            viewer.title = "TifxyzInkDataset load failed"
            status_label.setText(f"load failed: {exc}")
            next_button.setEnabled(True)
            next_multi_wrap_button.setEnabled(bool(multi_wrap_indices))
            print(f"napari sample load failed: {exc}")

        def _request_sample_load(idx):
            request_id = int(load_state["request_id"]) + 1
            load_state["request_id"] = request_id
            sample_cursor["idx"] = int(idx)
            viewer.title = f"TifxyzInkDataset loading sample {sample_cursor['idx']}..."
            status_label.setText(f"loading sample {sample_cursor['idx']}...")
            next_button.setEnabled(False)
            next_multi_wrap_button.setEnabled(False)

            @thread_worker
            def _load_sample():
                return _build_napari_sample_data(ds[sample_cursor["idx"]])

            worker = _load_sample()
            load_state["worker"] = worker
            worker.returned.connect(
                lambda sample_data, request_id=request_id: _apply_sample_data(sample_data, request_id)
            )
            worker.errored.connect(
                lambda exc, request_id=request_id: _handle_sample_error(exc, request_id)
            )
            worker.start()

        def _find_next_multi_wrap_idx(start_idx):
            if not multi_wrap_indices:
                return None
            dataset_len = len(ds)
            for offset in range(1, dataset_len + 1):
                candidate_idx = (int(start_idx) + offset) % dataset_len
                if candidate_idx in multi_wrap_indices:
                    return candidate_idx
            return None

        next_button = QPushButton("next")
        next_button.setEnabled(False)
        next_multi_wrap_button = QPushButton("next multi-wrap")
        next_multi_wrap_button.setEnabled(False)
        status_label = QLabel("waiting for initial sample...")
        controls = QWidget()
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(status_label)
        controls_layout.addWidget(next_button)
        controls_layout.addWidget(next_multi_wrap_button)
        controls.setLayout(controls_layout)

        def _on_next_clicked():
            _request_sample_load((sample_cursor["idx"] + 1) % len(ds))

        def _on_next_multi_wrap_clicked():
            next_multi_wrap_idx = _find_next_multi_wrap_idx(sample_cursor["idx"])
            if next_multi_wrap_idx is None:
                status_label.setText("no multi-wrap samples found")
                return
            _request_sample_load(next_multi_wrap_idx)

        next_button.clicked.connect(_on_next_clicked)
        next_multi_wrap_button.clicked.connect(_on_next_multi_wrap_clicked)
        viewer.window.add_dock_widget(controls, area="right", name="sample")

        QTimer.singleShot(0, lambda: _request_sample_load(sample_cursor["idx"]))

        napari.run()
