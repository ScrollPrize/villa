import os
import warnings
import cc3d
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from torch.utils.data import Dataset

from koine_machines.common.common import (
    _normalize_patch_size_zyx,
    _normalize_vectors_last_axis,
    _points_to_voxels,
    _project_label_from_sampled_grid,
    _read_volume_crop_from_patch_dict,
    _sample_patch_supervision_grid,
)
from koine_machines.data.patch_finding import find_segment_patches
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
        self.label_version = config.get("label_version")
        self.mode = str(config.get("mode", "default")).strip().lower()
        self.use_normal_pooled_3d = self.mode == "normal_pooled_3d"
        self.patch_size = config["patch_size"]                                          # 3d vol crop / model input patch size
        self.patch_size_zyx = _normalize_patch_size_zyx(self.patch_size)  
        config_wrap_mode = config.get("wrap_mode", "single")
        if wrap_mode is None:
            wrap_mode = config_wrap_mode
        self.wrap_mode = str(wrap_mode).strip().lower()
        if self.use_normal_pooled_3d and self.wrap_mode != "single":
            raise ValueError("mode='normal_pooled_3d' currently requires wrap_mode='single'")
        
        fg_distance_value = config.get("fg_distance", config.get("label_distance", 10))
        self.fg_distance = (
            (float(fg_distance_value), float(fg_distance_value))
            if np.isscalar(fg_distance_value)
            else tuple(float(v) for v in fg_distance_value)
        )
        bg_distance_value = config.get("bg_distance", self.fg_distance)
        self.bg_distance = (
            (float(bg_distance_value), float(bg_distance_value))
            if np.isscalar(bg_distance_value)
            else tuple(float(v) for v in bg_distance_value)
        )
        self.normal_sample_step = float(config.get("normal_sample_step", 0.5))
        self.surface_bbox_pad = float(config.get("surface_bbox_pad", 2.0))
        self.surface_distance_clip = float(config.get("surface_distance_clip", 10.0))
        self.input_channels = 2 if self.use_normal_pooled_3d else 1
        self.overlap_fraction = float(config.get("overlap_fraction", 0.25))             # amount of overlap (stride) in train/val patches, as a percentage of the patch size
        self.patch_finding_workers = int(config.get("patch_finding_workers", 4))                                                                               # reserved for patch-finding parallelism
        self.patch_cache_force_recompute = bool(config.get("patch_cache_force_recompute", False))
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

        # Recompute in_patch from augmented local_grid so downstream
        # geometry_valid_src does not include points that moved outside
        # the crop, which would corrupt the weighted resize and cause
        # surface-target / surface-valid misalignment.
        aug_local = augmented_grid["local_grid"]
        crop_size_tuple = tuple(
            int(v) for v in sampled_grid.get("crop_size", self.patch_size_zyx)
        )
        in_bounds = np.isfinite(aug_local).all(axis=-1)
        for axis in range(3):
            in_bounds &= aug_local[..., axis] >= 0.0
            in_bounds &= aug_local[..., axis] < float(crop_size_tuple[axis])
        augmented_grid["in_patch"] = sampled_grid["valid_interp"] & in_bounds

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

    @staticmethod
    def _resize_masked_surface_array(array, valid_mask, size_yx, *, mode, eps=1e-6):
        array_np = np.asarray(array, dtype=np.float32)
        valid_np = np.asarray(valid_mask, dtype=np.float32)
        if array_np.shape[:2] != valid_np.shape:
            raise ValueError(
                f"Masked surface resize expects array shape[:2] {array_np.shape[:2]} to match "
                f"valid_mask shape {valid_np.shape}"
            )

        tensor = torch.as_tensor(array_np, dtype=torch.float32)
        mask_tensor = torch.as_tensor(valid_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(
                f"Expected 2D or 3D array for masked surface resize, got shape {tuple(array_np.shape)}"
            )

        kwargs = {}
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False

        weighted = F.interpolate(tensor * mask_tensor, size=size_yx, mode=mode, **kwargs)[0]
        weights = F.interpolate(mask_tensor, size=size_yx, mode=mode, **kwargs)[0, 0]
        support = weights > float(eps)
        safe_weights = torch.where(support, weights, torch.ones_like(weights))
        resized = weighted / safe_weights.unsqueeze(0)
        resized = torch.where(support.unsqueeze(0), resized, torch.zeros_like(resized))

        if array_np.ndim == 2:
            return resized[0].cpu().numpy(), support.cpu().numpy()
        return resized.permute(1, 2, 0).cpu().numpy(), support.cpu().numpy()

    @staticmethod
    def _resize_normal_pooled_surface_arrays(
        *,
        local_grid,
        normals_zyx,
        geometry_valid_src,
        class_codes,
        supervision_valid_src,
        size_yx,
        eps=1e-6,
    ):
        local_grid_tensor = torch.as_tensor(
            np.asarray(local_grid, dtype=np.float32),
            dtype=torch.float32,
        ).permute(2, 0, 1).unsqueeze(0)
        normals_tensor = torch.as_tensor(
            np.asarray(normals_zyx, dtype=np.float32),
            dtype=torch.float32,
        ).permute(2, 0, 1).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            np.asarray(geometry_valid_src, dtype=np.float32),
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        weighted_geometry = torch.cat(
            [local_grid_tensor, normals_tensor],
            dim=1,
        ) * mask_tensor
        resized_geometry = F.interpolate(
            weighted_geometry,
            size=size_yx,
            mode="bilinear",
            align_corners=False,
        )[0]
        weights = F.interpolate(
            mask_tensor,
            size=size_yx,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        support = weights > float(eps)
        safe_weights = torch.where(support, weights, torch.ones_like(weights))
        resized_geometry = resized_geometry / safe_weights.unsqueeze(0)
        resized_geometry = torch.where(
            support.unsqueeze(0),
            resized_geometry,
            torch.zeros_like(resized_geometry),
        )

        label_tensor = torch.as_tensor(
            np.stack(
                [
                    np.asarray(
                        (class_codes == 1) & supervision_valid_src,
                        dtype=np.float32,
                    ),
                    np.asarray(supervision_valid_src, dtype=np.float32),
                ],
                axis=0,
            ),
            dtype=torch.float32,
        ).unsqueeze(0)
        resized_labels = F.interpolate(
            label_tensor,
            size=size_yx,
            mode="bilinear",
            align_corners=False,
        )[0]
        positive_support = resized_labels[0] > float(eps)
        supervision_support = resized_labels[1] > float(eps)

        return (
            resized_geometry[:3].permute(1, 2, 0).cpu().numpy(),
            resized_geometry[3:].permute(1, 2, 0).cpu().numpy(),
            support.cpu().numpy(),
            positive_support.cpu().numpy().astype(np.float32, copy=False),
            supervision_support.cpu().numpy(),
        )

    def _build_normal_pooled_surface_sample(self, sampled_grid):
        geometry_valid_src = (
            sampled_grid["in_patch"]
            & sampled_grid["valid_interp"]
            & sampled_grid["normals_valid"]
        )
        
        label_valid_src = sampled_grid["in_patch"] & sampled_grid["valid_interp"]
        supervision_valid_src = label_valid_src & (sampled_grid["class_codes"] != 100)

        (
            surface_points_zyx,
            surface_normals_zyx,
            geometry_supported,
            surface_targets_2d,
            surface_valid_2d,
        ) = self._resize_normal_pooled_surface_arrays(
            local_grid=sampled_grid["local_grid"],
            normals_zyx=sampled_grid["normals_zyx"],
            geometry_valid_src=geometry_valid_src,
            class_codes=sampled_grid["class_codes"],
            supervision_valid_src=supervision_valid_src,
            size_yx=self.surface_target_size_yx,
        )
        surface_normals_zyx, normals_nonzero = _normalize_vectors_last_axis(
            surface_normals_zyx
        )

        surface_valid_2d &= geometry_supported
        surface_valid_2d &= normals_nonzero
        surface_valid_2d &= self._compute_surface_point_mask(
            surface_points_zyx,
            sampled_grid.get("crop_size", self.patch_size_zyx),
            np.ones(self.surface_target_size_yx, dtype=bool),
        )
        assert bool(np.any(surface_valid_2d)), (
            "normal_pooled_3d supervision must contain at least one valid surface sample"
        )
        assert bool(np.any(surface_targets_2d[surface_valid_2d] > 0.0)), (
            "normal_pooled_3d supervision must contain at least one positive surface label"
        )

        surface_targets_2d[~surface_valid_2d] = 0.0
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
        }

    @staticmethod
    def _is_resampleable_normal_pooled_error(exc):
        message = str(exc)
        return message in {
            "normal_pooled_3d supervision must contain at least one valid surface sample",
            "normal_pooled_3d supervision must contain at least one positive surface label",
        }

    def _build_sample_for_index(self, idx):
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
            "wrap_mode": self.wrap_mode,
            "idx": int(idx),
        }

    def __getitem__(self, idx):
        idx = int(idx)
        if not self.use_normal_pooled_3d:
            return self._build_sample_for_index(idx)

        attempt_idx = 0
        num_patches = len(self.patches)
        candidate_idx = idx
        while True:
            try:
                return self._build_sample_for_index(candidate_idx)
            except AssertionError as exc:
                if not self._is_resampleable_normal_pooled_error(exc):
                    raise
                attempt_idx += 1
                warnings.warn(
                    (
                        "Resampling invalid normal_pooled_3d sample "
                        f"(requested_idx={idx}, sampled_idx={int(candidate_idx)}, "
                        f"attempt={attempt_idx}): {exc}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                candidate_idx = int(np.random.randint(num_patches))

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
