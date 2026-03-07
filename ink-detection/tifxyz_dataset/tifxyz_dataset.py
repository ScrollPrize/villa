import os
import numpy as np
import torch

from torch.utils.data import Dataset

from .common import (
    _normalize_patch_size_zyx,
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
    ):
        self.apply_augmentation = apply_augmentation
        self.apply_perturbation = bool(apply_perturbation)
        self.patch_size = config["patch_size"]                                          # 3d vol crop / model input patch size
        self.patch_size_zyx = _normalize_patch_size_zyx(self.patch_size)  
        
        bg_distance = config["bg_distance"]                                             # [positive, negative]
        self.bg_distance = (float(bg_distance[0]), float(bg_distance[1]))
        
        label_distance = config["label_distance"]                                       # [positive, negative]
        self.label_distance = (float(label_distance[0]), float(label_distance[1]))

        self.normal_sample_step = float(config.get("normal_sample_step", 0.5))
        self.surface_bbox_pad = float(config.get("surface_bbox_pad", 2.0))
      
        self.overlap_fraction = float(config.get("overlap_fraction", 0.25))             # amount of overlap (stride) in train/val patches, as a percentage of the patch size
        self.min_positive_fraction = float(config.get("min_positive_fraction", 0.01))   # minimum amount of labeled voxels in a candidate bbox to be added to our patches list, as a percentage of the total voxels
        self.min_span_ratio = float(config.get("min_span_ratio", 0.50))                 # the "span" in this instance is how far across the principle "direction" axis the segment should span (bbox local)
        self.patch_finding_workers = int(
            config.get("patch_finding_workers", 4)
        )                                                                               # workers for both z-band generation and bbox filtering
        self.patch_cache_force_recompute = bool(
            config.get("patch_cache_force_recompute", False)
        )
        self.patch_cache_filename = str(config["patch_cache_filename"])

        self._segment_grid_cache = {}
        self._segment_labels_and_mask_cache = {}
        self._segment_normal_cache = {}

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

    def __getitem__(self, idx):
        patch = self.patches[idx]

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
        
        segment = patch["segment"]
        
        sampled_grid = _sample_patch_supervision_grid(
            self,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            extra_bbox_pad=max(max(self.bg_distance), max(self.label_distance)) + 1.0,
        )

        positive_point_mask = sampled_grid["in_patch"] & (sampled_grid["class_codes"] == 1)
        assert bool(np.any(positive_point_mask)), (
            "sampled_grid must contain labeled points inside the patch"
        )
        positive_label_vox = _points_to_voxels(
            sampled_grid["local_grid"][positive_point_mask],
            crop_size,
        )

        background_point_mask = sampled_grid["in_patch"] & (sampled_grid["class_codes"] == 0)
        background_label_vox = _points_to_voxels(
            sampled_grid["local_grid"][background_point_mask],
            crop_size,
        )

        surface_point_mask = sampled_grid["in_patch"]
        assert sampled_grid["local_grid"].size > 0 and bool(np.any(surface_point_mask)), (
            "sampled_grid must contain in-patch surface points"
        )
        surface_vox = voxelize_surface_grid_masked(
            sampled_grid["local_grid"],
            crop_size,
            surface_point_mask,
        ).astype(np.float32, copy=False)

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
            label_distance=self.label_distance,
            require_points=True,
        )
        assert bool(np.any(label_projection_vox > 0.0)), "label projection must produce non-empty supervision"
        projected_loss_mask[label_projection_vox > 0.0] = 1.0

        labeled_vox_at_surface = np.full(crop_size, 2.0, dtype=np.float32)
        labeled_vox_at_surface[background_label_vox > 0.0] = 0.0
        labeled_vox_at_surface[positive_label_vox > 0.0] = 1.0

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
