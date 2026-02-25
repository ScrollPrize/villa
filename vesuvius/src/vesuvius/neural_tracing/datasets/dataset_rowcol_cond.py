import zarr
import vesuvius.tifxyz as tifxyz
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import tifffile
from pathlib import Path
from vesuvius.neural_tracing.datasets.common import (
    ChunkPatch,
    _compute_triplet_edt_bbox,
    _compute_wrap_order_stats,
    _extract_wrap_ids,
    _filter_triplet_overlap_chunks,
    _prepare_cond_surface_keypoints,
    _parse_z_range,
    _read_volume_crop_from_patch,
    _require_augmented_keypoints,
    _segment_overlaps_z_range,
    _signed_distance_field,
    _should_attempt_cond_local_perturb,
    _triplet_close_contact_fractions,
    _trim_to_world_bbox,
    _triplet_wraps_compatible,
    _upsample_world_triplet,
    _validate_result_tensors,
    create_band_mask,
    compute_heatmap_targets,
    edt_dilate_binary_mask,
    voxelize_surface_grid,
)
from vesuvius.neural_tracing.datasets.patch_finding import find_world_chunk_patches
from vesuvius.neural_tracing.datasets.direction_helpers import (
    build_triplet_direction_priors_from_conditioning_surface,
    maybe_swap_triplet_branch_channels,
)
from vesuvius.neural_tracing.datasets.augmentation import (
    augment_split_payload,
    augment_triplet_payload,
)
from vesuvius.neural_tracing.datasets.dataset_defaults import (
    setdefault_rowcol_cond_dataset_config,
    validate_rowcol_cond_dataset_config,
)
from vesuvius.neural_tracing.datasets.conditioning import (
    create_centered_conditioning,
    create_split_conditioning,
)
from vesuvius.neural_tracing.datasets.perturbation import maybe_perturb_conditioning_surface
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
import random
from scipy import ndimage
import warnings

class EdtSegDataset(Dataset):
    def __init__(
            self,
            config,
            apply_augmentation: bool = True,
            apply_perturbation: bool = True,
            patch_metadata=None
    ):
        self.config = config
        self.apply_augmentation = apply_augmentation
        self.apply_perturbation = bool(apply_perturbation)

        crop_size_cfg = config.get('crop_size', 128)
        if isinstance(crop_size_cfg, (list, tuple)):
            if len(crop_size_cfg) != 3:
                raise ValueError(f"crop_size must be an int or a list of 3 ints, got {crop_size_cfg}")
            self.crop_size = tuple(int(x) for x in crop_size_cfg)
        else:
            size = int(crop_size_cfg)
            self.crop_size = (size, size, size)

        target_size = self.crop_size
        self._heatmap_axes = [torch.arange(s, dtype=torch.float32) for s in self.crop_size]

        setdefault_rowcol_cond_dataset_config(config)
        validate_rowcol_cond_dataset_config(config)
        self.displacement_supervision = str(config['displacement_supervision']).lower()
        self.use_dense_displacement = bool(config['use_dense_displacement'])
        self.use_triplet_wrap_displacement = bool(config['use_triplet_wrap_displacement'])
        if not self.use_triplet_wrap_displacement:
            # Regular split is dense-supervision only.
            self.use_dense_displacement = True
        self.use_triplet_direction_priors = bool(config['use_triplet_direction_priors'])
        self.triplet_direction_prior_mask = str(config['triplet_direction_prior_mask']).lower()
        self.triplet_random_channel_swap_prob = 0.5
        self.triplet_close_check_enabled = bool(config['triplet_close_check_enabled'])
        self.triplet_close_distance_voxels = float(config['triplet_close_distance_voxels'])
        self.triplet_close_fraction_threshold = float(config['triplet_close_fraction_threshold'])
        self.triplet_edt_bbox_padding_voxels = float(config['triplet_edt_bbox_padding_voxels'])
        self.triplet_close_print = bool(config['triplet_close_print'])
        self._validate_result_tensors_enabled = bool(config['validate_result_tensors'])

        aug_config = config.get('augmentation', {})
        if apply_augmentation and aug_config.get('enabled', True):
            self._augmentations = create_training_transforms(
                patch_size=self.crop_size,
                no_spatial=False,
                no_scaling=False,
                only_spatial_and_intensity=aug_config.get('only_spatial_and_intensity', False),
            )
        else:
            self._augmentations = None
        self._cond_local_perturb_active = _should_attempt_cond_local_perturb(
            config=self.config,
            apply_perturbation=bool(self.apply_perturbation),
        )

        self.sample_mode = str(config['sample_mode']).lower()
        self._triplet_neighbor_lookup = {}
        self._triplet_lookup_stats = {}
        self._triplet_overlap_kept_indices = tuple()

        if patch_metadata is None:
            patches = []
            for dataset_idx, dataset in enumerate(config['datasets']):
                volume_path = dataset['volume_path']
                volume_scale = dataset['volume_scale']
                volume = zarr.open_group(volume_path, mode='r')
                segments_path = dataset['segments_path']
                z_range = _parse_z_range(dataset.get('z_range', None))
                dataset_segments = list(tifxyz.load_folder(segments_path))

                # retarget to the proper scale
                retarget_factor = 2 ** volume_scale
                scaled_segments = []
                dropped_by_z_range = 0
                for i, seg in enumerate(dataset_segments):
                    seg_scaled = seg.retarget(retarget_factor)
                    if not _segment_overlaps_z_range(seg_scaled, z_range):
                        dropped_by_z_range += 1
                        continue
                    seg_scaled.volume = volume
                    scaled_segments.append(seg_scaled)

                if not scaled_segments:
                    warnings.warn(
                        f"No segments remain after z_range filtering for dataset_idx={dataset_idx} "
                        f"(segments_path={segments_path}, z_range={z_range}); skipping dataset entry."
                    )
                    continue

                ref_scale = int(config['patch_count_reference_scale'])
                if config['scale_normalize_patch_counts']:
                    count_scale = float(2 ** (volume_scale - ref_scale))
                    count_scale_sq = count_scale * count_scale
                else:
                    count_scale_sq = 1.0

                min_points_per_wrap = max(1, int(round(
                    float(config['min_points_per_wrap']) * count_scale_sq
                )))
                edge_touch_min_count = max(1, int(round(
                    float(config['edge_touch_min_count']) * count_scale_sq
                )))


                cache_dir = Path(segments_path) / ".patch_cache" if segments_path else None
                chunk_results = find_world_chunk_patches(
                    segments=scaled_segments,
                    target_size=target_size,
                    overlap_fraction=config['overlap_fraction'],
                    min_span_ratio=config['min_span_ratio'],
                    edge_touch_frac=config['edge_touch_frac'],
                    edge_touch_min_count=edge_touch_min_count,
                    edge_touch_pad=config['edge_touch_pad'],
                    min_points_per_wrap=min_points_per_wrap,
                    bbox_pad_2d=config['bbox_pad_2d'],
                    require_all_valid_in_bbox=config['require_all_valid_in_bbox'],
                    skip_chunk_if_any_invalid=config['skip_chunk_if_any_invalid'],
                    inner_bbox_fraction=config['inner_bbox_fraction'],
                    cache_dir=cache_dir,
                    force_recompute=config['force_recompute_patches'],
                    verbose=config.get('verbose', False),
                    chunk_pad=config.get('chunk_pad', 0.0),
                )

                for chunk in chunk_results:
                    wraps_in_chunk = []
                    for w in chunk["wraps"]:
                        seg_idx = w["segment_idx"]
                        wraps_in_chunk.append({
                            "segment": scaled_segments[seg_idx],
                            "bbox_2d": tuple(w["bbox_2d"]),
                            "wrap_id": w["wrap_id"],
                            "segment_idx": seg_idx,
                        })

                    patches.append(ChunkPatch(
                        chunk_id=tuple(chunk["chunk_id"]),
                        volume=volume,
                        scale=volume_scale,
                        world_bbox=tuple(chunk["bbox_3d"]),
                        wraps=wraps_in_chunk,
                        segments=scaled_segments,
                    ))

            if self.use_triplet_wrap_displacement:
                patches, self._triplet_overlap_kept_indices = _filter_triplet_overlap_chunks(
                    patches,
                    config=self.config,
                )

            self.patches = patches
            self.sample_index = self._build_sample_index()
            if self.use_triplet_wrap_displacement:
                self._triplet_neighbor_lookup = self._build_triplet_neighbor_lookup()
                self.sample_index = [
                    (patch_idx, wrap_idx)
                    for patch_idx, wrap_idx in self.sample_index
                    if (patch_idx, wrap_idx) in self._triplet_neighbor_lookup
                ]
                if not self.sample_index:
                    raise ValueError(
                        "Triplet mode enabled but no wraps have same/adjacent neighbors on both sides."
                    )
            spec = self.config['cond_percent']
            low, high = float(spec[0]), float(spec[1])

            self._cond_percent_min, self._cond_percent_max = low, high
        else:
            self._load_patch_metadata(patch_metadata)

        self._dense_axis_cache = {}
        self._cc_structure_26 = np.ones((3, 3, 3), dtype=np.uint8)
        self._closing_structure_3 = np.ones((3, 3, 3), dtype=bool)

    def __len__(self):
        return len(self.sample_index)

    def _build_sample_index(self):
        if self.sample_mode == 'chunk':
            return [(patch_idx, None) for patch_idx in range(len(self.patches))]

        # wrap mode: each (chunk, wrap) pair is a unique dataset sample
        sample_index = []
        for patch_idx, patch in enumerate(self.patches):
            for wrap_idx in range(len(patch.wraps)):
                sample_index.append((patch_idx, wrap_idx))
        return sample_index

    def _load_patch_metadata(self, patch_metadata):
        self.sample_mode = patch_metadata.get('sample_mode', self.sample_mode)
        self.patches = patch_metadata['patches']
        self.sample_index = list(patch_metadata['sample_index'])
        self._triplet_neighbor_lookup = patch_metadata.get('triplet_neighbor_lookup', {})
        self._triplet_lookup_stats = patch_metadata.get('triplet_lookup_stats', {})
        self._triplet_overlap_kept_indices = tuple(
            int(i) for i in patch_metadata.get('triplet_overlap_kept_indices', tuple())
        )
        self._cond_percent_min, self._cond_percent_max = patch_metadata['cond_percent']

    def export_patch_metadata(self):
        return {
            'sample_mode': self.sample_mode,
            'patches': self.patches,
            'sample_index': tuple(self.sample_index),
            'triplet_neighbor_lookup': self._triplet_neighbor_lookup,
            'triplet_lookup_stats': self._triplet_lookup_stats,
            'triplet_overlap_kept_indices': tuple(getattr(self, '_triplet_overlap_kept_indices', tuple())),
            'cond_percent': (self._cond_percent_min, self._cond_percent_max),
            'use_triplet_wrap_displacement': self.use_triplet_wrap_displacement,
        }

    def _build_triplet_neighbor_lookup(self):
        """Build (patch_idx, wrap_idx) -> neighbor-wrap metadata for triplet mode.

        A wrap is kept only when it has one compatible neighbor on each side in
        local wrap ordering. Branches are intentionally unordered in triplet mode:
        we assign a deterministic side-based mapping so channel layout is stable,
        and rely on random branch swapping during training.
        """
        lookup = {}
        order_stats = {
            "candidate_triplets": 0,
            "kept_triplets": 0,
            "dropped_missing_neighbors": 0,
        }
        for patch_idx, patch in enumerate(self.patches):
            wrap_stats = []
            for wrap_idx, wrap in enumerate(patch.wraps):
                s = _compute_wrap_order_stats(wrap)
                if s is None:
                    continue
                seg = wrap.get("segment")
                seg_path = getattr(seg, "path", None)
                seg_name = Path(seg_path).name if seg_path is not None else ""
                wrap_ids = _extract_wrap_ids(seg_name)
                if not wrap_ids:
                    wrap_ids = _extract_wrap_ids(getattr(seg, "uuid", ""))
                wrap_stats.append({
                    "wrap_idx": wrap_idx,
                    "segment_idx": int(wrap["segment_idx"]),
                    "wrap_ids": wrap_ids,
                    "x_median": s["x_median"],
                    "y_median": s["y_median"],
                })

            if len(wrap_stats) < 3:
                continue

            x_medians = np.array([s["x_median"] for s in wrap_stats], dtype=np.float32)
            y_medians = np.array([s["y_median"] for s in wrap_stats], dtype=np.float32)
            x_spread = float(np.max(x_medians) - np.min(x_medians))
            y_spread = float(np.max(y_medians) - np.min(y_medians))
            order_axis = "x" if x_spread >= y_spread else "y"

            if order_axis == "x":
                ordered = sorted(wrap_stats, key=lambda s: (s["x_median"], s["wrap_idx"]))
            else:
                ordered = sorted(wrap_stats, key=lambda s: (s["y_median"], s["wrap_idx"]))

            for pos, target in enumerate(ordered):
                prev_neighbor = None
                for left_pos in range(pos - 1, -1, -1):
                    candidate = ordered[left_pos]
                    if _triplet_wraps_compatible(target, candidate):
                        prev_neighbor = candidate
                        break

                next_neighbor = None
                for right_pos in range(pos + 1, len(ordered)):
                    candidate = ordered[right_pos]
                    if _triplet_wraps_compatible(target, candidate):
                        next_neighbor = candidate
                        break

                if prev_neighbor is None or next_neighbor is None:
                    order_stats["dropped_missing_neighbors"] += 1
                    continue

                order_stats["candidate_triplets"] += 1
                lookup[(patch_idx, target["wrap_idx"])] = {
                    "behind_wrap_idx": int(prev_neighbor["wrap_idx"]),
                    "front_wrap_idx": int(next_neighbor["wrap_idx"]),
                }
                order_stats["kept_triplets"] += 1
        self._triplet_lookup_stats = order_stats
        return lookup

    def _extract_wrap_world_surface(self, patch: ChunkPatch, wrap: dict, require_all_valid: bool = True):
        """Extract one wrap as upsampled world-coordinate [H, W, 3] (ZYX)."""
        seg = wrap["segment"]
        r_min, r_max, c_min, c_max = wrap["bbox_2d"]

        seg_h, seg_w = seg._valid_mask.shape
        r_min = max(0, r_min)
        r_max = min(seg_h - 1, r_max)
        c_min = max(0, c_min)
        c_max = min(seg_w - 1, c_max)
        if r_max < r_min or c_max < c_min:
            return None

        seg.use_stored_resolution()
        scale_y, scale_x = seg._scale
        x_s, y_s, z_s, valid_s = seg[r_min:r_max + 1, c_min:c_max + 1]
        if x_s.size == 0:
            return None
        if valid_s is not None:
            if require_all_valid and not valid_s.all():
                return None
            if not require_all_valid and not valid_s.any():
                return None

        x_full, y_full, z_full = _upsample_world_triplet(x_s, y_s, z_s, scale_y, scale_x)
        trimmed = _trim_to_world_bbox(x_full, y_full, z_full, patch.world_bbox)
        if trimmed is None:
            return None
        x_full, y_full, z_full = trimmed
        return np.stack([z_full, y_full, x_full], axis=-1)

    def _extract_wrap_world_surface_by_index(self, patch_idx: int, wrap_idx: int, require_all_valid: bool = True):
        """Extract one wrap surface by (patch_idx, wrap_idx)."""
        patch = self.patches[patch_idx]
        wrap = patch.wraps[wrap_idx]
        return self._extract_wrap_world_surface(patch, wrap, require_all_valid=require_all_valid)

    def _conditioning_from_surface(
        self,
        *,
        cond_surface_local,
        cond_seg_gt: torch.Tensor,
    ):
        if cond_surface_local is None:
            return cond_seg_gt.clone()

        surface_np = cond_surface_local.detach().cpu().numpy().astype(np.float32, copy=False)
        perturbed_surface = maybe_perturb_conditioning_surface(
            surface_np,
            config=self.config,
            apply_perturbation=bool(self.apply_perturbation),
        )
        if perturbed_surface is surface_np:
            return cond_seg_gt.clone()

        cond_np = voxelize_surface_grid(
            np.asarray(perturbed_surface, dtype=np.float32),
            self.crop_size,
        ).astype(np.float32, copy=False)
        if not bool(cond_np.any()):
            return None

        cond_seg = torch.from_numpy(cond_np).to(torch.float32)
        if not bool(torch.isfinite(cond_seg).all()):
            return None
        return cond_seg

    def _restore_cond_surface_from_augmented(
        self,
        *,
        augmented: dict,
        cond_surface_keypoints: torch.Tensor,
        cond_surface_shape,
        mode: str,
    ):
        augmented_keypoints = _require_augmented_keypoints(
            augmented,
            cond_surface_keypoints.shape,
            mode=mode,
        )
        return augmented_keypoints.reshape(*cond_surface_shape, 3).contiguous()

    def _resolve_conditioning_segmentation(
        self,
        *,
        mask_bundle: dict,
        cond_seg_gt: torch.Tensor,
        cond_surface_local,
        cond_local_perturb_active: bool,
    ):
        if cond_local_perturb_active and cond_surface_local is not None:
            return self._conditioning_from_surface(
                cond_surface_local=cond_surface_local,
                cond_seg_gt=cond_seg_gt,
            )
        cond_seg = mask_bundle.get("cond")
        if cond_seg is None:
            cond_seg = cond_seg_gt.clone()
        return cond_seg

    def create_neighbor_masks(self, idx: int, patch_idx: int, wrap_idx: int):
        patch = self.patches[patch_idx]
        conditioning = create_centered_conditioning(self, idx, patch_idx, wrap_idx, patch)
        if conditioning is None:
            return None
        center_zyxs_unperturbed = conditioning["center_zyxs_unperturbed"]
        behind_zyxs = conditioning["behind_zyxs"]
        front_zyxs = conditioning["front_zyxs"]
        min_corner = conditioning["min_corner"]
        max_corner = conditioning["max_corner"]
        crop_size = self.crop_size

        vol_crop = _read_volume_crop_from_patch(
            patch,
            crop_size,
            min_corner,
            max_corner,
        )

        center_local_gt = (center_zyxs_unperturbed - min_corner).astype(np.float32)
        behind_local = (behind_zyxs - min_corner).astype(np.float32)
        front_local = (front_zyxs - min_corner).astype(np.float32)

        center_seg_gt = voxelize_surface_grid(center_local_gt, crop_size).astype(np.float32)
        behind_seg = voxelize_surface_grid(behind_local, crop_size).astype(np.float32)
        front_seg = voxelize_surface_grid(front_local, crop_size).astype(np.float32)

        if not center_seg_gt.any() or not behind_seg.any() or not front_seg.any():
            return None

        return {
            "vol": torch.from_numpy(vol_crop).to(torch.float32),
            "cond_gt": torch.from_numpy(center_seg_gt).to(torch.float32),
            "behind_seg": torch.from_numpy(behind_seg).to(torch.float32),
            "front_seg": torch.from_numpy(front_seg).to(torch.float32),
            "center_surface_local": torch.from_numpy(center_local_gt).to(torch.float32),
        }

    def create_neighbor_targets(
        self,
        *,
        cond_seg_gt: torch.Tensor,
        behind_seg: torch.Tensor,
        front_seg: torch.Tensor,
        cond_surface_local: torch.Tensor | None = None,
        idx: int,
        patch_idx: int,
        wrap_idx: int,
    ):
        crop_size = self.crop_size
        cond_np = cond_seg_gt.detach().cpu().numpy()
        behind_np = behind_seg.detach().cpu().numpy()
        front_np = front_seg.detach().cpu().numpy()

        weight_mode = str(self.config["triplet_dense_weight_mode"]).lower()
        need_neighbor_distances = weight_mode == "band"
        cond_bin_full = cond_np > 0.5
        behind_bin_full = behind_np > 0.5
        front_bin_full = front_np > 0.5
        triplet_gt_vector_dilation_radius = max(
            0.0,
            float(self.config["triplet_gt_vector_dilation_radius"]),
        )
        if triplet_gt_vector_dilation_radius > 0.0:
            behind_bin_for_gt = edt_dilate_binary_mask(
                behind_bin_full,
                triplet_gt_vector_dilation_radius,
            )
            front_bin_for_gt = edt_dilate_binary_mask(
                front_bin_full,
                triplet_gt_vector_dilation_radius,
            )
        else:
            behind_bin_for_gt = behind_bin_full
            front_bin_for_gt = front_bin_full
        if not behind_bin_for_gt.any() or not front_bin_for_gt.any():
            return None

        band_padding = max(0.0, float(self.config["triplet_band_padding_voxels"]))
        band_pct = float(self.config["triplet_band_distance_percentile"])
        band_pct = min(100.0, max(1.0, band_pct))

        if need_neighbor_distances:
            triplet_edt_bbox = _compute_triplet_edt_bbox(
                cond_mask=cond_bin_full,
                behind_mask=behind_bin_for_gt,
                front_mask=front_bin_for_gt,
                padding_voxels=self.triplet_edt_bbox_padding_voxels,
            )
            behind_disp_work = None
            front_disp_work = None
            d_behind_work = None
            d_front_work = None
            if triplet_edt_bbox is not None:
                behind_disp_work, _, d_behind_work = self._compute_dense_displacement_field(
                    behind_bin_for_gt,
                    return_weights=False,
                    return_distances=True,
                    bbox_slices=triplet_edt_bbox,
                )
                front_disp_work, _, d_front_work = self._compute_dense_displacement_field(
                    front_bin_for_gt,
                    return_weights=False,
                    return_distances=True,
                    bbox_slices=triplet_edt_bbox,
                )
                if (
                    behind_disp_work is not None and
                    front_disp_work is not None and
                    d_behind_work is not None and
                    d_front_work is not None
                ):
                    support_mask = cond_bin_full | behind_bin_for_gt | front_bin_for_gt
                    if (
                        not np.isfinite(d_behind_work[support_mask]).all() or
                        not np.isfinite(d_front_work[support_mask]).all()
                    ):
                        behind_disp_work = None
                        front_disp_work = None
                        d_behind_work = None
                        d_front_work = None

            if (
                behind_disp_work is None or
                front_disp_work is None or
                d_behind_work is None or
                d_front_work is None
            ):
                behind_disp_work, _, d_behind_work = self._compute_dense_displacement_field(
                    behind_bin_for_gt,
                    return_weights=False,
                    return_distances=True,
                )
                front_disp_work, _, d_front_work = self._compute_dense_displacement_field(
                    front_bin_for_gt,
                    return_weights=False,
                    return_distances=True,
                )
        else:
            behind_disp_work, _ = self._compute_dense_displacement_field(behind_bin_for_gt, return_weights=False)
            front_disp_work, _ = self._compute_dense_displacement_field(front_bin_for_gt, return_weights=False)
            d_behind_work = None
            d_front_work = None
        if behind_disp_work is None or front_disp_work is None:
            return None
        behind_disp_np = behind_disp_work
        front_disp_np = front_disp_work
        if self.triplet_close_check_enabled:
            cond_behind_frac, cond_front_frac, behind_front_frac = _triplet_close_contact_fractions(
                cond_mask=cond_bin_full,
                behind_mask=behind_bin_for_gt,
                front_mask=front_bin_for_gt,
                behind_disp_np=behind_disp_np,
                front_disp_np=front_disp_np,
                max_distance_voxels=self.triplet_close_distance_voxels,
            )
            max_close_frac = max(cond_behind_frac, cond_front_frac, behind_front_frac)
            if max_close_frac > self.triplet_close_fraction_threshold:
                if self.triplet_close_print:
                    print(
                        "Triplet close-contact reject "
                        f"idx={idx} patch={patch_idx} wrap={wrap_idx} "
                        f"cond-behind={cond_behind_frac:.4f} "
                        f"cond-front={cond_front_frac:.4f} "
                        f"behind-front={behind_front_frac:.4f} "
                        f"thr={self.triplet_close_fraction_threshold:.4f} "
                        f"dist<={self.triplet_close_distance_voxels:.3f}"
                    )
                return None

        if weight_mode == "all":
            dense_weight_np = np.ones((1, *crop_size), dtype=np.float32)
        elif weight_mode == "band":
            if d_behind_work is None or d_front_work is None:
                return None
            dense_band = create_band_mask(
                cond_bin_full=cond_bin_full,
                d_front_work=d_front_work,
                d_behind_work=d_behind_work,
                front_disp_work=front_disp_work,
                behind_disp_work=behind_disp_work,
                band_pct=band_pct,
                band_padding=band_padding,
                cc_structure_26=self._cc_structure_26,
                closing_structure_3=self._closing_structure_3,
            )
            if dense_band is None:
                return None
            dense_weight_np = dense_band[None]
        else:
            raise ValueError(
                "triplet_dense_weight_mode must be one of {'all', 'band'}, "
                f"got {weight_mode!r}"
            )
        if float(dense_weight_np.sum()) <= 0:
            return None

        dense_gt_np = np.concatenate([behind_disp_np, front_disp_np], axis=0)
        dir_priors_np = None
        if self.use_triplet_direction_priors:
            if cond_surface_local is None:
                return None
            cond_surface_np = cond_surface_local.detach().cpu().numpy().astype(np.float32, copy=False)
            dir_priors_np = build_triplet_direction_priors_from_conditioning_surface(
                crop_size,
                cond_bin_full,
                cond_surface_np,
                mask_mode=self.triplet_direction_prior_mask,
            )
            if dir_priors_np is None:
                return None
        dense_gt_np, dir_priors_np, triplet_channel_order_np = maybe_swap_triplet_branch_channels(
            dense_gt_np,
            dir_priors_np,
            swap_prob=self.triplet_random_channel_swap_prob,
            rng=random,
        )
        dense_gt_disp = torch.from_numpy(dense_gt_np).to(torch.float32)
        dense_loss_weight = torch.from_numpy(dense_weight_np).to(torch.float32)

        result = {
            "dense_gt_displacement": dense_gt_disp,  # (6, D, H, W): channel0(dz,dy,dx), channel1(dz,dy,dx)
            "dense_loss_weight": dense_loss_weight,  # (1, D, H, W)
            "triplet_channel_order": torch.from_numpy(triplet_channel_order_np).to(torch.int64),  # [2]: slot0/slot1 -> canonical A/B
        }
        if dir_priors_np is not None:
            result["dir_priors"] = torch.from_numpy(dir_priors_np).to(torch.float32)
        return result

    def _get_dense_axis_offsets(self, shape_zyx):
        shape_key = tuple(int(s) for s in shape_zyx)
        axes = self._dense_axis_cache.get(shape_key)
        if axes is None:
            d, h, w = shape_key
            axes = (
                np.arange(d, dtype=np.float32)[:, None, None],
                np.arange(h, dtype=np.float32)[None, :, None],
                np.arange(w, dtype=np.float32)[None, None, :],
            )
            self._dense_axis_cache[shape_key] = axes
        return axes

    def _compute_dense_displacement_field(
        self,
        full_surface_mask: np.ndarray,
        return_weights: bool = True,
        return_distances: bool = False,
        bbox_slices=None,
    ):
        """Compute nearest-surface displacement vector for every voxel.

        Args:
            full_surface_mask: (D, H, W) binary/boolean mask of GT surface voxels.
            return_distances: whether to also return nearest-surface Euclidean distance map.
            bbox_slices: optional (z, y, x) slices to compute on a sub-volume and
                scatter back into full-volume outputs.

        Returns:
            disp_field: (3, D, H, W) with components (dz, dy, dx)
            weight_mask: (1, D, H, W) per-voxel supervision weights
            distance_map: (D, H, W) nearest-surface distance (optional)
        """
        surface = np.asarray(full_surface_mask) > 0.5
        if bbox_slices is not None:
            if surface.ndim != 3:
                raise ValueError(f"full_surface_mask must be 3D, got shape {tuple(surface.shape)}")
            if len(bbox_slices) != 3:
                raise ValueError("bbox_slices must be a tuple/list of (z, y, x) slices")
            z_slice, y_slice, x_slice = bbox_slices
            sub_surface = surface[z_slice, y_slice, x_slice]
            if sub_surface.size == 0 or not sub_surface.any():
                if return_distances:
                    return None, None, None
                return None, None

            if return_distances:
                sub_disp, sub_weights, sub_dist = self._compute_dense_displacement_field(
                    sub_surface,
                    return_weights=return_weights,
                    return_distances=True,
                    bbox_slices=None,
                )
            else:
                sub_disp, sub_weights = self._compute_dense_displacement_field(
                    sub_surface,
                    return_weights=return_weights,
                    return_distances=False,
                    bbox_slices=None,
                )
                sub_dist = None
            if sub_disp is None:
                if return_distances:
                    return None, None, None
                return None, None

            disp_full = np.zeros((3, *surface.shape), dtype=np.float32)
            disp_full[:, z_slice, y_slice, x_slice] = sub_disp

            if return_weights:
                weights_full = np.zeros((1, *surface.shape), dtype=np.float32)
                if sub_weights is not None:
                    weights_full[:, z_slice, y_slice, x_slice] = sub_weights
            else:
                weights_full = None

            if return_distances:
                dist_full = np.full(surface.shape, np.inf, dtype=np.float32)
                if sub_dist is not None:
                    dist_full[z_slice, y_slice, x_slice] = sub_dist
                return disp_full, weights_full, dist_full
            return disp_full, weights_full

        if not surface.any():
            if return_distances:
                return None, None, None
            return None, None

        # Nearest foreground (surface) index per voxel.
        if return_distances:
            distances, nearest_idx = ndimage.distance_transform_edt(
                ~surface, return_distances=True, return_indices=True
            )
            distances = distances.astype(np.float32, copy=False)
        else:
            nearest_idx = ndimage.distance_transform_edt(
                ~surface, return_distances=False, return_indices=True
            )
            distances = None
        disp = nearest_idx.astype(np.float32, copy=False)
        z_axis, y_axis, x_axis = self._get_dense_axis_offsets(surface.shape)
        disp[0] -= z_axis
        disp[1] -= y_axis
        disp[2] -= x_axis
        weights = np.ones((1, *surface.shape), dtype=np.float32) if return_weights else None
        if return_distances:
            return disp, weights, distances
        return disp, weights

    def create_split_masks(self, idx: int, patch_idx: int, wrap_idx: int):
        patch = self.patches[patch_idx]
        crop_size = self.crop_size  # tuple (D, H, W)
        target_shape = crop_size

        conditioning = create_split_conditioning(self, idx, patch_idx, wrap_idx, patch)
        if conditioning is None:
            return None
        wrap = conditioning["wrap"]
        seg = conditioning["seg"]
        r_min = conditioning["r_min"]
        r_max = conditioning["r_max"]
        c_min = conditioning["c_min"]
        c_max = conditioning["c_max"]
        conditioning_percent = conditioning["conditioning_percent"]
        cond_direction = conditioning["cond_direction"]
        cond_zyxs_unperturbed = conditioning["cond_zyxs_unperturbed"]
        masked_zyxs = conditioning["masked_zyxs"]
        min_corner = conditioning["min_corner"]
        max_corner = conditioning["max_corner"]

        vol_crop = _read_volume_crop_from_patch(
            patch,
            target_shape,
            min_corner,
            max_corner,
        )

        # convert cond and masked coords to crop-local coords (float for line interpolation)
        cond_zyxs_unperturbed_local_float = (cond_zyxs_unperturbed - min_corner).astype(np.float32)
        masked_zyxs_local_float = (masked_zyxs - min_corner).astype(np.float32)

        crop_shape = target_shape

        # voxelize with line interpolation between adjacent grid points
        cond_segmentation_gt = voxelize_surface_grid(cond_zyxs_unperturbed_local_float, crop_shape)
        masked_segmentation = voxelize_surface_grid(masked_zyxs_local_float, crop_shape)

        # make sure we actually have some conditioning
        if not cond_segmentation_gt.any():
            return None

        cond_segmentation_gt_raw = cond_segmentation_gt.copy()

        # add thickness to conditioning segmentation via dilation
        use_dilation = self.config.get('use_dilation', False)
        if use_dilation:
            dilation_radius = self.config['dilation_radius']
            cond_segmentation = edt_dilate_binary_mask(
                cond_segmentation_gt > 0.5,
                dilation_radius,
            ).astype(np.float32, copy=False)
        else:
            cond_segmentation = cond_segmentation_gt

        use_segmentation = self.config['use_segmentation']
        use_sdt = self.config['use_sdt']
        full_segmentation = None
        full_segmentation_raw = None
        if use_sdt:
            # combine cond + masked into full segmentation
            full_segmentation = np.maximum(cond_segmentation, masked_segmentation)
        if use_segmentation:
            full_segmentation_raw = np.maximum(cond_segmentation_gt_raw, masked_segmentation)

        if use_sdt:
            # if already dilated, just compute SDT directly; otherwise dilate first
            if use_dilation:
                seg_dilated = full_segmentation
            else:
                dilation_radius = self.config['dilation_radius']
                seg_dilated = edt_dilate_binary_mask(
                    full_segmentation > 0.5,
                    dilation_radius,
                ).astype(np.float32, copy=False)
            sdt = _signed_distance_field(seg_dilated)

        if use_segmentation:
            dilation_radius = self.config['dilation_radius']
            full_segmentation_raw_bin = full_segmentation_raw > 0.5
            seg_dilated = edt_dilate_binary_mask(
                full_segmentation_raw_bin,
                dilation_radius,
            ).astype(np.float32, copy=False)
            seg_skel = full_segmentation_raw_bin.astype(np.float32, copy=False)

        # generate heatmap targets for expected positions in masked region
        use_heatmap = self.config['use_heatmap_targets']
        if use_heatmap:
            effective_step = int(self.config['heatmap_step_size'] * (2 ** patch.scale))
            h_s_full = r_max - r_min + 1
            w_s_full = c_max - c_min + 1
            r_cond_s = min(max(int(round(h_s_full * conditioning_percent)), 1), h_s_full - 1)
            c_cond_s = min(max(int(round(w_s_full * conditioning_percent)), 1), w_s_full - 1)
            if cond_direction == "down":
                r_split_s = r_min + (h_s_full - r_cond_s)
            else:
                r_split_s = r_min + r_cond_s
            if cond_direction == "right":
                c_split_s = c_min + (w_s_full - c_cond_s)
            else:
                c_split_s = c_min + c_cond_s
            heatmap_tensor = compute_heatmap_targets(
                cond_direction=cond_direction,
                r_split=r_split_s, c_split=c_split_s,
                r_min_full=r_min, r_max_full=r_max + 1,
                c_min_full=c_min, c_max_full=c_max + 1,
                patch_seg=seg,
                min_corner=min_corner,
                crop_size=crop_size,
                step_size=effective_step,
                step_count=self.config['heatmap_step_count'],
                sigma=self.config['heatmap_sigma'],
                axis_1d=self._heatmap_axes[0],
            )
            if heatmap_tensor is None:
                return None

        # other wrap conditioning: find and voxelize other wraps from the same segment
        use_other_wrap_cond = self.config['use_other_wrap_cond']
        other_wraps_vox = np.zeros(crop_shape, dtype=np.float32)
        if use_other_wrap_cond:
            primary_segment_idx = wrap["segment_idx"]
            other_wraps_list = [w for w in patch.wraps
                                if w["segment_idx"] == primary_segment_idx and w is not wrap]

            # if another wrap exists, get it with some probablity
            if other_wraps_list and random.random() < self.config['other_wrap_prob']:
                z_min_w, z_max_w, y_min_w, y_max_w, x_min_w, x_max_w = patch.world_bbox

                for other_wrap in other_wraps_list:
                    other_seg = other_wrap["segment"]
                    or_min, or_max, oc_min, oc_max = other_wrap["bbox_2d"]

                    other_seg_h, other_seg_w = other_seg._valid_mask.shape
                    or_min = max(0, or_min)
                    or_max = min(other_seg_h - 1, or_max)
                    oc_min = max(0, oc_min)
                    oc_max = min(other_seg_w - 1, oc_max)
                    if or_max < or_min or oc_max < oc_min:
                        continue

                    other_seg.use_stored_resolution()
                    o_scale_y, o_scale_x = other_seg._scale

                    ox_s, oy_s, oz_s, ovalid_s = other_seg[or_min:or_max+1, oc_min:oc_max+1]
                    if not ovalid_s.all():
                        continue

                    ox_full, oy_full, oz_full = _upsample_world_triplet(ox_s, oy_s, oz_s, o_scale_y, o_scale_x)
                    trimmed = _trim_to_world_bbox(
                        ox_full, oy_full, oz_full, (z_min_w, z_max_w, y_min_w, y_max_w, x_min_w, x_max_w)
                    )
                    if trimmed is None:
                        continue
                    ox_full, oy_full, oz_full = trimmed

                    other_zyxs = np.stack([oz_full, oy_full, ox_full], axis=-1)
                    other_zyxs_local = (other_zyxs - min_corner).astype(np.float32)

                    other_vox = voxelize_surface_grid(other_zyxs_local, crop_shape)
                    other_wraps_vox = np.maximum(other_wraps_vox, other_vox)

        result = {
            "vol": torch.from_numpy(vol_crop).to(torch.float32),
            "masked_seg": torch.from_numpy(masked_segmentation).to(torch.float32),
            "cond_gt": torch.from_numpy(cond_segmentation_gt_raw).to(torch.float32),
            "other_wraps": torch.from_numpy(other_wraps_vox).to(torch.float32),
            "cond_surface_local": torch.from_numpy(cond_zyxs_unperturbed_local_float).to(torch.float32),
        }
        if use_segmentation:
            result["segmentation"] = torch.from_numpy(seg_dilated).to(torch.float32)
            result["segmentation_skel"] = torch.from_numpy(seg_skel).to(torch.float32)
        if use_sdt:
            result["sdt"] = torch.from_numpy(sdt).to(torch.float32)
        if use_heatmap:
            result["heatmap_target"] = heatmap_tensor.to(torch.float32)
        return result

    def create_split_targets(
        self,
        *,
        cond_seg_gt: torch.Tensor,
        masked_seg: torch.Tensor,
    ):
        full_dense_surface = torch.maximum(masked_seg, cond_seg_gt)
        dense_disp_np, dense_weight_np = self._compute_dense_displacement_field(
            full_dense_surface.detach().cpu().numpy()
        )
        if dense_disp_np is None:
            return None
        dense_gt_disp = torch.from_numpy(dense_disp_np).to(torch.float32)
        dense_loss_weight = torch.from_numpy(dense_weight_np).to(torch.float32)
        return {
            "dense_gt_displacement": dense_gt_disp,  # (3, D, H, W)
            "dense_loss_weight": dense_loss_weight,  # (1, D, H, W)
        }

    def __getitem__(self, idx):
        patch_idx, wrap_idx = self.sample_index[idx]
        cond_local_perturb_active = bool(
            getattr(
                self,
                "_cond_local_perturb_active",
                _should_attempt_cond_local_perturb(
                    config=self.config,
                    apply_perturbation=bool(self.apply_perturbation),
                ),
            )
        )
        if self.use_triplet_wrap_displacement:
            mask_bundle = self.create_neighbor_masks(idx, patch_idx, wrap_idx)
            if mask_bundle is None:
                return self[np.random.randint(len(self))]

            vol_crop = mask_bundle["vol"]
            cond_seg_gt = mask_bundle["cond_gt"]
            behind_seg = mask_bundle["behind_seg"]
            front_seg = mask_bundle["front_seg"]
            cond_surface_local, cond_surface_shape, cond_surface_keypoints, valid_cond_surface = (
                _prepare_cond_surface_keypoints(mask_bundle.get("center_surface_local"))
            )
            if not valid_cond_surface:
                return self[np.random.randint(len(self))]

            triplet_augmented = augment_triplet_payload(
                augmentations=self._augmentations,
                crop_size=self.crop_size,
                vol_crop=vol_crop,
                cond_seg_gt=cond_seg_gt,
                behind_seg=behind_seg,
                front_seg=front_seg,
                cond_surface_local=cond_surface_local,
                cond_surface_keypoints=cond_surface_keypoints,
                cond_surface_shape=cond_surface_shape,
                restore_cond_surface_fn=self._restore_cond_surface_from_augmented,
            )
            vol_crop = triplet_augmented["vol_crop"]
            cond_seg_gt = triplet_augmented["cond_seg_gt"]
            behind_seg = triplet_augmented["behind_seg"]
            front_seg = triplet_augmented["front_seg"]
            cond_surface_local = triplet_augmented["cond_surface_local"]
            cond_seg = self._resolve_conditioning_segmentation(
                mask_bundle=mask_bundle,
                cond_seg_gt=cond_seg_gt,
                cond_surface_local=cond_surface_local,
                cond_local_perturb_active=cond_local_perturb_active,
            )
            if cond_seg is None:
                return self[np.random.randint(len(self))]

            target_payload = self.create_neighbor_targets(
                cond_seg_gt=cond_seg_gt,
                behind_seg=behind_seg,
                front_seg=front_seg,
                cond_surface_local=cond_surface_local,
                idx=idx,
                patch_idx=patch_idx,
                wrap_idx=wrap_idx,
            )
            if target_payload is None:
                return self[np.random.randint(len(self))]

            result = {
                "vol": vol_crop,
                "cond": cond_seg,
            }
            result.update(target_payload)
        else:
            use_other_wrap_cond = self.config['use_other_wrap_cond']
            use_sdt = self.config['use_sdt']
            use_heatmap = self.config['use_heatmap_targets']
            use_segmentation = self.config['use_segmentation']

            mask_bundle = self.create_split_masks(idx, patch_idx, wrap_idx)
            if mask_bundle is None:
                return self[np.random.randint(len(self))]

            vol_crop = mask_bundle["vol"]
            masked_seg = mask_bundle["masked_seg"]
            cond_seg_gt = mask_bundle["cond_gt"]
            other_wraps_tensor = mask_bundle["other_wraps"]
            cond_surface_local, cond_surface_shape, cond_surface_keypoints, valid_cond_surface = (
                _prepare_cond_surface_keypoints(mask_bundle.get("cond_surface_local"))
            )
            if not valid_cond_surface:
                return self[np.random.randint(len(self))]
            if use_segmentation:
                full_seg = mask_bundle["segmentation"]
                seg_skel = mask_bundle["segmentation_skel"]
            if use_sdt:
                sdt_tensor = mask_bundle["sdt"]
            if use_heatmap:
                heatmap_tensor = mask_bundle["heatmap_target"]

            split_augmented = augment_split_payload(
                augmentations=self._augmentations,
                crop_size=self.crop_size,
                vol_crop=vol_crop,
                masked_seg=masked_seg,
                other_wraps_tensor=other_wraps_tensor,
                cond_seg_gt=cond_seg_gt,
                cond_surface_local=cond_surface_local,
                cond_surface_keypoints=cond_surface_keypoints,
                cond_surface_shape=cond_surface_shape,
                restore_cond_surface_fn=self._restore_cond_surface_from_augmented,
                use_segmentation=use_segmentation,
                use_sdt=use_sdt,
                use_heatmap=use_heatmap,
                full_seg=full_seg if use_segmentation else None,
                seg_skel=seg_skel if use_segmentation else None,
                sdt_tensor=sdt_tensor if use_sdt else None,
                heatmap_tensor=heatmap_tensor if use_heatmap else None,
            )
            vol_crop = split_augmented["vol_crop"]
            masked_seg = split_augmented["masked_seg"]
            other_wraps_tensor = split_augmented["other_wraps_tensor"]
            cond_seg_gt = split_augmented["cond_seg_gt"]
            cond_surface_local = split_augmented["cond_surface_local"]
            if use_segmentation:
                full_seg = split_augmented["full_seg"]
                seg_skel = split_augmented["seg_skel"]
            if use_sdt:
                sdt_tensor = split_augmented["sdt_tensor"]
            if use_heatmap:
                heatmap_tensor = split_augmented["heatmap_tensor"]
            cond_seg = self._resolve_conditioning_segmentation(
                mask_bundle=mask_bundle,
                cond_seg_gt=cond_seg_gt,
                cond_surface_local=cond_surface_local,
                cond_local_perturb_active=cond_local_perturb_active,
            )
            if cond_seg is None:
                return self[np.random.randint(len(self))]

            target_payload = self.create_split_targets(
                cond_seg_gt=cond_seg_gt,
                masked_seg=masked_seg,
            )
            if target_payload is None:
                return self[np.random.randint(len(self))]

            result = {
                "vol": vol_crop,                 # raw volume crop
                "cond": cond_seg,                # conditioning segmentation
                "masked_seg": masked_seg,        # masked (target) segmentation
            }

            if use_other_wrap_cond:
                result["other_wraps"] = other_wraps_tensor  # other wraps from same segment as context
            if use_sdt:
                result["sdt"] = sdt_tensor  # signed distance transform of full (dilated) segmentation
            if use_heatmap:
                result["heatmap_target"] = heatmap_tensor  # (D, H, W) gaussian heatmap at expected positions
            if use_segmentation:
                result["segmentation"] = full_seg  # full segmentation (cond + masked)
                result["segmentation_skel"] = seg_skel  # skeleton for medial surface recall loss
            result.update(target_payload)

        if not _validate_result_tensors(
            result,
            idx,
            enabled=self._validate_result_tensors_enabled,
        ):
            return self[np.random.randint(len(self))]
        return result
    
