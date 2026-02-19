import edt
import zarr
import vesuvius.tifxyz as tifxyz
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import tifffile
from pathlib import Path
from vesuvius.neural_tracing.datasets.common import ChunkPatch, compute_heatmap_targets, voxelize_surface_grid
from vesuvius.neural_tracing.datasets.patch_finding import find_world_chunk_patches
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.image_proc.intensity.normalization import normalize_zscore
import random
from vesuvius.neural_tracing.datasets.extrapolation import compute_extrapolation
import cv2
from scipy import ndimage
from collections import OrderedDict
import warnings

import os
os.environ['OMP_NUM_THREADS'] = '1' # this is set to 1 because by default the edt package uses omp to threads the edt call
                                    # which is problematic if you use multiple dataloader workers (thread contention smokes cpu)


class EdtSegDataset(Dataset):
    def __init__(
            self,
            config,
            apply_augmentation: bool = True
    ):
        self.config = config
        self.apply_augmentation = apply_augmentation

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

        config.setdefault('use_sdt', False)
        config.setdefault('dilation_radius', 1)  # voxels
        config.setdefault('cond_percent', [0.5, 0.5])
        config.setdefault('use_extrapolation', True)
        config.setdefault('use_dense_displacement', False)
        config.setdefault('extrapolation_method', 'linear_edge')
        config.setdefault('supervise_conditioning', False)
        config.setdefault('cond_supervision_weight', 0.1)
        config.setdefault('force_recompute_patches', False)
        config.setdefault('use_heatmap_targets', False)
        config.setdefault('heatmap_step_size', 10)
        config.setdefault('heatmap_step_count', 5)
        config.setdefault('heatmap_sigma', 2.0)
        config.setdefault('use_segmentation', False)

        # other wrap conditioning: provide other wraps from same segment as context
        config.setdefault('use_other_wrap_cond', False)
        config.setdefault('other_wrap_prob', 0.5)  # probability of including other wraps when available
        config.setdefault('sample_mode', 'wrap')  # 'wrap' = each wrap is a sample, 'chunk' = random wrap per chunk
        config.setdefault('use_triplet_wrap_displacement', False)
        config.setdefault('triplet_dense_weight_mode', 'band')  # band|conditioning|all|neighbors
        config.setdefault('triplet_band_padding_voxels', 4.0)
        config.setdefault('triplet_band_distance_percentile', 95.0)
        config.setdefault('triplet_dense_roi_enabled', True)
        config.setdefault('triplet_dense_roi_padding_voxels', 8.0)
        config.setdefault('triplet_dense_roi_adaptive_padding', False)
        config.setdefault('triplet_dense_roi_max_padding_voxels', 128.0)
        config.setdefault('triplet_surface_cache_max_items', 2048)
        config.setdefault('triplet_order_abs_margin', 0.25)
        config.setdefault('triplet_order_rel_margin', 0.08)
        config.setdefault('triplet_order_warn_drop_fraction', 0.05)
        config.setdefault('triplet_order_min_band_points', 32)
        config.setdefault('triplet_order_debug_examples', 0)
        config.setdefault('enable_volume_crop_cache', True)
        config.setdefault('volume_crop_cache_max_items', 8)
        config.setdefault('validate_result_tensors', True)

        config.setdefault('overlap_fraction', 0.0)
        config.setdefault('min_span_ratio', 1.0)
        config.setdefault('edge_touch_frac', 0.1)
        config.setdefault('edge_touch_min_count', 10)
        config.setdefault('edge_touch_pad', 0)
        config.setdefault('min_points_per_wrap', 100)
        # Point-count thresholds in patch finding depend on tifxyz point density,
        # which changes with volume_scale after retargeting. Keep scale=1 behavior
        # as the default reference and scale counts quadratically for other scales.
        config.setdefault('scale_normalize_patch_counts', True)
        config.setdefault('patch_count_reference_scale', 0)
        config.setdefault('bbox_pad_2d', 0)
        config.setdefault('require_all_valid_in_bbox', True)
        config.setdefault('skip_chunk_if_any_invalid', False)
        config.setdefault('min_cond_span', 0.3)
        config.setdefault('inner_bbox_fraction', 0.7)
        config.setdefault('filter_oob_extrap_points', True)
        cond_local_perturb = dict(config.get('cond_local_perturb') or {})
        cond_local_perturb.setdefault('enabled', True)
        cond_local_perturb.setdefault('probability', 0.35)
        cond_local_perturb.setdefault('num_blobs', [1, 3])
        cond_local_perturb.setdefault('points_affected', 10)
        cond_local_perturb.setdefault('sigma_fraction_range', [0.04, 0.10])
        cond_local_perturb.setdefault('amplitude_range', [0.25, 1.25])
        cond_local_perturb.setdefault('radius_sigma_mult', 2.5)
        cond_local_perturb.setdefault('max_total_displacement', 6.0)
        cond_local_perturb.setdefault('apply_without_augmentation', False)
        config['cond_local_perturb'] = cond_local_perturb
        config.setdefault('rbf_downsample_factor', 4)
        config.setdefault('rbf_edge_downsample_factor', 8)
        config.setdefault('rbf_max_points', None)
        config.setdefault('rbf_edge_band_frac', 0.10)
        config.setdefault('rbf_edge_band_cells', None)
        config.setdefault('rbf_edge_min_points', 128)
        config.setdefault('debug_extrapolation_oob', False)
        config.setdefault('debug_extrapolation_oob_every', 100)
        config.setdefault('displacement_supervision', 'vector')
        self.displacement_supervision = str(config.get('displacement_supervision', 'vector')).lower()
        if self.displacement_supervision not in {'vector', 'normal_scalar'}:
            raise ValueError(
                "displacement_supervision must be 'vector' or 'normal_scalar', "
                f"got {self.displacement_supervision!r}"
            )
        self.use_dense_displacement = bool(config.get('use_dense_displacement', False))
        self.use_triplet_wrap_displacement = bool(config.get('use_triplet_wrap_displacement', False))
        if self.displacement_supervision == 'normal_scalar' and self.use_dense_displacement:
            raise ValueError("displacement_supervision='normal_scalar' is not supported with use_dense_displacement=True")
        if self.use_triplet_wrap_displacement:
            if not self.use_dense_displacement:
                raise ValueError("use_triplet_wrap_displacement=True requires use_dense_displacement=True")
            if config.get('use_extrapolation', True):
                raise ValueError("use_triplet_wrap_displacement=True requires use_extrapolation=False")
            if config.get('use_other_wrap_cond', False):
                raise ValueError("use_triplet_wrap_displacement=True is not compatible with use_other_wrap_cond")
            if config.get('use_sdt', False):
                raise ValueError("use_triplet_wrap_displacement=True is not compatible with use_sdt")
            if config.get('use_heatmap_targets', False):
                raise ValueError("use_triplet_wrap_displacement=True is not compatible with use_heatmap_targets")
            if config.get('use_segmentation', False):
                raise ValueError("use_triplet_wrap_displacement=True is not compatible with use_segmentation")
        self._needs_point_normals = self.displacement_supervision == 'normal_scalar'
        self._validate_result_tensors_enabled = bool(config.get('validate_result_tensors', True))

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

        patches = []

        for dataset in config['datasets']:
            volume_path = dataset['volume_path']
            volume_scale = dataset['volume_scale']
            volume = zarr.open_group(volume_path, mode='r')
            segments_path = dataset['segments_path']
            dataset_segments = list(tifxyz.load_folder(segments_path))

            # retarget to the proper scale
            retarget_factor = 2 ** volume_scale
            scaled_segments = []
            for i, seg in enumerate(dataset_segments):
                if i == 0:
                    if config['verbose']:
                        print(f"  [DEBUG PRE-RETARGET] seg._scale={seg._scale}, shape={seg._z.shape}")
                        print(f"  [DEBUG PRE-RETARGET] z range: {seg._z[seg._valid_mask].min():.2f} to {seg._z[seg._valid_mask].max():.2f}")
                seg_scaled = seg.retarget(retarget_factor)
                if i == 0:
                    if config['verbose']:
                        print(f"  [DEBUG POST-RETARGET factor={retarget_factor}] seg._scale={seg_scaled._scale}, shape={seg_scaled._z.shape}")
                        print(f"  [DEBUG POST-RETARGET] z range: {seg_scaled._z[seg_scaled._valid_mask].min():.2f} to {seg_scaled._z[seg_scaled._valid_mask].max():.2f}")
                seg_scaled.volume = volume
                scaled_segments.append(seg_scaled)

            ref_scale = int(config.get('patch_count_reference_scale', 0))
            if config.get('scale_normalize_patch_counts', True):
                count_scale = float(2 ** (volume_scale - ref_scale))
                count_scale_sq = count_scale * count_scale
            else:
                count_scale_sq = 1.0

            min_points_per_wrap = max(1, int(round(
                float(config.get('min_points_per_wrap', 100)) * count_scale_sq
            )))
            edge_touch_min_count = max(1, int(round(
                float(config.get('edge_touch_min_count', 10)) * count_scale_sq
            )))

            if config.get('verbose', False):
                print(
                    "  [DEBUG PATCH COUNTS] "
                    f"volume_scale={volume_scale}, ref_scale={ref_scale}, "
                    f"count_scale_sq={count_scale_sq:.3f}, "
                    f"min_points_per_wrap={min_points_per_wrap}, "
                    f"edge_touch_min_count={edge_touch_min_count}"
                )

            cache_dir = Path(segments_path) / ".patch_cache" if segments_path else None
            chunk_results = find_world_chunk_patches(
                segments=scaled_segments,
                target_size=target_size,
                overlap_fraction=config.get('overlap_fraction', 0.0),
                min_span_ratio=config.get('min_span_ratio', 1.0),
                edge_touch_frac=config.get('edge_touch_frac', 0.1),
                edge_touch_min_count=edge_touch_min_count,
                edge_touch_pad=config.get('edge_touch_pad', 0),
                min_points_per_wrap=min_points_per_wrap,
                bbox_pad_2d=config.get('bbox_pad_2d', 0),
                require_all_valid_in_bbox=config.get('require_all_valid_in_bbox', True),
                skip_chunk_if_any_invalid=config.get('skip_chunk_if_any_invalid', False),
                inner_bbox_fraction=config.get('inner_bbox_fraction', 0.7),
                cache_dir=cache_dir,
                force_recompute=config.get('force_recompute_patches', False),
                verbose=True,
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

        self.patches = patches
        self.sample_mode = str(config.get('sample_mode', 'wrap')).lower()
        if self.sample_mode not in {'wrap', 'chunk'}:
            raise ValueError(f"sample_mode must be 'wrap' or 'chunk', got {self.sample_mode!r}")
        self.sample_index = self._build_sample_index()
        self._triplet_neighbor_lookup = {}
        self._triplet_lookup_stats = {}
        self._triplet_surface_cache_max_items = int(config.get('triplet_surface_cache_max_items', 2048))
        if self._triplet_surface_cache_max_items < 0:
            self._triplet_surface_cache_max_items = 0
        self._triplet_surface_cache = OrderedDict()
        self._enable_volume_crop_cache = bool(config.get('enable_volume_crop_cache', True))
        self._volume_crop_cache_max_items = int(config.get('volume_crop_cache_max_items', 8))
        if self._volume_crop_cache_max_items < 0:
            self._volume_crop_cache_max_items = 0
        self._volume_crop_cache = OrderedDict()
        self._dense_axis_cache = {}
        if self.use_triplet_wrap_displacement:
            if self.sample_mode != 'wrap':
                raise ValueError("use_triplet_wrap_displacement=True requires sample_mode='wrap'")
            self._triplet_neighbor_lookup = self._build_triplet_neighbor_lookup()
            self.sample_index = [
                (patch_idx, wrap_idx)
                for patch_idx, wrap_idx in self.sample_index
                if (patch_idx, wrap_idx) in self._triplet_neighbor_lookup
            ]
            if not self.sample_index:
                raise ValueError(
                    "Triplet mode enabled but no wraps have same-segment neighbors on both sides."
                )
        self._cond_percent_min, self._cond_percent_max = self._parse_cond_percent()

        if config.get('verbose', False):
            total_wraps = sum(len(p.wraps) for p in self.patches)
            print(
                f"RowCol dataset built: chunks={len(self.patches)}, "
                f"wraps={total_wraps}, sample_mode={self.sample_mode}, "
                f"samples={len(self.sample_index)}"
            )
            if self.use_triplet_wrap_displacement:
                print(f"Triplet-wrap samples={len(self.sample_index)}")

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

    def _compute_wrap_order_stats(self, wrap):
        """Compute robust world-space stats/points used for triplet wrap ordering."""
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
        x_s, y_s, z_s, valid_s = seg[r_min:r_max + 1, c_min:c_max + 1]
        if x_s.size == 0:
            return None

        if valid_s is not None:
            if not valid_s.any():
                return None
            x_vals = x_s[valid_s]
            y_vals = y_s[valid_s]
            z_vals = z_s[valid_s]
        else:
            x_vals = x_s.reshape(-1)
            y_vals = y_s.reshape(-1)
            z_vals = z_s.reshape(-1)

        finite = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(z_vals)
        if not finite.any():
            return None
        x_vals = x_vals[finite]
        y_vals = y_vals[finite]
        z_vals = z_vals[finite]

        if x_vals.size == 0 or y_vals.size == 0 or z_vals.size == 0:
            return None

        points_zyx = np.stack([z_vals, y_vals, x_vals], axis=1).astype(np.float64, copy=False)

        return {
            "z_median": float(np.median(z_vals)),
            "z_min": float(np.min(z_vals)),
            "z_max": float(np.max(z_vals)),
            "x_median": float(np.median(x_vals)),
            "y_median": float(np.median(y_vals)),
            "x_span": float(np.max(x_vals) - np.min(x_vals)),
            "y_span": float(np.max(y_vals) - np.min(y_vals)),
            "points_zyx": points_zyx,
        }

    @staticmethod
    def _score_wrap_xy_distance_in_band(
        points_zyx: np.ndarray,
        band_z_min: float,
        band_z_max: float,
        center_y: float,
        center_x: float,
        min_band_points: int,
    ):
        """Return robust in-band XY distance score and robust in-band XY center."""
        if points_zyx is None or points_zyx.size == 0:
            return None
        z_vals = points_zyx[:, 0]
        in_band = (z_vals >= band_z_min) & (z_vals <= band_z_max)
        if int(np.count_nonzero(in_band)) < int(min_band_points):
            return None
        pts = points_zyx[in_band]
        if pts.size == 0:
            return None
        y_vals = pts[:, 1]
        x_vals = pts[:, 2]
        dy = y_vals - center_y
        dx = x_vals - center_x
        radial = np.sqrt((dy * dy) + (dx * dx))
        if radial.size == 0 or not np.isfinite(radial).any():
            return None
        score = float(np.median(radial))
        xy_center = np.array([float(np.median(y_vals)), float(np.median(x_vals))], dtype=np.float64)
        if not np.isfinite(score) or not np.isfinite(xy_center).all():
            return None
        return {
            "score": score,
            "xy_center": xy_center,
            "points_in_band": int(pts.shape[0]),
        }

    @staticmethod
    def _volume_center_zyx(patch: ChunkPatch) -> np.ndarray:
        """Global center of the source volume used by this patch, in ZYX coordinates."""
        volume = patch.volume
        if isinstance(volume, zarr.Group):
            volume = volume[str(patch.scale)]
        shape_zyx = np.array(volume.shape, dtype=np.float64)
        return 0.5 * np.maximum(shape_zyx - 1.0, 0.0)

    def _build_triplet_neighbor_lookup(self):
        """Build (patch_idx, wrap_idx) -> neighbor-wrap metadata for triplet mode.

        Adjacent neighbors are chosen from the per-segment wrap ordering. Then
        we orient them so "front" is always the adjacent neighbor closer to the
        volume center within the local Z band occupied by the candidate triplet,
        and "behind" is the farther one.
        """
        lookup = {}
        abs_margin = max(0.0, float(self.config.get("triplet_order_abs_margin", 0.25)))
        rel_margin = max(0.0, float(self.config.get("triplet_order_rel_margin", 0.08)))
        warn_drop_fraction = float(self.config.get("triplet_order_warn_drop_fraction", 0.05))
        warn_drop_fraction = float(np.clip(warn_drop_fraction, 0.0, 1.0))
        min_band_points = max(1, int(self.config.get("triplet_order_min_band_points", 32)))
        debug_examples = max(0, int(self.config.get("triplet_order_debug_examples", 0)))
        debug_lines = []
        order_stats = {
            "candidate_triplets": 0,
            "kept_triplets": 0,
            "dropped_ambiguous": 0,
            "dropped_missing_band_points": 0,
        }
        sep_values = []
        for patch_idx, patch in enumerate(self.patches):
            vol_center_zyx = self._volume_center_zyx(patch)
            wraps_by_segment = {}
            for wrap_idx, wrap in enumerate(patch.wraps):
                wraps_by_segment.setdefault(wrap["segment_idx"], []).append((wrap_idx, wrap))

            for segment_idx in sorted(wraps_by_segment):
                wraps_in_seg = sorted(wraps_by_segment[segment_idx], key=lambda x: x[0])
                if len(wraps_in_seg) < 3:
                    continue

                wrap_stats = []
                for wrap_idx, wrap in wraps_in_seg:
                    s = self._compute_wrap_order_stats(wrap)
                    if s is None:
                        continue
                    wrap_stats.append({
                        "wrap_idx": wrap_idx,
                        "z_median": s["z_median"],
                        "z_min": s["z_min"],
                        "z_max": s["z_max"],
                        "x_median": s["x_median"],
                        "y_median": s["y_median"],
                        "x_span": s["x_span"],
                        "y_span": s["y_span"],
                        "points_zyx": s["points_zyx"],
                    })

                if len(wrap_stats) < 3:
                    continue

                x_spans = np.array([s["x_span"] for s in wrap_stats], dtype=np.float32)
                y_spans = np.array([s["y_span"] for s in wrap_stats], dtype=np.float32)
                principal_axis = "x" if float(np.median(x_spans)) >= float(np.median(y_spans)) else "y"
                order_axis = "y" if principal_axis == "x" else "x"

                if order_axis == "x":
                    ordered = sorted(wrap_stats, key=lambda s: (s["x_median"], s["wrap_idx"]))
                else:
                    ordered = sorted(wrap_stats, key=lambda s: (s["y_median"], s["wrap_idx"]))

                for pos in range(1, len(ordered) - 1):
                    order_stats["candidate_triplets"] += 1
                    target = ordered[pos]
                    prev_neighbor = ordered[pos - 1]
                    next_neighbor = ordered[pos + 1]

                    # Use a z-band-aware center reference: clamp the global center Z
                    # into the local Z band occupied by this triplet's wraps.
                    band_z_min = min(
                        target["z_min"], prev_neighbor["z_min"], next_neighbor["z_min"]
                    )
                    band_z_max = max(
                        target["z_max"], prev_neighbor["z_max"], next_neighbor["z_max"]
                    )
                    center_z = float(np.clip(vol_center_zyx[0], band_z_min, band_z_max))
                    prev_score = self._score_wrap_xy_distance_in_band(
                        prev_neighbor["points_zyx"],
                        band_z_min=band_z_min,
                        band_z_max=band_z_max,
                        center_y=float(vol_center_zyx[1]),
                        center_x=float(vol_center_zyx[2]),
                        min_band_points=min_band_points,
                    )
                    next_score = self._score_wrap_xy_distance_in_band(
                        next_neighbor["points_zyx"],
                        band_z_min=band_z_min,
                        band_z_max=band_z_max,
                        center_y=float(vol_center_zyx[1]),
                        center_x=float(vol_center_zyx[2]),
                        min_band_points=min_band_points,
                    )
                    if prev_score is None or next_score is None:
                        order_stats["dropped_missing_band_points"] += 1
                        if len(debug_lines) < debug_examples:
                            debug_lines.append(
                                f"drop missing-band patch={patch_idx} seg={segment_idx} "
                                f"target={target['wrap_idx']} prev={prev_neighbor['wrap_idx']} next={next_neighbor['wrap_idx']} "
                                f"z_band=[{band_z_min:.2f},{band_z_max:.2f}] center_z={center_z:.2f}"
                            )
                        continue

                    sep_xy = float(np.linalg.norm(prev_score["xy_center"] - next_score["xy_center"]))
                    if np.isfinite(sep_xy):
                        sep_values.append(sep_xy)
                    else:
                        sep_xy = 0.0
                    margin = max(abs_margin, rel_margin * sep_xy)
                    gap = abs(prev_score["score"] - next_score["score"])

                    if gap <= margin:
                        order_stats["dropped_ambiguous"] += 1
                        if len(debug_lines) < debug_examples:
                            debug_lines.append(
                                f"drop ambiguous patch={patch_idx} seg={segment_idx} target={target['wrap_idx']} "
                                f"prev={prev_neighbor['wrap_idx']} next={next_neighbor['wrap_idx']} "
                                f"prev_score={prev_score['score']:.3f} next_score={next_score['score']:.3f} "
                                f"gap={gap:.3f} margin={margin:.3f} sep_xy={sep_xy:.3f}"
                            )
                        continue

                    # Enforce orientation: front is the adjacent neighbor closer
                    # to the z-band-aware XY volume center, behind is farther.
                    if prev_score["score"] < next_score["score"]:
                        front = prev_neighbor
                        behind = next_neighbor
                        front_score = prev_score["score"]
                        behind_score = next_score["score"]
                    elif next_score["score"] < prev_score["score"]:
                        front = next_neighbor
                        behind = prev_neighbor
                        front_score = next_score["score"]
                        behind_score = prev_score["score"]
                    else:
                        order_stats["dropped_ambiguous"] += 1
                        continue
                    lookup[(patch_idx, target["wrap_idx"])] = {
                        "behind_wrap_idx": behind["wrap_idx"],
                        "front_wrap_idx": front["wrap_idx"],
                        "principal_axis": principal_axis,
                        "order_axis": order_axis,
                        "front_score": float(front_score),
                        "behind_score": float(behind_score),
                        "decision_gap": float(gap),
                        "decision_margin": float(margin),
                        "band_z_min": float(band_z_min),
                        "band_z_max": float(band_z_max),
                    }
                    order_stats["kept_triplets"] += 1
        dropped_total = int(order_stats["candidate_triplets"] - order_stats["kept_triplets"])
        drop_fraction = (
            float(dropped_total) / float(order_stats["candidate_triplets"])
            if order_stats["candidate_triplets"] > 0 else 0.0
        )
        sep_arr = np.asarray(sep_values, dtype=np.float32)
        sep_p10 = float(np.percentile(sep_arr, 10)) if sep_arr.size > 0 else 0.0
        sep_p50 = float(np.percentile(sep_arr, 50)) if sep_arr.size > 0 else 0.0
        sep_p90 = float(np.percentile(sep_arr, 90)) if sep_arr.size > 0 else 0.0
        order_stats["drop_fraction"] = drop_fraction
        order_stats["sep_xy_p10"] = sep_p10
        order_stats["sep_xy_p50"] = sep_p50
        order_stats["sep_xy_p90"] = sep_p90
        self._triplet_lookup_stats = order_stats
        if dropped_total > 0:
            base_msg = (
                "Triplet ordering dropped candidates: "
                f"kept={order_stats['kept_triplets']}/{order_stats['candidate_triplets']} "
                f"(drop={drop_fraction * 100.0:.2f}%), "
                f"ambiguous={order_stats['dropped_ambiguous']}, "
                f"missing_band_points={order_stats['dropped_missing_band_points']}, "
                f"sep_xy[p10/p50/p90]=[{sep_p10:.2f}, {sep_p50:.2f}, {sep_p90:.2f}]"
            )
            warnings.warn(base_msg, RuntimeWarning)
            if drop_fraction >= warn_drop_fraction:
                warnings.warn(
                    "Triplet ordering drop fraction exceeded warning threshold: "
                    f"drop={drop_fraction * 100.0:.2f}% threshold={warn_drop_fraction * 100.0:.2f}%",
                    RuntimeWarning,
                )
        if debug_lines:
            for line in debug_lines:
                print(f"[triplet-order-debug] {line}")
        return lookup

    def _parse_cond_percent(self):
        spec = self.config['cond_percent']
        if not isinstance(spec, (list, tuple)) or len(spec) != 2:
            raise ValueError("cond_percent must be [min, max], e.g. [0.1, 0.5]")

        low, high = float(spec[0]), float(spec[1])
        if not (0.0 < low <= high < 1.0):
            raise ValueError("cond_percent values must satisfy 0 < min <= max < 1")
        return low, high

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

        x_full, y_full, z_full = self._upsample_world_triplet(x_s, y_s, z_s, scale_y, scale_x)
        trimmed = self._trim_to_world_bbox(x_full, y_full, z_full, patch.world_bbox)
        if trimmed is None:
            return None
        x_full, y_full, z_full = trimmed
        return np.stack([z_full, y_full, x_full], axis=-1)

    def _extract_wrap_world_surface_cached(self, patch_idx: int, wrap_idx: int, require_all_valid: bool = True):
        """LRU-cache extracted wrap world surfaces for triplet mode."""
        patch = self.patches[patch_idx]
        wrap = patch.wraps[wrap_idx]

        if self._triplet_surface_cache_max_items <= 0:
            return self._extract_wrap_world_surface(patch, wrap, require_all_valid=require_all_valid)

        cache_key = (patch_idx, wrap_idx, int(bool(require_all_valid)))
        cached = self._triplet_surface_cache.get(cache_key)
        if cached is not None:
            self._triplet_surface_cache.move_to_end(cache_key)
            return cached

        surface_zyxs = self._extract_wrap_world_surface(patch, wrap, require_all_valid=require_all_valid)
        if surface_zyxs is None:
            return None

        self._triplet_surface_cache[cache_key] = surface_zyxs
        if len(self._triplet_surface_cache) > self._triplet_surface_cache_max_items:
            self._triplet_surface_cache.popitem(last=False)
        return surface_zyxs

    @staticmethod
    def _read_volume_crop_from_patch(patch: ChunkPatch, crop_size, min_corner, max_corner):
        volume = patch.volume
        if isinstance(volume, zarr.Group):
            volume = volume[str(patch.scale)]

        vol_crop = np.zeros(crop_size, dtype=volume.dtype)
        vol_shape = volume.shape
        src_starts = np.maximum(min_corner, 0)
        src_ends = np.minimum(max_corner, np.array(vol_shape, dtype=np.int64))
        dst_starts = src_starts - min_corner
        dst_ends = dst_starts + (src_ends - src_starts)

        if np.all(src_ends > src_starts):
            vol_crop[
                dst_starts[0]:dst_ends[0],
                dst_starts[1]:dst_ends[1],
                dst_starts[2]:dst_ends[2],
            ] = volume[
                src_starts[0]:src_ends[0],
                src_starts[1]:src_ends[1],
                src_starts[2]:src_ends[2],
            ]
        return normalize_zscore(vol_crop)

    def _load_volume_crop_from_patch(
        self,
        patch: ChunkPatch,
        crop_size,
        min_corner,
        max_corner,
        cache_key=None,
    ):
        use_cache = (
            self._enable_volume_crop_cache and
            self._volume_crop_cache_max_items > 0 and
            cache_key is not None
        )
        if not use_cache:
            return self._read_volume_crop_from_patch(patch, crop_size, min_corner, max_corner)

        cached = self._volume_crop_cache.get(cache_key)
        if cached is not None:
            self._volume_crop_cache.move_to_end(cache_key)
            return cached.copy()

        vol_crop = self._read_volume_crop_from_patch(patch, crop_size, min_corner, max_corner)
        self._volume_crop_cache[cache_key] = vol_crop
        if len(self._volume_crop_cache) > self._volume_crop_cache_max_items:
            self._volume_crop_cache.popitem(last=False)
        return vol_crop.copy()

    def _validate_result_tensors(self, result: dict, idx: int):
        if not self._validate_result_tensors_enabled:
            return True
        for key, tensor in result.items():
            if tensor.numel() == 0:
                print(f"WARNING: Empty tensor for '{key}' at index {idx}, resampling...")
                return False
            if not np.isfinite(tensor.numpy()).all():
                print(f"WARNING: Non-finite values in '{key}' at index {idx}, resampling...")
                return False
        return True

    def _getitem_triplet_wrap_displacement(self, idx: int, patch_idx: int, wrap_idx: int):
        patch = self.patches[patch_idx]
        triplet_meta = self._triplet_neighbor_lookup.get((patch_idx, wrap_idx))
        if triplet_meta is None:
            return self[np.random.randint(len(self))]

        center_zyxs = self._extract_wrap_world_surface_cached(patch_idx, wrap_idx, require_all_valid=True)
        behind_zyxs = self._extract_wrap_world_surface_cached(
            patch_idx, triplet_meta["behind_wrap_idx"], require_all_valid=True
        )
        front_zyxs = self._extract_wrap_world_surface_cached(
            patch_idx, triplet_meta["front_wrap_idx"], require_all_valid=True
        )
        if center_zyxs is None or behind_zyxs is None or front_zyxs is None:
            return self[np.random.randint(len(self))]

        center_zyxs_unperturbed = center_zyxs
        center_zyxs_perturbed = self._maybe_perturb_conditioning_surface(center_zyxs_unperturbed)

        crop_size = self.crop_size
        z_min, _, y_min, _, x_min, _ = patch.world_bbox
        min_corner = np.round([z_min, y_min, x_min]).astype(np.int64)
        max_corner = min_corner + np.array(crop_size)
        vol_cache_key = (
            patch_idx,
            int(min_corner[0]), int(min_corner[1]), int(min_corner[2]),
            int(crop_size[0]), int(crop_size[1]), int(crop_size[2]),
        )

        vol_crop = self._load_volume_crop_from_patch(
            patch,
            crop_size,
            min_corner,
            max_corner,
            cache_key=vol_cache_key,
        )

        center_local_gt = (center_zyxs_unperturbed - min_corner).astype(np.float64)
        behind_local = (behind_zyxs - min_corner).astype(np.float64)
        front_local = (front_zyxs - min_corner).astype(np.float64)

        center_seg_gt = voxelize_surface_grid(center_local_gt, crop_size).astype(np.float32)
        if center_zyxs_perturbed is center_zyxs_unperturbed:
            # Common path: perturbation skipped, so cond and GT segmentations are identical.
            center_seg = center_seg_gt.copy()
        else:
            center_local = (center_zyxs_perturbed - min_corner).astype(np.float64)
            center_seg = voxelize_surface_grid(center_local, crop_size).astype(np.float32)
        behind_seg = voxelize_surface_grid(behind_local, crop_size).astype(np.float32)
        front_seg = voxelize_surface_grid(front_local, crop_size).astype(np.float32)

        if not center_seg.any() or not center_seg_gt.any() or not behind_seg.any() or not front_seg.any():
            return self[np.random.randint(len(self))]

        masked_seg = np.maximum(behind_seg, front_seg).astype(np.float32)

        vol_crop = torch.from_numpy(vol_crop).to(torch.float32)
        center_seg = torch.from_numpy(center_seg).to(torch.float32)
        center_seg_gt = torch.from_numpy(center_seg_gt).to(torch.float32)
        behind_seg = torch.from_numpy(behind_seg).to(torch.float32)
        front_seg = torch.from_numpy(front_seg).to(torch.float32)
        masked_seg = torch.from_numpy(masked_seg).to(torch.float32)

        if self._augmentations is not None:
            seg_list = torch.stack([masked_seg, center_seg, center_seg_gt, behind_seg, front_seg], dim=0)
            augmented = self._augmentations(
                image=vol_crop[None],
                segmentation=seg_list,
                crop_shape=crop_size,
            )
            vol_crop = augmented["image"].squeeze(0)
            masked_seg = augmented["segmentation"][0]
            center_seg = augmented["segmentation"][1]
            center_seg_gt = augmented["segmentation"][2]
            behind_seg = augmented["segmentation"][3]
            front_seg = augmented["segmentation"][4]

        cond_np = center_seg_gt.numpy()
        behind_np = behind_seg.numpy()
        front_np = front_seg.numpy()

        weight_mode = str(self.config.get("triplet_dense_weight_mode", "band")).lower()
        need_neighbor_distances = weight_mode == "band"
        cond_bin_full = cond_np > 0.5
        behind_bin_full = behind_np > 0.5
        front_bin_full = front_np > 0.5

        full_slices = (
            slice(0, int(crop_size[0])),
            slice(0, int(crop_size[1])),
            slice(0, int(crop_size[2])),
        )
        use_dense_roi = bool(self.config.get("triplet_dense_roi_enabled", True)) and weight_mode != "all"
        dense_roi_slices = full_slices
        dense_roi_bounds = None
        roi_pad = max(0, int(round(float(self.config.get("triplet_dense_roi_padding_voxels", 8.0)))))
        if use_dense_roi:
            union_bin = cond_bin_full | behind_bin_full | front_bin_full
            dense_roi_bounds = self._mask_bounds_zyx(union_bin)
            if dense_roi_bounds is None:
                return self[np.random.randint(len(self))]
            dense_roi_slices = self._bounds_to_slices(dense_roi_bounds, roi_pad, crop_size)

        band_padding = max(0.0, float(self.config.get("triplet_band_padding_voxels", 4.0)))
        band_pct = float(self.config.get("triplet_band_distance_percentile", 95.0))
        band_pct = min(100.0, max(1.0, band_pct))

        adaptive_roi_for_band = (
            need_neighbor_distances and
            use_dense_roi and
            bool(self.config.get("triplet_dense_roi_adaptive_padding", True))
        )
        roi_pad_max = max(int(crop_size[0]), int(crop_size[1]), int(crop_size[2]))
        if adaptive_roi_for_band:
            roi_pad_max = max(
                roi_pad,
                int(round(float(self.config.get("triplet_dense_roi_max_padding_voxels", float(roi_pad_max)))))
            )

        while True:
            behind_for_disp = behind_np[dense_roi_slices] if use_dense_roi else behind_np
            front_for_disp = front_np[dense_roi_slices] if use_dense_roi else front_np
            if need_neighbor_distances:
                behind_disp_work, _, d_behind_work = self._compute_dense_displacement_field(
                    behind_for_disp,
                    return_weights=False,
                    return_distances=True,
                )
                front_disp_work, _, d_front_work = self._compute_dense_displacement_field(
                    front_for_disp,
                    return_weights=False,
                    return_distances=True,
                )
            else:
                behind_disp_work, _ = self._compute_dense_displacement_field(behind_for_disp, return_weights=False)
                front_disp_work, _ = self._compute_dense_displacement_field(front_for_disp, return_weights=False)
                d_behind_work = None
                d_front_work = None

            if behind_disp_work is None or front_disp_work is None:
                return self[np.random.randint(len(self))]
            if not adaptive_roi_for_band:
                break

            cond_bin_roi = cond_bin_full[dense_roi_slices]
            if cond_bin_roi.sum() == 0 or d_behind_work is None or d_front_work is None:
                return self[np.random.randint(len(self))]
            cond_mask_roi = cond_bin_roi > 0
            cond_to_front = d_front_work[cond_mask_roi]
            cond_to_behind = d_behind_work[cond_mask_roi]
            if cond_to_front.size == 0 or cond_to_behind.size == 0:
                return self[np.random.randint(len(self))]

            front_radius = float(np.percentile(cond_to_front, band_pct)) + band_padding
            behind_radius = float(np.percentile(cond_to_behind, band_pct)) + band_padding
            required_pad = int(np.ceil(max(front_radius, behind_radius)))
            next_pad = min(roi_pad_max, max(roi_pad, required_pad))
            if next_pad <= roi_pad:
                break
            roi_pad = next_pad
            dense_roi_slices = self._bounds_to_slices(dense_roi_bounds, roi_pad, crop_size)

        if use_dense_roi:
            behind_disp_np = np.zeros((3, *crop_size), dtype=np.float32)
            front_disp_np = np.zeros((3, *crop_size), dtype=np.float32)
            roi_index = (slice(None),) + dense_roi_slices
            behind_disp_np[roi_index] = behind_disp_work.astype(np.float32, copy=False)
            front_disp_np[roi_index] = front_disp_work.astype(np.float32, copy=False)
        else:
            behind_disp_np = behind_disp_work.astype(np.float32, copy=False)
            front_disp_np = front_disp_work.astype(np.float32, copy=False)

        if weight_mode == "conditioning":
            dense_weight_np = (cond_np > 0.5).astype(np.float32, copy=False)[None]
        elif weight_mode == "neighbors":
            dense_weight_np = (np.maximum(behind_np, front_np) > 0.5).astype(np.float32, copy=False)[None]
        elif weight_mode == "all":
            dense_weight_np = np.ones((1, *crop_size), dtype=np.float32)
        elif weight_mode == "band":
            if d_behind_work is None or d_front_work is None:
                return self[np.random.randint(len(self))]

            cond_bin = cond_bin_full[dense_roi_slices].astype(np.uint8, copy=False) if use_dense_roi \
                else cond_bin_full.astype(np.uint8, copy=False)
            if cond_bin.sum() == 0:
                return self[np.random.randint(len(self))]

            cond_mask = cond_bin > 0
            cond_to_front = d_front_work[cond_mask]
            cond_to_behind = d_behind_work[cond_mask]
            if cond_to_front.size == 0 or cond_to_behind.size == 0:
                return self[np.random.randint(len(self))]

            # Build a dense slab between front/back using displacement geometry:
            # inside points tend to have front/back displacement vectors pointing
            # in opposite directions (non-positive dot product).
            d_sum_work = d_front_work + d_behind_work
            cond_sum = (cond_to_front + cond_to_behind).astype(np.float32, copy=False)
            if cond_sum.size == 0:
                return self[np.random.randint(len(self))]

            sum_threshold = float(np.percentile(cond_sum, band_pct)) + (2.0 * band_padding)
            vector_dot = np.sum(front_disp_work * behind_disp_work, axis=0, dtype=np.float32)
            dense_band = (vector_dot <= 0.0) & (d_sum_work <= sum_threshold)
            if not dense_band.any():
                return self[np.random.randint(len(self))]

            cc_structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connected neighborhood

            # Remove isolated islands: keep only components connected to conditioning.
            labels, num_labels = ndimage.label(dense_band, structure=cc_structure)
            if num_labels <= 0:
                return self[np.random.randint(len(self))]
            touching = np.unique(labels[cond_mask])
            touching = touching[touching > 0]
            if touching.size == 0:
                return self[np.random.randint(len(self))]
            keep = np.zeros(num_labels + 1, dtype=bool)
            keep[touching] = True

            dense_band = keep[labels]
            # Fill tiny holes inside the slab.
            dense_band = ndimage.binary_closing(
                dense_band,
                structure=np.ones((3, 3, 3), dtype=bool),
                iterations=1,
            )
            if not dense_band.any():
                return self[np.random.randint(len(self))]

            # Closing can create small detached islands; keep only cond-connected components.
            labels, num_labels = ndimage.label(dense_band, structure=cc_structure)
            if num_labels <= 0:
                return self[np.random.randint(len(self))]
            touching = np.unique(labels[cond_mask])
            touching = touching[touching > 0]
            if touching.size == 0:
                return self[np.random.randint(len(self))]
            keep = np.zeros(num_labels + 1, dtype=bool)
            keep[touching] = True
            dense_band = keep[labels].astype(np.float32, copy=False)
            if use_dense_roi:
                dense_weight_np = np.zeros((1, *crop_size), dtype=np.float32)
                dense_weight_np[(0,) + dense_roi_slices] = dense_band
            else:
                dense_weight_np = dense_band[None]
        else:
            raise ValueError(
                "triplet_dense_weight_mode must be one of {'conditioning', 'neighbors', 'all', 'band'}, "
                f"got {weight_mode!r}"
            )
        if float(dense_weight_np.sum()) <= 0:
            return self[np.random.randint(len(self))]

        dense_gt_np = np.concatenate([behind_disp_np, front_disp_np], axis=0).astype(np.float32, copy=False)
        dense_gt_disp = torch.from_numpy(dense_gt_np).to(torch.float32)
        dense_loss_weight = torch.from_numpy(dense_weight_np).to(torch.float32)

        result = {
            "vol": vol_crop,
            "cond": center_seg,
            "masked_seg": masked_seg,
            "dense_gt_displacement": dense_gt_disp,  # (6, D, H, W): behind(dz,dy,dx), front(dz,dy,dx)
            "dense_loss_weight": dense_loss_weight,  # (1, D, H, W)
        }
        if not self._validate_result_tensors(result, idx):
            return self[np.random.randint(len(self))]
        return result

    @staticmethod
    def _coords_in_bounds(coords_zyx: torch.Tensor, shape_zyx) -> torch.Tensor:
        d, h, w = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
        return (
            (coords_zyx[:, 0] >= 0) & (coords_zyx[:, 0] <= d - 1) &
            (coords_zyx[:, 1] >= 0) & (coords_zyx[:, 1] <= h - 1) &
            (coords_zyx[:, 2] >= 0) & (coords_zyx[:, 2] <= w - 1)
        )

    @staticmethod
    def _upsample_world_triplet(x_s, y_s, z_s, scale_y: float, scale_x: float):
        """Upsample (x, y, z) sampled grids in one cv2 call."""
        h_s, w_s = x_s.shape
        h_up = int(round(h_s / scale_y))
        w_up = int(round(w_s / scale_x))
        xyz_s = np.stack([x_s, y_s, z_s], axis=-1)
        xyz_up = cv2.resize(xyz_s, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
        return xyz_up[..., 0], xyz_up[..., 1], xyz_up[..., 2]

    @staticmethod
    def _trim_to_world_bbox(x_full, y_full, z_full, world_bbox):
        """Keep the minimal row/col slab that intersects the world bbox."""
        z_min, z_max, y_min, y_max, x_min, x_max = world_bbox
        in_bounds = (
            (z_full >= z_min) & (z_full < z_max) &
            (y_full >= y_min) & (y_full < y_max) &
            (x_full >= x_min) & (x_full < x_max)
        )
        if not in_bounds.any():
            return None

        valid_rows = np.any(in_bounds, axis=1)
        valid_cols = np.any(in_bounds, axis=0)
        row_idx = np.flatnonzero(valid_rows)
        col_idx = np.flatnonzero(valid_cols)
        if row_idx.size == 0 or col_idx.size == 0:
            return None

        r0, r1 = int(row_idx[0]), int(row_idx[-1])
        c0, c1 = int(col_idx[0]), int(col_idx[-1])
        return (
            x_full[r0:r1 + 1, c0:c1 + 1],
            y_full[r0:r1 + 1, c0:c1 + 1],
            z_full[r0:r1 + 1, c0:c1 + 1],
        )

    @staticmethod
    def _mask_bounds_zyx(mask: np.ndarray):
        """Return inclusive (z0, z1, y0, y1, x0, x1) for True voxels, or None."""
        if mask.size == 0:
            return None
        z_any = np.any(mask, axis=(1, 2))
        y_any = np.any(mask, axis=(0, 2))
        x_any = np.any(mask, axis=(0, 1))
        if not z_any.any() or not y_any.any() or not x_any.any():
            return None
        z_idx = np.flatnonzero(z_any)
        y_idx = np.flatnonzero(y_any)
        x_idx = np.flatnonzero(x_any)
        return (
            int(z_idx[0]), int(z_idx[-1]),
            int(y_idx[0]), int(y_idx[-1]),
            int(x_idx[0]), int(x_idx[-1]),
        )

    @staticmethod
    def _bounds_to_slices(bounds_zyx, pad_voxels: int, shape_zyx):
        if bounds_zyx is None:
            return (
                slice(0, int(shape_zyx[0])),
                slice(0, int(shape_zyx[1])),
                slice(0, int(shape_zyx[2])),
            )
        z0, z1, y0, y1, x0, x1 = bounds_zyx
        pad = max(0, int(pad_voxels))
        d, h, w = int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2])
        return (
            slice(max(0, z0 - pad), min(d, z1 + pad + 1)),
            slice(max(0, y0 - pad), min(h, y1 + pad + 1)),
            slice(max(0, x0 - pad), min(w, x1 + pad + 1)),
        )

    @staticmethod
    def _compute_surface_normals(surface_zyxs: np.ndarray) -> np.ndarray:
        """Estimate per-point unit normals from local row/col tangents."""
        h, w, _ = surface_zyxs.shape
        if h < 2 or w < 2:
            return np.zeros_like(surface_zyxs, dtype=np.float32)

        surface = surface_zyxs.astype(np.float32, copy=False)
        row_tangent = np.empty_like(surface)
        col_tangent = np.empty_like(surface)

        row_tangent[1:-1] = surface[2:] - surface[:-2]
        row_tangent[0] = surface[1] - surface[0]
        row_tangent[-1] = surface[-1] - surface[-2]

        col_tangent[:, 1:-1] = surface[:, 2:] - surface[:, :-2]
        col_tangent[:, 0] = surface[:, 1] - surface[:, 0]
        col_tangent[:, -1] = surface[:, -1] - surface[:, -2]

        normals = np.cross(col_tangent, row_tangent)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-6)
        normals[norms[..., 0] <= 1e-6] = 0.0
        return normals.astype(np.float32, copy=False)

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
    ):
        """Compute nearest-surface displacement vector for every voxel.

        Args:
            full_surface_mask: (D, H, W) binary/boolean mask of GT surface voxels.
            return_distances: whether to also return nearest-surface Euclidean distance map.

        Returns:
            disp_field: (3, D, H, W) with components (dz, dy, dx)
            weight_mask: (1, D, H, W) per-voxel supervision weights
            distance_map: (D, H, W) nearest-surface distance (optional)
        """
        surface = full_surface_mask > 0.5
        if not surface.any():
            if return_distances:
                return None, None, None
            return None, None

        # Distance transform on inverse mask gives nearest foreground (surface) index per voxel.
        edt_out = ndimage.distance_transform_edt(
            ~surface, return_distances=return_distances, return_indices=True
        )
        if return_distances:
            distances, nearest_idx = edt_out
            distances = distances.astype(np.float32, copy=False)
        else:
            nearest_idx = edt_out
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

    def _maybe_perturb_conditioning_surface(self, cond_zyxs: np.ndarray) -> np.ndarray:
        """Apply local normal-direction pushes with Gaussian falloff on small regions."""
        cfg = self.config.get('cond_local_perturb', {})
        apply_without_aug = bool(cfg.get('apply_without_augmentation', False))
        if (not self.apply_augmentation and not apply_without_aug) or not cfg.get('enabled', True):
            return cond_zyxs
        if random.random() >= float(cfg.get('probability', 0.35)):
            return cond_zyxs

        cond_h, cond_w, _ = cond_zyxs.shape
        if cond_h < 2 or cond_w < 2:
            return cond_zyxs

        normals = self._compute_surface_normals(cond_zyxs)
        valid_normal_idx = np.argwhere(np.linalg.norm(normals, axis=-1) > 1e-6)
        if len(valid_normal_idx) == 0:
            return cond_zyxs

        blob_cfg = cfg.get('num_blobs', [1, 3])
        if isinstance(blob_cfg, (list, tuple)) and len(blob_cfg) == 2:
            min_blobs = max(1, int(blob_cfg[0]))
            max_blobs = max(min_blobs, int(blob_cfg[1]))
        else:
            min_blobs = max_blobs = 1
        n_blobs = random.randint(min_blobs, max_blobs)

        sigma_cfg = cfg.get('sigma_fraction_range', [0.04, 0.10])
        if isinstance(sigma_cfg, (list, tuple)) and len(sigma_cfg) == 2:
            sigma_lo_frac = max(0.01, float(sigma_cfg[0]))
            sigma_hi_frac = max(sigma_lo_frac, float(sigma_cfg[1]))
        else:
            sigma_lo_frac, sigma_hi_frac = 0.04, 0.10
        sigma_scale = float(min(cond_h, cond_w))
        sigma_lo = max(0.3, sigma_lo_frac * sigma_scale)
        sigma_hi = max(sigma_lo, sigma_hi_frac * sigma_scale)

        amp_cfg = cfg.get('amplitude_range', [0.25, 1.25])
        if isinstance(amp_cfg, (list, tuple)) and len(amp_cfg) == 2:
            amp_lo = max(0.0, float(amp_cfg[0]))
            amp_hi = max(amp_lo, float(amp_cfg[1]))
        else:
            amp_lo, amp_hi = 0.25, 1.25

        radius_sigma_mult = max(0.5, float(cfg.get('radius_sigma_mult', 2.5)))
        max_total_disp = max(0.0, float(cfg.get('max_total_displacement', 1.5)))
        if max_total_disp <= 0.0:
            return cond_zyxs

        rr, cc = np.meshgrid(np.arange(cond_h), np.arange(cond_w), indexing='ij')
        disp_along_normal = np.zeros((cond_h, cond_w), dtype=np.float32)
        points_affected = int(cfg.get('points_affected', 10))
        use_k_neighborhood = points_affected > 0

        for _ in range(n_blobs):
            seed_r, seed_c = valid_normal_idx[np.random.randint(len(valid_normal_idx))]

            dr = rr - float(seed_r)
            dc = cc - float(seed_c)
            dist2 = dr * dr + dc * dc

            if use_k_neighborhood:
                flat_dist2 = dist2.reshape(-1)
                k = min(points_affected, flat_dist2.size)
                if k <= 0:
                    continue
                kth_idx = np.argpartition(flat_dist2, k - 1)[k - 1]
                radius2 = float(flat_dist2[kth_idx])
                local_mask = dist2 <= radius2
                sigma = max(0.3, np.sqrt(max(radius2, 1e-6)) / max(radius_sigma_mult, 1e-3))
            else:
                sigma = random.uniform(sigma_lo, sigma_hi)
                radius2 = (radius_sigma_mult * sigma) ** 2
                local_mask = dist2 <= radius2

            if not np.any(local_mask):
                continue

            amp = random.uniform(amp_lo, amp_hi)
            signed_amp = amp if random.random() < 0.5 else -amp
            falloff = np.exp(-0.5 * dist2 / max(sigma * sigma, 1e-6))
            disp_along_normal[local_mask] += (signed_amp * falloff[local_mask]).astype(np.float32)

        if not np.any(disp_along_normal):
            return cond_zyxs

        disp_along_normal = np.clip(disp_along_normal, -max_total_disp, max_total_disp)
        perturbed = cond_zyxs.astype(np.float32, copy=True)
        perturbed += normals * disp_along_normal[..., None]
        return perturbed.astype(cond_zyxs.dtype, copy=False)

    def __getitem__(self, idx):
        patch_idx, wrap_idx = self.sample_index[idx]
        if self.use_triplet_wrap_displacement:
            return self._getitem_triplet_wrap_displacement(idx, patch_idx, wrap_idx)

        patch = self.patches[patch_idx]
        crop_size = self.crop_size  # tuple (D, H, W)
        target_shape = crop_size

        # in wrap mode, use the indexed wrap; in chunk mode, choose randomly (legacy behavior)
        wrap = patch.wraps[wrap_idx] if wrap_idx is not None else random.choice(patch.wraps)
        seg = wrap["segment"]
        r_min, r_max, c_min, c_max = wrap["bbox_2d"]

        # clamp bbox to segment bounds (bbox is inclusive in stored resolution)
        seg_h, seg_w = seg._valid_mask.shape
        r_min = max(0, r_min)
        r_max = min(seg_h - 1, r_max)
        c_min = max(0, c_min)
        c_max = min(seg_w - 1, c_max)
        if r_max < r_min or c_max < c_min:
            return self[np.random.randint(len(self))]

        seg.use_stored_resolution()
        scale_y, scale_x = seg._scale
        x_full_s, y_full_s, z_full_s, valid_full_s = seg[r_min:r_max+1, c_min:c_max+1]

        # if any sample contains an invalid point, just grab a new one
        if not valid_full_s.all():
            return self[np.random.randint(len(self))]

        # upsampling here instead of in the tifxyz module because of the annoyances with 
        # handling coords in dif scales
        x_full, y_full, z_full = self._upsample_world_triplet(x_full_s, y_full_s, z_full_s, scale_y, scale_x)
        trimmed = self._trim_to_world_bbox(x_full, y_full, z_full, patch.world_bbox)
        if trimmed is None:
            return self[np.random.randint(len(self))]
        x_full, y_full, z_full = trimmed
        h_up, w_up = x_full.shape  # update dimensions after crop

        # split into cond and mask on the upsampled grid
        conditioning_percent = random.uniform(self._cond_percent_min, self._cond_percent_max)
        if h_up < 2 and w_up < 2:
            return self[np.random.randint(len(self))]

        valid_directions = []
        if w_up >= 2:
            valid_directions.extend(["left", "right"])
        if h_up >= 2:
            valid_directions.extend(["up", "down"])
        if not valid_directions:
            return self[np.random.randint(len(self))]

        r_cond_up = int(round(h_up * conditioning_percent))
        c_cond_up = int(round(w_up * conditioning_percent))
        if h_up >= 2:
            r_cond_up = min(max(r_cond_up, 1), h_up - 1)
        if w_up >= 2:
            c_cond_up = min(max(c_cond_up, 1), w_up - 1)

        # Split boundaries measured from top/left in the upsampled frame.
        r_split_up_top = r_cond_up
        c_split_up_left = c_cond_up

        cond_direction = random.choice(valid_directions)

        if cond_direction == "left":
            # conditioning is left, mask the right
            x_cond, y_cond, z_cond = x_full[:, :c_split_up_left], y_full[:, :c_split_up_left], z_full[:, :c_split_up_left]
            x_mask, y_mask, z_mask = x_full[:, c_split_up_left:], y_full[:, c_split_up_left:], z_full[:, c_split_up_left:]
            cond_row_off, cond_col_off = 0, 0
            mask_row_off, mask_col_off = 0, c_split_up_left
        elif cond_direction == "right":
            # conditioning is right, mask the left
            c_split_up_left = w_up - c_cond_up
            x_cond, y_cond, z_cond = x_full[:, c_split_up_left:], y_full[:, c_split_up_left:], z_full[:, c_split_up_left:]
            x_mask, y_mask, z_mask = x_full[:, :c_split_up_left], y_full[:, :c_split_up_left], z_full[:, :c_split_up_left]
            cond_row_off, cond_col_off = 0, c_split_up_left
            mask_row_off, mask_col_off = 0, 0
        elif cond_direction == "up":
            # conditioning is up, mask the bottom
            x_cond, y_cond, z_cond = x_full[:r_split_up_top, :], y_full[:r_split_up_top, :], z_full[:r_split_up_top, :]
            x_mask, y_mask, z_mask = x_full[r_split_up_top:, :], y_full[r_split_up_top:, :], z_full[r_split_up_top:, :]
            cond_row_off, cond_col_off = 0, 0
            mask_row_off, mask_col_off = r_split_up_top, 0
        elif cond_direction == "down":
            # conditioning is down, mask the top
            r_split_up_top = h_up - r_cond_up
            x_cond, y_cond, z_cond = x_full[r_split_up_top:, :], y_full[r_split_up_top:, :], z_full[r_split_up_top:, :]
            x_mask, y_mask, z_mask = x_full[:r_split_up_top, :], y_full[:r_split_up_top, :], z_full[:r_split_up_top, :]
            cond_row_off, cond_col_off = r_split_up_top, 0
            mask_row_off, mask_col_off = 0, 0

        cond_h, cond_w = x_cond.shape
        mask_h, mask_w = x_mask.shape
        if cond_h == 0 or cond_w == 0 or mask_h == 0 or mask_w == 0:
            return self[np.random.randint(len(self))]

        uv_cond = np.stack(np.meshgrid(
            np.arange(cond_h) + cond_row_off,
            np.arange(cond_w) + cond_col_off,
            indexing='ij'
        ), axis=-1)

        uv_mask = np.stack(np.meshgrid(
            np.arange(mask_h) + mask_row_off,
            np.arange(mask_w) + mask_col_off,
            indexing='ij'
        ), axis=-1)

        cond_zyxs = np.stack([z_cond, y_cond, x_cond], axis=-1)
        masked_zyxs = np.stack([z_mask, y_mask, x_mask], axis=-1)
        cond_zyxs_unperturbed = cond_zyxs.copy()
        cond_zyxs = self._maybe_perturb_conditioning_surface(cond_zyxs)

        # use world_bbox directly as crop position, this is the crop returned by find_patches
        z_min, z_max, y_min, y_max, x_min, x_max = patch.world_bbox
        min_corner = np.round([z_min, y_min, x_min]).astype(np.int64)
        max_corner = min_corner + np.array(crop_size)
        vol_cache_key = (
            patch_idx,
            int(min_corner[0]), int(min_corner[1]), int(min_corner[2]),
            int(crop_size[0]), int(crop_size[1]), int(crop_size[2]),
        )

        # if we're extrapolating, compute it with the extrapolation module
        if self.config['use_extrapolation']:
            method_name = self.config['extrapolation_method']
            rbf_downsample = int(self.config.get('rbf_downsample_factor', 2))
            edge_downsample_cfg = self.config.get('rbf_edge_downsample_factor', None)
            edge_downsample = rbf_downsample if edge_downsample_cfg is None else int(edge_downsample_cfg)
            selected_downsample = edge_downsample if method_name == 'rbf_edge_only' else rbf_downsample

            extrap_result = compute_extrapolation(
                uv_cond=uv_cond,
                zyx_cond=cond_zyxs,
                uv_mask=uv_mask,
                zyx_mask=masked_zyxs,
                min_corner=min_corner,
                crop_size=crop_size,
                method=method_name,
                downsample_factor=selected_downsample,
                rbf_max_points=self.config.get('rbf_max_points'),
                edge_band_frac=float(self.config.get('rbf_edge_band_frac', 0.10)),
                edge_band_cells=self.config.get('rbf_edge_band_cells'),
                edge_min_points=int(self.config.get('rbf_edge_min_points', 128)),
                cond_direction=cond_direction,
                degrade_prob=self.config.get('extrap_degrade_prob', 0.0),
                degrade_curvature_range=self.config.get('extrap_degrade_curvature_range', (0.001, 0.01)),
                degrade_gradient_range=self.config.get('extrap_degrade_gradient_range', (0.05, 0.2)),
                debug_no_in_bounds=bool(self.config.get('debug_extrapolation_oob', False)),
                debug_no_in_bounds_every=int(self.config.get('debug_extrapolation_oob_every', 100)),
                return_gt_normals=self._needs_point_normals,
            )
            if extrap_result is None:
                return self[np.random.randint(len(self))]
            extrap_surface = extrap_result['extrap_surface']
            extrap_coords_local = extrap_result['extrap_coords_local']
            gt_coords_local = extrap_result['gt_coords_local']
            gt_normals_local = extrap_result.get('gt_normals_local')

        vol_crop = self._load_volume_crop_from_patch(
            patch,
            target_shape,
            min_corner,
            max_corner,
            cache_key=vol_cache_key,
        )

        # convert cond and masked coords to crop-local coords (float for line interpolation)
        cond_zyxs_local_float = (cond_zyxs - min_corner).astype(np.float64)
        cond_zyxs_unperturbed_local_float = (cond_zyxs_unperturbed - min_corner).astype(np.float64)
        masked_zyxs_local_float = (masked_zyxs - min_corner).astype(np.float64)

        crop_shape = target_shape

        # voxelize with line interpolation between adjacent grid points
        cond_segmentation = voxelize_surface_grid(cond_zyxs_local_float, crop_shape)
        cond_segmentation_gt = voxelize_surface_grid(cond_zyxs_unperturbed_local_float, crop_shape)
        masked_segmentation = voxelize_surface_grid(masked_zyxs_local_float, crop_shape)

        # make sure we actually have some conditioning
        if not cond_segmentation.any():
            return self[np.random.randint(len(self))]
        if self.use_dense_displacement and not cond_segmentation_gt.any():
            return self[np.random.randint(len(self))]

        cond_segmentation_raw = cond_segmentation.copy()
        cond_segmentation_gt_raw = cond_segmentation_gt.copy()

        # add thickness to conditioning segmentation via dilation
        use_dilation = self.config.get('use_dilation', False)
        if use_dilation:
            dilation_radius = self.config.get('dilation_radius', 1.0)
            dist_from_cond = edt.edt(1 - cond_segmentation, parallel=1)
            cond_segmentation = (dist_from_cond <= dilation_radius).astype(np.float32)

        use_segmentation = self.config.get('use_segmentation', False)
        use_sdt = self.config['use_sdt']
        use_dense_displacement = self.use_dense_displacement
        full_segmentation = None
        full_segmentation_raw = None
        if use_sdt:
            # combine cond + masked into full segmentation
            full_segmentation = np.maximum(cond_segmentation, masked_segmentation)
        if use_segmentation:
            full_segmentation_raw = np.maximum(cond_segmentation_raw, masked_segmentation)

        if use_sdt:
            # if already dilated, just compute SDT directly; otherwise dilate first
            if use_dilation:
                seg_dilated = full_segmentation
            else:
                dilation_radius = self.config.get('dilation_radius', 1.0)
                distance_from_surface = edt.edt(1 - full_segmentation, parallel=1)
                seg_dilated = (distance_from_surface <= dilation_radius).astype(np.float32)
            sdt = edt.sdf(seg_dilated, parallel=1).astype(np.float32)

        if use_segmentation:
            dilation_radius = self.config.get('dilation_radius', 1.0)
            distance_from_surface = edt.edt(1 - full_segmentation_raw, parallel=1)
            seg_dilated = (distance_from_surface <= dilation_radius).astype(np.float32)
            seg_skel = (distance_from_surface == 0).astype(np.float32)

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
                return self[np.random.randint(len(self))]

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

                    ox_full, oy_full, oz_full = self._upsample_world_triplet(ox_s, oy_s, oz_s, o_scale_y, o_scale_x)
                    trimmed = self._trim_to_world_bbox(
                        ox_full, oy_full, oz_full, (z_min_w, z_max_w, y_min_w, y_max_w, x_min_w, x_max_w)
                    )
                    if trimmed is None:
                        continue
                    ox_full, oy_full, oz_full = trimmed

                    other_zyxs = np.stack([oz_full, oy_full, ox_full], axis=-1)
                    other_zyxs_local = (other_zyxs - min_corner).astype(np.float64)

                    other_vox = voxelize_surface_grid(other_zyxs_local, crop_shape)
                    other_wraps_vox = np.maximum(other_wraps_vox, other_vox)

        vol_crop = torch.from_numpy(vol_crop).to(torch.float32)
        masked_seg = torch.from_numpy(masked_segmentation).to(torch.float32)
        cond_seg = torch.from_numpy(cond_segmentation).to(torch.float32)
        cond_seg_gt = torch.from_numpy(cond_segmentation_gt_raw).to(torch.float32)
        other_wraps_tensor = torch.from_numpy(other_wraps_vox).to(torch.float32)
        if use_segmentation:
            full_seg = torch.from_numpy(seg_dilated).to(torch.float32)
            seg_skel = torch.from_numpy(seg_skel).to(torch.float32)

        use_extrapolation = self.config['use_extrapolation']
        if use_extrapolation:
            sample_coords_local = extrap_coords_local
            target_coords_local = gt_coords_local
            point_weights_local = np.ones(sample_coords_local.shape[0], dtype=np.float32)
            point_normals_local = None
            if self._needs_point_normals:
                if gt_normals_local is None:
                    raise ValueError("Expected gt_normals_local for normal_scalar supervision, got None")
                point_normals_local = gt_normals_local.astype(np.float32, copy=False)

            if self.config.get('supervise_conditioning', False):
                cond_sample_coords_local = (cond_zyxs - min_corner).reshape(-1, 3).astype(np.float32)
                cond_target_coords_local = (cond_zyxs_unperturbed - min_corner).reshape(-1, 3).astype(np.float32)
                cond_weight = max(0.0, float(self.config.get('cond_supervision_weight', 0.1)))
                cond_normals_local = None
                if self._needs_point_normals:
                    cond_normals_grid = self._compute_surface_normals(cond_zyxs_unperturbed)
                    cond_normals_local = cond_normals_grid.reshape(-1, 3).astype(np.float32)

                cond_in_bounds = (
                    (cond_sample_coords_local[:, 0] >= 0) & (cond_sample_coords_local[:, 0] < crop_size[0]) &
                    (cond_sample_coords_local[:, 1] >= 0) & (cond_sample_coords_local[:, 1] < crop_size[1]) &
                    (cond_sample_coords_local[:, 2] >= 0) & (cond_sample_coords_local[:, 2] < crop_size[2]) &
                    (cond_target_coords_local[:, 0] >= 0) & (cond_target_coords_local[:, 0] < crop_size[0]) &
                    (cond_target_coords_local[:, 1] >= 0) & (cond_target_coords_local[:, 1] < crop_size[1]) &
                    (cond_target_coords_local[:, 2] >= 0) & (cond_target_coords_local[:, 2] < crop_size[2])
                )
                cond_sample_coords_local = cond_sample_coords_local[cond_in_bounds]
                cond_target_coords_local = cond_target_coords_local[cond_in_bounds]
                if cond_normals_local is not None:
                    cond_normals_local = cond_normals_local[cond_in_bounds]

                if cond_sample_coords_local.shape[0] > 0:
                    sample_coords_local = np.concatenate([sample_coords_local, cond_sample_coords_local], axis=0)
                    target_coords_local = np.concatenate([target_coords_local, cond_target_coords_local], axis=0)
                    cond_weights = np.full(cond_sample_coords_local.shape[0], cond_weight, dtype=np.float32)
                    point_weights_local = np.concatenate([point_weights_local, cond_weights], axis=0)
                    if point_normals_local is not None and cond_normals_local is not None:
                        point_normals_local = np.concatenate([point_normals_local, cond_normals_local], axis=0)

            extrap_surf = torch.from_numpy(extrap_surface).to(torch.float32)
            extrap_coords = torch.from_numpy(sample_coords_local).to(torch.float32)
            gt_coords = torch.from_numpy(target_coords_local).to(torch.float32)
            point_weights = torch.from_numpy(point_weights_local).to(torch.float32)
            point_normals = None
            if point_normals_local is not None:
                point_normals = torch.from_numpy(point_normals_local).to(torch.float32)
            n_points = len(extrap_coords)

        use_sdt = self.config['use_sdt']
        if use_sdt:
            sdt_tensor = torch.from_numpy(sdt).to(torch.float32)

        if self._augmentations is not None:
            seg_list = [masked_seg, cond_seg, other_wraps_tensor]
            seg_keys = ['masked_seg', 'cond_seg', 'other_wraps']
            if use_dense_displacement:
                seg_list.append(cond_seg_gt)
                seg_keys.append('cond_seg_gt')
            if use_segmentation:
                seg_list.append(full_seg)
                seg_keys.append('full_seg')
                seg_list.append(seg_skel)
                seg_keys.append('seg_skel')
            if use_extrapolation:
                seg_list.append(extrap_surf)
                seg_keys.append('extrap_surf')

            dist_list = []
            dist_keys = []
            if use_sdt:
                dist_list.append(sdt_tensor)
                dist_keys.append('sdt')

            aug_kwargs = {
                'image': vol_crop[None],  # [1, D, H, W]
                'segmentation': torch.stack(seg_list, dim=0),
                'crop_shape': crop_size,
            }
            if dist_list:
                aug_kwargs['dist_map'] = torch.stack(dist_list, dim=0)
            if use_extrapolation:
                # stack both coordinate sets together - they get the same keypoint transform
                # we will split them after augmentation and compute displacement from the difference
                aug_kwargs['keypoints'] = torch.cat([extrap_coords, gt_coords], dim=0)
                if point_normals is not None:
                    aug_kwargs['point_normals'] = point_normals
                    aug_kwargs['vector_keys'] = ['point_normals']
            if use_heatmap:
                aug_kwargs['heatmap_target'] = heatmap_tensor[None]  # (1, D, H, W)
                aug_kwargs['regression_keys'] = ['heatmap_target']

            augmented = self._augmentations(**aug_kwargs)

            vol_crop = augmented['image'].squeeze(0)
            for i, key in enumerate(seg_keys):
                if key == 'masked_seg':
                    masked_seg = augmented['segmentation'][i]
                elif key == 'cond_seg':
                    cond_seg = augmented['segmentation'][i]
                elif key == 'other_wraps':
                    other_wraps_tensor = augmented['segmentation'][i]
                elif key == 'cond_seg_gt':
                    cond_seg_gt = augmented['segmentation'][i]
                elif key == 'full_seg':
                    full_seg = augmented['segmentation'][i]
                elif key == 'seg_skel':
                    seg_skel = augmented['segmentation'][i]
                elif key == 'extrap_surf':
                    extrap_surf = augmented['segmentation'][i]

            if dist_list:
                for i, key in enumerate(dist_keys):
                    if key == 'sdt':
                        sdt_tensor = augmented['dist_map'][i]

            if use_extrapolation:
                all_coords = augmented['keypoints']
                extrap_coords = all_coords[:n_points]
                gt_coords = all_coords[n_points:]
                if point_normals is not None:
                    point_normals = augmented['point_normals']
                # compute displacement AFTER augmentation 
                # both coordinate sets received the same spatial transform, so their
                # difference (displacement) is now in the post-augmentation coordinate system
                gt_disp = gt_coords - extrap_coords
            if use_heatmap:
                heatmap_tensor = augmented['heatmap_target'].squeeze(0)
        else:
            # No augmentation - compute displacement directly from coordinates
            if use_extrapolation:
                gt_disp = gt_coords - extrap_coords

        if use_extrapolation and self.config.get('filter_oob_extrap_points', True):
            in_bounds = self._coords_in_bounds(extrap_coords, vol_crop.shape)
            if not torch.all(in_bounds):
                extrap_coords = extrap_coords[in_bounds]
                gt_disp = gt_disp[in_bounds]
                point_weights = point_weights[in_bounds]
                if point_normals is not None:
                    point_normals = point_normals[in_bounds]
            if extrap_coords.shape[0] == 0:
                return self[np.random.randint(len(self))]

        dense_gt_disp = None
        dense_loss_weight = None
        if use_dense_displacement:
            # Build dense GT from augmented surfaces so supervision is in the same frame
            # as transformed inputs/labels.
            full_dense_surface = torch.maximum(masked_seg, cond_seg_gt)
            dense_disp_np, dense_weight_np = self._compute_dense_displacement_field(
                full_dense_surface.numpy()
            )
            if dense_disp_np is None:
                return self[np.random.randint(len(self))]
            dense_gt_disp = torch.from_numpy(dense_disp_np).to(torch.float32)
            dense_loss_weight = torch.from_numpy(dense_weight_np).to(torch.float32)

        result = {
            "vol": vol_crop,                 # raw volume crop
            "cond": cond_seg,                # conditioning segmentation
            "masked_seg": masked_seg,        # masked (target) segmentation
        }

        if use_other_wrap_cond:
            result["other_wraps"] = other_wraps_tensor  # other wraps from same segment as context

        if use_extrapolation:
            result["extrap_surface"] = extrap_surf     # extrapolated surface voxelization
            result["extrap_coords"] = extrap_coords    # (N, 3) coords for sampling predicted field
            result["gt_displacement"] = gt_disp        # (N, 3) ground truth displacement
            result["point_weights"] = point_weights    # (N,) per-point supervision weights
            if point_normals is not None:
                result["point_normals"] = point_normals  # (N, 3) normals for normal-scalar supervision

        if use_dense_displacement:
            result["dense_gt_displacement"] = dense_gt_disp  # (3, D, H, W)
            result["dense_loss_weight"] = dense_loss_weight  # (1, D, H, W)

        if use_sdt:
            result["sdt"] = sdt_tensor                 # signed distance transform of full (dilated) segmentation

        if use_heatmap:
            result["heatmap_target"] = heatmap_tensor  # (D, H, W) gaussian heatmap at expected positions

        if use_segmentation:
            result["segmentation"] = full_seg           # full segmentation (cond + masked)
            result["segmentation_skel"] = seg_skel      # skeleton for medial surface recall loss

        # Validate all tensors are non-empty and contain no NaN/Inf
        for key, tensor in result.items():
            if tensor.numel() == 0:
                print(f"WARNING: Empty tensor for '{key}' at index {idx}, resampling...")
                return self[np.random.randint(len(self))]
            if torch.isnan(tensor).any():
                print(f"WARNING: NaN values in '{key}' at index {idx}, resampling...")
                return self[np.random.randint(len(self))]
            if torch.isinf(tensor).any():
                print(f"WARNING: Inf values in '{key}' at index {idx}, resampling...")
                return self[np.random.randint(len(self))]

        return result
    


if __name__ == "__main__":
    config_path = "/home/sean/Documents/villa/vesuvius/src/vesuvius/neural_tracing/configs/config_rowcol_cond.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_ds = EdtSegDataset(config)
    print(f"Dataset has {len(train_ds)} patches")

    out_dir = Path("/tmp/edt_seg_debug")
    out_dir.mkdir(exist_ok=True)

    # Debug: check wrap distribution per chunk
    from collections import Counter
    wraps_per_chunk = []
    same_seg_wraps_per_chunk = []
    for patch in train_ds.patches[:2500]:
        wraps_per_chunk.append(len(patch.wraps))
        seg_idx_counts = Counter(w["segment_idx"] for w in patch.wraps)
        max_same_seg = max(seg_idx_counts.values()) if seg_idx_counts else 0
        same_seg_wraps_per_chunk.append(max_same_seg)

    print(f"Wraps per chunk: min={min(wraps_per_chunk)}, max={max(wraps_per_chunk)}, avg={sum(wraps_per_chunk)/len(wraps_per_chunk):.1f}")
    print(f"Max same-segment wraps per chunk: min={min(same_seg_wraps_per_chunk)}, max={max(same_seg_wraps_per_chunk)}")
    print(f"Chunks with >1 same-segment wrap: {sum(1 for x in same_seg_wraps_per_chunk if x > 1)}/{len(same_seg_wraps_per_chunk)}")

    num_samples = min(25, len(train_ds))
    for i in range(num_samples):
        sample = train_ds[i]

        # Save 3D volumes as tif
        for key in ['vol', 'cond', 'masked_seg', 'extrap_surface', 'other_wraps', 'sdt', 'heatmap_target',
                    'segmentation', 'segmentation_skel', 'dense_gt_displacement', 'dense_loss_weight']:
            if key in sample:
                subdir = out_dir / key
                subdir.mkdir(exist_ok=True)
                tifffile.imwrite(subdir / f"{i:03d}.tif", sample[key].numpy())

        # Print info about point data
        print(f"[{i+1}/{num_samples}] Sample {i:03d}:")
        if 'extrap_coords' in sample:
            print(f"  extrap_coords shape: {sample['extrap_coords'].shape}")
            print(f"  gt_displacement shape: {sample['gt_displacement'].shape}")
            print(f"  displacement magnitude range: [{sample['gt_displacement'].norm(dim=-1).min():.2f}, {sample['gt_displacement'].norm(dim=-1).max():.2f}]")

    print(f"Output saved to {out_dir}")
