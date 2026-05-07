import vesuvius.tifxyz as tifxyz
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from vesuvius.neural_tracing.datasets.common import (
    ChunkPatch,
    _compute_wrap_order_stats,
    _extract_wrap_ids,
    _prepare_cond_surface_keypoints,
    _parse_z_range,
    _read_volume_crop_from_patch,
    _require_augmented_keypoints,
    _segment_overlaps_z_range,
    _should_attempt_cond_local_perturb,
    _trim_to_world_bbox,
    _triplet_wraps_compatible,
    _upsample_world_triplet,
    _validate_result_tensors,
    open_zarr_group,
    voxelize_surface_grid,
)
from vesuvius.neural_tracing.datasets.patch_finding import find_world_chunk_patches
from vesuvius.neural_tracing.datasets.direction_helpers import (
    build_away_from_conditioning_trace_targets,
)
from vesuvius.neural_tracing.datasets.augmentation import (
    augment_split_payload,
)
from vesuvius.neural_tracing.datasets.triplet_resampling import (
    choose_replacement_index,
)
from vesuvius.neural_tracing.datasets.dataset_defaults import (
    setdefault_rowcol_cond_dataset_config,
    validate_rowcol_cond_dataset_config,
)
from vesuvius.neural_tracing.datasets.conditioning import (
    create_split_conditioning,
)
from vesuvius.neural_tracing.datasets.perturbation import maybe_perturb_conditioning_surface
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
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
        setdefault_rowcol_cond_dataset_config(config)
        validate_rowcol_cond_dataset_config(config)
        self.use_trace_validity_targets = bool(config.get('use_trace_validity_targets', False)) or (
            float(config.get('lambda_trace_validity', 0.0)) > 0.0
        )
        self.use_neighbor_sheet_context = bool(config.get('use_neighbor_sheet_context', False))
        self.neighbor_sheet_required = bool(config.get('neighbor_sheet_required', False))
        if self.neighbor_sheet_required:
            self.use_neighbor_sheet_context = True
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

        self._triplet_neighbor_lookup = {}
        self._triplet_lookup_stats = {}

        if patch_metadata is None:
            patches = []
            for dataset_idx, dataset in enumerate(config['datasets']):
                volume_path = dataset['volume_path']
                volume_scale = dataset['volume_scale']
                volume = open_zarr_group(
                    volume_path,
                    auth_json_path=dataset.get('volume_auth_json'),
                    config=config,
                )
                segments_path = dataset['segments_path']
                z_range = _parse_z_range(dataset.get('z_range', None))
                dataset_segments = list(tifxyz.load_folder(segments_path))

                # retarget to the proper scale
                retarget_factor = 2 ** volume_scale
                scaled_segments = []
                dropped_by_z_range = 0
                for i, seg in enumerate(dataset_segments):
                    seg_scaled = seg if retarget_factor == 1 else seg.retarget(retarget_factor)
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

            self.patches = patches
            self.sample_index = self._build_sample_index()
            if self.use_neighbor_sheet_context:
                self._triplet_neighbor_lookup = self._build_triplet_neighbor_lookup()
            if self.neighbor_sheet_required:
                self.sample_index = [
                    (patch_idx, wrap_idx)
                    for patch_idx, wrap_idx in self.sample_index
                    if (patch_idx, wrap_idx) in self._triplet_neighbor_lookup
                ]
                if not self.sample_index:
                    raise ValueError(
                        "neighbor_sheet_required enabled but no wraps have same/adjacent neighbors on both sides."
                    )
            spec = self.config['cond_percent']
            self._cond_percent_min, self._cond_percent_max = float(spec[0]), float(spec[1])
        else:
            self._load_patch_metadata(patch_metadata)

    def __len__(self):
        return len(self.sample_index)

    def _build_sample_index(self):
        sample_index = []
        for patch_idx, patch in enumerate(self.patches):
            for wrap_idx in range(len(patch.wraps)):
                sample_index.append((patch_idx, wrap_idx))
        return sample_index

    def _load_patch_metadata(self, patch_metadata):
        self.patches = patch_metadata['patches']
        self.sample_index = list(patch_metadata['sample_index'])
        self._triplet_neighbor_lookup = patch_metadata.get('triplet_neighbor_lookup', {})
        self._triplet_lookup_stats = patch_metadata.get('triplet_lookup_stats', {})
        self._cond_percent_min, self._cond_percent_max = patch_metadata['cond_percent']

    def export_patch_metadata(self):
        return {
            'patches': self.patches,
            'sample_index': tuple(self.sample_index),
            'triplet_neighbor_lookup': self._triplet_neighbor_lookup,
            'triplet_lookup_stats': self._triplet_lookup_stats,
            'cond_percent': (self._cond_percent_min, self._cond_percent_max),
        }

    def _build_triplet_neighbor_lookup(self):
        """Build (patch_idx, wrap_idx) -> neighbor-wrap metadata for triplet mode.

        A wrap is kept only when it has one compatible neighbor on each side in
        local wrap ordering. The same lookup also provides neighbor-sheet context
        for trace-validity targets.
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

    def create_split_masks(self, idx: int, patch_idx: int, wrap_idx: int):
        patch = self.patches[patch_idx]
        crop_size = self.crop_size  # tuple (D, H, W)
        target_shape = crop_size

        conditioning = create_split_conditioning(self, idx, patch_idx, wrap_idx, patch)
        if conditioning is None:
            return None
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

        neighbor_segmentation = None
        if self.use_neighbor_sheet_context:
            neighbor_segmentation = self._build_neighbor_sheet_mask(
                patch_idx=patch_idx,
                wrap_idx=wrap_idx,
                min_corner=min_corner,
                crop_shape=crop_shape,
            )
            if neighbor_segmentation is None and self.neighbor_sheet_required:
                return None

        result = {
            "vol": torch.from_numpy(vol_crop).to(torch.float32),
            "masked_seg": torch.from_numpy(masked_segmentation).to(torch.float32),
            "cond_gt": torch.from_numpy(cond_segmentation_gt).to(torch.float32),
            "cond_surface_local": torch.from_numpy(cond_zyxs_unperturbed_local_float).to(torch.float32),
            "masked_surface_local": torch.from_numpy(masked_zyxs_local_float).to(torch.float32),
            "cond_direction": cond_direction,
        }
        if neighbor_segmentation is not None:
            result["neighbor_seg"] = torch.from_numpy(neighbor_segmentation).to(torch.float32)
        return result

    def _build_neighbor_sheet_mask(self, *, patch_idx: int, wrap_idx: int, min_corner, crop_shape):
        triplet_meta = self._triplet_neighbor_lookup.get((patch_idx, wrap_idx))
        if triplet_meta is None:
            return None

        neighbor_vox = np.zeros(crop_shape, dtype=np.float32)
        for key in ("behind_wrap_idx", "front_wrap_idx"):
            neighbor_surface = self._extract_wrap_world_surface_by_index(
                patch_idx,
                int(triplet_meta[key]),
                require_all_valid=True,
            )
            if neighbor_surface is None:
                return None
            neighbor_local = (neighbor_surface - min_corner).astype(np.float32, copy=False)
            neighbor_vox = np.maximum(neighbor_vox, voxelize_surface_grid(neighbor_local, crop_shape))

        if not bool(neighbor_vox.any()):
            return None
        return neighbor_vox.astype(np.float32, copy=False)

    def _resample_item(self, idx, reason, patch_idx=None, wrap_idx=None, attempted_indices=None):
        attempted = set(int(i) for i in (attempted_indices or ()))
        if 0 <= int(idx) < len(self):
            attempted.add(int(idx))

        new_idx = choose_replacement_index(
            self.sample_index,
            attempted_indices=attempted,
        )
        if new_idx is None:
            raise RuntimeError(
                f"Unable to resample item after exhausting {len(attempted)} unique indices; "
                f"last idx={idx}, reason={reason}"
            )
        patch_str = f" patch={patch_idx}" if patch_idx is not None else ""
        wrap_str = f" wrap={wrap_idx}" if wrap_idx is not None else ""
        if bool(self.config.get("verbose", False)):
            print(
                f"[EdtSegDataset] Resampling item idx={idx}{patch_str}{wrap_str} "
                f"reason={reason} replacement_idx={new_idx}"
            )
        return self.__getitem__(new_idx, _attempted_indices=attempted)

    def __getitem__(self, idx, _attempted_indices=None):
        idx = int(idx)
        if _attempted_indices is None:
            _attempted_indices = set()
        if idx in _attempted_indices:
            return self._resample_item(
                idx,
                "duplicate resample index encountered",
                attempted_indices=_attempted_indices,
            )
        _attempted_indices.add(idx)

        patch_idx, wrap_idx = self.sample_index[idx]
        mask_bundle = self.create_split_masks(idx, patch_idx, wrap_idx)
        if mask_bundle is None:
            return self._resample_item(
                idx,
                "split mask bundle missing",
                patch_idx=patch_idx,
                wrap_idx=wrap_idx,
                attempted_indices=_attempted_indices,
            )

        vol_crop = mask_bundle["vol"]
        masked_seg = mask_bundle["masked_seg"]
        cond_seg_gt = mask_bundle["cond_gt"]
        cond_direction = mask_bundle["cond_direction"]
        neighbor_seg_tensor = mask_bundle.get("neighbor_seg", None)
        cond_surface_local, cond_surface_shape, cond_surface_keypoints, valid_cond_surface = (
            _prepare_cond_surface_keypoints(mask_bundle.get("cond_surface_local"))
        )
        if not valid_cond_surface:
            return self._resample_item(
                idx,
                "split conditioning surface invalid",
                patch_idx=patch_idx,
                wrap_idx=wrap_idx,
                attempted_indices=_attempted_indices,
            )
        masked_surface_local, masked_surface_shape, masked_surface_keypoints, valid_masked_surface = (
            _prepare_cond_surface_keypoints(mask_bundle.get("masked_surface_local"))
        )
        if not valid_masked_surface:
            return self._resample_item(
                idx,
                "split masked surface invalid",
                patch_idx=patch_idx,
                wrap_idx=wrap_idx,
                attempted_indices=_attempted_indices,
            )

        split_augmented = augment_split_payload(
            augmentations=self._augmentations,
            crop_size=self.crop_size,
            vol_crop=vol_crop,
            masked_seg=masked_seg,
            cond_seg_gt=cond_seg_gt,
            cond_surface_local=cond_surface_local,
            cond_surface_keypoints=cond_surface_keypoints,
            cond_surface_shape=cond_surface_shape,
            restore_cond_surface_fn=self._restore_cond_surface_from_augmented,
            masked_surface_local=masked_surface_local,
            masked_surface_keypoints=masked_surface_keypoints,
            masked_surface_shape=masked_surface_shape,
            neighbor_seg_tensor=neighbor_seg_tensor,
        )
        vol_crop = split_augmented["vol_crop"]
        masked_seg = split_augmented["masked_seg"]
        cond_seg_gt = split_augmented["cond_seg_gt"]
        cond_surface_local = split_augmented["cond_surface_local"]
        masked_surface_local = split_augmented["masked_surface_local"]
        neighbor_seg_tensor = split_augmented.get("neighbor_seg_tensor", neighbor_seg_tensor)
        if self._cond_local_perturb_active and cond_surface_local is not None:
            cond_seg = self._conditioning_from_surface(
                cond_surface_local=cond_surface_local,
                cond_seg_gt=cond_seg_gt,
            )
        else:
            cond_seg = cond_seg_gt.clone()
        if cond_seg is None:
            return self._resample_item(
                idx,
                "split conditioning segmentation unresolved",
                patch_idx=patch_idx,
                wrap_idx=wrap_idx,
                attempted_indices=_attempted_indices,
            )

        result = {
            "vol": vol_crop,
            "cond": cond_seg,
            "masked_seg": masked_seg,
            "cond_gt": cond_seg_gt,
            "cond_direction": cond_direction,
        }

        if self.use_trace_validity_targets and neighbor_seg_tensor is not None:
            result["neighbor_seg"] = neighbor_seg_tensor
        cond_surface_np = cond_surface_local.detach().cpu().numpy().astype(np.float32, copy=False)
        masked_surface_np = masked_surface_local.detach().cpu().numpy().astype(np.float32, copy=False)
        if cond_direction == "left":
            trace_surface_np = np.concatenate([cond_surface_np, masked_surface_np], axis=1)
        elif cond_direction == "right":
            trace_surface_np = np.concatenate([masked_surface_np, cond_surface_np], axis=1)
        elif cond_direction == "up":
            trace_surface_np = np.concatenate([cond_surface_np, masked_surface_np], axis=0)
        elif cond_direction == "down":
            trace_surface_np = np.concatenate([masked_surface_np, cond_surface_np], axis=0)
        else:
            trace_surface_np = None
        trace_payload = build_away_from_conditioning_trace_targets(
            self.crop_size,
            cond_direction,
            cond_surface_local=trace_surface_np,
            masked_surface_local=None,
            include_conditioning=True,
            include_masked=False,
            dilation_radius=0.0,
            surface_attract_radius=0.0,
        )
        if trace_payload is None:
            return self._resample_item(
                idx,
                "split trace target unavailable",
                patch_idx=patch_idx,
                wrap_idx=wrap_idx,
                attempted_indices=_attempted_indices,
            )
        result["velocity_dir"] = torch.from_numpy(trace_payload["velocity_dir"]).to(torch.float32)
        result["velocity_loss_weight"] = torch.from_numpy(trace_payload["trace_loss_weight"]).to(torch.float32)
        result["trace_loss_weight"] = torch.from_numpy(trace_payload["trace_loss_weight"]).to(torch.float32)

        if not _validate_result_tensors(
            result,
            idx,
            enabled=self._validate_result_tensors_enabled,
        ):
            return self._resample_item(
                idx,
                "result tensor validation failed",
                patch_idx=patch_idx,
                wrap_idx=wrap_idx,
                attempted_indices=_attempted_indices,
            )
        return result
    
