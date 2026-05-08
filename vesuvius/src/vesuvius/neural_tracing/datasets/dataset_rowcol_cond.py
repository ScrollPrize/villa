import vesuvius.tifxyz as tifxyz
import hashlib
import numpy as np
import torch
import time
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
    voxelize_surface_grid_into,
)
from vesuvius.neural_tracing.datasets.patch_finding import find_world_chunk_patches
from vesuvius.neural_tracing.datasets.direction_helpers import (
    build_split_surface_masks_and_trace_targets,
)
from vesuvius.neural_tracing.datasets.copy_neighbor_targets import (
    build_copy_neighbor_targets,
)
from vesuvius.neural_tracing.datasets.augmentation import (
    augment_split_payload,
)
from vesuvius.neural_tracing.datasets.triplet_resampling import (
    choose_replacement_index,
)
from vesuvius.neural_tracing.datasets.rowcol_cond_config import (
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

        setdefault_rowcol_cond_dataset_config(config)
        validate_rowcol_cond_dataset_config(config)
        self._validate_result_tensors_enabled = bool(config['validate_result_tensors'])
        self._profile_create_split_masks_enabled = bool(config.get("profile_create_split_masks", False))
        self._profile_create_split_masks = self._new_create_split_masks_profile()

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
                for i, seg in enumerate(dataset_segments):
                    seg_scaled = seg if retarget_factor == 1 else seg.retarget(retarget_factor)
                    if not _segment_overlaps_z_range(seg_scaled, z_range):
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
                    target_size=self.crop_size,
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
            self._triplet_neighbor_lookup = self._build_triplet_neighbor_lookup()
            self.sample_index = [
                (patch_idx, wrap_idx)
                for patch_idx, wrap_idx in self.sample_index
                if (patch_idx, wrap_idx) in self._triplet_neighbor_lookup
            ]
            if not self.sample_index:
                raise ValueError("No wraps have same/adjacent neighbors on both sides.")
            if str(self.config.get("training_mode", "rowcol_hidden")) == "copy_neighbors":
                self.sample_index = self._build_copy_neighbor_pair_records(self.sample_index)
                if not self.sample_index:
                    raise ValueError("No copy_neighbors source-target pair records remain after filtering.")
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
        self._cond_percent_min, self._cond_percent_max = patch_metadata['cond_percent']

    def export_patch_metadata(self):
        return {
            'patches': self.patches,
            'sample_index': tuple(self.sample_index),
            'triplet_neighbor_lookup': self._triplet_neighbor_lookup,
            'cond_percent': (self._cond_percent_min, self._cond_percent_max),
        }

    def _build_copy_neighbor_pair_records(self, triplet_middle_index):
        source_filter = str(self.config.get("copy_neighbor_source_position", "random"))
        target_filter = str(self.config.get("copy_neighbor_target_side", "random"))
        record_mode = str(self.config.get("copy_neighbor_pair_record_mode", "all_directed"))
        seed = int(self.config.get("seed", 0)) + int(self.config.get("copy_neighbor_pair_seed_offset", 0))

        records = []
        for patch_idx, middle_wrap_idx in triplet_middle_index:
            triplet_meta = self._triplet_neighbor_lookup[(patch_idx, middle_wrap_idx)]
            behind_wrap_idx = int(triplet_meta["behind_wrap_idx"])
            middle_wrap_idx = int(middle_wrap_idx)
            front_wrap_idx = int(triplet_meta["front_wrap_idx"])
            position_to_wrap = {
                "behind": behind_wrap_idx,
                "middle": middle_wrap_idx,
                "front": front_wrap_idx,
            }
            candidates = [
                ("behind", "middle", "front"),
                ("middle", "behind", "behind"),
                ("middle", "front", "front"),
                ("front", "middle", "behind"),
            ]
            triplet_records = []
            for source_pos, target_pos, target_side in candidates:
                if source_filter != "random" and source_pos != source_filter:
                    continue
                if target_filter != "random" and target_side != target_filter:
                    continue
                if source_pos == target_pos:
                    continue
                triplet_records.append({
                    "patch_idx": int(patch_idx),
                    "middle_wrap_idx": int(middle_wrap_idx),
                    "source_wrap_idx": int(position_to_wrap[source_pos]),
                    "target_wrap_idx": int(position_to_wrap[target_pos]),
                    "source_triplet_pos": source_pos,
                    "target_side": target_side,
                    "triplet_wrap_indices": (
                        int(behind_wrap_idx),
                        int(middle_wrap_idx),
                        int(front_wrap_idx),
                    ),
                })

            if record_mode == "one_per_triplet" and triplet_records:
                key = f"{seed}:{patch_idx}:{middle_wrap_idx}".encode("utf8")
                digest = hashlib.sha256(key).digest()
                chosen = int.from_bytes(digest[:8], "little") % len(triplet_records)
                records.append(triplet_records[chosen])
            else:
                records.extend(triplet_records)
        return records

    @staticmethod
    def _new_create_split_masks_profile():
        return {
            "attempts": 0,
            "successes": 0,
            "total": 0.0,
            "conditioning": 0.0,
            "volume_read": 0.0,
            "coord_convert": 0.0,
            "neighbor_total": 0.0,
            "neighbor_extract": 0.0,
            "neighbor_voxelize": 0.0,
            "tensor_convert": 0.0,
        }

    def create_split_masks_profile_summary(self) -> dict:
        profile = dict(self._profile_create_split_masks)
        successes = int(profile.get("successes", 0))
        attempts = int(profile.get("attempts", 0))
        denom = max(successes, 1)
        profile["mean_total"] = float(profile["total"]) / denom
        profile["mean_by_stage"] = {
            key: float(profile[key]) / denom
            for key in (
                "conditioning",
                "volume_read",
                "coord_convert",
                "neighbor_total",
                "neighbor_extract",
                "neighbor_voxelize",
                "tensor_convert",
            )
        }
        profile["success_rate"] = float(successes) / max(attempts, 1)
        return profile

    def _build_triplet_neighbor_lookup(self):
        """Build (patch_idx, wrap_idx) -> neighbor-wrap metadata for triplet mode.

        A wrap is kept only when it has one compatible neighbor on each side in
        local wrap ordering. The same lookup also provides neighbor-sheet context
        for trace-validity targets.
        """
        lookup = {}
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
                    continue

                lookup[(patch_idx, target["wrap_idx"])] = {
                    "behind_wrap_idx": int(prev_neighbor["wrap_idx"]),
                    "front_wrap_idx": int(next_neighbor["wrap_idx"]),
                }
        return lookup

    def _extract_wrap_world_surface(self, patch: ChunkPatch, wrap: dict):
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
            if not valid_s.all():
                return None

        x_full, y_full, z_full = _upsample_world_triplet(x_s, y_s, z_s, scale_y, scale_x)
        trimmed = _trim_to_world_bbox(x_full, y_full, z_full, patch.world_bbox)
        if trimmed is None:
            return None
        x_full, y_full, z_full = trimmed
        return np.stack([z_full, y_full, x_full], axis=-1)

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

    def _surface_crop_support_ok(self, surface_local, *, allow_partial: bool = False):
        if bool(allow_partial):
            return True, "partial surface allowed"
        surface = np.asarray(surface_local, dtype=np.float32)
        if surface.ndim != 3 or int(surface.shape[-1]) != 3:
            return False, "surface shape invalid"
        finite = np.isfinite(surface).all(axis=-1)
        in_crop = (
            finite
            & (surface[..., 0] >= 0.0)
            & (surface[..., 1] >= 0.0)
            & (surface[..., 2] >= 0.0)
            & (surface[..., 0] <= self.crop_size[0] - 1)
            & (surface[..., 1] <= self.crop_size[1] - 1)
            & (surface[..., 2] <= self.crop_size[2] - 1)
        )
        min_valid_fraction = float(self.config.get("copy_neighbor_min_in_crop_valid_fraction", 0.25))
        valid_fraction = float(in_crop.mean()) if in_crop.size else 0.0
        if valid_fraction < min_valid_fraction:
            return False, f"in-crop fraction {valid_fraction:.3f} below {min_valid_fraction:.3f}"

        strip_width = max(1, int(self.config.get("copy_neighbor_surface_edge_strip_width", 2)))
        edge_valid_frac = float(self.config.get("copy_neighbor_surface_edge_valid_frac", 0.10))
        row_strip = min(strip_width, in_crop.shape[0])
        col_strip = min(strip_width, in_crop.shape[1])
        row_low = float(in_crop[:row_strip].mean()) if row_strip > 0 else 0.0
        row_high = float(in_crop[-row_strip:].mean()) if row_strip > 0 else 0.0
        col_low = float(in_crop[:, :col_strip].mean()) if col_strip > 0 else 0.0
        col_high = float(in_crop[:, -col_strip:].mean()) if col_strip > 0 else 0.0
        row_spans = min(row_low, row_high) >= edge_valid_frac
        col_spans = min(col_low, col_high) >= edge_valid_frac
        if not (row_spans or col_spans):
            return False, "surface does not span row or column edge strips"
        return True, "ok"

    def create_copy_neighbor_masks(self, record: dict):
        patch_idx = int(record["patch_idx"])
        patch = self.patches[patch_idx]
        source_wrap_idx = int(record["source_wrap_idx"])
        target_wrap_idx = int(record["target_wrap_idx"])
        source_world = self._extract_wrap_world_surface(patch, patch.wraps[source_wrap_idx])
        target_world = self._extract_wrap_world_surface(patch, patch.wraps[target_wrap_idx])
        if source_world is None or target_world is None:
            return None

        finite_source = np.isfinite(source_world).all(axis=-1)
        if not bool(finite_source.any()):
            return None
        source_points = source_world[finite_source]
        center = np.median(source_points, axis=0)
        min_corner = np.round(center - 0.5 * np.asarray(self.crop_size, dtype=np.float32)).astype(np.int64)
        max_corner = min_corner + np.asarray(self.crop_size, dtype=np.int64)

        source_local = (source_world - min_corner).astype(np.float32, copy=False)
        target_local = (target_world - min_corner).astype(np.float32, copy=False)
        ok, _ = self._surface_crop_support_ok(source_local)
        if not ok:
            return None
        target_ok, _ = self._surface_crop_support_ok(
            target_local,
            allow_partial=bool(self.config.get("copy_neighbor_allow_partial_target", False)),
        )
        if not target_ok:
            return None

        vol_crop = _read_volume_crop_from_patch(
            patch,
            self.crop_size,
            min_corner,
            max_corner,
        )
        return {
            "vol": torch.from_numpy(vol_crop).to(torch.float32),
            "source_surface_local": source_local,
            "target_surface_local": target_local,
            "min_corner": min_corner,
            "record": record,
        }

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

    def create_split_masks(self, patch_idx: int, wrap_idx: int):
        profile_enabled = self._profile_create_split_masks_enabled
        profile_start = time.perf_counter() if profile_enabled else 0.0
        last = profile_start
        stage_times = None
        if profile_enabled:
            self._profile_create_split_masks["attempts"] += 1
            stage_times = {
                "conditioning": 0.0,
                "volume_read": 0.0,
                "coord_convert": 0.0,
                "neighbor_total": 0.0,
                "neighbor_extract": 0.0,
                "neighbor_voxelize": 0.0,
                "tensor_convert": 0.0,
            }

        def record_stage(name: str) -> None:
            nonlocal last
            if stage_times is None:
                return
            now = time.perf_counter()
            stage_times[name] += now - last
            last = now

        patch = self.patches[patch_idx]
        crop_size = self.crop_size  # tuple (D, H, W)
        target_shape = crop_size

        conditioning = create_split_conditioning(self, patch_idx, wrap_idx, patch)
        record_stage("conditioning")
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
        record_stage("volume_read")

        # convert cond and masked coords to crop-local coords (float for line interpolation)
        cond_zyxs_unperturbed_local_float = (cond_zyxs_unperturbed - min_corner).astype(np.float32)
        masked_zyxs_local_float = (masked_zyxs - min_corner).astype(np.float32)
        record_stage("coord_convert")

        crop_shape = target_shape

        neighbor_start = time.perf_counter() if stage_times is not None else 0.0
        neighbor_segmentation = self._build_neighbor_sheet_mask(
            patch_idx=patch_idx,
            wrap_idx=wrap_idx,
            min_corner=min_corner,
            crop_shape=crop_shape,
            stage_times=stage_times,
        )
        if stage_times is not None:
            now = time.perf_counter()
            stage_times["neighbor_total"] += now - neighbor_start
            last = now
        if neighbor_segmentation is None:
            return None

        result = {
            "vol": torch.from_numpy(vol_crop).to(torch.float32),
            "cond_surface_local": cond_zyxs_unperturbed_local_float,
            "masked_surface_local": masked_zyxs_local_float,
            "cond_direction": cond_direction,
        }
        result["neighbor_seg"] = torch.from_numpy(neighbor_segmentation).to(torch.float32)
        record_stage("tensor_convert")
        if stage_times is not None:
            profile = self._profile_create_split_masks
            profile["successes"] += 1
            profile["total"] += time.perf_counter() - profile_start
            for name, seconds in stage_times.items():
                profile[name] += seconds
        return result

    def _build_neighbor_sheet_mask(self, *, patch_idx: int, wrap_idx: int, min_corner, crop_shape, stage_times=None):
        triplet_meta = self._triplet_neighbor_lookup.get((patch_idx, wrap_idx))
        if triplet_meta is None:
            return None

        patch = self.patches[patch_idx]
        neighbor_vox = np.zeros(crop_shape, dtype=np.uint8)
        for key in ("behind_wrap_idx", "front_wrap_idx"):
            neighbor_wrap_idx = int(triplet_meta[key])
            extract_start = time.perf_counter() if stage_times is not None else 0.0
            neighbor_surface = self._extract_wrap_world_surface(
                patch,
                patch.wraps[neighbor_wrap_idx],
            )
            extract_done = time.perf_counter() if stage_times is not None else 0.0
            if stage_times is not None:
                stage_times["neighbor_extract"] += extract_done - extract_start
            if neighbor_surface is None:
                return None
            neighbor_local = (neighbor_surface - min_corner).astype(np.float32, copy=False)
            voxelize_surface_grid_into(neighbor_vox, neighbor_local)
            if stage_times is not None:
                voxel_done = time.perf_counter()
                stage_times["neighbor_voxelize"] += voxel_done - extract_done

        if not bool(neighbor_vox.any()):
            return None
        return neighbor_vox

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

    def _getitem_copy_neighbors(self, idx: int, _attempted_indices):
        record = self.sample_index[idx]
        patch_idx = int(record["patch_idx"])
        source_wrap_idx = int(record["source_wrap_idx"])
        target_wrap_idx = int(record["target_wrap_idx"])
        mask_bundle = self.create_copy_neighbor_masks(record)
        if mask_bundle is None:
            return self._resample_item(
                idx,
                "copy-neighbor mask bundle missing",
                patch_idx=patch_idx,
                wrap_idx=source_wrap_idx,
                attempted_indices=_attempted_indices,
            )

        vol_crop = mask_bundle["vol"]
        source_surface_source = torch.from_numpy(mask_bundle["source_surface_local"]).to(torch.float32)
        target_surface_source = torch.from_numpy(mask_bundle["target_surface_local"]).to(torch.float32)
        source_surface_local, source_surface_shape, source_keypoints, valid_source = (
            _prepare_cond_surface_keypoints(source_surface_source)
        )
        target_surface_local, target_surface_shape, target_keypoints, valid_target = (
            _prepare_cond_surface_keypoints(target_surface_source)
        )
        if not valid_source or not valid_target:
            return self._resample_item(
                idx,
                "copy-neighbor source/target surface invalid",
                patch_idx=patch_idx,
                wrap_idx=source_wrap_idx,
                attempted_indices=_attempted_indices,
            )

        augmented = augment_split_payload(
            augmentations=self._augmentations,
            crop_size=self.crop_size,
            vol_crop=vol_crop,
            masked_seg=None,
            cond_seg_gt=None,
            cond_surface_local=source_surface_local,
            cond_surface_keypoints=source_keypoints,
            cond_surface_shape=source_surface_shape,
            restore_cond_surface_fn=self._restore_cond_surface_from_augmented,
            masked_surface_local=target_surface_local,
            masked_surface_keypoints=target_keypoints,
            masked_surface_shape=target_surface_shape,
            neighbor_seg_tensor=None,
        )
        vol_crop = augmented["vol_crop"]
        source_surface_local = augmented["cond_surface_local"]
        target_surface_local = augmented["masked_surface_local"]
        source_np = source_surface_local.detach().cpu().numpy().astype(np.float32, copy=False)
        target_np = target_surface_local.detach().cpu().numpy().astype(np.float32, copy=False)

        ok, _ = self._surface_crop_support_ok(source_np)
        target_ok, _ = self._surface_crop_support_ok(
            target_np,
            allow_partial=bool(self.config.get("copy_neighbor_allow_partial_target", False)),
        )
        if not ok or not target_ok:
            return self._resample_item(
                idx,
                "copy-neighbor augmented surface support failed",
                patch_idx=patch_idx,
                wrap_idx=source_wrap_idx,
                attempted_indices=_attempted_indices,
            )

        source_valid = np.isfinite(source_np).all(axis=-1)
        target_valid = np.isfinite(target_np).all(axis=-1)
        if not bool(source_valid.any()) or not bool(target_valid.any()):
            return self._resample_item(
                idx,
                "copy-neighbor side hint centers unavailable",
                patch_idx=patch_idx,
                wrap_idx=source_wrap_idx,
                attempted_indices=_attempted_indices,
            )
        source_center = np.median(source_np[source_valid], axis=0)
        target_center = np.median(target_np[target_valid], axis=0)
        side_hint_vector = (target_center - source_center).astype(np.float32)
        side_norm = float(np.linalg.norm(side_hint_vector))
        if not np.isfinite(side_norm) or side_norm <= 1e-6:
            return self._resample_item(
                idx,
                "copy-neighbor side hint vector invalid",
                patch_idx=patch_idx,
                wrap_idx=source_wrap_idx,
                attempted_indices=_attempted_indices,
            )
        side_hint_vector /= np.float32(side_norm)

        payload = build_copy_neighbor_targets(
            self.crop_size,
            source_np,
            target_np,
            side_hint_vector,
            self.config,
        )
        if payload is None:
            return self._resample_item(
                idx,
                "copy-neighbor targets unavailable",
                patch_idx=patch_idx,
                wrap_idx=source_wrap_idx,
                attempted_indices=_attempted_indices,
            )

        cond_gt = torch.from_numpy(payload.cond_gt).to(torch.float32)
        if self._cond_local_perturb_active:
            cond_seg = self._conditioning_from_surface(
                cond_surface_local=source_surface_local,
                cond_seg_gt=cond_gt,
            )
        else:
            cond_seg = cond_gt.clone()
        if cond_seg is None:
            return self._resample_item(
                idx,
                "copy-neighbor conditioning segmentation unresolved",
                patch_idx=patch_idx,
                wrap_idx=source_wrap_idx,
                attempted_indices=_attempted_indices,
            )

        side_hint = np.broadcast_to(
            side_hint_vector[:, None, None, None],
            (3, *self.crop_size),
        ).astype(np.float32, copy=True)
        result = {
            "vol": vol_crop,
            "cond": cond_seg,
            "side_hint": torch.from_numpy(side_hint).to(torch.float32),
            "target_seg": torch.from_numpy(payload.target_seg).to(torch.float32),
            "domain": torch.from_numpy(payload.domain).to(torch.float32),
            "velocity_dir": torch.from_numpy(payload.velocity_dir).to(torch.float32),
            "velocity_loss_weight": torch.from_numpy(payload.velocity_loss_weight).to(torch.float32),
            "progress_phi": torch.from_numpy(payload.progress_phi).to(torch.float32),
            "progress_phi_weight": torch.from_numpy(payload.progress_phi_weight).to(torch.float32),
            "surface_attract": torch.from_numpy(payload.surface_attract).to(torch.float32),
            "surface_attract_weight": torch.from_numpy(payload.surface_attract_weight).to(torch.float32),
            "stop": torch.from_numpy(payload.stop).to(torch.float32),
            "stop_weight": torch.from_numpy(payload.stop_weight).to(torch.float32),
            "target_edt": torch.from_numpy(payload.target_edt).to(torch.float32),
            "endpoint_seed_points": torch.from_numpy(payload.endpoint_seed_points).to(torch.float32),
            "endpoint_seed_mask": torch.from_numpy(payload.endpoint_seed_mask).to(torch.float32),
            "endpoint_step_count": torch.tensor(
                int(payload.debug.get("endpoint_step_count", self.config.get("copy_neighbor_endpoint_steps", 8))),
                dtype=torch.long,
            ),
        }

        if not _validate_result_tensors(
            result,
            idx,
            enabled=self._validate_result_tensors_enabled,
        ):
            return self._resample_item(
                idx,
                "copy-neighbor result tensor validation failed",
                patch_idx=patch_idx,
                wrap_idx=target_wrap_idx,
                attempted_indices=_attempted_indices,
            )
        return result

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

        if str(self.config.get("training_mode", "rowcol_hidden")) == "copy_neighbors":
            return self._getitem_copy_neighbors(idx, _attempted_indices)

        patch_idx, wrap_idx = self.sample_index[idx]
        mask_bundle = self.create_split_masks(patch_idx, wrap_idx)
        if mask_bundle is None:
            return self._resample_item(
                idx,
                "split mask bundle missing",
                patch_idx=patch_idx,
                wrap_idx=wrap_idx,
                attempted_indices=_attempted_indices,
            )

        vol_crop = mask_bundle["vol"]
        cond_direction = mask_bundle["cond_direction"]
        neighbor_seg_tensor = mask_bundle["neighbor_seg"]
        cond_surface_source = mask_bundle.get("cond_surface_local")
        masked_surface_source = mask_bundle.get("masked_surface_local")
        if isinstance(cond_surface_source, np.ndarray):
            cond_surface_source = torch.from_numpy(cond_surface_source).to(torch.float32)
        if isinstance(masked_surface_source, np.ndarray):
            masked_surface_source = torch.from_numpy(masked_surface_source).to(torch.float32)
        cond_surface_local, cond_surface_shape, cond_surface_keypoints, valid_cond_surface = (
            _prepare_cond_surface_keypoints(cond_surface_source)
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
            _prepare_cond_surface_keypoints(masked_surface_source)
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
            masked_seg=None,
            cond_seg_gt=None,
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
        cond_surface_local = split_augmented["cond_surface_local"]
        masked_surface_local = split_augmented["masked_surface_local"]
        neighbor_seg_tensor = split_augmented.get("neighbor_seg_tensor", neighbor_seg_tensor)
        cond_surface_np = cond_surface_local.detach().cpu().numpy().astype(np.float32, copy=False)
        masked_surface_np = masked_surface_local.detach().cpu().numpy().astype(np.float32, copy=False)
        fused_payload = build_split_surface_masks_and_trace_targets(
            self.crop_size,
            cond_direction,
            cond_surface_local=cond_surface_np,
            masked_surface_local=masked_surface_np,
        )
        if fused_payload is None or not bool(fused_payload["cond_gt"].any()):
            return self._resample_item(
                idx,
                "split surface targets unavailable",
                patch_idx=patch_idx,
                wrap_idx=wrap_idx,
                attempted_indices=_attempted_indices,
            )
        masked_seg = torch.from_numpy(fused_payload["masked_seg"]).to(torch.float32)
        cond_seg_gt = torch.from_numpy(fused_payload["cond_gt"]).to(torch.float32)
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

        result["neighbor_seg"] = neighbor_seg_tensor
        result["velocity_dir"] = torch.from_numpy(fused_payload["velocity_dir"]).to(torch.float32)
        result["velocity_loss_weight"] = torch.from_numpy(fused_payload["trace_loss_weight"]).to(torch.float32)
        result["trace_loss_weight"] = torch.from_numpy(fused_payload["trace_loss_weight"]).to(torch.float32)

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
    
