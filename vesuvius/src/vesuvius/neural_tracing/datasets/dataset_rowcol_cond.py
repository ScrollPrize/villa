import vesuvius.tifxyz as tifxyz
import hashlib
import os
import numpy as np
import torch
import time
import random
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from vesuvius.neural_tracing.datasets.common import (
    ChunkPatch,
    _prepare_cond_surface_keypoints,
    _parse_z_range,
    _read_volume_crop_from_patch,
    _require_augmented_keypoints,
    _segment_overlaps_z_range,
    _should_attempt_cond_local_perturb,
    _trim_to_world_bbox,
    _upsample_world_surface,
    _validate_result_tensors,
    open_zarr_group,
    voxelize_surface_grid,
    voxelize_surface_grid_into,
)
from vesuvius.neural_tracing.datasets.chunk_finding import find_training_chunks
from vesuvius.neural_tracing.datasets.direction_helpers import (
    build_triplet_direction_priors_from_conditioning_surface,
    build_split_surface_masks_and_trace_targets,
)
from vesuvius.neural_tracing.datasets.augmentation import (
    augment_split_payload,
    augment_triplet_payload,
)
from vesuvius.neural_tracing.datasets.neighbor_resampling import (
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
        self.use_triplet_direction_priors = bool(config['use_triplet_direction_priors'])
        self.triplet_direction_prior_mask = str(config['triplet_direction_prior_mask']).lower()

        if patch_metadata is None:
            patches = []
            dataset_iter = tqdm(
                list(enumerate(config['datasets'])),
                desc="Patch finding datasets",
            )
            for dataset_idx, dataset in dataset_iter:
                volume_path = dataset['volume_path']
                volume_scale = dataset['volume_scale']
                dataset_iter.set_postfix_str(f"dataset_idx={dataset_idx}")
                volume = open_zarr_group(
                    volume_path,
                    auth_json_path=dataset.get('volume_auth_json'),
                    config=config,
                )
                segments_path = dataset['segments_path']
                z_range = _parse_z_range(dataset.get('z_range', None))
                dataset_segments = list(tqdm(
                    tifxyz.load_folder(segments_path),
                    desc=f"Loading segments dataset {dataset_idx}",
                    unit="segment",
                ))

                # retarget to the proper scale
                retarget_factor = 2 ** volume_scale
                scaled_segments = []
                segment_iter = tqdm(
                    dataset_segments,
                    desc=f"Retarget/filter segments dataset {dataset_idx}",
                    unit="segment",
                )
                for seg in segment_iter:
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

                cache_dir = Path(segments_path) / ".patch_cache" if segments_path else None
                chunk_results = find_training_chunks(
                    segments=scaled_segments,
                    volume=volume,
                    scale=volume_scale,
                    target_size=self.crop_size,
                    overlap_fraction=config['overlap_fraction'],
                    min_points_per_wrap=min_points_per_wrap,
                    bbox_pad_2d=config['bbox_pad_2d'],
                    cache_dir=cache_dir,
                    force_recompute=config['force_recompute_patches'],
                    verbose=config.get('verbose', False),
                    chunk_pad=config.get('chunk_pad', 0.0),
                    terminal_chunk_guard_voxels=config.get('terminal_chunk_guard_voxels', None),
                    training_mode=config.get('training_mode', 'rowcol_hidden'),
                )

                for chunk in tqdm(
                    chunk_results,
                    desc=f"Materializing chunk metadata dataset {dataset_idx}",
                    unit="chunk",
                ):
                    if bool(chunk.get("no_neighboring_wraps", False)):
                        continue
                    neighbor_sets = {
                        int(k): tuple(int(v) for v in values)
                        for k, values in chunk.get("neighbor_sets", {}).items()
                    }
                    wraps_in_chunk = []
                    for w in chunk["wraps"]:
                        seg_idx = w["segment_idx"]
                        wrap_idx = int(w.get("wrap_idx", len(wraps_in_chunk)))
                        wraps_in_chunk.append({
                            "segment": scaled_segments[seg_idx],
                            "bbox_2d": tuple(w["bbox_2d"]),
                            "wrap_id": w["wrap_id"],
                            "wrap_label": int(w.get("wrap_label", w["wrap_id"])),
                            "wrap_idx": wrap_idx,
                            "segment_idx": seg_idx,
                            "neighbor_wrap_indices": neighbor_sets.get(wrap_idx, tuple()),
                        })

                    patch = ChunkPatch(
                        chunk_id=tuple(chunk["chunk_id"]),
                        volume=volume,
                        scale=volume_scale,
                        world_bbox=tuple(chunk["bbox_3d"]),
                        wraps=wraps_in_chunk,
                        segments=scaled_segments,
                        dataset_idx=dataset_idx,
                    )
                    patch.neighbor_sets = neighbor_sets
                    patch.eligible_source_wrap_indices = tuple(
                        int(v) for v in chunk.get("eligible_source_wrap_indices", ())
                    )
                    patches.append(patch)

            self.patches = patches
            self.sample_index = self._build_sample_index()
            self._neighbor_lookup = self._build_neighbor_lookup()
            if str(self.config.get("training_mode", "rowcol_hidden")) == "copy_neighbors":
                self.sample_index = [
                    (patch_idx, wrap_idx)
                    for patch_idx, wrap_idx in self.sample_index
                    if self._has_lower_and_upper_neighbors(patch_idx, wrap_idx)
                ]
                if not self.sample_index:
                    raise ValueError("No copy_neighbors source wraps have adjacent neighbors on both sides.")
            else:
                self.sample_index = [
                    (patch_idx, wrap_idx)
                    for patch_idx, wrap_idx in self.sample_index
                    if self._has_lower_and_upper_neighbors(patch_idx, wrap_idx)
                ]
                if not self.sample_index:
                    raise ValueError("No wraps have adjacent neighbors on both sides.")
            spec = self.config['cond_percent']
            self._cond_percent_min, self._cond_percent_max = float(spec[0]), float(spec[1])
        else:
            self._load_patch_metadata(patch_metadata)

    def __len__(self):
        return len(self.sample_index)

    def _build_sample_index(self):
        sample_index = []
        for patch_idx, patch in enumerate(self.patches):
            eligible = getattr(patch, "eligible_source_wrap_indices", None)
            wrap_indices = eligible if eligible is not None else range(len(patch.wraps))
            for wrap_idx in wrap_indices:
                sample_index.append((patch_idx, wrap_idx))
        return sample_index

    def _load_patch_metadata(self, patch_metadata):
        self.patches = patch_metadata['patches']
        self.sample_index = list(patch_metadata['sample_index'])
        self._neighbor_lookup = patch_metadata.get('neighbor_lookup', {})
        self._cond_percent_min, self._cond_percent_max = patch_metadata['cond_percent']

    def export_patch_metadata(self):
        return {
            'patches': self.patches,
            'sample_index': tuple(self.sample_index),
            'neighbor_lookup': self._neighbor_lookup,
            'cond_percent': (self._cond_percent_min, self._cond_percent_max),
        }

    def _build_copy_neighbor_pair_records(self, source_index):
        target_filter = str(self.config.get("copy_neighbor_target_side", "random"))
        record_mode = str(self.config.get("copy_neighbor_pair_record_mode", "all_directed"))
        seed = int(self.config.get("seed", 0)) + int(self.config.get("copy_neighbor_pair_seed_offset", 0))

        records = []
        for patch_idx, source_wrap_idx in source_index:
            neighbor_meta = self._neighbor_lookup.get((patch_idx, source_wrap_idx))
            if neighbor_meta is None:
                continue
            source_wrap_idx = int(source_wrap_idx)
            pair_records = []
            for target_wrap_idx in neighbor_meta["neighbor_wrap_indices"]:
                target_wrap_idx = int(target_wrap_idx)
                if target_wrap_idx == source_wrap_idx:
                    continue
                target_side = self._neighbor_side(patch_idx, source_wrap_idx, target_wrap_idx)
                if target_filter != "random" and target_side != target_filter:
                    continue
                pair_records.append({
                    "patch_idx": int(patch_idx),
                    "source_wrap_idx": int(source_wrap_idx),
                    "target_wrap_idx": int(target_wrap_idx),
                    "target_side": target_side,
                    "neighbor_wrap_indices": tuple(int(v) for v in neighbor_meta["neighbor_wrap_indices"]),
                })

            if record_mode == "one_per_source" and pair_records:
                key = f"{seed}:{patch_idx}:{source_wrap_idx}".encode("utf8")
                digest = hashlib.sha256(key).digest()
                chosen = int.from_bytes(digest[:8], "little") % len(pair_records)
                records.append(pair_records[chosen])
            else:
                records.extend(pair_records)
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

    def _build_neighbor_lookup(self):
        """Build (patch_idx, wrap_idx) -> precomputed neighbor-wrap metadata."""
        lookup = {}
        for patch_idx, patch in enumerate(self.patches):
            for wrap_idx, wrap in enumerate(patch.wraps):
                neighbor_indices = tuple(int(v) for v in wrap.get("neighbor_wrap_indices", ()))
                if not neighbor_indices:
                    continue
                source_label = int(wrap.get("wrap_label", wrap.get("wrap_id", -1)))
                lower = [
                    idx for idx in neighbor_indices
                    if int(patch.wraps[idx].get("wrap_label", patch.wraps[idx].get("wrap_id", -1))) < source_label
                ]
                upper = [
                    idx for idx in neighbor_indices
                    if int(patch.wraps[idx].get("wrap_label", patch.wraps[idx].get("wrap_id", -1))) > source_label
                ]
                selected = tuple([int(lower[0])] if lower else []) + tuple([int(upper[0])] if upper else [])
                lookup[(patch_idx, wrap_idx)] = {
                    "neighbor_wrap_indices": tuple(int(v) for v in neighbor_indices),
                    "lower_neighbor_wrap_indices": tuple(int(v) for v in lower),
                    "upper_neighbor_wrap_indices": tuple(int(v) for v in upper),
                    "rowcol_neighbor_wrap_indices": selected,
                }
        return lookup

    def _has_lower_and_upper_neighbors(self, patch_idx: int, wrap_idx: int) -> bool:
        neighbor_meta = self._neighbor_lookup.get((int(patch_idx), int(wrap_idx)))
        if neighbor_meta is None:
            return False
        return bool(neighbor_meta["lower_neighbor_wrap_indices"]) and bool(neighbor_meta["upper_neighbor_wrap_indices"])

    def _neighbor_side(self, patch_idx: int, source_wrap_idx: int, target_wrap_idx: int) -> str:
        patch = self.patches[int(patch_idx)]
        source = patch.wraps[int(source_wrap_idx)]
        target = patch.wraps[int(target_wrap_idx)]
        source_label = int(source.get("wrap_label", source.get("wrap_id", -1)))
        target_label = int(target.get("wrap_label", target.get("wrap_id", -1)))
        return "lower" if target_label < source_label else "upper"

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

        x_full, y_full, z_full = _upsample_world_surface(x_s, y_s, z_s, scale_y, scale_x)
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

    def create_copy_neighbor_masks(self, patch_idx: int, source_wrap_idx: int):
        patch_idx = int(patch_idx)
        patch = self.patches[patch_idx]
        source_wrap_idx = int(source_wrap_idx)
        neighbor_meta = self._neighbor_lookup.get((patch_idx, source_wrap_idx))
        if neighbor_meta is None:
            return None
        lower = tuple(int(v) for v in neighbor_meta.get("lower_neighbor_wrap_indices", ()))
        upper = tuple(int(v) for v in neighbor_meta.get("upper_neighbor_wrap_indices", ()))
        if not lower or not upper:
            return None
        lower_wrap_idx = int(lower[0])
        upper_wrap_idx = int(upper[0])

        source_world = self._extract_wrap_world_surface(patch, patch.wraps[source_wrap_idx])
        lower_world = self._extract_wrap_world_surface(patch, patch.wraps[lower_wrap_idx])
        upper_world = self._extract_wrap_world_surface(patch, patch.wraps[upper_wrap_idx])
        if source_world is None or lower_world is None or upper_world is None:
            return None

        z_min, _, y_min, _, x_min, _ = patch.world_bbox
        min_corner = np.round([z_min, y_min, x_min]).astype(np.int64)
        max_corner = min_corner + np.asarray(self.crop_size, dtype=np.int64)

        source_local = (source_world - min_corner).astype(np.float32, copy=False)
        lower_local = (lower_world - min_corner).astype(np.float32, copy=False)
        upper_local = (upper_world - min_corner).astype(np.float32, copy=False)

        vol_crop = _read_volume_crop_from_patch(
            patch,
            self.crop_size,
            min_corner,
            max_corner,
        )
        cond_gt = voxelize_surface_grid(source_local, self.crop_size).astype(np.float32, copy=False)
        behind_seg = voxelize_surface_grid(lower_local, self.crop_size).astype(np.float32, copy=False)
        front_seg = voxelize_surface_grid(upper_local, self.crop_size).astype(np.float32, copy=False)
        if not cond_gt.any() or not behind_seg.any() or not front_seg.any():
            return None
        return {
            "vol": torch.from_numpy(vol_crop).to(torch.float32),
            "cond_gt": torch.from_numpy(cond_gt).to(torch.float32),
            "behind_seg": torch.from_numpy(behind_seg).to(torch.float32),
            "front_seg": torch.from_numpy(front_seg).to(torch.float32),
            "source_surface_local": source_local,
            "lower_surface_local": lower_local,
            "upper_surface_local": upper_local,
            "min_corner": min_corner,
            "lower_wrap_idx": lower_wrap_idx,
            "upper_wrap_idx": upper_wrap_idx,
        }

    def create_neighbor_targets(
        self,
        *,
        cond_seg_gt: torch.Tensor,
        behind_seg: torch.Tensor,
        front_seg: torch.Tensor,
        cond_surface_local: torch.Tensor | None = None,
    ):
        crop_size = self.crop_size
        cond_np = cond_seg_gt.detach().cpu().numpy()
        behind_np = behind_seg.detach().cpu().numpy()
        front_np = front_seg.detach().cpu().numpy()

        cond_bin_full = cond_np > 0.5
        behind_bin_full = behind_np > 0.5
        front_bin_full = front_np > 0.5
        if not behind_bin_full.any() or not front_bin_full.any():
            return None

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

        result = {}
        if dir_priors_np is not None:
            result["dir_priors"] = torch.from_numpy(dir_priors_np).to(torch.float32)
        return result

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
        neighbor_meta = self._neighbor_lookup.get((patch_idx, wrap_idx))
        if neighbor_meta is None:
            return None
        neighbor_indices = tuple(int(v) for v in neighbor_meta.get("rowcol_neighbor_wrap_indices", ()))
        if len(neighbor_indices) != 2:
            return None

        patch = self.patches[patch_idx]
        neighbor_vox = np.zeros(crop_shape, dtype=np.uint8)
        for neighbor_wrap_idx in neighbor_indices:
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
        patch_idx, source_wrap_idx = self.sample_index[idx]
        patch_idx = int(patch_idx)
        source_wrap_idx = int(source_wrap_idx)
        mask_bundle = self.create_copy_neighbor_masks(patch_idx, source_wrap_idx)
        if mask_bundle is None:
            return self._resample_item(
                idx,
                "copy-neighbor triplet mask bundle missing",
                patch_idx=patch_idx,
                wrap_idx=source_wrap_idx,
                attempted_indices=_attempted_indices,
            )

        vol_crop = mask_bundle["vol"]
        cond_seg_gt = mask_bundle["cond_gt"]
        behind_seg = mask_bundle["behind_seg"]
        front_seg = mask_bundle["front_seg"]
        source_surface_source = torch.from_numpy(mask_bundle["source_surface_local"]).to(torch.float32)
        source_surface_local, source_surface_shape, source_keypoints, valid_source = (
            _prepare_cond_surface_keypoints(source_surface_source)
        )
        if not valid_source:
            return self._resample_item(
                idx,
                "copy-neighbor source surface invalid",
                patch_idx=patch_idx,
                wrap_idx=source_wrap_idx,
                attempted_indices=_attempted_indices,
            )

        augmented = augment_triplet_payload(
            augmentations=self._augmentations,
            crop_size=self.crop_size,
            vol_crop=vol_crop,
            cond_seg_gt=cond_seg_gt,
            behind_seg=behind_seg,
            front_seg=front_seg,
            cond_surface_local=source_surface_local,
            cond_surface_keypoints=source_keypoints,
            cond_surface_shape=source_surface_shape,
            restore_cond_surface_fn=self._restore_cond_surface_from_augmented,
        )
        vol_crop = augmented["vol_crop"]
        cond_seg_gt = augmented["cond_seg_gt"]
        behind_seg = augmented["behind_seg"]
        front_seg = augmented["front_seg"]
        source_surface_local = augmented["cond_surface_local"]

        target_payload = self.create_neighbor_targets(
            cond_seg_gt=cond_seg_gt,
            behind_seg=behind_seg,
            front_seg=front_seg,
            cond_surface_local=source_surface_local,
        )
        if target_payload is None:
            return self._resample_item(
                idx,
                "copy-neighbor dense displacement targets unavailable",
                patch_idx=patch_idx,
                wrap_idx=source_wrap_idx,
                attempted_indices=_attempted_indices,
            )

        if self._cond_local_perturb_active:
            cond_seg = self._conditioning_from_surface(
                cond_surface_local=source_surface_local,
                cond_seg_gt=cond_seg_gt,
            )
        else:
            cond_seg = cond_seg_gt.clone()
        if cond_seg is None:
            return self._resample_item(
                idx,
                "copy-neighbor conditioning segmentation unresolved",
                patch_idx=patch_idx,
                wrap_idx=source_wrap_idx,
                attempted_indices=_attempted_indices,
            )

        result = {
            "vol": vol_crop,
            "cond": cond_seg,
            "cond_gt": cond_seg_gt,
            "behind_seg": behind_seg,
            "front_seg": front_seg,
            "triplet_swap_enabled": torch.tensor(float(bool(self.apply_augmentation)), dtype=torch.float32),
        }
        result.update(target_payload)

        if not _validate_result_tensors(
            result,
            idx,
            enabled=self._validate_result_tensors_enabled,
        ):
            return self._resample_item(
                idx,
                "copy-neighbor result tensor validation failed",
                patch_idx=patch_idx,
                wrap_idx=source_wrap_idx,
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
    
