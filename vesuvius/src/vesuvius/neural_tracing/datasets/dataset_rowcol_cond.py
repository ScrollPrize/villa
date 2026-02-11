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

        config.setdefault('overlap_fraction', 0.0)
        config.setdefault('min_span_ratio', 1.0)
        config.setdefault('edge_touch_frac', 0.1)
        config.setdefault('edge_touch_min_count', 10)
        config.setdefault('edge_touch_pad', 0)
        config.setdefault('min_points_per_wrap', 100)
        config.setdefault('bbox_pad_2d', 0)
        config.setdefault('require_all_valid_in_bbox', True)
        config.setdefault('skip_chunk_if_any_invalid', False)
        config.setdefault('min_cond_span', 0.3)
        config.setdefault('inner_bbox_fraction', 0.7)
        cond_local_perturb = dict(config.get('cond_local_perturb') or {})
        cond_local_perturb.setdefault('enabled', True)
        cond_local_perturb.setdefault('probability', 0.35)
        cond_local_perturb.setdefault('num_blobs', [1, 3])
        cond_local_perturb.setdefault('points_affected', 10)
        cond_local_perturb.setdefault('sigma_fraction_range', [0.04, 0.10])
        cond_local_perturb.setdefault('amplitude_range', [0.25, 1.25])
        cond_local_perturb.setdefault('radius_sigma_mult', 2.5)
        cond_local_perturb.setdefault('max_total_displacement', 6.0)
        config['cond_local_perturb'] = cond_local_perturb

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

            cache_dir = Path(segments_path) / ".patch_cache" if segments_path else None
            chunk_results = find_world_chunk_patches(
                segments=scaled_segments,
                target_size=target_size,
                overlap_fraction=config.get('overlap_fraction', 0.0),
                min_span_ratio=config.get('min_span_ratio', 1.0),
                edge_touch_frac=config.get('edge_touch_frac', 0.1),
                edge_touch_min_count=config.get('edge_touch_min_count', 10),
                edge_touch_pad=config.get('edge_touch_pad', 0),
                min_points_per_wrap=config.get('min_points_per_wrap', 100),
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
        self._cond_percent_min, self._cond_percent_max = self._parse_cond_percent()

    def __len__(self):
        return len(self.patches)

    def _parse_cond_percent(self):
        spec = self.config['cond_percent']
        if not isinstance(spec, (list, tuple)) or len(spec) != 2:
            raise ValueError("cond_percent must be [min, max], e.g. [0.1, 0.5]")

        low, high = float(spec[0]), float(spec[1])
        if not (0.0 < low <= high < 1.0):
            raise ValueError("cond_percent values must satisfy 0 < min <= max < 1")
        return low, high

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

    def _maybe_perturb_conditioning_surface(self, cond_zyxs: np.ndarray) -> np.ndarray:
        """Apply local normal-direction pushes with Gaussian falloff on small regions."""
        cfg = self.config.get('cond_local_perturb', {})
        if not self.apply_augmentation or not cfg.get('enabled', True):
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

        patch = self.patches[idx]
        crop_size = self.crop_size  # tuple (D, H, W)
        target_shape = crop_size

        # select one wrap randomly for conditioning/masked split
        wrap = random.choice(patch.wraps)
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
        h_s, w_s = x_full_s.shape
        h_up = int(round(h_s / scale_y))
        w_up = int(round(w_s / scale_x))
        x_full = cv2.resize(x_full_s, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
        y_full = cv2.resize(y_full_s, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
        z_full = cv2.resize(z_full_s, (w_up, h_up), interpolation=cv2.INTER_LINEAR)

        z_min, z_max, y_min, y_max, x_min, x_max = patch.world_bbox
        in_bounds = (
            (z_full >= z_min) & (z_full < z_max) &
            (y_full >= y_min) & (y_full < y_max) &
            (x_full >= x_min) & (x_full < x_max)
        )

        valid_rows = np.any(in_bounds, axis=1)
        valid_cols = np.any(in_bounds, axis=0)
        if not valid_rows.any() or not valid_cols.any():
            return self[np.random.randint(len(self))]
        r0, r1 = np.where(valid_rows)[0][[0, -1]]
        c0, c1 = np.where(valid_cols)[0][[0, -1]]
        x_full = x_full[r0:r1+1, c0:c1+1]
        y_full = y_full[r0:r1+1, c0:c1+1]
        z_full = z_full[r0:r1+1, c0:c1+1]
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

        # if we're extrapolating, compute it with the extrapolation module
        if self.config['use_extrapolation']:
            extrap_result = compute_extrapolation(
                uv_cond=uv_cond,
                zyx_cond=cond_zyxs,
                uv_mask=uv_mask,
                zyx_mask=masked_zyxs,
                min_corner=min_corner,
                crop_size=crop_size,
                method=self.config['extrapolation_method'],
                cond_direction=cond_direction,
                degrade_prob=self.config.get('extrap_degrade_prob', 0.0),
                degrade_curvature_range=self.config.get('extrap_degrade_curvature_range', (0.001, 0.01)),
                degrade_gradient_range=self.config.get('extrap_degrade_gradient_range', (0.05, 0.2)),
            )
            if extrap_result is None:
                return self[np.random.randint(len(self))]
            extrap_surface = extrap_result['extrap_surface']
            extrap_coords_local = extrap_result['extrap_coords_local']
            gt_coords_local = extrap_result['gt_coords_local']

        volume = patch.volume
        if isinstance(volume, zarr.Group):
            volume = volume[str(patch.scale)]

        vol_crop = np.zeros(target_shape, dtype=volume.dtype)
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

        vol_crop = normalize_zscore(vol_crop)

        # convert cond and masked coords to crop-local coords (float for line interpolation)
        cond_zyxs_local_float = (cond_zyxs - min_corner).astype(np.float64)
        masked_zyxs_local_float = (masked_zyxs - min_corner).astype(np.float64)

        crop_shape = target_shape

        # voxelize with line interpolation between adjacent grid points
        cond_segmentation = voxelize_surface_grid(cond_zyxs_local_float, crop_shape)
        masked_segmentation = voxelize_surface_grid(masked_zyxs_local_float, crop_shape)

        # make sure we actually have some conditioning
        cond_nz = np.nonzero(cond_segmentation)
        if len(cond_nz[0]) == 0:
            return self[np.random.randint(len(self))]

        cond_segmentation_raw = cond_segmentation.copy()

        # add thickness to conditioning segmentation via dilation
        use_dilation = self.config.get('use_dilation', False)
        if use_dilation:
            dilation_radius = self.config.get('dilation_radius', 1.0)
            dist_from_cond = edt.edt(1 - cond_segmentation, parallel=1)
            cond_segmentation = (dist_from_cond <= dilation_radius).astype(np.float32)

        use_segmentation = self.config.get('use_segmentation', False)
        use_sdt = self.config['use_sdt']
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

                    oh_s, ow_s = ox_s.shape
                    oh_up = int(round(oh_s / o_scale_y))
                    ow_up = int(round(ow_s / o_scale_x))
                    ox_full = cv2.resize(ox_s, (ow_up, oh_up), interpolation=cv2.INTER_LINEAR)
                    oy_full = cv2.resize(oy_s, (ow_up, oh_up), interpolation=cv2.INTER_LINEAR)
                    oz_full = cv2.resize(oz_s, (ow_up, oh_up), interpolation=cv2.INTER_LINEAR)

                    in_bounds = (
                        (oz_full >= z_min_w) & (oz_full < z_max_w) &
                        (oy_full >= y_min_w) & (oy_full < y_max_w) &
                        (ox_full >= x_min_w) & (ox_full < x_max_w)
                    )
                    if not in_bounds.any():
                        continue
                    valid_rows = np.any(in_bounds, axis=1)
                    valid_cols = np.any(in_bounds, axis=0)
                    if not valid_rows.any() or not valid_cols.any():
                        continue
                    r0, r1 = np.where(valid_rows)[0][[0, -1]]
                    c0, c1 = np.where(valid_cols)[0][[0, -1]]
                    ox_full = ox_full[r0:r1+1, c0:c1+1]
                    oy_full = oy_full[r0:r1+1, c0:c1+1]
                    oz_full = oz_full[r0:r1+1, c0:c1+1]

                    other_zyxs = np.stack([oz_full, oy_full, ox_full], axis=-1)
                    other_zyxs_local = (other_zyxs - min_corner).astype(np.float64)

                    other_vox = voxelize_surface_grid(other_zyxs_local, crop_shape)
                    other_wraps_vox = np.maximum(other_wraps_vox, other_vox)

        vol_crop = torch.from_numpy(vol_crop).to(torch.float32)
        masked_seg = torch.from_numpy(masked_segmentation).to(torch.float32)
        cond_seg = torch.from_numpy(cond_segmentation).to(torch.float32)
        other_wraps_tensor = torch.from_numpy(other_wraps_vox).to(torch.float32)
        if use_segmentation:
            full_seg = torch.from_numpy(seg_dilated).to(torch.float32)
            seg_skel = torch.from_numpy(seg_skel).to(torch.float32)

        use_extrapolation = self.config['use_extrapolation']
        if use_extrapolation:
            sample_coords_local = extrap_coords_local
            target_coords_local = gt_coords_local
            point_weights_local = np.ones(sample_coords_local.shape[0], dtype=np.float32)

            if self.config.get('supervise_conditioning', False):
                cond_sample_coords_local = (cond_zyxs - min_corner).reshape(-1, 3).astype(np.float32)
                cond_target_coords_local = (cond_zyxs_unperturbed - min_corner).reshape(-1, 3).astype(np.float32)
                cond_weight = max(0.0, float(self.config.get('cond_supervision_weight', 0.1)))

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

                if cond_sample_coords_local.shape[0] > 0:
                    sample_coords_local = np.concatenate([sample_coords_local, cond_sample_coords_local], axis=0)
                    target_coords_local = np.concatenate([target_coords_local, cond_target_coords_local], axis=0)
                    cond_weights = np.full(cond_sample_coords_local.shape[0], cond_weight, dtype=np.float32)
                    point_weights_local = np.concatenate([point_weights_local, cond_weights], axis=0)

            extrap_surf = torch.from_numpy(extrap_surface).to(torch.float32)
            extrap_coords = torch.from_numpy(sample_coords_local).to(torch.float32)
            gt_coords = torch.from_numpy(target_coords_local).to(torch.float32)
            point_weights = torch.from_numpy(point_weights_local).to(torch.float32)
            n_points = len(extrap_coords)

        use_sdt = self.config['use_sdt']
        if use_sdt:
            sdt_tensor = torch.from_numpy(sdt).to(torch.float32)

        if self._augmentations is not None:
            seg_list = [masked_seg, cond_seg, other_wraps_tensor]
            seg_keys = ['masked_seg', 'cond_seg', 'other_wraps']
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
                    'segmentation', 'segmentation_skel']:
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
