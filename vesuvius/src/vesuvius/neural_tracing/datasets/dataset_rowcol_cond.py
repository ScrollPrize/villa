import edt
import zarr
import vesuvius.tifxyz as tifxyz
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import tifffile
from pathlib import Path
from vesuvius.neural_tracing.datasets.common import Patch, compute_heatmap_targets
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.image_proc.intensity.normalization import normalize_zscore
import random
from vesuvius.neural_tracing.datasets.extrapolation import compute_extrapolation

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

        # Parse crop_size - can be int (cubic) or list of 3 ints [D, H, W]
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

        config.setdefault('use_sdf', True)
        config.setdefault('use_sdt', False)
        config.setdefault('dilation_radius', 1)  # voxels
        config.setdefault('cond_percent', 0.5)
        config.setdefault('use_extrapolation', True)
        config.setdefault('extrapolation_method', 'linear_edge')
        config.setdefault('force_recompute_patches', False)
        config.setdefault('use_heatmap_targets', False)
        config.setdefault('heatmap_step_size', 10)
        config.setdefault('heatmap_step_count', 5)
        config.setdefault('heatmap_sigma', 2.0)
        
        # Setup augmentations
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

            for seg in dataset_segments:
                # retarget segment to match the volume resolution level
                retarget_factor = 2 ** volume_scale
                seg_scaled = seg.retarget(retarget_factor)
                seg_scaled.volume = volume
                seg_patches = seg_scaled.get_patches_3d(
                    target_size,
                    force_recompute=config.get('force_recompute_patches', False),
                )

                for grid_bbox, world_bbox in seg_patches:
                    patches.append(Patch(
                        seg=seg_scaled,
                        volume=volume,
                        scale=volume_scale,
                        grid_bbox=grid_bbox,
                        world_bbox=world_bbox,
                    ))

        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        
        patch = self.patches[idx]
        patch.seg.use_full_resolution() # scale/interpolate to "full" resolution
                                        # slicing is lazy, we only access the part we need at this res

        r_min, r_max, c_min, c_max = patch.grid_bbox
        crop_size = self.crop_size  # tuple (D, H, W)
        target_shape = crop_size

        # world_bbox is centered on surface centroid and extends to target size
        fallback_min_corner = None
        if patch.world_bbox is not None:
            z_min, z_max, y_min, y_max, x_min, x_max = patch.world_bbox
            fallback_min_corner = np.round([z_min, y_min, x_min]).astype(np.int64)

        scale_y, scale_x = patch.seg._scale
        r_min_full = int(r_min / scale_y)
        r_max_full = int(r_max / scale_y)
        c_min_full = int(c_min / scale_x)
        c_max_full = int(c_max / scale_x)

        conditioning_percent = self.config['cond_percent']
        r_split = r_min_full + round((r_max_full - r_min_full) * conditioning_percent)
        c_split = c_min_full + round((c_max_full - c_min_full) * conditioning_percent)

        cond_direction = random.choice(["left", "right", "up", "down"])

        if cond_direction == "left":
            # the left half of the patch is conditioning, mask the right half
            x_cond, y_cond, z_cond, valid_cond = patch.seg[r_min_full:r_max_full, c_min_full:c_split]
            x_mask, y_mask, z_mask, valid_mask = patch.seg[r_min_full:r_max_full, c_split:c_max_full]
            
            rows = np.arange(r_min_full, r_max_full)
            uv_cond = np.stack(np.meshgrid(rows, np.arange(c_min_full, c_split), indexing='ij'), axis=-1)
            uv_mask = np.stack(np.meshgrid(rows, np.arange(c_split, c_max_full), indexing='ij'), axis=-1)
                               
        elif cond_direction == "right":
            # the right half of the patch is conditioning, mask the left half
            x_cond, y_cond, z_cond, valid_cond = patch.seg[r_min_full:r_max_full, c_split:c_max_full]
            x_mask, y_mask, z_mask, valid_mask = patch.seg[r_min_full:r_max_full, c_min_full:c_split]

            rows = np.arange(r_min_full, r_max_full)
            uv_cond = np.stack(np.meshgrid(rows, np.arange(c_split, c_max_full), indexing='ij'), axis=-1)
            uv_mask = np.stack(np.meshgrid(rows, np.arange(c_min_full, c_split), indexing='ij'), axis=-1)

        elif cond_direction == "up":
            # the top half of the patch is conditioning, mask the bottom half
            x_cond, y_cond, z_cond, valid_cond = patch.seg[r_min_full:r_split, c_min_full:c_max_full]
            x_mask, y_mask, z_mask, valid_mask = patch.seg[r_split:r_max_full, c_min_full:c_max_full]

            cols = np.arange(c_min_full, c_max_full)
            uv_cond = np.stack(np.meshgrid(np.arange(r_min_full, r_split), cols, indexing='ij'), axis=-1)
            uv_mask = np.stack(np.meshgrid(np.arange(r_split, r_max_full), cols, indexing='ij'), axis=-1)

        elif cond_direction == "down":
            # the bottom half of the patch is conditioning, mask the top half
            x_cond, y_cond, z_cond, valid_cond = patch.seg[r_split:r_max_full, c_min_full:c_max_full]
            x_mask, y_mask, z_mask, valid_mask = patch.seg[r_min_full:r_split, c_min_full:c_max_full]

            cols = np.arange(c_min_full, c_max_full)
            uv_cond = np.stack(np.meshgrid(np.arange(r_split, r_max_full), cols, indexing='ij'), axis=-1)
            uv_mask = np.stack(np.meshgrid(np.arange(r_min_full, r_split), cols, indexing='ij'), axis=-1)

        # if either half contains invalid points, grab a different sample
        if not valid_cond.all() or not valid_mask.all():
            return self[np.random.randint(len(self))]

        cond_zyxs = np.stack([z_cond, y_cond, x_cond], axis=-1)
        masked_zyxs = np.stack([z_mask, y_mask, x_mask], axis=-1)

        # Center the crop on the combined conditioning+masked surface coords.
        combined_zyxs = np.concatenate(
            [cond_zyxs.reshape(-1, 3), masked_zyxs.reshape(-1, 3)],
            axis=0,
        )
        if combined_zyxs.size == 0:
            if fallback_min_corner is None:
                return self[np.random.randint(len(self))]
            min_corner = fallback_min_corner
        else:
            finite_mask = np.isfinite(combined_zyxs).all(axis=1)
            if not finite_mask.any():
                if fallback_min_corner is None:
                    return self[np.random.randint(len(self))]
                min_corner = fallback_min_corner
            else:
                center_zyx = combined_zyxs[finite_mask].mean(axis=0)
                min_corner = np.round(center_zyx - np.array(crop_size) / 2.0).astype(np.int64)

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
            )
            if extrap_result is None:
                return self[np.random.randint(len(self))]
            extrap_surface = extrap_result['extrap_surface']
            extrap_coords_local = extrap_result['extrap_coords_local']
            gt_displacement = extrap_result['gt_displacement']

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

        masked_segmentation = np.zeros(target_shape, dtype=np.float32)
        cond_segmentation = np.zeros(target_shape, dtype=np.float32)

        # convert cond and masked coords to crop-local coords
        cond_zyxs_local = (cond_zyxs - min_corner).astype(np.int64)
        masked_zyxs_local = (masked_zyxs - min_corner).astype(np.int64)

        crop_shape = target_shape

        cond_in_bounds = (
            (cond_zyxs_local[..., 0] >= 0) & (cond_zyxs_local[..., 0] < crop_shape[0]) &
            (cond_zyxs_local[..., 1] >= 0) & (cond_zyxs_local[..., 1] < crop_shape[1]) &
            (cond_zyxs_local[..., 2] >= 0) & (cond_zyxs_local[..., 2] < crop_shape[2])
        )
        cond_zyxs_local = cond_zyxs_local[cond_in_bounds]

        masked_in_bounds = (
            (masked_zyxs_local[..., 0] >= 0) & (masked_zyxs_local[..., 0] < crop_shape[0]) &
            (masked_zyxs_local[..., 1] >= 0) & (masked_zyxs_local[..., 1] < crop_shape[1]) &
            (masked_zyxs_local[..., 2] >= 0) & (masked_zyxs_local[..., 2] < crop_shape[2])
        )

        masked_zyxs_local = masked_zyxs_local[masked_in_bounds]
        cond_segmentation[cond_zyxs_local[:, 0], cond_zyxs_local[:, 1], cond_zyxs_local[:, 2]] = 1
        masked_segmentation[masked_zyxs_local[:, 0], masked_zyxs_local[:, 1], masked_zyxs_local[:, 2]] = 1

        if self.config['use_sdt']:
            # Combine cond + masked into full segmentation
            full_segmentation = np.zeros(target_shape, dtype=np.float32)
            all_zyxs_local = np.concatenate([cond_zyxs_local, masked_zyxs_local], axis=0)

            # Filter to in-bounds
            in_bounds = (
                (all_zyxs_local[..., 0] >= 0) & (all_zyxs_local[..., 0] < crop_shape[0]) &
                (all_zyxs_local[..., 1] >= 0) & (all_zyxs_local[..., 1] < crop_shape[1]) &
                (all_zyxs_local[..., 2] >= 0) & (all_zyxs_local[..., 2] < crop_shape[2])
            )
            all_zyxs_local = all_zyxs_local[in_bounds]
            full_segmentation[all_zyxs_local[:, 0], all_zyxs_local[:, 1], all_zyxs_local[:, 2]] = 1

            dilation_radius = self.config.get('dilation_radius', 1.0)
            distance_from_surface = edt.edt(1 - full_segmentation, parallel=1)
            seg_dilated = (distance_from_surface <= dilation_radius).astype(np.float32)
            sdt = edt.sdf(seg_dilated, parallel=1).astype(np.float32)

        # Generate heatmap targets for expected positions in masked region
        use_heatmap = self.config['use_heatmap_targets']
        if use_heatmap:
            effective_step = int(self.config['heatmap_step_size'] * (2 ** patch.scale))
            heatmap_tensor = compute_heatmap_targets(
                cond_direction=cond_direction,
                r_split=r_split, c_split=c_split,
                r_min_full=r_min_full, r_max_full=r_max_full,
                c_min_full=c_min_full, c_max_full=c_max_full,
                patch_seg=patch.seg,
                min_corner=min_corner,
                crop_size=crop_size,
                step_size=effective_step,
                step_count=self.config['heatmap_step_count'],
                sigma=self.config['heatmap_sigma'],
                axis_1d=self._heatmap_axes[0],
            )
            if heatmap_tensor is None:
                return self[np.random.randint(len(self))]

        vol_crop = torch.from_numpy(vol_crop).to(torch.float32)
        masked_seg = torch.from_numpy(masked_segmentation).to(torch.float32)
        cond_seg = torch.from_numpy(cond_segmentation).to(torch.float32)

        use_extrapolation = self.config['use_extrapolation']
        if use_extrapolation:
            extrap_surf = torch.from_numpy(extrap_surface).to(torch.float32)
            extrap_coords = torch.from_numpy(extrap_coords_local).to(torch.float32)
            gt_disp = torch.from_numpy(gt_displacement).to(torch.float32)

        use_sdt = self.config['use_sdt']
        if use_sdt:
            sdt_tensor = torch.from_numpy(sdt).to(torch.float32)

        if self._augmentations is not None:
            seg_list = [masked_seg, cond_seg]
            seg_keys = ['masked_seg', 'cond_seg']
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
                aug_kwargs['keypoints'] = extrap_coords
                aug_kwargs['gt_displacement'] = gt_disp
                aug_kwargs['vector_keys'] = ['gt_displacement']
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
                elif key == 'extrap_surf':
                    extrap_surf = augmented['segmentation'][i]

            if dist_list:
                for i, key in enumerate(dist_keys):
                    if key == 'sdt':
                        sdt_tensor = augmented['dist_map'][i]

            if use_extrapolation:
                extrap_coords = augmented['keypoints']
                gt_disp = augmented['gt_displacement']
            if use_heatmap:
                heatmap_tensor = augmented['heatmap_target'].squeeze(0)

        result = {
            "vol": vol_crop,                 # raw volume crop
            "cond": cond_seg,                # conditioning segmentation
            "masked_seg": masked_seg,        # masked (target) segmentation
        }

        if use_extrapolation:
            result["extrap_surface"] = extrap_surf     # extrapolated surface voxelization
            result["extrap_coords"] = extrap_coords    # (N, 3) coords for sampling predicted field
            result["gt_displacement"] = gt_disp        # (N, 3) ground truth displacement

        if use_sdt:
            result["sdt"] = sdt_tensor                 # signed distance transform of full (dilated) segmentation

        if use_heatmap:
            result["heatmap_target"] = heatmap_tensor  # (D, H, W) gaussian heatmap at expected positions

        # Validate all tensors are non-empty and contain no NaN/Inf
        for key, tensor in result.items():
            if tensor.numel() == 0:
                raise ValueError(f"Empty tensor for '{key}' at index {idx}")
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN values in '{key}' at index {idx}")
            if torch.isinf(tensor).any():
                raise ValueError(f"Inf values in '{key}' at index {idx}")

        return result
    


if __name__ == "__main__":
    config_path = "/home/sean/Documents/villa/vesuvius/src/vesuvius/neural_tracing/configs/config_rowcol_cond.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_ds = EdtSegDataset(config)
    print(f"Dataset has {len(train_ds)} patches")

    out_dir = Path("/tmp/edt_seg_debug")
    out_dir.mkdir(exist_ok=True)

    num_samples = min(10, len(train_ds))
    for i in range(num_samples):
        sample = train_ds[i]

        # Save 3D volumes as tif
        for key in ['vol', 'cond', 'masked_seg', 'extrap_surface', 'sdt', 'heatmap_target']:
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
