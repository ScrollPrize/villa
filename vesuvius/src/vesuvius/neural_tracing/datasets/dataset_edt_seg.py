import edt
import zarr
import vesuvius.tifxyz as tifxyz
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import tifffile
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass
from vesuvius.tifxyz import Tifxyz

from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.image_proc.intensity.normalization import normalize_zscore

import os                                                                                                               
os.environ['OMP_NUM_THREADS'] = '1' # this is set to 1 because by default the edt package uses omp to threads the edt call 
                                    # which is problematic if you use multiple dataloader workers (thread contention smokes cpu)




def sample_unit_vector():
    """
    Uniform random unit vector on the sphere via Marsaglia method.
    This gives the plane an arbitrary 3D rotation (not axis-aligned).
    """
    while True:
        u1, u2 = np.random.uniform(-1, 1, 2)
        s = u1**2 + u2**2
        if s < 1:
            return np.array([2*u1*np.sqrt(1-s), 2*u2*np.sqrt(1-s), 1-2*s])


def compute_signed_distance_to_plane(shape, normal, center=None):
    """Compute signed distance from each voxel to plane through center."""
    D, H, W = shape
    if center is None:
        center = np.array([D/2, H/2, W/2])
    z, y, x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack([z, y, x], axis=-1).astype(np.float32) - center
    return np.dot(coords, normal)


def sample_plane_mask(seg_mask, target_range=(0.1, 0.4), max_attempts=10):
    """
    Sample a plane that splits seg_mask voxels into target ratio.

    Args:
        seg_mask: Binary segmentation mask [D, H, W]
        target_range: Tuple (min_ratio, max_ratio) for the smaller side
        max_attempts: Number of random plane orientations to try

    Returns:
        plane_mask: Boolean array [D, H, W] - True for the smaller (conditioning) side
    """
    seg_voxels = seg_mask > 0
    if not seg_voxels.any():
        return np.zeros_like(seg_mask, dtype=bool)

    total_seg = seg_voxels.sum()

    # Compute centroid of segmentation voxels so plane passes through the GT
    seg_coords = np.argwhere(seg_voxels)  # [N, 3] array of (z, y, x)
    seg_centroid = seg_coords.mean(axis=0)  # [3,] centroid

    for _ in range(max_attempts):
        normal = sample_unit_vector()
        signed_dist = compute_signed_distance_to_plane(seg_mask.shape, normal, center=seg_centroid)
        seg_dists = signed_dist[seg_voxels]

        # Binary search for offset achieving target split
        lo, hi = float(seg_dists.min()), float(seg_dists.max())

        for _ in range(50):
            mid = (lo + hi) / 2
            ratio = (seg_dists < mid).sum() / total_seg

            if target_range[0] <= ratio <= target_range[1]:
                # Found valid split
                return signed_dist < mid
            elif ratio < target_range[0]:
                lo = mid
            else:
                hi = mid

        # Check final ratio
        ratio = (seg_dists < mid).sum() / total_seg
        if target_range[0] <= ratio <= target_range[1]:
            return signed_dist < mid

    # Fallback: return best-effort mask from last attempt
    return signed_dist < mid

@dataclass
class Patch:
    seg: Tifxyz                           # Reference to the segment
    volume: zarr.Array                    # zarr volume
    scale: float                          # volume_scale from config
    grid_bbox: Tuple[int, int, int, int]  # (row_min, row_max, col_min, col_max) in the tifxyz grid
    world_bbox: Tuple[float, ...]         # (z_min, z_max, y_min, y_max, x_min, x_max) in world coordinates (volume coordinates)

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
        config.setdefault('dilation_radius', 1)  # voxels

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
                seg_patches = seg_scaled.get_patches_3d(target_size)

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
        r_min, r_max, c_min, c_max = patch.grid_bbox

        # patch locations in the tifxyz grid are discovered/computed at downscaled (on-disk) resolution , 
        # we have to scale them up to match the real coordinates at full resolution
        scale_y, scale_x = patch.seg._scale
        r_min_full = int(r_min / scale_y)
        r_max_full = int(r_max / scale_y)
        c_min_full = int(c_min / scale_x)
        c_max_full = int(c_max / scale_x)

        # here we set the segment to "full resolution" , which interpolates/scales the on-disk tifxyz to "full" res scale
        # this is necessary because i want to use the tifxyz points as a dense segmentation mask
        patch.seg.use_full_resolution()
        x, y, z, valid = patch.seg[r_min_full:r_max_full, c_min_full:c_max_full]
        zyxs = np.stack([z, y, x], axis=-1)

        volume = patch.volume
        if isinstance(volume, zarr.Group):
            volume = volume[str(patch.scale)]

        crop_size = self.crop_size  # tuple (D, H, W)
        target_shape = crop_size

        # world_bbox is centered on surface centroid and extends to target size
        z_min, z_max, y_min, y_max, x_min, x_max = patch.world_bbox
        min_corner = np.round([z_min, y_min, x_min]).astype(np.int64)
        max_corner = min_corner + np.array(crop_size)

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

        gt_segmentation = np.zeros(target_shape, dtype=np.float32)

        # get valid coords, and convert to crop-local coords so we dont have gigantic arrays
        z_coords = zyxs[..., 0][valid]
        y_coords = zyxs[..., 1][valid]
        x_coords = zyxs[..., 2][valid]

        z_idx = (z_coords - min_corner[0]).astype(np.int64)
        y_idx = (y_coords - min_corner[1]).astype(np.int64)
        x_idx = (x_coords - min_corner[2]).astype(np.int64)

        crop_shape = target_shape
        in_bounds = (
            (z_idx >= 0) & (z_idx < crop_shape[0]) &
            (y_idx >= 0) & (y_idx < crop_shape[1]) &
            (x_idx >= 0) & (x_idx < crop_shape[2])
        )
        z_idx, y_idx, x_idx = z_idx[in_bounds], y_idx[in_bounds], x_idx[in_bounds]

        gt_segmentation[z_idx, y_idx, x_idx] = 1

        # if by some change we get no valid points , just grab a different random sample
        if len(z_idx) == 0:
            return self[np.random.randint(len(self))]

        # sample plane mask for conditioning input
        # - place an arbitrarily rotated plane within the volume
        # - shift it along its normals until it meets our "split" ratio
        # - provide the "smaller" side of the plane as conditioning input
        # - create a mask of this side, to mask out the loss on the segmentation
        plane_mask = sample_plane_mask(
            gt_segmentation,
            target_range=(
                self.config.get('plane_split_ratio_min', 0.05),
                self.config.get('plane_split_ratio_max', 0.30),
            ),
            max_attempts=10,
        )

        # seg gt on the smaller side of the plane, zeroed on the larger side
        cond = gt_segmentation * plane_mask.astype(np.float32)

        # doing 50 edts like this might look stupid (and it may well be), but i need to add some "thickness"
        # to the segmentation gt, and an edt happens to be the smoothest/most accurate way to do so.
        distance_from_surface = edt.edt(1 - gt_segmentation, parallel=1)
        seg_gt_dilated = (distance_from_surface <= self.config['dilation_radius']).astype(np.float32)

        if self.config['use_sdf']:
            seg_dt = edt.sdf(seg_gt_dilated, parallel=1)
        else:
            seg_dt = edt.edt(seg_gt_dilated, parallel=1)

        # skeleton is the centerline of the dilated region = original surface points
        # (where distance_from_surface == 0)
        skeleton = (distance_from_surface == 0).astype(np.float32)

        vol_crop = torch.from_numpy(vol_crop).to(torch.float32)
        seg_gt_dilated = torch.from_numpy(seg_gt_dilated).to(torch.float32)
        seg_dt = torch.from_numpy(seg_dt).to(torch.float32)
        skeleton = torch.from_numpy(skeleton).to(torch.float32)
        cond = torch.from_numpy(cond).to(torch.float32)
        plane_mask = torch.from_numpy(plane_mask.astype(np.float32)).to(torch.float32)

        if self._augmentations is not None:
            # Augmentations expect [C, D, H, W] tensors
            # image: intensity augmentation applied
            # dist_map: geometric transforms only (for dt, skeleton)
            # segmentation: geometric transforms only (for seg, cond, plane_mask)
            while True:
                augmented = self._augmentations(
                    image=vol_crop[None],                                              # [1, D, H, W]
                    dist_map=torch.stack([seg_dt, skeleton], dim=0),                   # [2, D, H, W]
                    segmentation=torch.stack([seg_gt_dilated, cond, plane_mask], dim=0),  # [3, D, H, W]
                )
                if augmented['image'] is not None and augmented['dist_map'] is not None:
                    break

            vol_crop = augmented['image'].squeeze(0)       # [D, H, W]
            seg_gt_dilated = augmented['segmentation'][0]  # [D, H, W]
            cond = augmented['segmentation'][1]            # [D, H, W]
            plane_mask = augmented['segmentation'][2]      # [D, H, W]
            seg_dt = augmented['dist_map'][0]              # [D, H, W]
            skeleton = augmented['dist_map'][1]            # [D, H, W]

        data_dict = {
            "vol":  vol_crop,         # raw volume crop
            "seg":  seg_gt_dilated,   # segmentation gt (full)
            "dt":   seg_dt,           # edt/sdf
            "cond": cond,             # partial seg GT as conditioning (plane-masked)
            "skel": skeleton,         # skeleton for medial surface recall loss
            "plane_mask": plane_mask, # 1=known (conditioned), 0=unknown (to predict)
        }

        return data_dict
    


if __name__ == "__main__":
    config_path = "/home/sean/Documents/villa/vesuvius/src/vesuvius/neural_tracing/configs/config_test.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_ds = EdtSegDataset(config)
    print(f"Dataset has {len(train_ds)} patches")

    out_dir = Path("/tmp/edt_seg_debug")
    out_dir.mkdir(exist_ok=True)

    num_samples = min(5, len(train_ds))
    for i in range(num_samples):
        sample = train_ds[i]
        for key, tensor in sample.items():
            subdir = out_dir / key
            subdir.mkdir(exist_ok=True)
            tifffile.imwrite(subdir / f"{i:03d}.tif", tensor.numpy())

        print(f"[{i+1}/{num_samples}] Saved sample {i:03d}")

    print(f"Output saved to {out_dir}")
