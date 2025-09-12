import os
import cv2
import zarr
import json
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Patch:
    zyxs: torch.Tensor
    scale: torch.Tensor
    volume: zarr.Array

    def _get_face_indices(self):
        h, w = torch.tensor(self.zyxs.shape[:2]) - 1
        indices = torch.arange(h * w).view(h, w)
        top_left = indices[:-1, :-1].flatten()
        top_right = indices[:-1, 1:].flatten()
        bottom_left = indices[1:, :-1].flatten()
        bottom_right = indices[1:, 1:].flatten()
        return torch.cat([
            torch.stack([bottom_left, top_left, top_right], dim=1),
            torch.stack([bottom_left, top_right, bottom_right], dim=1)
        ], dim=0)

    def __post_init__(self):
        # Construct the valid *quads* mask; the ij'th element says whether all four corners of the quad with min-corner at ij are valid
        self.valid_mask = torch.any(self.zyxs[:-1, :-1] != -1, dim=-1) & torch.any(self.zyxs[1:, :-1] != -1, dim=-1) & torch.any(self.zyxs[:-1, 1:] != -1, dim=-1) & torch.any(self.zyxs[1:, 1:] != -1, dim=-1)
        self.valid_indices = torch.stack(torch.where(self.valid_mask), dim=-1)
        assert len(self.valid_indices) > 0
        self.area = (~self.valid_mask).sum() * (1 / self.scale).prod()

    def retarget(self, factor):
        # Retarget the patch to a volume downsampled by the given factor
        return Patch(
            torch.where((self.zyxs == -1).all(dim=-1, keepdim=True), -1, self.zyxs / factor),
            self.scale * factor,
            self.volume
        )


class LineDataset(torch.utils.data.IterableDataset):
    def __init__(self, config):
        self._config = config
        self._patches = load_datasets(config)

    def __iter__(self):

        areas = torch.tensor([patch.area for patch in self._patches])
        area_weights = areas / areas.sum()
        point_spacing = self._config['point_spacing']
        total_points = self._config['context_points'] + self._config['generated_points']

        while True:
            patch = random.choices(self._patches, weights=area_weights)[0]

            # Sample a random valid quad in the patch, then a point in that quad
            random_idx = torch.randint(len(patch.valid_indices) - 1, size=[])
            start_quad_ij = patch.valid_indices[random_idx]
            start_ij = start_quad_ij + torch.rand(size=[2])
            
            # Sample second point at random angle; reject if outside the patch or quad not valid
            angle = torch.rand(size=[]) * 2 * torch.pi
            distance = point_spacing * total_points
            end_ij = start_ij + distance * torch.tensor([torch.cos(angle), torch.sin(angle)])
            if torch.any(end_ij < 0) or torch.any(end_ij >= torch.tensor(patch.zyxs.shape[:2])):
                continue
            if not patch.valid_mask[*end_ij.int()]:
                continue

            # FIXME: sometimes should sample shorter context (min length of two) to enable bootstrapping traces
            #  from just a pair of points

            coeffs = torch.linspace(0, 1, total_points)
            ijs = torch.lerp(start_ij, end_ij, coeffs[:, None])
            zyxs = F.grid_sample(patch.zyxs, ijs[None, :, ::-1] / patch.zyxs.shape[:2] * 2 - 1, align_corners=True, mode='bilinear', padding_mode='border')

            yield {
                'zyxs': zyxs,
                'patch': ...,
            }


class PatchInCubeDataset(torch.utils.data.IterableDataset):

    def __init__(self, config):
        self._config = config
        self._patches = load_datasets(config)

    def _crop_volume_with_padding(self, volume, center_zyx, crop_size):
        """Crop volume around center point, padding with zeros if out of bounds."""
        crop_min = (center_zyx - crop_size // 2).int()
        crop_max = crop_min + crop_size
        
        # Clamp to volume bounds
        actual_min = torch.maximum(crop_min, torch.zeros_like(crop_min))
        actual_max = torch.minimum(crop_max, torch.tensor(volume.shape))
        
        # Extract valid portion and convert to tensor
        volume_crop = torch.from_numpy(volume[
            actual_min[0]:actual_max[0], 
            actual_min[1]:actual_max[1], 
            actual_min[2]:actual_max[2]
        ]).to(torch.float32) / 255.
        
        # Pad if needed
        pad_before = actual_min - crop_min
        pad_after = crop_max - actual_max
        if torch.any(pad_before > 0) or torch.any(pad_after > 0):
            paddings = pad_before[2], pad_after[2], pad_before[1], pad_after[1], pad_before[0], pad_after[0]
            volume_crop = F.pad(volume_crop, paddings, mode='constant', value=0)
        
        return volume_crop, crop_min

    def _mark_context_point(self, volume, point):
        center = point.int()
        offsets = torch.tensor([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
        for offset in offsets:
            volume[tuple(center + offset)] = 1.

    def _get_zyx_from_patch(self, ij, patch):
        normalized_ij = ij / torch.tensor(patch.zyxs.shape[:2]) * 2 - 1
        interpolated = F.grid_sample(
            rearrange(patch.zyxs, 'h w c -> 1 c h w'), 
            rearrange(normalized_ij.flip(-1), 'xy -> 1 1 1 xy'), 
            align_corners=True, 
            mode='bilinear', 
            padding_mode='border'
        )
        return rearrange(interpolated, '1 c 1 1 -> c')

    def __iter__(self):

        areas = torch.tensor([patch.area for patch in self._patches])
        area_weights = areas / areas.sum()
        context_point_distance = self._config['context_point_distance']
        num_context_points = self._config['num_context_points']
        assert num_context_points >= 1  # ...since we always include one point at the center of the patch
        crop_size = torch.tensor(self._config['crop_size'])

        while True:
            patch = random.choices(self._patches, weights=area_weights)[0]

            # Sample a random valid quad in the patch, then a point in that quad
            random_idx = torch.randint(len(patch.valid_indices) - 1, size=[])
            start_quad_ij = patch.valid_indices[random_idx]
            center_ij = start_quad_ij + torch.rand(size=[2])

            # Sample other nearby random points as additional context; reject if outside patch
            angle = torch.rand(size=[num_context_points]) * 2 * torch.pi
            distance = context_point_distance * patch.scale
            context_ij = center_ij + distance * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
            if torch.any(context_ij < 0) or torch.any(context_ij >= torch.tensor(patch.zyxs.shape[:2])):
                continue
            if not patch.valid_mask[*context_ij.int().T].all():
                continue

            center_zyx = self._get_zyx_from_patch(center_ij, patch)
            context_zyxs = [self._get_zyx_from_patch(context_ij, patch) for context_ij in context_ij]

            # Crop ROI out of the volume; mark context points
            volume_crop, min_corner_zyx = self._crop_volume_with_padding(patch.volume, center_zyx, crop_size)
            self._mark_context_point(volume_crop, center_zyx - min_corner_zyx)
            for context_zyx in context_zyxs:
                self._mark_context_point(volume_crop, context_zyx - min_corner_zyx)

            # Build the confidence map, by rasterising the surface
            quad_centers = 0.5 * (patch.zyxs[1:, 1:] + patch.zyxs[:-1, :-1])
            quad_in_crop = patch.valid_mask & torch.all(quad_centers >= min_corner_zyx, dim=-1) & torch.all(quad_centers < min_corner_zyx + crop_size, dim=-1)
            filtered_quads_zyxs = torch.stack([
                torch.stack([
                    patch.zyxs[:-1, :-1][quad_in_crop],
                    patch.zyxs[:-1, 1:][quad_in_crop],
                ], dim=1),
                torch.stack([
                    patch.zyxs[1:, :-1][quad_in_crop],
                    patch.zyxs[1:, 1:][quad_in_crop],
                ], dim=1),
            ], dim=1)  # quad, top/bottom, left/right, zyx
            oversample_factor = 2
            points_per_side = (1 / patch.scale + 0.5).int() * oversample_factor
            v_points = torch.arange(points_per_side[0], dtype=torch.float32) / points_per_side[0]
            u_points = torch.arange(points_per_side[1], dtype=torch.float32) / points_per_side[1]
            points_covering_quads = torch.lerp(filtered_quads_zyxs[:, None, 0, :], filtered_quads_zyxs[:, None, 1, :], v_points[None, :, None, None])
            points_covering_quads = torch.lerp(points_covering_quads[:, :, None, 0], points_covering_quads[:, :, None, 1], u_points[None, None, :, None])
            indices_in_crop = (points_covering_quads - min_corner_zyx + 0.5).int().clip(0, crop_size - 1)
            confidence = torch.zeros([crop_size, crop_size, crop_size], dtype=torch.float32)
            confidence[*indices_in_crop.view(-1, 3).T] = 1.

            # Also construct UV map on the surface
            # filtered_quads_uvs = torch.stack(torch.where(quad_in_crop), dim=-1)
            # uv_grid = torch.stack(torch.meshgrid(u_points, v_points, indexing='ij'), dim=-1)
            # interpolated_uvs = filtered_quads_uvs[:, None, None, :] + uv_grid
            # uvs = torch.zeros([crop_size, crop_size, crop_size, 2], dtype=torch.float32)
            # uvs[*indices_in_crop.view(-1, 3).T] = interpolated_uvs.view(-1, 2)

            # Also construct UV map, on the surface and extrapolated into free space
            from torch_geometric.nn.unpool import knn_interpolate
            filtered_quads_uvs = torch.stack(torch.where(quad_in_crop), dim=-1)
            uv_grid = torch.stack(torch.meshgrid(u_points, v_points, indexing='ij'), dim=-1)
            interpolated_uvs = filtered_quads_uvs[:, None, None, :] + uv_grid
            undersample_factor = 4
            uvs = knn_interpolate(
                x=interpolated_uvs[:, ::undersample_factor, ::undersample_factor, :].reshape(-1, 2),
                pos_x=indices_in_crop[:, ::undersample_factor, ::undersample_factor, :].reshape(-1, 3).float(),
                pos_y=torch.stack(torch.meshgrid(*[torch.arange(crop_size)] *  3, indexing='ij'), dim=-1).view(-1, 3).float(),
                k=1,
            ).reshape([crop_size, crop_size, crop_size, 2])
            uvs = (uvs - center_ij) / ...
            # FIXME: we need to have UVs in a range that is diffusion-compatible, and also that
            #  has consistent semantic meaning across different patches

            #TODO: include full 2d slices for additional context

            yield {
                'volume': ...,
                'confidence': ...,
                'uv': ...,
            }


class PatchWithCubeDataset(torch.utils.data.IterableDataset):

    def __init__(self, config):
        self._config = config
        self._patches = load_datasets(config)

    def _get_crop_from_volume(self, volume, center_zyx, crop_size):
        """Crop volume around center point, padding with zeros if needed"""
        crop_min = (center_zyx - crop_size // 2).int()
        crop_max = crop_min + crop_size

        # Clamp to volume bounds
        actual_min = torch.maximum(crop_min, torch.zeros_like(crop_min))
        actual_max = torch.minimum(crop_max, torch.tensor(volume.shape))

        # Extract valid portion and convert to tensor
        volume_crop = torch.from_numpy(volume[
            actual_min[0]:actual_max[0],
            actual_min[1]:actual_max[1],
            actual_min[2]:actual_max[2]
        ]).to(torch.float32) / 255.

        # Pad if needed
        pad_before = actual_min - crop_min
        pad_after = crop_max - actual_max
        if torch.any(pad_before > 0) or torch.any(pad_after > 0):
            paddings = pad_before[2], pad_after[2], pad_before[1], pad_after[1], pad_before[0], pad_after[0]
            volume_crop = F.pad(volume_crop, paddings, mode='constant', value=0)

        return volume_crop

    def _get_crop_from_patch(self, ij, crop_size_vx, patch):
        # Note ij is measured in quads, while crop_size_vx is measured in voxels
        crop_size = (crop_size_vx * patch.scale).int()
        grid = torch.stack(torch.meshgrid(torch.arange(crop_size[0]), torch.arange(crop_size[1])), dim=-1) + ij
        normalized_grid = grid / torch.tensor(patch.zyxs.shape[:2]) * 2 - 1
        interpolated = F.grid_sample(
            rearrange(patch.zyxs, 'y x zyx -> 1 zyx y x'),
            normalized_grid.flip(-1).unsqueeze(0),
            align_corners=True,
            mode='bilinear',
            padding_mode='zeros'
        )
        interpolated = rearrange(interpolated, '1 zyx y x -> y x zyx')
        valid_mask = F.grid_sample(
            rearrange(patch.valid_mask.float(), 'y x -> 1 1 y x'),
            normalized_grid.flip(-1).unsqueeze(0),
            align_corners=True,
            mode='nearest',
            padding_mode='zeros'
        ).squeeze(1).squeeze(0)
        valid_mask = valid_mask != 0.
        return torch.where(valid_mask[..., None], interpolated, -1), valid_mask

    def __iter__(self):

        areas = torch.tensor([patch.area for patch in self._patches])
        area_weights = areas / areas.sum()
        volume_crop_size = torch.tensor(self._config['crop_size'])
        patch_crop_size_vx = torch.tensor(self._config['patch_size'] * self._config['step_size'])

        while True:
            patch = random.choices(self._patches, weights=area_weights)[0]

            # Sample a random valid quad in the patch, then a point in that quad
            # This defines the center of our ROI (in both patch and volume)
            random_idx = torch.randint(len(patch.valid_indices) - 1, size=[])
            start_quad_ij = patch.valid_indices[random_idx]
            center_ij = start_quad_ij + torch.rand(size=[2])

            # Extract 2D crop from patch, and 3D crop from volume
            patch_crop, patch_valid = self._get_crop_from_patch(center_ij, patch_crop_size_vx, patch)
            if not patch_valid[patch_crop.shape[0] // 2, patch_crop.shape[1] // 2]:
                continue  # this can happen due to rounding errors
            center_zyx = patch_crop[patch_crop.shape[0] // 2, patch_crop.shape[1] // 2]
            volume_crop = self._get_crop_from_volume(patch.volume, center_zyx, volume_crop_size)

            # Relativise and normalise the patch coordinates
            patch_crop = torch.where(patch_valid[..., None], (patch_crop - center_zyx) / patch_crop_size_vx, -1)

            #TODO: include full 2d slices for additional context

            yield {
                'volume': volume_crop,
                'patch_zyx': patch_crop,
                'patch_valid': patch_valid,
            }


def load_datasets(config):
    all_patches = []
    for dataset in config['datasets']:
        ome_zarr = zarr.open(dataset['volume_path'], mode='r')
        volume_scale = dataset['volume_scale']
        volume = ome_zarr[str(volume_scale)]
        
        patches = load_tifxyz_patches(dataset['segments_path'], dataset.get('z_range', None), volume)

        if 'roi_path' in dataset:
            # If specified, roi_path is a proofreader log; filter & crop patches to approved cubes
            with open(dataset['roi_path'], 'r') as f:
                proofreader_json = json.load(f)
            assert tuple(proofreader_json['metadata']['volume_shape']) == ome_zarr['0'].shape
            approved_cubes = [{
                'min_zyx': patch['coords'],
                'size': patch['patch_size'],
            } for patch in proofreader_json['approved_patches']]
            patches = filter_patches_by_roi(patches, approved_cubes)

        patches = [patch.retarget(2 ** volume_scale) for patch in patches]

        all_patches.extend(patches)
        
    print(f'loaded {len(all_patches)} patches in total')
    return all_patches


def load_tifxyz_patches(segments_path, z_range,volume):

    segment_paths = glob.glob(segments_path + "/*")
    segment_paths = sorted([path for path in segment_paths if os.path.isdir(path)])
    print(f'found {len(segment_paths)} tifxyz patches')

    all_patches = []
    for segment_path in tqdm(segment_paths, desc='loading tifxyz patches'):
        try:  # TODO: remove
            with open(f'{segment_path}/meta.json', 'r') as meta_json:
                metadata = json.load(meta_json)
                bbox = metadata['bbox']
                scale = torch.tensor(metadata['scale'])
            if z_range is not None and (bbox['min'][2] > z_range[1] or bbox['max'][2] < z_range[0]):
                continue
            zyxs = torch.from_numpy(np.stack([
                cv2.imread( f'{segment_path}/{coord}.tif', flags=cv2.IMREAD_UNCHANGED)
                for coord in 'zyx'
            ], axis=-1))
            all_patches.append(Patch(zyxs, scale, volume))
        except Exception as e:
            print(f'error loading {segment_path}: {e}')
            continue

    print(f'loaded {len(all_patches)} tifxyz patches from {segments_path}')
    return all_patches


def filter_patches_by_roi(patches, approved_cubes):
    filtered_patches = []
    for patch in patches:
        # For each point, check if in any approved cube
        point_in_roi = torch.zeros(patch.zyxs.shape[:2], dtype=torch.bool)
        for cube in approved_cubes:
            min_zyx = torch.tensor(cube['min_zyx'], dtype=patch.zyxs.dtype)
            size = torch.tensor(cube['size'], dtype=patch.zyxs.dtype)
            in_cube = torch.all(patch.zyxs >= min_zyx, dim=-1) & torch.all(patch.zyxs < min_zyx + size, dim=-1)
            point_in_roi |= in_cube
        # Mask out points outside approved cube and crop; drop the patch if none left
        patch.zyxs[~point_in_roi] = torch.tensor([-1, -1, -1], dtype=patch.zyxs.dtype)
        valid_mask = torch.any(patch.zyxs != -1, dim=-1)
        if torch.any(valid_mask):
            valid_rows = torch.where(torch.any(valid_mask, dim=1))[0]
            valid_cols = torch.where(torch.any(valid_mask, dim=0))[0]
            cropped_zyxs = patch.zyxs[
                valid_rows[0] : valid_rows[-1] + 1,
                valid_cols[0] : valid_cols[-1] + 1
            ]
            filtered_patches.append(Patch(cropped_zyxs, patch.scale, patch.volume))
    return filtered_patches

