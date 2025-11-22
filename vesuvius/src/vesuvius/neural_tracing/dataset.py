import copy
import os
import cv2
import zarr
import json
import glob
import torch
import random
import numpy as np
import scipy.ndimage
import networkx as nx
from tqdm import tqdm
from einops import rearrange
from functools import lru_cache
import torch.nn.functional as F
from dataclasses import dataclass

import vesuvius.neural_tracing.augmentation as augmentation



@dataclass
class Patch:
    zyxs: torch.Tensor
    scale: torch.Tensor
    volume: zarr.Array
    volume_name: str | None = None

    def __post_init__(self):
        # Construct the valid *quads* mask; the ij'th element says whether all four corners of the quad with min-corner at ij are valid
        self.valid_quad_mask = torch.any(self.zyxs[:-1, :-1] != -1, dim=-1) & torch.any(self.zyxs[1:, :-1] != -1, dim=-1) & torch.any(self.zyxs[:-1, 1:] != -1, dim=-1) & torch.any(self.zyxs[1:, 1:] != -1, dim=-1)
        self.valid_quad_indices = torch.stack(torch.where(self.valid_quad_mask), dim=-1)
        self.valid_vertex_mask = torch.any(self.zyxs != -1, dim=-1)
        assert len(self.valid_quad_indices) > 0
        self.area = (~self.valid_quad_mask).sum() * (1 / self.scale).prod()
        self.quad_centers = torch.where(self.valid_quad_mask[..., None], 0.5 * (self.zyxs[1:, 1:] + self.zyxs[:-1, :-1]), torch.tensor(-1.))

    def retarget(self, factor):
        # Retarget the patch to a volume downsampled by the given factor
        return Patch(
            torch.where((self.zyxs == -1).all(dim=-1, keepdim=True), -1, self.zyxs / factor),
            self.scale * factor,
            self.volume,
            self.volume_name,
        )


class PatchInCubeDataset(torch.utils.data.IterableDataset):

    def __init__(self, config):
        self._config = config
        self._patches = load_datasets(config)
        self._augmentations = augmentation.get_training_augmentations(config['crop_size'], config['augmentation']['no_spatial'], config['augmentation']['only_spatial_and_intensity'])

    def __iter__(self):

        areas = torch.tensor([patch.area for patch in self._patches])
        area_weights = areas / areas.sum()
        # context_point_distance = self._config['context_point_distance']
        # num_context_points = self._config['num_context_points']
        # assert num_context_points >= 1  # ...since we always include one point at the center of the patch
        crop_size = torch.tensor(self._config['crop_size'])

        sample_idx = 0
        while True:
            sample_idx += 1
            patch = random.choices(self._patches, weights=area_weights)[0]

            # Sample a random valid quad in the patch, then a point in that quad
            random_idx = torch.randint(len(patch.valid_quad_indices) - 1, size=[])
            start_quad_ij = patch.valid_quad_indices[random_idx]
            center_ij = start_quad_ij + torch.rand(size=[2])
            # TODO: maybe erode inwards before doing this, so we can directly
            #  avoid sampling points that are too near the edge of the patch

            # # Sample other nearby random points as additional context; reject if outside patch
            # angle = torch.rand(size=[num_context_points]) * 2 * torch.pi
            # distance = context_point_distance * patch.scale
            # context_ij = center_ij + distance * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
            # if torch.any(context_ij < 0) or torch.any(context_ij >= torch.tensor(patch.zyxs.shape[:2])):
            #     continue
            # if not patch.valid_quad_mask[*context_ij.int().T].all():
            #     continue

            center_zyx = get_zyx_from_patch(center_ij, patch)
            # context_zyxs = [self._get_zyx_from_patch(context_ij, patch) for context_ij in context_ij]

            # Crop ROI out of the volume; mark context points
            volume_crop, min_corner_zyx = self._get_crop_from_volume(patch.volume, center_zyx, crop_size)
            # self._mark_context_point(volume_crop, center_zyx - min_corner_zyx)
            # for context_zyx in context_zyxs:
            #     self._mark_context_point(volume_crop, context_zyx - min_corner_zyx)

            # Find quads that are in the volume crop, and reachable from the start quad without leaving the crop

            # FIXME: instead check any corner in crop, and clamp to bounds later
            quad_centers = 0.5 * (patch.zyxs[1:, 1:] + patch.zyxs[:-1, :-1])
            quad_in_crop = patch.valid_quad_mask & torch.all(quad_centers >= min_corner_zyx, dim=-1) & torch.all(quad_centers < min_corner_zyx + crop_size, dim=-1)
            
            # Build neighbor graph of quads that are in the volume crop
            G = nx.Graph()
            quad_indices = torch.stack(torch.where(quad_in_crop), dim=-1)
            
            # Add nodes for each quad in crop using (i, j) as node ID
            for i, j in quad_indices:
                G.add_node((i.item(), j.item()))
            
            # Add edges between neighboring quads
            for i, j in quad_indices:
                # Check 4-connected neighbors
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for ni, nj in neighbors:
                    if (0 <= ni < quad_in_crop.shape[0] and 
                        0 <= nj < quad_in_crop.shape[1] and 
                        quad_in_crop[ni, nj]):
                        G.add_edge((i.item(), j.item()), (ni.item(), nj.item()))
            
            # Find reachable quads starting from start_quad_ij
            start_node = (start_quad_ij[0].item(), start_quad_ij[1].item())
            if not G.has_node(start_node):
                print('WARNING: start_quad_ij not in crop')
                continue
            reachable_quads = nx.node_connected_component(G, start_node)
            reachable_quads = torch.tensor(list(reachable_quads))
            
            # Create new mask with only reachable quads
            quad_reachable_in_crop = torch.zeros_like(quad_in_crop)
            quad_reachable_in_crop[*reachable_quads.T] = True
            
            # Rasterise the (cropped, reachable) surface patch
            filtered_quads_zyxs = torch.stack([
                torch.stack([
                    patch.zyxs[:-1, :-1][quad_reachable_in_crop],
                    patch.zyxs[:-1, 1:][quad_reachable_in_crop],
                ], dim=1),
                torch.stack([
                    patch.zyxs[1:, :-1][quad_reachable_in_crop],
                    patch.zyxs[1:, 1:][quad_reachable_in_crop],
                ], dim=1),
            ], dim=1)  # quad, top/bottom, left/right, zyx
            oversample_factor = 2
            points_per_side = (1 / patch.scale + 0.5).int() * oversample_factor
            v_points = torch.arange(points_per_side[0], dtype=torch.float32) / points_per_side[0]
            u_points = torch.arange(points_per_side[1], dtype=torch.float32) / points_per_side[1]
            points_covering_quads = torch.lerp(filtered_quads_zyxs[:, None, 0, :], filtered_quads_zyxs[:, None, 1, :], v_points[None, :, None, None])
            points_covering_quads = torch.lerp(points_covering_quads[:, :, None, 0], points_covering_quads[:, :, None, 1], u_points[None, None, :, None])
            indices_in_crop = (points_covering_quads - min_corner_zyx + 0.5).int().clip(0, crop_size - 1)
            rasterised = torch.zeros([crop_size, crop_size, crop_size], dtype=torch.float32)
            rasterised[*indices_in_crop.view(-1, 3).T] = 1.

            # Construct UV map on the surface
            filtered_quads_uvs = torch.stack(torch.where(quad_reachable_in_crop), dim=-1)
            uv_grid = torch.stack(torch.meshgrid(u_points, v_points, indexing='ij'), dim=-1)
            interpolated_uvs = filtered_quads_uvs[:, None, None, :] + uv_grid
            uvs = torch.zeros([crop_size, crop_size, crop_size, 2], dtype=torch.float32)
            uvs[*indices_in_crop.view(-1, 3).T] = interpolated_uvs.view(-1, 2)

            # Extend into free space: find EDT, and use feature transform to get nearest UV
            edt, ft = scipy.ndimage.morphology.distance_transform_edt((rasterised == 0).numpy(), return_indices=True)
            edt /= ((crop_size // 2) ** 2 * 3) ** 0.5 + 1.  # worst case: only center point is fg, hence max(edt) is 'radius' to corners
            edt = torch.exp(-edt / 0.25)
            edt = edt.to(torch.float32) * 2 - 1  # ...so it's "signed" but the zero point is arbitrary
            uvs = uvs[*ft]
            uvs = (uvs - center_ij) / patch.scale / (crop_size * 2)  # *2 is somewhat arbitrary; worst-ish-case = patch wrapping round three sides of cube
            uvws = torch.cat([uvs, edt[..., None]], dim=-1).to(torch.float32)

            localiser = build_localiser(center_zyx, min_corner_zyx, crop_size)

            #TODO: include full 2d slices for additional context
            #  if so, need to augment them consistently with the 3d crop -> tricky for geometric transforms

            # FIXME: the loop is a hack because some augmentation sometimes randomly returns None
            #  we should instead just remove the relevant augmentation (or fix it!)
            while True:  
                augmented = self._augmentations(image=volume_crop[None], dist_map=torch.cat([localiser[None], rearrange(uvws, 'z y x c -> c z y x')], dim=0))
                if augmented['dist_map'] is not None:
                    break
            volume_crop = augmented['image'].squeeze(0)
            localiser = augmented['dist_map'][0]
            uvws = rearrange(augmented['dist_map'][1:], 'c z y x -> z y x c')
            if not torch.isfinite(volume_crop).all() or not torch.isfinite(localiser).all() or not torch.isfinite(uvws).all():
                # FIXME: why do these NaNs happen occasionally?
                print(f"[DataWarning] NaN/Inf encountered (sample {sample_idx}, volume={getattr(patch, 'volume_name', None)})")
                continue  

            yield {
                'volume': volume_crop,
                'localiser': localiser,
                'uvw': uvws,
            }


class HeatmapDatasetV2(torch.utils.data.IterableDataset):
    _channel_range_cache = {}
    _kernel_offsets_cache = {}
    _kernel_value_cache = {}

    def __init__(self, config, patches_for_split):
        self._config = config
        self._patches = patches_for_split
        self._heatmap_sigma = float(config.get('heatmap_sigma', 2.0))
        self._augmentations = augmentation.get_training_augmentations(config['crop_size'], config['augmentation']['allow_transposes'], config['augmentation']['only_spatial_and_intensity'])
        self._perturb_prob = config['point_perturbation']['perturb_probability']
        self._uv_max_perturbation = config['point_perturbation']['uv_max_perturbation']  # measured in voxels
        self._w_max_perturbation = config['point_perturbation']['w_max_perturbation']  # measured in voxels
        self._main_component_distance_factor = config['point_perturbation']['main_component_distance_factor']

        self._perturb_cache_key = None
        self._perturb_cache_value = None
        self._use_zscore = config.get("zscore", True)
        self._sampling = {}
        self._quad_bboxes = {}

        for patch in self._patches:
            top_left = patch.zyxs[:-1, :-1]
            top_right = patch.zyxs[:-1, 1:]
            bottom_left = patch.zyxs[1:, :-1]
            bottom_right = patch.zyxs[1:, 1:]
            quad_corners = torch.stack([
                torch.stack([top_left, top_right], dim=2),
                torch.stack([bottom_left, bottom_right], dim=2),
            ], dim=2)  # shape: (h-1, w-1, 2, 2, 3)
            points_per_side = (1 / patch.scale + 0.5).int()
            v_points = torch.arange(points_per_side[0], dtype=torch.float32) / points_per_side[0]
            u_points = torch.arange(points_per_side[1], dtype=torch.float32) / points_per_side[1]
            self._sampling[id(patch)] = {
                "quad_corners": quad_corners,
                "v_points": v_points,
                "u_points": u_points,
                "uv_weights": self._make_quad_weights(v_points, u_points),
                "quad_corners_flat": quad_corners.view(*quad_corners.shape[:2], 4, 3),
            }

            valid_centers = patch.quad_centers[patch.valid_quad_mask]
            if len(valid_centers) == 0:
                bbox_min = torch.zeros(3, dtype=patch.quad_centers.dtype)
                bbox_max = bbox_min
            else:
                bbox_min = valid_centers.min(dim=0).values
                bbox_max = valid_centers.max(dim=0).values
            self._quad_bboxes[id(patch)] = (bbox_min, bbox_max)

    def _normalize_volume_crop(self, volume_crop: torch.Tensor) -> torch.Tensor:
        if self._use_zscore:
            mean = torch.mean(volume_crop)
            std = torch.std(volume_crop).clamp(min=1e-8)
            return (volume_crop - mean) / std
        return volume_crop

    def _get_crop_from_volume(self, volume, center_zyx, crop_size):
        if self._use_zscore:
            volume_crop, min_corner_zyx = get_crop_from_volume(volume, center_zyx, crop_size, normalize=False)
            volume_crop = self._normalize_volume_crop(volume_crop)
            return volume_crop, min_corner_zyx

        volume_crop, min_corner_zyx = get_crop_from_volume(volume, center_zyx, crop_size)
        volume_crop = self._normalize_volume_crop(volume_crop)
        return volume_crop, min_corner_zyx

    def _sample_points_from_quads(self, patch, quad_mask):
        """Sample points finely from quads specified by the mask"""
        if not torch.any(quad_mask):
            return torch.empty(0, 3)
        
        filtered_quads_zyxs = torch.stack([
            torch.stack([
                patch.zyxs[:-1, :-1][quad_mask],
                patch.zyxs[:-1, 1:][quad_mask],
            ], dim=1),
            torch.stack([
                patch.zyxs[1:, :-1][quad_mask],
                patch.zyxs[1:, 1:][quad_mask],
            ], dim=1),
        ], dim=1)  # quad, top/bottom, left/right, zyx
        
        points_per_side = (1 / patch.scale + 0.5).int()
        v_points = torch.arange(points_per_side[0], dtype=torch.float32) / points_per_side[0]
        u_points = torch.arange(points_per_side[1], dtype=torch.float32) / points_per_side[1]
        points_covering_quads = torch.lerp(filtered_quads_zyxs[:, None, 0, :], filtered_quads_zyxs[:, None, 1, :], v_points[None, :, None, None])
        points_covering_quads = torch.lerp(points_covering_quads[:, :, None, 0], points_covering_quads[:, :, None, 1], u_points[None, None, :, None])
        
        return points_covering_quads.view(-1, 3)

    @staticmethod
    def _make_quad_weights(v_points, u_points):
        one_minus_v = 1 - v_points
        one_minus_u = 1 - u_points
        w_tl = one_minus_v[:, None] * one_minus_u[None, :]
        w_tr = one_minus_v[:, None] * u_points[None, :]
        w_bl = v_points[:, None] * one_minus_u[None, :]
        w_br = v_points[:, None] * u_points[None, :]
        weights = torch.stack([w_tl, w_tr, w_bl, w_br], dim=-1)
        return weights.view(-1, 4)

    def _get_quads_in_crop(self, patch, min_corner_zyx, crop_size):
        """Get mask of quads that fall within the crop region."""
        bbox_min, bbox_max = self._quad_bboxes[id(patch)]
        crop_min = min_corner_zyx.to(dtype=bbox_min.dtype)
        crop_size_tensor = torch.as_tensor(crop_size, dtype=bbox_min.dtype, device=crop_min.device)
        crop_max = crop_min + crop_size_tensor
        if (bbox_max < crop_min).any() or (bbox_min >= crop_max).any():
            return torch.zeros_like(patch.valid_quad_mask)

        return patch.valid_quad_mask & torch.all(patch.quad_centers >= crop_min, dim=-1) & torch.all(patch.quad_centers < crop_max, dim=-1)

    def _get_patch_points_in_crop(self, patch, min_corner_zyx, crop_size):
        """Get finely sampled points from a patch that fall within the crop region"""
        bbox_min, bbox_max = self._quad_bboxes[id(patch)]
        crop_min = min_corner_zyx.to(dtype=bbox_min.dtype)
        crop_size_tensor = torch.as_tensor(crop_size, dtype=bbox_min.dtype)
        crop_max = crop_min + crop_size_tensor
        if (bbox_max < crop_min).any() or (bbox_min >= crop_max).any():
            return torch.empty((0, 3), dtype=torch.float32)

        quad_in_crop = self._get_quads_in_crop(patch, min_corner_zyx, crop_size)
        if not torch.any(quad_in_crop):
            return torch.empty((0, 3), dtype=torch.float32)

        info = self._sampling[id(patch)]
        filtered_quads_zyxs = info["quad_corners_flat"][quad_in_crop]
        weights = info["uv_weights"]
        points_covering_quads = torch.einsum("kc,ncd->nkd", weights, filtered_quads_zyxs)

        return points_covering_quads.reshape(-1, 3)

    def _get_current_patch_center_component_mask(self, current_patch, center_ij, min_corner_zyx, crop_size):
        """Get the mask of the connected component containing the center point"""
        quad_in_crop = self._get_quads_in_crop(current_patch, min_corner_zyx, crop_size)
        
        if not torch.any(quad_in_crop):
            return torch.zeros_like(quad_in_crop)
        
        center_quad = center_ij.int()
        if (center_quad[0] < 0 or center_quad[0] >= quad_in_crop.shape[0] or 
            center_quad[1] < 0 or center_quad[1] >= quad_in_crop.shape[1] or
            not quad_in_crop[center_quad[0], center_quad[1]]):
            return torch.zeros_like(quad_in_crop)
        
        component_mask = torch.zeros_like(quad_in_crop)
        stack = [(center_quad[0].item(), center_quad[1].item())]
        
        while stack:
            i, j = stack.pop()
            if (0 <= i < quad_in_crop.shape[0] and 0 <= j < quad_in_crop.shape[1] and 
                quad_in_crop[i, j] and not component_mask[i, j]):
                component_mask[i, j] = True
                stack.extend([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])
        
        return component_mask

    def _make_cache_key(self, patch, center_ij, min_corner_zyx, crop_size):
        return (
            id(patch),
            tuple(center_ij.tolist()),
            tuple(min_corner_zyx.tolist()),
            int(crop_size),
        )

    def _compute_cached_patch_points(self, current_patch, center_ij, min_corner_zyx, crop_size):
        """Pre-compute the patch-point sets for a given crop context."""
        quad_main_component = self._get_current_patch_center_component_mask(current_patch, center_ij, min_corner_zyx, crop_size)

        all_patch_points = []
        for other_patch in self._patches:
            if other_patch is current_patch or other_patch.volume is not current_patch.volume:
                continue
            patch_points = self._get_patch_points_in_crop(other_patch, min_corner_zyx, crop_size)
            if len(patch_points) > 0:
                all_patch_points.append(patch_points)

        quad_in_crop = self._get_quads_in_crop(current_patch, min_corner_zyx, crop_size)
        quad_excluding_main = quad_in_crop & ~quad_main_component
        other_patch_points = self._sample_points_from_quads(current_patch, quad_excluding_main)
        if len(other_patch_points) > 0:
            all_patch_points.append(other_patch_points)

        return all_patch_points

    def _get_cached_patch_points(self, current_patch, center_ij, min_corner_zyx, crop_size):
        """Pre-compute and cache all patch points for efficient distance calculations."""
        key = self._make_cache_key(current_patch, center_ij, min_corner_zyx, crop_size)
        if key != self._perturb_cache_key:
            self._perturb_cache_key = key
            raw_points = self._compute_cached_patch_points(current_patch, center_ij, min_corner_zyx, crop_size)
            concat = None
            if raw_points and any(p.numel() > 0 for p in raw_points):
                concat = torch.cat([p for p in raw_points if p.numel() > 0], dim=0)
            self._perturb_cache_value = {"points": raw_points, "concat": concat}
        return self._perturb_cache_value

    def _get_distance_to_nearest_patch_cached(self, point_zyx, cached_patch_points):
        """Calculate the distance to the nearest patch using pre-computed patch points"""
        if not cached_patch_points:
            return float('inf')
        if isinstance(cached_patch_points, dict):
            concat = cached_patch_points.get("concat", None)
            if concat is None or concat.numel() == 0:
                return float('inf')
            return torch.norm(concat - point_zyx, dim=-1).min().item()

        min_distance = float('inf')
        for patch_points in cached_patch_points:
            if patch_points.numel() == 0:
                continue
            distances = torch.norm(patch_points - point_zyx, dim=-1)
            min_distance = min(min_distance, distances.min().item())
        return min_distance

    def _get_perturbed_zyx_from_patch(self, point_ij, patch, center_ij, min_corner_zyx, crop_size, is_center_point=False):
        """Apply random 3D perturbation to a point and return the perturbed 3D coordinates"""
        if is_center_point:
            # For center point, only apply normal perturbation, skip uv perturbation
            perturbed_ij = point_ij
            perturbed_zyx = get_zyx_from_patch(point_ij, patch)
        else:
            # For conditioning points, apply both uv and normal perturbations
            # Generate random 2D offset within the uv threshold (in voxels)
            offset_magnitude = torch.rand([]) * self._uv_max_perturbation
            offset_angle = torch.rand([]) * 2 * torch.pi
            offset_uv_voxels = offset_magnitude * torch.tensor([torch.cos(offset_angle), torch.sin(offset_angle)])
            
            # Convert uv offset from voxels to patch coordinates using patch scale
            offset_2d = offset_uv_voxels * patch.scale
            
            # Apply 2D offset
            perturbed_ij = point_ij + offset_2d
            
            # Clamp to patch bounds
            perturbed_ij = torch.clamp(perturbed_ij, torch.zeros([]), torch.tensor(patch.zyxs.shape[:2]) - 1.001)
            
            # Check if the perturbed point is still valid
            if not patch.valid_quad_mask[*perturbed_ij.int()]:
                return get_zyx_from_patch(point_ij, patch)  # Return original 3D point if invalid
            
            # Convert to 3D coordinates
            perturbed_zyx = get_zyx_from_patch(perturbed_ij, patch)
        
        # Estimate quad normal at this point for 3D perturbation
        # Get neighboring points to estimate the surface normal
        i, j = perturbed_ij.int()
        h, w = patch.zyxs.shape[:2]
        
        # Get surrounding points for normal estimation
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and patch.valid_vertex_mask[ni, nj]:
                    neighbors.append(patch.zyxs[ni, nj])
        
        if len(neighbors) < 3:
            # Not enough neighbors for normal estimation, skip 3D perturbation
            final_zyx = perturbed_zyx
        else:
            # Estimate surface normal using cross product of two edges
            normal = torch.linalg.cross(neighbors[1] - neighbors[0], neighbors[2] - neighbors[0], dim=-1)
            normal_norm = torch.norm(normal)
            if normal_norm > 1e-6:
                normal = normal / normal_norm
                # Apply random 3D offset along normal direction using w threshold
                normal_offset_magnitude = (torch.rand([]) * 2 - 1) * self._w_max_perturbation
                
                # Pre-compute patch points once for efficient distance calculations
                cached_patch_points = self._get_cached_patch_points(patch, center_ij, min_corner_zyx, crop_size)
                
                # Find a perturbation size that is acceptable, i.e. doesn't bring us too close to another patch
                while abs(normal_offset_magnitude) >= 1.0:
                    nearest_patch_distance = self._get_distance_to_nearest_patch_cached(perturbed_zyx, cached_patch_points)
                    if abs(normal_offset_magnitude) <= nearest_patch_distance * self._main_component_distance_factor:
                        break
                    normal_offset_magnitude *= 0.8
                else:
                    normal_offset_magnitude = 0.
                
                # Apply the acceptable perturbation
                final_zyx = perturbed_zyx + normal_offset_magnitude * normal
            else:
                # Normal is too small, skip 3D perturbation
                final_zyx = perturbed_zyx
        
        return final_zyx

    @classmethod
    def _get_channel_range(cls, num_channels, device):
        key = (num_channels, device)
        if key not in cls._channel_range_cache:
            cls._channel_range_cache[key] = torch.arange(num_channels, device=device)
        return cls._channel_range_cache[key]

    @classmethod
    def _get_kernel_offsets(cls, device, sigma):
        sigma = float(sigma)
        if sigma not in cls._kernel_offsets_cache:
            _, kernel_size = _get_gaussian_kernel(sigma)
            radius = kernel_size // 2
            coords = torch.arange(kernel_size) - radius
            z_off, y_off, x_off = torch.meshgrid(coords, coords, coords, indexing='ij')
            cls._kernel_offsets_cache[sigma] = torch.stack([z_off, y_off, x_off], dim=-1).view(-1, 3)
        return cls._kernel_offsets_cache[sigma].to(device=device)

    @classmethod
    def _get_kernel_values(cls, device, dtype, sigma):
        key = (device, dtype, float(sigma))
        if key not in cls._kernel_value_cache:
            kernel, _ = _get_gaussian_kernel(sigma)
            cls._kernel_value_cache[key] = kernel.to(device=device, dtype=dtype).reshape(-1)
        return cls._kernel_value_cache[key]

    @classmethod
    def _collect_coords(cls, all_zyxs, min_corner_zyx, crop_size_int, device, dtype):
        channel_count = all_zyxs[0].shape[0]
        coords_accum = []
        channel_accum = []
        channel_range = cls._get_channel_range(channel_count, device)
        min_corner = min_corner_zyx.to(device=device, dtype=dtype)

        for zyxs in all_zyxs:
            coords = (zyxs.to(device=device, dtype=dtype) - min_corner + 0.5).to(torch.int64)
            valid_mask = (coords >= 0).all(dim=1) & (coords < crop_size_int).all(dim=1)
            if not torch.any(valid_mask):
                continue
            coords_accum.append(coords[valid_mask])
            channel_accum.append(channel_range[valid_mask])

        if not coords_accum:
            return channel_count, None, None

        all_coords = torch.cat(coords_accum, dim=0)
        all_channels = torch.cat(channel_accum, dim=0)
        return channel_count, all_coords, all_channels

    @classmethod
    def _scatter_heatmaps(cls, all_zyxs, min_corner_zyx, crop_size):
        crop_size_int = int(crop_size)
        dtype = all_zyxs[0].dtype
        device = all_zyxs[0].device
        channel_count, coords, channels = cls._collect_coords(all_zyxs, min_corner_zyx, crop_size_int, device, dtype)
        heatmaps = torch.zeros((channel_count, crop_size_int, crop_size_int, crop_size_int), device=device, dtype=dtype)

        if coords is None:
            return heatmaps

        z, y, x = coords.unbind(dim=1)
        values = torch.ones_like(channels, dtype=dtype)
        heatmaps.index_put_((channels, z, y, x), values, accumulate=True)
        return heatmaps

    @classmethod
    def make_heatmaps(cls, all_zyxs, min_corner_zyx, crop_size, apply_gaussian=True, sigma: float = 2.0):
        if not apply_gaussian:
            return cls._scatter_heatmaps(all_zyxs, min_corner_zyx, crop_size)

        crop_size_int = int(crop_size)
        dtype = all_zyxs[0].dtype
        device = all_zyxs[0].device
        channel_count, coords, channels = cls._collect_coords(all_zyxs, min_corner_zyx, crop_size_int, device, dtype)
        heatmaps = torch.zeros((channel_count, crop_size_int, crop_size_int, crop_size_int), device=device, dtype=dtype)

        if coords is None:
            return heatmaps

        kernel_offsets = cls._get_kernel_offsets(device, sigma)
        kernel_values = cls._get_kernel_values(device, dtype, sigma)

        expanded_coords = coords[:, None, :] + kernel_offsets[None, :, :]
        in_bounds = (expanded_coords >= 0).all(dim=-1) & (expanded_coords < crop_size_int).all(dim=-1)

        if torch.any(in_bounds):
            valid_positions = expanded_coords[in_bounds]
            valid_channels = channels[:, None].expand(-1, kernel_offsets.shape[0])[in_bounds]
            valid_values = kernel_values.expand(coords.shape[0], -1)[in_bounds]

            z, y, x = valid_positions.unbind(dim=1)
            heatmaps.index_put_((valid_channels, z, y, x), valid_values, accumulate=True)

        return heatmaps

    def __iter__(self):

        areas = torch.tensor([patch.area for patch in self._patches])
        area_weights = areas / areas.sum()
        crop_size = torch.tensor(self._config['crop_size'])
        step_size = torch.tensor(self._config['step_size'])
        step_count = torch.tensor(self._config['step_count'])
        heatmap_sigma = self._heatmap_sigma

        sample_idx = 0
        while True:
            sample_idx += 1
            patch = random.choices(self._patches, weights=area_weights)[0]

            # Sample a random valid quad in the patch, then a point in that quad
            random_idx = torch.randint(len(patch.valid_quad_indices) - 1, size=[])
            start_quad_ij = patch.valid_quad_indices[random_idx]
            center_ij = start_quad_ij + torch.rand(size=[2])
            center_zyx = get_zyx_from_patch(center_ij, patch)

            # Sample rows of points along U & V axes
            uv_deltas = torch.arange(1, step_count + 1)[:, None] * step_size * patch.scale
            u_pos_shifted_ijs = center_ij + uv_deltas * torch.tensor([1, 0])
            u_neg_shifted_ijs = center_ij - uv_deltas * torch.tensor([1, 0])
            v_pos_shifted_ijs = center_ij + uv_deltas * torch.tensor([0, 1])
            v_neg_shifted_ijs = center_ij - uv_deltas * torch.tensor([0, 1])

            def valid_steps(shifted_ijs):
                ij_int = shifted_ijs.long()
                h, w = patch.valid_vertex_mask.shape[:2]
                in_bounds = (ij_int[:, 0] >= 0) & (ij_int[:, 0] < h) & (ij_int[:, 1] >= 0) & (ij_int[:, 1] < w)
                valid = torch.zeros(shifted_ijs.shape[0], dtype=torch.bool, device=shifted_ijs.device)
                if in_bounds.any():
                    valid[in_bounds] = patch.valid_vertex_mask[ij_int[in_bounds, 0], ij_int[in_bounds, 1]]
                return valid

            u_pos_valid = valid_steps(u_pos_shifted_ijs)
            u_neg_valid = valid_steps(u_neg_shifted_ijs)
            v_pos_valid = valid_steps(v_pos_shifted_ijs)
            v_neg_valid = valid_steps(v_neg_shifted_ijs)

            # Randomly flip positive and negative directions, as a form of augmentation since they're arbitrary
            if torch.rand([]) < 0.5:
                u_pos_shifted_ijs, u_neg_shifted_ijs = u_neg_shifted_ijs, u_pos_shifted_ijs
                u_pos_valid, u_neg_valid = u_neg_valid, u_pos_valid
            if torch.rand([]) < 0.5:
                v_pos_shifted_ijs, v_neg_shifted_ijs = v_neg_shifted_ijs, v_pos_shifted_ijs
                v_pos_valid, v_neg_valid = v_neg_valid, v_pos_valid

            # Similarly, randomly swap U & V axes
            if torch.rand([]) < 0.5:
                u_pos_shifted_ijs, v_pos_shifted_ijs = v_pos_shifted_ijs, u_pos_shifted_ijs
                u_neg_shifted_ijs, v_neg_shifted_ijs = v_neg_shifted_ijs, u_neg_shifted_ijs
                u_pos_valid, v_pos_valid = v_pos_valid, u_pos_valid
                u_neg_valid, v_neg_valid = v_neg_valid, u_neg_valid

            # Apply perturbations to center and negative points
            if torch.rand([]) < self._perturb_prob:
                min_corner_zyx = (center_zyx - crop_size // 2).int()
                
                # Perturb center point in 3D (only normal perturbation, no uv)
                center_zyx = self._get_perturbed_zyx_from_patch(center_ij, patch, center_ij, min_corner_zyx, crop_size, is_center_point=True)
                
                # Perturb negative points (context points) in 3D (both uv and normal)
                def _perturb_or_lookup(shifted_ijs, valid_mask):
                    zyxs = []
                    for i in range(len(shifted_ijs)):
                        if valid_mask[i]:
                            zyxs.append(self._get_perturbed_zyx_from_patch(shifted_ijs[i], patch, center_ij, min_corner_zyx, crop_size, is_center_point=False))
                        else:
                            zyxs.append(get_zyx_from_patch(shifted_ijs[i], patch))
                    return torch.stack(zyxs)

                u_neg_shifted_zyxs = _perturb_or_lookup(u_neg_shifted_ijs, u_neg_valid)
                v_neg_shifted_zyxs = _perturb_or_lookup(v_neg_shifted_ijs, v_neg_valid)
                
            else:
                # No perturbation applied, use original coordinates
                u_neg_shifted_zyxs = get_zyx_from_patch(u_neg_shifted_ijs, patch)
                v_neg_shifted_zyxs = get_zyx_from_patch(v_neg_shifted_ijs, patch)
            
            # Get crop volume (with optional normalization) and its min-corner (which may be slightly negative)
            volume_crop, min_corner_zyx = self._get_crop_from_volume(patch.volume, center_zyx, crop_size)

            # Map to 3D space and construct heatmaps
            u_pos_shifted_zyxs = get_zyx_from_patch(u_pos_shifted_ijs, patch)
            v_pos_shifted_zyxs = get_zyx_from_patch(v_pos_shifted_ijs, patch)

            def make_in_out_heatmaps(pos_shifted_zyxs, neg_shifted_zyxs, cond, suppress_out=None):
                if cond:
                    # Conditioning on this direction: include one negative point as input, and all positive as output
                    assert suppress_out is None
                    in_heatmaps = self.make_heatmaps([neg_shifted_zyxs[:1]], min_corner_zyx, crop_size, sigma=heatmap_sigma)
                    out_heatmaps = self.make_heatmaps([pos_shifted_zyxs], min_corner_zyx, crop_size, sigma=heatmap_sigma)
                else:
                    # Not conditioning on this direction: include all positive and negative points as output, and nothing as input
                    in_heatmaps = torch.zeros([1, crop_size, crop_size, crop_size])
                    out_points = ([pos_shifted_zyxs] if suppress_out != 'pos' else []) + ([neg_shifted_zyxs] if suppress_out != 'neg' else [])
                    out_heatmaps = self.make_heatmaps(out_points, min_corner_zyx, crop_size, sigma=heatmap_sigma) if out_points else torch.zeros([pos_shifted_zyxs.shape[0], crop_size, crop_size, crop_size])
                return in_heatmaps, out_heatmaps

            # TODO: is there a nicer way to arrange this that expresses the logic/cases better?

            u_cond, v_cond = torch.rand([2]) < 0.75

            diag_ij = None
            suppress_out_u = suppress_out_v = None
            
            if (u_cond ^ v_cond) and torch.rand([]) < 0.6:
                # With 60% probability in one-cond case, also condition on a diagonal that is adjacent
                # to the conditioned-on (1st neg) point, and suppress output (positive/negative) heatmaps
                # for the perpendicular (not-cond) direction on the same side as the diagonal
                diag_is_pos = torch.rand([]) < 0.5
                if u_cond:
                    diag_ij = torch.stack([u_neg_shifted_ijs[0, 0], (v_pos_shifted_ijs if diag_is_pos else v_neg_shifted_ijs)[0, 1]])
                    suppress_out_v = 'neg' if diag_is_pos else 'pos'
                else:
                    diag_ij = torch.stack([(u_pos_shifted_ijs if diag_is_pos else u_neg_shifted_ijs)[0, 0], v_neg_shifted_ijs[0, 1]])
                    suppress_out_u = 'neg' if diag_is_pos else 'pos'
            if (u_cond & v_cond) and torch.rand([]) < 0.5:
                # With 50% probability in two-cond case, also condition on a diagonal in a 'to x from' direction, so
                # adjacent to exactly one of the conditioned-on (1st neg) points
                if torch.rand([]) < 0.5:
                    diag_ij = torch.stack([u_neg_shifted_ijs[0, 0], v_pos_shifted_ijs[0, 1]])
                else:
                    diag_ij = torch.stack([u_pos_shifted_ijs[0, 0], v_neg_shifted_ijs[0, 1]])
            if diag_ij is not None:
                if torch.any(diag_ij < 0) or torch.any(diag_ij >= torch.tensor(patch.valid_vertex_mask.shape[:2])):
                    continue
                if not patch.valid_vertex_mask[*diag_ij.int()]:
                    continue
                diag_zyx = get_zyx_from_patch(diag_ij, patch)
            else:
                diag_zyx = None

            if (u_cond and not u_neg_valid[0]) or (v_cond and not v_neg_valid[0]):
                continue

            def build_out_channel_mask(cond, suppress_out, pos_valid, neg_valid):
                if cond:
                    return pos_valid
                mask = torch.zeros_like(pos_valid)
                if suppress_out != 'pos':
                    mask |= pos_valid
                if suppress_out != 'neg':
                    mask |= neg_valid
                return mask

            u_out_channel_mask = build_out_channel_mask(u_cond, suppress_out_u, u_pos_valid, u_neg_valid)
            v_out_channel_mask = build_out_channel_mask(v_cond, suppress_out_v, v_pos_valid, v_neg_valid)
            if not (u_out_channel_mask.any() or v_out_channel_mask.any()):
                # No valid supervision remaining
                continue

            # *_in_heatmaps always have a single plane, either the first negative point or empty
            # *_out_heatmaps always have one plane per step, and may contain only positive or both positive and negative points
            u_in_heatmaps, u_out_heatmaps = make_in_out_heatmaps(u_pos_shifted_zyxs, u_neg_shifted_zyxs, u_cond, suppress_out_u)
            v_in_heatmaps, v_out_heatmaps = make_in_out_heatmaps(v_pos_shifted_zyxs, v_neg_shifted_zyxs, v_cond, suppress_out_v)
            if ~u_cond and ~v_cond:
                # In this case U & V are (nearly) indistinguishable, so don't force the model to separate them
                u_out_heatmaps = v_out_heatmaps = torch.maximum(u_out_heatmaps, v_out_heatmaps)
                combined_mask = u_out_channel_mask | v_out_channel_mask
                u_out_channel_mask = v_out_channel_mask = combined_mask
            if diag_zyx is not None:
                diag_in_heatmaps = self.make_heatmaps([diag_zyx[None]], min_corner_zyx, crop_size, sigma=heatmap_sigma)
            else:
                diag_in_heatmaps = torch.zeros_like(u_in_heatmaps)
            uv_heatmaps_both = torch.cat([u_in_heatmaps, v_in_heatmaps, diag_in_heatmaps, u_out_heatmaps, v_out_heatmaps], dim=0)
            out_channel_mask = torch.cat([u_out_channel_mask, v_out_channel_mask])

            # Build localiser volume
            localiser = build_localiser(center_zyx, min_corner_zyx, crop_size)

            #TODO: include full 2d slices for additional context
            #  if so, need to augment them consistently with the 3d crop -> tricky for geometric transforms

            # FIXME: the loop is a hack because some augmentation sometimes randomly returns None
            #  we should instead just remove the relevant augmentation (or fix it!)
            # TODO: consider interaction of augmentation with localiser -- logically should follow translations of
            #  the center-point, since the heatmaps do, but not follow rotations/scales; however in practice maybe
            #  ok since it's 'just more augmentation' that won't be applied during tracing
            while True:
                augmented = self._augmentations(image=volume_crop[None], dist_map=localiser[None], regression_target=uv_heatmaps_both)
                if augmented['dist_map'] is not None:
                    break
            volume_crop = augmented['image'].squeeze(0)
            localiser = augmented['dist_map'].squeeze(0)
            uv_heatmaps_both = rearrange(augmented['regression_target'], 'c z y x -> z y x c')
            if not torch.isfinite(volume_crop).all() or not torch.isfinite(localiser).all() or not torch.isfinite(uv_heatmaps_both).all():
                # FIXME: why do these NaNs happen occasionally?
                print(f"[DataWarning] NaN/Inf encountered (sample {sample_idx}, volume={getattr(patch, 'volume_name', None)})")
                continue

            uv_heatmaps_in = uv_heatmaps_both[..., :3]
            uv_heatmaps_out = uv_heatmaps_both[..., 3:]
            out_channel_mask_expanded = out_channel_mask.to(device=uv_heatmaps_out.device, dtype=uv_heatmaps_out.dtype).view(1, 1, 1, -1)
            uv_heatmaps_out_mask = out_channel_mask_expanded.expand_as(uv_heatmaps_out)
            uv_heatmaps_out = uv_heatmaps_out * uv_heatmaps_out_mask

            yield {
                'volume': volume_crop,
                'localiser': localiser,
                'uv_heatmaps_in': uv_heatmaps_in,
                'uv_heatmaps_out': uv_heatmaps_out,
                'uv_heatmaps_out_mask': uv_heatmaps_out_mask,
            }


class ForwardAlignedHeatmapDataset(HeatmapDatasetV2):
    """
    Aligns each crop to a local canonical frame before augmentation so that the positive U
    direction becomes +X, the V direction becomes +Y, and the surface normal becomes +Z. This
    reduces directional variance and makes "forward" consistent across samples.
    """

    _local_grid_cache = {}

    def __init__(self, config, patches_for_split):
        cfg = copy.deepcopy(config)
        forward_cfg = cfg.get("forward_align", {})
        aug_cfg = cfg.setdefault("augmentation", {})
        aug_cfg["allow_transposes"] = forward_cfg.get("allow_transposes", False)

        self._forward_align_cfg = forward_cfg
        super().__init__(cfg, patches_for_split)

    @staticmethod
    def _safe_normalize(vec, eps=1e-6):
        norm = torch.linalg.norm(vec)
        if (not torch.isfinite(norm).item()) or norm.item() < eps:
            return None
        return vec / norm

    def _estimate_local_frame(self, patch, center_ij):
        """Build an orthonormal frame from patch geometry; returns R whose columns are (fwd, side, normal) in zyx."""
        device = patch.zyxs.device
        dtype = patch.zyxs.dtype

        def sample(offset):
            return get_zyx_from_patch(center_ij + offset, patch).to(device=device, dtype=dtype)

        u_offset = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        v_offset = torch.tensor([1.0, 0.0], device=device, dtype=dtype)

        forward_vec = sample(u_offset) - sample(-u_offset)
        side_vec = sample(v_offset) - sample(-v_offset)

        forward = self._safe_normalize(forward_vec)
        side = self._safe_normalize(side_vec)
        if forward is None or side is None:
            return None

        normal = self._safe_normalize(torch.linalg.cross(forward, side))
        if normal is None:
            return None

        side = self._safe_normalize(side - torch.dot(side, forward) * forward - torch.dot(side, normal) * normal) or torch.linalg.cross(normal, forward)
        forward = self._safe_normalize(forward - torch.dot(forward, normal) * normal) or torch.linalg.cross(side, normal)

        if side is None or forward is None:
            return None

        if self._forward_align_cfg.get("prefer_positive_normal_z", True) and normal[0] < 0:
            normal = -normal
            forward = -forward

        return torch.stack([forward, side, normal], dim=1)  # columns: x/y/z in global zyx coords

    @classmethod
    def _get_local_grid(cls, crop_size, device, dtype):
        key = (int(crop_size), device, dtype)
        if key not in cls._local_grid_cache:
            size = int(crop_size)
            z, y, x = torch.meshgrid(
                torch.arange(size, device=device),
                torch.arange(size, device=device),
                torch.arange(size, device=device),
                indexing="ij",
            )
            center = (size - 1) / 2.0
            coords_local = torch.stack([x - center, y - center, z - center], dim=-1).to(dtype=dtype)
            cls._local_grid_cache[key] = (coords_local.view(-1, 3), center)
        return cls._local_grid_cache[key]

    def _make_rotation_grid(self, R, crop_size, device, dtype):
        coords_local_flat, center = self._get_local_grid(crop_size, device, dtype)
        rotated = coords_local_flat @ R.T + center  # back to global zyx voxel coordinates
        coords = rotated.view(int(crop_size), int(crop_size), int(crop_size), 3)

        norm = (crop_size - 1)
        grid = torch.empty((1, int(crop_size), int(crop_size), int(crop_size), 3), device=device, dtype=dtype)
        grid[..., 0] = coords[..., 2] * 2.0 / norm - 1.0  # x
        grid[..., 1] = coords[..., 1] * 2.0 / norm - 1.0  # y
        grid[..., 2] = coords[..., 0] * 2.0 / norm - 1.0  # z
        return grid

    def _apply_forward_alignment(self, volume_crop, localiser, uv_heatmaps_both, patch, center_ij):
        frame = self._estimate_local_frame(patch, center_ij)
        if frame is None:
            return volume_crop, localiser, uv_heatmaps_both

        crop_size = volume_crop.shape[0]
        grid = self._make_rotation_grid(frame.to(device=volume_crop.device, dtype=volume_crop.dtype), crop_size, volume_crop.device, volume_crop.dtype)

        volume_rot = F.grid_sample(
            volume_crop[None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0).squeeze(0)

        localiser_rot = F.grid_sample(
            localiser[None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0).squeeze(0)

        uv_both_rot = F.grid_sample(
            uv_heatmaps_both.permute(3, 0, 1, 2)[None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0).permute(1, 2, 3, 0)

        return volume_rot, localiser_rot, uv_both_rot

    def __iter__(self):
        areas = torch.tensor([patch.area for patch in self._patches])
        area_weights = areas / areas.sum()
        crop_size = torch.tensor(self._config['crop_size'])
        step_size = torch.tensor(self._config['step_size'])
        step_count = torch.tensor(self._config['step_count'])
        heatmap_sigma = self._heatmap_sigma

        sample_idx = 0
        while True:
            sample_idx += 1
            patch = random.choices(self._patches, weights=area_weights)[0]

            random_idx = torch.randint(len(patch.valid_quad_indices) - 1, size=[])
            start_quad_ij = patch.valid_quad_indices[random_idx]
            center_ij = start_quad_ij + torch.rand(size=[2])
            center_zyx = get_zyx_from_patch(center_ij, patch)

            uv_deltas = torch.arange(1, step_count + 1)[:, None] * step_size * patch.scale
            u_pos_shifted_ijs = center_ij + uv_deltas * torch.tensor([1, 0])
            u_neg_shifted_ijs = center_ij - uv_deltas * torch.tensor([1, 0])
            v_pos_shifted_ijs = center_ij + uv_deltas * torch.tensor([0, 1])
            v_neg_shifted_ijs = center_ij - uv_deltas * torch.tensor([0, 1])

            def valid_steps(shifted_ijs):
                ij_int = shifted_ijs.long()
                h, w = patch.valid_vertex_mask.shape[:2]
                in_bounds = (ij_int[:, 0] >= 0) & (ij_int[:, 0] < h) & (ij_int[:, 1] >= 0) & (ij_int[:, 1] < w)
                valid = torch.zeros(shifted_ijs.shape[0], dtype=torch.bool, device=shifted_ijs.device)
                if in_bounds.any():
                    valid[in_bounds] = patch.valid_vertex_mask[ij_int[in_bounds, 0], ij_int[in_bounds, 1]]
                return valid

            u_pos_valid = valid_steps(u_pos_shifted_ijs)
            u_neg_valid = valid_steps(u_neg_shifted_ijs)
            v_pos_valid = valid_steps(v_pos_shifted_ijs)
            v_neg_valid = valid_steps(v_neg_shifted_ijs)

            if torch.rand([]) < 0.5:
                u_pos_shifted_ijs, u_neg_shifted_ijs = u_neg_shifted_ijs, u_pos_shifted_ijs
                u_pos_valid, u_neg_valid = u_neg_valid, u_pos_valid
            if torch.rand([]) < 0.5:
                v_pos_shifted_ijs, v_neg_shifted_ijs = v_neg_shifted_ijs, v_pos_shifted_ijs
                v_pos_valid, v_neg_valid = v_neg_valid, v_pos_valid
            if torch.rand([]) < 0.5:
                u_pos_shifted_ijs, v_pos_shifted_ijs = v_pos_shifted_ijs, u_pos_shifted_ijs
                u_neg_shifted_ijs, v_neg_shifted_ijs = v_neg_shifted_ijs, u_neg_shifted_ijs
                u_pos_valid, v_pos_valid = v_pos_valid, u_pos_valid
                u_neg_valid, v_neg_valid = v_neg_valid, u_neg_valid

            if torch.rand([]) < self._perturb_prob:
                min_corner_zyx = (center_zyx - crop_size // 2).int()

                center_zyx = self._get_perturbed_zyx_from_patch(center_ij, patch, center_ij, min_corner_zyx, crop_size, is_center_point=True)

                def _perturb_or_lookup(shifted_ijs, valid_mask):
                    zyxs = []
                    for i in range(len(shifted_ijs)):
                        if valid_mask[i]:
                            zyxs.append(self._get_perturbed_zyx_from_patch(shifted_ijs[i], patch, center_ij, min_corner_zyx, crop_size, is_center_point=False))
                        else:
                            zyxs.append(get_zyx_from_patch(shifted_ijs[i], patch))
                    return torch.stack(zyxs)

                u_neg_shifted_zyxs = _perturb_or_lookup(u_neg_shifted_ijs, u_neg_valid)
                v_neg_shifted_zyxs = _perturb_or_lookup(v_neg_shifted_ijs, v_neg_valid)

            else:
                u_neg_shifted_zyxs = get_zyx_from_patch(u_neg_shifted_ijs, patch)
                v_neg_shifted_zyxs = get_zyx_from_patch(v_neg_shifted_ijs, patch)

            volume_crop, min_corner_zyx = self._get_crop_from_volume(patch.volume, center_zyx, crop_size)

            u_pos_shifted_zyxs = get_zyx_from_patch(u_pos_shifted_ijs, patch)
            v_pos_shifted_zyxs = get_zyx_from_patch(v_pos_shifted_ijs, patch)

            def make_in_out_heatmaps(pos_shifted_zyxs, neg_shifted_zyxs, cond, suppress_out=None):
                if cond:
                    in_heatmaps = self.make_heatmaps([neg_shifted_zyxs[:1]], min_corner_zyx, crop_size, sigma=heatmap_sigma)
                    out_heatmaps = self.make_heatmaps([pos_shifted_zyxs], min_corner_zyx, crop_size, sigma=heatmap_sigma)
                else:
                    in_heatmaps = torch.zeros([1, crop_size, crop_size, crop_size])
                    out_points = ([pos_shifted_zyxs] if suppress_out != 'pos' else []) + ([neg_shifted_zyxs] if suppress_out != 'neg' else [])
                    out_heatmaps = self.make_heatmaps(out_points, min_corner_zyx, crop_size, sigma=heatmap_sigma) if out_points else torch.zeros([pos_shifted_zyxs.shape[0], crop_size, crop_size, crop_size])
                return in_heatmaps, out_heatmaps

            u_cond, v_cond = torch.rand([2]) < 0.75

            diag_ij = None
            suppress_out_u = suppress_out_v = None

            if (u_cond ^ v_cond) and torch.rand([]) < 0.6:
                diag_is_pos = torch.rand([]) < 0.5
                if u_cond:
                    diag_ij = torch.stack([u_neg_shifted_ijs[0, 0], (v_pos_shifted_ijs if diag_is_pos else v_neg_shifted_ijs)[0, 1]])
                    suppress_out_v = 'neg' if diag_is_pos else 'pos'
                else:
                    diag_ij = torch.stack([(u_pos_shifted_ijs if diag_is_pos else u_neg_shifted_ijs)[0, 0], v_neg_shifted_ijs[0, 1]])
                    suppress_out_u = 'neg' if diag_is_pos else 'pos'
            if (u_cond & v_cond) and torch.rand([]) < 0.5:
                if torch.rand([]) < 0.5:
                    diag_ij = torch.stack([u_neg_shifted_ijs[0, 0], v_pos_shifted_ijs[0, 1]])
                else:
                    diag_ij = torch.stack([u_pos_shifted_ijs[0, 0], v_neg_shifted_ijs[0, 1]])
            if diag_ij is not None:
                if torch.any(diag_ij < 0) or torch.any(diag_ij >= torch.tensor(patch.valid_vertex_mask.shape[:2])):
                    continue
                if not patch.valid_vertex_mask[*diag_ij.int()]:
                    continue
                diag_zyx = get_zyx_from_patch(diag_ij, patch)
            else:
                diag_zyx = None

            if (u_cond and not u_neg_valid[0]) or (v_cond and not v_neg_valid[0]):
                continue

            def build_out_channel_mask(cond, suppress_out, pos_valid, neg_valid):
                if cond:
                    return pos_valid
                mask = torch.zeros_like(pos_valid)
                if suppress_out != 'pos':
                    mask |= pos_valid
                if suppress_out != 'neg':
                    mask |= neg_valid
                return mask

            u_out_channel_mask = build_out_channel_mask(u_cond, suppress_out_u, u_pos_valid, u_neg_valid)
            v_out_channel_mask = build_out_channel_mask(v_cond, suppress_out_v, v_pos_valid, v_neg_valid)
            if not (u_out_channel_mask.any() or v_out_channel_mask.any()):
                continue

            u_in_heatmaps, u_out_heatmaps = make_in_out_heatmaps(u_pos_shifted_zyxs, u_neg_shifted_zyxs, u_cond, suppress_out_u)
            v_in_heatmaps, v_out_heatmaps = make_in_out_heatmaps(v_pos_shifted_zyxs, v_neg_shifted_zyxs, v_cond, suppress_out_v)
            if ~u_cond and ~v_cond:
                u_out_heatmaps = v_out_heatmaps = torch.maximum(u_out_heatmaps, v_out_heatmaps)
                combined_mask = u_out_channel_mask | v_out_channel_mask
                u_out_channel_mask = v_out_channel_mask = combined_mask
            if diag_zyx is not None:
                diag_in_heatmaps = self.make_heatmaps([diag_zyx[None]], min_corner_zyx, crop_size, sigma=heatmap_sigma)
            else:
                diag_in_heatmaps = torch.zeros_like(u_in_heatmaps)
            uv_heatmaps_both = torch.cat([u_in_heatmaps, v_in_heatmaps, diag_in_heatmaps, u_out_heatmaps, v_out_heatmaps], dim=0)
            out_channel_mask = torch.cat([u_out_channel_mask, v_out_channel_mask])

            localiser = build_localiser(center_zyx, min_corner_zyx, crop_size)

            volume_crop, localiser, uv_heatmaps_both = self._apply_forward_alignment(volume_crop, localiser, rearrange(uv_heatmaps_both, 'c z y x -> z y x c'), patch, center_ij)

            while True:
                augmented = self._augmentations(image=volume_crop[None], dist_map=localiser[None], regression_target=uv_heatmaps_both.permute(3, 0, 1, 2))
                if augmented['dist_map'] is not None:
                    break
            volume_crop = augmented['image'].squeeze(0)
            localiser = augmented['dist_map'].squeeze(0)
            uv_heatmaps_both = rearrange(augmented['regression_target'], 'c z y x -> z y x c')
            if not torch.isfinite(volume_crop).all() or not torch.isfinite(localiser).all() or not torch.isfinite(uv_heatmaps_both).all():
                print(f"[DataWarning] NaN/Inf encountered (sample {sample_idx}, volume={getattr(patch, 'volume_name', None)})")
                continue

            uv_heatmaps_in = uv_heatmaps_both[..., :3]
            uv_heatmaps_out = uv_heatmaps_both[..., 3:]
            uv_heatmaps_out_mask = out_channel_mask.to(device=uv_heatmaps_out.device, dtype=uv_heatmaps_out.dtype).view(1, 1, 1, -1).expand_as(uv_heatmaps_out)
            uv_heatmaps_out = uv_heatmaps_out * uv_heatmaps_out_mask

            yield {
                'volume': volume_crop,
                'localiser': localiser,
                'uv_heatmaps_in': uv_heatmaps_in,
                'uv_heatmaps_out': uv_heatmaps_out,
                'uv_heatmaps_out_mask': uv_heatmaps_out_mask,
            }


def mark_context_point(volume, point, value=1.):
    center = (point + 0.5).int()
    offsets = torch.tensor([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0], [0, 0, -2], [0, 0, 2]])
    for offset in offsets:
        volume[tuple(center + offset)] = value


def get_zyx_from_patch(ij, patch):
    original_shape = ij.shape
    batch_dims = original_shape[:-1]
    ij_flat = ij.view(-1, 2)
    normalized_ij = ij_flat / torch.tensor(patch.zyxs.shape[:2]) * 2 - 1
    interpolated = F.grid_sample(
        rearrange(patch.zyxs, 'h w c -> 1 c h w'),
        rearrange(normalized_ij.flip(-1), 'b xy -> 1 b 1 xy'),
        align_corners=True,
        mode='bilinear',
        padding_mode='border'
    )
    return rearrange(interpolated, '1 c b 1 -> b c').view(*batch_dims, -1)


@lru_cache(maxsize=None)
def _get_gaussian_kernel(sigma: float = 2.0):
    """Build and cache a 3D Gaussian kernel for heatmap smoothing."""
    sigma = float(sigma)
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    z_coords, y_coords, x_coords = torch.meshgrid(coords, coords, coords, indexing='ij')
    kernel = torch.exp(-(z_coords**2 + y_coords**2 + x_coords**2) / (2 * sigma**2))
    return kernel, kernel_size

def make_heatmaps(all_zyxs, min_corner_zyx, crop_size, sigma: float = 2.0):
    return HeatmapDatasetV2.make_heatmaps(all_zyxs, min_corner_zyx, crop_size, apply_gaussian=True, sigma=sigma)


def load_datasets(config):
    train_patches = []
    val_patches = []
    for dataset in config['datasets']:
        volume_path = dataset['volume_path']
        if False:
            # TODO: how does this interact with multiple dataloader workers? cache is created before fork, hence presumably not shared?
            if "://" in volume_path or "::" in volume_path:
                store = zarr.storage.FSStore(volume_path, mode='r')
            else:
                store = zarr.storage.DirectoryStore(volume_path)
            store = zarr.storage.LRUStoreCache(store, max_size=12*1024**3)
            ome_zarr = zarr.open_group(store, mode='r')
        else:
            ome_zarr = zarr.open(volume_path, mode='r')

        volume_scale = dataset['volume_scale']
        volume = ome_zarr[str(volume_scale)]

        patches = load_tifxyz_patches(dataset['segments_path'], dataset.get('z_range', None), volume, volume_path)
        patches_wrt_volume_scale = dataset.get('segments_scale', 0)  # if specified, patches are assumed to already target the volume at this scale

        if 'roi_path' in dataset:
            # If specified, roi_path is a proofreader log; filter & crop patches to approved cubes
            with open(dataset['roi_path'], 'r') as f:
                proofreader_json = json.load(f)
            assert tuple(proofreader_json['metadata']['volume_shape']) == ome_zarr[str(patches_wrt_volume_scale)].shape
            approved_cubes = [{
                'min_zyx': patch['coords'],
                'size': patch['patch_size'],
            } for patch in proofreader_json['approved_patches']]
            patches = filter_patches_by_roi(patches, approved_cubes)

        patches = [patch.retarget(2 ** (volume_scale - patches_wrt_volume_scale)) for patch in patches]

        num_val_per_volume = config.get('num_val_segments_per_volume', 1)
        train_patches.extend(patches[num_val_per_volume:])
        val_patches.extend(patches[:num_val_per_volume])
        
    print(f'loaded {len(train_patches)} train patches and {len(val_patches)} val patches')
    return train_patches, val_patches


def load_tifxyz_patches(segments_path, z_range, volume, volume_name=None):

    segment_paths = glob.glob(segments_path + "/*")
    segment_paths = sorted([path for path in segment_paths if os.path.isdir(path)])
    print(f'found {len(segment_paths)} tifxyz patches')

    all_patches = []
    for segment_path in tqdm(segment_paths, desc='loading tifxyz patches'):
        try:  # TODO: remove
            # TODO: move this bit to a method in tifxyz.py
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
            all_patches.append(Patch(zyxs, scale, volume, volume_name))
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
            filtered_patches.append(Patch(cropped_zyxs, patch.scale, patch.volume, patch.volume_name))
    return filtered_patches


def get_crop_from_volume(volume, center_zyx, crop_size, normalize=True):
    """Crop volume around center point, padding with zeros if needed.
    When normalize is True, scales to [-1, 1]; otherwise returns raw values as float32."""
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
    ]).to(torch.float32)
    if normalize:
        # TODO: should instead always use standardised uint8 volumes!
        if volume.dtype == np.uint8:
            volume_crop = volume_crop / 255.
        else:
            # Guard against empty/constant crops (amax == 0) or NaNs in the source data
            max_val = torch.nan_to_num(volume_crop.abs().amax(), nan=0.0, posinf=0.0, neginf=0.0)
            max_val = max_val.clamp_min(1e-6)
            volume_crop = volume_crop / max_val
        volume_crop = torch.nan_to_num(volume_crop * 2 - 1, nan=-1.0, posinf=1.0, neginf=-1.0)

    # Pad if needed
    pad_before = actual_min - crop_min
    pad_after = crop_max - actual_max
    if torch.any(pad_before > 0) or torch.any(pad_after > 0):
        paddings = pad_before[2], pad_after[2], pad_before[1], pad_after[1], pad_before[0], pad_after[0]
        volume_crop = F.pad(volume_crop, paddings, mode='constant', value=0)

    return volume_crop, crop_min


def build_localiser(center_zyx, min_corner_zyx, crop_size):
    localiser = torch.linalg.norm(torch.stack(torch.meshgrid(*[torch.arange(crop_size)] * 3, indexing='ij'), dim=-1).to(torch.float32) - crop_size // 2, dim=-1)
    localiser = localiser / localiser.amax() * 2 - 1
    mark_context_point(localiser, center_zyx - min_corner_zyx, value=0.)
    return localiser
