import os
import cv2
import zarr
import json
import glob
import torch
import random
import numpy as np
import scipy.ndimage
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

    def __post_init__(self):
        # Construct the valid *quads* mask; the ij'th element says whether all four corners of the quad with min-corner at ij are valid
        self.valid_quad_mask = torch.any(self.zyxs[:-1, :-1] != -1, dim=-1) & torch.any(self.zyxs[1:, :-1] != -1, dim=-1) & torch.any(self.zyxs[:-1, 1:] != -1, dim=-1) & torch.any(self.zyxs[1:, 1:] != -1, dim=-1)
        self.valid_quad_indices = torch.stack(torch.where(self.valid_quad_mask), dim=-1)
        self.valid_vertex_mask = torch.any(self.zyxs != -1, dim=-1)
        assert len(self.valid_quad_indices) > 0
        self.area = self.valid_quad_mask.sum() * (1 / self.scale).prod()
        self.quad_centers = torch.where(self.valid_quad_mask[..., None], 0.5 * (self.zyxs[1:, 1:] + self.zyxs[:-1, :-1]), torch.tensor(-1.))

    def retarget(self, factor):
        # Retarget the patch to a volume downsampled by the given factor
        return Patch(
            torch.where((self.zyxs == -1).all(dim=-1, keepdim=True), -1, self.zyxs / factor),
            self.scale * factor,
            self.volume
        )

class HeatmapDatasetV2(torch.utils.data.IterableDataset):
    _channel_range_cache = {}
    _kernel_offsets_cache = {}
    _kernel_value_cache = {}
    _quad_weight_cache = {}

    def __init__(self, config, patches_for_split, multistep_count, bidirectional):
        self._config = config
        self._patches = patches_for_split
        self._multistep_count = multistep_count
        self._bidirectional = bidirectional
        self._heatmap_sigma = float(config.get('heatmap_sigma', 2.0))
        self._augmentations = augmentation.get_training_augmentations(config['crop_size'], config['augmentation']['allow_transposes'],config['augmentation']['allow_mirroring'], config['augmentation']['only_spatial_and_intensity'])
        self._perturb_prob = config['point_perturbation']['perturb_probability']
        self._uv_max_perturbation = config['point_perturbation']['uv_max_perturbation']  # measured in voxels
        self._w_max_perturbation = config['point_perturbation']['w_max_perturbation']  # measured in voxels
        self._main_component_distance_factor = config['point_perturbation']['main_component_distance_factor']

        self._perturb_cache_key = None
        self._perturb_cache_value = None
        self._volume_patch_bboxes = {}
        self._sampling = {}
        self._quad_bboxes = {}
        self._vertex_normals = {}

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
            volume_key = id(patch.volume)
            self._volume_patch_bboxes.setdefault(volume_key, []).append((patch, bbox_min, bbox_max))
            self._vertex_normals[id(patch)] = self._compute_vertex_normals(patch)

    def _reset_iter_caches(self):
        self._perturb_cache_key = None
        self._perturb_cache_value = None

    def _make_cache_key(self, patch, center_ij, min_corner_zyx, crop_size):
        return (
            id(patch),
            tuple(center_ij.tolist()),
            tuple(min_corner_zyx.tolist()),
            int(crop_size),
        )

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

    @classmethod
    def _get_quad_weights(cls, points_per_side, device):
        key = (int(points_per_side[0]), int(points_per_side[1]), device)
        if key not in cls._quad_weight_cache:
            v_points = torch.arange(points_per_side[0], dtype=torch.float32, device=device) / points_per_side[0]
            u_points = torch.arange(points_per_side[1], dtype=torch.float32, device=device) / points_per_side[1]
            cls._quad_weight_cache[key] = cls._make_quad_weights(v_points, u_points)
        return cls._quad_weight_cache[key]

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
        quad_corners_flat = filtered_quads_zyxs.view(filtered_quads_zyxs.shape[0], 4, 3)
        weights = self._get_quad_weights(points_per_side, device=filtered_quads_zyxs.device)
        points_covering_quads = torch.einsum("kc,ncd->nkd", weights, quad_corners_flat)

        return points_covering_quads.reshape(-1, 3)

    def _get_quads_in_crop(self, patch, min_corner_zyx, crop_size):
        """Get mask of quads that fall within the crop region."""
        bbox_min, bbox_max = self._quad_bboxes[id(patch)]
        crop_min = min_corner_zyx.to(dtype=bbox_min.dtype)
        crop_size_tensor = torch.as_tensor(crop_size, dtype=bbox_min.dtype, device=crop_min.device)
        crop_max = crop_min + crop_size_tensor
        if (bbox_max < crop_min).any() or (bbox_min >= crop_max).any():
            return torch.zeros_like(patch.valid_quad_mask)

        return patch.valid_quad_mask & torch.all(patch.quad_centers >= crop_min, dim=-1) & torch.all(patch.quad_centers < crop_max, dim=-1)

    @staticmethod
    def _compute_vertex_normals(patch):
        """Pre-compute unit normals per vertex from valid quads."""
        zyxs = patch.zyxs.to(dtype=torch.float32)
        normals = torch.zeros_like(zyxs, dtype=torch.float32)

        tl = zyxs[:-1, :-1]
        tr = zyxs[:-1, 1:]
        bl = zyxs[1:, :-1]
        quad_normals = torch.linalg.cross(tr - tl, bl - tl)
        quad_normals = quad_normals * patch.valid_quad_mask[..., None]

        normals[:-1, :-1] += quad_normals
        normals[:-1, 1:] += quad_normals
        normals[1:, :-1] += quad_normals
        normals[1:, 1:] += quad_normals

        norms = torch.linalg.norm(normals, dim=-1, keepdim=True)
        normals = torch.where(norms > 1e-6, normals / norms, torch.zeros_like(normals))
        normals = torch.where(patch.valid_vertex_mask[..., None], normals, torch.zeros_like(normals))
        return normals

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
            component_mask = torch.zeros_like(quad_in_crop)
            return component_mask
        
        center_quad = center_ij.int()
        if (center_quad[0] < 0 or center_quad[0] >= quad_in_crop.shape[0] or 
            center_quad[1] < 0 or center_quad[1] >= quad_in_crop.shape[1] or
            not quad_in_crop[center_quad[0], center_quad[1]]):
            component_mask = torch.zeros_like(quad_in_crop)
            return component_mask

        # use scipy label instead of dfs; use 8-connectivity (including diagonals) to avoid dashed lines
        structure = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]], dtype=np.int8)
        labeled, _ = scipy.ndimage.label(quad_in_crop.cpu().numpy(), structure=structure)
        label = labeled[center_quad[0].item(), center_quad[1].item()]
        if label == 0:
            component_mask = torch.zeros_like(quad_in_crop)
            return component_mask

        component_mask = torch.as_tensor(labeled == label, device=quad_in_crop.device)
        return component_mask

    def _compute_cached_patch_points(self, current_patch, center_ij, min_corner_zyx, crop_size):
        """Pre-compute the patch-point sets for a given crop context."""
        quad_main_component = self._get_current_patch_center_component_mask(current_patch, center_ij, min_corner_zyx, crop_size)

        all_patch_points = []
        crop_min = min_corner_zyx.to(dtype=current_patch.zyxs.dtype, device=current_patch.zyxs.device, non_blocking=True)
        crop_max = crop_min + torch.as_tensor(crop_size, dtype=crop_min.dtype, device=crop_min.device)

        volume_key = id(current_patch.volume)

        # Check all other patches from the same volume
        for other_patch, bbox_min, bbox_max in self._volume_patch_bboxes.get(volume_key, []):
            if other_patch is current_patch:
                continue  # handled separately below
            if (bbox_max < crop_min).any() or (bbox_min >= crop_max).any():
                continue  # cheap reject: patch bbox does not overlap crop
            patch_points = self._get_patch_points_in_crop(other_patch, min_corner_zyx, crop_size)
            if len(patch_points) > 0:
                all_patch_points.append(patch_points)

        # Check other parts of the current patch (excluding main component)
        quad_in_crop = self._get_quads_in_crop(current_patch, min_corner_zyx, crop_size)
        quad_excluding_main = quad_in_crop & ~quad_main_component

        # Sample points from remaining parts of current patch
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
            self._perturb_cache_value = {"concat": concat}
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
                print("[DataWarning] empty patch_points in distance check")
                continue
            # Calculate minimum distance to any point in this patch
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
        i, j = perturbed_ij.int()
        normal = self._vertex_normals[id(patch)][i, j]
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
            # Normal is too small or invalid, skip 3D perturbation
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
        return cls._kernel_offsets_cache[sigma].to(device=device, non_blocking=True)

    @classmethod
    def _get_kernel_values(cls, device, dtype, sigma):
        key = (device, dtype, float(sigma))
        if key not in cls._kernel_value_cache:
            kernel, _ = _get_gaussian_kernel(sigma)
            cls._kernel_value_cache[key] = kernel.to(device=device, dtype=dtype, non_blocking=True).reshape(-1)
        return cls._kernel_value_cache[key]

    @classmethod
    def _collect_coords(cls, all_zyxs, min_corner_zyx, crop_size, device, dtype):
        """Collect valid coordinates within the crop bounds.

        Args:
            crop_size: int for cubic, or tuple/list of 3 ints [D, H, W] for anisotropic
        """
        # Normalize crop_size to tensor
        if isinstance(crop_size, (list, tuple)):
            crop_size_tensor = torch.tensor(crop_size, device=device, dtype=torch.int64)
        else:
            crop_size_tensor = torch.tensor([crop_size, crop_size, crop_size], device=device, dtype=torch.int64)

        channel_count = all_zyxs[0].shape[0]
        coords_accum = []
        channel_accum = []
        channel_range = cls._get_channel_range(channel_count, device)
        min_corner = min_corner_zyx.to(device=device, dtype=dtype, non_blocking=True)

        for zyxs in all_zyxs:
            coords = (zyxs.to(device=device, dtype=dtype, non_blocking=True) - min_corner + 0.5).to(torch.int64)
            valid_mask = (coords >= 0).all(dim=1) & (coords < crop_size_tensor).all(dim=1)
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
        # Normalize crop_size to tuple of 3 ints [D, H, W]
        if isinstance(crop_size, (list, tuple)):
            crop_size_dhw = tuple(crop_size)
        else:
            crop_size_dhw = (crop_size, crop_size, crop_size)
        dtype = all_zyxs[0].dtype
        device = all_zyxs[0].device
        channel_count, coords, channels = cls._collect_coords(all_zyxs, min_corner_zyx, crop_size, device, dtype)
        heatmaps = torch.zeros((channel_count, *crop_size_dhw), device=device, dtype=dtype)

        if coords is None:
            return heatmaps

        z, y, x = coords.unbind(dim=1)
        values = torch.ones_like(channels, dtype=dtype)
        heatmaps.index_put_((channels, z, y, x), values, accumulate=True)
        return heatmaps

    @classmethod
    def _make_heatmaps_with_grad(cls, all_zyxs, min_corner_zyx, crop_size, sigma):

        if len(all_zyxs) != 1 or all_zyxs[0].shape[0] != 1:
            raise NotImplementedError  # for now we only support the special case of one heatmap in this differentiable version!
        zyx = all_zyxs[0]

        # Normalize crop_size to tuple of 3 ints [D, H, W]
        if isinstance(crop_size, (list, tuple)):
            crop_size_dhw = tuple(crop_size)
        else:
            crop_size_dhw = (crop_size, crop_size, crop_size)

        coords = min_corner_zyx + torch.stack(torch.meshgrid(
            torch.arange(crop_size_dhw[0], device=zyx.device),
            torch.arange(crop_size_dhw[1], device=zyx.device),
            torch.arange(crop_size_dhw[2], device=zyx.device),
            indexing='ij'
        ), dim=-1)
        heatmap = torch.exp(-((coords - zyx) ** 2).sum(dim=-1) / (2 * sigma ** 2))

        return heatmap.unsqueeze(0)

    @classmethod
    def make_heatmaps(cls, all_zyxs, min_corner_zyx, crop_size, apply_gaussian=True, sigma: float = 2.0):
        if not apply_gaussian:
            return cls._scatter_heatmaps(all_zyxs, min_corner_zyx, crop_size)

        if any([zyxs.requires_grad for zyxs in all_zyxs]) or min_corner_zyx.requires_grad:
            return cls._make_heatmaps_with_grad(all_zyxs, min_corner_zyx, crop_size, sigma)

        # Normalize crop_size to tuple of 3 ints [D, H, W]
        if isinstance(crop_size, (list, tuple)):
            crop_size_dhw = tuple(crop_size)
        else:
            crop_size_dhw = (crop_size, crop_size, crop_size)
        crop_size_tensor = torch.tensor(crop_size_dhw)

        device = all_zyxs[0].device
        channel_count, coords, channels = cls._collect_coords(all_zyxs, min_corner_zyx, crop_size, device, torch.float32)
        heatmaps = torch.zeros((channel_count, *crop_size_dhw), device=device, dtype=torch.float32)

        if coords is None:
            return heatmaps

        kernel_offsets = cls._get_kernel_offsets(device, sigma)
        kernel_values = cls._get_kernel_values(device, torch.float32, sigma)

        expanded_coords = coords[:, None, :] + kernel_offsets[None, :, :]
        crop_size_bounds = crop_size_tensor.to(device=device)
        in_bounds = (expanded_coords >= 0).all(dim=-1) & (expanded_coords < crop_size_bounds).all(dim=-1)

        if torch.any(in_bounds):
            valid_positions = expanded_coords[in_bounds]
            valid_channels = channels[:, None].expand(-1, kernel_offsets.shape[0])[in_bounds]
            valid_values = kernel_values.expand(coords.shape[0], -1)[in_bounds]

            z, y, x = valid_positions.unbind(dim=1)
            heatmaps.index_put_((valid_channels, z, y, x), valid_values, accumulate=True)

        return heatmaps

    def _decide_conditioning(self, use_multistep, u_neg_valid, v_neg_valid, u_pos_shifted_ijs, u_neg_shifted_ijs, v_pos_shifted_ijs, v_neg_shifted_ijs, patch):
        """Decide conditioning directions and diagonal point. Returns None to signal resample needed.

        Returns:
            dict with keys: u_cond, v_cond, suppress_out_u, suppress_out_v, diag_zyx
            or None if this sample should be skipped
        """
        if not use_multistep:
            u_cond, v_cond = torch.rand([2]) < 0.75
        else:
            # For now, multi-step only works for a 'chain', i.e. conditioning on exactly one of u/v
            u_cond, v_cond = torch.tensor([True, False] if torch.rand([]) < 0.5 else [False, True])

        # Can't condition on a missing point
        if (u_cond and not u_neg_valid[0]) or (v_cond and not v_neg_valid[0]):
            return None

        diag_ij = None
        diag_zyx = None
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
                return None
            if not patch.valid_vertex_mask[*diag_ij.int()]:
                return None
            diag_zyx = get_zyx_from_patch(diag_ij, patch)

        return {
            'u_cond': u_cond,
            'v_cond': v_cond,
            'suppress_out_u': suppress_out_u,
            'suppress_out_v': suppress_out_v,
            'diag_zyx': diag_zyx,
        }

    def _build_final_heatmaps(
        self,
        min_corner_zyx,
        crop_size,
        heatmap_sigma,
        u_pos_shifted_zyxs,
        u_neg_shifted_zyxs,
        v_pos_shifted_zyxs,
        v_neg_shifted_zyxs,
        u_neg_shifted_zyxs_unperturbed=None,
        v_neg_shifted_zyxs_unperturbed=None,
        u_cond=None,
        v_cond=None,
        suppress_out_u=None,
        suppress_out_v=None,
        diag_zyx=None,
        center_zyx_unperturbed=None,
    ):
        """Build heatmaps using u/v direction conditioning."""

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

        # *_in_heatmaps always have a single plane, either the first negative point or empty
        # *_out_heatmaps always have one plane per step, and may contain only positive or both positive and negative points
        u_in_heatmaps, u_out_heatmaps = make_in_out_heatmaps(u_pos_shifted_zyxs, u_neg_shifted_zyxs, u_cond, suppress_out_u)
        v_in_heatmaps, v_out_heatmaps = make_in_out_heatmaps(v_pos_shifted_zyxs, v_neg_shifted_zyxs, v_cond, suppress_out_v)
        if ~u_cond and ~v_cond:
            # In this case U & V are (nearly) indistinguishable, so don't force the model to separate them
            u_out_heatmaps = v_out_heatmaps = torch.maximum(u_out_heatmaps, v_out_heatmaps)
        if diag_zyx is not None:
            diag_in_heatmaps = self.make_heatmaps([diag_zyx[None]], min_corner_zyx, crop_size, sigma=heatmap_sigma)
        else:
            diag_in_heatmaps = torch.zeros_like(u_in_heatmaps)

        uv_heatmaps_in_all = torch.cat([u_in_heatmaps, v_in_heatmaps, diag_in_heatmaps], dim=0)
        uv_heatmaps_out_all = torch.cat([u_out_heatmaps, v_out_heatmaps], dim=0)
        condition_channels = uv_heatmaps_in_all.shape[0]
        uv_heatmaps_both = torch.cat([uv_heatmaps_in_all, uv_heatmaps_out_all], dim=0)

        if center_zyx_unperturbed is not None:
            maybe_center_heatmap = {
                'center_heatmap': self.make_heatmaps([center_zyx_unperturbed[None]], min_corner_zyx, crop_size, sigma=heatmap_sigma)
            }
        else:
            maybe_center_heatmap = {}

        # Generate srf_overlap mask if enabled
        srf_overlap_mask = None
        if self._config.get('aux_srf_overlap', False):
            from vesuvius.neural_tracing.surf_overlap_loss import render_surf_overlap_mask
            srf_overlap_thickness = self._config.get('srf_overlap_thickness', 2.0)
            # Use first step coordinates for srf_overlap (step 0)
            srf_overlap_mask = render_surf_overlap_mask(
                u_neg_shifted_zyxs[0], u_pos_shifted_zyxs[0],
                v_neg_shifted_zyxs[0], v_pos_shifted_zyxs[0],
                min_corner_zyx, crop_size, thickness=srf_overlap_thickness
            )

        return {
            'uv_heatmaps_both': uv_heatmaps_both,
            'condition_channels': condition_channels,
            'srf_overlap_mask': srf_overlap_mask,
            **maybe_center_heatmap,
        }

    def _should_flip_uv_directions(self):
        """Whether to randomly flip positive/negative directions as augmentation."""
        return bool(self._config.get("flip_uv_directions", True))

    def _should_swap_uv_axes(self):
        """Whether to apply UV axis swap augmentation. Override in subclasses."""
        return True

    def _build_batch_dict(
        self,
        volume_crop,
        localiser,
        uv_heatmaps_in,
        uv_heatmaps_out,
        seg,
        seg_mask,
        normals,
        normals_mask,
        center_heatmap,
        srf_overlap_mask=None,
    ):
        """Build the batch dictionary. Override in subclasses to add masking."""
        batch_dict = {
            'volume': volume_crop,
            'localiser': localiser,
            'uv_heatmaps_in': uv_heatmaps_in,
            'uv_heatmaps_out': uv_heatmaps_out,
            **({'center_heatmap': center_heatmap} if center_heatmap is not None else {}),
        }

        if self._config.get("aux_segmentation", False) and seg is not None:
            batch_dict.update({'seg': seg, 'seg_mask': seg_mask})
        if self._config.get("aux_normals", False) and normals is not None:
            batch_dict.update({'normals': normals, 'normals_mask': normals_mask})
        if self._config.get("aux_srf_overlap", False) and srf_overlap_mask is not None:
            batch_dict.update({'srf_overlap_mask': srf_overlap_mask})

        return batch_dict

    def __iter__(self):

        areas = torch.tensor([patch.area for patch in self._patches])
        area_weights = areas / areas.sum()
        crop_size = torch.tensor(self._config['crop_size'])
        step_size = torch.tensor(self._config['step_size'])
        multistep_prob = float(self._config.get('multistep_prob', 0.0))
        multistep_count = self._multistep_count
        heatmap_sigma = self._heatmap_sigma

        crop_size += step_size * torch.tensor(self._config['step_count']) * (multistep_count - 1) * 2

        while True:
            self._reset_iter_caches()
            patch = random.choices(self._patches, weights=area_weights)[0]

            # Decide whether to return multi-step targets for this data point
            if multistep_count > 1 and torch.rand([]) < multistep_prob:
                step_count = multistep_count * torch.tensor(self._config['step_count'])
                use_multistep = True
            else:
                step_count = torch.tensor(self._config['step_count'])
                use_multistep = False

            # Sample a random valid quad in the patch, then a point in that quad
            random_idx = torch.randint(len(patch.valid_quad_indices), size=[])
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
                """Mark which shifted points remain in-bounds and on valid vertices."""
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

            # If any step along U or V would fall outside the patch or onto an invalid vertex, resample.
            if not (u_pos_valid.all() and u_neg_valid.all() and v_pos_valid.all() and v_neg_valid.all()):
                continue

            # Randomly flip positive and negative directions, as a form of augmentation since they're arbitrary
            if self._should_flip_uv_directions():
                if torch.rand([]) < 0.5:
                    u_pos_shifted_ijs, u_neg_shifted_ijs = u_neg_shifted_ijs, u_pos_shifted_ijs
                    u_pos_valid, u_neg_valid = u_neg_valid, u_pos_valid
                if torch.rand([]) < 0.5:
                    v_pos_shifted_ijs, v_neg_shifted_ijs = v_neg_shifted_ijs, v_pos_shifted_ijs
                    v_pos_valid, v_neg_valid = v_neg_valid, v_pos_valid

            # Decide conditioning directions early so we know which points to perturb
            cond_result = self._decide_conditioning(
                use_multistep, u_neg_valid, v_neg_valid,
                u_pos_shifted_ijs, u_neg_shifted_ijs, v_pos_shifted_ijs, v_neg_shifted_ijs, patch
            )
            if cond_result is None:
                continue
            u_cond = cond_result['u_cond']
            v_cond = cond_result['v_cond']
            suppress_out_u = cond_result['suppress_out_u']
            suppress_out_v = cond_result['suppress_out_v']
            diag_zyx = cond_result['diag_zyx']

            # Anchor the main surface component to the *unperturbed* conditioning context
            center_zyx_unperturbed = center_zyx.clone()
            anchor_candidates_world = [
                get_zyx_from_patch(u_neg_shifted_ijs[0], patch),
                get_zyx_from_patch(v_neg_shifted_ijs[0], patch),
                center_zyx_unperturbed,
            ]

            # Get negative coordinates (unperturbed)
            u_neg_shifted_zyxs_unperturbed = get_zyx_from_patch(u_neg_shifted_ijs, patch)
            v_neg_shifted_zyxs_unperturbed = get_zyx_from_patch(v_neg_shifted_ijs, patch)

            # Start with unperturbed, then optionally apply perturbation
            u_neg_shifted_zyxs = u_neg_shifted_zyxs_unperturbed.clone()
            v_neg_shifted_zyxs = v_neg_shifted_zyxs_unperturbed.clone()

            # Apply perturbations only to the directions we're conditioning on
            if torch.rand([]) < self._perturb_prob:
                min_corner_zyx = (center_zyx - crop_size // 2).int()

                # Perturb center point in 3D (only normal perturbation, no uv)
                center_zyx = self._get_perturbed_zyx_from_patch(center_ij, patch, center_ij, min_corner_zyx, crop_size, is_center_point=True)

                # Only perturb the first negative point for each conditioned direction
                if u_cond:
                    u_neg_shifted_zyxs[0] = self._get_perturbed_zyx_from_patch(u_neg_shifted_ijs[0], patch, center_ij, min_corner_zyx, crop_size, is_center_point=False)
                if v_cond:
                    v_neg_shifted_zyxs[0] = self._get_perturbed_zyx_from_patch(v_neg_shifted_ijs[0], patch, center_ij, min_corner_zyx, crop_size, is_center_point=False)

            # Get crop volume and its min-corner (which may be slightly negative)
            volume_crop, min_corner_zyx = get_crop_from_volume(patch.volume, center_zyx, crop_size)

            seg = seg_mask = None
            normals = normals_mask = None
            surface_pts_crop = None
            normals_covering_quads = None

            needs_surface = self._config.get("aux_segmentation", False) or self._config.get("aux_normals", False)
            surface_mask = None
            anchor_voxel = None
            if needs_surface:
                quad_in_crop = self._get_quads_in_crop(patch, min_corner_zyx, crop_size)

                if torch.any(quad_in_crop):
                    info = self._sampling[id(patch)]
                    weights = info["uv_weights"]  # [P, 4]

                    quad_corners = info["quad_corners_flat"][quad_in_crop]
                    points_covering_quads = torch.einsum("kc,ncd->nkd", weights, quad_corners).reshape(-1, 3)
                    surface_pts_crop = (points_covering_quads - min_corner_zyx.to(points_covering_quads.device, non_blocking=True)).round().long()
                    in_bounds = ((surface_pts_crop >= 0) & (surface_pts_crop < crop_size)).all(dim=1)
                    surface_pts_crop = surface_pts_crop[in_bounds]
                    if surface_pts_crop.numel() > 0:
                        surface_mask = torch.zeros((1, crop_size, crop_size, crop_size), device=surface_pts_crop.device, dtype=torch.float32)
                        z, y, x = surface_pts_crop.unbind(dim=1)
                        surface_mask.index_put_((torch.zeros_like(z), z, y, x), torch.ones_like(z, dtype=torch.float32), accumulate=True)
                        surface_mask = (surface_mask > 0).float()
                        # Choose a component anchor from *unperturbed* conditioning points, then center
                        for cand in anchor_candidates_world:
                            voxel = (cand - min_corner_zyx).round().long()
                            voxel = torch.clamp(voxel, 0, crop_size - 1)
                            if surface_mask[0, voxel[0], voxel[1], voxel[2]] > 0:
                                anchor_voxel = voxel
                                break
                        # Keep only the connected component touching the chosen anchor to avoid fragmented masks
                        if anchor_voxel is not None and surface_mask[0, anchor_voxel[0], anchor_voxel[1], anchor_voxel[2]] > 0:
                            structure = np.ones((3, 3, 3), dtype=np.int8)
                            labeled, _ = scipy.ndimage.label(surface_mask[0].cpu().numpy(), structure=structure)
                            label = labeled[anchor_voxel[0].item(), anchor_voxel[1].item(), anchor_voxel[2].item()]
                            if label != 0:
                                surface_mask_np = (labeled == label)[None].astype(np.float32)
                                surface_mask = torch.from_numpy(surface_mask_np).to(surface_mask.device, non_blocking=True)
                    if self._config.get("aux_normals", False):
                        vertex_normals = self._vertex_normals[id(patch)]
                        filtered_quad_normals = torch.stack([
                            torch.stack([vertex_normals[:-1, :-1][quad_in_crop], vertex_normals[:-1, 1:][quad_in_crop]], dim=1),
                            torch.stack([vertex_normals[1:, :-1][quad_in_crop], vertex_normals[1:, 1:][quad_in_crop]], dim=1),
                        ], dim=1)  # [N, 2, 2, 3]
                        quad_normals_flat = filtered_quad_normals.view(filtered_quad_normals.shape[0], 4, 3)
                        normals_covering_quads = torch.einsum("kc,ncd->nkd", weights, quad_normals_flat).reshape(-1, 3)
                        normals_covering_quads = normals_covering_quads[in_bounds]

            if self._config.get("aux_normals", False):
                # Build dense normals by bilinearly interpolating vertex normals within each quad and splatting to voxels.
                if normals_covering_quads is None or surface_pts_crop is None or surface_pts_crop.numel() == 0:
                    continue
                normals = torch.zeros((3, crop_size, crop_size, crop_size), device=surface_pts_crop.device, dtype=torch.float32)
                counts = torch.zeros((1, crop_size, crop_size, crop_size), device=surface_pts_crop.device, dtype=torch.float32)
                z, y, x = surface_pts_crop.unbind(dim=1)
                norms = torch.linalg.norm(normals_covering_quads, dim=1, keepdim=True)
                normals_covering_quads = torch.where(norms > 1e-6, normals_covering_quads / norms, torch.zeros_like(normals_covering_quads))
                normals.index_put_((torch.arange(3, device=surface_pts_crop.device)[:, None], z[None], y[None], x[None]), normals_covering_quads.T, accumulate=True)
                counts.index_put_((torch.zeros_like(z), z, y, x), torch.ones_like(z, dtype=torch.float32), accumulate=True)
                mask = (counts > 0).float()
                normals = torch.where(mask > 0, normals / torch.clamp(counts, min=1e-6), torch.zeros_like(normals))
                # Use the same occupancy mask for normals and segmentation to avoid mismatches
                normals_mask = surface_mask if surface_mask is not None else mask
                if anchor_voxel is not None:
                    normals_mask[0, anchor_voxel[0], anchor_voxel[1], anchor_voxel[2]] = 1.0

            if self._config.get("aux_segmentation", False):
                # Build segmentation from the same surface occupancy as normals to keep both labels aligned.
                if surface_mask is None:
                    continue
                seg = surface_mask.squeeze(0).clone()

                center_voxel = (center_zyx - min_corner_zyx).round().long()
                center_voxel = torch.clamp(center_voxel, 0, crop_size - 1)
                seg[center_voxel[0], center_voxel[1], center_voxel[2]] = 1.0
                # Only supervise where we have a label
                seg_mask = seg.clone()

            # Map to 3D space and construct heatmaps
            u_pos_shifted_zyxs = get_zyx_from_patch(u_pos_shifted_ijs, patch)
            v_pos_shifted_zyxs = get_zyx_from_patch(v_pos_shifted_ijs, patch)

            heatmap_result = self._build_final_heatmaps(
                min_corner_zyx=min_corner_zyx,
                crop_size=crop_size,
                heatmap_sigma=heatmap_sigma,
                u_pos_shifted_zyxs=u_pos_shifted_zyxs,
                u_neg_shifted_zyxs=u_neg_shifted_zyxs,
                v_pos_shifted_zyxs=v_pos_shifted_zyxs,
                v_neg_shifted_zyxs=v_neg_shifted_zyxs,
                u_neg_shifted_zyxs_unperturbed=u_neg_shifted_zyxs_unperturbed,
                v_neg_shifted_zyxs_unperturbed=v_neg_shifted_zyxs_unperturbed,
                u_cond=u_cond,
                v_cond=v_cond,
                suppress_out_u=suppress_out_u,
                suppress_out_v=suppress_out_v,
                diag_zyx=diag_zyx,
                center_zyx_unperturbed=center_zyx_unperturbed if self._bidirectional else None,
            )

            uv_heatmaps_both = heatmap_result['uv_heatmaps_both']
            heatmap_num_in_channels = heatmap_result['condition_channels']
            maybe_center_heatmap = heatmap_result['center_heatmap'] if 'center_heatmap' in heatmap_result else None
            srf_overlap_mask = heatmap_result.get('srf_overlap_mask', None)

            # Build localiser volume
            localiser = build_localiser(center_zyx, min_corner_zyx, crop_size)

            #TODO: include full 2d slices for additional context
            #  if so, need to augment them consistently with the 3d crop -> tricky for geometric transforms

            # FIXME: the loop is a hack because some augmentation sometimes randomly returns None
            #  we should instead just remove the relevant augmentation (or fix it!)
            # TODO: consider interaction of augmentation with localiser -- logically should follow translations of
            #  the center-point, since the heatmaps do, but not follow rotations/scales; however in practice maybe
            #  ok since it's 'just more augmentation' that won't be applied during tracing
            uv_channels = uv_heatmaps_both.shape[0]
            regression_target = torch.cat([uv_heatmaps_both] + ([maybe_center_heatmap] if maybe_center_heatmap is not None else []), dim=0)
            seg_for_aug = seg[None] if seg is not None else None
            while True:
                augmented = self._augmentations(
                    image=volume_crop[None],
                    dist_map=localiser[None],
                    regression_target=regression_target,
                    segmentation=seg_for_aug,
                    normals=normals,
                    normals_mask=normals_mask,
                )
                if augmented['dist_map'] is not None:
                    break
            volume_crop = augmented['image'].squeeze(0)
            localiser = augmented['dist_map'].squeeze(0)
            regression_aug = rearrange(augmented['regression_target'], 'c z y x -> z y x c')
            uv_heatmaps_both = regression_aug[..., :uv_channels]
            maybe_center_heatmap = regression_aug[..., uv_channels] if uv_channels < regression_aug.shape[-1] else None
            if seg is not None:
                seg_aug = augmented.get('segmentation', None)
                if seg_aug is None:
                    continue
                seg = rearrange(seg_aug, 'c z y x -> z y x c')[..., 0]
                # dilate the seg label to add a bit of thickness
                seg = F.max_pool3d(seg[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
                seg_mask = seg
            if normals is not None:
                # Use explicitly tracked normals so vector components are permuted correctly
                normals_aug = augmented.get('normals', None)
                if normals_aug is not None:
                    normals = rearrange(normals_aug, 'c z y x -> z y x c')
                normals_mask_aug = augmented.get('normals_mask', None)
                if normals_mask_aug is not None:
                    if normals_mask_aug.ndim == 4 and normals_mask_aug.shape[0] == 1:
                        normals_mask = normals_mask_aug.squeeze(0)
                    elif normals_mask_aug.ndim == 4:
                        # fallback: drop channel dim by taking first channel
                        normals_mask = normals_mask_aug[0]
                    else:
                        normals_mask = normals_mask_aug
            if not torch.isfinite(volume_crop).all() or not torch.isfinite(localiser).all() or not torch.isfinite(uv_heatmaps_both).all():
                # FIXME: why do these NaNs happen occasionally?
                continue

            uv_heatmaps_in = uv_heatmaps_both[..., :heatmap_num_in_channels]
            uv_heatmaps_out = uv_heatmaps_both[..., heatmap_num_in_channels:]

            if multistep_count > 1 and not use_multistep:
                # Allocate blank channels in output heatmaps
                # FIXME: if we supported multi-step loss for non-chain cases, then we could always enable it hence wouldn't need this
                assert uv_heatmaps_out.shape[-1] == 2 * step_count
                other_step_zeros = torch.zeros([*uv_heatmaps_out.shape[:-1], (multistep_count - 1) * step_count], device=uv_heatmaps_out.device, dtype=uv_heatmaps_out.dtype)
                uv_heatmaps_out = torch.cat([
                    uv_heatmaps_out[..., :step_count],
                    other_step_zeros,
                    uv_heatmaps_out[..., step_count:],
                    other_step_zeros,
                ], dim=-1)

            # As an additional augmentation, randomly swap U & V axes in heatmaps
            # We can't do this earlier due to how diagonal points are constructed
            if self._should_swap_uv_axes() and torch.rand([]) < 0.5:
                assert uv_heatmaps_in.shape[-1] == 3 and uv_heatmaps_out.shape[-1] % 2 == 0
                uv_heatmaps_in = uv_heatmaps_in[..., [1, 0, 2]]
                uv_heatmaps_out = torch.cat([uv_heatmaps_out[..., uv_heatmaps_out.shape[-1] // 2:], uv_heatmaps_out[..., :uv_heatmaps_out.shape[-1] // 2]], dim=-1)

            batch_dict = self._build_batch_dict(
                volume_crop=volume_crop,
                localiser=localiser,
                uv_heatmaps_in=uv_heatmaps_in,
                uv_heatmaps_out=uv_heatmaps_out,
                seg=seg,
                seg_mask=seg_mask,
                normals=normals,
                normals_mask=normals_mask,
                center_heatmap=maybe_center_heatmap,
            )

            yield batch_dict


def mark_context_point(volume, point, value=1.):
    center = (point + 0.5).int()
    offsets = torch.tensor([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0], [0, 0, -2], [0, 0, 2]])
    for offset in offsets:
        volume[tuple(center + offset)] = value


def get_zyx_from_patch(ij, patch):
    original_shape = ij.shape
    batch_dims = original_shape[:-1]
    ij_flat = ij.view(-1, 2)
    # Align grid coords to vertex centers: -1 -> [0,0], +1 -> [H-1, W-1]
    denom = torch.as_tensor(patch.zyxs.shape[:2], device=ij_flat.device, dtype=ij_flat.dtype) - 1
    denom = torch.clamp(denom, min=1)  # avoid divide-by-zero on degenerate patches
    normalized_ij = ij_flat / denom * 2 - 1
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


def load_datasets(config, shard_idx=None, total_shards=None):
    train_patches = []
    val_patches = []
    for dataset in config['datasets']:
        volume_path = dataset['volume_path']
        if config.get("use_volume_store_cache", False):
            # TODO: how does this interact with multiple dataloader workers? cache is created before fork, hence presumably not shared?
            if "://" in volume_path or "::" in volume_path:
                store = zarr.storage.FSStore(volume_path, mode='r')
            else:
                store = zarr.storage.DirectoryStore(volume_path)
            cache_gb = float(config.get("cache_max_gb", config.get("volume_cache_gb", 12)))
            store = zarr.storage.LRUStoreCache(store, max_size=int(cache_gb * 1024**3))
            ome_zarr = zarr.open_group(store, mode='r')
        else:
            ome_zarr = zarr.open(volume_path, mode='r')

        volume_scale = dataset['volume_scale']
        volume = ome_zarr[str(volume_scale)]

        patches = load_tifxyz_patches(dataset['segments_path'], dataset.get('z_range', None), volume)
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
        min_segments_for_val = config.get('min_segments_for_val', 5)
        if len(patches) < min_segments_for_val:
            # Not enough segments - use all for training, none for validation
            train_patches.extend(patches)
        else:
            train_patches.extend(patches[num_val_per_volume:])
            val_patches.extend(patches[:num_val_per_volume])
        
    print(f'loaded {len(train_patches)} train patches and {len(val_patches)} val patches')

    if shard_idx is not None and total_shards > 1:
        assert shard_idx < total_shards
        assert len(train_patches) >= total_shards and len(val_patches) >= total_shards
        train_patches = train_patches[shard_idx::total_shards]
        val_patches = val_patches[shard_idx::total_shards]
        print(f'shard #{shard_idx}: retaining {len(train_patches)} train patches and {len(val_patches)} val patches')

    return train_patches, val_patches


def load_tifxyz_patches(segments_path, z_range, volume):

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


def get_crop_from_volume(volume, center_zyx, crop_size, normalize=True):
    """Crop volume around center point, padding with zeros if needed.

    Guarantees the returned crop is exactly `crop_size` on each axis, even if the
    requested center lies far outside the volume (previously this could yield
    huge tensors and blow up concatenation in inference).

    Args:
        volume: The source volume array
        center_zyx: Center coordinates for the crop
        crop_size: Size of the crop (int for cubic, or tuple/list for anisotropic [D, H, W])
        normalize: If True, normalize intensity to [-1, 1] range. If False, return
                   raw float32 values (useful when caller handles normalization).
    """
    # Normalize crop_size to tensor of 3 ints [D, H, W]
    if isinstance(crop_size, (list, tuple)):
        crop_size_tensor = torch.tensor(crop_size)
    else:
        crop_size_tensor = torch.tensor([crop_size, crop_size, crop_size])

    crop_min = (center_zyx - crop_size_tensor // 2).int()
    crop_max = crop_min + crop_size_tensor

    vol_shape = torch.tensor(volume.shape)

    # Clamp requested bounds to the volume and ensure max >= min so slicing is safe
    actual_min = torch.maximum(crop_min, torch.zeros_like(crop_min))
    actual_max = torch.minimum(crop_max, vol_shape)
    actual_max = torch.maximum(actual_max, actual_min)

    # Extract valid portion (may be empty if far outside the volume)
    volume_crop = torch.from_numpy(volume[
        actual_min[0]:actual_max[0],
        actual_min[1]:actual_max[1],
        actual_min[2]:actual_max[2]
    ]).to(torch.float32)

    if normalize:
        if volume_crop.numel() > 0:
            # TODO: should instead always use standardised uint8 volumes!
            if volume.dtype == np.uint8:
                volume_crop = volume_crop / 255.
            else:
                max_val = volume_crop.amax()
                volume_crop = volume_crop / max_val if max_val > 0 else volume_crop
        else:
            volume_crop = torch.zeros((0, 0, 0), dtype=torch.float32)
        volume_crop = volume_crop * 2 - 1
    else:
        if volume_crop.numel() == 0:
            volume_crop = torch.zeros((0, 0, 0), dtype=torch.float32)

    # Compute padding so final shape is exactly crop_size in each dimension.
    pad_before = torch.clamp(actual_min - crop_min, min=0)
    pad_before = torch.minimum(pad_before, crop_size_tensor)

    current_shape = torch.tensor(volume_crop.shape)
    pad_after = crop_size_tensor - current_shape - pad_before
    pad_after = torch.clamp(pad_after, min=0)

    if torch.any(pad_before > 0) or torch.any(pad_after > 0):
        paddings = (
            int(pad_before[2].item()), int(pad_after[2].item()),
            int(pad_before[1].item()), int(pad_after[1].item()),
            int(pad_before[0].item()), int(pad_after[0].item()),
        )
        volume_crop = F.pad(volume_crop, paddings, mode='constant', value=0)

    # Ensure any edge cases still return the desired shape.
    expected_shape = tuple(crop_size_tensor.tolist())
    if volume_crop.shape != expected_shape:
        volume_crop = volume_crop[:expected_shape[0], :expected_shape[1], :expected_shape[2]]

    # min_corner reflects the coordinate of voxel [0,0,0] after padding/truncation
    min_corner = actual_min - pad_before
    return volume_crop, min_corner


def build_localiser(center_zyx, min_corner_zyx, crop_size):
    # Normalize crop_size to tuple of 3 ints [D, H, W]
    if isinstance(crop_size, (list, tuple)):
        crop_size_dhw = tuple(crop_size)
    else:
        crop_size_dhw = (crop_size, crop_size, crop_size)
    localiser = torch.linalg.norm(
        torch.stack(torch.meshgrid(
            torch.arange(crop_size_dhw[0]),
            torch.arange(crop_size_dhw[1]),
            torch.arange(crop_size_dhw[2]),
            indexing='ij'
        ), dim=-1).to(torch.float32) - torch.tensor(crop_size_dhw).float() / 2,
        dim=-1
    )
    localiser = localiser / localiser.amax() * 2 - 1
    mark_context_point(localiser, center_zyx - min_corner_zyx, value=0.)
    return localiser
