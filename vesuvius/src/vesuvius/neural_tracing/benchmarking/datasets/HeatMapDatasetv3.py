import torch
from einops import rearrange
from vesuvius.neural_tracing.dataset import (HeatmapDatasetV2,
                                             kernel,
                                             kernel_size,
                                             get_zyx_from_patch)

# here we're just accumulating the implementations which achieve the speedups we're looking for
# current prog:
# - replaced make_heatmaps with a class method and swapped the fft_conv for a sparse gaussian splat ( mean timings : 18.5 ms -> 0.50 ms )
# - replaced _get_perturbed_zyx_from_patch by caching across perturbations that have the same (patch, center_ij, crop) context and vectorizing over patches (48.07 ms -> 2.11 ms)
# - replaced _get_cached_patch_points with last-call memoization for the same (patch, center_ij, crop) context (43.64 ms -> 0.01 ms)
# - replaced _get_patch_points_in_crop with bbox rejection + precomputed sampling grids (0.87 ms -> 0.60 ms)
# - replaced _get_quads_in_crop with bbox rejection (0.84 ms -> 0.65 ms)


# current timings for full sample iter for 50 samples
# === HeatmapDatasetV2 ===
# Iterator timings: mean 276.09 ms | median 272.13 ms | min 59.66 ms | max 760.15 ms
#
# === HeatmapDatasetV3 ===
# Iterator timings: mean 105.19 ms | median 86.37 ms | min 18.49 ms | max 520.93 ms // OLD
# Iterator timings: mean 98.32 ms | median 79.59 ms | min 18.94 ms | max 441.92 ms // OLD
# Iterator timings: mean 91.23 ms | median 70.69 ms | min 16.80 ms | max 573.13 ms // OLD
# Iterator timings: mean 91.04 ms | median 75.23 ms | min 18.92 ms | max 555.06 ms // CURRENT


def _scatter_heatmaps(all_zyxs, min_corner_zyx, crop_size):
    crop_size_int = int(crop_size)
    dtype = all_zyxs[0].dtype
    device = all_zyxs[0].device
    min_corner = min_corner_zyx.to(device=device, dtype=dtype)
    heatmaps = torch.zeros((crop_size_int, crop_size_int, crop_size_int, all_zyxs[0].shape[0]), device=device, dtype=dtype)

    def scatter(zyxs):
        coords = torch.cat([
            (zyxs.to(device=device, dtype=dtype) - min_corner + 0.5).int(),
            torch.arange(zyxs.shape[0], device=device)[:, None]
        ], dim=1)
        coords = coords[(coords[..., :3] >= 0).all(dim=1) & (coords[..., :3] < crop_size_int).all(dim=1)]
        if len(coords) > 0:
            heatmaps[*coords.T] = 1.

    for zyxs in all_zyxs:
        scatter(zyxs)

    return heatmaps

class HeatmapDatasetV3(HeatmapDatasetV2):
    def __init__(self, config, patches_for_split):
        super().__init__(config, patches_for_split)
        self._perturb_cache_key = None
        self._perturb_cache_value = None
        self._sampling = {}
        self._quad_bboxes = {}
        for patch in self._patches:
            # Precompute quad corners and u/v grids for sampling
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
            }

            # Precompute quad-center AABB for quick rejection
            valid_centers = patch.quad_centers[patch.valid_quad_mask]
            if len(valid_centers) == 0:
                bbox_min = torch.zeros(3, dtype=patch.quad_centers.dtype)
                bbox_max = bbox_min
            else:
                bbox_min = valid_centers.min(dim=0).values
                bbox_max = valid_centers.max(dim=0).values
            self._quad_bboxes[id(patch)] = (bbox_min, bbox_max)

    @classmethod
    def make_heatmaps(cls, all_zyxs, min_corner_zyx, crop_size):
        crop_size_int = int(crop_size)
        radius = kernel_size // 2
        dtype = all_zyxs[0].dtype
        device = all_zyxs[0].device
        min_corner = min_corner_zyx.to(device=device, dtype=dtype)

        heatmaps = torch.zeros((crop_size_int, crop_size_int, crop_size_int, all_zyxs[0].shape[0]), device=device,
                               dtype=dtype)
        kernel_t = kernel.to(device=device, dtype=dtype)

        for zyxs in all_zyxs:
            coords = (zyxs.to(device=device, dtype=dtype) - min_corner + 0.5).int()
            valid_mask = (coords >= 0).all(dim=1) & (coords < crop_size_int).all(dim=1)
            if not torch.any(valid_mask):
                continue
            coords = coords[valid_mask]
            channel_indices = torch.arange(zyxs.shape[0], device=device)[valid_mask]

            for coord, channel_idx in zip(coords, channel_indices):
                z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
                z0 = max(0, z - radius)
                y0 = max(0, y - radius)
                x0 = max(0, x - radius)
                z1 = min(crop_size_int, z + radius + 1)
                y1 = min(crop_size_int, y + radius + 1)
                x1 = min(crop_size_int, x + radius + 1)

                kz0 = z0 - (z - radius)
                ky0 = y0 - (y - radius)
                kx0 = x0 - (x - radius)
                kz1 = kz0 + (z1 - z0)
                ky1 = ky0 + (y1 - y0)
                kx1 = kx0 + (x1 - x0)

                heatmaps[z0:z1, y0:y1, x0:x1, int(channel_idx)] += kernel_t[kz0:kz1, ky0:ky1, kx0:kx1]

        # Match HeatmapDatasetV2 return layout: (channels, z, y, x)
        return rearrange(heatmaps, 'z y x c -> c z y x')

    # Last-call memoized cache for _get_cached_patch_points (best-performing alt)
    def _make_cache_key(self, patch, center_ij, min_corner_zyx, crop_size):
        return (
            id(patch),
            tuple(center_ij.tolist()),
            tuple(min_corner_zyx.tolist()),
            int(crop_size),
        )

    def _get_cached_patch_points(self, patch, center_ij, min_corner_zyx, crop_size):
        key = self._make_cache_key(patch, center_ij, min_corner_zyx, crop_size)
        if key != self._perturb_cache_key:
            self._perturb_cache_key = key
            self._perturb_cache_value = super()._get_cached_patch_points(patch, center_ij, min_corner_zyx, crop_size)
        return self._perturb_cache_value

    def _get_distance_to_nearest_patch_cached(self, point_zyx, cached_patch_points):
        if not cached_patch_points:
            return float('inf')
        all_points = torch.cat(cached_patch_points, dim=0)
        if all_points.numel() == 0:
            return float('inf')
        return torch.norm(all_points - point_zyx, dim=-1).min().item()

    def _get_quads_in_crop(self, patch, min_corner_zyx, crop_size):
        bbox_min, bbox_max = self._quad_bboxes[id(patch)]
        crop_min = min_corner_zyx.to(dtype=bbox_min.dtype)
        crop_size_tensor = torch.as_tensor(crop_size, dtype=bbox_min.dtype, device=crop_min.device)
        crop_max = crop_min + crop_size_tensor
        if (bbox_max < crop_min).any() or (bbox_min >= crop_max).any():
            return torch.zeros_like(patch.valid_quad_mask)

        return patch.valid_quad_mask & torch.all(patch.quad_centers >= crop_min, dim=-1) & torch.all(patch.quad_centers < crop_max, dim=-1)

    def _get_patch_points_in_crop(self, patch, min_corner_zyx, crop_size):
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
        filtered_quads_zyxs = info["quad_corners"][quad_in_crop]

        points_covering_quads = torch.lerp(
            filtered_quads_zyxs[:, None, 0, :],
            filtered_quads_zyxs[:, None, 1, :],
            info["v_points"][None, :, None, None],
        )
        points_covering_quads = torch.lerp(
            points_covering_quads[:, :, None, 0],
            points_covering_quads[:, :, None, 1],
            info["u_points"][None, None, :, None],
        )

        return points_covering_quads.view(-1, 3)

    def _get_perturbed_zyx_from_patch(self, point_ij, patch, center_ij, min_corner_zyx, crop_size, is_center_point=False):
        if is_center_point:
            perturbed_ij = point_ij
            perturbed_zyx = get_zyx_from_patch(point_ij, patch)
        else:
            offset_magnitude = torch.rand([]) * self._uv_max_perturbation
            offset_angle = torch.rand([]) * 2 * torch.pi
            offset_uv_voxels = offset_magnitude * torch.tensor([torch.cos(offset_angle), torch.sin(offset_angle)])

            offset_2d = offset_uv_voxels * patch.scale
            perturbed_ij = point_ij + offset_2d

            perturbed_ij = torch.clamp(perturbed_ij, torch.zeros([]), torch.tensor(patch.zyxs.shape[:2]) - 1.001)

            if not patch.valid_quad_mask[*perturbed_ij.int()]:
                return get_zyx_from_patch(point_ij, patch)

            perturbed_zyx = get_zyx_from_patch(perturbed_ij, patch)

        i, j = perturbed_ij.int()
        h, w = patch.zyxs.shape[:2]

        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and patch.valid_vertex_mask[ni, nj]:
                    neighbors.append(patch.zyxs[ni, nj])

        if len(neighbors) < 3:
            final_zyx = perturbed_zyx
        else:
            normal = torch.linalg.cross(neighbors[1] - neighbors[0], neighbors[2] - neighbors[0], dim=-1)
            normal_norm = torch.norm(normal)
            if normal_norm > 1e-6:
                normal = normal / normal_norm
                normal_offset_magnitude = (torch.rand([]) * 2 - 1) * self._w_max_perturbation

                cached_patch_points = self._get_cached_patch_points(patch, center_ij, min_corner_zyx, crop_size)

                while abs(normal_offset_magnitude) >= 1.0:
                    nearest_patch_distance = self._get_distance_to_nearest_patch_cached(perturbed_zyx, cached_patch_points)
                    if abs(normal_offset_magnitude) <= nearest_patch_distance * self._main_component_distance_factor:
                        break
                    normal_offset_magnitude *= 0.8
                else:
                    normal_offset_magnitude = 0.

                final_zyx = perturbed_zyx + normal_offset_magnitude * normal
            else:
                final_zyx = perturbed_zyx

        return final_zyx
