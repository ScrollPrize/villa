import torch
import numpy as np
import scipy.ndimage

import vesuvius.neural_tracing.dataset as base_dataset
from vesuvius.neural_tracing.dataset import HeatmapDatasetV2, get_zyx_from_patch


def _bilinear_lookup(ij, patch):
    """
    Fast bilinear lookup of 3D coords from patch.zyxs without grid_sample.
    """
    orig_shape = ij.shape
    ij_flat = ij.view(-1, 2).to(dtype=torch.float32)

    h, w, _ = patch.zyxs.shape
    i = ij_flat[:, 0].clamp(0, h - 1)
    j = ij_flat[:, 1].clamp(0, w - 1)

    i0 = torch.floor(i)
    j0 = torch.floor(j)
    i1 = (i0 + 1).clamp(max=h - 1)
    j1 = (j0 + 1).clamp(max=w - 1)

    di = i - i0
    dj = j - j0

    i0 = i0.to(dtype=torch.int64)
    i1 = i1.to(dtype=torch.int64)
    j0 = j0.to(dtype=torch.int64)
    j1 = j1.to(dtype=torch.int64)

    zyxs = patch.zyxs.to(dtype=torch.float32)
    tl = zyxs[i0, j0]
    tr = zyxs[i0, j1]
    bl = zyxs[i1, j0]
    br = zyxs[i1, j1]

    top = tl * (1 - dj)[:, None] + tr * dj[:, None]
    bottom = bl * (1 - dj)[:, None] + br * dj[:, None]
    interp = top * (1 - di)[:, None] + bottom * di[:, None]
    return interp.view(*orig_shape[:-1], 3)


class HeatmapDatasetV2BilinearLookup(HeatmapDatasetV2):
    """
    Variant that swaps grid_sample for a manual bilinear lookup to reduce per-call overhead.
    Uses a scoped monkey-patch of get_zyx_from_patch while iterating.
    """

    def __iter__(self):
        orig = base_dataset.get_zyx_from_patch
        base_dataset.get_zyx_from_patch = _bilinear_lookup
        try:
            for sample in super().__iter__():
                yield sample
        finally:
            base_dataset.get_zyx_from_patch = orig


class HeatmapDatasetV2CoarseDistance(HeatmapDatasetV2):
    """
    Variant that uses quad centers instead of densely sampled quad points when
    computing cached patch points for nearest-distance checks. This trades a
    coarser distance estimate for lower memory/compute.
    """

    def _compute_cached_patch_points(self, current_patch, center_ij, min_corner_zyx, crop_size):
        quad_main_component = self._get_current_patch_center_component_mask(
            current_patch, center_ij, min_corner_zyx, crop_size
        )

        all_patch_points = []
        crop_min = min_corner_zyx.to(dtype=current_patch.zyxs.dtype, device=current_patch.zyxs.device)
        crop_max = crop_min + torch.as_tensor(crop_size, dtype=crop_min.dtype, device=crop_min.device)

        volume_key = id(current_patch.volume)
        for other_patch, bbox_min, bbox_max in self._volume_patch_bboxes.get(volume_key, []):
            if other_patch is current_patch:
                continue
            if (bbox_max < crop_min).any() or (bbox_min >= crop_max).any():
                continue
            quad_in_crop = self._get_quads_in_crop(other_patch, min_corner_zyx, crop_size)
            if torch.any(quad_in_crop):
                centers = other_patch.quad_centers[quad_in_crop]
                if centers.numel() > 0:
                    all_patch_points.append(centers)

        quad_in_crop = self._get_quads_in_crop(current_patch, min_corner_zyx, crop_size)
        quad_excluding_main = quad_in_crop & ~quad_main_component
        centers = current_patch.quad_centers[quad_excluding_main]
        if centers.numel() > 0:
            all_patch_points.append(centers)

        return all_patch_points


class HeatmapDatasetV2PrecomputedNormals(HeatmapDatasetV2):
    """
    Compatibility alias: HeatmapDatasetV2 now precomputes normals by default.
    """

    pass


class HeatmapDatasetV2TorchComponent(HeatmapDatasetV2):
    """
    Variant of HeatmapDatasetV2 that replaces scipy.ndimage.label with a torch flood-fill
    for the center component mask to avoid host round-trips.
    """

    def _get_current_patch_center_component_mask(self, current_patch, center_ij, min_corner_zyx, crop_size):
        quad_in_crop = self._get_quads_in_crop(current_patch, min_corner_zyx, crop_size)
        if not torch.any(quad_in_crop):
            return torch.zeros_like(quad_in_crop)

        center_quad = center_ij.int()
        if (
            center_quad[0] < 0
            or center_quad[0] >= quad_in_crop.shape[0]
            or center_quad[1] < 0
            or center_quad[1] >= quad_in_crop.shape[1]
            or not quad_in_crop[center_quad[0], center_quad[1]]
        ):
            return torch.zeros_like(quad_in_crop)

        component_mask = torch.zeros_like(quad_in_crop)
        frontier = torch.zeros_like(quad_in_crop)
        frontier[center_quad[0], center_quad[1]] = True

        while frontier.any():
            # 4-neighbour expansion
            neighbors = torch.zeros_like(frontier)
            neighbors[:-1] |= frontier[1:]
            neighbors[1:] |= frontier[:-1]
            neighbors[:, :-1] |= frontier[:, 1:]
            neighbors[:, 1:] |= frontier[:, :-1]

            component_mask |= frontier
            frontier = neighbors & quad_in_crop & ~component_mask

        return component_mask


class HeatmapDatasetV2BBoxGuard(HeatmapDatasetV2):
    """
    Variant that adds a coarse bbox distance guard before running full nearest-point scans.
    """

    def _compute_cached_patch_points(self, current_patch, center_ij, min_corner_zyx, crop_size):
        quad_main_component = self._get_current_patch_center_component_mask(
            current_patch, center_ij, min_corner_zyx, crop_size
        )

        all_patch_points = []
        coarse_bboxes = []
        crop_min = min_corner_zyx.to(dtype=current_patch.zyxs.dtype, device=current_patch.zyxs.device)
        crop_max = crop_min + torch.as_tensor(crop_size, dtype=crop_min.dtype, device=crop_min.device)

        volume_key = id(current_patch.volume)
        for other_patch, bbox_min, bbox_max in self._volume_patch_bboxes.get(volume_key, []):
            if other_patch is current_patch:
                continue
            if (bbox_max < crop_min).any() or (bbox_min >= crop_max).any():
                continue
            patch_points = self._get_patch_points_in_crop(other_patch, min_corner_zyx, crop_size)
            if len(patch_points) > 0:
                all_patch_points.append(patch_points)
                coarse_bboxes.append((bbox_min, bbox_max))

        quad_in_crop = self._get_quads_in_crop(current_patch, min_corner_zyx, crop_size)
        quad_excluding_main = quad_in_crop & ~quad_main_component
        other_patch_points = self._sample_points_from_quads(current_patch, quad_excluding_main)
        if len(other_patch_points) > 0:
            all_patch_points.append(other_patch_points)
            coarse_bboxes.append(self._quad_bboxes[id(current_patch)])

        return all_patch_points, coarse_bboxes

    def _get_cached_patch_points(self, current_patch, center_ij, min_corner_zyx, crop_size):
        key = self._make_cache_key(current_patch, center_ij, min_corner_zyx, crop_size)
        if key != self._perturb_cache_key:
            self._perturb_cache_key = key
            raw_points, coarse_bboxes = self._compute_cached_patch_points(
                current_patch, center_ij, min_corner_zyx, crop_size
            )
            concat = None
            if raw_points and any(p.numel() > 0 for p in raw_points):
                concat = torch.cat([p for p in raw_points if p.numel() > 0], dim=0)
            self._perturb_cache_value = {"concat": concat, "bboxes": coarse_bboxes}
        return self._perturb_cache_value

    @staticmethod
    def _min_distance_to_bbox(point_zyx, bbox_min, bbox_max):
        clamped = torch.maximum(torch.minimum(point_zyx, bbox_max), bbox_min)
        return torch.norm(point_zyx - clamped)

    def _get_min_bbox_distance(self, point_zyx, cached_patch_points):
        if not cached_patch_points:
            return float("inf")
        bboxes = cached_patch_points.get("bboxes", None) if isinstance(cached_patch_points, dict) else None
        if not bboxes:
            return float("inf")
        point_device = point_zyx.device
        min_dist = float("inf")
        for bbox_min, bbox_max in bboxes:
            dist = self._min_distance_to_bbox(
                point_zyx, bbox_min.to(device=point_device), bbox_max.to(device=point_device)
            )
            if dist < min_dist:
                min_dist = dist
        return min_dist

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
        normal = self._vertex_normals[id(patch)][i, j]
        normal_norm = torch.norm(normal)
        if normal_norm > 1e-6:
            normal = normal / normal_norm
            normal_offset_magnitude = (torch.rand([]) * 2 - 1) * self._w_max_perturbation
            cached_patch_points = self._get_cached_patch_points(patch, center_ij, min_corner_zyx, crop_size)
            coarse_min_distance = self._get_min_bbox_distance(perturbed_zyx, cached_patch_points)
            threshold = self._main_component_distance_factor

            if abs(normal_offset_magnitude) <= coarse_min_distance * threshold:
                nearest_patch_distance = coarse_min_distance
            else:
                while abs(normal_offset_magnitude) >= 1.0:
                    nearest_patch_distance = self._get_distance_to_nearest_patch_cached(
                        perturbed_zyx, cached_patch_points
                    )
                    if abs(normal_offset_magnitude) <= nearest_patch_distance * threshold:
                        break
                    normal_offset_magnitude *= 0.8
                else:
                    normal_offset_magnitude = 0.0

            final_zyx = perturbed_zyx + normal_offset_magnitude * normal
        else:
            final_zyx = perturbed_zyx

        return final_zyx


class HeatmapDatasetV2LRUStoreCache(HeatmapDatasetV2):
    """
    Variant that reopens the zarr volumes behind each patch with an LRUStoreCache
    to cache hot chunks in memory. Useful for benchmarking disk I/O bottlenecks
    without affecting the default dataset behaviour.
    """

    def __init__(self, config, patches):
        config = dict(config)
        cache_gb = config.get("cache_max_gb", config.get("volume_cache_gb", None))
        if cache_gb is None:
            config["cache_max_gb"] = 4
        super().__init__(config, patches)


__all__ = [
    "HeatmapDatasetV2BilinearLookup",
    "HeatmapDatasetV2CoarseDistance",
    "HeatmapDatasetV2PrecomputedNormals",
    "HeatmapDatasetV2TorchComponent",
    "HeatmapDatasetV2BBoxGuard",
    "HeatmapDatasetV2LRUStoreCache",
    "HeatmapDatasetV2DistanceTransform",
]


class HeatmapDatasetV2DistanceTransform(HeatmapDatasetV2):
    """
    Variant that builds a per-crop binary occupancy grid and runs a single 3D
    distance transform; each distance query becomes an O(1) lookup.
    """

    def _compute_cached_patch_points(self, current_patch, center_ij, min_corner_zyx, crop_size):
        quad_main_component = self._get_current_patch_center_component_mask(
            current_patch, center_ij, min_corner_zyx, crop_size
        )

        crop_size_int = int(crop_size)
        occupancy = np.zeros((crop_size_int, crop_size_int, crop_size_int), dtype=np.bool_)

        def _points_to_indices(points):
            if points.numel() == 0:
                return None
            coords = torch.round(points - min_corner_zyx).to(dtype=torch.int64)
            in_bounds = (coords >= 0).all(dim=1) & (coords < crop_size_int).all(dim=1)
            if not torch.any(in_bounds):
                return None
            coords = coords[in_bounds]
            return coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), coords[:, 2].cpu().numpy()

        crop_min = min_corner_zyx.to(dtype=current_patch.zyxs.dtype, device=current_patch.zyxs.device)
        crop_max = crop_min + torch.as_tensor(crop_size, dtype=crop_min.dtype, device=crop_min.device)

        volume_key = id(current_patch.volume)
        for other_patch, bbox_min, bbox_max in self._volume_patch_bboxes.get(volume_key, []):
            if other_patch is current_patch:
                continue
            if (bbox_max < crop_min).any() or (bbox_min >= crop_max).any():
                continue
            patch_points = self._get_patch_points_in_crop(other_patch, min_corner_zyx, crop_size)
            idx = _points_to_indices(patch_points)
            if idx is not None:
                occupancy[idx] = True

        quad_in_crop = self._get_quads_in_crop(current_patch, min_corner_zyx, crop_size)
        quad_excluding_main = quad_in_crop & ~quad_main_component
        other_patch_points = self._sample_points_from_quads(current_patch, quad_excluding_main)
        idx = _points_to_indices(other_patch_points)
        if idx is not None:
            occupancy[idx] = True

        # Distance in voxels to nearest occupied voxel
        if occupancy.any():
            distance_field = scipy.ndimage.distance_transform_edt(~occupancy).astype(np.float32)
        else:
            distance_field = np.full_like(occupancy, np.inf, dtype=np.float32)
        return distance_field

    def _get_cached_patch_points(self, current_patch, center_ij, min_corner_zyx, crop_size):
        key = self._make_cache_key(current_patch, center_ij, min_corner_zyx, crop_size)
        if key != self._perturb_cache_key:
            self._perturb_cache_key = key
            distance_field = self._compute_cached_patch_points(current_patch, center_ij, min_corner_zyx, crop_size)
            self._perturb_cache_value = {
                "distance_field": torch.from_numpy(distance_field),
                "min_corner": min_corner_zyx.to(dtype=torch.float32).clone(),
            }
        return self._perturb_cache_value

    def _get_distance_to_nearest_patch_cached(self, point_zyx, cached_patch_points):
        if not cached_patch_points:
            return float("inf")
        distance_field = cached_patch_points.get("distance_field", None)
        if distance_field is None:
            return float("inf")

        min_corner = cached_patch_points["min_corner"]
        rel = torch.round(point_zyx - min_corner).to(dtype=torch.int64)
        crop_size = distance_field.shape[0]
        if ((rel < 0) | (rel >= crop_size)).any():
            return float("inf")

        z, y, x = rel.tolist()
        return distance_field[z, y, x].item()
