import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from tifxyz import Patch


_LINK_THREADS_DEFAULT_CAP = 16


@dataclass(frozen=True)
class PointPatchLink:
    """Link between a point collection entry and a patch."""
    point_id: int
    collection_id: int
    collection_name: str
    point_zyx: List[float]
    ij_coords: List[float]
    distance: float
    winding_annotation: float


def load_point_collection(filename: str) -> Optional[Dict[int, Dict[str, Any]]]:
    """Load point collection from JSON file and return as dictionary.

    `kind` must be one of PCL_KINDS and is stamped onto each loaded pcl; it
    determines whether the pcl partakes in patch attachment (cross_patch) or is
    consumed as a free-floating ordered strip (unattached).
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        # Check version
        if not data.get("vc_pointcollections_json_version") == "1":
            print(f"Error: Unsupported JSON version in {filename}")
            return None

        collections = {}

        # Load collections
        collections_data = data.get("collections", {})
        for id_str, collection_data in collections_data.items():
            collection_id = int(id_str)
            collection = {
                "id": collection_id,
                "name": collection_data["name"],
                "points": {},
                "metadata": collection_data.get("metadata", {}),
                "color": collection_data.get("color", [0.0, 0.0, 0.0])
            }

            # Load points
            points_data = collection_data.get("points", {})
            for point_id_str, point_data in points_data.items():
                point_id = int(point_id_str)
                point = {
                    "id": point_id,
                    "collectionId": collection_id,
                    "p": point_data["p"],
                    "winding_annotation": point_data.get("wind_a") if point_data.get("wind_a") is not None else float('nan'),
                    "creation_time": point_data.get("creation_time", 0)
                }
                collection["points"][point_id] = point

            collections[collection_id] = collection

        total_points = sum(len(col["points"]) for col in collections.values())
        print(f"Loaded point collection with {len(collections)} collections ({total_points} points)")
        return collections

    except Exception as e:
        print(f"Error loading point collection from {filename}: {e}")
        return None


def _load_surface_index_backend():
    try:
        from vc import surface_index
        return surface_index
    except ImportError:
        pass

    try:
        import vc_surface_index
        return vc_surface_index
    except ImportError:
        return None


def can_use_surface_index_backend(patches: Dict[str, Patch]) -> bool:
    return _load_surface_index_backend() is not None


def _build_surface_patch_index(
    patches: Dict[str, Patch],
    tolerance: float,
):
    surface_index = _load_surface_index_backend()
    if surface_index is None:
        return None

    surface_ids = []
    surface_defs = []
    for patch_id, patch in patches.items():
        zyx = patch.zyxs.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        scale = patch.scale.detach().cpu().numpy() if hasattr(patch.scale, 'detach') else patch.scale
        surface_ids.append(patch_id)
        surface_defs.append(surface_index.QuadSurface(patch_id, zyx, float(scale[0]), float(scale[1])))

    if not surface_defs:
        return None

    index = surface_index.SurfacePatchIndex()
    index.rebuild(surface_defs, bbox_padding=tolerance, sampling_stride=1)
    return index, surface_ids


def _record_point_patch_link(
    links: Dict[str, List[PointPatchLink]],
    collection: Dict[str, Any],
    collection_id: int,
    point_id: int,
    point: Dict[str, Any],
    patch_id: str,
    distance: float,
    ij_coords: List[float],
) -> None:
    point_zyx = np.asarray(point.get('zyx', point['p'][::-1]), dtype=np.float32).tolist()
    point['on_patch'] = {'id': patch_id, 'distance': float(distance), 'ij': list(ij_coords)}
    links.setdefault(patch_id, []).append(
        PointPatchLink(
            point_id=point_id,
            collection_id=collection_id,
            collection_name=collection['name'],
            point_zyx=point_zyx,
            ij_coords=list(ij_coords),
            distance=float(distance),
            winding_annotation=float(point['winding_annotation']),
        )
    )


def _link_thread_count(num_items: int) -> int:
    if num_items <= 1:
        return 1
    override = os.environ.get('FIT_SPIRAL_LINK_THREADS')
    if override:
        try:
            requested = int(override)
        except ValueError:
            requested = 0
        if requested > 0:
            return max(1, min(requested, num_items))
    return max(1, min(num_items, os.cpu_count() or 1, _LINK_THREADS_DEFAULT_CAP))


def _best_hit_per_point(offsets, surf_idx, distances, ijs, *, target_idxs=None, patch_areas=None):
    if len(surf_idx) == 0:
        return None

    point_idx = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    keep = surf_idx >= 0
    if target_idxs is not None:
        keep &= np.isin(surf_idx, target_idxs)
    if not keep.any():
        return None

    points = point_idx[keep]
    surfaces = surf_idx[keep]
    dists = distances[keep]
    hit_ijs = ijs[keep]

    if patch_areas is None:
        # Primary sort: point. Within each point, nearest distance wins; surface
        # index is a deterministic tie-break.
        order = np.lexsort((surfaces, dists, points))
    else:
        areas = patch_areas[surfaces]
        # Local fit_spiral behavior: for general PCLs, prefer largest patch area,
        # then nearest distance. Surface index only breaks exact ties.
        order = np.lexsort((surfaces, dists, -areas, points))

    sorted_points = points[order]
    _, first = np.unique(sorted_points, return_index=True)
    winners = order[first]
    return points[winners], surfaces[winners], dists[winners], hit_ijs[winners]


def _run_collection_link_workers(work_items, worker_fn, desc):
    merged: Dict[str, List[PointPatchLink]] = {}

    def merge(local):
        for patch_id, patch_links in local.items():
            if patch_links:
                merged.setdefault(patch_id, []).extend(patch_links)

    max_workers = _link_thread_count(len(work_items))
    if max_workers <= 1:
        for item in tqdm(work_items, desc):
            merge(worker_fn(item))
        return merged

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for local in tqdm(executor.map(worker_fn, work_items), desc, total=len(work_items)):
            merge(local)
    return merged


def _link_points_to_patches_with_surface_index(
    patches: Dict[str, Patch],
    point_collections: Dict[int, Dict[str, Any]],
    tolerance: float,
    distance_scale: float,
    built_index=None,
    general_hit_policy: str = 'nearest',
) -> Optional[Dict[str, List[PointPatchLink]]]:
    if built_index is None:
        built_index = _build_surface_patch_index(patches, tolerance)
    if built_index is None:
        return None
    index, surface_ids = built_index

    links: Dict[str, List[PointPatchLink]] = {}
    if not point_collections:
        return links

    print(f'linking points to patches with vc.surface_index ({len(surface_ids)} patches)')

    if not hasattr(index, 'locate_all_xyz_batch'):
        for collection_id, collection in tqdm(point_collections.items(), 'linking points to patches'):
            for point_id, point in collection['points'].items():
                point_zyx = np.asarray(point.get('zyx', point['p'][::-1]), dtype=np.float32)
                hit = index.locate_xyz(point_zyx[::-1].copy(), tolerance)
                if hit is None:
                    continue
                _record_point_patch_link(
                    links,
                    collection,
                    collection_id,
                    point_id,
                    point,
                    hit['id'],
                    float(hit['distance']) / distance_scale,
                    hit['ij'],
                )
        return links

    patch_areas = None
    if general_hit_policy == 'largest_area':
        patch_areas = np.asarray([float(patches[patch_id].area) for patch_id in surface_ids], dtype=np.float64)

    def worker(item):
        collection_id, collection = item
        point_items = list(collection['points'].items())
        if not point_items:
            return {}

        point_xyzs = np.ascontiguousarray(
            np.stack([
                np.asarray(point.get('zyx', point['p'][::-1]), dtype=np.float32)[::-1]
                for _, point in point_items
            ], axis=0),
            dtype=np.float32,
        )
        offsets, surf_idx, distances, ijs = index.locate_all_xyz_batch(point_xyzs, tolerance)
        best = _best_hit_per_point(offsets, surf_idx, distances, ijs, patch_areas=patch_areas)
        if best is None:
            return {}

        local: Dict[str, List[PointPatchLink]] = {}
        win_points, win_surfaces, win_distances, win_ijs = best
        for k in range(len(win_points)):
            point_id, point = point_items[int(win_points[k])]
            _record_point_patch_link(
                local,
                collection,
                collection_id,
                point_id,
                point,
                surface_ids[int(win_surfaces[k])],
                float(win_distances[k]) / distance_scale,
                [float(win_ijs[k][0]), float(win_ijs[k][1])],
            )
        return local

    index_links = _run_collection_link_workers(
        list(point_collections.items()), worker, 'linking points to patches'
    )
    for patch_id, patch_links in index_links.items():
        links.setdefault(patch_id, []).extend(patch_links)

    return links


def _link_between_patch_collections_with_surface_index(
    links: Dict[str, List[PointPatchLink]],
    patches: Dict[str, Patch],
    between_collections,
    tolerance: float,
    distance_scale: float,
    built_index=None,
) -> bool:
    if built_index is None:
        built_index = _build_surface_patch_index(patches, tolerance)
    if built_index is None:
        return False
    index, surface_ids = built_index
    if not hasattr(index, 'locate_all_xyz_batch'):
        return False

    surface_id_to_idx = {surface_id: idx for idx, surface_id in enumerate(surface_ids)}
    use_subset_query = hasattr(index, 'locate_all_xyz_batch_in')
    print(
        f'linking between-patch pcls with vc.surface_index ({len(surface_ids)} patches'
        f'{", subset query" if use_subset_query else ""})'
    )

    def worker(item):
        collection_id, collection, targets = item
        target_idxs = np.fromiter(
            (surface_id_to_idx[patch_id] for patch_id in targets if patch_id in surface_id_to_idx),
            dtype=np.int64,
        )
        if len(target_idxs) == 0:
            return {}

        point_items = list(collection['points'].items())
        if not point_items:
            return {}

        point_xyzs = np.ascontiguousarray(
            np.stack([
                np.asarray(point.get('zyx', point['p'][::-1]), dtype=np.float32)[::-1]
                for _, point in point_items
            ], axis=0),
            dtype=np.float32,
        )
        if use_subset_query:
            offsets, surf_idx, distances, ijs = index.locate_all_xyz_batch_in(
                point_xyzs, target_idxs.astype(np.int32), tolerance
            )
            best = _best_hit_per_point(offsets, surf_idx, distances, ijs)
        else:
            offsets, surf_idx, distances, ijs = index.locate_all_xyz_batch(point_xyzs, tolerance)
            best = _best_hit_per_point(offsets, surf_idx, distances, ijs, target_idxs=target_idxs)
        if best is None:
            return {}

        local: Dict[str, List[PointPatchLink]] = {}
        win_points, win_surfaces, win_distances, win_ijs = best
        for k in range(len(win_points)):
            point_id, point = point_items[int(win_points[k])]
            _record_point_patch_link(
                local,
                collection,
                collection_id,
                point_id,
                point,
                surface_ids[int(win_surfaces[k])],
                float(win_distances[k]) / distance_scale,
                [float(win_ijs[k][0]), float(win_ijs[k][1])],
            )
        return local

    between_links = _run_collection_link_workers(
        between_collections, worker, 'linking between-patch pcls'
    )
    for patch_id, patch_links in between_links.items():
        links.setdefault(patch_id, []).extend(patch_links)
    return True


_BETWEEN_PATCHES_PREFIX = 'between_patches__'


def _resolve_between_patches_targets(
    collection: Dict[str, Any],
    patches: Dict[str, Patch],
) -> Optional[Dict[str, Patch]]:
    """Resolve a "between_patches__XXX__YYY" collection to its named patch pair.

    These collections are written by connect_overlapping_patches.py to connect
    a specific pair of patches; their points should attach only to that pair.
    Returns ``{XXX: patch, YYY: patch}`` when the name encodes a pair and both
    patches are present, otherwise ``None`` (so the collection falls back to the
    general nearest-patch search). Patch ids may themselves contain the ``__``
    separator, so the split is disambiguated by requiring both halves to name an
    existing patch.
    """
    name = collection.get('name', '') or ''
    if not name.startswith(_BETWEEN_PATCHES_PREFIX):
        return None
    rest = name[len(_BETWEEN_PATCHES_PREFIX):]
    idx = rest.find('__')
    while idx != -1:
        a, b = rest[:idx], rest[idx + 2:]
        if a in patches and b in patches:
            return {a: patches[a], b: patches[b]}
        idx = rest.find('__', idx + 1)
    return None


def _link_collection_to_patch_subset(
    links: Dict[str, List[PointPatchLink]],
    collection_id: int,
    collection: Dict[str, Any],
    candidate_patches: Dict[str, Patch],
    tolerance: float,
    hit_policy: str = 'nearest',
) -> None:
    """Attach each point of one collection to the nearest of ``candidate_patches``.

    Brute-force projection (``Patch.project``) restricted to the given patch
    subset; the nearest patch within ``tolerance`` wins. Shared by the general
    fallback (all patches) and the "between_patches" special case (the named
    pair only).
    """
    device = torch.device('cpu')
    tolerance_t = torch.tensor(tolerance, device=device, dtype=torch.float32)

    for point_id, point in collection['points'].items():
        point_zyx = torch.as_tensor(point.get('zyx', point['p'][::-1]), dtype=torch.float32, device=device)

        nearest_patch_id = None
        nearest_distance = torch.tensor(float('inf'), device=device)
        nearest_ij = None
        best_area = -1.0

        for patch_id, patch in candidate_patches.items():
            ij_coord, distance = patch.project(point_zyx)
            if distance > tolerance_t:
                continue
            if hit_policy == 'largest_area':
                area = float(patch.area)
                is_better = area > best_area or (area == best_area and distance < nearest_distance)
            else:
                is_better = distance < nearest_distance
            if is_better:
                nearest_distance = distance
                nearest_patch_id = patch_id
                nearest_ij = ij_coord
                best_area = float(patch.area)

        if nearest_patch_id:
            _record_point_patch_link(
                links,
                collection,
                collection_id,
                point_id,
                point,
                nearest_patch_id,
                float(nearest_distance.cpu().item()),
                nearest_ij.tolist(),
            )


def link_points_to_patches(
    patches: Dict[str, Patch],
    point_collections: Dict[int, Dict[str, Any]],
    tolerance: float = 10.0,
    surface_index_tolerance: Optional[float] = None,
    distance_scale: float = 1.0,
    general_hit_policy: str = 'nearest',
) -> Dict[str, List[PointPatchLink]]:
    """Process point collections and link them to patches.

    Special case: a collection named "between_patches__XXX__YYY" (written by
    connect_overlapping_patches.py) attaches its points only to patches XXX and
    YYY, when both exist. Such collections are handled here by projecting onto
    just that pair; all other collections go through the general nearest-patch
    search (surface index when available, else brute force).

    ``general_hit_policy='largest_area'`` preserves fit_spiral's historical
    behavior for general PCLs: choose the largest-area hit within tolerance,
    then nearest distance. Between-patch PCLs always choose nearest within
    their named pair.
    """
    links: Dict[str, List[PointPatchLink]] = {}

    use_surface_index = surface_index_tolerance is not None and can_use_surface_index_backend(patches)
    subset_tolerance = surface_index_tolerance / distance_scale if use_surface_index else tolerance

    general_collections: Dict[int, Dict[str, Any]] = {}
    between_collections = []
    for collection_id, collection in point_collections.items():
        targets = _resolve_between_patches_targets(collection, patches)
        if targets is None:
            general_collections[collection_id] = collection
        else:
            between_collections.append((collection_id, collection, targets))

    built_surface_index = None
    if use_surface_index and (between_collections or general_collections):
        built_surface_index = _build_surface_patch_index(patches, surface_index_tolerance)

    if between_collections:
        linked_between = False
        if built_surface_index is not None:
            linked_between = _link_between_patch_collections_with_surface_index(
                links,
                patches,
                between_collections,
                surface_index_tolerance,
                distance_scale,
                built_index=built_surface_index,
            )
        if not linked_between:
            for collection_id, collection, targets in tqdm(
                between_collections, 'linking between-patch pcls'
            ):
                _link_collection_to_patch_subset(
                    links, collection_id, collection, targets, subset_tolerance
                )
        print(f'attached {len(between_collections)} between_patches collection(s) to their named patch pairs only')

    if surface_index_tolerance is not None:
        index_links = _link_points_to_patches_with_surface_index(
            patches,
            general_collections,
            surface_index_tolerance,
            distance_scale,
            built_index=built_surface_index,
            general_hit_policy=general_hit_policy,
        )
        if index_links is not None:
            for patch_id, patch_links in index_links.items():
                links.setdefault(patch_id, []).extend(patch_links)
            return links

    for collection_id, collection in tqdm(general_collections.items(), 'linking points to patches'):
        _link_collection_to_patch_subset(
            links, collection_id, collection, patches, tolerance, hit_policy=general_hit_policy
        )

    return links


def normalise_pcl_winding_annotations(point_collections):
    # Per-pcl: if every point has a winding annotation, leave alone; if none has one, set them all to 0;
    # if mixed, print a warning and strip the unannotated points.
    for pcl_id, pcl in point_collections.items():
        points = pcl['points']
        annotated = [pid for pid, p in points.items() if np.isfinite(p['winding_annotation'])]
        unannotated = [pid for pid, p in points.items() if not np.isfinite(p['winding_annotation'])]
        # Record whether the pcl carried any winding annotation in the source json,
        # *before* the all-unannotated case below 0-fills it. This is the only signal
        # that distinguishes a "same-winding" pcl (fiber / new_same_wind: no wind_a) from
        # a deliberate relative-winding pcl once both have finite annotations loaded.
        pcl['has_winding_annotations'] = bool(annotated)
        if not unannotated:
            continue
        if not annotated:
            for p in points.values():
                p['winding_annotation'] = 0.0
            continue
        print(
            f'WARNING: pcl {pcl_id} ({pcl.get("name", "?")}) has mixed winding-number annotations: '
            f'{len(annotated)} annotated, {len(unannotated)} missing — stripping unannotated points'
        )
        for pid in unannotated:
            del points[pid]
