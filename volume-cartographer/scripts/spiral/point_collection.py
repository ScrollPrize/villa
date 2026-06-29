import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from tifxyz import Patch


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


def _link_points_to_patches_with_surface_index(
    patches: Dict[str, Patch],
    point_collections: Dict[int, Dict[str, Any]],
    tolerance: float,
    distance_scale: float,
) -> Optional[Dict[str, List[PointPatchLink]]]:
    surface_index = _load_surface_index_backend()
    if surface_index is None:
        return None

    surface_defs = []
    for patch_id, patch in patches.items():
        zyx = patch.zyxs.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        scale = patch.scale.detach().cpu().numpy() if hasattr(patch.scale, 'detach') else patch.scale
        surface_defs.append(surface_index.QuadSurface(patch_id, zyx, float(scale[0]), float(scale[1])))

    links: Dict[str, List[PointPatchLink]] = {}
    if not surface_defs or not point_collections:
        return links

    print(f'linking points to patches with vc.surface_index ({len(surface_defs)} patches)')
    index = surface_index.SurfacePatchIndex()
    index.rebuild(surface_defs, bbox_padding=tolerance, sampling_stride=1)

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

        for patch_id, patch in candidate_patches.items():
            ij_coord, distance = patch.project(point_zyx)
            if distance < nearest_distance and distance <= tolerance_t:
                nearest_distance = distance
                nearest_patch_id = patch_id
                nearest_ij = ij_coord

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
) -> Dict[str, List[PointPatchLink]]:
    """Process point collections and link them to patches.

    Special case: a collection named "between_patches__XXX__YYY" (written by
    connect_overlapping_patches.py) attaches its points only to patches XXX and
    YYY, when both exist. Such collections are handled here by projecting onto
    just that pair; all other collections go through the general nearest-patch
    search (surface index when available, else brute force).

    The between_patches subset match is always brute force, so it must use the
    same effective cutoff (in brute-force/voxel units) as the regime the general
    collections use: the surface index accepts points within
    ``surface_index_tolerance`` of a patch and reports distances divided by
    ``distance_scale``, which is the cutoff ``surface_index_tolerance /
    distance_scale`` in brute-force units; the brute-force paths report distances
    directly and use ``tolerance``.
    """
    links: Dict[str, List[PointPatchLink]] = {}

    # The general collections take the surface-index path iff a surface-index
    # tolerance is requested and the backend is available; everything else is
    # brute force. Match that regime's cutoff for the between_patches subset.
    use_surface_index = surface_index_tolerance is not None and can_use_surface_index_backend(patches)
    subset_tolerance = surface_index_tolerance / distance_scale if use_surface_index else tolerance

    # Pull out the "between_patches" collections whose named pair both exist and
    # attach each only to its pair; the rest go through the general search.
    general_collections: Dict[int, Dict[str, Any]] = {}
    n_between = 0
    for collection_id, collection in point_collections.items():
        targets = _resolve_between_patches_targets(collection, patches)
        if targets is None:
            general_collections[collection_id] = collection
        else:
            n_between += 1
            _link_collection_to_patch_subset(links, collection_id, collection, targets, subset_tolerance)
    if n_between:
        print(f'attached {n_between} between_patches collection(s) to their named patch pairs only')

    if surface_index_tolerance is not None:
        index_links = _link_points_to_patches_with_surface_index(
            patches, general_collections, surface_index_tolerance, distance_scale
        )
        if index_links is not None:
            for patch_id, patch_links in index_links.items():
                links.setdefault(patch_id, []).extend(patch_links)
            return links

    for collection_id, collection in tqdm(general_collections.items(), 'linking points to patches'):
        _link_collection_to_patch_subset(links, collection_id, collection, patches, tolerance)

    return links


def _point_cache_key(point):
    """Stable per-point cache key: the exact (x, y, z) from the pcl json. Independent
    of the winding annotation (applied fresh at stamp time), so editing an annotation
    is a cache hit and only moving a point's position invalidates its entry."""
    return tuple(point['p'])


def _surface_index_link_queries(patches, queries_xyz, tolerance, distance_scale):
    """Link a batch of world-space xyz query points to their nearest patch via the
    surface index. ``queries_xyz`` is an (N, 3) float32 array in xyz order. Returns a
    length-N list of hit dicts ``{'id', 'distance', 'ij'}`` (distance already divided
    by ``distance_scale``; ij = (grid_y, grid_x), matching ``locate_xyz``) or None per
    point with no hit. Returns None entirely if the surface-index backend is missing.

    Uses the GIL-releasing ``locate_all_xyz_batch`` (one C++ call over all points) when
    available, picking the nearest hit with a known id per point; otherwise falls back
    to a per-point ``locate_xyz`` loop. The index is rebuilt once for the whole batch.
    """
    surface_index = _load_surface_index_backend()
    if surface_index is None:
        return None

    surface_defs = []
    for patch_id, patch in patches.items():
        zyx = patch.zyxs.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        scale = patch.scale.detach().cpu().numpy() if hasattr(patch.scale, 'detach') else patch.scale
        surface_defs.append(surface_index.QuadSurface(patch_id, zyx, float(scale[0]), float(scale[1])))

    index = surface_index.SurfacePatchIndex()
    index.rebuild(surface_defs, bbox_padding=tolerance, sampling_stride=1)

    n = len(queries_xyz)
    results = [None] * n
    xyz = np.ascontiguousarray(queries_xyz, dtype=np.float32)
    if hasattr(index, 'locate_xyz_nearest_batch') and hasattr(index, 'surface_ids'):
        # One GIL-released, multithreaded C++ call: single-nearest per point, compact
        # arrays. Distances on unknown surfaces (surf_idx < 0) are dropped.
        surf_idx, distance, ij = index.locate_xyz_nearest_batch(xyz, tolerance)
        surf_idx = np.asarray(surf_idx); distance = np.asarray(distance); ij = np.asarray(ij)
        ids = index.surface_ids()
        hit_mask = surf_idx >= 0
        for k in np.nonzero(hit_mask)[0]:
            results[int(k)] = {
                'id': ids[int(surf_idx[k])],
                'distance': float(distance[k]) / distance_scale,
                'ij': [float(ij[k, 0]), float(ij[k, 1])],
            }
    else:
        for k in range(n):
            hit = index.locate_xyz(np.ascontiguousarray(xyz[k]), tolerance)
            if hit is None:
                continue
            results[k] = {'id': hit['id'], 'distance': float(hit['distance']) / distance_scale,
                          'ij': list(hit['ij'])}
    return results


def _brute_force_link_queries(patches, queries_zyx, tolerance):
    """Brute-force fallback (no surface-index backend): project each zyx query point
    onto every patch and keep the nearest within ``tolerance``. Returns a list of hit
    dicts ``{'id', 'distance', 'ij'}`` or None, matching _surface_index_link_queries."""
    device = torch.device('cpu')
    tol_t = torch.tensor(tolerance, device=device, dtype=torch.float32)
    results = []
    for zyx in queries_zyx:
        pt = torch.as_tensor(zyx, dtype=torch.float32, device=device)
        best_id, best_ij = None, None
        best_d = torch.tensor(float('inf'), device=device)
        for patch_id, patch in patches.items():
            ij_coord, distance = patch.project(pt)
            if distance < best_d and distance <= tol_t:
                best_d, best_id, best_ij = distance, patch_id, ij_coord
        results.append(None if best_id is None else
                       {'id': best_id, 'distance': float(best_d.cpu().item()), 'ij': best_ij.tolist()})
    return results


def link_points_to_patches_pointcache(
    patches: Dict[str, Patch],
    point_collections: Dict[int, Dict[str, Any]],
    tolerance: float,
    surface_index_tolerance: Optional[float],
    distance_scale: float,
    cache: Dict[Any, Any],
    brute_force_only: bool = False,
) -> Dict[str, List[PointPatchLink]]:
    """Per-point-cached variant of link_points_to_patches.

    ``cache`` maps ``_point_cache_key(point) -> hit_or_None`` and is mutated in place
    with any newly-linked points (the caller persists it). Only points whose POSITION
    is new get linked; winding annotations are read fresh from ``point_collections`` at
    stamp time. When there are no new points the surface index is never rebuilt, so a
    re-run (or an annotation-only edit) does no linking work at all.

    ``between_patches__…`` collections keep their dedicated brute-force pair attachment
    (cheap, not cached). Mirrors link_points_to_patches' cutoff handling.
    """
    links: Dict[str, List[PointPatchLink]] = {}
    use_surface_index = (not brute_force_only and surface_index_tolerance is not None
                         and can_use_surface_index_backend(patches))
    subset_tolerance = surface_index_tolerance / distance_scale if use_surface_index else tolerance

    # between_patches collections: dedicated named-pair attachment, unchanged.
    general_collections: Dict[int, Dict[str, Any]] = {}
    n_between = 0
    for collection_id, collection in point_collections.items():
        targets = _resolve_between_patches_targets(collection, patches)
        if targets is None:
            general_collections[collection_id] = collection
        else:
            n_between += 1
            _link_collection_to_patch_subset(links, collection_id, collection, targets, subset_tolerance)
    if n_between:
        print(f'attached {n_between} between_patches collection(s) to their named patch pairs only')

    # Gather every general point with its cache key; collect cache misses (new positions).
    point_refs = []
    missing = {}  # key -> zyx (dedup repeated positions)
    for collection_id, collection in general_collections.items():
        for point_id, point in collection['points'].items():
            point_zyx = np.asarray(point.get('zyx', point['p'][::-1]), dtype=np.float32)
            key = _point_cache_key(point)
            point_refs.append((key, collection_id, collection, point_id, point))
            if key not in cache and key not in missing:
                missing[key] = point_zyx

    if missing:
        miss_keys = list(missing.keys())
        miss_zyx = [missing[k] for k in miss_keys]
        eff_tol = surface_index_tolerance if use_surface_index else tolerance
        hits = None
        if use_surface_index:
            miss_xyz = np.stack([z[::-1] for z in miss_zyx]).astype(np.float32)
            print(f'linking {len(miss_keys)} new point(s) to patches via surface index (batched)')
            hits = _surface_index_link_queries(patches, miss_xyz, eff_tol, distance_scale)
        if hits is None:  # no surface-index backend (or disabled) -> brute force the misses
            print(f'linking {len(miss_keys)} new point(s) to patches via brute-force projection')
            hits = _brute_force_link_queries(patches, miss_zyx, tolerance)
        for key, hit in zip(miss_keys, hits):
            cache[key] = hit  # hit dict or None (explicit "no patch" is cached too)

    # Stamp every general point from the cache, applying its fresh annotation.
    for key, collection_id, collection, point_id, point in point_refs:
        hit = cache.get(key)
        if hit is None:
            continue
        _record_point_patch_link(links, collection, collection_id, point_id, point,
                                 hit['id'], hit['distance'], hit['ij'])

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


