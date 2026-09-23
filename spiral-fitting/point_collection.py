import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

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


# Side rules for PatchLinkOptions.side_rules: which side of a collection's
# points a patch surface may lie on. "Front" is the inward side, toward the
# umbilicus / the neighbouring winding with the lower winding number.
SIDE_FRONT = 'front'
SIDE_BEHIND = 'behind'
SIDE_RULES = (SIDE_FRONT, SIDE_BEHIND)


@dataclass(frozen=True)
class PatchLinkOptions:
    """Point-to-patch linking behaviour shared by every linking path.

    ``window_points`` / ``window_min_points``: a candidate patch (one the
    point itself hits within tolerance) is eligible only when at least
    ``window_min_points`` of the ``window_points`` consecutive points (in
    point-id order, centred on the point being linked, the point included)
    hit it within tolerance. Eligible candidates are ranked as always:
    largest area, then nearest, for general collections; nearest for
    between-patch ones. With ``window_min_points`` of 1 every candidate is
    eligible, which is the single-point behaviour. Even window counts round
    up to the next odd count so the window stays centred; where the window is
    clipped at the collection's ends the requirement is clipped to the
    members available.

    ``side_rules`` maps a collection id to SIDE_FRONT or SIDE_BEHIND: its
    points may only attach to patches whose surface lies on that side of the
    point. ``inward_direction`` maps ``(N, 3)`` zyx points to ``(N, 3)`` unit
    vectors pointing to the front side (inward: toward the umbilicus, or the
    fitted spiral's decreasing-winding direction), in the frame the points
    and patches are expressed in. With ``s`` the projection foot's offset from
    the point along that vector, a SIDE_FRONT collection keeps hits with
    ``s >= -side_margin`` and a SIDE_BEHIND one hits with ``s <= side_margin``
    (margin in point units). Rejected hits are dropped before selection, so
    in a window they count as misses.

    ``allowed_patches`` / ``rejected_patches`` map a collection id to patch
    ids, e.g. from a human review of fiber-to-patch placements: a collection
    with an allowed set only attaches to those patches, and never to a
    rejected one. Collections without an entry are unaffected. Like the side
    rules, the filter drops hits before the window gate and selection.
    """
    window_points: int = 1
    window_min_points: int = 1
    side_rules: Mapping[int, str] = None
    inward_direction: Optional[Callable[[np.ndarray], np.ndarray]] = None
    side_margin: float = 0.0
    allowed_patches: Mapping[int, frozenset] = None
    rejected_patches: Mapping[int, frozenset] = None

    def __post_init__(self):
        if int(self.window_points) < 1:
            raise ValueError(
                f'window_points must be >= 1, got {self.window_points!r}')
        object.__setattr__(self, 'window_points', int(self.window_points))
        if not 1 <= int(self.window_min_points) <= self.window_points:
            raise ValueError(
                f'window_min_points must be in [1, window_points], got '
                f'{self.window_min_points!r} with window_points='
                f'{self.window_points}')
        object.__setattr__(self, 'window_min_points', int(self.window_min_points))
        rules = dict(self.side_rules or {})
        for collection_id, rule in rules.items():
            if rule not in SIDE_RULES:
                raise ValueError(
                    f'side rule for collection {collection_id!r} must be one of '
                    f'{SIDE_RULES}, got {rule!r}')
        object.__setattr__(self, 'side_rules', rules)
        if rules and self.inward_direction is None:
            raise ValueError('side rules need an inward_direction function')
        for name in ('allowed_patches', 'rejected_patches'):
            object.__setattr__(self, name, {
                collection_id: frozenset(patch_ids)
                for collection_id, patch_ids in dict(getattr(self, name) or {}).items()})

    def patch_filter(self, collection_id):
        """(allowed or None, rejected) patch ids for a collection, or None
        when neither applies."""
        allowed = self.allowed_patches.get(collection_id)
        rejected = self.rejected_patches.get(collection_id, frozenset())
        if allowed is None and not rejected:
            return None
        return allowed, rejected

    @property
    def window_half(self) -> int:
        """Window members on each side of the linked point."""
        return self.window_points // 2

    @property
    def gates_on_window(self) -> bool:
        return self.window_half > 0 and self.window_min_points > 1

    def side_rule(self, collection_id) -> Optional[str]:
        return self.side_rules.get(collection_id)


def umbilicus_inward_direction(z_to_umbilicus_yx):
    """Inward-direction function (see PatchLinkOptions) pointing from each
    point to the umbilicus at its z, in the yx plane. ``z_to_umbilicus_yx``
    maps an array of z to ``(N, 2)`` yx centres. Points on the axis get a
    zero vector, which no side rule can reject."""
    def inward(zyxs):
        zyxs = np.asarray(zyxs, dtype=np.float64).reshape(-1, 3)
        centres = np.asarray(
            z_to_umbilicus_yx(zyxs[:, 0]), dtype=np.float64).reshape(-1, 2)
        direction = np.zeros_like(zyxs)
        direction[:, 1:] = centres - zyxs[:, 1:]
        norms = np.linalg.norm(direction, axis=1)
        defined = norms > 0
        direction[defined] /= norms[defined, None]
        return direction
    return inward


DEFAULT_LINK_OPTIONS = PatchLinkOptions()


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

    except FileNotFoundError:
        # Default configs list optional annotation files (e.g.
        # drawn_control_points.json) that may not exist in a dataset.
        print(f"WARNING: point collection file not found, skipping: {filename}")
        return None
    except Exception as e:
        print(f"Error loading point collection from {filename}: {e}")
        return None


def _load_surface_index_backend():
    try:
        from vc_spiral import surface_index
        return surface_index
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
            collection_name=collection.get('name', ''),
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


def _ordered_point_items(collection: Dict[str, Any]):
    """``(point_id, point)`` items in point-id order: the chain order every
    linking window is taken along (see spiral_helpers.SequenceChain)."""
    return sorted(collection['points'].items(), key=lambda kv: int(kv[0]))


def _point_zyxs(point_items) -> np.ndarray:
    return np.ascontiguousarray(np.stack([
        np.asarray(point.get('zyx', point['p'][::-1]), dtype=np.float32)
        for _, point in point_items
    ], axis=0), dtype=np.float32)


def _projection_feet(point_idx, surf_idx, ijs, patch_for_surface):
    """Projection foot (zyx) of every hit, NaN where it cannot be evaluated."""
    feet = np.full((len(point_idx), 3), np.nan, dtype=np.float32)
    for surface in np.unique(surf_idx):
        patch = patch_for_surface(int(surface))
        if patch is None or not hasattr(patch, 'ij_to_zyx'):
            continue
        rows = np.flatnonzero(surf_idx == surface)
        ij = torch.as_tensor(np.asarray(ijs[rows], dtype=np.float32))
        zyx, valid = patch.ij_to_zyx(ij)
        if not bool(valid.all()):
            # A foot exactly on the last grid row/column is out of range for
            # the bilinear lookup; nudge it into the adjacent quad.
            h, w = patch.zyxs.shape[:2]
            nudged = ij.clone()
            nudged[:, 0] = nudged[:, 0].clamp(0.0, h - 1 - 1e-3)
            nudged[:, 1] = nudged[:, 1].clamp(0.0, w - 1 - 1e-3)
            zyx_nudged, valid_nudged = patch.ij_to_zyx(nudged)
            fill = (~valid) & valid_nudged
            zyx[fill] = zyx_nudged[fill]
            valid = valid | valid_nudged
        valid_np = valid.cpu().numpy()
        feet[rows[valid_np]] = zyx.cpu().numpy()[valid_np]
    return feet


def _side_reject_mask(point_zyxs, point_idx, surf_idx, ijs, patch_for_surface, rule, options):
    """True for hits on the wrong side of their point under ``rule`` (see
    PatchLinkOptions.side_rules). A foot that cannot be evaluated, or a point
    with an undefined inward direction, is never rejected."""
    reject = np.zeros(len(point_idx), dtype=bool)
    if len(point_idx) == 0:
        return reject
    feet = _projection_feet(point_idx, surf_idx, ijs, patch_for_surface)
    known = np.isfinite(feet).all(axis=1)
    if not known.any():
        return reject
    points = point_zyxs[point_idx[known]].astype(np.float64)
    inward = np.asarray(options.inward_direction(points), dtype=np.float64).reshape(-1, 3)
    defined = np.linalg.norm(inward, axis=1) > 0
    offset = ((feet[known].astype(np.float64) - points) * inward).sum(axis=1)
    margin = float(options.side_margin)
    if rule == SIDE_FRONT:
        wrong = offset < -margin
    else:
        wrong = offset > margin
    reject[known] = defined & wrong
    return reject


def _window_hit_gate(point_idx, surf_idx, num_points, half, min_points):
    """True for hits whose patch is hit by at least ``min_points`` of the
    point's window (the point itself included; see PatchLinkOptions). The
    requirement is clipped to the window members available at the
    collection's ends."""
    keep = np.zeros(len(point_idx), dtype=bool)
    for surface in np.unique(surf_idx):
        rows = np.flatnonzero(surf_idx == surface)
        hit = np.zeros(num_points, dtype=np.int64)
        hit[point_idx[rows]] = 1
        cumulative = np.concatenate(([0], np.cumsum(hit)))
        centre = point_idx[rows]
        lo = np.maximum(centre - half, 0)
        hi = np.minimum(centre + half, num_points - 1)
        count = cumulative[hi + 1] - cumulative[lo]
        required = np.minimum(min_points, hi - lo + 1)
        keep[rows] = count >= required
    return keep


def _best_hit_per_point(point_idx, surf_idx, distances, ijs, *, patch_areas=None):
    if len(surf_idx) == 0:
        return None
    # Primary sort: point. Then, for general PCLs, the largest patch area (the
    # historical fit_spiral preference); then nearest distance. Surface index
    # is a deterministic tie-break.
    keys = [surf_idx, distances]
    if patch_areas is not None:
        keys.append(-patch_areas[surf_idx])
    keys.append(point_idx)
    order = np.lexsort(keys)
    sorted_points = point_idx[order]
    _, first = np.unique(sorted_points, return_index=True)
    winners = order[first]
    return point_idx[winners], surf_idx[winners], distances[winners], ijs[winners]


def _link_from_hits(
    collection: Dict[str, Any],
    collection_id: int,
    point_items,
    point_zyxs: np.ndarray,
    point_idx: np.ndarray,
    surf_idx: np.ndarray,
    distances: np.ndarray,
    ijs: np.ndarray,
    *,
    distance_scale: float,
    surface_ids,
    patch_for_surface,
    patch_areas,
    options: PatchLinkOptions,
    target_idxs=None,
) -> Dict[str, List[PointPatchLink]]:
    """Choose one patch per point from its candidate hits and record the links.

    ``point_idx``/``surf_idx``/``distances``/``ijs`` are flat, one row per
    (point, patch) hit within the query tolerance; ``point_idx`` indexes
    ``point_items`` (id order). Points already carrying ``on_patch`` still
    serve as window members but are never re-linked.
    """
    local: Dict[str, List[PointPatchLink]] = {}
    if len(surf_idx) == 0:
        return local
    keep = surf_idx >= 0
    if target_idxs is not None:
        keep &= np.isin(surf_idx, target_idxs)
    patch_filter = options.patch_filter(collection_id)
    if patch_filter is not None:
        allowed, rejected = patch_filter
        for surface in np.unique(surf_idx[keep]):
            patch_id = surface_ids[int(surface)]
            if (allowed is not None and patch_id not in allowed) or patch_id in rejected:
                keep &= surf_idx != surface
    if not keep.all():
        point_idx, surf_idx, distances, ijs = (
            point_idx[keep], surf_idx[keep], distances[keep], ijs[keep])
    rule = options.side_rule(collection_id)
    if rule is not None and len(point_idx):
        reject = _side_reject_mask(
            point_zyxs, point_idx, surf_idx, ijs, patch_for_surface, rule, options)
        if reject.any():
            accept = ~reject
            point_idx, surf_idx, distances, ijs = (
                point_idx[accept], surf_idx[accept], distances[accept], ijs[accept])
    if options.gates_on_window and len(point_idx):
        eligible = _window_hit_gate(
            point_idx, surf_idx, len(point_items), options.window_half,
            options.window_min_points)
        if not eligible.all():
            point_idx, surf_idx, distances, ijs = (
                point_idx[eligible], surf_idx[eligible], distances[eligible],
                ijs[eligible])
    best = _best_hit_per_point(
        point_idx, surf_idx, distances, ijs, patch_areas=patch_areas)
    if best is None:
        return local
    win_points, win_surfaces, win_distances, win_ijs = best
    for k in range(len(win_points)):
        point_id, point = point_items[int(win_points[k])]
        if 'on_patch' in point:
            continue
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


def _hits_from_offsets(offsets):
    """Flat point index for the surface index's CSR-style hit offsets."""
    return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))


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
    options: PatchLinkOptions = DEFAULT_LINK_OPTIONS,
) -> Optional[Dict[str, List[PointPatchLink]]]:
    if built_index is None:
        built_index = _build_surface_patch_index(patches, tolerance)
    if built_index is None:
        return None
    index, surface_ids = built_index

    links: Dict[str, List[PointPatchLink]] = {}
    if not point_collections:
        return links

    print(f'linking points to patches with vc_spiral.surface_index ({len(surface_ids)} patches)')

    if not hasattr(index, 'locate_all_xyz_batch'):
        if options.gates_on_window or options.side_rules:
            # The single-hit legacy query cannot count window hits or test
            # the side rules; let the brute-force path handle it.
            return None
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

    def patch_for_surface(surface):
        return patches.get(surface_ids[surface])

    def worker(item):
        collection_id, collection = item
        point_items = _ordered_point_items(collection)
        if not point_items:
            return {}

        point_zyxs = _point_zyxs(point_items)
        offsets, surf_idx, distances, ijs = index.locate_all_xyz_batch(
            np.ascontiguousarray(point_zyxs[:, ::-1]), tolerance)
        return _link_from_hits(
            collection, collection_id, point_items, point_zyxs,
            _hits_from_offsets(offsets), surf_idx, distances, ijs,
            distance_scale=distance_scale,
            surface_ids=surface_ids, patch_for_surface=patch_for_surface,
            patch_areas=patch_areas, options=options)

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
    options: PatchLinkOptions = DEFAULT_LINK_OPTIONS,
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
        f'linking between-patch pcls with vc_spiral.surface_index ({len(surface_ids)} patches'
        f'{", subset query" if use_subset_query else ""})'
    )

    def patch_for_surface(surface):
        return patches.get(surface_ids[surface])

    def worker(item):
        collection_id, collection, targets = item
        target_idxs = np.fromiter(
            (surface_id_to_idx[patch_id] for patch_id in targets if patch_id in surface_id_to_idx),
            dtype=np.int64,
        )
        if len(target_idxs) == 0:
            return {}

        point_items = _ordered_point_items(collection)
        if not point_items:
            return {}

        point_zyxs = _point_zyxs(point_items)
        point_xyzs = np.ascontiguousarray(point_zyxs[:, ::-1])
        if use_subset_query:
            offsets, surf_idx, distances, ijs = index.locate_all_xyz_batch_in(
                point_xyzs, target_idxs.astype(np.int32), tolerance
            )
            restrict_to = None
        else:
            offsets, surf_idx, distances, ijs = index.locate_all_xyz_batch(point_xyzs, tolerance)
            restrict_to = target_idxs
        return _link_from_hits(
            collection, collection_id, point_items, point_zyxs,
            _hits_from_offsets(offsets), surf_idx, distances, ijs,
            distance_scale=distance_scale,
            surface_ids=surface_ids, patch_for_surface=patch_for_surface,
            patch_areas=None, options=options, target_idxs=restrict_to)

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
    options: PatchLinkOptions = DEFAULT_LINK_OPTIONS,
) -> None:
    """Attach each point of one collection to the best of ``candidate_patches``.

    Brute-force projection (``Patch.project``) restricted to the given patch
    subset; every patch within ``tolerance`` is a candidate and the selection
    (nearest, largest area, window hit gate, side rules) is the same as the
    surface-index path's. Shared by the general fallback (all patches) and the
    "between_patches" special case (the named pair only).
    """
    point_items = _ordered_point_items(collection)
    if not point_items or not candidate_patches:
        return
    point_zyxs = _point_zyxs(point_items)
    surface_ids = list(candidate_patches)
    tolerance32 = np.float32(tolerance)

    hit_points: List[int] = []
    hit_surfaces: List[int] = []
    hit_distances: List[float] = []
    hit_ijs: List[List[float]] = []
    for surface, patch_id in enumerate(surface_ids):
        patch = candidate_patches[patch_id]
        for position in range(len(point_items)):
            point_zyx = torch.as_tensor(point_zyxs[position], dtype=torch.float32)
            ij_coord, distance = patch.project(point_zyx)
            distance32 = np.float32(distance.detach().cpu().item())
            if distance32 > tolerance32:
                continue
            hit_points.append(position)
            hit_surfaces.append(surface)
            hit_distances.append(distance32)
            hit_ijs.append(ij_coord.detach().cpu().reshape(-1)[:2].tolist())
    if not hit_points:
        return

    patch_areas = None
    if hit_policy == 'largest_area':
        patch_areas = np.asarray(
            [float(candidate_patches[patch_id].area) for patch_id in surface_ids],
            dtype=np.float64)
    local = _link_from_hits(
        collection, collection_id, point_items, point_zyxs,
        np.asarray(hit_points, dtype=np.int64),
        np.asarray(hit_surfaces, dtype=np.int64),
        np.asarray(hit_distances, dtype=np.float32),
        np.asarray(hit_ijs, dtype=np.float32).reshape(-1, 2),
        distance_scale=1.0, surface_ids=surface_ids,
        patch_for_surface=lambda surface: candidate_patches[surface_ids[surface]],
        patch_areas=patch_areas, options=options)
    for patch_id, patch_links in local.items():
        links.setdefault(patch_id, []).extend(patch_links)


def link_points_to_patches(
    patches: Dict[str, Patch],
    point_collections: Dict[int, Dict[str, Any]],
    tolerance: float = 10.0,
    surface_index_tolerance: Optional[float] = None,
    distance_scale: float = 1.0,
    general_hit_policy: str = 'nearest',
    options: Optional[PatchLinkOptions] = None,
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

    ``options`` (PatchLinkOptions) adds the window hit gate and the side
    rules on top of either policy; points already carrying ``on_patch`` are
    never re-linked.
    """
    options = options or DEFAULT_LINK_OPTIONS
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
                options=options,
            )
        if not linked_between:
            for collection_id, collection, targets in tqdm(
                between_collections, 'linking between-patch pcls'
            ):
                _link_collection_to_patch_subset(
                    links, collection_id, collection, targets, subset_tolerance,
                    options=options,
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
            options=options,
        )
        if index_links is not None:
            for patch_id, patch_links in index_links.items():
                links.setdefault(patch_id, []).extend(patch_links)
            return links

    for collection_id, collection in tqdm(general_collections.items(), 'linking points to patches'):
        _link_collection_to_patch_subset(
            links, collection_id, collection, patches, tolerance,
            hit_policy=general_hit_policy, options=options,
        )

    return links


def link_unattached_points_to_patches(
    point_collections: Dict[int, Dict[str, Any]],
    all_patches: Dict[str, Patch],
    new_patches: Dict[str, Patch],
    tolerance: float = 10.0,
    surface_index_tolerance: Optional[float] = None,
    distance_scale: float = 1.0,
    general_hit_policy: str = 'nearest',
    options: Optional[PatchLinkOptions] = None,
) -> Dict[int, int]:
    """Offer still-unattached points of resident collections to newly added patches.

    Points already carrying ``on_patch`` keep their attachment; only points
    without one are linked, and only against ``new_patches`` (a subset of
    ``all_patches``). Because an unattached point was already rejected by every
    other patch, this yields the same attachments a fresh start over
    ``all_patches`` would, at the cost of indexing the new patches only. The
    attached points still count as window members for ``options``' window
    hit gate.

    ``all_patches`` is consulted only to resolve "between_patches__XXX__YYY"
    collections to their named pair (see ``link_points_to_patches``): such a
    collection attaches to whichever of its named patches are new, and never to
    an unrelated new patch.

    Returns ``{collection_id: newly attached point count}`` for the collections
    that gained at least one attachment.
    """
    if not new_patches:
        return {}
    options = options or DEFAULT_LINK_OPTIONS
    general_shadows: Dict[int, Dict[str, Any]] = {}
    between_shadows = []
    shadows: Dict[int, Dict[str, Any]] = {}
    attached_before: Dict[int, set] = {}
    for collection_id, collection in point_collections.items():
        absolute = collection.get('metadata', {}).get('winding_is_absolute', False)
        already = {
            point_id for point_id, point in collection['points'].items()
            if 'on_patch' in point
        }
        if len(already) == len(collection['points']):
            continue
        # A shadow shares the point dicts, so attachments land on the resident
        # collection; the container is separate so nothing else is touched.
        # Every point is offered so windows see their attached neighbours;
        # the linker never re-links a point that already carries on_patch.
        shadow = {
            'name': collection.get('name', ''),
            'metadata': collection.get('metadata', {}),
            'points': {
                point_id: point for point_id, point in collection['points'].items()
                if point_id in already or not absolute
                or (np.isfinite(point['winding_annotation'])
                    and point['winding_annotation'] > 0)
            },
        }
        shadows[collection_id] = shadow
        attached_before[collection_id] = already
        targets = _resolve_between_patches_targets(collection, all_patches)
        if targets is None:
            general_shadows[collection_id] = shadow
        else:
            new_targets = {
                patch_id: patch for patch_id, patch in targets.items()
                if patch_id in new_patches
            }
            if new_targets:
                between_shadows.append((collection_id, shadow, new_targets))

    links: Dict[str, List[PointPatchLink]] = {}
    use_surface_index = (surface_index_tolerance is not None
                         and can_use_surface_index_backend(new_patches))
    subset_tolerance = (surface_index_tolerance / distance_scale
                        if use_surface_index else tolerance)
    for collection_id, shadow, new_targets in between_shadows:
        _link_collection_to_patch_subset(
            links, collection_id, shadow, new_targets, subset_tolerance,
            options=options)
    if general_shadows:
        link_points_to_patches(
            new_patches,
            general_shadows,
            tolerance=tolerance,
            surface_index_tolerance=surface_index_tolerance,
            distance_scale=distance_scale,
            general_hit_policy=general_hit_policy,
            options=options,
        )

    gained: Dict[int, int] = {}
    for collection_id, shadow in shadows.items():
        count = sum(
            1 for point_id, point in shadow['points'].items()
            if 'on_patch' in point and point_id not in attached_before[collection_id])
        if count:
            gained[collection_id] = count
    return gained


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
