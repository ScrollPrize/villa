import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import torch
from tqdm import tqdm

from tifxyz import Patch, interpolate_quad_triangles


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

    Analysis callers can set ``strict_min_points`` (no end-of-chain relaxation),
    ``rank_by_support`` (support count, RMS, center distance, stable patch ID),
    and ``max_wrong_side_fractions`` per collection. A fraction rule replaces
    per-hit side rejection with a local-window gate, retaining tolerated
    wrong-side hits and excluding undefined directions from support. Fraction
    equality passes. ``report_counts`` receives per-collection candidate and
    attachment counts; callers using workers must make it thread-safe.

    ``strips`` switches from per-point selection to strip-level
    winner-take-all: a strip is one collection's points hitting one patch,
    from the first hit to the last (``strip_max_gap``, default None, limits
    the consecutive points without a hit before a new strip starts). Strips
    with fewer than ``window_min_points`` hits are dropped (the window itself
    is unused). ``strip_rank`` orders eligible strips: ``'fit'`` by RMS hit
    distance in ``strip_fit_resolution`` bins, then more hits; ``'hits'`` by
    more hits, then fit bins. Both then prefer larger patch area and patch ID.
    Each strip claims every point of its span that no better strip has
    claimed, so a stretch of points never alternates between patches;
    ``strip_claim_gap`` (default None: the whole span) limits the claim to
    the hits and holes of at most that many points between them, so a long
    hole inside a strip stays open to other patches. What a losing strip
    keeps is split at claimed points and re-checked: fragments with fewer
    than ``window_min_points`` free hits are dropped. With a
    fraction rule the wrong-side gate applies per strip.

    ``target_patches`` maps a fiber collection id to the patch ids its points
    may attach to (its ``explicit_link:`` tags, see
    spiral_helpers.explicit_patch_links_from_tags); hits on any other patch
    are dropped before selection. With ``restrict_untargeted`` a fiber
    collection absent from the map attaches to no patch at all. Only fiber
    collections (``sampling_group == 'fibers'`` or
    ``metadata.logical_input_kind == 'fiber'``) are restricted; regular and
    absolute-winding collections never are. ``None`` (the default) restricts
    nothing.
    """
    window_points: int = 1
    window_min_points: int = 1
    side_rules: Mapping[int, str] = None
    inward_direction: Optional[Callable[[np.ndarray], np.ndarray]] = None
    side_margin: float = 0.0
    # Opt-in analysis policies; defaults preserve the training linker.
    strict_min_points: bool = False
    rank_by_support: bool = False
    max_wrong_side_fractions: Mapping[int, float] = None
    report_counts: Optional[Callable[[dict], None]] = None
    strips: bool = False
    strip_max_gap: Optional[int] = None
    strip_claim_gap: Optional[int] = None
    strip_rank: str = 'fit'
    strip_fit_resolution: float = 0.25
    target_patches: Optional[Mapping[int, frozenset]] = None
    restrict_untargeted: bool = False

    def __post_init__(self):
        if int(self.window_points) < 1:
            raise ValueError(
                f'window_points must be >= 1, got {self.window_points!r}')
        object.__setattr__(self, 'window_points', int(self.window_points))
        strip_mode = bool(self.strips)
        if not 1 <= int(self.window_min_points) <= (
                max(self.window_points, int(self.window_min_points)) if strip_mode else self.window_points):
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
        fractions = dict(self.max_wrong_side_fractions or {})
        for cid, fraction in fractions.items():
            if cid not in rules or not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError('wrong-side fractions need a side rule and a value in [0, 1]')
        object.__setattr__(self, 'max_wrong_side_fractions', fractions)
        for name in ('strip_max_gap', 'strip_claim_gap'):
            value = getattr(self, name)
            if value is not None:
                if int(value) < 0:
                    raise ValueError(f'{name} must be >= 0, got {value!r}')
                object.__setattr__(self, name, int(value))
        if not (np.isfinite(self.strip_fit_resolution) and self.strip_fit_resolution > 0):
            raise ValueError(
                f'strip_fit_resolution must be > 0, got {self.strip_fit_resolution!r}')
        if self.strip_rank not in ('fit', 'hits'):
            raise ValueError(f"strip_rank must be 'fit' or 'hits', got {self.strip_rank!r}")
        if self.target_patches is not None:
            targets = {}
            for collection_id, patch_ids in dict(self.target_patches).items():
                if isinstance(patch_ids, str) or not all(
                        isinstance(patch_id, str) for patch_id in patch_ids):
                    raise ValueError(
                        f'target patches for collection {collection_id!r} must be a '
                        f'set of patch id strings, got {patch_ids!r}')
                targets[collection_id] = frozenset(patch_ids)
            object.__setattr__(self, 'target_patches', targets)
        elif self.restrict_untargeted:
            raise ValueError('restrict_untargeted needs a target_patches map')
        object.__setattr__(self, 'restrict_untargeted', bool(self.restrict_untargeted))

    @property
    def strip_mode(self) -> bool:
        return bool(self.strips)

    @property
    def restricts_patches(self) -> bool:
        """Whether any collection's candidate patches may be restricted."""
        return self.target_patches is not None

    def allowed_patches(self, collection_id, collection) -> Optional[frozenset]:
        """Patch ids ``collection`` may attach to, or ``None`` when it is
        unrestricted. Only fiber collections are ever restricted."""
        if self.target_patches is None:
            return None
        metadata = collection.get('metadata') or {}
        is_fiber = (collection.get('sampling_group') == 'fibers'
                    or metadata.get('logical_input_kind') == 'fiber')
        if not is_fiber:
            return None
        allowed = self.target_patches.get(collection_id)
        if allowed is None and self.restrict_untargeted:
            return frozenset()
        return allowed

    @property
    def window_half(self) -> int:
        """Window members on each side of the linked point."""
        return self.window_points // 2

    @property
    def gates_on_window(self) -> bool:
        return (self.window_min_points > 1 and (self.window_half > 0 or self.strict_min_points)
                and not self.strip_mode)

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


class NoValidLinkQuads(ValueError):
    """Sampling left a patch without any usable coarse cells."""


class SampledLinkPatch(Patch):
    """Link-only coarse quads, separated by invalid rows/columns.

    Separators let both the native vertex-based index and Patch.project use
    exactly the same valid coarse cells, without bridging holes. Returned ij
    coordinates are in this packed grid until ``original_ij`` maps them back.
    """

    def __init__(self, source, stride):
        vertices = source.zyxs.detach().cpu()
        h, w = vertices.shape[:2]
        self.sample_rows = np.unique(np.r_[np.arange(0, h, stride), h - 1])
        self.sample_cols = np.unique(np.r_[np.arange(0, w, stride), w - 1])
        rows, cols = self.sample_rows, self.sample_cols
        # A coarse cell is usable only if every underlying fine quad is valid.
        invalid = (~source.valid_quad_mask.cpu().numpy()).astype(np.int64)
        prefix = np.pad(invalid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        bad = (prefix[rows[1:, None], cols[None, 1:]]
               - prefix[rows[:-1, None], cols[None, 1:]]
               - prefix[rows[1:, None], cols[None, :-1]]
               + prefix[rows[:-1, None], cols[None, :-1]])
        valid = torch.from_numpy(bad == 0)
        if not bool(valid.any()):
            raise NoValidLinkQuads('no valid sampled quads')
        grid = torch.full((3 * len(valid) - 1, 3 * valid.shape[1] - 1, 3), -1., dtype=torch.float32)
        for i in range(2):
            for j in range(2):
                corners = vertices[rows[i:len(rows) - 1 + i, None], cols[None, j:len(cols) - 1 + j]]
                grid[i::3, j::3] = torch.where(valid[..., None], corners, -1.)
        super().__init__(grid, source.scale, source.overlapping_ids, None, uuid=source.uuid)
        self.area = source.area  # largest-area selection still uses original area

    def _cell_coordinates(self, ij):
        ij = torch.as_tensor(ij, dtype=torch.float32)
        cell = torch.floor(ij / 3).long()
        cell[..., 0].clamp_(0, len(self.sample_rows) - 2)
        cell[..., 1].clamp_(0, len(self.sample_cols) - 2)
        uv = ij - 3 * cell
        return cell, uv

    def original_ij(self, ij):
        cell, uv = self._cell_coordinates(ij)
        rows = torch.as_tensor(self.sample_rows, device=uv.device)
        cols = torch.as_tensor(self.sample_cols, device=uv.device)
        r, c = cell.unbind(-1)
        return torch.stack([rows[r] + uv[..., 0] * (rows[r + 1] - rows[r]),
                            cols[c] + uv[..., 1] * (cols[c + 1] - cols[c])], dim=-1)

    def ij_to_zyx(self, ij, *, triangulated=True):
        # Match the two triangles used by both projection backends, including
        # exact edge vertices; bilinear interpolation would move a curved foot.
        cell, uv = self._cell_coordinates(ij)
        r, c = (3 * cell).unbind(-1)
        grid = self.zyxs.to(uv.device)
        a, b, d, e = grid[r, c], grid[r + 1, c], grid[r, c + 1], grid[r + 1, c + 1]
        u, v = uv[..., :1], uv[..., 1:]
        zyx = interpolate_quad_triangles(a, d, b, e, u, v)
        valid = ((uv >= 0) & (uv <= 1)).all(-1) & self.valid_quad_mask.to(uv.device)[r, c]
        return zyx, valid


def sample_patches_for_linking(patches, stride):
    """Return link-only geometry; stride 1 reuses the original patches."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError('patch stride must be a positive integer')
    if stride == 1:
        return patches
    sampled = {}
    for pid, patch in patches.items():
        try:
            sampled[pid] = SampledLinkPatch(patch, stride)
        except NoValidLinkQuads:
            continue
    return sampled


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
    grids = []
    scales = []
    for patch_id, patch in patches.items():
        zyx = patch.zyxs.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        scale = patch.scale.detach().cpu().numpy() if hasattr(patch.scale, 'detach') else patch.scale
        surface_ids.append(patch_id)
        grids.append(zyx)
        scales.append((float(scale[0]), float(scale[1])))

    if not grids:
        return None

    index = surface_index.SurfacePatchIndex()
    if hasattr(index, 'rebuild_grids'):
        # Bulk build: the same index, with the geometry copies in parallel.
        scales_np = np.asarray(scales, dtype=np.float32).reshape(-1, 2)
        index.rebuild_grids(surface_ids, grids,
                            np.ascontiguousarray(scales_np[:, 0]),
                            np.ascontiguousarray(scales_np[:, 1]),
                            bbox_padding=tolerance, sampling_stride=1)
    else:
        surface_defs = [surface_index.QuadSurface(patch_id, zyx, si, sj)
                        for patch_id, zyx, (si, sj) in zip(surface_ids, grids, scales)]
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


def _projection_feet(point_idx, surf_idx, ijs, patch_for_surface, *, triangulated=False,
                     interpolate=None):
    """Projection foot (zyx) of every hit, NaN where it cannot be evaluated.

    ``interpolate(surface_indices, ijs, triangulated)`` -- the native surface
    index's ``interpolate_ij``, a bit-exact twin of ``Patch.ij_to_zyx`` plus the
    out-of-range nudge below over every indexed patch -- evaluates all hits in
    one call; without it each surface's feet are lifted with its own patch."""
    feet = np.full((len(point_idx), 3), np.nan, dtype=np.float32)
    if interpolate is not None and len(point_idx):
        zyx, valid = interpolate(
            np.ascontiguousarray(surf_idx, dtype=np.int32),
            np.ascontiguousarray(np.asarray(ijs, dtype=np.float32).reshape(-1, 2)),
            bool(triangulated))
        feet[valid] = zyx[valid]
        return feet
    for surface in np.unique(surf_idx):
        patch = patch_for_surface(int(surface))
        if patch is None or not hasattr(patch, 'ij_to_zyx'):
            continue
        rows = np.flatnonzero(surf_idx == surface)
        ij = torch.as_tensor(np.asarray(ijs[rows], dtype=np.float32))
        kwargs = {'triangulated': True} if triangulated and isinstance(patch, Patch) else {}
        zyx, valid = patch.ij_to_zyx(ij, **kwargs)
        if not bool(valid.all()):
            # A foot exactly on the last grid row/column is out of range for
            # the bilinear lookup; nudge it into the adjacent quad.
            h, w = patch.zyxs.shape[:2]
            nudged = ij.clone()
            nudged[:, 0] = nudged[:, 0].clamp(0.0, h - 1 - 1e-3)
            nudged[:, 1] = nudged[:, 1].clamp(0.0, w - 1 - 1e-3)
            zyx_nudged, valid_nudged = patch.ij_to_zyx(nudged, **kwargs)
            fill = (~valid) & valid_nudged
            zyx[fill] = zyx_nudged[fill]
            valid = valid | valid_nudged
        valid_np = valid.cpu().numpy()
        feet[rows[valid_np]] = zyx.cpu().numpy()[valid_np]
    return feet


def _side_measurements(point_zyxs, point_idx, surf_idx, ijs, patch_for_surface, rule, options,
                       interpolate=None, point_directions=None):
    """Return wrong-side and defined masks for candidate projection feet.

    ``point_directions`` (num_points, 3), when given, holds the collection's
    inward directions already evaluated (at least) at every point with a
    candidate hit, in place of calling ``options.inward_direction``."""
    reject = np.zeros(len(point_idx), dtype=bool)
    defined_hits = np.zeros(len(point_idx), dtype=bool)
    if len(point_idx) == 0:
        return reject, defined_hits
    feet = _projection_feet(point_idx, surf_idx, ijs, patch_for_surface,
                            triangulated=bool(options.max_wrong_side_fractions),
                            interpolate=interpolate)
    known = np.isfinite(feet).all(axis=1)
    if not known.any():
        return reject, defined_hits
    points = point_zyxs[point_idx[known]].astype(np.float64)
    unique_points, inverse = np.unique(point_idx[known], return_inverse=True)
    if point_directions is not None:
        directions = point_directions[unique_points]
    else:
        directions = options.inward_direction(point_zyxs[unique_points].astype(np.float64))
    inward = np.asarray(directions, dtype=np.float64).reshape(-1, 3)[inverse]
    defined = np.isfinite(inward).all(axis=1) & (np.linalg.norm(inward, axis=1) > 0)
    offset = ((feet[known].astype(np.float64) - points) * inward).sum(axis=1)
    margin = float(options.side_margin)
    if rule == SIDE_FRONT:
        wrong = offset < -margin
    else:
        wrong = offset > margin
    reject[known] = defined & wrong
    defined_hits[known] = defined
    return reject, defined_hits


def _side_reject_mask(point_zyxs, point_idx, surf_idx, ijs, patch_for_surface, rule, options):
    """Legacy per-hit rule: undefined directions never reject a hit."""
    return _side_measurements(
        point_zyxs, point_idx, surf_idx, ijs, patch_for_surface, rule, options)[0]


def _window_hit_statistics(point_idx, surf_idx, distances, num_points, half, wrong):
    """Support, RMS distance, and wrong-side count before candidate gating."""
    counts = np.zeros(len(point_idx), dtype=np.int64)
    rms = np.zeros(len(point_idx), dtype=np.float64)
    wrong_counts = np.zeros(len(point_idx), dtype=np.int64)
    for surface in np.unique(surf_idx):
        rows = np.flatnonzero(surf_idx == surface)
        positions = point_idx[rows]
        values = np.zeros((num_points, 3), dtype=np.float64)
        values[positions, 0] = 1
        values[positions, 1] = np.asarray(distances[rows], dtype=np.float64) ** 2
        values[positions, 2] = wrong[rows]
        cumulative = np.vstack([np.zeros(3), values.cumsum(axis=0)])
        lo = np.maximum(positions - half, 0)
        hi = np.minimum(positions + half + 1, num_points)
        totals = cumulative[hi] - cumulative[lo]
        counts[rows] = totals[:, 0].astype(np.int64)
        rms[rows] = np.sqrt(np.maximum(totals[:, 1], 0) / totals[:, 0])
        wrong_counts[rows] = totals[:, 2].astype(np.int64)
    return counts, rms, wrong_counts


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


def _strip_select(point_idx, surf_idx, distances, wrong, num_points, options, fraction,
                  surface_ranks, patch_for_surface):
    """Strip-level winner-take-all over the flat hits (see PatchLinkOptions).

    Returns the kept hit rows (at most one per point) and a counts dict:
    ``support_rejected`` hits in strips or fragments below ``window_min_points``,
    ``side_rejected`` hits in strips over the wrong-side fraction,
    ``overruled`` hits in eligible strips that a better strip claimed, and
    ``strips_kept``.
    """
    counts = dict(support_rejected=0, side_rejected=0, overruled=0, strips_kept=0)
    keep_rows = []
    if len(point_idx) == 0:
        return np.zeros(0, dtype=np.int64), counts
    gap = options.strip_max_gap if options.strip_max_gap is not None else num_points
    claim_gap = options.strip_claim_gap if options.strip_claim_gap is not None else gap
    min_hits = options.window_min_points
    resolution = float(options.strip_fit_resolution)
    strips = []
    areas = {}

    def area(surface):
        if surface not in areas:
            patch = patch_for_surface(surface)
            areas[surface] = -float(patch.area) if patch is not None and hasattr(patch, 'area') else 0.0
        return areas[surface]
    # One hit per (point, surface): keep the nearest.
    order = np.lexsort((distances, point_idx, surf_idx))
    point_idx_o, surf_idx_o = point_idx[order], surf_idx[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (point_idx_o[1:] != point_idx_o[:-1]) | (surf_idx_o[1:] != surf_idx_o[:-1])
    order = order[first]
    surfaces = surf_idx[order]
    bounds = np.flatnonzero(np.diff(surfaces)) + 1
    for rows in np.split(order, bounds):
        positions = point_idx[rows]  # ascending within a surface
        breaks = np.flatnonzero(np.diff(positions) > gap + 1) + 1
        for strip_rows in np.split(rows, breaks):
            n = len(strip_rows)
            if n < min_hits:
                counts['support_rejected'] += n
                continue
            wrong_count = int(wrong[strip_rows].sum())
            if fraction is not None and wrong_count > fraction * n:
                counts['side_rejected'] += n
                continue
            rms = float(np.sqrt(np.mean(np.asarray(distances[strip_rows], dtype=np.float64) ** 2)))
            surface = int(surf_idx[strip_rows[0]])
            fit_bin = int(np.floor(rms / resolution))
            lead = (fit_bin, -n) if options.strip_rank == 'fit' else (-n, fit_bin)
            key = lead + (area(surface), int(surface_ranks[surface]), int(point_idx[strip_rows[0]]))
            strips.append((key, strip_rows))
    strips.sort(key=lambda item: item[0])
    owner = np.full(num_points, -1, dtype=np.int64)
    for _, strip_rows in strips:
        positions = point_idx[strip_rows]
        free = owner[positions] < 0
        overruled = int((~free).sum())
        strip_rows, positions = strip_rows[free], positions[free]
        if len(positions):
            # Fragments: split where the strip's own gap rule breaks or where a
            # better strip owns a point in between.
            lo = positions[0]
            owned = np.concatenate(([0], np.cumsum(owner[lo:positions[-1] + 1] >= 0)))
            claimed_between = owned[positions[1:] - lo] - owned[positions[:-1] - lo + 1] > 0
            breaks = np.flatnonzero((np.diff(positions) > claim_gap + 1) | claimed_between) + 1
            for fragment in np.split(strip_rows, breaks):
                if len(fragment) < min_hits:
                    counts['support_rejected'] += len(fragment)
                    continue
                span = point_idx[fragment]
                owner[span[0]:span[-1] + 1] = surf_idx[fragment[0]]
                keep_rows.append(fragment)
                counts['strips_kept'] += 1
        counts['overruled'] += overruled
    kept = np.concatenate(keep_rows) if keep_rows else np.zeros(0, dtype=np.int64)
    return kept, counts


def _surface_id_ranks(surface_ids):
    """Position of each surface's id in the sorted id list: an integer key that
    sorts exactly like the id strings (ids are unique), for _best_hit_per_point."""
    ranks = np.empty(len(surface_ids), dtype=np.int64)
    ranks[np.argsort(np.asarray(surface_ids), kind='stable')] = np.arange(len(surface_ids))
    return ranks


def _best_hit_per_point(point_idx, surf_idx, distances, ijs, *, patch_areas=None,
                        support=None, rms=None, surface_ids=None, surface_ranks=None):
    if len(surf_idx) == 0:
        return None
    # Primary sort: point. Then, for general PCLs, the largest patch area (the
    # historical fit_spiral preference); then nearest distance. Surface index
    # is a deterministic tie-break.
    keys = [surf_idx, distances]
    if patch_areas is not None:
        keys.append(-patch_areas[surf_idx])
    if support is not None:
        # Patch-id order tie-break: the precomputed integer ranks order exactly
        # like the id strings (see _surface_id_ranks) without a string sort
        # per collection.
        if surface_ranks is None:
            surface_ranks = _surface_id_ranks(surface_ids)
        keys = [surface_ranks[surf_idx], distances, rms, -support]
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
    interpolate=None,
    surface_ranks=None,
    point_directions=None,
) -> Dict[str, List[PointPatchLink]]:
    """Choose one patch per point from its candidate hits and record the links.

    ``point_idx``/``surf_idx``/``distances``/``ijs`` are flat, one row per
    (point, patch) hit within the query tolerance; ``point_idx`` indexes
    ``point_items`` (id order). Points already carrying ``on_patch`` still
    serve as window members but are never re-linked.
    """
    if collection.get('metadata', {}).get('winding_is_absolute', False):
        # Absolute anchors are independent measurements, often on successive
        # windings. Neighbour support and fiber side rules do not apply.
        # Retain reporting and the caller's distance/patch-selection policy.
        options = PatchLinkOptions(report_counts=options.report_counts)
    local: Dict[str, List[PointPatchLink]] = {}
    if len(surf_idx) == 0:
        return local
    keep = surf_idx >= 0
    if target_idxs is not None:
        keep &= np.isin(surf_idx, target_idxs)
    allowed = options.allowed_patches(collection_id, collection)
    if allowed is not None:
        # Explicit patch links: only the named patches remain candidates (an
        # empty set drops every hit, so the collection stays unattached).
        allowed_idxs = np.fromiter(
            (surface for surface, patch_id in enumerate(surface_ids) if patch_id in allowed),
            dtype=np.int64)
        keep &= np.isin(surf_idx, allowed_idxs)
    if not keep.all():
        point_idx, surf_idx, distances, ijs = (
            point_idx[keep], surf_idx[keep], distances[keep], ijs[keep])
    rule = options.side_rule(collection_id)
    fraction = options.max_wrong_side_fractions.get(collection_id)
    counts_report = dict(candidate_hits=len(point_idx), undefined_side_hits=0,
                         side_rejected_candidates=0, support_rejected_candidates=0,
                         attached_points=0)
    wrong = np.zeros(len(point_idx), dtype=bool)
    if rule is not None and len(point_idx):
        wrong, defined = _side_measurements(
            point_zyxs, point_idx, surf_idx, ijs, patch_for_surface, rule, options,
            interpolate=interpolate, point_directions=point_directions)
        reject = ~defined if fraction is not None else wrong
        counts_report['undefined_side_hits'] = int((~defined).sum())
        if fraction is None:
            counts_report['side_rejected_candidates'] = int(reject.sum())
        if reject.any():
            accept = ~reject
            point_idx, surf_idx, distances, ijs = (
                point_idx[accept], surf_idx[accept], distances[accept], ijs[accept])
            wrong = wrong[accept]
    support = rms = None
    if options.strip_mode:
        if surface_ranks is None:
            surface_ranks = _surface_id_ranks(surface_ids)
        kept, strip_counts = _strip_select(
            point_idx, surf_idx, distances, wrong, len(point_items), options, fraction,
            surface_ranks, patch_for_surface)
        counts_report['support_rejected_candidates'] = strip_counts['support_rejected']
        counts_report['side_rejected_candidates'] += strip_counts['side_rejected']
        counts_report['strip_overruled_candidates'] = strip_counts['overruled']
        counts_report['strips_kept'] = strip_counts['strips_kept']
        point_idx, surf_idx, distances, ijs = (
            point_idx[kept], surf_idx[kept], distances[kept], ijs[kept])
    elif options.rank_by_support or options.strict_min_points or fraction is not None:
        support, rms, wrong_counts = _window_hit_statistics(
            point_idx, surf_idx, distances, len(point_items), options.window_half, wrong)
        required = options.window_min_points
        if not options.strict_min_points:
            required = np.minimum(required, np.minimum(point_idx + options.window_half + 1, len(point_items))
                                  - np.maximum(point_idx - options.window_half, 0))
        supported = support >= required
        side_ok = np.ones(len(point_idx), dtype=bool)
        if fraction is not None:
            side_ok = wrong_counts <= fraction * support
        counts_report['support_rejected_candidates'] = int((~supported).sum())
        counts_report['side_rejected_candidates'] += int((supported & ~side_ok).sum())
        eligible = supported & side_ok
        point_idx, surf_idx, distances, ijs, support, rms = (
            point_idx[eligible], surf_idx[eligible], distances[eligible],
            ijs[eligible], support[eligible], rms[eligible])
    elif options.gates_on_window and len(point_idx):
        eligible = _window_hit_gate(point_idx, surf_idx, len(point_items),
                                    options.window_half, options.window_min_points)
        counts_report['support_rejected_candidates'] = int((~eligible).sum())
        if not eligible.all():
            point_idx, surf_idx, distances, ijs = (
                point_idx[eligible], surf_idx[eligible], distances[eligible],
                ijs[eligible])
    best = _best_hit_per_point(
        point_idx, surf_idx, distances, ijs, patch_areas=patch_areas,
        support=support if options.rank_by_support else None,
        rms=rms, surface_ids=surface_ids, surface_ranks=surface_ranks)
    counts_report['attached_points'] = 0 if best is None else sum(
        'on_patch' not in point_items[int(i)][1] for i in best[0])
    if options.report_counts is not None:
        options.report_counts(counts_report)
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


def _run_collection_link_workers(work_items, worker_fn, desc, *, merge=None, results=None):
    """Run ``worker_fn`` over ``work_items`` on the linking thread pool, in item
    order, folding each result into ``results`` with ``merge(results, result)``.
    The default merge extends a patch-id -> links dict."""
    if results is None:
        results = {}
    if merge is None:
        def merge(merged, local):
            for patch_id, patch_links in local.items():
                if patch_links:
                    merged.setdefault(patch_id, []).extend(patch_links)

    max_workers = _link_thread_count(len(work_items))
    if max_workers <= 1:
        for item in tqdm(work_items, desc):
            merge(results, worker_fn(item))
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for local in tqdm(executor.map(worker_fn, work_items), desc, total=len(work_items)):
            merge(results, local)
    return results


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
        if (options.gates_on_window or options.side_rules or options.rank_by_support
                or options.report_counts or options.strip_mode or options.restricts_patches):
            # The single-hit legacy query cannot count window hits, test the
            # side rules or restrict candidate patches; let the brute-force
            # path handle it.
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
    surface_ranks = (_surface_id_ranks(surface_ids)
                     if options.rank_by_support or options.strip_mode else None)

    def patch_for_surface(surface):
        return patches.get(surface_ids[surface])

    # Side rules lift every candidate hit's projection foot. The native index
    # can do that for all hits of a collection in one call (a bit-exact twin of
    # Patch.ij_to_zyx over its stored geometry) when every indexed patch is a
    # plain Patch; sampled link patches interpolate differently and keep the
    # per-patch path.
    interpolate = None
    if (options.side_rules and hasattr(index, 'interpolate_ij')
            and all(type(patches[patch_id]) is Patch for patch_id in surface_ids)):
        interpolate = index.interpolate_ij

    def locate(item):
        collection_id, collection = item
        point_items = _ordered_point_items(collection)
        if not point_items:
            return None
        point_zyxs = _point_zyxs(point_items)
        offsets, surf_idx, distances, ijs = index.locate_all_xyz_batch(
            np.ascontiguousarray(point_zyxs[:, ::-1]), tolerance)
        return (collection_id, collection, point_items, point_zyxs,
                _hits_from_offsets(offsets), surf_idx, distances, ijs)

    # Phase 1: candidate hits of every collection (the native query releases
    # the GIL, so this threads well).
    located = _run_collection_link_workers(
        list(point_collections.items()), locate, 'locating patch hits',
        merge=lambda results, item: results.append(item), results=[])
    located = [item for item in located if item is not None]

    # Phase 2: side rules need the inward winding direction at every point with
    # a candidate hit. Evaluate them for all ruled collections in one call --
    # the model direction is a device evaluation whose cost is per call, not per
    # point -- and hand each collection its own (num_points, 3) table.
    directions_by_collection = {}
    if options.side_rules:
        batch = []
        for collection_id, collection, point_items, point_zyxs, point_idx, *_ in located:
            if (options.side_rule(collection_id) is None or len(point_idx) == 0
                    or collection.get('metadata', {}).get('winding_is_absolute', False)):
                continue
            hit_points = np.unique(point_idx)
            batch.append((collection_id, len(point_items), hit_points, point_zyxs[hit_points]))
        if batch:
            directions = np.asarray(options.inward_direction(
                np.concatenate([zyxs for *_, zyxs in batch]).astype(np.float64)),
                dtype=np.float64).reshape(-1, 3)
            start = 0
            for collection_id, num_points, hit_points, _ in batch:
                table = np.full((num_points, 3), np.nan, dtype=np.float64)
                table[hit_points] = directions[start:start + len(hit_points)]
                directions_by_collection[collection_id] = table
                start += len(hit_points)

    # Phase 3: choose and record one patch per point.
    def worker(item):
        collection_id, collection, point_items, point_zyxs, point_idx, surf_idx, distances, ijs = item
        return _link_from_hits(
            collection, collection_id, point_items, point_zyxs,
            point_idx, surf_idx, distances, ijs,
            distance_scale=distance_scale,
            surface_ids=surface_ids, patch_for_surface=patch_for_surface,
            patch_areas=patch_areas, options=options,
            interpolate=interpolate, surface_ranks=surface_ranks,
            point_directions=directions_by_collection.get(collection_id))

    index_links = _run_collection_link_workers(located, worker, 'linking points to patches')
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
    built_index=None,
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
    never re-linked. Collections marked ``metadata.winding_is_absolute``
    attach each point independently, without window or side rules.

    ``built_index`` -- ``_build_surface_patch_index(patches,
    surface_index_tolerance)`` output -- lets callers linking several groups
    of collections against the same patches build the index once.
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

    built_surface_index = built_index if use_surface_index else None
    if built_surface_index is None and use_surface_index and (between_collections or general_collections):
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
