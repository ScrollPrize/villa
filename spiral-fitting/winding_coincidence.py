"""Certify shared-surface XYZ agreement or locally aligned normal thickness."""
from collections import defaultdict
from dataclasses import dataclass
import itertools
import copy

import numpy as np


THETA_SCALE = 10000.  # Index conditioning only; theta remains measured in turns.


@dataclass
class Surface:
    mesh: object
    low: np.ndarray
    high: np.ndarray
    window_counts: dict | None = None


def refine_normal_grid(parameter, xyz, quads, cells, extent, window_size=3):
    """Uniform integer refinement preserves every original triangle exactly.

    Pick the nearest integer subdivision to the requested physical footprint.
    A common factor along both grid axes preserves original quad diagonals.
    Missing parent quads produce no child quads. This is interpolation, never
    smoothing or a change to the geometry exported for the source patch.
    """
    points = xyz[quads]
    row_length = np.median(np.linalg.norm(points[:, 2]-points[:, 0], axis=1))
    col_length = np.median(np.linalg.norm(points[:, 1]-points[:, 0], axis=1))
    spacing = (row_length+col_length)/2
    factor = max(1, int(np.floor((window_size-1)*spacing/extent+.5)))
    metadata = dict(subdivisions=factor, source_median_edge_voxels=[float(row_length), float(col_length)])
    if factor == 1:
        return parameter, xyz, quads, cells, metadata
    # Unique refined vertices share one index even across parent-cell borders.
    dr, dc = np.mgrid[:factor+1, :factor+1]
    locations = cells[:, None, :]*factor + np.column_stack((dr.ravel(), dc.ravel()))[None]
    refined_cells, first = np.unique(locations.reshape(-1, 2), axis=0, return_index=True)
    parent = first // ((factor+1)**2)
    local = refined_cells-cells[parent]*factor
    u, v = local[:, 0]/factor, local[:, 1]/factor
    lower = u+v <= 1
    weights = np.column_stack((np.maximum(1-u-v, 0),
                               np.where(lower, v, 1-u),
                               np.where(lower, u, 1-v), np.maximum(u+v-1, 0)))
    def interpolate(values):
        return np.einsum('nk,nkd->nd', weights, values[quads[parent]])
    refined_parameter, refined_xyz = interpolate(parameter), interpolate(xyz)
    dr, dc = np.mgrid[:factor, :factor]
    cells = (cells[:, None, :]*factor + np.column_stack((dr.ravel(), dc.ravel()))[None]).reshape(-1, 2)
    # Search integer packed coordinates rather than allocating a dense grid over holes.
    width = int(refined_cells[:, 1].max()+2)
    packed = refined_cells[:, 0]*width+refined_cells[:, 1]
    corner = cells[:, None, :]+np.array([[0,0],[0,1],[1,0],[1,1]])[None]
    quads = np.searchsorted(packed, corner[..., 0]*width+corner[..., 1])
    return refined_parameter, refined_xyz, quads, cells, metadata


def normal_grid_mesh(parameter, xyz, quads, cells, window_size=3, neighborhood_voxels=None):
    """Sliding intact source-grid windows; never fill a missing quad.

    Each original quad is covered by at least one window. Complete 3x3
    neighborhoods take precedence; uncovered boundary/hole quads use 2x2.
    Normals are least-squares plane normals fitted within each source window,
    before combining surfaces. Coordinates are not moved or smoothed.
    """
    from vc_spiral.winding_coincidence import ThicknessMesh
    if window_size not in (2, 3):
        raise ValueError('normal grid size must be 2 or 3')
    metadata = {}
    if neighborhood_voxels is not None:
        if isinstance(neighborhood_voxels, bool) or not np.isfinite(neighborhood_voxels) or neighborhood_voxels <= 0:
            raise ValueError('normal neighborhood must be finite and positive')
        parameter, xyz, quads, cells, metadata = refine_normal_grid(
            parameter, xyz, quads, cells, neighborhood_voxels, window_size)
    lookup = {tuple(rc): i for i, rc in enumerate(cells)}
    groups, covered = [], set()
    if window_size == 3:
        for row, col in sorted(lookup):
            group = [lookup.get((row+dr, col+dc)) for dr, dc in ((0,0),(0,1),(1,0),(1,1))]
            if all(q is not None for q in group):
                groups.append(group)
                covered.update(group)
    groups.extend([[q] for q in range(len(quads)) if q not in covered])
    faces = np.concatenate((quads[:, [0,1,2]], quads[:, [3,2,1]])).astype(np.int32)
    window_faces = np.full((len(groups), 8), -1, dtype=np.int32)
    normals = np.empty((len(groups), 3), dtype=np.float64)
    # Batch the small eigensystems separately by window size.
    for count in (1, 4):
        indices = [i for i, g in enumerate(groups) if len(g) == count]
        if not indices:
            continue
        vertices = np.array([np.unique(quads[groups[i]].ravel()) for i in indices])
        points = xyz[vertices]
        if count == 4 and neighborhood_voxels is not None:
            grid = points.reshape(-1, 3, 3, 3)
            sides = np.column_stack((np.linalg.norm(np.diff(grid[:, :, 1], axis=1), axis=2).sum(axis=1),
                                     np.linalg.norm(np.diff(grid[:, 1, :], axis=1), axis=2).sum(axis=1)))
            metadata['window_side_median_voxels'] = np.median(sides, axis=0).tolist()
        centered = points - points.mean(axis=1, keepdims=True)
        eigenvalues, vectors = np.linalg.eigh(np.einsum('nvi,nvj->nij', centered, centered))
        if np.any(eigenvalues[:, 1] <= 1e-12):
            raise ValueError('Undefined local surface normal')
        normals[indices] = vectors[:, :, 0]
        for i in indices:
            fs = groups[i]+[q+len(quads) for q in groups[i]]
            window_faces[i, :len(fs)] = fs
    return ThicknessMesh(np.ascontiguousarray(parameter), np.ascontiguousarray(xyz),
                         faces, window_faces, normals), dict(
        windows_3x3=sum(len(g)==4 for g in groups),
        windows_2x2=sum(len(g)==1 for g in groups), **metadata)


def patch_surface(fields, pid, label, metric="xyz", window_size=3, neighborhood_voxels=None):
    """Use every accepted fine face, including its outer boundary and holes."""
    from vc_spiral.winding_coincidence import Mesh
    field = fields.get(pid)
    select = np.flatnonzero(field['labels'] == label)
    rows, cols = field['graph'].node_qi[select], field['graph'].node_qj[select]
    grid = fields.patches[pid].zyxs.cpu().numpy()
    corner = np.stack([np.column_stack((rows+dr, cols+dc))
                       for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1))], axis=1)
    flat = corner[..., 0]*grid.shape[1] + corner[..., 1]
    used, inverse = np.unique(flat, return_inverse=True)
    inverse = inverse.reshape(-1, 4)
    zyx = grid.reshape(-1, 3)[used].astype(np.float64)
    theta = fields.frame(zyx)
    delta = theta[inverse] - field['theta'][select, None]
    lifts = field['lift'][select, None] + (delta < -.5) - (delta > .5)
    lo = np.full(len(used), np.iinfo(np.int64).max, dtype=np.int64)
    hi = np.full(len(used), np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(lo, inverse.ravel(), lifts.ravel())
    np.maximum.at(hi, inverse.ravel(), lifts.ravel())
    if np.any(lo != hi):
        raise ValueError('Inconsistent winding transport at shared patch corner')
    parameter = np.column_stack(((theta+lo)*THETA_SCALE, zyx[:, 0]))
    faces = np.concatenate((inverse[:, [0, 1, 2]], inverse[:, [3, 2, 1]])).astype(np.int32)
    counts = None
    if metric == 'normal_grid':
        mesh, counts = normal_grid_mesh(parameter, zyx[:, ::-1], inverse,
                                       np.column_stack((rows, cols)), window_size, neighborhood_voxels)
    else:
        mesh = Mesh(np.ascontiguousarray(parameter), np.ascontiguousarray(zyx[:, ::-1]), faces)
    return Surface(mesh, parameter.min(axis=0), parameter.max(axis=0), counts)


class CoincidenceGuard:
    """Monotone acceptance index; rejected surfaces never enter its coverage.

    A mesh comparison examines the vertices of every triangle intersection.
    Convexity of the norm bounds the XYZ separation throughout each shared
    polygon, not just at a set of probes. Self intersections are checked too.
    """
    cell_size = np.array([THETA_SCALE/16, 256.])

    def __init__(self, fields, patch_nodes, tolerance=3., metric="xyz", window_size=3, neighborhood_voxels=None):
        self.fields, self.patch_nodes, self.tolerance = fields, patch_nodes, tolerance
        self.metric, self.window_size = metric, window_size
        self.neighborhood_voxels = neighborhood_voxels
        self.surfaces = {}
        self.bins = defaultdict(set)
        self.pending = None
        self.checked_candidates = self.checked_pairs = self.vertices = 0
        self.max_accepted_distance = 0.

    def _keys(self, low, high):
        lo, hi = (np.floor(x/self.cell_size).astype(int) for x in (low, high))
        return itertools.product(range(lo[0], hi[0]+1), range(lo[1], hi[1]+1))

    def fork(self):
        """Stage a region merge without modifying either accepted region."""
        result = copy.copy(self)
        result.surfaces = self.surfaces.copy()
        result.bins = defaultdict(set, {key: values.copy() for key, values in self.bins.items()})
        result.pending = None
        return result

    def check(self, node, offset):
        self.pending = None
        if node not in self.patch_nodes:
            return None  # Fiber components carry topology, not patch area.
        try:
            if self.neighborhood_voxels is not None and self.metric == 'normal_grid':
                surface = patch_surface(self.fields, *self.patch_nodes[node], self.metric,
                                        self.window_size, self.neighborhood_voxels)
            else:
                surface = (patch_surface(self.fields, *self.patch_nodes[node]) if self.metric == 'xyz'
                           else patch_surface(self.fields, *self.patch_nodes[node], self.metric, self.window_size))
        except ValueError as error:
            self.checked_candidates += 1
            return dict(reason='invalid_theta_height_surface', source=node, detail=str(error))
        try:
            return self.check_surface(node, offset, surface)
        except ValueError as error:
            return dict(reason='invalid_local_surface', source=node, detail=str(error))

    def check_surface(self, node, offset, surface):
        """Also used to independently certify serialized, unflattened output."""
        self.pending = None
        self.checked_candidates += 1
        maximum = 0.
        if self.metric == 'xyz':
            check = surface.mesh.check_self(self.tolerance)
            self.vertices += check['intersection_vertices']
            if check['conflict']:
                return self._conflict(node, node, offset, check)
            maximum = check['max_distance']
        shift = np.array([offset*THETA_SCALE, 0.])
        low, high = surface.low+shift, surface.high+shift
        neighbours = set()
        for key in self._keys(low, high):
            neighbours.update(self.bins.get(key, ()))
        for other in sorted(neighbours):
            if self.metric == 'normal_grid' and node.rsplit(':', 1)[0] == other.rsplit(':', 1)[0] and node.startswith('patch:'):
                continue  # Disconnected components of one source are not a patch pair.
            saved, saved_offset = self.surfaces[other]
            saved_shift = np.array([saved_offset*THETA_SCALE, 0.])
            if np.any(low > saved.high+saved_shift) or np.any(saved.low+saved_shift > high):
                continue
            try:
                check = surface.mesh.compare(saved.mesh, (offset-saved_offset)*THETA_SCALE, self.tolerance)
            except ValueError as error:
                return dict(reason='invalid_local_surface', source=node, other=other, detail=str(error))
            self.checked_pairs += 1
            self.vertices += check['intersection_vertices']
            if check['conflict']:
                return self._conflict(node, other, saved_offset, check)
            maximum = max(maximum, check['max_distance'])
        self.pending = node, offset, surface, maximum
        return None

    def _conflict(self, node, other, offset, check):
        theta, height = check.pop('parameter')
        return dict(reason=('normal_grid_thickness' if self.metric == 'normal_grid' else 'theta_height_separation'), source=node, other=other,
                    theta=theta/THETA_SCALE+offset, height=height,
                    tolerance=self.tolerance, **check)

    def accept(self, node, offset):
        if node not in self.patch_nodes:
            return
        if self.pending is None or self.pending[:2] != (node, offset):
            raise RuntimeError('Patch accepted without a coincidence certificate')
        _, _, surface, maximum = self.pending
        self.surfaces[node] = surface, offset
        shift = np.array([offset*THETA_SCALE, 0.])
        for key in self._keys(surface.low+shift, surface.high+shift):
            self.bins[key].add(node)
        self.max_accepted_distance = max(self.max_accepted_distance, maximum)
        self.pending = None

    def report(self):
        if self.metric == 'normal_grid':
            sides = [s.window_counts['window_side_median_voxels'] for s, _ in self.surfaces.values()
                     if s.window_counts and 'window_side_median_voxels' in s.window_counts]
            return dict(neighborhood_voxels=self.neighborhood_voxels,
                patch_median_window_side_percentiles=(np.percentile(sides, [0,5,50,95,100], axis=0).tolist() if sides else None),
                tolerance=self.tolerance, metric=self.metric, grid_size=self.window_size,
                checked_candidates=self.checked_candidates, accepted_patches=len(self.surfaces),
                checked_patch_pairs=self.checked_pairs, tested_intersection_vertices=self.vertices,
                max_accepted_thickness=self.max_accepted_distance,
                window_counts={key: sum((s.window_counts or {}).get(key, 0) for s, _ in self.surfaces.values())
                               for key in ('windows_3x3', 'windows_2x2')},
                scope='sliding intact grid windows, local tangent-plane overlap, distinct source patches only',
                alignment='shared flat tangent plane; theta/source-Z broad phase; same winding only',
                normal='sign-aligned average of separately fitted least-squares window normals',
                measurement='max(n dot p) - min(n dot p) across both clipped window surfaces',
                interpolation='piecewise linear original XYZ on accepted quad triangles',
                certified=True)
        return dict(tolerance=self.tolerance, checked_candidates=self.checked_candidates,
                    accepted_patches=len(self.surfaces), checked_patch_pairs=self.checked_pairs,
                    tested_intersection_vertices=self.vertices,
                    max_accepted_separation=self.max_accepted_distance,
                    scope='all fine triangle intersections in unwrapped theta/source Z, including self intersections',
                    interpolation='piecewise linear original XYZ and theta on accepted quad triangles',
                    certified=True)
