"""Geometry-aware vertical guides, metric checks, and legacy fit experiments.

Production workspace conversion uses winding_slim and the flatboi/Pastix driver.

The deformation is shared by all sources. XYZ plus the existing winding-chart
coordinate distinguishes projection overlaps from genuine spatial neighbors.
Vertical centerlines supply column constraints; patch triangles supply the
metric. Source Z supplies a row reference with a configurable dead band.
"""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import factorized
from scipy.ndimage import gaussian_filter1d

from winding_arclength import source_grid


def coordinates(points):
    return np.column_stack((points[:, :3], points[:, 3] * .25))


class VerticalGuides:
    def __init__(self, lines, coarse_spacing=250.):
        if not lines:
            raise ValueError('No vertical fibers available for straightening')
        self.lines, self.coords, self.trees, self.heights = [], [], [], []
        arcs = []
        coarse, owners, columns = [], [], []
        for i, original in enumerate(lines):
            line = np.asarray(original, dtype=float).copy()
            if line.ndim != 2 or line.shape[1] != 5 or len(line) < 2 or not np.isfinite(line).all():
                raise ValueError('Invalid vertical fiber geometry')
            if line[-1, 2] < line[0, 2]:
                line = line[::-1].copy()
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line[:, :3], axis=0), axis=1))]
            if len(line) < 2 or arc[-1] <= 0 or not np.isfinite(line).all():
                raise ValueError('Invalid vertical fiber geometry')
            coord = coordinates(line)
            idx = np.unique(np.r_[np.searchsorted(arc, np.arange(0, arc[-1], coarse_spacing)), len(line)-1])
            self.lines.append(line)
            arcs.append(arc)
            self.coords.append(coord)
            self.trees.append(cKDTree(coord))
            coarse.append(coord[idx])
            owners.append(np.full(len(idx), i))
            columns.append(np.median(line[:, 3]))
        self.initial_columns = np.asarray(columns)
        self.coarse_tree = cKDTree(np.concatenate(coarse))
        self.owners = np.concatenate(owners)
        self.fine_tree = cKDTree(np.concatenate(self.coords))
        self.fine_owners = np.repeat(np.arange(len(lines)), [len(line) for line in self.lines])
        # A row reference is a common function of Z, not a requirement that
        # one display unit equal one unit of Z. Measure vertical length too:
        # curved/slanted fibers have more arclength than height span.
        lo = min(line[:, 2].min() for line in self.lines)
        hi = max(line[:, 2].max() for line in self.lines)
        if hi <= lo:
            raise ValueError('Vertical guides need nonzero overall Z extent')
        self.row_z = np.linspace(lo, hi, max(2, int(np.ceil((hi-lo)/256))+1))
        speeds = [[] for _ in range(len(self.row_z)-1)]
        bends = []
        for line, arc, coord in zip(self.lines, arcs, self.coords):
            z = line[:, 2]
            low, high = np.minimum(z[:-1], z[1:]), np.maximum(z[:-1], z[1:])
            dz = high-low
            ds = np.diff(arc)
            bends.extend(coord[:-1][np.diff(z) < 0])
            for i, (a, b) in enumerate(zip(self.row_z[:-1], self.row_z[1:])):
                a, b = max(a, z.min()), min(b, z.max())
                if b-a >= 32:
                    fraction = np.clip((np.minimum(high, b)-np.maximum(low, a))/np.maximum(dz, 1e-20), 0., 1.)
                    fraction[(dz == 0) & (low >= a) & (low < b)] = 1.
                    speeds[i].append(np.sum(ds*fraction)/(b-a))
        self.bend_tree = cKDTree(np.asarray(bends)) if bends else None
        speed = np.array([np.median(values) if values else np.nan for values in speeds])
        valid = np.flatnonzero(np.isfinite(speed))
        if len(valid):
            speed = np.interp(np.arange(len(speed)), valid, speed[valid])
            speed = gaussian_filter1d(speed, 1., mode='nearest')
        else:
            speed[:] = 1.
        self.row_v = np.r_[lo, lo+np.cumsum(speed*np.diff(self.row_z))]
        # Preserve acquisition order and full 3D length through S-bends.
        # Source Z is a soft row reference, never the material path coordinate.
        self.heights = [arc + np.median(self.row_reference(line[:, 2])-arc)
                        for line, arc in zip(self.lines, arcs)]

    def row_reference(self, z):
        """Shared measured vertical arclength, linearly continued at the ends."""
        result = np.interp(z, self.row_z, self.row_v)
        for index in (0, -1):
            neighbor = 1 if index == 0 else -2
            slope = (self.row_v[neighbor]-self.row_v[index])/(self.row_z[neighbor]-self.row_z[index])
            mask = z < self.row_z[0] if index == 0 else z > self.row_z[-1]
            result = np.where(mask, self.row_v[index]+slope*(z-self.row_z[index]), result)
        return result

    def contacts(self, radius=1.):
        """Keep overlapping vertical traces together, not in independent columns."""
        pairs = set()
        for i, coord in enumerate(self.coords):
            idx = np.linspace(0, len(coord)-1, min(len(coord), 2048), dtype=int)
            distance, neighbor = self.fine_tree.query(coord[idx], k=list(range(1, min(8, self.fine_tree.n)+1)), workers=4)
            other = self.fine_owners[neighbor]
            for j in np.unique(other[(distance <= radius) & (other != i)]):
                pairs.add(tuple(sorted((i, int(j)))))
        return sorted(pairs)

    def contact_targets(self, pairs):
        target = np.zeros((len(pairs), 2))
        for k, (i, j) in enumerate(pairs):
            distance, index = self.trees[j].query(self.coords[i])
            keep = distance <= 1.
            target[k, 1] = np.median(self.heights[j][index[keep]]-self.heights[i][keep])
        return target

    def row_weights(self, points):
        """A Z row prior must yield near a real bend that reverses source Z."""
        if self.bend_tree is None:
            return np.ones(len(points))
        distance, _ = self.bend_tree.query(coordinates(points), workers=4)
        return distance**2/(distance**2+250.**2)

    def project(self, i, query):
        """Nearest polyline segment, including bends with uneven sample spacing.

        The nearest vertex alone need not border the nearest segment. Expand
        candidate vertices until the longest-segment bound excludes all unseen
        segments: a segment's closest endpoint is at most half its length away
        from its interior projection.
        """
        coord = self.coords[i]
        half_squared = np.max(np.sum(np.diff(coord, axis=0)**2, axis=1))*.25
        result_j = np.empty(len(query), dtype=int)
        result_t = np.empty(len(query))
        result_distance = np.empty(len(query))
        pending = np.arange(len(query))
        k = min(8, len(coord))
        while len(pending):
            q = query[pending]
            distances, vertices = self.trees[i].query(q, k=list(range(1, k+1)))
            j = np.clip(np.concatenate((vertices-1, vertices), axis=1), 0, len(coord)-2)
            a = coord[j]
            delta = coord[j+1]-a
            t = np.clip(np.sum((q[:, None]-a)*delta, axis=2)/np.maximum(np.sum(delta*delta, axis=2), 1e-20), 0, 1)
            squared = np.sum((q[:, None]-a-t[:, :, None]*delta)**2, axis=2)
            best = squared.argmin(axis=1)
            rows = np.arange(len(q))
            result_j[pending] = j[rows, best]
            result_t[pending] = t[rows, best]
            result_distance[pending] = squared[rows, best]
            if k == len(coord):
                break
            uncertain = distances[:, -1]**2 <= result_distance[pending]+half_squared
            pending = pending[uncertain]
            k = min(k*2, len(coord))
        return result_j, result_t, result_distance

    def basis(self, points, with_distance=False):
        """Return column weights and the fixed part of horizontal placement.

        Each candidate curve is projected onto its original polyline, including
        local Z reversals. This avoids sorting a curve by height and changing its
        connectivity. Coarse samples select smoothly weighted candidate curves.
        """
        n = len(self.lines)
        query = coordinates(points)
        k = min(64, len(self.owners))
        distances, indices = self.coarse_tree.query(query, k=list(range(1, k+1)), workers=4)
        candidate = self.owners[indices]
        radius = np.maximum(distances[:, -1, None] * (1 + 1e-6), 1e-10)
        coarse_weight = np.maximum(0., 1 - distances / radius) ** 2
        key = np.arange(len(points))[:, None] * n + candidate
        unique, inverse = np.unique(key, return_inverse=True)
        weights = np.bincount(inverse.ravel(), weights=coarse_weight.ravel(), minlength=len(unique))
        rows, cols = unique // n, unique % n
        near_u = np.empty(len(unique))
        height_offset = np.empty(len(unique))
        squared_distance = np.empty(len(unique))
        for i in np.unique(cols):
            selected = np.flatnonzero(cols == i)
            q = query[rows[selected]]
            j, t, squared_distance[selected] = self.project(i, q)
            near_u[selected] = self.lines[i][j, 3]*(1-t) + self.lines[i][j+1, 3]*t
            near_z = self.lines[i][j, 2]*(1-t) + self.lines[i][j+1, 2]*t
            height_offset[selected] = (self.heights[i][j]*(1-t) + self.heights[i][j+1]*t
                                       - self.row_reference(near_z))
        weights /= np.maximum(squared_distance, 1e-20)
        weights /= np.bincount(rows, weights=weights, minlength=len(points))[rows]
        matrix = sparse.coo_matrix((weights, (rows, cols)), shape=(len(points), n)).tocsr()
        base_u = points[:, 3] - np.bincount(rows, weights=weights*near_u, minlength=len(points))
        base_v = self.row_reference(points[:, 2]) + np.bincount(rows, weights=weights*height_offset, minlength=len(points))
        base = np.column_stack((base_u, base_v))
        if with_distance:
            nearest = np.full(len(points), np.inf)
            np.minimum.at(nearest, rows, squared_distance)
            return matrix, base, np.sqrt(nearest)
        return matrix, base


class StraighteningMap:
    def __init__(self, guides, centers):
        self.guides = guides
        self.centers = np.asarray(centers)
        self.tree = cKDTree(self.centers)
        self.columns = np.column_stack((guides.initial_columns, np.zeros(len(guides.lines))))
        self.correction = np.zeros((len(centers), 2))

    def free_basis(self, points, near=None):
        query = coordinates(points)
        k = min(24, len(self.centers))
        distance, indices = self.tree.query(query, k=list(range(1, k+1)), workers=4)
        radius = np.maximum(distance[:, -1, None]*(1+1e-6), 1e-10)
        r = np.minimum(distance/radius, 1.)
        weights = (1-r)**4 * (4*r+1)
        weights /= weights.sum(axis=1)[:, None]
        if near is None:
            _, _, near = self.guides.basis(points, with_distance=True)
        # Correction vanishes on the full guide polylines, not just vertices.
        weights *= (near**2 / (near**2 + 50**2))[:, None]
        return sparse.coo_matrix((weights.ravel(), (np.repeat(np.arange(len(points)), k), indices.ravel())),
                                 shape=(len(points), len(self.centers))).tocsr()

    def basis(self, points):
        guide, base, near = self.guides.basis(points, with_distance=True)
        return sparse.hstack((guide, self.free_basis(points, near)), format='csr'), base

    def __call__(self, points):
        matrix, base = self.basis(points)
        return base + matrix @ np.concatenate((self.columns, self.correction))


class SourceHeightReference:
    """Only a soft source-Z reference; no inferred vertical guide constraints."""
    def __init__(self, z):
        lo, hi = float(np.min(z)), float(np.max(z))
        self.row_z = np.array([lo, max(hi, lo+1.)])
        self.row_v = self.row_z.copy()
        self.lines, self.heights = [], []

    def row_reference(self, z):
        return np.asarray(z, dtype=float)


def collect_guides(payload, sources, *, allow_source_z_reference=False):
    # Use the same manual/automatic/geometric precedence as fiber linking.
    from spiral_helpers import classify_fiber_hv
    lines, ids = [], []
    missing = 0
    for source in sources:
        if not source['fiber'] or source.get('quarantined', False):
            continue
        line = source_grid(payload, source)[1]
        direction = source.get('direction')
        if direction not in ('H', 'V'):
            classification = source.get('hv_classification')
            if classification is None:
                try:
                    classification = json.loads(Path(source['path']).read_text()).get('hv_classification')
                except (OSError, ValueError):
                    missing += 1
            direction = classify_fiber_hv(classification, line[:, :3][:, ::-1],
                                          min_z_fraction=.8, min_auto_certainty=.5)
            source['direction'] = direction
        if direction == 'V':
            lines.append(line.copy())
            ids.append(source['id'])
    if not lines and allow_source_z_reference:
        return SourceHeightReference(payload[payload[:, 0] != -1, 2]), ids, missing
    return VerticalGuides(lines), ids, missing


def sample_triangles(payload, sources, samples=4, interior=False, with_orientation=False):
    chunks, orientations = [], []
    for source in sources:
        if source['fiber']:
            continue
        grid = source_grid(payload, source)
        a, b, c = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1]
        idx = np.flatnonzero((a[..., 0] != -1) & (b[..., 0] != -1) & (c[..., 0] != -1))
        if len(idx):
            # Grid connectivity carries orientation through a physical S-bend.
            # A local sign change in the old theta/Z projection can be the
            # fold we need to unfold, not a reflection to preserve.
            uv1, uv2 = b[..., 3:]-a[..., 3:], c[..., 3:]-a[..., 3:]
            area = (uv1[..., 0]*uv2[..., 1]-uv1[..., 1]*uv2[..., 0]).ravel()[idx]
            nonzero = area[np.abs(area) > 1e-8]
            orientation = np.sign(np.median(nonzero)) if len(nonzero) else 1.
            if orientation == 0:
                orientation = np.sign(nonzero[np.argmax(np.abs(nonzero))])
            count = min(len(idx), samples)
            positions = ((np.arange(count)+.5)/count*(len(idx)-1)).astype(int) if interior else np.linspace(0, len(idx)-1, count, dtype=int)
            idx = idx[positions]
            chunks.append(np.stack([v.reshape(-1, 5)[idx] for v in (a, b, c)], axis=1))
            orientations.append(np.full(len(idx), orientation))
    if not chunks:
        raise ValueError('Straightening needs patch triangles to constrain spacing')
    result = np.concatenate(chunks)
    return (result, np.concatenate(orientations)) if with_orientation else result


def triangle_frames(triangles, orientation=None):
    d1 = triangles[:, 1, :3] - triangles[:, 0, :3]
    d2 = triangles[:, 2, :3] - triangles[:, 0, :3]
    length = np.linalg.norm(d1, axis=1)
    along = np.sum(d1*d2, axis=1) / np.maximum(length, 1e-20)
    across = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(length, 1e-20)
    if orientation is None:
        u1 = triangles[:, 1, 3:] - triangles[:, 0, 3:]
        u2 = triangles[:, 2, 3:] - triangles[:, 0, 3:]
        orientation = u1[:, 0]*u2[:, 1] - u1[:, 1]*u2[:, 0]
    valid = (length > 1e-6) & (across > 1e-6) & (np.abs(orientation) > 1e-8)
    return valid, length[valid], along[valid], across[valid]*np.sign(orientation[valid])


def distortion_report(triangles, uv, orientation=None):
    valid, length, along, across = triangle_frames(triangles, orientation)
    if not valid.any():
        raise ValueError('No usable triangles for distortion validation')
    uv = uv[valid]
    dx = (uv[:, 1]-uv[:, 0])/length[:, None]
    dy = ((uv[:, 2]-uv[:, 0])-dx*along[:, None])/across[:, None]
    jacobian = np.stack((dx, dy), axis=2)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return dict(triangles=int(valid.sum()), inverted=int(np.count_nonzero(np.linalg.det(jacobian) <= 0)),
                singular_value_percentiles=np.percentile(singular, [1, 5, 50, 95, 99]).tolist(),
                area_ratio_percentiles=np.percentile(np.prod(singular,axis=1), [1, 5, 50, 95, 99]).tolist(),
                scale_fraction_within_20_percent=float(np.mean((singular>=.8)&(singular<=1.2))),
                area_fraction_within_20_percent=float(np.mean((np.prod(singular,axis=1)>=.8)&(np.prod(singular,axis=1)<=1.2))),
                mean_squared_log_stretch=float(np.mean(np.log(np.maximum(singular, 1e-12))**2)))


def validate_straightening(model, triangles, orientation=None):
    """Evaluate independent triangles before changing any workspace placement."""
    flat = triangles.reshape(-1, 5)
    uv = np.concatenate([model(flat[i:i+20000]) for i in range(0, len(flat), 20000)]).reshape(-1, 3, 2)
    before = distortion_report(triangles, triangles[:, :, 3:], orientation)
    after = distortion_report(triangles, uv, orientation)
    return dict(before=before, after=after,
                accepted=bool(after['inverted'] == 0 and
                              after['mean_squared_log_stretch'] <= before['mean_squared_log_stretch']+.01))


def fit_straightening(guides, triangles, row_band=40., iterations=50, progress=print,
                     orientation_iterations=500, row_slope_weight=1., orientation=None):
    """Fit near-isometric triangle frames while retaining approximate Z rows.

    This is a constrained least-squares fit, not a certification of injectivity.
    The caller must inspect the returned inversion/distortion diagnostics before
    publishing a workspace.
    """
    if (not np.isfinite(row_band) or row_band < 0 or iterations < 1 or orientation_iterations < 0
            or not np.isfinite(row_slope_weight) or row_slope_weight < 0):
        raise ValueError('Invalid straightening fit settings')
    valid, length, along, across = triangle_frames(triangles, orientation)
    triangles = triangles[valid]
    if not len(triangles):
        raise ValueError('No usable patch triangles for straightening')
    flat = triangles.reshape(-1, 5)
    coord = coordinates(flat)
    _, ids = np.unique(np.floor(coord/192).astype(np.int64), axis=0, return_index=True)
    model = StraighteningMap(guides, coord[ids])
    matrices, bases = [], []
    for start in range(0, len(flat), 20000):
        matrix, base = model.basis(flat[start:start+20000])
        matrices.append(matrix)
        bases.append(base + matrix[:, :len(guides.lines)] @ model.columns)
    matrix = sparse.vstack(matrices, format='csr')
    base = np.concatenate(bases).reshape(-1, 3, 2)
    dx = sparse.diags(1/length) @ (matrix[1::3]-matrix[0::3])
    dy = sparse.diags(1/across) @ ((matrix[2::3]-matrix[0::3])-sparse.diags(along)@dx)
    sx = (base[:, 1]-base[:, 0])/length[:, None]
    sy = ((base[:, 2]-base[:, 0])-sx*along[:, None])/across[:, None]
    operator = sparse.vstack((dx, dy), format='csr')
    initial = np.concatenate((sx, sy))
    reference = guides.row_reference(triangles[:, :, 2])
    zx = (reference[:, 1]-reference[:, 0])/length
    zy = ((reference[:, 2]-reference[:, 0])-zx*along)/across
    source_slope = np.concatenate((zx, zy))
    initial_row_drift = base[:, :, 1].ravel() - reference.ravel()
    row_weights = guides.row_weights(flat)
    slope_weights = np.tile(row_weights.reshape(-1, 3).min(axis=1), 2)
    row_operator = sparse.diags(row_weights)@matrix
    slope_operator = sparse.diags(slope_weights)@operator
    count = len(model.centers)
    k = min(7, count)
    neighbor_distances, neighbors = model.tree.query(model.centers, k=list(range(1, k+1)))
    a = np.repeat(np.arange(count), k-1)
    b = neighbors[:, 1:].ravel()
    scale = 1/np.maximum(neighbor_distances[:, 1:].ravel(), 1e-10)
    differences = sparse.coo_matrix((np.column_stack((-scale, scale)).ravel(),
                                    (np.repeat(np.arange(len(a)), 2), np.column_stack((a, b)).ravel())),
                                   shape=(len(a), count)).tocsr()
    regularizer = sparse.block_diag((sparse.eye(len(guides.lines))*1e-8,
                                     (differences.T@differences)*.1+sparse.eye(count)*1e-8), format='csc')
    contacts = guides.contacts()
    contact_matrix = sparse.coo_matrix((np.tile([1., -1.], len(contacts)),
                                        (np.repeat(np.arange(len(contacts)), 2), np.asarray(contacts, dtype=int).ravel())),
                                       shape=(len(contacts), operator.shape[1])).tocsr()
    contact_start = contact_matrix[:, :len(guides.lines)]@model.columns
    contact_rhs = 100.*contact_matrix.T@(guides.contact_targets(contacts)-contact_start)
    regularizer += 100.*(contact_matrix.T@contact_matrix).tocsc()
    progress(f'Fitting {len(guides.lines)} verticals with {len(triangles)} patch triangles and {count} deformation controls')
    normal = (operator.T@operator).tocsc() + regularizer
    solve_u = factorized(normal)
    solve_v = factorized(normal + row_slope_weight*(operator.T@slope_operator).tocsc()
                        + .01*(matrix.T@row_operator).tocsc())
    row_slope_rhs = row_slope_weight*slope_operator.T@(source_slope-initial[:, 1])
    solution = np.zeros((operator.shape[1], 2))
    value = initial.copy()
    nf = len(triangles)
    for iteration in range(iterations):
        jacobian = np.stack((value[:nf], value[nf:]), axis=2)
        u, _, vt = np.linalg.svd(jacobian)
        u[:, :, -1] *= np.linalg.det(u@vt)[:, None]
        rotation = u@vt
        rhs = operator.T @ (np.concatenate((rotation[:, :, 0], rotation[:, :, 1]))-initial) + contact_rhs
        solution[:, 0] = solve_u(rhs[:, 0])
        drift = initial_row_drift + matrix @ solution[:, 1]
        solution[:, 1] = solve_v(rhs[:, 1] + row_slope_rhs
                                + .01*row_operator.T@(np.clip(drift, -row_band, row_band)-initial_row_drift))
        value = initial + operator@solution
        if iteration % 10 == 0:
            progress(f'Straightening iteration {iteration+1}/{iterations}')
    # Jointly correct U and V: fixing row coordinates while repairing area can
    # force extreme horizontal stretching. The local projection bounds both
    # physical stretches and excludes reflection, while the shared global fit
    # retains exact guide columns and the soft row reference.
    jacobian = np.stack((value[:nf], value[nf:]), axis=2)
    if np.any(np.linalg.det(jacobian) <= 0) and orientation_iterations:
        progress('Correcting local orientation and stretch with both coordinates free')
        dual = np.zeros_like(jacobian)
        best_solution = solution.copy()
        best_inversions = np.count_nonzero(np.linalg.det(jacobian) <= 0)
        for iteration in range(orientation_iterations):
            u, singular, vt = np.linalg.svd(jacobian+dual)
            sign = np.linalg.det(u@vt)
            u[:, :, -1] *= sign[:, None]
            singular[:, -1] *= sign
            singular = np.clip((singular+1.)/2., .25, 4.)
            projected = (u*singular[:, None, :])@vt
            target = projected-dual
            rhs = operator.T@(np.concatenate((target[:, :, 0], target[:, :, 1]))-initial)+contact_rhs
            solution[:, 0] = solve_u(rhs[:, 0])
            drift = initial_row_drift + matrix@solution[:, 1]
            solution[:, 1] = solve_v(rhs[:, 1] + row_slope_rhs
                                    + .01*row_operator.T@(np.clip(drift, -row_band, row_band)-initial_row_drift))
            value = initial + operator@solution
            jacobian = np.stack((value[:nf], value[nf:]), axis=2)
            dual += jacobian-projected
            area = np.linalg.det(jacobian)
            inversions = np.count_nonzero(area <= 0)
            if inversions < best_inversions:
                best_inversions = inversions
                best_solution = solution.copy()
            if iteration % 50 == 0:
                progress(f'Orientation iteration {iteration+1}/{orientation_iterations}: '
                         f'{np.count_nonzero(area <= 0)} inverted triangles, minimum area ratio {area.min():.4g}')
            if iteration >= 50 and area.min() >= .05 and np.max(np.abs(jacobian-projected)) < .01:
                break
            if iteration >= 50 and inversions > max(1, best_inversions)*1.25:
                progress('Orientation correction did not converge; retaining the best candidate for validation')
                break
        solution = best_solution
        value = initial + operator@solution
    model.columns += solution[:len(guides.lines)]
    model.correction = solution[len(guides.lines):]
    jacobian = np.stack((value[:nf], value[nf:]), axis=2)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    report = dict(vertical_fibers=len(guides.lines), training_triangles=nf, deformation_controls=count,
                  vertical_contact_pairs=contacts,
                  row_slope_weight=row_slope_weight,
                  row_band=row_band, row_drift_percentiles=np.percentile(np.abs(initial_row_drift+matrix@solution[:, 1]), [50, 95, 100]).tolist(),
                  singular_value_percentiles=np.percentile(singular, [1, 5, 50, 95, 99]).tolist(),
                  inverted_training_triangles=int(np.count_nonzero(np.linalg.det(jacobian) <= 0)))
    return model, report
