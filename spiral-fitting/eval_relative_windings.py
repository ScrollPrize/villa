"""Score a relative-windings annotation against hand-traced per-wrap truth meshes.

Truth meshes are tifxyz directories named ``w<index>.tifxyz``; the index is the
winding. A PointCollections JSON v1 file (``relative_windings.json``) carries
``wind_a`` per point, of which only differences inside one collection mean
anything. Each point is linked to the nearest truth mesh within ``tolerance``
voxels (point ``p`` is [x, y, z]; tifxyz vertices are stored in full-resolution
voxel zyx, ``scale`` only describes the grid sampling, so no rescaling is done).
Consecutive linked points (ordered by ``creation_time``, ties and missing
times keep file order) give a predicted step ``d_pred = round(wind_a[j] - wind_a[i])``
(``wind_a`` is nominally an integer; a fractional value is rounded, not truncated)
and a true step ``d_true = w[j] - w[i]``.

Linking uses a batched nearest-mesh ``Patch.project`` rather than
``point_collection.link_points_to_patches``, which projects one point at a time
(minutes on 1M-vertex meshes).

theta convention (for ``--exclude``): theta = degrees(atan2(y - uy, x - ux)) mod 360,
with (ux, uy) the umbilicus interpolated at the point's z. UNVERIFIED against
VC3D's own angle convention.

``--make-control`` writes a synthetic annotation from the meshes themselves.
"""
import argparse
import json
import math
import os
import re

import numpy as np
import torch

from point_collection import load_point_collection
from tifxyz import Patch, load_tifxyz

SEED = 343
N_BOOT = 2000


def load_truth_meshes(directory):
    meshes = []
    for name in sorted(os.listdir(directory)):
        m = re.fullmatch(r'w(\d+)\.tifxyz', name)
        if m and os.path.isdir(os.path.join(directory, name)):
            meshes.append((int(m.group(1)), load_tifxyz(os.path.join(directory, name))))
    if not meshes:
        raise ValueError(f'no w<index>.tifxyz directories in {directory}')
    return sorted(meshes, key=lambda m: m[0])


def max_face_edge(mesh):
    """Longest edge over the valid triangles of a patch (both quad diagonals included)."""
    v, q = mesh.zyxs.to(torch.float32), mesh.valid_quad_mask
    tl, tr, bl, br = v[:-1, :-1][q], v[:-1, 1:][q], v[1:, :-1][q], v[1:, 1:][q]
    edges = [tl - bl, tr - tl, tr - bl, br - tr, br - bl]
    return float(max(torch.linalg.norm(e, dim=-1).max() for e in edges))


def project_near(mesh, zyx, reach, chunk_size=8):
    """Distance from each point to the patch, exact wherever it is <= reach - longest face edge, else
    >= that bound (inf when no vertex lies within reach). A face within distance t of a point has every
    vertex within t + longest edge of it, so projecting onto the grid window spanned by the vertices
    inside that ball (plus one cell) finds every such face; points with no vertex in the ball are skipped."""
    from scipy.spatial import cKDTree
    ij = mesh.valid_vertex_indices.numpy()
    tree = cKDTree(mesh.zyxs[mesh.valid_vertex_mask].to(torch.float64).numpy())
    h, w = mesh.zyxs.shape[:2]
    out = torch.full((len(zyx),), float('inf'))
    for n, near in enumerate(tree.query_ball_point(zyx.to(torch.float64).numpy(), reach)):
        if not near:
            continue
        (i0, j0), (i1, j1) = ij[near].min(0), ij[near].max(0)
        win = mesh.zyxs[max(i0 - 1, 0):min(i1 + 2, h), max(j0 - 1, 0):min(j1 + 2, w)]
        valid = torch.any(win != -1, dim=-1)
        if not bool((valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]).any()):
            continue
        sub = Patch(win.clone(), mesh.scale, None, None)
        out[n] = sub.project(zyx[n:n + 1], chunk_size=chunk_size)[1][0]
    return out


def link_points(points, meshes, tolerance, chunk_size=8, prefilter=True):
    """Winding index of the nearest mesh within tolerance for each (N, 3) xyz point, else None.

    prefilter: project each point only onto the grid window of its nearby vertices (KD-tree), see
    project_near. Same links as the brute-force projection over every face (prefilter=False)."""
    zyx = torch.as_tensor(np.asarray(points, dtype=np.float32).reshape(-1, 3)[:, ::-1].copy())
    if prefilter:
        dists = torch.stack([project_near(m, zyx, tolerance + max_face_edge(m), chunk_size)
                             for _, m in meshes])  # (M, N), exact where <= tolerance
    else:
        dists = torch.stack([m.project(zyx, chunk_size=chunk_size)[1] for _, m in meshes])  # (M, N)
    best_d, best_i = dists.min(dim=0)
    return [meshes[int(i)][0] if float(d) <= tolerance else None
            for d, i in zip(best_d, best_i)]


def umbilicus_theta(xyz, umbilicus):
    cps = sorted(umbilicus['control_points'], key=lambda c: c['z'])
    zs = [c['z'] for c in cps]
    ux = np.interp(xyz[2], zs, [c['x'] for c in cps])
    uy = np.interp(xyz[2], zs, [c['y'] for c in cps])
    return math.degrees(math.atan2(xyz[1] - uy, xyz[0] - ux)) % 360.0


def in_sector(theta, lo, hi):
    lo, hi = lo % 360.0, hi % 360.0
    return lo <= theta <= hi if lo <= hi else (theta >= lo or theta <= hi)


def score_collection(points):
    """points: dicts with winding_annotation, truth (int|None), creation_time, excluded (bool)."""
    order = sorted(range(len(points)), key=lambda i: (points[i].get('creation_time', 0), i))
    pts = [points[i] for i in order]
    ok = [p['truth'] is not None and not p.get('excluded') and np.isfinite(p['winding_annotation'])
          for p in pts]
    exact = l1 = sign = pairs = 0
    for a, b, oa, ob in zip(pts, pts[1:], ok, ok[1:]):
        if not (oa and ob):
            continue
        # round, not truncate: int(1.9 - 0.0) would read a 1.9-turn step as 1
        d_pred = int(round(b['winding_annotation'] - a['winding_annotation']))
        d_true = b['truth'] - a['truth']
        pairs += 1
        exact += d_pred == d_true
        l1 += abs(d_pred - d_true)
        sign += d_pred * d_true < 0
    return {'n_pairs': pairs, 'n_exact': int(exact), 'sum_abs_err': int(l1),
            'sign_violations': int(sign), 'n_points': len(pts),
            'n_unlinked': sum(1 for p in pts if p['truth'] is None and not p.get('excluded')),
            'n_excluded': sum(1 for p in pts if p.get('excluded'))}


def _rates(rows):
    n = sum(r['n_pairs'] for r in rows)
    if n == 0:
        return float('nan'), float('nan')
    return sum(r['n_exact'] for r in rows) / n, sum(r['sum_abs_err'] for r in rows) / n


def pooled_with_interval(rows, seed=SEED, n_boot=N_BOOT):
    exact, l1 = _rates(rows)
    rng = np.random.default_rng(seed)
    boots = np.array([_rates([rows[i] for i in rng.integers(0, len(rows), len(rows))])
                      for _ in range(n_boot)])
    lo_hi = lambda c: [float(np.nanpercentile(boots[:, c], 2.5)), float(np.nanpercentile(boots[:, c], 97.5))]
    return {'exact_step_share': exact, 'exact_step_share_ci95': lo_hi(0),
            'mean_abs_err': l1, 'mean_abs_err_ci95': lo_hi(1),
            'sign_violations': sum(r['sign_violations'] for r in rows),
            'n_pairs': sum(r['n_pairs'] for r in rows),
            'n_unlinked': sum(r['n_unlinked'] for r in rows),
            'n_excluded': sum(r['n_excluded'] for r in rows),
            'n_collections': len(rows), 'seed': seed, 'n_boot': n_boot}


def evaluate(collections, meshes, tolerance=10.0, umbilicus=None, exclude=None, prefilter=True):
    """collections: dict from load_point_collection. Returns the result dict."""
    rows = []
    all_pts = {cid: [dict(p) for p in col['points'].values()] for cid, col in collections.items()}  # file order
    flat = [p['p'] for pts in all_pts.values() for p in pts]
    linked = iter(link_points(flat, meshes, tolerance, prefilter=prefilter) if flat else [])  # one pass: one KD-tree per mesh
    for cid, pts in all_pts.items():
        truth = [next(linked) for _ in pts]
        for p, t in zip(pts, truth):
            p['truth'] = t
            p['excluded'] = bool(t is not None and any(
                e['winding'] == t and in_sector(umbilicus_theta(p['p'], umbilicus), *e['theta_deg'])
                for e in (exclude or [])))
        rows.append({'collection_id': cid, **score_collection(pts)})
    result = pooled_with_interval(rows)
    result.update({'tolerance': tolerance, 'exclude': exclude or [],
                   'theta_convention': 'degrees(atan2(y-uy, x-ux)) mod 360, umbilicus interpolated at point z (unverified vs VC3D)',
                   'per_collection': rows})
    return result


def make_control(meshes, kind, n_per_mesh=60, seed=SEED):
    """Synthetic relative_windings.json dict. Points are mesh vertices; wind_a = mesh index."""
    rng = np.random.default_rng(seed)
    pool = {w: [] for w, _ in meshes}
    for w, m in meshes:
        idx = m.valid_vertex_indices
        for k in rng.choice(len(idx), n_per_mesh, replace=False):
            i, j = idx[int(k)].tolist()
            z, y, x = m.zyxs[i, j].tolist()
            pool[w].append([x, y, z])
    windings = [w for w, _ in meshes]
    collections, cid = {}, 0
    while any(pool.values()):
        length = int(rng.integers(4, 7))
        seq, left = [], {w: len(v) for w, v in pool.items()}
        for w in (int(x) for x in rng.choice(windings, length)):
            if left[w]:
                left[w] -= 1
                seq.append(w)
        if len(seq) < 2:
            for w in windings:
                pool[w].clear()
            break
        shift_at = int(rng.integers(1, len(seq))) if kind == 'negative' and rng.random() < 0.3 else None
        points = {str(k): {'p': pool[int(w)].pop(), 'wind_a': int(w) + (1 if shift_at is not None and k >= shift_at else 0),
                           'creation_time': 1000 + k} for k, w in enumerate(seq)}
        collections[str(cid)] = {'name': f'control_{kind}_{cid}', 'points': points}
        cid += 1
    return {'vc_pointcollections_json_version': '1', 'collections': collections}


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    ap.add_argument('--meshes', required=True)
    ap.add_argument('--collection', required=True, help='input file, or output path with --make-control')
    ap.add_argument('--make-control', choices=['positive', 'negative'])
    ap.add_argument('--n-per-mesh', type=int, default=60)
    ap.add_argument('--umbilicus')
    ap.add_argument('--exclude')
    ap.add_argument('--tolerance', type=float, default=10.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-prefilter', action='store_true',
                    help='brute-force projection over every face (slow; same links)')
    args = ap.parse_args()

    meshes = load_truth_meshes(args.meshes)
    if args.make_control:
        with open(args.collection, 'w') as f:
            json.dump(make_control(meshes, args.make_control, args.n_per_mesh), f)
    exclude = json.load(open(args.exclude)) if args.exclude else None
    umbilicus = json.load(open(args.umbilicus)) if args.umbilicus else None
    if exclude and not umbilicus:
        ap.error('--exclude needs --umbilicus')
    collections = load_point_collection(args.collection)
    if collections is None:
        raise SystemExit('could not load point collection')
    result = evaluate(collections, meshes, args.tolerance, umbilicus, exclude, prefilter=not args.no_prefilter)
    result['mesh_windings'] = [w for w, _ in meshes]
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    print(f"exact_step_share {result['exact_step_share']:.4f} CI95 {result['exact_step_share_ci95']}  "
          f"mean_abs_err {result['mean_abs_err']:.4f} CI95 {result['mean_abs_err_ci95']}  "
          f"pairs {result['n_pairs']} sign_viol {result['sign_violations']} unlinked {result['n_unlinked']}")


if __name__ == '__main__':
    main()
