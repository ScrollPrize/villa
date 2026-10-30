"""Checkpoint-free check of relative-winding annotations (point-collection JSONs) before a spiral fit.

For every relative-winding step (consecutive points of one collection, in placement order) it reports:
  * the radial step per winding relative to the umbilicus, against the global median and a local null (median
    per-winding step of other collections' steps nearby in z, theta and radius);
  * for each binary surface-prediction zarr given with --pred (in the annotation voxel frame), the number of
    predicted sheets crossed along the step's 3D segment: foreground runs along the segment, endpoints snapped
    to their run (<= --snap-tol voxels); an off-sheet endpoint gets the half index of the gap it is in;
    sheets crossed = |index difference| (a half-integer when exactly one endpoint is off-sheet);
  * for steps with delta != 0: `over` / `under` (every prediction counts at least one sheet more / fewer than
    |delta|), `over_strict` (over, and every endpoint snapped in every prediction) and `suggested_delta`
    (minimum count over predictions for strict over-counts). Delta-0 steps get counts but no flags.
Cross-collection consistency (loops, absolute anchors via patch attachment) is left to find_inconsistent_windings.py.

Only relative collections (wind_a set, not winding_is_absolute) are checked; other --pcl files are loaded and ignored.
Example (PHercParis4 spiral inputs, 9.6 um frame):
  python check_winding_annotations.py --umbilicus umbilicus.json --pcl relative_windings.json \
      --pred https://.../surface-m7-L2-th0.2.zarr/0 --pred https://.../surface-recto-2um-ps256-L0-th0.45.zarr/2 \
      --output winding_check.json
No checkpoint, patches or GPU are needed; only the prediction chunks under the annotated segments are read.
"""
import collections
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

import click
import numpy as np
import zarr

from point_collection import load_point_collection
from umbilicus import json_umbilicus_z_to_yx

SAMPLES_PER_VOXEL = 4


def classify(collection):
    if (collection.get('metadata') or {}).get('winding_is_absolute', False):
        return 'abs'
    if all(math.isnan(p['winding_annotation']) for p in collection['points'].values()):
        return 'same'
    return 'rel'


def ordered_points(collection):
    """Points in placement order (creation_time, then id), as (xyz array, winding, point ids)."""
    pts = sorted(collection['points'].values(), key=lambda p: (p['creation_time'], p['id']))
    return (np.array([p['p'] for p in pts], float).reshape(-1, 3), [p['winding_annotation'] for p in pts],
            [p['id'] for p in pts])


def load_chains(pcl_paths):
    chains = []
    for path in pcl_paths:
        cols = load_point_collection(path)
        if cols is None:
            raise click.ClickException(f'could not load point collection {path}')
        for cid, c in cols.items():
            kind = classify(c)
            xyz, w, pid = ordered_points(c)
            if kind == 'rel':
                keep = [i for i, x in enumerate(w) if not math.isnan(x)]
                xyz, w, pid = xyz[keep], [w[i] for i in keep], [pid[i] for i in keep]
            if len(xyz):
                chains.append(dict(file=path, id=cid, name=c['name'], kind=kind, xyz=xyz, w=w, pid=pid))
    return chains


def polar(umb, xyz):
    yx = umb(xyz[:, 2])
    d = xyz[:, :2] - yx[:, ::-1]
    return np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0])


def radial_checks(steps, umb, dz=300.0, dtheta=0.35, dr=200.0, min_refs=5):
    """Per-winding radial step of each nonzero step, global robust z-score and local-null ratio."""
    if not steps:
        return
    a = np.array([s['p0'] for s in steps])
    b = np.array([s['p1'] for s in steps])
    ra, ta = polar(umb, a)
    rb, _ = polar(umb, b)
    d = np.array([s['delta'] for s in steps], float)
    per = (rb - ra) / d
    sense = 1.0 if np.median(per) >= 0 else -1.0
    per = per * sense
    med = np.median(per)
    mad = np.median(np.abs(per - med)) * 1.4826 + 1e-9
    mz, mr = (a[:, 2] + b[:, 2]) / 2, (ra + rb) / 2
    owner = np.array([s['chain'] for s in steps])
    for i, s in enumerate(steps):
        th = np.abs(np.angle(np.exp(1j * (ta - ta[i]))))  # wrapped: no branch cut at theta = +-pi
        nb = (np.abs(mz - mz[i]) < dz) & (th < dtheta) & (np.abs(mr - mr[i]) < dr) & (owner != owner[i])
        local = float(np.median(per[nb])) if nb.sum() >= min_refs else None
        s.update(per_winding=round(float(per[i]), 2), global_suspect=bool(per[i] <= 0 or abs(per[i] - med) / mad > 4),
                 local_spacing=None if local is None else round(local, 2),
                 # two-sided: more than 2.5x or less than 1/2.5 of the local spacing; no verdict if local <= 0
                 local_suspect=bool(local is not None and local > 0 and (per[i] <= 0 or not 1 / 2.5 <= per[i] / local <= 2.5)))


class Prediction:
    """Binary surface prediction (zarr array, z/y/x) read chunk by chunk; chunks kept bit-packed in memory."""

    def __init__(self, url, threshold=None, threads=16):
        self.url = url
        self.arr = zarr.open_array(url, mode='r')
        self.shape = self.arr.shape
        self.c = self.arr.chunks
        self.threads = threads
        # dtype-aware default: binary 0/1 or 0/255 integer volumes -> > 0; float probabilities -> > 0.5
        self.threshold = threshold if threshold is not None else (0.5 if np.dtype(self.arr.dtype).kind == 'f' else 0)
        self.warned = threshold is not None or np.dtype(self.arr.dtype).kind == 'f'
        self.mem = {}

    def _read(self, k):
        sl = tuple(slice(k[i] * self.c[i], min((k[i] + 1) * self.c[i], self.shape[i])) for i in range(3))
        block = np.zeros(self.c, bool)
        raw = self.arr[sl]
        if not self.warned and len(np.unique(raw)) > 2:  # integer default (> 0) assumes a binary volume
            self.warned = True
            click.echo(f'WARNING: {self.url} has more than 2 values in a chunk; default threshold > 0 assumes a binary '
                       'prediction, pass --threshold for probability maps', err=True)
        a = raw > self.threshold
        block[:a.shape[0], :a.shape[1], :a.shape[2]] = a
        return k, np.packbits(block, axis=2)

    def prefetch(self, zyx):
        keys = set(map(tuple, (zyx // np.array(self.c)).tolist())) - set(self.mem)
        keys = [k for k in keys if all(0 <= k[i] * self.c[i] < self.shape[i] for i in range(3))]
        with ThreadPoolExecutor(self.threads) as ex:
            for k, v in ex.map(self._read, sorted(keys)):
                self.mem[k] = v
        return len(keys)

    def sample(self, zyx):
        out = np.zeros(len(zyx), bool)
        ck = zyx // np.array(self.c)
        loc = zyx - ck * np.array(self.c)
        for k in set(map(tuple, ck.tolist())):
            m = np.all(ck == k, axis=1)
            block = self.mem.get(k)
            if block is not None:
                l = loc[m]
                out[m] = (block[l[:, 0], l[:, 1], l[:, 2] >> 3] >> (7 - (l[:, 2] & 7))) & 1
        return out


def chain_path(xyz, pad):
    """3D polyline samples (extended `pad` voxels past both ends) and the sample index of each point."""
    d0, d1 = xyz[1] - xyz[0], xyz[-1] - xyz[-2]
    q = np.vstack([xyz[0] - pad * d0 / (np.linalg.norm(d0) + 1e-9), xyz, xyz[-1] + pad * d1 / (np.linalg.norm(d1) + 1e-9)])
    parts = []
    for i in range(len(q) - 1):
        n = max(1, int(np.ceil(np.linalg.norm(q[i + 1] - q[i]) * SAMPLES_PER_VOXEL)))
        t = np.arange(n) / n
        parts.append(q[i][None] + t[:, None] * (q[i + 1] - q[i])[None])
    starts = np.cumsum([0] + [len(p) for p in parts])
    return np.vstack(parts), [int(starts[i]) for i in range(1, len(xyz) + 1)]


def foreground_runs(fg):
    """(start, end) sample ranges of foreground after closing background gaps shorter than SAMPLES_PER_VOXEL samples (one voxel)."""
    fg = fg.copy()
    edges = np.r_[0, np.flatnonzero(np.diff(fg.astype(np.int8))) + 1, len(fg)]
    for j in range(1, len(edges) - 2):
        s, e = edges[j], edges[j + 1]
        if not fg[s] and e - s < SAMPLES_PER_VOXEL:
            fg[s:e] = True
    edges = np.r_[0, np.flatnonzero(np.diff(fg.astype(np.int8))) + 1, len(fg)]
    return [(edges[j], edges[j + 1]) for j in range(len(edges) - 1) if fg[edges[j]] and edges[j + 1] - edges[j] >= 2]


def sheets_crossed(fg, idx, snap_tol):
    """Per step: (sheets crossed, both endpoints snapped). Endpoint index = its run index if snapped, else
    (number of run centres before it) - 0.5, i.e. the gap it lies in; crossed = |index difference|."""
    runs = foreground_runs(fg)
    centres = np.array([(a + b - 1) / 2 for a, b in runs]) if runs else np.zeros(0)
    snap = []
    for s in idx:
        hit = [k for k, (a, b) in enumerate(runs) if a <= s < b]
        if not hit and runs:
            dist = [min(abs(s - a), abs(s - (b - 1))) for a, b in runs]
            k = int(np.argmin(dist))
            hit = [k] if dist[k] <= snap_tol * SAMPLES_PER_VOXEL else []
        snap.append(hit[0] if hit else None)
    pos = [float(j) if j is not None else float((centres < s).sum()) - 0.5 for s, j in zip(idx, snap)]
    out = []
    for i in range(len(idx) - 1):
        out.append((abs(pos[i + 1] - pos[i]), snap[i] is not None and snap[i + 1] is not None))
    return out


def check(chains, umb, preds, pad=6.0, snap_tol=4.0):
    rel = [k for k, c in enumerate(chains) if c['kind'] == 'rel' and len(c['xyz']) >= 2]
    steps = []
    for k in rel:
        c = chains[k]
        for i in range(len(c['xyz']) - 1):
            steps.append(dict(collection=c['name'], file=c['file'], id=c['id'], chain=k, step=i, point0=c['pid'][i], point1=c['pid'][i + 1], w0=c['w'][i], w1=c['w'][i + 1],
                              delta=int(round(c['w'][i + 1] - c['w'][i])), p0=c['xyz'][i], p1=c['xyz'][i + 1],
                              length=round(float(np.linalg.norm(c['xyz'][i + 1] - c['xyz'][i])), 1), cross=[]))
    radial_checks([s for s in steps if s['delta'] != 0], umb)
    paths = {k: chain_path(chains[k]['xyz'], pad) for k in rel}
    by_chain = collections.defaultdict(list)
    for s in steps:
        by_chain[s['chain']].append(s)
    pred_meta = []
    allz = np.vstack([np.rint(paths[k][0][:, ::-1]).astype(int) for k in rel]) if rel else np.zeros((0, 3), int)
    mx = np.max(np.vstack([chains[k]['xyz'] for k in rel]), axis=0) if rel else np.zeros(3)
    mn = np.min(np.vstack([chains[k]['xyz'] for k in rel]), axis=0) if rel else np.zeros(3)
    for p in preds:
        # bounds check only: annotation coordinates must fall inside the prediction array (z, y, x)
        frame_ok = bool(mn.min() >= 0 and mx[2] < p.shape[0] and mx[1] < p.shape[1] and mx[0] < p.shape[2])
        t0 = time.time()
        n_chunks = p.prefetch(allz)
        snapped = total = 0
        for k in rel:
            xyz, idx = paths[k]
            fg = p.sample(np.rint(xyz[:, ::-1]).astype(int))
            res = sheets_crossed(fg, idx, snap_tol)
            for s, (n, ok) in zip(by_chain[k], res):
                s['cross'].append(dict(n=float(n), snapped=bool(ok)))
            snapped += sum(ok for _, ok in res)
            total += len(res)
        pred_meta.append(dict(url=p.url, shape=list(p.shape), frame_ok=frame_ok, chunks=n_chunks,
                              steps_snapped_frac=round(snapped / max(1, total), 4), seconds=round(time.time() - t0, 1)))
    for s in steps:
        counts, d = [x['n'] for x in s['cross']], abs(s['delta'])
        active = bool(counts) and d != 0
        s['over'] = active and all(n >= d + 1 for n in counts)
        s['under'] = active and all(n <= d - 1 for n in counts)
        s['over_strict'] = s['over'] and all(x['snapped'] for x in s['cross'])
        s['suggested_delta'] = int(np.sign(s['delta'])) * int(min(counts)) if s['over_strict'] else None
        s['p0'], s['p1'] = [round(float(v), 1) for v in s['p0']], [round(float(v), 1) for v in s['p1']]
        del s['chain']
    return steps, pred_meta


@click.command()
@click.option('--umbilicus', required=True, type=click.Path(exists=True, dir_okay=False), help='umbilicus.json (control_points x/y/z).')
@click.option('--pcl', 'pcl_paths', required=True, multiple=True, type=click.Path(exists=True, dir_okay=False),
              help='Point-collection JSON (relative, same and/or absolute windings). Repeatable.')
@click.option('--pred', 'pred_urls', multiple=True, help='Binary surface-prediction zarr array (path or URL) in the annotation frame. Repeatable.')
@click.option('--pad', default=6.0, type=float, help='Voxels sampled past each chain end.')
@click.option('--snap-tol', default=4.0, type=float, help='Max distance (voxels) from a point to its predicted sheet.')
@click.option('--threshold', default=None, type=float,
              help='Foreground if value > threshold. Default: 0 for integer/bool arrays, 0.5 for float arrays.')
@click.option('--output', required=True, type=click.Path(dir_okay=False), help='Output JSON report.')
def main(umbilicus, pcl_paths, pred_urls, pad, snap_tol, threshold, output):
    t0 = time.time()
    umb = json_umbilicus_z_to_yx(umbilicus)
    chains = load_chains(pcl_paths)
    preds = [Prediction(u, threshold) for u in pred_urls]
    steps, pred_meta = check(chains, umb, preds, pad, snap_tol)
    summary = dict(relative_steps=len(steps), global_suspects=sum(s.get('global_suspect', False) for s in steps),
                   local_suspects=sum(s.get('local_suspect', False) for s in steps), predictions=pred_meta, over=sum(s['over'] for s in steps), under=sum(s['under'] for s in steps),
                   over_strict=sum(s['over_strict'] for s in steps), seconds=round(time.time() - t0, 1))
    with open(output, 'w') as f:
        json.dump(dict(summary=summary, steps=steps), f, indent=1)
    click.echo(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
