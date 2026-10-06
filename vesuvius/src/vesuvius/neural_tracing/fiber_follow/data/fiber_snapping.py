"""Snap fiber polylines to the centre of a fiber presence prediction (batched on the GPU, tiled over the volume).

Coordinates are presence voxels (native / ``native_per_voxel``, minus ``offset``), xyz. One pass: every fiber is
resampled every ``step`` voxels; at each point the presence is sampled on a disk of radius ``radius`` perpendicular
to the local tangent (grid ``grid`` voxels) and averaged over ``along`` voxels each way along the fiber; the point's
offset is the centroid of the disk region at or above half way between the disk minimum and maximum. Offsets are
smoothed along each fiber (Gaussian ``sigma``, in voxels) and applied. ``passes`` repeats this on the snapped curve.
Validated on Paris4 AFV against manual annotations (output/paris4_annotation_audit_20261006): label distance to the
manual fiber median 1.46 -> 0.68 trace voxels, within 1 voxel 25% -> 77% (radius 2.5, sigma 2, two passes).

The presence volume is read in ``tile``-voxel blocks (plus a margin covering the disk) in spatial order on worker
threads, so each block is read once per pass; all points of a block are sampled in one ``grid_sample`` call.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SnapConfig:
    radius: float = 2.5
    sigma: float = 2.
    passes: int = 2
    step: float = 1.
    grid: float = .5
    along: float = 1.
    tangent_sigma: float = 3.
    tile: int = 128
    threads: int = 16

    def __post_init__(self):
        if min(self.radius, self.sigma, self.step, self.grid, self.tangent_sigma) <= 0 or self.along < 0:
            raise ValueError('Snapping radius, sigma, step, grid and tangent sigma must be positive')
        if type(self.passes) is not int or self.passes < 1 or self.tile < 8 or self.threads < 1:
            raise ValueError('Snapping needs at least one pass, tiles of 8+ voxels and one thread')

    def to_dict(self):
        return asdict(self)


def resample(points, step):
    """Polyline (N, 3) resampled every ``step`` along its arclength, both ends kept."""
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.r_[0., np.cumsum(segments)]
    if s[-1] <= 0:
        return points[:1].copy()
    count = max(2, int(np.ceil(s[-1]/step))+1)
    arcs = np.linspace(0., s[-1], count)
    return np.stack([np.interp(arcs, s, points[:, k]) for k in range(3)], 1)


def disk_offsets(cfg):
    """(K, 3) local (u, v, w) sample offsets: the perpendicular disk, repeated at each along-fiber slice, and the
    (D, 2) disk coordinates."""
    g = np.arange(-cfg.radius, cfg.radius+1e-9, cfg.grid)
    u, v = np.meshgrid(g, g, indexing='ij')
    keep = np.hypot(u, v) <= cfg.radius+1e-9
    disk = np.stack((u[keep], v[keep]), 1)
    slices = np.arange(-cfg.along, cfg.along+1e-9, cfg.grid) if cfg.along else np.zeros(1)
    return np.concatenate([np.c_[disk, np.full(len(disk), w)] for w in slices]), disk, len(slices)


def frames(tangents):
    """Unit perpendicular axes e1, e2 (N, 3) for unit tangents (N, 3)."""
    axis = np.eye(3)[np.argmin(np.abs(tangents), axis=1)]
    e1 = np.cross(tangents, axis)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(tangents, e1)


class PresenceTiles:
    """Blocks of a zyx presence array (anything with ``read(start_zyx, size_zyx)`` and ``shape``) around xyz tiles,
    returned in pinned memory when ``pinned`` (for asynchronous transfer to the GPU)."""
    def __init__(self, array, tile, margin, pinned=False):
        self.array, self.tile, self.margin, self.pinned = array, int(tile), int(margin), pinned
        self.shape = np.asarray(array.shape)

    def block(self, key):
        lo = np.asarray(key)*self.tile-self.margin  # zyx
        hi = lo+self.tile+2*self.margin
        a, b = np.maximum(lo, 0), np.minimum(hi, self.shape)
        out = torch.zeros(tuple(hi-lo), dtype=torch.uint8, pin_memory=self.pinned)
        if (b > a).all():
            out.numpy()[tuple(slice(x-l, y-l) for x, y, l in zip(a, b, lo))] = self.array.read(tuple(a), tuple(b-a))
        return lo, out


def point_offsets(points, tangents, tiles, cfg, device):
    """World offsets (N, 3) from each point to the presence centroid on its perpendicular disk.

    Points are sorted by tile once, so each tile's points are a contiguous slice of arrays uploaded once; blocks
    are read ahead on worker threads and every tile's offsets land in one device buffer copied back at the end."""
    local, disk, slices = disk_offsets(cfg)
    keys = np.floor(points[:, ::-1]/tiles.tile).astype(np.int64)  # zyx tile of each point
    order = np.lexsort(keys.T[::-1])
    keys = keys[order]
    breaks = np.r_[0, np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1))+1, len(order)]
    e1, e2 = frames(tangents[order])
    as_device = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32).to(device, non_blocking=True)
    p, t, a, b = (as_device(x) for x in (points[order], tangents[order], e1, e2))
    local_t, disk_t = as_device(local), as_device(disk)
    out = torch.zeros(len(order), 2, device=device)
    ahead = 2*cfg.threads
    with ThreadPoolExecutor(cfg.threads) as pool:
        pending = [pool.submit(tiles.block, tuple(keys[breaks[i]])) for i in range(min(ahead, len(breaks)-1))]
        for index in range(len(breaks)-1):
            if index+ahead < len(breaks)-1:
                pending.append(pool.submit(tiles.block, tuple(keys[breaks[index+ahead]])))
            lo, block = pending[index].result()
            pending[index] = None
            volume = block.to(device, non_blocking=True).float()[None, None]  # (1, 1, D, H, W)
            g = slice(breaks[index], breaks[index+1])
            world = (p[g, None]+local_t[None, :, 0:1]*a[g, None]+local_t[None, :, 1:2]*b[g, None]
                     +local_t[None, :, 2:3]*t[g, None])  # (n, K, 3) xyz
            scale = torch.tensor([2/(n-1) for n in block.shape[::-1]], device=device)
            grid = (world-torch.tensor(lo[::-1].tolist(), dtype=torch.float32, device=device))*scale-1
            values = F.grid_sample(volume, grid[None, :, :, None], mode='bilinear', padding_mode='border',
                                   align_corners=True)[0, 0, :, :, 0]  # (n, K)
            values = values.reshape(-1, slices, len(disk)).mean(1)
            floor = values.min(1, keepdim=True).values
            top = values.max(1, keepdim=True).values
            weight = torch.where(values >= floor+.5*(top-floor), values-floor, torch.zeros_like(values))
            total = weight.sum(1, keepdim=True)
            out[g] = torch.where(total > 0, weight@disk_t/total.clamp_min(1e-12), torch.zeros_like(out[g]))
    uv = out.cpu().numpy().astype(np.float64)
    offsets = np.empty_like(points)
    offsets[order] = uv[:, :1]*e1+uv[:, 1:]*e2
    return offsets


def snap_polylines(polylines, presence, cfg=SnapConfig(), device=None):
    """Snapped copies of xyz polylines (presence voxels), each resampled every ``cfg.step``."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    margin = int(np.ceil(np.hypot(cfg.radius, cfg.along)))+2
    tiles = PresenceTiles(presence, cfg.tile, margin, pinned=torch.device(device).type == 'cuda')
    curves = [np.asarray(p, np.float64) for p in polylines]
    for _ in range(cfg.passes):
        curves = [resample(c, cfg.step) for c in curves]
        lengths = np.array([len(c) for c in curves])
        points = np.concatenate(curves)
        tangents = np.empty_like(points)
        start = 0
        for c in curves:
            n = len(c)
            if n == 1:
                tangents[start] = (1., 0., 0.)
            else:
                g = np.gradient(gaussian_filter1d(c, cfg.tangent_sigma/cfg.step, axis=0, mode='nearest'), axis=0)
                tangents[start:start+n] = g/np.linalg.norm(g, axis=1, keepdims=True).clip(1e-12)
            start += n
        offsets = point_offsets(points, tangents, tiles, cfg, device)
        out, start = [], 0
        for c, n in zip(curves, lengths):
            o = offsets[start:start+n]
            out.append(c+(gaussian_filter1d(o, cfg.sigma/cfg.step, axis=0, mode='nearest') if n > 1 else o))
            start += n
        curves = out
    return curves


# Files: native base-voxel geometry mapped to presence voxels as native/scale - offset, written every output_step.
def snap_native(polylines, presence, scale, offset, cfg, device, output_step):
    curves = snap_polylines([p/scale-offset for p in polylines], presence, cfg, device)
    return [(resample(c, output_step)+offset)*scale for c in curves]


def snap_afv(source, destination, presence, scale, offset, cfg, device, output_step, limit, provenance):
    from vesuvius.neural_tracing.fiber_follow.data.afv import read_afv_polylines, write_afv
    fibers = read_afv_polylines(source)
    if limit:
        fibers = fibers[:limit]
    snapped = snap_native([p for _, p in fibers], presence, scale, offset, cfg, device, output_step)
    return write_afv(source, destination, [(row, xyz) for (row, _), xyz in zip(fibers, snapped)],
                     dict(snapping=provenance, snapped_from=str(Path(source).resolve())))


def snap_json(paths, destination, presence, scale, offset, cfg, device, output_step, provenance):
    destination.mkdir(parents=True, exist_ok=False)
    documents = [json.loads(p.read_text()) for p in paths]
    lines = [np.asarray(d['line_points'], np.float64) for d in documents]
    snapped = snap_native(lines, presence, scale, offset, cfg, device, output_step)
    for path, document, line, new in zip(paths, documents, lines, snapped):
        s_old = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        s_new = np.r_[0., np.cumsum(np.linalg.norm(np.diff(new, axis=0), axis=1))]
        for control in document.get('control_points', []):
            # Same arclength fraction on the snapped line, at the nearest written point.
            nearest = np.argmin(np.linalg.norm(line-np.asarray(control['position']), axis=1))
            fraction = s_old[nearest]/max(s_old[-1], 1e-12)
            control['position'] = new[np.argmin(np.abs(s_new-fraction*s_new[-1]))].tolist()
        document['line_points'] = new.tolist()
        (destination/path.name).write_text(json.dumps(document))
    (destination/'snapping.json').write_text(json.dumps(dict(provenance, files=[p.name for p in paths]), indent=1))
    return dict(fibers=len(paths), points=int(sum(len(s) for s in snapped)))
