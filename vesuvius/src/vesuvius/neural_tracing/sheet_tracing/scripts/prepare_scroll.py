"""Prepare another scroll's segments as training data (open data, vesuvius-challenge-open-data).

Every segment of ``--scroll`` registered onto the volume ``--volume-id`` (``tifxyz-transformed``), coordinates
multiplied by ``--coord-scale`` into local level-0 voxels (0.25 for full-resolution 2.4 um coordinates and remote
level 2 as local level 0), is masked and split by z:

* Segments run out of the scanned volume in places. Nodes outside the array, in chunks absent from the bucket, or
  where the CT is empty (every voxel of their 8^3 block is 0), and nodes within ``--margin`` voxels of those along
  the surface, are set invalid (-1), so no episode seeds or supervises there and no crop sits mostly in the void.
* Nodes with z in ``--heldout-z LO HI`` (default: the middle ``--heldout-fraction`` of the segments' z extent) go to
  ``OUT/heldout_patches`` for benchmarks, nodes more than ``--gap`` voxels away from that range to
  ``OUT/train_patches`` for training; each copy is cropped to its valid nodes and kept if it has at least
  ``--min-nodes`` of them. (Separate directories from ``prepare_unseen.py``'s ``OUT/patches``, which can hold
  unregistered segments.)

The CT the training episodes (and benchmarks on the held-out part) can reach, chunks within ``--pad`` voxels of any
kept node, is mirrored into ``OUT/volume.zarr`` with local level k = remote level ``--levels[k]`` (chunking and
compression kept). Existing chunks are reused, so it can be rerun or pointed at a ``prepare_unseen.py`` output of the
same volume. Train with ``data.patches_dir`` += OUT/train_patches and ``data.patch_volumes`` += OUT/volume.zarr.

  python scripts/prepare_scroll.py --metadata metadata.json --scroll PHerc1667 --volume-id 20251217075048 \\
      --coord-scale 0.25 --levels 2 3 --out data/unseen/PHerc1667
"""
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import numpy as np
from scipy import ndimage

from open_data import ROOT, chunk_name, download_chunks, mirror_array, read_segment, segment_path, write_segment

BLOCK = 8  # occupancy cell (local level-0 voxels) for the empty-CT test


class Occupancy:
    """Which BLOCK^3 cells of the local level-0 volume hold any non-zero CT, chunk by chunk, fetching missing chunks
    (and remembering the ones absent from the bucket in ``absent.json`` next to the level)."""

    def __init__(self, url: str, level_dir: Path, zarray: dict, workers: int):
        self.url, self.dir, self.zarray, self.workers = url, level_dir, zarray, workers
        self.shape, self.chunk = np.array(zarray['shape']), np.array(zarray['chunks'])
        self.cells = {}  # chunk key -> (chunk/BLOCK) bool, or None if absent
        self.absent_file = level_dir/'absent.json'
        self.absent = {tuple(k) for k in json.loads(self.absent_file.read_text())} if self.absent_file.is_file() else set()
        self.lock = threading.Lock()
        if zarray.get('compressor'):
            import numcodecs
            self.codec = numcodecs.get_codec(zarray['compressor'])
        else:
            self.codec = None

    def _cells(self, blob: bytes) -> np.ndarray:
        raw = self.codec.decode(blob) if self.codec else blob
        a = np.frombuffer(raw, np.dtype(self.zarray['dtype'])).reshape(self.chunk)
        c = self.chunk//BLOCK
        return a.reshape(c[0], BLOCK, c[1], BLOCK, c[2], BLOCK).max((1, 3, 5)) > 0

    def load(self, keys) -> int:
        """Occupancy of these chunks (zyx keys), fetching the ones not on disk; returns the bytes downloaded."""
        jobs = []
        for key in keys:
            if key in self.cells:
                continue
            if key in self.absent:
                self.cells[key] = None
                continue
            dst = self.dir/chunk_name(self.zarray, key)
            if dst.is_file():
                self.cells[key] = self._cells(dst.read_bytes())
            else:
                jobs.append((f'{self.url}/{chunk_name(self.zarray, key)}', dst, key))

        key_of = {dst: key for _, dst, key in jobs}

        def done(job, blob):
            cells, key = None if blob is None else self._cells(blob), key_of[job[1]]
            with self.lock:
                self.cells[key] = cells
                if blob is None:
                    self.absent.add(key)
        got = download_chunks([(url, dst) for url, dst, _ in jobs], self.workers, done=done)
        self.absent_file.write_text(json.dumps(sorted(self.absent)))
        return sum(got)

    def empty(self, xyz: np.ndarray) -> np.ndarray:
        """For level-0 points (N, 3) xyz: outside the array, in an absent chunk, or in an all-zero cell."""
        zyx = np.floor(xyz[:, ::-1]).astype(np.int64)
        inside = np.all((zyx >= 0) & (zyx < self.shape), 1)
        out = ~inside
        keys = zyx[inside]//self.chunk
        grid = -(-self.shape//self.chunk)
        flat = (keys[:, 0]*grid[1]+keys[:, 1])*grid[2]+keys[:, 2]
        uniq, inverse = np.unique(flat, return_inverse=True)
        uniq = [(int(f//(grid[1]*grid[2])), int(f//grid[2] % grid[1]), int(f % grid[2])) for f in uniq]
        self.load(uniq)
        local = (zyx[inside] % self.chunk)//BLOCK
        result = np.ones(len(keys), bool)
        order = np.argsort(inverse, kind='stable')
        bounds = np.searchsorted(inverse[order], np.arange(len(uniq)+1))
        for i, key in enumerate(uniq):
            cells = self.cells[key]
            if cells is None:
                continue
            sel = order[bounds[i]:bounds[i+1]]
            result[sel] = ~cells[local[sel, 0], local[sel, 1], local[sel, 2]]
        out[np.flatnonzero(inside)] = result
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--metadata', required=True, help='open-data metadata.json (decompressed)')
    ap.add_argument('--scroll', required=True)
    ap.add_argument('--volume-id', required=True)
    ap.add_argument('--coord-scale', type=float, required=True)
    ap.add_argument('--levels', type=int, nargs='+', default=[2, 3], help='remote levels for local levels 0, 1, ...')
    ap.add_argument('--margin', type=float, default=32., help='voxels along the surface kept clear of empty CT')
    ap.add_argument('--heldout-z', type=float, nargs=2, metavar=('LO', 'HI'), help='local level-0 z range held out')
    ap.add_argument('--heldout-fraction', type=float, default=.2)
    ap.add_argument('--gap', type=float, default=256., help='voxels of z between the training and held-out nodes')
    ap.add_argument('--min-nodes', type=int, default=20000)
    ap.add_argument('--pad', type=float, default=96., help='voxels of CT mirrored around every kept node')
    ap.add_argument('--max-gib', type=float, default=300., help='stop before mirroring more than this')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args(argv)

    meta = json.loads(Path(args.metadata).read_text())['samples'][args.scroll]
    vpath = meta['volumes'][args.volume_id]['data'][0]['origins'][0]['path'].rstrip('/')
    segs = sorted((g for g in meta['segments'].values()
                   if segment_path(g, 'tifxyz-transformed', args.volume_id)), key=lambda g: g['long_id'])
    out = Path(args.out)
    attrs = dict(source=f'{ROOT}/{vpath}', levels=args.levels, scroll=args.scroll)
    zarrays = [mirror_array(f'{ROOT}/{vpath}/{remote}', out/'volume.zarr'/str(level), attrs)
               for level, remote in enumerate(args.levels)]
    occupancy = Occupancy(f'{ROOT}/{vpath}/{args.levels[0]}', out/'volume.zarr'/'0', zarrays[0], args.workers)
    if args.heldout_z:
        lo, hi = args.heldout_z
    else:  # the middle of the segments' z extent on this volume (metadata bounding boxes)
        boxes = [np.array(g['properties']['volume_coverage'][args.volume_id]['bbox_transformed'])*args.coord_scale
                 for g in segs if args.volume_id in (g['properties'].get('volume_coverage') or {})]
        z0, z1 = min(b[0][2] for b in boxes), max(b[1][2] for b in boxes)
        z0, z1 = max(z0, 0.), min(z1, float(zarrays[0]['shape'][0]))
        mid, half = (z0+z1)/2, args.heldout_fraction*(z1-z0)/2
        lo, hi = mid-half, mid+half
    print(f'{len(segs)} registered segments; held-out z [{lo:.0f}, {hi:.0f}], training z outside '
          f'[{lo-args.gap:.0f}, {hi+args.gap:.0f}]', flush=True)

    summary, kept_points = [], []
    for g in segs:
        name = g['long_id']
        xyz, smeta = read_segment(segment_path(g, 'tifxyz-transformed', args.volume_id), args.coord_scale)
        valid = np.all(xyz != -1, -1) & np.all(np.isfinite(xyz), -1)
        void = np.zeros(valid.shape, bool)
        void[valid] = occupancy.empty(xyz[valid])
        cell = 1./float(smeta['scale'][0])
        near = ndimage.distance_transform_edt(~void, sampling=cell) < args.margin if void.any() else void
        kept = valid & ~near
        z = xyz[..., 2]
        row = dict(segment=name, valid=int(valid.sum()), empty=int(void.sum()), kept=int(kept.sum()))
        for part, sel in (('train_patches', kept & ((z < lo-args.gap) | (z > hi+args.gap))),
                          ('heldout_patches', kept & (z >= lo) & (z <= hi))):
            row[part] = int(sel.sum())
            if row[part] < args.min_nodes:
                continue
            rows, cols = np.flatnonzero(sel.any(1)), np.flatnonzero(sel.any(0))
            r, c = slice(rows[0], rows[-1]+1), slice(cols[0], cols[-1]+1)
            part_xyz = np.where(sel[r, c, None], xyz[r, c], -1).astype(np.float32)
            write_segment(out/part/name, part_xyz, smeta)
            step = max(1, int(32/cell))  # nodes ~32 vx apart are enough to find the chunks around them
            kept_points.append(part_xyz[::step, ::step][sel[r, c][::step, ::step]])
        summary.append(row)
        print(json.dumps(row), flush=True)
        del xyz, valid, void, near, kept, z

    # the CT within --pad of every kept node, at every level
    points = np.concatenate(kept_points) if kept_points else np.zeros((0, 3), np.float32)
    jobs = []
    for level, (remote, zarray) in enumerate(zip(args.levels, zarrays)):
        size, shape = np.array(zarray['chunks']), np.array(zarray['shape'])
        keys = set()
        for off in np.ndindex(3, 3, 3):
            zyx = (points[:, ::-1]+(np.array(off)-1)*args.pad)/(1 << level)
            k = np.unique(np.floor(zyx/size).astype(np.int64), axis=0)
            k = k[np.all((k >= 0) & (k*size < shape), 1)]
            keys.update(map(tuple, k.tolist()))
        missing = [k for k in keys if not (out/'volume.zarr'/str(level)/chunk_name(zarray, k)).is_file()
                   and not (level == 0 and k in occupancy.absent)]
        print(f'level {level}: {len(keys)} chunks around the kept nodes, {len(missing)} to fetch', flush=True)
        jobs += [(f'{ROOT}/{vpath}/{remote}/{chunk_name(zarray, k)}', out/'volume.zarr'/str(level)/chunk_name(zarray, k))
                 for k in missing]
    estimate = len(jobs)*np.prod(zarrays[0]['chunks'])/2**30  # uncompressed upper bound
    if estimate > args.max_gib:
        raise SystemExit(f'{len(jobs)} chunks (up to {estimate:.0f} GiB) exceed --max-gib {args.max_gib}')
    got = download_chunks(jobs, args.workers)
    print(f'downloaded {sum(1 for g in got if g)} chunks ({sum(got)/2**30:.2f} GiB), {sum(1 for g in got if not g)} '
          f'absent -> {out}', flush=True)
    (out/'prepare_scroll.json').write_text(json.dumps(dict(vars(args), heldout_z=[lo, hi], segments=summary), indent=1))


if __name__ == '__main__':
    main()
