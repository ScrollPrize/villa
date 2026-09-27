"""Live, worker-local access to immutable, automatically validated negative paths.

The producer atomically appends finished shards to bank.json. No native tracing
or image reading takes place in this loader. Missing coverage stays unknown.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bulk import BANK_VERSION, digest
from vesuvius.neural_tracing.fiber_follow.shared.components import crop_indices, volume_at
from vesuvius.neural_tracing.fiber_follow.shared.data import fiber_manifest
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at, crop_local_grid


class NeighborBank:
    """A live append-only bank; safe with persistent fork/spawn loader workers."""
    def __init__(self, path, fibers, band, *, grid_scale=8., refresh_seconds=30., cache_bytes=64 << 20):
        if not np.isfinite(refresh_seconds) or refresh_seconds < 0 or cache_bytes < 1:
            raise ValueError('Invalid negative-bank refresh interval or cache size')
        self.root = Path(path).resolve()
        if self.root.name == 'bank.json':
            self.root = self.root.parent
        self.run = json.loads((self.root/'run.json').read_text())
        if self.run.get('version') != BANK_VERSION or digest({k:v for k,v in self.run.items() if k != 'digest'}) != self.run.get('digest'):
            raise ValueError('Invalid negative-bank run metadata')
        if self.run['mining']['grid_scale'] != grid_scale:
            raise ValueError('Negative-bank coordinate scale differs from training')
        if band is None or self.run['excluded_z'] != [band.lo, band.hi]:
            raise ValueError('Negative-bank holdout differs from training')
        records = {r['name']:(i,r) for i,r in enumerate(self.run['fibers'])}
        self.fibers, self.band = fibers, band
        self.fiber_ids = []
        for current in fiber_manifest(fibers):
            entry = records.get(current['name'])
            if entry is None or not entry[1]['training_fiber']:
                raise ValueError(f'Negative bank has no training annotation {current["name"]}')
            if any(entry[1].get(k) != v for k,v in current.items()):
                raise ValueError(f'Negative-bank annotation changed: {current["name"]}')
            self.fiber_ids.append(entry[0])
        self.refresh_seconds, self.cache_bytes = float(refresh_seconds), int(cache_bytes)
        self.exclusion = float(self.run['mining']['exclusion'])
        self._known, self._by_fiber = {}, {}
        self._manifest_sha = None
        self._signature, self._next_refresh = None, 0.
        self._cache, self._trees = OrderedDict(), OrderedDict()
        self._cached_bytes = 0
        self._pid = os.getpid()
        self.refresh(force=True)

    def __getstate__(self):
        return dict(self.__dict__, _cache=OrderedDict(), _trees=OrderedDict(), _cached_bytes=0,
                    _next_refresh=0., _pid=None)

    def _worker(self):
        if self._pid != os.getpid():
            self._cache, self._trees = OrderedDict(), OrderedDict()
            self._cached_bytes, self._next_refresh = 0, 0.
            self._pid = os.getpid()

    def refresh(self, *, force=False):
        self._worker()
        now = time.monotonic()
        if not force and now < self._next_refresh:
            return False
        self._next_refresh = now+self.refresh_seconds
        path = self.root/'bank.json'
        try:
            stat = path.stat()
        except FileNotFoundError:
            return False  # producer may not have published its first shard yet
        signature = (stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if signature == self._signature:
            return False
        value = json.loads(path.read_text())
        if value.get('version') != BANK_VERSION or value.get('run_digest') != self.run['digest']:
            raise ValueError('Negative-bank run changed during training')
        if value.get('sha256') != digest({k:v for k,v in value.items() if k != 'sha256'}):
            raise ValueError('Negative-bank index checksum mismatch')
        for key in ('fibers','mining','excluded_z'):
            if value[key] != self.run[key]:
                raise ValueError(f'Negative-bank {key} changed during training')
        incoming = {s['path']:s for s in value['shards']}
        if len(incoming) != len(value['shards']):
            raise ValueError('Duplicate negative-bank shard')
        if any(incoming.get(k) != v for k,v in self._known.items()):
            raise ValueError('Published negative-bank shards were removed or modified')
        by_fiber = {}
        for key, shard in incoming.items():
            if not (self.root/key).resolve().is_relative_to(self.root):
                raise ValueError('Negative shard path escapes bank directory')
            fi = shard['fiber']
            if not 0 <= fi < len(self.run['fibers']):
                raise ValueError('Invalid negative-bank fiber index')
            if shard['training_candidates'] and not self.run['fibers'][fi]['training_fiber']:
                raise ValueError('Held-out annotation marked training eligible')
            if shard['training_candidates']:
                by_fiber.setdefault(fi,[]).append(shard)
        for rows in by_fiber.values():
            rows.sort(key=lambda s:(s['begin'],s['path']))
        self._known, self._by_fiber = incoming, by_fiber
        self._manifest_sha, self._signature = value['sha256'], signature
        return True

    @property
    def shard_count(self):
        return len(self._known)

    def provenance(self):
        """Record a snapshot boundary, while permitting append-only growth on resume."""
        self.refresh(force=True)
        return dict(version=1, run_digest=self.run['digest'], manifest_sha256=self._manifest_sha,
                    shard_hashes={k:v['bank_sha256'] for k,v in sorted(self._known.items())})

    def validate_resume(self, saved):
        if not saved or saved.get('version') != 1 or saved.get('run_digest') != self.run['digest']:
            raise ValueError('Checkpoint negative-bank run differs')
        current = self.provenance()['shard_hashes']
        if any(current.get(k) != v for k,v in saved['shard_hashes'].items()):
            raise ValueError('Checkpoint negative-bank shards were removed or modified')

    def validate_volume(self, spec):
        groups = self.run['prediction_manifest']['groups']
        if any(Path(g['zarr']).parent.parent.resolve() != Path(spec.fiber_zarr_dir).resolve()
               or Path(g['zarr']).name != str(spec.fiber_level) for g in groups.values()):
            raise ValueError('Negative-bank prediction volume differs from training')
        if Path(self.run['ct']).parent.resolve() != Path(spec.ct_zarr).resolve():
            raise ValueError('Negative-bank CT source differs from training')

    def _shard(self, entry):
        key = entry['path']
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        raw = (self.root/key/'bank.npz').read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['bank_sha256']:
            raise ValueError(f'Negative-bank shard checksum mismatch: {key}')
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            data = {k:archive[k] for k in ('points','offsets','arc_ranges','train_eligible','anchors')}
        p, offsets, ranges, eligible = (data[k] for k in ('points','offsets','arc_ranges','train_eligible'))
        n = entry['candidates']
        if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or offsets.shape != (n+1,)
                or offsets.dtype.kind not in 'iu' or offsets[0] != 0 or offsets[-1] != len(p)
                or np.any(np.diff(offsets) < 2) or ranges.shape != (n,2) or not np.isfinite(ranges).all()
                or np.any(ranges[:,0] >= ranges[:,1]) or eligible.shape != (n,) or eligible.dtype != np.dtype(bool)
                or int(eligible.sum()) != entry['training_candidates'] or data['anchors'].shape != (n,)):
            raise ValueError(f'Invalid negative-bank geometry: {key}')
        lines = {}
        for i in np.flatnonzero(eligible):
            line = p[offsets[i]:offsets[i+1]]
            if line[:,2].min()-2 < self.band.hi and line[:,2].max()+2 >= self.band.lo:
                raise ValueError(f'Training negative crosses holdout: {key}')
            arc = arclength(line)
            # Same quarter-voxel arclength interpolation as the target annotation.
            lines[i] = interp_at(line,arc,np.arange(0.,arc[-1]+1e-9,.25))
        data['lines'] = lines
        data['bytes'] = sum(v.nbytes for v in data.values() if isinstance(v,np.ndarray))+sum(p.nbytes for p in lines.values())
        while self._cache and self._cached_bytes+data['bytes'] > self.cache_bytes:
            _, old = self._cache.popitem(last=False)
            self._cached_bytes -= old['bytes']
        if data['bytes'] <= self.cache_bytes:
            self._cache[key] = data
            self._cached_bytes += data['bytes']
        return data

    def paths(self, fiber_index, original_arc, *, half_window=64.):
        """Only validated training paths near this target's original arclength."""
        self.refresh()
        lo, hi = original_arc-half_window, original_arc+half_window
        padding = self.run['mining']['block_size']
        lines = []
        for shard in self._by_fiber.get(self.fiber_ids[fiber_index],()):
            if shard['anchor_range'][0] > hi+padding or shard['anchor_range'][1] < lo-padding:
                continue
            data = self._shard(shard)
            for i,line in data['lines'].items():
                a,b = data['arc_ranges'][i]
                if a <= hi and b >= lo:
                    lines.append(line)
        return lines

    def draw_path(self, rng):
        """Uniform over published training paths belonging to this dataset."""
        self.refresh()
        local_ids = {global_id:local_id for local_id,global_id in enumerate(self.fiber_ids)}
        shards = [s for fi,rows in self._by_fiber.items() if fi in local_ids for s in rows]
        if not shards:
            return None
        sizes = np.array([s['training_candidates'] for s in shards],dtype=float)
        entry = shards[int(rng.choice(len(shards),p=sizes/sizes.sum()))]
        data = self._shard(entry)
        i = int(rng.choice(list(data['lines'])))
        return local_ids[entry['fiber']],data['lines'][i],data['arc_ranges'][i]

    def _target_tree(self, fi):
        if fi not in self._trees:
            p = self.fibers[fi].points
            self._trees[fi] = (cKDTree(p),float(np.linalg.norm(np.diff(p,axis=0),axis=1).max()/2))
            if len(self._trees) > 8:
                self._trees.popitem(last=False)
        self._trees.move_to_end(fi)
        return self._trees[fi]

    def clear_of_target(self, fi, world):
        tree,gap = self._target_tree(fi)
        return tree.query(np.asarray(world).reshape(-1,3))[0]-gap > self.exclusion

    def candidates(self, item, crop, presence, rule):
        """Exact centerline queries, with whole-annotation clearance.

        Confidence labels rasterize only cells containing these line samples;
        their entire cell must clear the target exclusion tube. Query locations
        are never rasterized, expanded, jittered or snapped to the crop grid.
        """
        fi,t,reverse = item['fiber_ref']
        fiber = self.fibers[fi]
        lines = self.paths(fi,fiber.length-t if reverse else t)
        shape = (crop.depth,crop.width,crop.width)
        mask = np.zeros(shape,bool)
        empty = dict(foreign=mask, local=np.empty((0,3)), nearest=np.empty(0,np.int64),
                     counts=dict(foreign_components=0))
        if not lines or len(item['identity_curve']) < 3:
            return empty
        pos, frame = np.asarray(item['pos']), np.asarray(item['frame'])
        local = np.concatenate([(p-pos) @ frame for p in lines])
        indices = crop_indices(crop,local)
        near = np.all((indices >= 0) & (indices <= np.asarray(shape)-1),axis=1)
        local,indices = local[near],indices[near]
        if not len(local):
            return empty
        tree, gap = self._target_tree(fi)
        keep = volume_at(presence,crop,local,order=1) >= rule.threshold
        keep &= tree.query(local @ frame.T+pos)[0]-gap > max(self.exclusion,rule.own_radius)
        nearest = cKDTree(item['identity_curve']).query(local)[1]
        keep &= (nearest > 0) & (nearest < len(item['identity_curve'])-1)
        local,nearest = local[keep],nearest[keep]
        voxels = np.unique(np.rint(indices[keep]).astype(int),axis=0)
        points = crop_local_grid(crop)[tuple(voxels.T)]
        half_cell = np.sqrt(3)*crop.spacing/2
        keep = tree.query(points @ frame.T+pos)[0]-gap-half_cell > max(self.exclusion,rule.own_radius)
        mask[tuple(voxels[keep].T)] = True
        return dict(foreign=mask, local=local, nearest=nearest,
                    counts=dict(foreign_components=int(bool(len(local)))))
