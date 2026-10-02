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
from vesuvius.neural_tracing.fiber_follow.shared.components import crop_indices
from vesuvius.neural_tracing.fiber_follow.shared.data import fiber_manifest
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at


class NeighborBank:
    """A live append-only bank; safe with persistent fork/spawn loader workers."""
    def __init__(self, path, fibers, band, *, grid_scale=8., refresh_seconds=30., cache_bytes=64 << 20, training=True):
        if not np.isfinite(refresh_seconds) or refresh_seconds < 0 or cache_bytes < 1:
            raise ValueError('Invalid negative-bank refresh interval or cache size')
        self.root = Path(path).resolve()
        if self.root.name == 'bank.json':
            self.root = self.root.parent
        self.run = json.loads((self.root/'run.json').read_text())
        if self.run.get('version') not in (1, BANK_VERSION) or digest({k:v for k,v in self.run.items() if k != 'digest'}) != self.run.get('digest'):
            raise ValueError('Invalid negative-bank run metadata')
        if self.run['mining']['grid_scale'] != grid_scale:
            raise ValueError('Negative-bank coordinate scale differs from training')
        if band is None or self.run['excluded_z'] != [band.lo, band.hi]:
            raise ValueError('Negative-bank holdout differs from training')
        records = {r['name']:(i,r) for i,r in enumerate(self.run['fibers'])}
        self.fibers, self.band, self.training = fibers, band, training
        self.fiber_ids = []
        for current in fiber_manifest(fibers):
            entry = records.get(current['name'])
            if entry is None or entry[1]['training_fiber'] != training:
                raise ValueError(f'Negative bank has no {"training" if training else "evaluation"} annotation {current["name"]}')
            if any(entry[1].get(k) != v for k,v in current.items()):
                raise ValueError(f'Negative-bank annotation changed: {current["name"]}')
            self.fiber_ids.append(entry[0])
        self.refresh_seconds, self.cache_bytes = float(refresh_seconds), int(cache_bytes)
        self.exclusion = float(self.run['mining']['exclusion'])
        self._known, self._by_fiber = {}, {}
        self._manifest_sha = None
        self._signature, self._next_refresh = None, 0.
        self._cache, self._trees = OrderedDict(), OrderedDict()
        self._difficulty = OrderedDict()
        self._draw_cache = OrderedDict()
        self._arc_bounds = {}
        self._spatial_bounds = {}
        self._cached_bytes = 0
        self._pid = os.getpid()
        self.refresh(force=True)

    def __getstate__(self):
        return dict(self.__dict__, _cache=OrderedDict(), _trees=OrderedDict(), _difficulty=OrderedDict(),
                    _draw_cache=OrderedDict(), _cached_bytes=0,
                    _next_refresh=0., _pid=None)

    def _worker(self):
        if self._pid != os.getpid():
            self._cache, self._trees = OrderedDict(), OrderedDict()
            self._difficulty = OrderedDict()
            self._draw_cache = OrderedDict()
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
        if value.get('version') != self.run['version'] or value.get('run_digest') != self.run['digest']:
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
            available = shard['training_candidates'] if self.training else shard['candidates']-shard['training_candidates']
            if available:
                by_fiber.setdefault(fi,[]).append(shard)
        for rows in by_fiber.values():
            rows.sort(key=lambda s:(s['begin'],s['path']))
        self._known, self._by_fiber = incoming, by_fiber
        self._draw_cache.clear()
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

    def _archive(self, entry):
        raw = (self.root/entry['path']/'bank.npz').read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['bank_sha256']:
            raise ValueError(f'Negative-bank shard checksum mismatch: {entry["path"]}')
        return np.load(io.BytesIO(raw), allow_pickle=False)

    def _shard_arc_bounds(self, entry):
        """Mining anchors need not match a path's nearest annotated winding.

        Cache tiny verified arc envelopes independently of the geometry LRU.
        Legacy manifests have only anchor ranges, which cannot safely cull paths.
        """
        key = entry['path']
        if key not in self._arc_bounds:
            with self._archive(entry) as data:
                ranges = data['arc_ranges']
                if (ranges.shape != (entry['candidates'], 2) or not np.isfinite(ranges).all()
                        or np.any(ranges[:,0] >= ranges[:,1])):
                    raise ValueError(f'Invalid negative-bank arc ranges: {key}')
                self._arc_bounds[key] = ((float(ranges[:,0].min()), float(ranges[:,1].max()))
                                         if len(ranges) else (float('inf'), -float('inf')))
                points = data['points']
                if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                    raise ValueError(f'Invalid negative-bank points: {key}')
                self._spatial_bounds[key] = ((points.min(0), points.max(0)) if len(points)
                                             else (np.full(3, np.inf), np.full(3, -np.inf)))
        return self._arc_bounds[key]

    def _shard(self, entry):
        key = entry['path']
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        with self._archive(entry) as archive:
            data = {k:archive[k] for k in ('points','offsets','arc_ranges','train_eligible','anchors')}
            data['draw_eligible'] = archive['draw_eligible'] if 'draw_eligible' in archive else data['train_eligible'].copy()
        p, offsets, ranges, eligible = (data[k] for k in ('points','offsets','arc_ranges','train_eligible'))
        n = entry['candidates']
        if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or offsets.shape != (n+1,)
                or offsets.dtype.kind not in 'iu' or offsets[0] != 0 or offsets[-1] != len(p)
                or np.any(np.diff(offsets) < 2) or ranges.shape != (n,2) or not np.isfinite(ranges).all()
                or np.any(ranges[:,0] >= ranges[:,1]) or eligible.shape != (n,) or eligible.dtype != np.dtype(bool)
                or int(eligible.sum()) != entry['training_candidates'] or data['anchors'].shape != (n,)):
            raise ValueError(f'Invalid negative-bank geometry: {key}')
        draws = data['draw_eligible']
        if (draws.shape != (n,) or draws.dtype != np.dtype(bool) or np.any(draws & ~eligible)
                or ('draw_candidates' in entry and int(draws.sum()) != entry['draw_candidates'])
                or ('draw_indices' in entry and np.flatnonzero(draws).tolist() != entry['draw_indices'])):
            raise ValueError(f'Invalid negative-bank draw eligibility: {key}')
        if 'training_indices' in entry and np.flatnonzero(eligible).tolist() != entry['training_indices']:
            raise ValueError(f'Invalid negative-bank training eligibility: {key}')
        lines = {}
        for i in np.flatnonzero(eligible if self.training else ~eligible):
            line = p[offsets[i]:offsets[i+1]]
            if self.training and line[:,2].min()-2 < self.band.hi and line[:,2].max()+2 >= self.band.lo:
                raise ValueError(f'Training negative crosses holdout: {key}')
            arc = arclength(line)
            # Same quarter-voxel arclength interpolation as the target annotation.
            lines[i] = interp_at(line,arc,np.arange(0.,arc[-1]+1e-9,.25))
        data['lines'] = lines
        # Draw filtering uses the resampled path length, which can be slightly
        # shorter than the manifest's original path. Compute it once per shard.
        data['sampled_lengths'] = np.zeros(n, dtype=np.float64)
        for i, line in lines.items():
            data['sampled_lengths'][i] = arclength(line)[-1]
        self._arc_bounds[key] = ((float(ranges[:,0].min()), float(ranges[:,1].max()))
                                 if len(ranges) else (float('inf'), -float('inf')))
        self._spatial_bounds[key] = ((p.min(0), p.max(0)) if len(p)
                                     else (np.full(3, np.inf), np.full(3, -np.inf)))
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
        return [r['samples'] for r in self.path_records(fiber_index, original_arc, half_window=half_window)]

    def path_records(self, fiber_index, original_arc, *, half_window=64.):
        """Stable shard/path identifiers accompany exact relationship geometry."""
        self.refresh()
        lo, hi = original_arc-half_window, original_arc+half_window
        lines = []
        for shard in self._by_fiber.get(self.fiber_ids[fiber_index],()):
            lower, upper = self._shard_arc_bounds(shard)
            if lower > hi or upper < lo:
                continue
            data = self._shard(shard)
            for i,line in data['lines'].items():
                a,b = data['arc_ranges'][i]
                if a <= hi and b >= lo:
                    # Keep the original polyline: regular resampling can cut
                    # corners, which matters for continuous contact locations.
                    lines.append(self._path_record(shard, data, i))
        return lines

    @staticmethod
    def _path_record(shard, data, i):
        a, b = data['offsets'][i:i+2]
        return dict(points=data['points'][a:b], samples=data['lines'][i], shard=shard['path'], index=int(i))

    def spatial_records(self, fiber_index, world, *, radius=0.):
        """Trusted relationships intersecting a world-space bounding box.

        A nearby winding may have a distant annotation arc. Spatial queries
        therefore cannot be restricted to the intended head's progress window.
        """
        self.refresh()
        low, high = np.min(world, axis=0)-radius, np.max(world, axis=0)+radius
        records = []
        for shard in self._by_fiber.get(self.fiber_ids[fiber_index], ()):
            self._shard_arc_bounds(shard)
            a, b = self._spatial_bounds[shard['path']]
            if np.any(a > high) or np.any(b < low):
                continue
            data = self._shard(shard)
            for i in data['lines']:
                record = self._path_record(shard, data, i)
                p = record['points']
                if np.all(p.min(0) <= high) and np.all(p.max(0) >= low):
                    records.append(record)
        return records

    def draw_path(self, rng, *, unique=True, min_length=0., hard_fraction=0.):
        """Following draws use unique geometry; other roles may use all relationships.

        Version-2 length metadata selects fitting paths before loading geometry.
        Old shards use bounded rejection sampling without a full-bank scan.
        """
        if not np.isfinite(hard_fraction) or not 0 <= hard_fraction <= 1:
            raise ValueError('Hard bank fraction must be in [0,1]')
        if hard_fraction and rng.random() < hard_fraction:
            from .bank_geometry import difficulty_scores, DIFFICULTY_KINDS
            kind = int(rng.integers(len(DIFFICULTY_KINDS)))
            proposals = [self.draw_path(rng, unique=unique, min_length=min_length) for _ in range(8)]
            proposals = [p for p in proposals if p is not None]
            if not proposals:
                return None
            scores = []
            for fi, line, _ in proposals:
                key = (fi, hashlib.sha256(line.tobytes()).digest())
                if key not in self._difficulty:
                    self._difficulty[key] = difficulty_scores(line, self.fibers[fi])
                    if len(self._difficulty) > 128:
                        self._difficulty.popitem(last=False)
                scores.append(self._difficulty[key][kind])
            best = np.flatnonzero(np.asarray(scores) >= max(scores)-1e-12)
            return proposals[int(rng.choice(best))]
        if not self.training:
            return None
        self.refresh()
        local_ids, shards, sizes = self._draw_distribution(unique, min_length)
        if not shards:
            return None
        # Rejection can zero weights locally; never mutate the cached counts.
        sizes = sizes.copy()
        if not sizes.any():
            return None
        for _ in range(8 if min_length else 1):
            entry = shards[int(rng.choice(len(shards),p=sizes/sizes.sum()))]
            data = self._shard(entry)
            ids = [i for i in data['lines'] if (not unique or data['draw_eligible'][i])
                   and (not min_length or data['sampled_lengths'][i] >= min_length)]
            if ids:
                i = int(rng.choice(ids))
                return local_ids[entry['fiber']],data['lines'][i],data['arc_ranges'][i]
            sizes[shards.index(entry)] = 0
            if not sizes.any():
                break
        return None

    def _draw_distribution(self, unique, min_length):
        """Immutable metadata counts, invalidated whenever new shards appear."""
        key = (unique, min_length)
        if key not in self._draw_cache:
            local_ids = {global_id: local_id for local_id, global_id in enumerate(self.fiber_ids)}
            shards = [s for fi, rows in self._by_fiber.items() if fi in local_ids for s in rows]
            sizes = []
            for s in shards:
                count = s.get('draw_candidates', s['training_candidates']) if unique else s['training_candidates']
                indices = s.get('draw_indices' if unique else 'training_indices')
                if min_length and 'path_lengths' in s and indices is not None:
                    count = sum(s['path_lengths'][i] >= min_length for i in indices)
                elif min_length and 'path_lengths' in s and max(s['path_lengths'], default=0.) < min_length:
                    count = 0
                sizes.append(count)
            self._draw_cache[key] = (local_ids, shards, np.asarray(sizes, float))
            if len(self._draw_cache) > 32:
                self._draw_cache.popitem(last=False)
        self._draw_cache.move_to_end(key)
        return self._draw_cache[key]

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

    def state_target_tree(self, item):
        return self._target_tree(item['fiber_ref'][0])

    def clear_of_state(self, item, world):
        tree,gap = self.state_target_tree(item)
        return tree.query(np.asarray(world).reshape(-1,3))[0]-gap > self.exclusion

    def candidates(self, item, crop, rule, *, mask_crop=None, additional_banks=(), rasterize=True):
        """Exact centerline queries, with whole-annotation clearance.

        Presence support was established when the bank was mined; it is not
        re-checked on the training crop.

        Confidence labels rasterize only cells containing these line samples;
        their entire cell must clear the target exclusion tube. Query locations
        are never rasterized, expanded, jittered or snapped to the crop grid.
        Observation-only rows use the same geometry for sampling feedback but
        do not need a dense confidence-label mask.
        """
        fi,t,reverse = item['fiber_ref']
        lateral, forward = crop.lateral_coords[[0,-1]], crop.forward_coords[[0,-1]]
        corners = np.array([[a,b,z] for a in lateral for b in lateral for z in forward])
        world = corners @ np.asarray(item['frame']).T+item['pos']
        lines = [r['samples'] for bank in (self, *additional_banks)
                 for r in bank.spatial_records(fi, world)]
        mask_crop = crop if mask_crop is None else mask_crop
        shape = (mask_crop.depth,mask_crop.width,mask_crop.width)
        mask = np.zeros(shape,bool) if rasterize else None
        empty = dict(foreign=mask, local=np.empty((0,3)), nearest=np.empty(0,np.int64), path_ids=np.empty(0,np.int64),
                     counts=dict(foreign_components=0))
        if not lines or len(item['identity_curve']) < 3:
            return empty
        pos, frame = np.asarray(item['pos']), np.asarray(item['frame'])
        local = np.concatenate([(p-pos) @ frame for p in lines])
        path_ids = np.repeat(np.arange(len(lines)),[len(p) for p in lines])
        indices = crop_indices(crop,local)
        near = np.all((indices >= 0) & (indices <= np.array([crop.depth,crop.width,crop.width])-1),axis=1)
        local,indices,path_ids = local[near],indices[near],path_ids[near]
        if not len(local):
            return empty
        tree, gap = self.state_target_tree(item)
        keep = tree.query(local @ frame.T+pos)[0]-gap > max(self.exclusion,rule.own_radius)
        nearest = cKDTree(item['identity_curve']).query(local)[1]
        keep &= (nearest > 0) & (nearest < len(item['identity_curve'])-1)
        local,nearest,path_ids = local[keep],nearest[keep],path_ids[keep]
        if rasterize:
            voxels = np.unique(np.rint(crop_indices(mask_crop,local)).astype(int),axis=0)
            voxels = voxels[np.all((voxels >= 0) & (voxels < np.asarray(shape)),axis=1)]
            # Only these occupied cells are queried; avoid materializing the
            # full million-voxel coordinate grid for every decision.
            lateral, forward = mask_crop.lateral_coords, mask_crop.forward_coords
            points = np.column_stack((lateral[voxels[:, 2]], lateral[voxels[:, 1]],
                                      forward[voxels[:, 0]]))
            half_cell = np.sqrt(3)*mask_crop.spacing/2
            keep = tree.query(points @ frame.T+pos)[0]-gap-half_cell > max(self.exclusion,rule.own_radius)
            mask[tuple(voxels[keep].T)] = True
        return dict(foreign=mask, local=local, nearest=nearest, path_ids=path_ids,
                    counts=dict(foreign_components=int(bool(len(local)))))
