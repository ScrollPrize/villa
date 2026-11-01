"""On-policy replay: collected DAgger decisions, their labels and the replay index used for sampling."""
from __future__ import annotations

import os
from pathlib import Path
import json

import numpy as np

from vesuvius.neural_tracing.tracers.shared.annotations.traced import fiber_manifest
from vesuvius.neural_tracing.tracers.fibers.data.tasks import TRAVEL_STRATA, stable_key
from vesuvius.neural_tracing.tracers.fibers.data.state_labels import (
    DEPARTURE_PATIENCE, facts, REPLAY_CLASS, REPLAY_CLASSES)


EVENT_TABLE_SIZE = 1 << 18


class ReplayIndex:
    """Eligibility index by class, fiber, episode and event; rows are drawn last.

    Fibers are uniform, then episodes, then events, then rows, so dense windows cannot
    dominate merely by containing more decisions. Ordinary following is additionally
    stratified by travelled length. Each event group carries a stable key for the
    shared reuse counter.
    """
    def __init__(self, caches):
        self.caches = list(caches)
        self.groups = {name: {} for name in REPLAY_CLASSES}
        self.ordinary = [dict() for _ in range(len(TRAVEL_STRATA)-1)]
        for ci, op in enumerate(self.caches):
            identity = op.provenance["cache_id"]
            classes = np.asarray(op.replay_class)
            for name, index in REPLAY_CLASS.items():
                rows = np.flatnonzero(classes == index)
                if not len(rows):
                    continue
                if name == 'ordinary':
                    strata = np.digitize(np.asarray(op.travelled)[rows], TRAVEL_STRATA[1:-1])
                    keys = np.c_[np.asarray(op.episode)[rows], strata]
                else:
                    strata = None
                    keys = np.c_[np.asarray(op.episode)[rows], np.asarray(op.event_id)[rows]]
                for key in np.unique(keys, axis=0):
                    members = rows[(keys == key).all(1)]
                    episode, event = int(key[0]), int(key[1])
                    group = (ci, episode, event, members, stable_key(identity, episode, name, event))
                    fi = int(op.fiber_idx[members[0]])
                    table = self.ordinary[event] if name == 'ordinary' else self.groups[name]
                    table.setdefault(fi, {}).setdefault((ci, episode), []).append(group)

    def counts(self):
        tables = dict(self.groups, ordinary=None)
        result = {}
        for name, table in tables.items():
            tables_ = self.ordinary if name == 'ordinary' else [table]
            groups = [g for t in tables_ for episodes in t.values() for gs in episodes.values() for g in gs]
            result[name] = dict(fibers=len({f for t in tables_ for f in t}), events=len(groups),
                                rows=int(sum(len(g[3]) for g in groups)))
        return result

    def draw(self, name, rng, eligible):
        """One event group from an eligible cache, or None."""
        tables = [t for t in self.ordinary if t] if name == 'ordinary' else [self.groups[name]]
        if name == 'ordinary' and tables:
            tables = [tables[int(rng.integers(len(tables)))]]
        for table in tables:
            fibers = [f for f, episodes in table.items() if any(e[0] in eligible for e in episodes)]
            if not fibers:
                return None
            episodes = table[fibers[int(rng.integers(len(fibers)))]]
            choices = [key for key in episodes if key[0] in eligible]
            groups = episodes[choices[int(rng.integers(len(choices)))]]
            return groups[int(rng.integers(len(groups)))]
        return None


def replay_facts(op, j, cfg):
    """Stored trace facts of one replay row, relabeled under this run's label contract."""
    return facts(match_distance=float(op.match_distance[j]), window_distance=float(op.window_distance[j]),
                 match_valid=bool(op.match_valid[j]), match_ambiguous=bool(op.match_ambiguous[j]),
                 switched=bool(op.switched[j]), beyond_end=bool(op.beyond_end[j]),
                 tolerance=cfg.label_tolerance, max_recovery_distance=cfg.max_recovery_distance,
                 t=float(op.t[j]), excursion=bool(op.bad_run[j] >= DEPARTURE_PATIENCE), bad_run=int(op.bad_run[j]),
                 bad_run_start=float(op.bad_run_start[j]), departure_distance=float(op.departure_distance[j]),
                 boundary_distance=float(op.boundary_distance[j]), switch_distance=float(op.switch_distance[j]))


def replay_labeler_state(op, j):
    """Exact ``TraceLabeler`` state after row ``j``; restarts resume from it."""
    finite = lambda value: None if not np.isfinite(value) else float(value)
    return dict(t=float(op.t[j]), last_travelled=float(op.travelled[j]), bad_run=int(op.bad_run[j]),
                bad_run_start=finite(op.bad_run_start[j]), started=True,
                departure_distance=finite(op.departure_distance[j]),
                boundary_distance=finite(op.boundary_distance[j]), switch=None)


class OnPolicyStates:
    """Decision states (pos, frame, own-trace history) visited by a policy on GT fibers.

    One schema: every field below is required and validated on load; there are no
    defaults for absent fields. Loaded as memory-mapped .npy files so DataLoader workers
    share one copy (pickling sends only the directory path). Each trace's committed
    polyline is stored once in ``track_pos``; rows reference it by ``seq_start:seq_end``
    (the exclusive end includes the decision head). Nothing here except the observed
    geometry ever becomes a model input.
    """

    # name: (dtype, trailing shape); None trailing dims are checked against arrays.
    FIELDS = dict(
        fiber_idx=('i8', ()), t=('f8', ()), reverse=('?', ()), pos=('f8', (3,)), frame=('f8', (3, 3)),
        hist=('f8', (None, 3)), hmask=('f4', (None,)),
        seed_pos=('f8', (3,)), seed_tangent=('f8', (3,)), seed_age=('f4', ()), seed_valid=('?', ()),
        heading_start=('i8', ()), travelled=('f8', ()), episode=('i8', ()), source_row=('i8', ()),
        seq_start=('i8', ()), seq_end=('i8', ()),
        # Current supervision (state_labels.classify) and its trace facts.
        supervision=('i1', ()),
        match_distance=('f4', ()), window_distance=('f4', ()), match_valid=('?', ()), match_ambiguous=('?', ()),
        switched=('?', ()), beyond_end=('?', ()),
        # Historical events (NaN when absent) and resumable departure patience.
        departure_distance=('f8', ()), boundary_distance=('f8', ()), switch_distance=('f8', ()),
        bad_run=('i4', ()), bad_run_start=('f8', ()),
        # Replay membership.
        replay_class=('i1', ()), event_id=('i8', ()))
    TRACK = dict(track_pos=('f8', (3,)))

    def __init__(self, *, manifest, provenance, **arrays):
        expected = set(self.FIELDS) | set(self.TRACK)
        if set(arrays) != expected:
            raise ValueError(f'Replay fields differ from the schema: missing {sorted(expected-set(arrays))}, '
                             f'unexpected {sorted(set(arrays)-expected)}')
        for key in expected:
            setattr(self, key, np.asarray(arrays[key]))
        self._dir = None
        self.manifest = manifest
        self.provenance = provenance
        self.validate()

    def validate(self):
        n = len(self.pos)
        for key, (dtype, trailing) in {**self.FIELDS, **self.TRACK}.items():
            value = getattr(self, key)
            length = len(self.track_pos) if key in self.TRACK else n
            if len(value) != length or value.ndim != 1+len(trailing) or any(
                    expected is not None and actual != expected for expected, actual in zip(trailing, value.shape[1:])):
                raise ValueError(f'Replay field {key} has shape {value.shape}')
            if np.dtype(dtype).kind != value.dtype.kind:
                raise ValueError(f'Replay field {key} has dtype {value.dtype}')
        if n and np.any((self.seq_start < 0) | (self.seq_start >= self.seq_end) | (self.seq_end > len(self.track_pos))):
            raise ValueError('Replay rows reference tracks outside the saved tracks')
        if 'step' not in self.provenance or 'volume' not in self.provenance:
            raise ValueError('Replay provenance must record its source step and volume')

    def __len__(self):
        return len(self.pos)

    def observed_prefix(self, j):
        """Actual committed polyline through this decision; no future vertices."""
        start, end = int(self.seq_start[j]), int(self.seq_end[j])
        path = self.track_pos[start:end]
        if not np.allclose(path[-1], self.pos[j], atol=1e-5, rtol=0):
            raise ValueError('Replay prefix does not end at decision head')
        if self.seed_valid[j] and not np.allclose(path[0], self.seed_pos[j], atol=1e-5, rtol=0):
            raise ValueError('Replay prefix does not start at its original seed')
        return path

    def save(self, path):
        np.savez(path, __metadata__=json.dumps(dict(fibers=self.manifest, provenance=self.provenance)),
                 **{k: getattr(self, k) for k in (*self.FIELDS, *self.TRACK)})

    def validate_fibers(self, fibers):
        if self.manifest != fiber_manifest(fibers):
            raise ValueError("On-policy cache has incompatible fibers or geometry; "
                             "recollect with collect.py using the current controlled-span labels")
        if len(self) and (np.any(self.fiber_idx < 0) or np.any(self.fiber_idx >= len(fibers))):
            raise ValueError("On-policy cache contains invalid fiber indices")
        if len(self):
            # Decode only referenced fibers in lazy collections, and compare
            # against the same geometry that supplies the collector's arcs.
            indices, inverse = np.unique(self.fiber_idx, return_inverse=True)
            lengths = np.array([fibers[int(i)].length for i in indices])[inverse]
            if np.any(~np.isfinite(self.t)) or np.any((self.t < 0) | (self.t > lengths)):
                raise ValueError("On-policy cache contains arc positions outside controlled spans")

    @classmethod
    def load(cls, path):
        """``path``: .npz from collect.py (mirrored once to a sibling ``_mmap`` directory of
        .npy files) or such a directory."""
        path = os.fspath(path)
        d = path[:-4] + "_mmap" if path.endswith(".npz") else path
        metadata_path = os.path.join(d, "metadata.json")
        names = (*cls.FIELDS, *cls.TRACK)
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as z:
                metadata = json.loads(str(z["__metadata__"].item()))
                missing = [k for k in names if k not in z.files]
                if missing:
                    raise ValueError(f'Replay archive lacks schema fields {missing}; recollect it')
                # Include archive identity so an overwritten NPZ cannot reuse stale mmap arrays.
                stat = os.stat(path)
                metadata["archive"] = [stat.st_size, stat.st_mtime_ns]
                existing = None
                if os.path.exists(metadata_path):
                    with open(metadata_path) as fh:
                        existing = json.load(fh)
                if existing != metadata or any(not os.path.exists(os.path.join(d, k + ".npy")) for k in names):
                    os.makedirs(d, exist_ok=True)
                    for k in names:
                        np.save(os.path.join(d, k + ".npy"), z[k])
                    with open(metadata_path, "w") as fh:
                        json.dump(metadata, fh)
        obj = cls.__new__(cls)
        obj._open(d)
        return obj

    def _open(self, d):
        with open(os.path.join(d, "metadata.json")) as fh:
            metadata = json.load(fh)
        self.manifest = metadata["fibers"]
        self.provenance = metadata["provenance"]
        self._dir = d
        missing = [k for k in (*self.FIELDS, *self.TRACK) if not os.path.exists(os.path.join(d, k + ".npy"))]
        if missing:
            raise ValueError(f'Replay cache lacks schema fields {missing}; recollect it')
        for k in (*self.FIELDS, *self.TRACK):
            setattr(self, k, np.load(os.path.join(d, k + ".npy"), mmap_mode="r"))
        self.validate()

    def __getstate__(self):
        if self._dir is None:
            return self.__dict__
        return {"_dir": self._dir}

    def __setstate__(self, state):
        if state.get("_dir") and len(state) == 1:
            self._open(state["_dir"])
        else:
            self.__dict__.update(state)


def own_replay_paths(paths, dagger_dir):
    """The replay caches a run collected itself (inside its own ``dagger_dir``); caches of other runs or models are
    dropped with a warning: DAgger states come only from the run's own recent checkpoints."""
    root = Path(dagger_dir).resolve()
    own = [p for p in paths if root in Path(p).resolve().parents]
    for p in paths:
        if p not in own:
            print(f'Warning: dropping replay cache from another run: {p}', flush=True)
    return own


def load_replay(paths):
    """Replay caches at ``paths``; one that cannot be loaded is skipped with a warning."""
    caches = []
    for path in paths:
        try:
            caches.append(OnPolicyStates.load(path))
        except (OSError, ValueError, KeyError) as error:
            print(f'Warning: skipping replay cache {path}: {error}', flush=True)
    return caches


def usable_replay(caches, fibers, n_history, grid_scale):
    """The caches whose fibers, history length and coordinate scale match this run; others are skipped with a
    warning (their states would be labeled against different data)."""
    usable = []
    for op in caches:
        try:
            op.validate_fibers(fibers)
            if op.hist.shape[1] != n_history:
                raise ValueError('replay history length differs from n_history')
            if op.provenance['volume']['grid_scale'] != grid_scale:
                raise ValueError('replay world coordinate scale differs from this run')
        except (ValueError, KeyError) as error:
            print(f'Warning: skipping replay cache {getattr(op, "_dir", "")}: {error}', flush=True)
        else:
            usable.append(op)
    return usable
