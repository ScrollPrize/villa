"""Paths recorded in frozen artifacts (seed manifests, checkpoints, configs) on another machine.

A path map ``OLD=NEW`` pairs (the ``TRACERS_PATH_MAP='OLD=NEW;OLD2=NEW2'`` variable, formerly
``FIBER_FOLLOW_PATH_MAP``, or a tool's ``--path-map OLD=NEW`` options) relocates a recorded path whose leading
components are OLD to NEW (longest prefix first) before it is compared with the paths of this run; the comparison
itself stays exact. Without one, recorded paths are used unchanged. Paths recorded under a tracer's former package
location (``MOVES``) are read as the same path under its current one.
"""
import os
from pathlib import PurePosixPath

MOVES = ((('neural_tracing', 'fiber_follow'), ('neural_tracing', 'tracers', 'fibers')),
         (('neural_tracing', 'sheet_tracing'), ('neural_tracing', 'tracers', 'sheets')))


def moved_path(path):
    """``path`` with a tracer package's former location replaced by its current one."""
    parts = PurePosixPath(str(path)).parts
    for old, new in MOVES:
        for i in range(len(parts)-len(old)+1):
            if parts[i:i+len(old)] == old:
                return str(PurePosixPath(*parts[:i], *new, *parts[i+len(old):]))
    return str(path)


def parse_path_map(items):
    """``OLD=NEW`` texts -> relocation pairs, longest OLD first."""
    pairs = []
    for item in items:
        old, sep, new = item.partition('=')
        if not sep or not old or not new:
            raise ValueError(f'Path map entries must be OLD=NEW: {item!r}')
        pairs.append((PurePosixPath(moved_path(old)), PurePosixPath(moved_path(new))))
    return sorted(pairs, key=lambda pair: len(pair[0].parts), reverse=True)


def path_map(value=None):
    """The relocation pairs of ``value`` (``OLD=NEW;...``), by default of the environment."""
    if value is None:
        value = os.environ.get('TRACERS_PATH_MAP', os.environ.get('FIBER_FOLLOW_PATH_MAP', ''))
    return parse_path_map(filter(None, (part.strip() for part in value.split(';'))))


def recorded_path(path, mapping=None):
    """``path`` as recorded elsewhere, relocated to this machine."""
    path = PurePosixPath(moved_path(path))
    for old, new in path_map() if mapping is None else mapping:
        if path == old or old in path.parents:
            return str(new/path.relative_to(old))
    return str(path)


def relocated_volume_key(key):
    """A CT normalization key ``<path>::<level>`` recorded elsewhere, relocated (remote stores unchanged)."""
    path, sep, level = str(key).rpartition('::')
    if not sep or '://' in path:
        return str(key)
    return recorded_path(path)+sep+level


def relocate_checkpoint(ck):
    """Relocate the local paths a checkpoint records for its volume and CT normalization, in place."""
    spec = ck.get('vol_spec') or {}
    for key in ('ct_zarr', 'cache_dir'):
        if isinstance(spec.get(key), str) and '://' not in spec[key]:
            spec[key] = recorded_path(spec[key])
    if isinstance(spec.get('ct_normalization'), dict) and 'volume' in spec['ct_normalization']:
        spec['ct_normalization']['volume'] = relocated_volume_key(spec['ct_normalization']['volume'])
    normalization = ck.get('ct_normalization')
    if isinstance(normalization, dict) and isinstance(normalization.get('volumes'), dict):
        records = {}
        for key, record in normalization['volumes'].items():
            if isinstance(record, dict) and 'volume' in record:  # each record names its volume too
                record = dict(record, volume=relocated_volume_key(record['volume']))
            records[relocated_volume_key(key)] = record
        normalization['volumes'] = records
    return ck
