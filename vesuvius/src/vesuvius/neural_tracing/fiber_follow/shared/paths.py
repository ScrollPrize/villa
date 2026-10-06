"""Paths recorded in frozen artifacts (seed manifests, checkpoints) on another machine.

``FIBER_FOLLOW_PATH_MAP='OLD=NEW;OLD2=NEW2'`` relocates a recorded path whose prefix is OLD to NEW
(longest prefix first) before it is compared with the paths of this run; the comparison itself stays
exact. Without the variable, recorded paths are used unchanged.
"""
import os
from pathlib import PurePosixPath


def path_map(value=None):
    value = os.environ.get('FIBER_FOLLOW_PATH_MAP', '') if value is None else value
    pairs = []
    for item in filter(None, (part.strip() for part in value.split(';'))):
        old, sep, new = item.partition('=')
        if not sep or not old or not new:
            raise ValueError(f'FIBER_FOLLOW_PATH_MAP entries must be OLD=NEW: {item!r}')
        pairs.append((PurePosixPath(old), PurePosixPath(new)))
    return sorted(pairs, key=lambda pair: len(pair[0].parts), reverse=True)


def recorded_path(path, mapping=None):
    """``path`` as recorded elsewhere, relocated to this machine."""
    path = PurePosixPath(str(path))
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
    for key in ('fiber_zarr_dir', 'ct_zarr', 'cache_dir'):
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
