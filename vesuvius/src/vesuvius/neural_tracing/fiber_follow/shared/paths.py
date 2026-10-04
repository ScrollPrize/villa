"""Paths recorded in frozen artifacts (seed manifests, negative-bank runs) on another machine.

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
