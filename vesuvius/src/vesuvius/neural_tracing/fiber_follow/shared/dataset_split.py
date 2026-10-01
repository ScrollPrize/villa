"""Stable whole-fiber splits, independent of spatial position and draw order."""
import hashlib
import math


def heldout_ids(ids, validation):
    if validation.get('strategy') != 'fiber_hash':
        raise ValueError('Expected validation.strategy="fiber_hash"')
    fraction = float(validation['fraction'])
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    ids = list(ids)
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError('Fiber split needs at least two unique IDs')
    seed = int(validation['seed'])
    order = sorted(ids, key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).digest())
    count = min(len(ids)-1,max(1,round(len(ids)*fraction)))
    return set(order[:count])
