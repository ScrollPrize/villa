"""Stable whole-fiber splits, independent of spatial position and draw order."""
import hashlib
import math


def validate_split_policy(validation):
    if validation.get('strategy') != 'fiber_hash':
        raise ValueError('Expected validation.strategy="fiber_hash"')
    if ('fraction' in validation) == ('count' in validation):
        raise ValueError('Specify exactly one validation fraction or count')
    if 'count' in validation:
        count = validation['count']
        if type(count) is not int or count < 1:
            raise ValueError('Validation count must be a positive integer')
    else:
        fraction = float(validation['fraction'])
        if not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError('Validation fraction must be between zero and one')
    int(validation['seed'])


def heldout_ids(ids, validation):
    validate_split_policy(validation)
    ids = list(ids)
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError('Fiber split needs at least two unique IDs')
    seed = int(validation['seed'])
    order = sorted(ids, key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).digest())
    count = (validation['count'] if 'count' in validation else
             min(len(ids)-1,max(1,round(len(ids)*float(validation['fraction'])))))
    if count >= len(ids):
        raise ValueError('Validation count must leave at least one training fiber')
    return set(order[:count])
