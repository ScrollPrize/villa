"""Metadata policy for new geometry derived from a selected object.

This is not a copy API: relationships, history, manual extension rows and
artifact payloads require the store's copy/export infrastructure.
"""
from copy import deepcopy


def derived_metadata(metadata, *, policy='portable', reference=None):
    """Copy application metadata, with an explicit policy for managed payloads.

    ``portable`` refuses extension/artifact descriptors: retaining their JSON
    without their payloads would produce a detached or misleading result.
    ``geometry_only`` intentionally omits those two managed fields for a new
    geometry result. It records that choice and the selected source reference;
    all other application metadata remains intact. Neither policy reads files
    or resolves storage locations.
    """
    if policy not in ('portable', 'geometry_only'):
        raise ValueError('metadata policy must be portable or geometry_only')
    managed = ('vcobj_extensions', 'artifacts')
    if policy == 'portable' and any(key in metadata for key in managed):
        raise ValueError('Managed extension/artifact data require a store-preserving copy, '
                         'or explicit metadata_policy="geometry_only" for derived geometry')
    result = deepcopy(metadata)
    if policy == 'geometry_only':
        for key in managed:
            result.pop(key, None)
        result['derivation'] = {'policy': policy, 'source': deepcopy(reference)}
    return result
