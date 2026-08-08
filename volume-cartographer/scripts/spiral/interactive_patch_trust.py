"""Trust routing for patches appended to a resident Spiral fit."""

import numpy as np


def select_interactive_patch_pool(
        record, verified_patches, unverified_patches,
        pending_verified_patches, pending_unverified_patches):
    """Return the only pool an uploaded patch may enter.

    The service validates this field before staging, but the resident fitter
    treats its input records as untrusted too. Keeping duplicate detection
    cross-pool prevents an identifier from changing trust level by shadowing a
    source patch, including one filtered out of the resident training pool.
    """
    input_id = record.get('id')
    if not isinstance(input_id, str) or not input_id:
        raise RuntimeError('An interactive patch record has no valid id')
    classification = record.get('classification', 'verified')
    if classification not in ('verified', 'unverified'):
        raise RuntimeError(
            f'Patch {input_id!r} has invalid classification {classification!r}')
    if any(input_id in patches for patches in (
            verified_patches, unverified_patches,
            pending_verified_patches, pending_unverified_patches)):
        raise RuntimeError(f'Patch {input_id!r} is already part of this session')
    target = (pending_verified_patches if classification == 'verified'
              else pending_unverified_patches)
    return classification, target


def collect_trusted_collection_points(collections, z_begin, z_end):
    """Return finite in-ROI PCL points in the fitter's z-y-x order."""
    chunks = []
    for pcl in collections.values():
        points = pcl.get('points', {})
        if not points:
            continue
        points_xyz = []
        for point in points.values():
            xyz = np.asarray(point.get('p'), dtype=np.float32)
            if xyz.shape != (3,):
                raise RuntimeError('Point-collection coordinates must be x-y-z triples')
            points_xyz.append(xyz)
        points_zyx = np.stack(points_xyz, axis=0)[:, ::-1]
        in_roi = (
            np.isfinite(points_zyx).all(axis=1)
            & (points_zyx[:, 0] >= z_begin)
            & (points_zyx[:, 0] < z_end))
        if in_roi.any():
            chunks.append(points_zyx[in_roi])
    return (np.concatenate(chunks, axis=0)
            if chunks else np.empty((0, 3), dtype=np.float32))
