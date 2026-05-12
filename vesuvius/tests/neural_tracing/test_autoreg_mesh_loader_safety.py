"""Tests that ChunkPatch and related dataset state are spawn-safe.

The prior autoreg_mesh node-nuke was triggered by ChunkPatch.volume holding a
zarr.Group backed by an FSStore over S3, which (a) got deep-pickled to each
spawned DataLoader worker and (b) carried boto3 / aiohttp thread state across
the spawn boundary. The fix is to attach only the volume_path string, then
resolve lazily per-PID via get_or_open_zarr_group. These tests guard against
regressions in pickle size and the lazy-resolution contract.
"""
from __future__ import annotations

import pickle

import numpy as np

from vesuvius.neural_tracing.datasets.common import (
    ChunkPatch,
    get_or_open_zarr_group,
)


def _make_patch_with_path() -> ChunkPatch:
    return ChunkPatch(
        chunk_id=(0, 0, 0),
        volume_path="s3://example/zarr-store",
        scale=0,
        world_bbox=(0.0, 64.0, 0.0, 64.0, 0.0, 64.0),
        wraps=[],
        segments=[],
    )


def test_chunkpatch_pickle_small_when_using_volume_path():
    """Pickling a ChunkPatch with only volume_path (no live volume handle)
    must be a few bytes, not megabytes. Anything larger means a zarr.Group
    snuck into a pickled field again."""
    patch = _make_patch_with_path()
    blob = pickle.dumps(patch)
    assert len(blob) < 2_000, f"ChunkPatch pickle too large: {len(blob)} bytes"


def test_chunkpatch_lazy_volume_unopened_before_access():
    """The lazy field volume_path should NOT trigger a zarr open at
    construction time."""
    patch = _make_patch_with_path()
    # The `volume` field is the legacy-only direct handle; with volume_path
    # set, it should remain None until get_volume() is called.
    assert patch.volume is None
    assert patch.volume_path == "s3://example/zarr-store"


def test_chunkpatch_legacy_volume_kwarg_still_works():
    """Existing tests pass volume=<np.ndarray>; that must still work."""
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    patch = ChunkPatch(
        chunk_id=(0, 0, 0),
        volume=arr,
        scale=0,
        world_bbox=(0.0, 4.0, 0.0, 4.0, 0.0, 4.0),
        wraps=[],
        segments=[],
    )
    # get_volume() returns the legacy handle when volume_path is None.
    assert patch.get_volume() is arr
    assert patch.volume_path is None


def test_get_or_open_zarr_group_cache_per_pid(tmp_path):
    """Two calls with the same path return the same group object within
    the same process (cached). After a PID change (simulated), the cache
    resets."""
    import zarr

    store = zarr.DirectoryStore(str(tmp_path / "z"))
    root = zarr.group(store=store, overwrite=True)
    root.create_dataset("0", shape=(4, 4, 4), dtype=np.uint8)

    g1 = get_or_open_zarr_group(str(tmp_path / "z"))
    g2 = get_or_open_zarr_group(str(tmp_path / "z"))
    # Same call -> same cached object.
    assert g1 is g2

    # Simulate a PID change to force cache reset and re-open. We tweak the
    # module-level _ZARR_CACHE_PID so the next call rebuilds the cache.
    import vesuvius.neural_tracing.datasets.common as common_mod
    common_mod._ZARR_CACHE_PID = -1
    g3 = get_or_open_zarr_group(str(tmp_path / "z"))
    assert g3 is not g1  # fresh open after PID change


def test_chunkpatch_get_volume_via_volume_path(tmp_path):
    """When volume_path is set, get_volume() must lazy-open the zarr store
    using the same helper used in the DataLoader workers."""
    import zarr

    store_path = tmp_path / "vol"
    store = zarr.DirectoryStore(str(store_path))
    root = zarr.group(store=store, overwrite=True)
    root.create_dataset("0", shape=(2, 2, 2), dtype=np.uint8)

    patch = ChunkPatch(
        chunk_id=(0, 0, 0),
        volume_path=str(store_path),
        scale=0,
        world_bbox=(0.0, 2.0, 0.0, 2.0, 0.0, 2.0),
        wraps=[],
        segments=[],
    )
    grp = patch.get_volume()
    assert isinstance(grp, zarr.Group)
    # repeat -> same cached object
    assert patch.get_volume() is grp
