"""Memory-conscious checkpoint loading helpers for Spiral fits."""

from __future__ import annotations

import torch

from checkpoint_migrations import migrate_legacy_gap_parameterization


def load_checkpoint_cpu(path):
    """Load a modern checkpoint with lazily mapped CPU tensor storages."""
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    return migrate_legacy_gap_parameterization(checkpoint)


# Frozen-epoch snapshot fields that shape the baked transform besides its
# parameters (see constraint_baking.snapshot_frozen_epoch). Listed here so
# the digest below stays aligned with snapshots_match().
_FROZEN_EPOCH_TENSOR_KEYS = (
    "umbilicus_zyx", "flow_min_corner_zyx", "flow_max_corner_zyx")
_FROZEN_EPOCH_SCALAR_KEYS = (
    "spiral_outward_sense", "flow_integration_steps",
    "flow_integration_solver", "model_config")


def _feed_tensor(digest, name, value):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest.update(
        f"{name}|{tensor.dtype}|{tuple(tensor.shape)}\n".encode("utf-8"))
    if tensor.numel():
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())


def _feed_state_dict(digest, prefix, state_dict):
    for key in sorted(state_dict):
        value = state_dict[key]
        if torch.is_tensor(value):
            _feed_tensor(digest, f"{prefix}{key}", value)
        else:
            digest.update(
                f"{prefix}{key}|{value!r}\n".encode("utf-8"))


def model_state_sha256(state_dict, frozen_epochs=(), z_begin=None,
                       z_end=None):
    """Content digest of everything that places the fitted surface.

    The live parameters alone do not determine the surface a fit publishes:
    after a constraint-bake reset the surface is pulled back through the
    frozen-epoch stack, and the run window bounds it. A checkpoint and a
    preview export that agree on this digest were produced from the same
    surface, which is what lets a service re-show the preview it already
    flattened for a checkpoint instead of flattening it again.
    """
    import hashlib
    import json

    digest = hashlib.sha256()
    digest.update(f"z_window|{z_begin!r}|{z_end!r}\n".encode("utf-8"))
    _feed_state_dict(digest, "live/", state_dict)
    for index, snapshot in enumerate(frozen_epochs or ()):
        prefix = f"frozen/{index}/"
        _feed_state_dict(
            digest, prefix + "state/", snapshot["spiral_and_transform"])
        for key in _FROZEN_EPOCH_TENSOR_KEYS:
            if snapshot.get(key) is not None:
                _feed_tensor(digest, prefix + key, snapshot[key])
        scalars = {key: snapshot.get(key) for key in _FROZEN_EPOCH_SCALAR_KEYS}
        digest.update(
            (prefix + json.dumps(scalars, sort_keys=True, default=repr)
             + "\n").encode("utf-8"))
    return digest.hexdigest()
