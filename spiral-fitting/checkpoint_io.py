"""Memory-conscious checkpoint loading helpers for Spiral fits."""

from __future__ import annotations

import torch


def load_checkpoint_cpu(path):
    """Load a modern checkpoint with lazily mapped CPU tensor storages."""
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )


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
                       z_end=None, flow_interpolant=None):
    """Content digest of everything that places the fitted surface.

    The live parameters alone do not determine the surface a fit publishes:
    after a constraint-bake reset the surface is pulled back through the
    frozen-epoch stack, and the run window bounds it. A checkpoint and a
    preview export that agree on this digest were produced from the same
    surface, which is what lets a service re-show the preview it already
    flattened for a checkpoint instead of flattening it again.

    ``flow_interpolant`` is the Cartesian flow field's interpolant
    (SpiralAndTransform.flow_interpolant): the same parameters place another
    surface under the upsampled field than under direct low-resolution
    sampling. Either one is fed. Before it was recorded, the digest of the
    same parameters was the same for both (CUDA stamped it on directly
    sampled surfaces, MPS and the CPU on upsampled ones), so no Cartesian
    digest equals an unrecorded one: a surface is never matched to a preview,
    commit or checkpoint digest of the other interpolant, at the cost of one
    re-flatten and re-commit per older surface. The other field types have a
    single interpolant (None) and keep the digests they always had.
    """
    import hashlib
    import json

    digest = hashlib.sha256()
    digest.update(f"z_window|{z_begin!r}|{z_end!r}\n".encode("utf-8"))
    if flow_interpolant is not None:
        if flow_interpolant not in ("direct_lr", "upsampled"):
            raise ValueError(f"unknown flow interpolant {flow_interpolant!r}")
        digest.update(f"flow_interpolant|{flow_interpolant}\n".encode("utf-8"))
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


# The checkpoint entry naming the flow field's interpolant (see
# checkpoint_flow_interpolant).
FLOW_INTERPOLANT_KEY = "flow_field_interpolant"


def checkpoint_flow_interpolant(checkpoint):
    """``(interpolant, note)``: the interpolant the checkpoint's flow field
    was fitted with, which every reconstruction must keep.

    Checkpoints record it (FLOW_INTERPOLANT_KEY) since the Metal kernels.
    For an older one it is what main integrated where the checkpoint was
    written, read from its RNG states: model_flow_field_direct_lr sampled the
    coarse lattice directly only in CUDA's Triton kernel, and fell back to the
    upsampled field on MPS (a checkpoint with ``torch_mps_rng_state``) and on
    the CPU (one written without CUDA RNG states). ``note`` explains such an
    inference, and is None otherwise. ``interpolant`` is None for the field
    types that have a single interpolant.
    """
    config = checkpoint.get("cfg") or {}
    field_type = config.get("model_flow_field_type",
                            config.get("flow_field_type", "cartesian"))
    if field_type != "cartesian":
        return None, None
    recorded = checkpoint.get(FLOW_INTERPOLANT_KEY)
    if recorded is not None:
        if recorded not in ("direct_lr", "upsampled"):
            raise ValueError(
                f"checkpoint records an unknown flow interpolant {recorded!r}")
        return recorded, None
    if not config.get("model_flow_field_direct_lr", False):
        return "upsampled", None
    if checkpoint.get("torch_mps_rng_state") is not None:
        return "upsampled", (
            "written on MPS before the Metal kernels, where "
            "model_flow_field_direct_lr fell back to the upsampled field")
    if not checkpoint.get("torch_cuda_rng_states"):
        return "upsampled", (
            "written without CUDA, where model_flow_field_direct_lr fell back "
            "to the upsampled field")
    return "direct_lr", (
        "inferred, not recorded: written on a CUDA machine, where the Triton "
        "kernel sampled the coarse lattice directly; a fit there that ran "
        "without it (FIT_SPIRAL_DEVICE=cpu, FIT_SPIRAL_DIRECT_LR=0, "
        "FIT_SPIRAL_TRITON=0 or no Triton) integrated the upsampled field: "
        "set FIT_SPIRAL_DIRECT_LR=0 to keep that")
