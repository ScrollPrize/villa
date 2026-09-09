"""Explicit depth support for centered surface inference and upload verification."""

import hashlib
from pathlib import Path

import numpy as np
import zarr

from vesuvius.ink_detection.inference.infer_surface3d import evaluated_depth_support
from vesuvius.label_zarr import create_v2_array


def ensure_depth_validity(group, shape, patch_depth, margin=0, source_depth_reversed=False):
    start, stop = evaluated_depth_support(shape, (patch_depth, 1, 1), "centered", margin, source_depth_reversed)
    values = np.zeros(shape[0], dtype=np.uint8)
    values[start:stop] = 1
    if "valid_depth" not in group:
        create_v2_array(group, "valid_depth", shape=values.shape, chunks=values.shape,
                        dtype=np.uint8, compressor=None, fill_value=0)
        group["valid_depth"][:] = values
    if not np.array_equal(group["valid_depth"][:], values):
        raise ValueError("Existing depth validity differs from the evaluated window")
    group["valid_depth"].attrs.update({"_ARRAY_DIMENSIONS": ["z"],
        "description": "Broadcast across XY at every image level; 1=evaluated depth, 0=unobserved depth"})
    group.attrs.update({"depth_mode": "centered", "supported_depth_zyx": [start, stop],
        "valid_depth_margin": margin, "input_depth_reversed": source_depth_reversed,
        "validity_array": "valid_depth", "validity_broadcast": "Z vector broadcast across XY at all six levels",
        "invalid_depth_semantics": "Zeros outside the evaluated window are unobserved placeholders, not negative predictions"})


def verified_depth_validity(path, attrs, shape):
    """Return content provenance; reject missing, corrupted or inconsistent support."""
    if attrs.get("depth_mode", "sliding") != "centered":
        return {}
    patch = attrs.get("patch_zyx")
    if not patch or attrs.get("validity_array") != "valid_depth":
        raise ValueError("Missing centered depth validity provenance")
    start, stop = evaluated_depth_support(shape, patch, "centered", attrs.get('valid_depth_margin', 0),
                                         attrs.get('input_depth_reversed', False))
    if attrs.get("supported_depth_zyx") != [start, stop]:
        raise ValueError("Centered depth bounds mismatch")
    support_path = Path(path) / "valid_depth"
    if not (support_path / ".zarray").is_file():
        raise ValueError("Missing centered depth validity array")
    support = zarr.open_array(str(support_path), mode="r")
    expected = np.zeros(shape[0], dtype=np.uint8)
    expected[start:stop] = 1
    if support.shape != expected.shape or support.dtype != np.uint8 or not np.array_equal(support[:], expected):
        raise ValueError("Centered depth validity array mismatch")
    digest = hashlib.sha256()
    for entry in sorted(support_path.rglob("*")):
        if entry.is_file():
            digest.update(str(entry.relative_to(support_path)).encode() + b"\0")
            digest.update(entry.read_bytes())
    return {"depth_mode": "centered", "supported_depth_zyx": [start, stop],
            "auxiliary_arrays": {"valid_depth": {"shape": [shape[0]], "sha256": digest.hexdigest()}}}
