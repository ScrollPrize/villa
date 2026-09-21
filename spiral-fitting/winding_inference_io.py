"""Shared checksums and metadata for exported winding crossing stores."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


ARTIFACT_TYPE = "winding_inference_crossings"
FORMAT_VERSION = 1


def _canonical_digest(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_array(root: Path, description: dict, *, verify: bool) -> np.ndarray:
    path = root / description["file"]
    if verify and _sha256(path) != description["sha256"]:
        raise ValueError(f"winding-inference array checksum mismatch: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(value.shape) != list(description["shape"]):
        raise ValueError(f"winding-inference array shape mismatch: {path}")
    if np.dtype(value.dtype).str != str(description["dtype"]):
        raise ValueError(f"winding-inference array dtype mismatch: {path}")
    return value


def manifest_fingerprint(manifest):
    identity_view = copy.deepcopy(manifest)
    identity_view.pop("fingerprint", None)
    identity_view.pop("elapsed_seconds", None)
    identity_view.pop("export_workers", None)
    identity_view.pop("rays_per_task", None)
    for shard in identity_view.get("shards", []):
        shard.pop("elapsed_seconds", None)
    return _canonical_digest(identity_view)
