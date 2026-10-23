"""Normalize Zarr v3 metadata using the shared native reader."""
import json
from .. import _reader


def normalize(metadata):
    if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "array":
        raise ValueError("expected a Zarr v3 array")
    return json.loads(_reader.normalize_zarr_metadata(json.dumps(metadata)))
