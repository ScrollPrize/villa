"""Read bounded final rays without silently masking invalid pixels."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .slab import CONTRACT, report_grid

def read_capture(path):
    path = Path(path)
    st = path.stat()
    if getattr(st, "st_flags", 0) & 0x40000000 or st.st_size > 64 * 1024**2:
        raise ValueError("Refusing offloaded or oversized capture")
    raw = path.read_bytes()
    def reject_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")
    def no_duplicate_keys(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError(f"Duplicate capture key: {k}")
            result[k] = v
        return result
    data = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=no_duplicate_keys)
    for key, value in {
        "schema": "vc-sampling-grid-v1",
        "stage": "final-base-and-dirs-before-readMultiSlice",
        "pixel_order": "row-major-baseXYZ-directionXYZ",
        "units": "level-index-voxel",
        "between_pixel_interpolation": "NOT_SPECIFIED_BY_RENDERER",
    }.items():
        if data.get(key) != value:
            raise ValueError(f"Unsupported capture {key}")
    shape = data.get("shape_hw")
    if (not isinstance(shape, list) or len(shape) != 2 or
        any(type(x) is not int or x < 2 for x in shape) or shape[0] * shape[1] > 512 * 512):
        raise ValueError("Invalid or oversized capture shape")
    level = data.get("level")
    if type(level) is not int or level < 0:
        raise ValueError("Invalid capture level")
    crop = data.get("crop_xy")
    if not isinstance(crop, list) or len(crop) != 2 or any(type(x) is not int or x < 0 for x in crop):
        raise ValueError("Invalid crop origin")
    if type(data.get("invalid_pixels")) is not int or data["invalid_pixels"] != 0:
        raise ValueError("Invalid pixels remain; no silent masking or interpolation")
    pixels = np.asarray(data.get("pixels"), dtype=np.float64)
    if pixels.shape != (shape[0] * shape[1], 6) or not np.isfinite(pixels).all():
        raise ValueError("Incomplete/nonfinite pixel rays")
    offsets = np.asarray(data.get("offsets"), dtype=np.float64)
    if offsets.ndim != 1 or not 1 <= offsets.size <= 65536 or not np.isfinite(offsets).all():
        raise ValueError("Invalid offset vector")
    # Producer serializes float32 at max_digits10: recover the actual stored
    # renderer values, not the slightly different decimal-to-float64 values.
    with np.errstate(over="ignore", invalid="ignore"):
        pixels32, offsets32 = pixels.astype(np.float32), offsets.astype(np.float32)
    if not np.isfinite(pixels32).all() or not np.isfinite(offsets32).all():
        raise ValueError("Values outside float32 range")
    grid = pixels32.astype(np.float64).reshape(*shape, 6)
    return data, grid[..., :3], grid[..., 3:], offsets32.astype(np.float64), hashlib.sha256(raw).hexdigest()


def report_capture(path, reference):
    data, q, n, offsets, digest = read_capture(path)
    low, high = float(offsets.min()), float(offsets.max())
    result = report_grid(q, n, reference, low, high, contract=CONTRACT,
                         units=f"level-{data['level']}-index-voxel")
    result["renderer_interpolation_match"] = "VERTEX_RAYS_ONLY; between-pixel interpolation is a diagnostic surrogate"
    result["capture"] = {
        "sha256": digest, "path": str(path), "schema": data["schema"],
        "stage": data["stage"], "level": data["level"], "crop_xy": data["crop_xy"],
        "offset_count": len(offsets), "offsets_sha256_float32_le": hashlib.sha256(offsets.astype("<f4").tobytes()).hexdigest(),
        "interval_rule": "minimum/maximum of actual offsets, including accumulation; no recentering",
        "full_renderer_execution": "NOT_ATTESTED_BY_THIS_READER; preserve the producing command/log separately",
        "quantization": "original producer float32 recovered before float64 diagnostic",
    }
    for item in result["diagonals"].values():
        for witness in item["failure_witnesses"]:
            witness["canvas_cell_xy"] = [data["crop_xy"][0] + witness["cell_col"],
                                           data["crop_xy"][1] + witness["cell_row"]]
    return result
