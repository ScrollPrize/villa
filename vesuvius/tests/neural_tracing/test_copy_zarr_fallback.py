"""Network-marked check of the zarr fallback reader in infer_rowcol_triplet_wraps.py (the path used when the
monorepo-only ``vc`` package is not installed).

Reads five small level-0 crops of the public PHerc0841 9.366 um volume through ``_open_zarr_volume_level`` and
compares them byte for byte with the raw chunks fetched over HTTPS and decoded with ``np.frombuffer`` (no zarr, no
vc). The fallback is called directly, so the test also runs where ``vc`` is installed. Skipped by default; run with:

    uv run pytest -m 'slow and network' \
        tests/neural_tracing/test_copy_zarr_fallback.py -v
"""

from __future__ import annotations

import itertools
import json
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps import _open_zarr_volume_level


pytestmark = [pytest.mark.slow, pytest.mark.network]

S3_URL = "s3://vesuvius-challenge-open-data/PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr"
HTTPS_URL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr"
)

# level-0 crops as ((z, y, x) start, (z, y, x) size)
CROPS = {
    "inside": ((2344, 3368, 3240), (32, 40, 48)),  # inside one chunk
    "chunk_edge": ((2412, 3436, 3308), (40, 40, 40)),  # across a corner shared by 8 chunks
    "missing_chunk": ((16512, 2800, 4736), (32, 32, 32)),  # half in a chunk the store does not have
    "vol_edge_below": ((-16, 4500, 4960), (32, 40, 40)),  # starts below z = 0
    "vol_edge_past": ((7500, 3300, 7680), (32, 32, 40)),  # runs past the last x voxel
}
HAS_DATA = {"inside", "chunk_edge", "missing_chunk", "vol_edge_below"}


def _get(url):
    """Bytes at url; None for a chunk the store does not have (403/404)."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                return None
            if attempt == 4:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise
        time.sleep(2**attempt)


@pytest.fixture(scope="module")
def zarray():
    meta = json.loads(_get(f"{HTTPS_URL}/0/.zarray"))
    # the raw reference decodes chunks with np.frombuffer, which is only valid for this encoding
    assert meta["zarr_format"] == 2
    assert meta["dtype"] == "|u1"
    assert meta["compressor"] is None
    assert meta.get("filters") is None
    assert meta.get("dimension_separator") == "/"
    assert meta.get("order", "C") == "C"
    return meta


@pytest.fixture(scope="module")
def level0():
    volume, level = _open_zarr_volume_level(S3_URL, 0)
    assert level == 0
    return volume


def _raw_crop(meta, start, size):
    shape = np.array(meta["shape"])
    chunks = np.array(meta["chunks"])
    start = np.array(start)
    size = np.array(size)
    out = np.full(tuple(size), meta["fill_value"] or 0, dtype=np.uint8)
    lo = np.maximum(start, 0)
    hi = np.minimum(start + size, shape)
    if np.any(hi <= lo):
        return out
    ranges = [range(lo[i] // chunks[i], (hi[i] - 1) // chunks[i] + 1) for i in range(3)]
    for key in itertools.product(*ranges):
        raw = _get(f"{HTTPS_URL}/0/" + "/".join(str(k) for k in key))
        if raw is None:
            continue
        chunk = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(chunks))
        base = np.array(key) * chunks
        a = np.maximum(lo, base)
        b = np.minimum(hi, base + chunks)
        out[tuple(slice(a[i] - start[i], b[i] - start[i]) for i in range(3))] = chunk[
            tuple(slice(a[i] - base[i], b[i] - base[i]) for i in range(3))
        ]
    return out


def test_level0_metadata(level0, zarray):
    assert level0.shape == tuple(zarray["shape"])
    assert level0.chunks == tuple(zarray["chunks"])
    assert level0.dtype == np.uint8


@pytest.mark.parametrize("name", sorted(CROPS))
def test_crop_matches_raw_chunks(level0, zarray, name):
    start, size = CROPS[name]
    want = _raw_crop(zarray, start, size)
    if name in HAS_DATA:
        assert want.any(), f"reference crop {name} is all fill; the crop no longer covers scroll data"
    got = level0[tuple(slice(s, s + n) for s, n in zip(start, size))]
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)
