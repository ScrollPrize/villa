"""Live open-data check: reading a downscaled tifxyz_normalized mesh must warn and name
the factor; a correctly scaled control must not warn.

Downloads ~2 small meshes and the catalog anonymously from the open-data bucket.
Run with VESUVIUS_LIVE_OPEN_DATA=1.
"""
import os
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pytest

from vesuvius.tifxyz import read_tifxyz

pytestmark = pytest.mark.skipif(
    os.environ.get("VESUVIUS_LIVE_OPEN_DATA") != "1",
    reason="set VESUVIUS_LIVE_OPEN_DATA=1 for live open-data tests",
)

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
FILES = ("meta.json", "x.tif", "y.tif", "z.tif")
# original_volume_downscale = 4 in the catalog; meta.json does not record it.
SCALED_SEG = "PHerc0139/segments/20260422000000-title_2026042222_zmid_flatboi"
SCALED_REF = SCALED_SEG + "/mesh/20260422000000-on-20260102150214-2.399um.tifxyz"
# original_volume_downscale = 1: coordinates are already original-volume voxels.
CONTROL_SEG = "PHerc0343P/segments/20250902170435--5_b2"
NORMALIZED = "/mesh/intermediate/tifxyz_normalized"


def _fetch(key, root):
    dst = root / key
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BUCKET}/{key}", dst)
    return dst


def _mesh(key, root):
    for f in FILES:
        _fetch(f"{key}/{f}", root)
    return root / key


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    root = Path(os.environ.get("VESUVIUS_LIVE_CACHE") or tmp_path_factory.mktemp("open-data"))
    catalog = _fetch("metadata.json", root)
    os.environ["VESUVIUS_OPEN_DATA_CATALOG"] = str(catalog)
    yield root
    os.environ.pop("VESUVIUS_OPEN_DATA_CATALOG", None)


def _read(path, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        surf = read_tifxyz(path, **kw)
    return surf, [w for w in rec if "original_volume_downscale" in str(w.message)]


def _valid_extent(surf):
    # bbox is computed from the valid points only.
    return np.asarray(surf.bbox, dtype=np.float64)


def test_downscaled_mesh_warns_with_factor(live):
    path = _mesh(SCALED_SEG + NORMALIZED, live)
    _, warns = _read(path)
    assert warns, "no warning for a mesh stored at 1/4 of its original volume"
    assert "original_volume_downscale=4" in str(warns[0].message)


def test_catalog_factor_matches_published_mesh(live):
    path = _mesh(SCALED_SEG + NORMALIZED, live)
    ref = _mesh(SCALED_REF, live)
    fixed, warns = _read(path, downscale="catalog")
    assert not warns
    published, _ = _read(ref)
    # The published original-volume mesh is the normalized mesh x4, regridded; extents agree
    # to within one source cell (20 voxels at scale 0.05, x4 = 80 voxels).
    assert np.allclose(_valid_extent(fixed), _valid_extent(published), atol=80)
    unscaled, _ = _read(path, downscale=1)
    assert not np.allclose(_valid_extent(unscaled), _valid_extent(published), atol=80)


def test_correctly_scaled_control_does_not_warn(live):
    path = _mesh(CONTROL_SEG + NORMALIZED, live)
    _, warns = _read(path)
    assert not warns
