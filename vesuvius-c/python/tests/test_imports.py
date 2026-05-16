"""Import-time smoke tests for the vesuvius-c Python bindings.

These verify the contract that survives without a network round-trip:

1. The module imports cleanly.
2. ``libvesuvius.so`` loads via ctypes.
3. The Python ``ctypes.Structure`` field layouts produce the same
   ``sizeof`` as the C structs declared in ``vesuvius-c/vesuvius-c.h``.

The tests skip cleanly if ``libvesuvius.so`` isn't built — building is a
separate concern handled by ``setup.py`` (which shells out to gcc and
needs ``libcurl-dev``, ``libblosc2-dev``, ``libjson-c-dev``). When run
after ``pip install .``, the fixture finds the built ``.so`` and the
tests run.
"""

from __future__ import annotations

import ctypes
import os
import sys

import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)


@pytest.fixture(scope="module")
def vesuvius_c():
    """Import the wrapper if ``libvesuvius.so`` is already built; skip otherwise."""
    so_path = os.path.join(_PKG_DIR, "libvesuvius.so")
    if not os.path.exists(so_path):
        pytest.skip(
            "libvesuvius.so not built; run `pip install .` from "
            "vesuvius-c/python/ first (requires libcurl-dev, libblosc2-dev, "
            "libjson-c-dev)."
        )
    if _PKG_DIR not in sys.path:
        sys.path.insert(0, _PKG_DIR)
    import vesuvius_c  # noqa: E402  (path is injected above)

    return vesuvius_c


def test_module_loads_and_exposes_public_api(vesuvius_c):
    assert vesuvius_c._lib is not None, "ctypes.CDLL did not produce a library handle"
    assert hasattr(vesuvius_c, "VesuviusVolume"), "VesuviusVolume class missing"
    assert callable(vesuvius_c.VesuviusVolume), "VesuviusVolume is not callable"


def test_structure_sizes_match_c_layout(vesuvius_c):
    """ctypes.Structure sizes must equal the C-side ``sizeof`` of each mirrored struct.

    Expected values are derived from ``vesuvius-c/vesuvius-c.h``:

    * ``volume``: ``char[1024] + char[1024] + zarr_metadata`` = 1024 + 1024 + 124 = 2172
    * ``zarr_metadata``: ``int32[3] + int32[3] + zarr_compressor_settings(76)
      + char[8] + int32 + char + 3 pad + int32 + char + 3 pad`` = 124
    * ``chunk``: ``int[3]`` (flexible-array ``data`` is excluded from sizeof) = 12

    A mismatch here means the ctypes mirror has drifted from the C
    declaration and chunk reads will return garbage.
    """
    assert ctypes.sizeof(vesuvius_c.Volume) == 2172
    assert ctypes.sizeof(vesuvius_c.ZarrMetadata) == 124
    assert ctypes.sizeof(vesuvius_c.Chunk) == 12


def test_c_symbol_signatures_attached(vesuvius_c):
    """Confirm the six C functions the wrapper uses have ctypes signatures set."""
    lib = vesuvius_c._lib
    for name in (
        "vs_zarr_parse_zarray",
        "vs_zarr_read_chunk",
        "vs_chunk_free",
        "vs_vol_new",
        "vs_vol_get_chunk",
        "vs_vol_free",
    ):
        fn = getattr(lib, name)
        assert fn.argtypes is not None, f"{name}.argtypes was not configured"
        assert fn.restype is not None or name in ("vs_chunk_free", "vs_vol_free"), (
            f"{name}.restype was not configured"
        )
