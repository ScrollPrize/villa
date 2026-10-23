"""Volume.download_inklabel builds the ink label URL with '/' on every platform.

It used os.path.join, which is ntpath.join on Windows, so the URL became
``https://.../54keV_7.91um\\20230827161847_inklabels.png``. The request failed,
the exception was only printed in verbose mode, and ``Volume.inklabel`` was
silently replaced by an all-zero placeholder of the segment's shape.
"""

from __future__ import annotations

import contextlib
import io
import ntpath
import types

import numpy as np
import pytest
from PIL import Image

import vesuvius.data.volume as volume_mod
from vesuvius.data.volume import Volume

SEGMENT_URL = "https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/20230827161847.zarr/"
EXPECTED_URL = "https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/20230827161847_inklabels.png"


def _segment(url: str) -> Volume:
    vol = Volume.__new__(Volume)
    vol.type = "segment"
    vol.url = url
    vol.verbose = False
    vol.anon = False
    vol.data = None
    vol.inklabel = None
    return vol


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((3, 5), 255, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def windows_os_path(monkeypatch):
    """Make the volume module see Windows' os.path, whatever the host OS is."""
    fake_os = types.SimpleNamespace(**{k: getattr(volume_mod.os, k) for k in dir(volume_mod.os)
                                       if not k.startswith("__")})
    fake_os.path = ntpath
    monkeypatch.setattr(volume_mod, "os", fake_os)


@pytest.fixture
def fake_fsspec(monkeypatch):
    requested = []
    payload = _png_bytes()

    @contextlib.contextmanager
    def fake_open(url, mode="rb", **kwargs):
        requested.append(url)
        if url != EXPECTED_URL:
            raise FileNotFoundError(url)
        yield io.BytesIO(payload)

    monkeypatch.setattr(volume_mod.fsspec, "open", fake_open)
    return requested


@pytest.mark.parametrize("use_windows_path", [False, True])
def test_inklabel_url_uses_forward_slashes(request, fake_fsspec, use_windows_path):
    if use_windows_path:
        request.getfixturevalue("windows_os_path")
    vol = _segment(SEGMENT_URL)

    vol.download_inklabel()

    assert fake_fsspec == [EXPECTED_URL]
    assert vol.inklabel.shape == (3, 5)
    assert int(vol.inklabel.max()) == 255


def test_local_segment_path_still_joins_with_os_path(tmp_path, monkeypatch):
    seg_dir = tmp_path / "segments"
    seg_dir.mkdir()
    (seg_dir / "123_inklabels.png").write_bytes(_png_bytes())
    vol = _segment(str(seg_dir / "123.zarr"))

    vol.download_inklabel()

    assert vol.inklabel.shape == (3, 5)


def test_missing_inklabel_warns_instead_of_failing_silently(fake_fsspec):
    vol = _segment("https://example.invalid/segments/999.zarr/")

    with pytest.warns(RuntimeWarning, match="all-zero placeholder"):
        vol.download_inklabel()

    assert not vol.inklabel.any()
