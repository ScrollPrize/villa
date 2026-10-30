from __future__ import annotations

import numpy as np
import tifffile

from vesuvius.tifxyz import Tifxyz, write_tifxyz


def _surface(offset: float, invalid_corner: bool = False) -> Tifxyz:
    coords = np.arange(16 * 16, dtype=np.float32).reshape(16, 16) + offset
    mask = np.ones((16, 16), dtype=bool)
    if invalid_corner:
        mask[:4, :4] = False
    return Tifxyz(
        _x=coords,
        _y=coords.copy(),
        _z=coords.copy(),
        uuid="0F8FAD5B-D9CB-469F-A165-70867728950E",
        _scale=(1.0, 1.0),
        _mask=mask,
        path=None,
    )


def test_overwrite_keeps_companion_files(tmp_path) -> None:
    seg = write_tifxyz(tmp_path / "seg", _surface(10.0))
    companions = {
        "seg_inklabels.png": b"label bytes",
        "generations.tif": b"generations bytes",
        "overlapping.json": b'{"overlapping": []}',
    }
    for name, payload in companions.items():
        (seg / name).write_bytes(payload)

    write_tifxyz(seg, _surface(1000.0), overwrite=True)

    for name, payload in companions.items():
        assert (seg / name).read_bytes() == payload
    assert tifffile.imread(seg / "x.tif")[0, 0] == 1000.0
    assert not list(tmp_path.glob(".tifxyz_tmp_*"))


def test_overwrite_replaces_the_files_it_writes(tmp_path) -> None:
    seg = write_tifxyz(tmp_path / "seg", _surface(10.0))
    assert tifffile.imread(seg / "mask.tif").min() == 255

    write_tifxyz(seg, _surface(10.0, invalid_corner=True), overwrite=True)

    mask = tifffile.imread(seg / "mask.tif")
    assert mask[:4, :4].max() == 0
    assert mask[4:, 4:].min() == 255
