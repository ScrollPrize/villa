"""A surface written without a uuid gets one minted identity, kept on the
in-memory surface so repeated writes do not each invent a new one."""

from __future__ import annotations

import json

import numpy as np

from vesuvius.tifxyz import Tifxyz, write_tifxyz


def _surface(uuid: str = "") -> Tifxyz:
    coords = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    return Tifxyz(
        _x=coords,
        _y=coords.copy(),
        _z=coords.copy(),
        uuid=uuid,
        _scale=(1.0, 1.0),
        path=None,
    )


def _meta_uuid(directory) -> str:
    return json.loads((directory / "meta.json").read_text(encoding="utf-8"))["uuid"]


def test_repeated_writes_of_unset_uuid_keep_one_identity(tmp_path) -> None:
    surface = _surface()
    first = write_tifxyz(tmp_path / "first.tifxyz", surface)
    second = write_tifxyz(tmp_path / "second.tifxyz", surface)

    assert surface.uuid
    assert _meta_uuid(first) == surface.uuid
    assert _meta_uuid(second) == surface.uuid


def test_explicit_uuid_is_written_as_is(tmp_path) -> None:
    surface = _surface(uuid="0F8FAD5B-D9CB-469F-A165-70867728950E")
    written = write_tifxyz(tmp_path / "named.tifxyz", surface)
    assert _meta_uuid(written) == "0F8FAD5B-D9CB-469F-A165-70867728950E"
