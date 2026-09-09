import json
from pathlib import Path

import numpy as np
import tifffile

from vesuvius.tifxyz import list_tifxyz


def _write_segment(path: Path, z_lo: float, z_hi: float, bbox_lo_z: float) -> Path:
    """A 4x6 grid whose valid z runs z_lo..z_hi, with a chosen meta.json bbox z_min."""
    path.mkdir(parents=True)
    x = np.full((4, 6), 100.0, dtype=np.float32)
    y = np.full((4, 6), 200.0, dtype=np.float32)
    z = np.linspace(z_lo, z_hi, 24, dtype=np.float32).reshape(4, 6)
    x[1, 2] = y[1, 2] = z[1, 2] = -1  # one genuinely missing point, not at either extreme
    for name, grid in (("x", x), ("y", y), ("z", z)):
        tifffile.imwrite(str(path / f"{name}.tif"), grid)
    meta = {
        "uuid": path.name,
        "scale": [0.05, 0.05],
        "bbox": [[100.0, 200.0, bbox_lo_z], [100.0, 200.0, float(z_hi)]],
        "format": "tifxyz",
        "type": "seg",
    }
    (path / "meta.json").write_text(json.dumps(meta))
    return path


def test_marker_bbox_falls_back_to_valid_coordinates(tmp_path: Path) -> None:
    # Published shape from villa #1618: bbox lower corner carries the -1 marker
    # while the real points start thousands of slices in.
    _write_segment(tmp_path / "marker", z_lo=3000.0, z_hi=4000.0, bbox_lo_z=-1.0)
    _write_segment(tmp_path / "honest", z_lo=11000.0, z_hi=12000.0, bbox_lo_z=11000.0)

    names = lambda zr: sorted(s.path.name for s in list_tifxyz(tmp_path, z_range=zr))
    assert names((0, 100)) == []            # was ["marker"]: z_min=-1 passes every range
    assert names((500, 1000)) == []         # was ["marker"]
    assert names((3000, 4000)) == ["marker"]
    assert names((11000, 12000)) == ["honest"]

    marker = next(s for s in list_tifxyz(tmp_path) if s.path.name == "marker")
    assert marker.z_min is not None and abs(marker.z_min - 3000.0) < 1e-3  # was -1.0
    honest = next(s for s in list_tifxyz(tmp_path) if s.path.name == "honest")
    assert honest.z_min == 11000.0  # stored bbox trusted as before


def test_all_points_missing_gives_no_bbox(tmp_path: Path) -> None:
    p = tmp_path / "empty"
    p.mkdir()
    for name in ("x", "y", "z"):
        tifffile.imwrite(str(p / f"{name}.tif"), np.full((2, 2), -1, dtype=np.float32))
    (p / "meta.json").write_text(json.dumps(
        {"uuid": "empty", "scale": [0.05, 0.05], "bbox": [[-1, -1, -1], [-1, -1, -1]]}))
    infos = list_tifxyz(tmp_path)
    assert len(infos) == 1 and infos[0].bbox is None
    assert list_tifxyz(tmp_path, z_range=(0, 10)) == infos  # no bbox: not filtered, as before


def test_read_tifxyz_object_gets_the_recomputed_bbox(tmp_path: Path) -> None:
    from vesuvius.tifxyz import read_tifxyz

    _write_segment(tmp_path / "marker", z_lo=3000.0, z_hi=4000.0, bbox_lo_z=-1.0)
    surface = read_tifxyz(tmp_path / "marker")
    assert surface.bbox is not None
    assert abs(surface.bbox[2] - 3000.0) < 1e-3  # was -1.0
    assert abs(surface.bbox[5] - 4000.0) < 1e-3

    # neural_tracing prefers seg.bbox over the coordinates when it is not None, so
    # before this change its z filter saw (-1, 4000) for this segment.
    try:
        from vesuvius.neural_tracing.datasets.common import _segment_z_bounds
    except Exception:  # optional heavy deps; the assertion above already covers the object
        return
    lo, hi = _segment_z_bounds(surface)
    assert abs(lo - 3000.0) < 1e-3 and abs(hi - 4000.0) < 1e-3


def test_tifxyzinfo_keeps_its_original_constructor_signature():
    """``bbox`` must stay accepted by name and by position (raised in review of #1731).

    ``TifxyzInfo`` is public API. Turning ``bbox`` into a lazily resolved property is only
    safe if construction is unchanged for existing callers. Without this, keyword use
    raises TypeError, and positional use is worse than an error: the bbox tuple lands in
    ``uuid`` and the uuid string in the stored bbox, failing later and somewhere else.
    """
    from pathlib import Path
    from vesuvius.tifxyz.reader import TifxyzInfo

    bbox = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    positional = TifxyzInfo(Path("/nonexistent"), (1.0, 1.0), bbox, "uuid-positional")
    assert positional.uuid == "uuid-positional"
    assert positional.bbox == bbox

    keyword = TifxyzInfo(Path("/nonexistent"), (1.0, 1.0), bbox=bbox, uuid="uuid-keyword")
    assert keyword.uuid == "uuid-keyword"
    assert keyword.bbox == bbox

    # the field name works too, and bbox wins when both are supplied
    by_field = TifxyzInfo(Path("/nonexistent"), (1.0, 1.0), uuid="u", stored_bbox=bbox)
    assert by_field.bbox == bbox

    # and a segment with no bbox stays None rather than raising
    assert TifxyzInfo(Path("/nonexistent"), (1.0, 1.0), uuid="u").bbox is None
