from __future__ import annotations

import pytest

from vc_reader import NotFoundError


@pytest.fixture
def vol(paris4):
    return paris4.volume(resolution=2.399, energy_kev=78)


def test_level_by_um(vol):
    lvl = vol.level(um=9.6)
    assert lvl.level == 2 and lvl.scale_factor == 4
    assert lvl.voxel_size_um == pytest.approx(9.6)
    assert vol.level(um=9.59) == lvl  # within the default relative tolerance
    assert [round(x.voxel_size_um, 3) for x in vol.levels] == [2.4, 4.8, 9.6, 19.2, 38.4, 76.8]


def test_level_errors(vol):
    with pytest.raises(NotFoundError, match="available: L0=2.4um"):
        vol.level(um=7.0)
    with pytest.raises(NotFoundError):
        vol.level(6)
    with pytest.raises(TypeError):
        vol.level()


def test_level_shape_and_url(vol):
    lvl = vol.level(2)
    # Matches the published PHercParis4/.../2/.zarray shape.
    assert lvl.expected_shape_zyx == (18946, 8174, 8174)
    assert lvl.url() == vol.url() + "2/"


def test_level_transforms_same_volume(vol):
    l0, l2 = vol.level(0), vol.level(2)
    assert l2.to_level0.apply((1, 2, 3)) == (4, 8, 12)
    assert l2.from_level0.apply((4, 8, 12)) == (1, 2, 3)
    assert l2.transform_to(l0).apply((1, 2, 3)) == (4, 8, 12)
    assert l0.transform_to(l2).apply((4, 8, 12)) == (1, 2, 3)


def test_level_transforms_across_volumes(paris4, vol):
    hi = paris4.volume(resolution=1.129)
    src, dst = vol.level(2), hi.level(1)
    p = (4000.0, 3000.0, 9000.0)
    expected = [c / 2 for c in vol.transform_to(hi).apply([c * 4 for c in p])]
    assert src.transform_to(dst).apply(p) == pytest.approx(expected, rel=1e-12)
    assert src.transform_to(hi) == src.transform_to(hi.level(0))


def test_paired_predictions(vol):
    by_key = {
        (p.artifact.type, p.artifact.model_id, p.artifact.parameters.get("level")): p
        for p in vol.level(um=9.6).predictions()
    }
    # Surface prediction computed on level 2: its own level 0 is the 9.6um grid.
    p = by_key[("surface-prediction-zarr", "20260413222639", 2)]
    assert p.sublevel == 0 and p.url().endswith("-L2-th0.2.zarr/0/")
    # Surface prediction and 3D ink computed on level 0: use their level 2.
    assert by_key[("surface-prediction-zarr", "20260413141734", 0)].sublevel == 2
    assert by_key[("ink-detection-3d-zarr", "20260428123845", None)].sublevel == 2
    # Not pyramids on the volume grid.
    assert by_key[("lasagna", "20260419180421", 2)].sublevel is None
    assert by_key[("umbilicus", None, None)].url() is None


def test_prediction_coarser_than_level_is_not_aligned(vol):
    preds = vol.level(1).predictions()
    l2_surface = next(p for p in preds if p.artifact.parameters.get("level") == 2)
    assert not l2_surface.aligned


def test_paired_bundle(vol):
    bundle = vol.level(um=9.6).paired()
    assert bundle.level.level == 2
    assert len(bundle.segments) == 3
    assert len(bundle.ink_labels) == 3
    assert len(bundle.predictions) == len(vol.derived_artifacts())


def test_open_level_reader(vol, monkeypatch):
    pytest.importorskip("numpy")
    from vc_reader.reader import ZarrReader

    seen = {}
    monkeypatch.setattr(ZarrReader, "open", classmethod(lambda cls, url, **kw: seen.update(url=url, **kw) or "reader"))
    assert vol.level(um=9.6).open(cache_dir="/c") == "reader"
    assert seen == {"url": vol.url() + "2/", "cache_dir": "/c"}
    pred = next(p for p in vol.level(2).predictions() if p.artifact.parameters.get("level") == 2)
    pred.open()
    assert seen["url"].endswith("-L2-th0.2.zarr/0/")
    lasagna = next(p for p in vol.level(2).predictions() if p.artifact.type == "lasagna")
    with pytest.raises(NotFoundError):
        lasagna.open()
