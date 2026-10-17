from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

pytest.importorskip("cc3d")
pytest.importorskip("skimage")

from vesuvius.image_proc.run import label_topology_audit as lta  # noqa: E402


def sheet(shape=(40, 40, 40), z0=18, z1=20):
    v = np.zeros(shape, np.uint8)
    v[z0:z1, 4:-4, 4:-4] = 1
    return v


def torus(shape=(40, 40, 40), R=10, r=4):
    z, y, x = np.indices(shape) - np.array(shape)[:, None, None, None] / 2
    return (((np.hypot(y, x) - R) ** 2 + z**2) <= r**2).astype(np.uint8)


def shell(shape=(40, 40, 40), r_out=12, r_in=9):
    d = np.sqrt(((np.indices(shape) - np.array(shape)[:, None, None, None] / 2) ** 2).sum(0))
    return ((d <= r_out) & (d > r_in)).astype(np.uint8)


def test_flat_sheet_has_no_holes_or_cavities_and_is_thin():
    m, _ = lta.audit(sheet())
    assert (m["b1_total_big"], m["b2_total_big"], m["n_components"]) == (0, 0, 1)
    assert 1 <= m["thickness"]["median"] <= 3


def test_torus_has_one_handle():
    m, _ = lta.audit(torus())
    assert (m["b1_total_big"], m["b2_total_big"]) == (1, 0)


def test_sheet_with_hole_has_one_handle():
    v = sheet()
    v[:, 18:24, 18:24] = 0
    m, _ = lta.audit(v)
    assert (m["b1_total_big"], m["b2_total_big"], m["n_components"]) == (1, 0, 1)


def test_hollow_shell_has_one_cavity():
    m, _ = lta.audit(shell())
    assert (m["b2_total_big"], m["b1_total_big"]) == (1, 0)


def test_specks_are_counted_as_tiny_fragments():
    v = sheet()
    for i in range(5):
        v[2, 2 + 6 * i, 2] = 1  # isolated single voxels
    m, _ = lta.audit(v)
    assert m["n_components"] == 6 and m["n_tiny"] == 5
    assert m["n_big_components"] == 1 and m["b1_total_big"] == 0


def test_ignore_voxels_are_not_foreground_and_mark_components_open():
    v = sheet()
    v[18:20, 20:24, 20:24] = 2
    m, _ = lta.audit(v)
    assert m["fg_voxels"] == int((v == 1).sum())
    assert m["unlabeled_frac"] > 0
    assert m["n_closed_big"] == 0  # touches ignore voxels
    # the same bytes read with another surface value are a different volume
    m2, _ = lta.audit(v, surface_value=2, ignore_value=0)
    assert m2["fg_voxels"] == 4 * 4 * 2


def test_interior_component_is_closed():
    v = np.zeros((40, 40, 40), np.uint8)
    v[18:20, 10:30, 10:30] = 1
    m, _ = lta.audit(v)
    assert m["n_closed_big"] == 1


def test_cli_after_delta_and_fail_on(tmp_path):
    before, after = sheet(), sheet()
    after[:, 18:24, 18:24] = 0
    tifffile.imwrite(tmp_path / "before.tif", before)
    tifffile.imwrite(tmp_path / "after.tif", after)
    out = tmp_path / "out"
    rc = lta.main(["--label", str(tmp_path / "before.tif"), "--after", str(tmp_path / "after.tif"),
                   "--out", str(out)])
    assert rc == 0
    delta = json.loads((out / "before.delta.json").read_text())
    assert delta["b1_total_big"] == [0, 1]
    assert (out / "aggregate.csv").exists()
    rc = lta.main(["--label", str(tmp_path / "after.tif"), "--out", str(out),
                   "--fail-on", "b1_total_big=0"])
    assert rc == 2
