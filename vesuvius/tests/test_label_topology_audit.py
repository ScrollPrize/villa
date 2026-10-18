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


def test_cavity_closed_only_diagonally_counts_as_cavity():
    # 26-connected foreground pairs with 6-connected background: a cavity whose only route
    # outside is a diagonal (edge) step between two removed wall voxels is still closed.
    # With the wrong pairing it would leak (b2=0) or get a spurious handle.
    v = np.zeros((30, 30, 30), np.uint8)
    v[4:26, 4:26, 4:26] = 1
    v[6:24, 6:24, 6:24] = 0          # 2-voxel-thick hollow cube
    v[4, 15, 15] = 0                 # outer wall layer
    v[5, 16, 15] = 0                 # inner wall layer, edge-adjacent only
    m, _ = lta.audit(v)
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
    # a regression only in the --after volume must also fail
    rc = lta.main(["--label", str(tmp_path / "before.tif"), "--after", str(tmp_path / "after.tif"),
                   "--out", str(out), "--fail-on", "b1_total_big=0"])
    assert rc == 2
    with pytest.raises(SystemExit):
        lta.main(["--label", str(tmp_path / "before.tif"), "--out", str(out), "--fail-on", "b1_total_big=abc"])


def test_batch_after_missing_pair_errors_before_writing(tmp_path):
    (tmp_path / "labels").mkdir(); (tmp_path / "after").mkdir()
    tifffile.imwrite(tmp_path / "labels" / "a.tif", sheet())
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        lta.main(["--batch", str(tmp_path), "--after", str(tmp_path / "after"), "--out", str(out)])
    assert not out.exists()


def test_hole_at_crop_face_is_excluded_from_interior_b1():
    # holes cut at the crop face are mostly unjudgeable slivers in the Kaggle crops; the
    # interior count is what ranked.csv sorts on, so a face hole must not inflate it
    face, mid = sheet(), sheet()
    face[:, 4:7, 18:22] = 0     # hole whose rim is the first labelled rows next to the face
    face[:, 3, 18:22] = 1       # close it off so it is a hole, not a notch
    mid[:, 18:22, 18:22] = 0
    mf, _ = lta.audit(face)
    mm, _ = lta.audit(mid)
    assert (mf["b1_total_big"], mf["b1_interior"]) == (1, 0)
    assert (mm["b1_total_big"], mm["b1_interior"]) == (1, 1)

    # a crop thinner than the trimmed margin has no interior
    tiny = np.zeros((10, 40, 40), np.uint8); tiny[2:8, 4:-4, 4:-4] = 1; tiny[2:8, 18:22, 18:22] = 0
    mt, _ = lta.audit(tiny)
    assert (mt["b1_total_big"], mt["b1_interior"]) == (1, 0)


def test_ranked_puts_interior_holes_first_and_drops_clean_crops():
    base = {k: 0 for k, _ in lta.SEVERITY}
    rows = [{"crop": "clean", **base},
            {"crop": "junctions", **base, "n_junction_clusters": 9},
            {"crop": "face_hole", **base, "b1_total_big": 5},
            {"crop": "hole", **base, "b1_interior": 1, "b1_total_big": 1}]
    r = lta.ranked(rows)
    assert [x["crop"] for x in r] == ["hole", "junctions", "face_hole"]
    assert r[0]["rank"] == 1 and r[0]["reason"] == "1 interior holes, 1 holes incl. crop faces"


def test_batch_writes_ranked_csv(tmp_path):
    (tmp_path / "labels").mkdir()
    holed = sheet(); holed[:, 18:24, 18:24] = 0
    tifffile.imwrite(tmp_path / "labels" / "a_clean.tif", sheet())
    tifffile.imwrite(tmp_path / "labels" / "b_holed.tif", holed)
    out = tmp_path / "out"
    assert lta.main(["--batch", str(tmp_path), "--out", str(out)]) == 0
    lines = (out / "ranked.csv").read_text().splitlines()
    assert len(lines) == 2 and lines[1].startswith("1,b_holed,")


def test_detect_holes_cpu_topology_matches_known_shapes():
    # detect_holes/hole_viewer now get their CPU Betti numbers from the shared helper;
    # both connectivity pairings must still give the textbook answers
    dh = pytest.importorskip("vesuvius.scripts.detect_holes")
    for fg, bg in ((26, 6), (6, 26)):
        topo = lambda v: dh.compute_component_topology(v, fg_connectivity=fg, bg_connectivity=bg, use_cpu=True)[:2]
        assert (topo(torus()), topo(shell()), topo(sheet())) == ((1, 0), (0, 1), (0, 0))
