from __future__ import annotations

import csv
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
    assert (mf["b1_total_big"], mf["b1_interior"], mf["b1_face"]) == (1, 0, 1)
    assert (mm["b1_total_big"], mm["b1_interior"], mm["b1_face"], mm["b1_unlocated"]) == (1, 1, 0, 0)

    # a crop thinner than twice the face margin has no interior
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
    assert r[0]["rank"] == 1 and r[0]["reason"] == "1 interior or unlocated holes, 1 holes incl. crop faces"


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


def test_holes_are_located_where_they_are():
    # locations are what make a flag checkable (and fixable with fix_holes.py), so each
    # small hole must come back at its own position
    v = sheet()
    v[:, 10:13, 10:13] = 0
    v[:, 26:29, 24:27] = 0
    m, flags = lta.audit(v)
    assert m["b1_total_big"] == m["n_holes_located"] == 2
    centres = sorted(tuple(h["center_zyx"][1:]) for h in m["holes"])
    assert centres == [(11, 11), (27, 25)]
    assert flags["hole"].sum() == 2 * 9 * 2 and not (flags["hole"] & (v == 1)).any()


def test_wide_hole_is_not_located_but_still_counts_as_interior():
    v = sheet()
    v[:, 12:28, 12:28] = 0           # 16 voxels across, wider than the 3-step closing reaches
    m, _ = lta.audit(v)
    assert (m["b1_total_big"], m["n_holes_located"], m["b1_interior"]) == (1, 0, 1)
    assert (m["b1_face"], m["b1_unlocated"]) == (0, 1)


def test_one_loop_closed_from_two_sides_is_counted_once():
    # tube A (long) and tube B (short, narrow bore) joined by a bridge: b1 = 2. An ignore plug
    # splits A's bore into two blobs, and filling either caps A. Counting both against the
    # original b1 would use up b1 on A and never locate B's hole.
    z, y, x = np.mgrid[:24, :16, :30]
    ra, rb = np.hypot(y - 7.5, x - 7.5), np.hypot(y - 7.5, x - 21.5)
    v = (ra < 4) & (ra >= 2) & (z >= 2) & (z < 22)
    v |= (rb < 4) & (rb >= 1) & (z >= 2) & (z < 8)
    v |= (z >= 2) & (z < 4) & (y >= 5) & (y < 11) & (x >= 10) & (x < 20) & ~((ra < 2) | (rb < 1))
    plug = (ra < 2) & (z >= 11) & (z < 13)
    assert lta.component_topology(v)["b1"] == 2
    holes, found = lta.locate_holes(v, 2, plug)
    assert sorted(h["center_zyx"][2] > 15 for h in holes) == [False, True]   # one per tube
    assert sum(h["fixes"] for h in holes) == 2
    assert not (found & plug).any()


def test_cli_hole_masks_and_hole_tiles(tmp_path):
    pytest.importorskip("matplotlib")
    v = sheet()
    v[:, 18:21, 18:21] = 0
    tifffile.imwrite(tmp_path / "lab.tif", v)
    tifffile.imwrite(tmp_path / "img.tif", (v * 200).astype(np.uint8))
    out = tmp_path / "out"
    assert lta.main(["--label", str(tmp_path / "lab.tif"), "--image", str(tmp_path / "img.tif"),
                     "--out", str(out), "--plot", "--hole-masks"]) == 0
    mask = tifffile.imread(out / "lab.hole_mask.tif")
    assert mask.sum() == 9 * 2 and mask[18:20, 18:21, 18:21].all()
    assert (out / "lab.holes.png").exists()


@pytest.mark.parametrize("limit", ["nan", "inf", "-inf"])
def test_cli_rejects_nonfinite_limits_before_writing(tmp_path, limit):
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        lta.main(["--label", str(tmp_path / "unused.tif"), "--out", str(out),
                  "--fail-on", f"b1_total_big={limit}"])
    assert not out.exists()


@pytest.mark.parametrize("extra", [["--erode", "0"], ["--erode", "-1"],
                                   ["--min-piece", "0"], ["--ignore-value", "1"]])
def test_cli_rejects_invalid_audit_options_before_writing(tmp_path, extra):
    tifffile.imwrite(tmp_path / "label.tif", sheet())
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        lta.main(["--label", str(tmp_path / "label.tif"), "--out", str(out), *extra])
    assert not out.exists()


@pytest.mark.parametrize("paired_option", ["--after", "--image"])
def test_cli_rejects_mismatched_volume_shapes_before_writing(tmp_path, paired_option):
    tifffile.imwrite(tmp_path / "label.tif", sheet())
    tifffile.imwrite(tmp_path / "mismatch.tif", sheet()[:20])
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        lta.main(["--label", str(tmp_path / "label.tif"), "--out", str(out),
                  paired_option, str(tmp_path / "mismatch.tif"), "--plot"])
    assert not out.exists()


def test_audit_rejects_2d_or_empty_input_with_clear_error():
    for label in [np.zeros((20, 20), np.uint8), np.zeros((0, 20, 20), np.uint8)]:
        with pytest.raises(ValueError, match="nonempty 3D"):
            lta.audit(label)


def test_singleton_axis_and_empty_foreground_are_supported():
    label = np.ones((1, 20, 20), np.uint8)
    metrics, _ = lta.audit(label)
    assert metrics["n_components"] == 1
    assert metrics["thickness"]["median"] == 1
    metrics, _ = lta.audit(np.zeros((10, 10, 10), np.uint8))
    assert metrics["n_components"] == metrics["b1_total_big"] == 0
    assert metrics["thickness"] == {}


def test_thickness_treats_all_crop_faces_as_truncated_boundaries():
    # A full 10-voxel cube has no observed external background. Padding supplies a
    # symmetric crop boundary; the lower-bound thickness must not exceed the crop.
    stats, _, edt = lta.thickness_stats(np.ones((10, 10, 10), bool))
    assert stats["max"] == 9
    assert np.array_equal(edt, edt[::-1, ::-1, ::-1])


def test_fusion_heuristic_flags_thick_lobes_joined_by_a_thin_bridge():
    label = np.zeros((30, 30, 50), np.uint8)
    label[8:22, 8:22, 4:18] = 1
    label[8:22, 8:22, 32:46] = 1
    label[14:16, 14:16, 18:32] = 1
    metrics, _ = lta.audit(label)
    assert metrics["n_components"] == metrics["n_fused_components"] == 1
    assert metrics["fused_voxel_frac"] == 1


def test_parse_roi_clips_and_rejects_empty():
    assert lta.parse_roi(None, (4, 5, 6)) == (slice(0, 4), slice(0, 5), slice(0, 6))
    assert lta.parse_roi("1:3,:,2:99", (4, 5, 6)) == (slice(1, 3), slice(0, 5), slice(2, 6))
    with pytest.raises(SystemExit):
        lta.parse_roi("1:3,0:5", (4, 5, 6))
    with pytest.raises(SystemExit):
        lta.parse_roi("3:1,:,:", (4, 5, 6))


def test_tile_windows_drops_partial_far_edge():
    roi = (slice(0, 5), slice(1, 5), slice(0, 4))
    wins = lta.tile_windows(roi, 2)
    # 2 full z tiles (4..5 dropped), 2 y tiles from 1, 2 x tiles
    assert len(wins) == 8
    assert wins[0] == (slice(0, 2), slice(1, 3), slice(0, 2))
    assert all(w[0].stop <= 5 and w[1].stop <= 5 and w[2].stop <= 4 for w in wins)
    assert lta.tile_windows(roi, None) == [roi]
    with pytest.raises(SystemExit):
        lta.tile_windows(roi, 6)


def test_cli_zarr_roi_tile_threshold(tmp_path):
    # a 0/255 mask in an OME-Zarr group with a hole in one tile only: the tiled run must find
    # the hole in that tile and name tiles by their window so results map back to the volume
    zarr = pytest.importorskip("zarr")
    v = np.zeros((40, 80, 40), np.uint8)
    v[18:20, 4:-4, 4:-4] = 255
    v[:, 58:64, 18:24] = 0  # hole in the y >= 40 tile only
    g = zarr.open_group(str(tmp_path / "pred.zarr"), mode="w")
    create = getattr(g, "create_array", None) or g.create_dataset
    create("0", shape=v.shape, chunks=(20, 20, 20), dtype=v.dtype)
    g["0"][:] = v
    g.attrs["multiscales"] = [{"version": "0.4", "datasets": [{"path": "0"}]}]
    out = tmp_path / "out"
    rc = lta.main(["--label", str(tmp_path / "pred.zarr"), "--threshold", "1", "--tile", "40",
                   "--out", str(out)])
    assert rc == 0
    rows = {r["crop"]: r for r in csv.DictReader(open(out / "aggregate.csv"))}
    assert set(rows) == {"pred_z0_y0_x0", "pred_z0_y40_x0"}
    assert (int(rows["pred_z0_y0_x0"]["b1_total_big"]), int(rows["pred_z0_y40_x0"]["b1_total_big"])) == (0, 1)
    assert (out / "ranked.csv").exists()  # tiled runs are ranked like a batch
    # --roi restricts the read; a window inside the clean tile has no hole and no ranking file
    out2 = tmp_path / "out2"
    rc = lta.main(["--label", str(tmp_path / "pred.zarr"), "--threshold", "1", "--roi", ":,0:40,:",
                   "--out", str(out2)])
    assert rc == 0
    m = json.loads((out2 / "pred.json").read_text())
    assert m["b1_total_big"] == 0 and m["shape"] == [40, 40, 40]
    assert not (out2 / "ranked.csv").exists()
    # without --threshold a 0/255 mask has no voxels equal to surface value 1: an empty audit,
    # which is why the threshold exists
    assert lta.main(["--label", str(tmp_path / "pred.zarr"), "--out", str(tmp_path / "out3")]) == 0
    assert json.loads((tmp_path / "out3" / "pred.json").read_text())["fg_voxels"] == 0


def test_cli_threshold_sweep_reads_once_and_names_cuts(tmp_path, monkeypatch):
    # a probability volume: a sheet of 200 with a weak (60) patch. Cutting at 50 keeps the
    # patch (no hole), cutting at 100 opens it. One sweep must give both answers from one read,
    # because the read is the expensive part for a volume served over HTTP.
    v = np.zeros((40, 40, 40), np.uint8)
    v[18:20, 4:-4, 4:-4] = 200
    v[18:20, 16:24, 16:24] = 60
    tifffile.imwrite(tmp_path / "prob.tif", v)
    reads = []
    orig = lta.read_window
    monkeypatch.setattr(lta, "read_window", lambda vol, win, threshold=None: (reads.append(1), orig(vol, win, threshold))[1])
    out = tmp_path / "out"
    assert lta.main(["--label", str(tmp_path / "prob.tif"), "--threshold", "50,100", "--out", str(out)]) == 0
    assert len(reads) == 1
    rows = {r["crop"]: r for r in csv.DictReader(open(out / "aggregate.csv"))}
    assert set(rows) == {"prob_t50", "prob_t100"}
    assert (int(rows["prob_t50"]["b1_total_big"]), int(rows["prob_t100"]["b1_total_big"])) == (0, 1)
    assert (out / "ranked.csv").exists()
    # a single threshold keeps the plain crop name
    out2 = tmp_path / "out2"
    assert lta.main(["--label", str(tmp_path / "prob.tif"), "--threshold", "100", "--out", str(out2)]) == 0
    assert (out2 / "prob.json").exists()
    with pytest.raises(SystemExit):
        lta.main(["--label", str(tmp_path / "prob.tif"), "--threshold", "50,x", "--out", str(tmp_path / "o3")])


def test_cli_batch_rejects_roi_and_tile(tmp_path):
    (tmp_path / "labels").mkdir()
    tifffile.imwrite(tmp_path / "labels" / "a.tif", sheet())
    with pytest.raises(SystemExit):
        lta.main(["--batch", str(tmp_path), "--tile", "20", "--out", str(tmp_path / "out")])


def test_fill_cut_off_by_ignore_ring_is_not_a_located_hole():
    # a fill that does not touch the sheet is an island: it raises the Euler number without
    # closing any loop, so counting it as a repair would report a hole fixed that is still open
    v = sheet()
    v[18:20, 9:14, 9:14] = 2
    v[18:20, 10:13, 10:13] = 0
    m, flags = lta.audit(v)
    assert m["n_holes_located"] == 0
    assert not flags["hole"].any()


def test_component_touching_ignore_only_at_a_corner_is_open():
    # foreground is 26-connected, so a corner contact with an ignore voxel is contact: the
    # component may continue through the unlabeled region and its topology is truncated
    v = np.zeros((40, 40, 40), np.uint8)
    v[18:20, 10:30, 10:30] = 1
    v[20, 30, 30] = 2
    m, _ = lta.audit(v)
    assert m["n_closed_big"] == 0


def neck_fixture(gap):
    label = np.zeros((30, 30, 50), np.uint8)
    label[8:22, 8:22, 4:18] = 1
    label[8:22, 8:22, 18 + gap:32 + gap] = 1
    label[14:16, 14:16, 18:18 + gap] = 1
    return label


def test_short_neck_between_thick_lobes_is_localised_not_just_detected():
    # The mask is what the overlay draws and what a reviewer follows to the defect. Zeroing it must
    # fail this test: the voxels have to sit on the neck's cross-section, not merely exist.
    metrics, flags = lta.audit(neck_fixture(gap=1))
    assert metrics["n_fused_components"] == 1
    assert metrics["bridge_voxels"] == int(flags["bridge"].sum()) > 0
    z, y, x = np.nonzero(flags["bridge"])
    assert z.min() >= 14 and z.max() <= 15 and y.min() >= 14 and y.max() <= 15
    assert x.min() >= 17 and x.max() <= 19


def test_long_bridge_is_detected_as_fused_but_the_mask_only_marks_necks_next_to_two_pieces():
    # Erosion deletes a 2-voxel-wide bridge, so the split is detected. The bridge mask only marks
    # voxels whose 3x3x3 neighbourhood touches two eroded pieces, so a long bridge is NOT localised.
    # `n_fused_components` means "erosion split detected"; `bridge_voxels` means "neck localised".
    metrics, flags = lta.audit(neck_fixture(gap=14))
    assert metrics["n_fused_components"] == 1
    assert metrics["bridge_voxels"] == int(flags["bridge"].sum()) == 0


def test_tiling_audits_each_window_before_loading_the_next(tmp_path, monkeypatch):
    # A tiled sweep over a large volume must hold one window at a time, not every window.
    v = np.zeros((40, 40, 80), np.uint8)
    v[18:20, 4:-4, 4:-4] = 1
    tifffile.imwrite(tmp_path / "v.tif", v)
    loads, seen = [], []
    real_read, real_audit = lta.read_window, lta.audit
    monkeypatch.setattr(lta, "read_window", lambda vol, w: (loads.append(w), real_read(vol, w))[1])
    monkeypatch.setattr(lta, "audit", lambda *x, **k: (seen.append(len(loads)), real_audit(*x, **k))[1])
    rc = lta.main(["--label", str(tmp_path / "v.tif"), "--tile", "40", "--out", str(tmp_path / "out")])
    assert rc == 0 and len(loads) == 2
    assert seen == [1, 2]


def test_empty_crop_cannot_reach_a_gate_with_a_missing_value(tmp_path):
    # An all-ignore crop has no thickness (thick_* are None), but every --fail-on key is a count or
    # fraction that is always defined, and thick_* are refused as gate keys rather than compared as None.
    tifffile.imwrite(tmp_path / "empty.tif", np.full((20, 20, 20), 2, np.uint8))
    args = ["--label", str(tmp_path / "empty.tif"), "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        lta.main(args + ["--fail-on", "thick_median=7"])
    assert lta.main(args + ["--fail-on", "n_components=0"]) == 0
    m, _ = lta.audit(np.full((20, 20, 20), 2, np.uint8))
    flat = lta.flat(m)
    assert all(flat[k] is not None for k in lta.flat_keys())


def test_url_input_is_zarr_unless_it_names_a_tif():
    from vesuvius.image_proc.run.label_topology_audit import _is_zarr
    assert _is_zarr("https://host/vol.zarr")
    assert _is_zarr("s3://bucket/vol")
    assert not _is_zarr("https://host/label.tif")
    assert not _is_zarr("https://host/label.TIFF")


def test_tif_url_is_fetched_and_decoded(monkeypatch, tmp_path):
    import io
    arr = (np.arange(27, dtype=np.uint8) % 2).reshape(3, 3, 3)
    tifffile.imwrite(tmp_path / "x.tif", arr)
    payload = (tmp_path / "x.tif").read_bytes()
    seen = []

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url: (seen.append(url), Resp(payload))[1])
    out = lta.open_volume("https://host/label.tif?x=1")
    assert seen == ["https://host/label.tif?x=1"] and np.array_equal(out, arr)


def test_threshold_values_that_share_a_crop_name_are_refused():
    with pytest.raises(ValueError, match="collide"):
        lta.parse_thresholds("0.5000001,0.5000002")
    assert lta.parse_thresholds("0.5,0.6") == [0.5, 0.6]
