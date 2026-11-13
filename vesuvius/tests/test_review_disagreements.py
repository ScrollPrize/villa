"""review_disagreements must rank model-drawn papyrus the label lacks, put the marker on
the component (not in a concave gap), keep the reader blind, and export exactly the
reviewed component, so a corrected label never gains voxels nobody looked at.
"""
import csv
import json
import pytest

import numpy as np
import tifffile
from vesuvius.scripts import review_disagreements as rd

SHAPE = (48, 48, 48)


def _dataset(tmp_path):
    data, pred = tmp_path / "data", tmp_path / "pred"
    for d in (data / "labels", data / "images", pred):
        d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    label = np.zeros(SHAPE, np.uint8)
    label[24, 4:20, 4:20] = 1                       # a sheet both agree on
    label[2, 2:6, 2:6] = 2                          # ignore region
    p = np.zeros(SHAPE, np.uint8)
    p[24, 4:20, 4:20] = 1
    p[24, 5:7, 5:7] = 1                             # inside the label: not far
    p[24, 22:25, 4:8] = 1                           # touches the label edge: within tolerance, not far
    # C-shaped far component: its centroid falls in the gap
    p[40, 30:46, 30:34] = 1
    p[40, 30:46, 42:46] = 1
    p[40, 30:34, 30:46] = 1
    # a second crop with a smaller far blob
    p2 = p.copy()
    p2[40] = 0
    p2[36:40, 36:44, 36:44] = 1
    for name, pr in (("crop_a", p), ("crop_b", p2), ("crop_c", np.where(label == 1, 1, 0).astype(np.uint8))):
        tifffile.imwrite(data / "labels" / f"{name}.tif", label)
        tifffile.imwrite(data / "images" / f"{name}.tif", rng.integers(0, 255, SHAPE, np.uint8))
        tifffile.imwrite(pred / f"{name}.tif", pr)
    return data, pred


def _stage(tmp_path, **kw):
    data, pred = _dataset(tmp_path)
    out = tmp_path / "rev"
    keys = rd.stage(data, pred, out, min_vox=30, half=20, **kw)
    return data, pred, out, keys


def test_ranking_finds_far_component_and_ignores_one_under_tolerance(tmp_path):
    data, pred = _dataset(tmp_path)
    rows, cands = rd.rank_crops(data, pred, min_vox=30)
    assert rows[-1]["crop"] == "crop_c"
    by = {r["crop"]: r for r in rows}
    assert by["crop_c"]["pred_far"] == 0 and by["crop_c"]["n_cand"] == 0 and cands["crop_c"] == []
    a = by["crop_a"]
    assert a["n_cand"] == 1                          # the 3-voxel-beyond blob is under tol
    assert a["largest"] == cands["crop_a"][0]["size"] == 3 * 64 - 2 * 16   # C: two arms and a cap, two 4x4 corners shared
    assert a["pred_far_frac"] > 0 and by["crop_b"]["largest"] == 4 * 8 * 8


def test_concave_component_marker_is_on_the_component(tmp_path):
    data, pred = _dataset(tmp_path)
    _, cands = rd.rank_crops(data, pred, min_vox=30)
    c = cands["crop_a"][0]
    p = tifffile.imread(pred / "crop_a.tif")
    marker, centre = json.loads(c["marker_zyx"]), np.array(json.loads(c["center_zyx"]))
    assert p[tuple(marker)] == 1
    assert p[tuple(centre)] == 0                     # the centroid itself is in the gap
    assert c["marker_to_centroid_vox"] > 0
    assert np.linalg.norm(np.array(marker) - centre) <= 8


def test_exclude_drops_a_crop(tmp_path):
    data, pred = _dataset(tmp_path)
    ex = tmp_path / "ex.csv"
    ex.write_text("crop\ncrop_a\n")
    rows, _ = rd.rank_crops(data, pred, min_vox=30, exclude=rd.load_excluded(ex))
    assert "crop_a" not in {r["crop"] for r in rows}


def test_stage_only_writes_blind_tiles_and_separate_key(tmp_path):
    data, pred = _dataset(tmp_path)
    out = tmp_path / "rev"
    rd.main(["--data", str(data), "--predictions", str(pred), "--out", str(out), "--min-vox", "30",
             "--half", "20", "--workers", "1", "--stage-only", "--per-crop", "1"])
    keys = list(csv.DictReader(open(out / "key.csv")))
    assert [k["code"] for k in keys] == ["S01", "S02"]
    assert {k["crop"] for k in keys} == {"crop_a", "crop_b"}
    assert sorted(p.name for p in (out / "tiles").iterdir()) == ["S01.png", "S02.png"]
    sheet = list(csv.DictReader(open(out / "sheet.csv")))
    assert [r["code"] for r in sheet] == ["S01", "S02"] and not any(r["verdict"] for r in sheet)
    assert "crop" not in open(out / "sheet.csv").read()
    assert not (out / "verdicts.csv").exists()
    assert (out / "queue.csv").exists() and (out / "candidates.csv").exists()


def test_resume_does_not_reask_and_prompt_hides_the_crop(tmp_path):
    _, _, out, _ = _stage(tmp_path)
    prompts = []
    answers = iter(["s", "q"])

    def ask(png, prompt):
        assert png.exists()
        prompts.append(prompt)
        return next(answers)

    rd.review(out, ask)
    assert not any("crop_" in p for p in prompts)
    asked = []
    rd.review(out, lambda png, p: asked.append(p) or "n")
    assert len(asked) == 1 and asked[0].startswith("S02")      # S01 was answered, not re-asked
    assert [r["verdict"] for r in csv.DictReader(open(out / "verdicts.csv"))] == ["SHEET", "NOT"]


def test_export_sets_exactly_the_component_for_sheet_and_nothing_for_not(tmp_path):
    data, pred, out, _ = _stage(tmp_path)
    code_of = {k["crop"]: k["code"] for k in csv.DictReader(open(out / "key.csv"))}
    rd.review(out, lambda png, p: "s" if p.startswith(code_of["crop_a"]) else "n")
    dest = tmp_path / "train"
    rd.main(["--data", str(data), "--predictions", str(pred), "--out", str(out), "--min-vox", "30",
             "--export-training", str(dest), "--reviewer", "tester"])
    rows = list(csv.DictReader(open(dest / "index.csv")))
    by = {r["crop"]: r for r in rows}
    assert by["crop_a"]["verdict"] == "SHEET" and by["crop_b"]["verdict"] == "NOT"
    ia, ib = (int(by[c]["file"].rsplit("_", 1)[1]) for c in ("crop_a", "crop_b"))
    label = tifffile.imread(data / "labels" / "crop_a.tif")
    p = tifffile.imread(pred / "crop_a.tif")
    fixed = tifffile.imread(dest / "labels" / f"crop_a_{ia:04d}_surface.tif")
    expected = np.zeros(SHAPE, bool)
    expected[40, 30:46, 30:34] = expected[40, 30:46, 42:46] = expected[40, 30:34, 30:46] = True
    assert ((fixed != label)).sum() == expected.sum() == int(by["crop_a"]["changed_voxels"])
    assert (fixed[expected] == 1).all() and (fixed[~expected] == label[~expected]).all()
    assert np.array_equal(tifffile.imread(dest / "labels" / f"crop_b_{ib:04d}_surface.tif"),
                          tifffile.imread(data / "labels" / "crop_b.tif"))
    assert by["crop_b"]["changed_voxels"] == "0"
    assert tifffile.imread(dest / "images" / f"crop_a_{ia:04d}.tif").shape == SHAPE
    prov = json.loads((dest / "provenance.json").read_text())
    assert prov["reviewer"] == "tester" and prov["tol"] == 3.0 and prov["min_vox"] == 30 and prov["n"] == 2


def test_unclear_is_not_exported(tmp_path):
    data, pred, out, _ = _stage(tmp_path)
    rd.review(out, lambda png, p: "u")
    dest = tmp_path / "train"
    assert rd.export_training(data, pred, out, dest, min_vox=30) == []
    assert not list(dest.glob("labels/*"))


def test_restage_refuses_to_overwrite_a_filled_sheet(tmp_path):
    # A reader's filled sheet is the only record of their calls; restaging must not blank it.
    data, pred = _dataset(tmp_path)
    out = tmp_path / "rev"
    args = ["--data", str(data), "--predictions", str(pred), "--out", str(out), "--min-vox", "30",
            "--half", "20", "--workers", "1", "--stage-only", "--per-crop", "1"]
    rd.main(args)
    (out / "sheet.csv").write_text("code,verdict,note\nS01,sheet,\nS02,,\n")
    with pytest.raises(SystemExit, match="already holds verdicts"):
        rd.main(args)
    assert "S01,sheet" in (out / "sheet.csv").read_text()


def test_ranks_stage_from_each_queue_position_and_record_the_rank(tmp_path):
    # Precision by rank needs tiles sampled below the top; the key must say where each came from.
    data, pred = _dataset(tmp_path)
    keys = rd.stage(data, pred, tmp_path / "rev", min_vox=30, half=20, limit=1, ranks=[2, 3])
    assert [(k["rank"], k["kind"]) for k in keys] == [(2, "candidate")]   # crop_c at rank 3 has no candidate
    queue = list(csv.DictReader(open(tmp_path / "rev" / "queue.csv")))
    assert keys[0]["crop"] == queue[1]["crop"]


def test_decoys_are_shuffled_blind_controls_on_and_off_the_sheet(tmp_path):
    # A reader who only ever sees candidates cannot show they say NOT when NOT is right.
    data, pred = _dataset(tmp_path)
    out = tmp_path / "rev"
    keys = rd.stage(data, pred, out, min_vox=30, half=20, limit=2, decoys=2, seed=1)
    kinds = [k["kind"] for k in keys]
    assert kinds.count("candidate") == 2 and kinds.count("decoy_covered") == 2 and kinds.count("decoy_off") == 2
    assert kinds != ["candidate", "decoy_covered", "decoy_off"] * 2        # shuffled, not in staging order
    assert [k["code"] for k in keys] == [f"S{i:02d}" for i in range(1, 7)]
    label = tifffile.imread(data / "labels" / "crop_a.tif")
    p = tifffile.imread(pred / "crop_a.tif")
    for k in keys:
        if k["crop"] != "crop_a" or k["kind"] == "candidate":
            continue
        z, y, x = json.loads(k["marker_zyx"])
        on = label[z, y, x] == 1 and p[z, y, x] == 1
        assert on == (k["kind"] == "decoy_covered")
        if k["kind"] == "decoy_off":
            assert label[z - 3:z + 4, y - 3:y + 4, x - 3:x + 4].max() == 0 and p[z - 3:z + 4, y - 3:y + 4, x - 3:x + 4].max() == 0
    assert sorted(q.name for q in (out / "tiles").iterdir()) == [f"S{i:02d}.png" for i in range(1, 7)]
    assert "kind" not in open(out / "sheet.csv").read()


def test_decoy_verdicts_are_scored_by_kind_and_never_exported(tmp_path):
    data, pred = _dataset(tmp_path)
    out = tmp_path / "rev"
    keys = rd.stage(data, pred, out, min_vox=30, half=20, limit=2, decoys=1, seed=3)
    kind_of = {k["code"]: k["kind"] for k in keys}
    rd.review(out, lambda png, prompt: "s" if kind_of[prompt.split()[0]] != "decoy_off" else "n")
    rows = rd.score(out)
    by = {(r["kind"], r["rank"]): r for r in rows}
    assert by[("candidate", "1")]["SHEET"] == 1 and by[("candidate", "2")]["SHEET"] == 1
    assert by[("decoy_covered", "1")]["SHEET"] == 1                 # one decoy per crop still alternates kinds
    assert by[("decoy_off", "2")]["NOT"] == 1 and ("decoy_off", "1") not in by
    index = rd.export_training(data, pred, out, tmp_path / "train", min_vox=30)
    assert {r["crop"] for r in index} == {"crop_a", "crop_b"} and len(index) == 2

