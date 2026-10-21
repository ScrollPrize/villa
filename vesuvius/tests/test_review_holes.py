"""review_holes turns a reviewer's calls into the exact inputs fix_holes.py needs.

A CUT verdict must reach fix_holes.py --veto unchanged, a FILL verdict must not, the
session must resume without re-asking, and the staged folder must be what --in-dir reads.
"""
import csv
import json
import numpy as np
import tifffile
from vesuvius.scripts import fix_holes, review_holes


def _dataset(tmp_path):
    data, audit = tmp_path / "data", tmp_path / "audit"
    (data / "labels").mkdir(parents=True)
    (data / "images").mkdir()
    audit.mkdir()
    label = np.zeros((20, 40, 40), np.uint8)
    label[10] = 1
    label[10, 8:12, 8:12] = 0           # hole A
    label[10, 28:32, 28:32] = 0         # hole B
    mask = np.zeros_like(label)
    mask[10, 8:12, 8:12] = 1
    mask[10, 28:32, 28:32] = 1
    tifffile.imwrite(data / "labels" / "crop_1.tif", label)
    tifffile.imwrite(data / "images" / "crop_1.tif", np.random.default_rng(0).integers(0, 255, label.shape, np.uint8))
    tifffile.imwrite(audit / "crop_1.hole_mask.tif", mask)
    holes = [{"center_zyx": [10, 10, 10], "voxels": 16, "interior": False},
             {"center_zyx": [10, 30, 30], "voxels": 16, "interior": True}]
    (audit / "crop_1.json").write_text(json.dumps({"holes": holes}))
    with open(audit / "ranked.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "crop", "reason"])
        w.writerow([1, "crop_1", "2 interior or unlocated holes"])
    return data, audit


def test_cut_reaches_veto_and_fill_does_not(tmp_path):
    data, audit = _dataset(tmp_path)
    out = tmp_path / "rev"
    answers = iter(["c", "f"])           # interior hole B asked first: CUT; then A: FILL
    seen = []

    def ask(png, prompt):
        assert png.exists()
        seen.append(prompt)
        return next(answers)

    v = review_holes.review(data, audit, out, ask)
    assert "zyx [10, 30, 30]" in seen[0] and "interior" in seen[0]
    assert {r["verdict"] for r in v.values()} == {"CUT", "FILL"}
    veto = fix_holes._load_veto(out / "veto.csv")
    assert veto == {"crop_1": [(10, 30, 30)]}
    # staged layout is what fix_holes reads; the vetoed hole stays open, the other is filled
    ok, msg = fix_holes._process_folder(
        out / "fix_in" / "crop_1", hole_mask_name="hole_mask.tif", label_glob="*_surf.tif", fg_value=1,
        ignore_value=None, suffix="t", methods=["2d_hole_fill"], closing_kernel=2, alpha=0.005, iters=1,
        mask_dilate=0, veto_centers=veto["crop_1"])
    assert ok and "[VETO] 1 hole" in msg
    fixed = tifffile.imread(out / "fix_in" / "crop_1" / "crop_1_surf_t.tif")
    assert fixed[10, 28:32, 28:32].sum() == 0 and fixed[10, 8:12, 8:12].sum() == 16


def test_resume_skips_answered_holes_and_quit_keeps_them(tmp_path):
    data, audit = _dataset(tmp_path)
    out = tmp_path / "rev"
    review_holes.review(data, audit, out, lambda png, p: "u")      # first session: both UNCLEAR
    # second session must ask nothing
    review_holes.review(data, audit, out, lambda png, p: (_ for _ in ()).throw(AssertionError("re-asked")))
    rows = list(csv.DictReader(open(out / "verdicts.csv")))
    assert [r["verdict"] for r in rows] == ["UNCLEAR", "UNCLEAR"]
    assert open(out / "veto.csv").read().strip() == "crop,center_zyx"

    out2 = tmp_path / "rev2"
    answers = iter(["c", "q"])
    review_holes.review(data, audit, out2, lambda png, p: next(answers))
    assert fix_holes._load_veto(out2 / "veto.csv") == {"crop_1": [(10, 30, 30)]}
    assert (out2 / "fix_in" / "crop_1" / "hole_mask.tif").exists()


def test_stage_only_lays_out_what_hole_viewer_reads(tmp_path):
    """--stage-only must give hole_viewer.py the CT, the hole list and the rank order,
    on top of the fix_holes.py layout, without asking a single question."""
    data, audit = _dataset(tmp_path)
    out = tmp_path / "rev"
    review_holes.main(["--data", str(data), "--audit", str(audit), "--out", str(out), "--stage-only", "--limit", "5"])
    d = out / "fix_in" / "crop_1"
    assert (d / "crop_1_surf.tif").exists() and (d / "hole_mask.tif").exists()
    assert tifffile.imread(d / "crop_1_img.tif").shape == (20, 40, 40)
    holes = json.loads((d / "holes.json").read_text())
    assert [h["center_zyx"] for h in holes] == [[10, 30, 30], [10, 10, 10]]   # interior first
    assert (out / "fix_in" / "order.txt").read_text().split() == ["crop_1"]
    assert not (out / "verdicts.csv").exists()
    # a second staging run must not duplicate the order entry
    review_holes.stage_ranked(data, audit, out)
    assert (out / "fix_in" / "order.txt").read_text().split() == ["crop_1"]


def test_export_training_keeps_each_verdict_with_the_ct_it_was_made_from(tmp_path):
    """A FILL/CUT model trains on the reviewer's call plus the CT around the hole; the
    export must centre the cube on the hole and carry the verdict unchanged."""
    data, audit = _dataset(tmp_path)
    out = tmp_path / "rev"
    answers = iter(["c", "f"])
    review_holes.review(data, audit, out, lambda png, p: next(answers))
    dest = tmp_path / "train"
    (data / "pred").mkdir()
    pred = np.zeros((20, 40, 40), np.uint8)
    pred[10] = 1                                   # the model drew the sheet with no holes
    tifffile.imwrite(data / "pred" / "crop_1.tif", pred)
    review_holes.main(["--data", str(data), "--audit", str(audit), "--out", str(out),
                       "--export-training", str(dest), "--half", "16", "--predictions", str(data / "pred"),
                       "--reviewer", "tester"])
    rows = list(csv.DictReader(open(dest / "index.csv")))
    prov = json.loads((dest / "provenance.json").read_text())
    assert prov["reviewer"] == "tester" and prov["n"] == 2 and prov["cube"] == 16
    assert [r["verdict"] for r in rows] == ["CUT", "FILL"]
    ex = np.load(dest / rows[0]["file"])
    assert ex["ct"].shape == ex["label"].shape == ex["mask"].shape == (16, 16, 16)
    assert str(ex["verdict"]) == "CUT" and list(ex["center_zyx"]) == [10, 30, 30]
    assert ex["mask"][8, 8, 8] == 1 and ex["label"][8, 8, 8] == 0     # cube centred on the hole
    assert ex["label"][8, 0, 0] == 1                                     # sheet around it
    assert ex["pred"][8, 8, 8] == 1 and ex["label"][8, 8, 8] == 0        # model drew it, label did not
