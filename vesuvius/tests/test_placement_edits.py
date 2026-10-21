"""placement_edits.py pairs each worst cell of a placement audit with what the annotator
did there, so an edit becomes a training example: a correct drag must read as a shift
equal to the offset the audit measured, and untouched cells must read as unmoved.
"""
import csv
import json
import numpy as np
import pytest
import tifffile
from vesuvius.image_proc.run import surface_placement_audit as spa
from vesuvius.scripts import placement_edits
from vesuvius.tifxyz.types import Tifxyz
from vesuvius.tifxyz.writer import write_tifxyz



def sheet_volume(shape=(64, 64, 64), z0=30.0, width=2.0):
    z = np.arange(shape[0])[:, None, None]
    v = 40 + 150 * np.exp(-0.5 * ((z - z0) / width) ** 2)
    return np.broadcast_to(v, shape).astype(np.uint8)


def _seg(path, z_right):
    yy, xx = np.meshgrid(np.linspace(4, 60, 40), np.linspace(4, 60, 40))
    z = np.where(xx >= 32, z_right, 30.0)
    return str(write_tifxyz(path, Tifxyz(_x=xx.astype(np.float32), _y=yy.astype(np.float32),
                                         _z=z.astype(np.float32), _scale=(1.0, 1.0), path=None)))


def test_drag_onto_the_sheet_reads_as_the_measured_offset(tmp_path):
    ct = tmp_path / "ct.tif"
    tifffile.imwrite(ct, sheet_volume())
    before = _seg(tmp_path / "before", 34.0)        # right half 4 voxels above the sheet
    after = _seg(tmp_path / "after", 30.0)          # annotator pulled it back on
    out = tmp_path / "audit"
    assert spa.main(["--surface", before, "--volume", str(ct), "--out", str(out), "--samples", "4000",
                     "--seed", "1", "--workers", "1", "--worst-cells", "10"]) == 0
    placement_edits.main(["--before", before, "--after", after, "--cells", str(out / "before.worst_cells.json"),
                          "--out", str(tmp_path / "edits"), "--volume", str(ct), "--half", "8"])
    rows = list(csv.DictReader(open(tmp_path / "edits" / "edits.csv")))
    assert rows and all(float(r["x"]) >= 32 for r in rows)
    for r in rows:
        assert float(r["median_shift"]) == pytest.approx(float(r["median_offset"]), abs=1.0)
        assert int(r["n_moved"]) == int(r["n"])
    arrs = np.load(tmp_path / "edits" / "edits.npz")
    assert arrs["ct_1"].shape == (16, 16, 16) and arrs["after_1"].shape == arrs["before_1"].shape
    prov = json.loads((tmp_path / "edits" / "provenance.json").read_text())
    assert prov["before"] == before and prov["audit"]["surface"] == "before" and prov["editor"]
    # --all-cells adds the on-sheet cells the audit never flagged, so the set has negatives,
    # and --prediction stores the model's surface next to the CT
    tifffile.imwrite(tmp_path / "pred.tif", (sheet_volume() > 100).astype(np.uint8))
    placement_edits.main(["--before", before, "--after", after, "--cells", str(out / "before.worst_cells.json"),
                          "--out", str(tmp_path / "all"), "--all-cells", "--volume", str(ct),
                          "--prediction", str(tmp_path / "pred.tif")])
    rows = list(csv.DictReader(open(tmp_path / "all" / "edits.csv")))
    left = [r for r in rows if r["flagged"] == "0"]
    assert left and all(float(r["x"]) < 32 and int(r["n_moved"]) == 0 for r in left)
    assert all(float(r["x"]) >= 32 for r in rows if r["flagged"] == "1")
    arrs = np.load(tmp_path / "all" / "edits.npz")
    assert arrs["pred_1"].shape == arrs["ct_1"].shape == (48, 48, 48) and arrs["pred_1"].max() == 1
    # no edit at all: every cell reads as unmoved
    placement_edits.main(["--before", before, "--after", before, "--cells", str(out / "before.worst_cells.json"),
                          "--out", str(tmp_path / "same")])
    assert all(int(r["n_moved"]) == 0 and abs(float(r["median_shift"])) < 1e-6
               for r in csv.DictReader(open(tmp_path / "same" / "edits.csv")))


def test_candidates_record_which_tracer_the_human_and_the_audit_each_pick(tmp_path):
    """With several tracers of one region, the export must say per cell which candidate the
    human's final surface sits on and which the audit would rank best; those two labels are
    what a tracer-choosing model trains on."""
    ct = tmp_path / "ct.tif"
    tifffile.imwrite(ct, sheet_volume())
    before = _seg(tmp_path / "before", 34.0)        # tracer A: right half 4 above the sheet
    on = _seg(tmp_path / "on", 30.0)                # tracer B: on the sheet
    under = _seg(tmp_path / "under", 26.0)          # tracer C: 4 below
    out = tmp_path / "audit"
    spa.main(["--surface", before, "--volume", str(ct), "--out", str(out), "--samples", "4000",
              "--seed", "1", "--workers", "1", "--worst-cells", "10"])
    placement_edits.main(["--before", before, "--after", on, "--cells", str(out / "before.worst_cells.json"),
                          "--out", str(tmp_path / "edits"), "--volume", str(ct), "--workers", "1",
                          "--candidate", f"B={on}", "--candidate", f"C={under}"])
    rows = list(csv.DictReader(open(tmp_path / "edits" / "edits.csv")))
    assert rows
    for r in rows:
        assert r["chosen"] == "B" and r["audit_pick"] == "B"
        assert float(r["dist_B"]) < 0.01 < float(r["dist_before"]) and abs(float(r["off_B"])) < 0.6
        assert abs(float(r["off_before"])) == pytest.approx(4.0, abs=0.6)
        assert abs(float(r["off_C"])) == pytest.approx(4.0, abs=0.6)
    arrs = np.load(tmp_path / "edits" / "edits.npz")
    assert arrs["cand_C_1"].shape == arrs["before_1"].shape
    prov = json.loads((tmp_path / "edits" / "provenance.json").read_text())
    assert prov["candidates"] == {"B": on, "C": under}
