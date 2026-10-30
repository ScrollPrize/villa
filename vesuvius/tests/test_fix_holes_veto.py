"""A vetoed hole must stay open while the un-vetoed hole in the same crop is still filled."""
import csv
import numpy as np
import tifffile
from vesuvius.scripts import fix_holes


def _crop(tmp_path, name):
    label = np.zeros((20, 40, 40), np.uint8)
    label[10] = 1                      # one flat sheet
    label[10, 8:12, 8:12] = 0          # hole A, centre (10, 9.5, 9.5)
    label[10, 28:32, 28:32] = 0        # hole B, centre (10, 29.5, 29.5)
    mask = np.zeros_like(label, bool)
    mask[10, 8:12, 8:12] = True
    mask[10, 28:32, 28:32] = True
    d = tmp_path / name
    d.mkdir()
    tifffile.imwrite(d / f"{name}_surf.tif", label)
    tifffile.imwrite(d / "hole_mask.tif", mask.astype(np.uint8))
    return d


def test_vetoed_hole_stays_open(tmp_path):
    d = _crop(tmp_path, "sample_1")
    veto = tmp_path / "veto.csv"
    with open(veto, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop", "center_zyx"])
        w.writerow(["sample_1", "[10, 10, 10]"])   # hole A is a real gap: leave it
    centers = fix_holes._load_veto(veto)
    assert centers == {"sample_1": [(10, 10, 10)]}
    ok, msg = fix_holes._process_folder(
        d, hole_mask_name="hole_mask.tif", label_glob="*_surf.tif", fg_value=1, ignore_value=None,
        suffix="t", methods=["2d_hole_fill"], closing_kernel=2, alpha=0.005, iters=1, mask_dilate=0,
        veto_centers=centers["sample_1"],
    )
    assert ok and "[VETO] 1 hole" in msg
    out = tifffile.imread(d / "sample_1_surf_t.tif")
    assert out[10, 8:12, 8:12].sum() == 0        # vetoed hole untouched
    assert out[10, 28:32, 28:32].sum() == 16     # the other hole filled


def test_far_centre_vetoes_nothing():
    mask = np.zeros((5, 5, 5), bool)
    mask[2, 2, 2] = True
    kept, n = fix_holes._apply_veto(mask, [(0, 0, 0)], max_dist=1.0)
    assert n == 0 and kept.sum() == 1
