"""vc_tomography.geometry: the reconstruction's terms from a scan's poses, and the files the reader refuses.
tests/test_geometry_solve.py covers the step that makes calibrations, its file and the poses it gives."""
import numpy as np
import pytest

from vc_tomography import geometry as G


def test_reconstruction_terms_hold_the_poses():
    """as the reconstruction's terms: the slice h lies at the row h - (z_pix + z_readjustment + row_offset) = h + row + origin,
    and the merged column the axis reads is the scan's axis"""
    rng = np.random.default_rng(3)
    n = 300
    H = 0.5 + 150.0 * np.arange(n) / 100
    layout = dict(rows=200, row_sign=-1.0)
    for s in (0, 1):
        p = dict(row=rng.normal(0, 2, n), axis=140.0 + rng.normal(0, 0.3, n), row_per_col=np.full(n, -1.5e-5), col_per_row=np.full(n, 1.5e-5),
                 col_scale=np.zeros(n), row_scale=np.full(n, 1e-3), angle_deg=3.6 * np.arange(n), scale_col=0.0)
        t = G.reconstruction_terms(p, H, layout, rot_center=210.25, column_offset=70.5 * (1 - s), z_readjustment=0.5 * s, row_origin=0.25)
        z_pix = -(H - H[0])
        assert np.allclose(-(z_pix * t["z_translation_scale"] + 0.5 * s + t["row_offset"]), p["row"] + 0.25, atol=2e-4)
        assert np.allclose(210.25 - 70.5 * (1 - s) + t["col_offset"], p["axis"], atol=2e-4)
        assert np.allclose(t["angles_deg"], p["angle_deg"])


def test_files_without_the_joint_model_are_refused():
    """a calibration file without "model": "joint" (here one with a motion stage) is refused"""
    old = dict(layout=dict(rows=200, cols=240, frames=[dict(rest1=10, helix=20, rest2=10)] * 2, row_sign=-1.0, coarse_move=70),
               row_time=dict(seconds_per_row=-2e-5, frame_period_s=0.033), poses=dict(motion={}))
    with pytest.raises((ValueError, RuntimeError), match="joint"):
        G.scan_poses(old, [np.zeros(40)] * 2, [np.zeros(40)] * 2)
    with pytest.raises((ValueError, RuntimeError), match="joint"):
        G.rolling_shutter_of(old)
