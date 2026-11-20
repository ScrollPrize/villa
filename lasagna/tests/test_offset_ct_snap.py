"""Tests of offset_ct_snap and of the lasagna-offset-ct-snap command.

The fast tests use flat windings in a synthetic block: they pin the behaviour of the code (snap, check, recovery, inputs,
exit codes, tie-breaking, orientation, and opt-in flags) and are not evidence about papyrus. The last test runs on one real
1 mm^2 window of PHerc0139 read from the published volume; it is opt-in (LASAGNA_OFFSET_CT_SNAP_REAL=1, about 200 MB of CT
fetched), like the other real-run tests.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from vc_reader import images, tifxyz
from vc_reader.reader import ZarrWriter

from lasagna import offset_ct_snap as ocs
from lasagna import snap_offset_surface as cli

WINDINGS_Z = (150, 190, 230)    # three flat windings, 8 voxels thick, 32 voxels of air between them
ORG = np.array([0.0, 0.0, 0.0])
DATA = Path(__file__).resolve().parent / "data" / "offset_ct_snap" / "pherc0139_lot4_23_w027"
REAL_CT = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/0"
REAL_ROI = ["12025.289", "15121.054", "15915.914", "12531.616", "15587.34", "16474.97"]
REAL_SHA = {"x.tif": "bc2b734839ffecb0c2add4d8cf64b6bc9459fc547a84da6fc88f9e094692326c", "y.tif": "19b1bd602c93849e06c6c67c7ef88e58bac045381cbcd8641b38f64087e6c2dd",
            "z.tif": "69649090db7e4660e250f586659ddfb32623606a1dd892440edd6aac2169ce19", "snap_state.tif": "e90ed50e795846a056a7f9fab2b1e5fb5d2f22cd7d4f0ba543b57fdcc8b8d7f9"}
REAL_REPORT = {"verdict": 1, "accepted": True, "vertices_in_domain": 19704, "delivered": 19452, "anchored": 3097, "lifted": 15637, "recovered": 718, "not_delivered": 252}


def block(hole: bool = False) -> np.ndarray:
    vol = np.full((380, 200, 200), 30, np.uint8)           # (z, y, x), air
    rng = np.random.default_rng(0)
    for z in WINDINGS_Z:
        vol[z - 4:z + 4] = 160
    vol = np.clip(vol.astype(int) + rng.integers(-6, 7, vol.shape), 0, 255).astype(np.uint8)
    if hole:
        vol[WINDINGS_Z[1] - 6:WINDINGS_Z[1] + 6, 90:110, 90:110] = 30
    return vol


def grid(z: float, n: int = 6, tilt: float = 0.0) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(40, 160, n), np.linspace(40, 160, n), indexing="ij")
    return np.stack([x, y, z + tilt * (x - 100.0) / 60.0], -1)


def gap_fit() -> np.ndarray:
    """Like a real Offset fit: partly in the air below the target winding, partly touching it."""
    return grid(WINDINGS_Z[1] - 7, tilt=9.0)


def run(fit: np.ndarray, hole: bool = False, **kwargs):
    return ocs.snap_offset_surface(grid(WINDINGS_Z[0]), fit, block(hole), ORG, +1, **kwargs)


def test_dense_grid_and_step():
    g = grid(100.0)
    assert ocs.grid_step(g) == pytest.approx(24.0)
    d = ocs.dense_grid(g, 24.0, 4)
    assert d.shape == (31, 31, 3) and np.allclose(d[0, 0], g[0, 0]) and np.allclose(d[-1, -1], g[-1, -1])


def test_fit_in_the_gap_is_snapped_onto_the_next_winding_and_accepted():
    r = run(gap_fit())
    assert r["verdict"] == 1 and r["accepted"]
    placed = r["domain"] & (r["state"] > 0)
    assert placed.mean() > 0.95
    assert np.abs(r["xyz"][..., 2][placed] - WINDINGS_Z[1]).max() <= 4.5


def test_fit_left_on_the_start_winding_is_refused():
    r = run(grid(WINDINGS_Z[0] + 2))
    assert r["verdict"] != 1 and not r["accepted"] and int(r["state"].sum()) == 0


def test_fit_two_windings_away_is_refused():
    r = run(grid(WINDINGS_Z[2] - 3))
    assert r["verdict"] == 2 and not r["accepted"]


def test_no_vertex_is_invented_over_a_hole():
    r = run(gap_fit(), hole=True)
    x, y = r["xyz"][..., 0], r["xyz"][..., 1]
    over_hole = (x > 94) & (x < 106) & (y > 94) & (y < 106)
    assert r["accepted"] and over_hole.any() and not (r["state"][over_hole] > 0).any()


def test_recovery_and_stayed_filter_branches():
    # A fit with a localized dip > follow_tol_vox (6) but <= recover_tol_vox (12) exercises state 3 (recovery),
    # and a localized corner left on the start winding exercises the stayed filter on an accepted surface.
    fit = gap_fit()
    fit[2, 2, 2] = WINDINGS_Z[1] - 11.0
    fit[0, 0, 2] = WINDINGS_Z[0]
    r = run(fit)
    assert r["accepted"] and r["recovered"] > 0 and (r["state"] == 3).any()
    assert r["state"][0, 0] == 0


def test_undecided_verdict_when_too_few_rays_are_valid():
    # A small 2x2 grid (step 120) refined with dense_step_vox=60 yields 3x3 = 9 rays (< min_valid_rays=30) -> verdict -1.
    p = dict(ocs.PARAMS, dense_step_vox=60)
    r = ocs.snap_offset_surface(grid(WINDINGS_Z[0], n=2), grid(WINDINGS_Z[1], n=2), block(), ORG, +1, params=p)
    assert r["verdict"] == -1 and not r["accepted"] and int(r["state"].sum()) == 0


def test_tie_between_one_gap_and_two_gaps_is_refused():
    # Half the grid on winding 1 (1 gap) and half on winding 2 (2 gaps) must not be accepted on a tie.
    fit = grid(WINDINGS_Z[1])
    fit[3:, :, 2] = WINDINGS_Z[2]
    r = run(fit)
    assert not r["accepted"] and r["verdict"] == 2


def test_transposed_grid_refusal_and_auto_orient():
    # Transposing (row, col) flips cross(d/d_row, d/d_col): refused by default with negative move, accepted with auto_orient.
    start_t = np.transpose(grid(WINDINGS_Z[0]), (1, 0, 2))
    fit_t = np.transpose(gap_fit(), (1, 0, 2))
    r_def = ocs.snap_offset_surface(start_t, fit_t, block(), ORG, +1)
    assert not r_def["accepted"] and r_def["verdict"] == 0 and r_def["median_fit_normal_move_vox"] < -10.0
    r_auto = ocs.snap_offset_surface(start_t, fit_t, block(), ORG, +1, auto_orient=True)
    assert r_auto["accepted"] and r_auto["auto_oriented"]


def test_opt_in_smooth_within_paper_and_fallback_march():
    # 1) smooth_within_paper keeps vertices on the target winding while reducing normal variance
    r_sm = run(gap_fit(), smooth_within_paper=True)
    placed = r_sm["domain"] & (r_sm["state"] > 0)
    assert r_sm["accepted"] and np.abs(r_sm["xyz"][..., 2][placed] - WINDINGS_Z[1]).max() <= 4.5
    # 2) allow_march_fallback recovers the next winding when the fit moved forward but overshot (verdict 2)
    r_fb = run(grid(WINDINGS_Z[2] - 3), allow_march_fallback=True)
    placed_fb = r_fb["domain"] & (r_fb["state"] == 4)
    assert r_fb["verdict"] == 2 and not r_fb["accepted"] and r_fb["fallback_marched"] and r_fb["fallback_refused"] is None
    assert placed_fb.mean() > 0.95 and np.abs(r_fb["xyz"][..., 2][placed_fb] - WINDINGS_Z[1]).max() <= 4.5
    # 3) consensus_target_v6f promotes the second ridge across a start-winding delamination blister pocket
    mask = np.ones((20, 20), dtype=bool)
    events, b1, b2 = [], [], []
    for k in range(400):
        if k < 100:
            # Delaminated rays: narrow split w0=20 at e0=42 followed by wider gap w1=55 at e1=115
            events.append((np.array([42.0, 115.0], np.float32), np.array([20.0, 55.0], np.float32)))
            b1.append(25.0); b2.append(80.0)
        else:
            # Clean rays: wide gap w0=65 at e0=115
            events.append((np.array([115.0], np.float32), np.array([65.0], np.float32)))
            b1.append(60.0); b2.append(np.nan)
    b1_m, d1_s, _, prom_m = ocs.consensus_target_v6f(events, np.array(b1, np.float32), np.array(b2, np.float32), mask)
    assert prom_m.sum() == 100 and np.abs(d1_s - 115.0).max() < 2.0 and (b1_m >= 55.0).all()


def test_fallback_march_is_refused_when_the_fit_does_not_confirm_the_direction():
    # V6-FS: a fit left on the start winding, or a wrong --offset sign, must be refused, never marched to some winding.
    r_stay = run(grid(WINDINGS_Z[0] + 2), allow_march_fallback=True)
    assert r_stay["verdict"] == 0 and not r_stay["fallback_marched"] and int(r_stay["state"].sum()) == 0
    assert "did not move forward" in r_stay["fallback_refused"]
    r_sign = ocs.snap_offset_surface(grid(WINDINGS_Z[0]), gap_fit(), block(), ORG, -1, allow_march_fallback=True)
    assert not r_sign["accepted"] and not r_sign["fallback_marched"] and int(r_sign["state"].sum()) == 0
    assert r_sign["median_fit_normal_move_vox"] < 0 and "did not move forward" in r_sign["fallback_refused"]
    # a fit that moved a little (10 voxels, tilted, verdict 0) floats in the lower half of the first air gap: nearer the
    # start packet than the next sheet, it designates no sheet and gives a march nothing to rely on
    r_part = run(grid(WINDINGS_Z[0] + 10, tilt=9.0), allow_march_fallback=True)
    assert r_part["verdict"] == 0
    assert not r_part["accepted"] and not r_part["fallback_marched"] and int(r_part["state"].sum()) == 0
    assert "past the middle of the first air gap" in r_part["fallback_refused"] and r_part["median_fit_normal_move_vox"] >= ocs.PARAMS["min_forward_vox"]
    assert r_part["evidence"]["fit_past_mid_gap_share"] < ocs.PARAMS["march_min_past_mid_gap_share"] and r_part["delivery_basis"] is None


def test_fallback_march_is_refused_on_an_undecided_check():
    # Verdict -1 (too few valid rays): the check could not be made, so there is nothing to march against.
    p = dict(ocs.PARAMS, dense_step_vox=60)
    r = ocs.snap_offset_surface(grid(WINDINGS_Z[0], n=2), grid(WINDINGS_Z[1], n=2), block(), ORG, +1, params=p, allow_march_fallback=True)
    assert r["verdict"] == -1 and not r["fallback_marched"] and int(r["state"].sum()) == 0
    assert "undecided" in r["fallback_refused"]


def test_fallback_march_reports_where_its_heights_came_from():
    r = run(grid(WINDINGS_Z[2] - 3), allow_march_fallback=True)
    assert r["fallback_marched"] and r["fallback_height_from"] in ("fit", "ct_consensus")


def test_repair_start_pockets_is_separate_and_keeps_a_clean_accepted_surface():
    # Without a start-winding pocket, the opt-in repair leaves an accepted surface as it is; without the flag nothing runs.
    for hole in (False, True):
        a, b = run(gap_fit(), hole), run(gap_fit(), hole, repair_start_pockets=True)
        assert a["accepted"] and b["accepted"] and not b["fallback_marched"]
        placed = a["state"] > 0
        assert np.array_equal(a["state"], b["state"]) and np.array_equal(a["xyz"][placed], b["xyz"][placed])


def test_legacy_absolute_move_guard_is_off_by_default_and_still_available():
    # The march is no longer gated on an absolute fit move (V6-FR: 45 voxels, tuned on one scroll). A fit that reached the
    # next winding after moving 44 voxels is delivered; the old guard can still be requested through the parameters.
    fit = grid(WINDINGS_Z[0] + 44)
    r = run(fit, allow_march_fallback=True)
    assert r["verdict"] == 2 and r["fallback_marched"] and r["fallback_refused"] is None and r["delivery_basis"] == "ct_march"
    placed = r["domain"] & (r["state"] > 0)
    assert np.abs(r["xyz"][..., 2][placed] - WINDINGS_Z[1]).max() <= 4.5
    r45 = run(fit, allow_march_fallback=True, params=dict(ocs.PARAMS, fallback_min_fit_move_vox=45.0))
    assert not r45["fallback_marched"] and "fallback_min_fit_move_vox" in r45["fallback_refused"] and int(r45["state"].sum()) == 0


def test_auto_orient_with_fallback_is_reported():
    start_t = np.transpose(grid(WINDINGS_Z[0]), (1, 0, 2))
    fit_t = np.transpose(grid(WINDINGS_Z[2] - 3), (1, 0, 2))
    r = ocs.snap_offset_surface(start_t, fit_t, block(), ORG, +1, auto_orient=True, allow_march_fallback=True)
    assert r["auto_oriented"] and r["median_fit_normal_move_vox"] > 0
    r_no = ocs.snap_offset_surface(start_t, fit_t, block(), ORG, +1, allow_march_fallback=True)
    assert not r_no["fallback_marched"] and int(r_no["state"].sum()) == 0


def test_fallback_march_never_changes_an_accepted_surface():
    for fit, hole in ((gap_fit(), False), (gap_fit(), True)):
        a, b = run(fit, hole), run(fit, hole, allow_march_fallback=True)
        assert a["accepted"] and b["accepted"] and not b["fallback_marched"]
        assert np.array_equal(a["state"], b["state"]) and np.array_equal(np.where((a["state"] > 0)[..., None], a["xyz"], 0), np.where((b["state"] > 0)[..., None], b["xyz"], 0))


def test_bad_arrays_are_errors():
    for start, fit, vol in ((grid(150.0, 6), grid(180.0, 7), block()), (grid(150.0), grid(180.0), block().astype(np.uint16)), (grid(150.0)[..., :2], grid(180.0)[..., :2], block())):
        with pytest.raises(ValueError):
            ocs.snap_offset_surface(start, fit, vol, ORG, +1)
    with pytest.raises(ValueError):
        ocs.snap_offset_surface(grid(150.0), gap_fit(), block(), ORG, 0)


def test_degenerate_inputs_are_errors():
    good, fit, vol = grid(WINDINGS_Z[0]), gap_fit(), block()
    i, j = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
    line = np.stack([40.0 + 3.0 * (6 * i + j), np.full(i.shape, 40.0), np.full(i.shape, 150.0)], -1)      # 36 distinct vertices, all on one line
    dup = good.copy(); dup[2] = dup[1]                                                                    # two identical rows
    nan = fit.copy(); nan[0, 0, 0] = np.nan
    gone = fit.copy(); gone[3, 3] = -1.0
    for start, f, v, org, sign in ((line, fit, vol, ORG, +1), (good, line, vol, ORG, +1), (dup, fit, vol, ORG, +1), (good, nan, vol, ORG, +1), (good, gone, vol, ORG, +1),
                                   (good, fit, np.zeros_like(vol), ORG, +1), (good, fit, vol[:1], ORG, +1), (good, fit, vol[:0], ORG, +1),
                                   (good, fit, vol, np.array([0.0, np.nan, 0.0]), +1), (good, fit, vol, ORG, 2), (good, fit, vol, ORG, 0.5)):
        with pytest.raises(ValueError):
            ocs.snap_offset_surface(start, f, v, org, sign)
    assert ocs.snap_offset_surface(good, fit, vol, ORG, +1)["accepted"]                                 # the same call with sound inputs still runs


def test_deterministic_and_stage_export_changes_nothing():
    stages = {}
    a, b = run(gap_fit()), ocs.snap_offset_surface(grid(WINDINGS_Z[0]), gap_fit(), block(), ORG, +1, stages=stages)
    assert np.array_equal(a["state"], b["state"]) and np.array_equal(a["xyz"], b["xyz"])
    assert [e["stage"] for e in stages["events"]] == ["grids_densified", "start_snapped", "fit_snapped", "rays_marched", "surface_checked", "vertices_recovered", "surface_delivered"]


def test_acceptance_needs_a_lead_of_the_one_gap_rays_and_says_so():
    # "One air gap" must lead "no gap" and "two gaps" by more than accept_margin_share of the rays. The synthetic block
    # only produces clear-cut surfaces (the snap keeps one continuous patch), so the rule is pinned through the parameter:
    # the lead is reported, a lead that does not exceed the margin is an explicit refusal, and a margin of 0 is the
    # strict plurality of the first versions.
    r = run(gap_fit())
    ev = r["evidence"]
    assert r["accepted"] and ev["acceptance_margin"] == pytest.approx(ev["one_gap_share"] - max(ev["no_gap_share"], ev["double_share"]), abs=1e-3)
    assert ev["acceptance_margin"] > ocs.PARAMS["accept_margin_share"] == 0.05
    tie = run(gap_fit(), params=dict(ocs.PARAMS, accept_margin_share=ev["acceptance_margin"]))
    assert not tie["accepted"] and tie["verdict"] == 0 and int(tie["state"].sum()) == 0
    assert "near tie" in tie["refusal_reason"] and "accept_margin_share" in tie["refusal_reason"]
    strict = run(gap_fit(), params=dict(ocs.PARAMS, accept_margin_share=0.0))
    assert strict["accepted"] and np.array_equal(strict["state"], r["state"]) and np.array_equal(strict["xyz"], r["xyz"])


TOUCH_Z = 174    # a winding whose ridge is 24 voxels above the start ridge, with papyrus and no air between the two


def touching_block() -> np.ndarray:
    """Start winding at z = 150 and the next one at z = 174 pressed against it: two ridges in one packet, no air gap."""
    vol = np.full((380, 200, 200), 30, np.uint8)
    rng = np.random.default_rng(1)
    vol[146:154] = 160
    vol[154:170] = 118                      # papyrus-level filling between the two ridges: a density valley, not air
    vol[170:178] = 160
    vol[226:234] = 160                      # a third winding, beyond a real air gap
    return np.clip(vol.astype(int) + rng.integers(-6, 7, vol.shape), 0, 255).astype(np.uint8)


def test_touching_windings_are_refused_by_default_and_delivered_as_the_fit_sheet_on_request():
    # No air gap can be counted between two windings that touch: the default refuses (as before). With the fallback, the
    # sheet the fit designates is delivered, flagged as such, and nothing is marched to the winding beyond the air.
    start, fit, vol = grid(WINDINGS_Z[0]), grid(TOUCH_Z), touching_block()
    r = ocs.snap_offset_surface(start, fit, vol, ORG, +1)
    assert r["verdict"] == 0 and not r["accepted"] and int(r["state"].sum()) == 0 and r["delivery_basis"] is None
    assert "touch without air" in r["refusal_reason"] and r["fallback_refused"] is None
    f = ocs.snap_offset_surface(start, fit, vol, ORG, +1, allow_march_fallback=True)
    placed = f["domain"] & (f["state"] > 0)
    assert f["verdict"] == 0 and not f["accepted"] and f["fallback_marched"] and f["delivery_basis"] == "fit_sheet"
    assert placed.mean() > 0.95 and not (f["state"] == 4).any()
    assert np.abs(f["xyz"][..., 2][placed] - TOUCH_Z).max() <= 4.5                      # on the fit's sheet, not on z = 230
    assert f["evidence"]["no_gap_share"] > 0.9 and f["evidence"]["step_height_vox"] == pytest.approx(TOUCH_Z - WINDINGS_Z[0], abs=3.0)


def test_fit_sheet_never_delivers_the_start_winding():
    # Negative controls of the fallback on touching windings: a fit left on the start ridge, and a reversed sign.
    start, vol = grid(WINDINGS_Z[0]), touching_block()
    stay = ocs.snap_offset_surface(start, grid(WINDINGS_Z[0] + 8), vol, ORG, +1, allow_march_fallback=True)
    assert not stay["fallback_marched"] and int(stay["state"].sum()) == 0 and stay["refusal_reason"]
    sign = ocs.snap_offset_surface(start, grid(TOUCH_Z), vol, ORG, -1, allow_march_fallback=True)
    assert not sign["fallback_marched"] and int(sign["state"].sum()) == 0 and "did not move forward" in sign["refusal_reason"]
    # a fit that moved less than one sheet thickness designates no other sheet, even if a few of its vertices reach one:
    # it must not be carried to the touching winding by propagation from those few
    short = grid(WINDINGS_Z[0] + 10, tilt=7.0)                       # median move 10 voxels, its high edge near z = 167
    assert ocs.PARAMS["min_forward_vox"] <= 10 < ocs.PARAMS["fit_sheet_min_step_vox"]
    r = ocs.snap_offset_surface(start, short, vol, ORG, +1, allow_march_fallback=True)
    assert not r["fallback_marched"] and int(r["state"].sum()) == 0 and "less than one sheet thickness" in r["refusal_reason"]


def test_fit_sheet_never_keeps_a_vertex_on_the_start_side():
    # Two thirds of the fit on the touching winding, one third left on the start ridge. No ridge closer to the start than
    # one sheet thickness is a candidate: the vertices left behind are either hooked to the fit's sheet by continuity with
    # their neighbours (here the sheet exists under them) or left out, never delivered on the start winding.
    fit = grid(TOUCH_Z)
    fit[4:, :, 2] = WINDINGS_Z[0] + 2
    f = ocs.snap_offset_surface(grid(WINDINGS_Z[0]), fit, touching_block(), ORG, +1, allow_march_fallback=True)
    placed = f["domain"] & (f["state"] > 0)
    assert f["delivery_basis"] == "fit_sheet" and placed.mean() >= 0.55
    assert f["xyz"][..., 2][placed].min() > WINDINGS_Z[0] + ocs.PARAMS["fit_sheet_min_step_vox"] - 4.5
    assert np.abs(f["xyz"][..., 2][placed] - TOUCH_Z).max() <= 4.5


def _flat(h):
    x, y = np.meshgrid(np.arange(12.0) * 4, np.arange(12.0) * 4, indexing="ij")
    return np.stack([x, y, np.broadcast_to(np.asarray(h, dtype=float), x.shape)], -1)


def test_pocket_repair_is_transactional():
    level = np.full((12, 12), 50.0)
    st = np.ones((12, 12), np.int8)
    # (a) vertices that were at the level of the accepted surface are moved one winding away: rolled back
    h_b = np.full((12, 12), 50.0); h_a = h_b.copy(); h_a[:, 6:] = 103.0
    st_a = st.copy(); st_a[:, 6:] = 4
    Q, state, kept, rolled = ocs.commit_repairs(_flat(h_b), st, h_b, _flat(h_a), st_a, h_a, level)
    assert kept == 0 and rolled == 72 and np.array_equal(Q, _flat(h_b)) and np.array_equal(state, st)
    # (b) a real pocket, a depression below the level, is brought back to it: kept
    h_b = np.full((12, 12), 50.0); h_b[4:8, 4:8] = 14.0; h_a = np.full((12, 12), 50.0); h_a[4:8, 4:8] = 47.0
    st_a = st.copy(); st_a[4:8, 4:8] = 3
    Q, state, kept, rolled = ocs.commit_repairs(_flat(h_b), st, h_b, _flat(h_a), st_a, h_a, level)
    assert kept == 16 and rolled == 0 and np.array_equal(Q, _flat(h_a)) and (state[4:8, 4:8] == 3).all()
    # (c) a pocket sent far beyond the level is not a repair: rolled back
    h_a2 = np.full((12, 12), 50.0); h_a2[4:8, 4:8] = 101.0
    Q, state, kept, rolled = ocs.commit_repairs(_flat(h_b), st, h_b, _flat(h_a2), st_a, h_a2, level)
    assert kept == 0 and rolled == 16 and np.array_equal(Q, _flat(h_b))
    # (d) nothing moved: nothing to commit
    assert ocs.commit_repairs(_flat(h_b), st, h_b, _flat(h_b), st, h_b, level)[2:] == (0, 0)


def test_report_states_the_basis_the_evidence_and_the_reason():
    acc = run(gap_fit(), repair_start_pockets=True)
    assert acc["delivery_basis"] == "air_gap" and acc["refusal_reason"] is None and acc["start_pockets_rolled_back"] == 0
    ev = acc["evidence"]
    assert set(ev) >= {"fit_move_vox", "one_gap_share", "no_gap_share", "double_share", "fit_left_start_packet_share", "fit_past_mid_gap_share", "step_height_vox"}
    assert ev["one_gap_share"] > 0.9 and ev["step_height_vox"] == pytest.approx(WINDINGS_Z[1] - WINDINGS_Z[0], abs=3.0)
    ref = run(grid(WINDINGS_Z[0] + 2))
    assert ref["delivery_basis"] is None and ref["refusal_reason"] and ref["evidence"]["no_gap_share"] > 0.9
    far = run(grid(WINDINGS_Z[2] - 3), allow_march_fallback=True)
    assert far["delivery_basis"] == "ct_march" and far["evidence"]["double_share"] > ocs.PARAMS["fit_sheet_max_double_share"]


def write_grid(folder: Path, xyz: np.ndarray) -> Path:
    folder.mkdir(parents=True)
    for k, c in enumerate("xyz"):
        images.write_tiff(str(folder / f"{c}.tif"), np.ascontiguousarray(xyz[..., k].astype(np.float32)), compression=None)
    (folder / "meta.json").write_text(json.dumps({"uuid": folder.name, "name": folder.name, "format": "tifxyz", "type": "seg", "scale": [1 / 24.0, 1 / 24.0]}))
    return folder


@pytest.fixture()
def world(tmp_path):
    vol = np.full((384, 256, 256), 30, np.uint8)
    vol[:380, :200, :200] = block()
    ZarrWriter.create(tmp_path / "ct.zarr", shape=vol.shape, chunks=(128, 128, 128), dtype="uint8")[:] = vol
    return {"ct": str(tmp_path / "ct.zarr"), "tmp": tmp_path, "start": write_grid(tmp_path / "start.tifxyz", grid(WINDINGS_Z[0])),
            "fit": write_grid(tmp_path / "fit.tifxyz", gap_fit()), "stay": write_grid(tmp_path / "stay.tifxyz", grid(WINDINGS_Z[0] + 2))}


def args(w, fit="fit", out="out.tifxyz", offset="1", ct=None):
    return ["--start", str(w["start"]), "--fit", str(w[fit]), "--offset", offset, "--ct", ct or w["ct"], "--out", str(w["tmp"] / out), "--roi", "2", "2", "2", "250", "250", "380"]


def test_command_writes_a_readable_tifxyz(world):
    assert cli.main(args(world)) == 0
    out = world["tmp"] / "out.tifxyz"
    assert sorted(p.name for p in out.iterdir()) == ["meta.json", "snap_report.json", "snap_state.tif", "x.tif", "y.tif", "z.tif"]
    rep = json.loads((out / "snap_report.json").read_text()); meta = json.loads((out / "meta.json").read_text())
    state = np.asarray(images.read(str(out / "snap_state.tif")))
    surf = tifxyz.read(out, mask=False)                                    # villa's own reader accepts the output
    assert surf.xyz.shape == (31, 31, 3) and np.array_equal(np.asarray(surf.valid), state > 0)
    assert rep["accepted"] and rep["delivered"] == int((state > 0).sum()) == rep["anchored"] + rep["lifted"] + rep["recovered"]
    assert np.abs(np.asarray(surf.xyz)[..., 2][state > 0] - WINDINGS_Z[1]).max() <= 4.5
    assert meta["extra_channels"] == [{"name": "snap_state", "file": "snap_state.tif", "dtype": "uint8", "range": [0, 3]}]
    assert meta["scale"][0] == pytest.approx(1 / 4.0) and meta["format"] == "tifxyz"


def test_command_refusal_and_errors(world, capsys):
    assert cli.main(args(world, fit="stay", out="refused.tifxyz")) == 2
    err = capsys.readouterr().err                                                                         # the refusal says why
    assert "refused, verdict 0" in err and "#2418" in err and "offset.json" in err
    assert cli.main(args(world, fit="fit", out="wrong_sign.tifxyz", offset="-1")) == 2
    err_sign = capsys.readouterr().err                                                                    # opposite move points to --offset / transposed grid
    assert "refused, verdict 0" in err_sign and "--offset" in err_sign
    assert int(np.asarray(images.read(str(world["tmp"] / "refused.tifxyz" / "snap_state.tif"))).sum()) == 0
    refused = tifxyz.read(world["tmp"] / "refused.tifxyz", mask=False)                                  # still a tifxyz villa reads: no vertex, the empty box
    assert not np.asarray(refused.valid).any()
    assert json.loads((world["tmp"] / "refused.tifxyz" / "meta.json").read_text())["bbox"] == [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
    assert cli.main(args(world, fit="stay", out="refused.tifxyz")) == 3                                  # never overwritten
    w = dict(world); w["absent"] = world["tmp"] / "absent.tifxyz"; w["small"] = write_grid(world["tmp"] / "small.tifxyz", grid(WINDINGS_Z[1], 5))
    for k, (case, extra) in enumerate((("absent", {}), ("small", {}), ("fit", {"offset": "0"}), ("fit", {"ct": str(world["tmp"] / "no_such.zarr")}))):
        assert cli.main(args(w, fit=case, out=f"bad{k}.tifxyz", **extra)) == 3
        assert not (world["tmp"] / f"bad{k}.tifxyz").exists()


def test_command_fallback_march_has_its_own_exit_code(world, capsys):
    far = write_grid(world["tmp"] / "far.tifxyz", grid(WINDINGS_Z[2] - 3))
    w = dict(world, far=far)
    assert cli.main(args(w, fit="far", out="marched.tifxyz") + ["--fallback-march"]) == 4
    err = capsys.readouterr().err
    assert "NOT accepted" in err and "--fallback-march" in err
    rep = json.loads((world["tmp"] / "marched.tifxyz" / "snap_report.json").read_text())
    state = np.asarray(images.read(str(world["tmp"] / "marched.tifxyz" / "snap_state.tif")))
    assert not rep["accepted"] and rep["fallback_marched"] and rep["marched"] == int((state == 4).sum()) == rep["delivered"] > 0
    assert rep["delivery_basis"] == "ct_march" and rep["refusal_reason"] is None and rep["evidence"]["double_share"] > 0.5
    assert cli.main(args(w, fit="fit", out="sign.tifxyz", offset="-1") + ["--fallback-march"]) == 2
    err = capsys.readouterr().err
    assert "refused" in err and "--fallback-march delivered nothing" in err
    assert int(np.asarray(images.read(str(world["tmp"] / "sign.tifxyz" / "snap_state.tif"))).sum()) == 0


def test_command_delivers_the_fit_sheet_with_exit_code_4_and_says_on_what_basis(tmp_path, capsys):
    vol = np.full((384, 256, 256), 30, np.uint8)
    vol[:380, :200, :200] = touching_block()
    ZarrWriter.create(tmp_path / "ct.zarr", shape=vol.shape, chunks=(128, 128, 128), dtype="uint8")[:] = vol
    w = {"ct": str(tmp_path / "ct.zarr"), "tmp": tmp_path, "start": write_grid(tmp_path / "start.tifxyz", grid(WINDINGS_Z[0])), "fit": write_grid(tmp_path / "fit.tifxyz", grid(TOUCH_Z))}
    assert cli.main(args(w, out="refused.tifxyz")) == 2
    rep = json.loads((tmp_path / "refused.tifxyz" / "snap_report.json").read_text())
    assert rep["delivery_basis"] is None and "touch without air" in rep["refusal_reason"] and rep["delivered"] == 0
    capsys.readouterr()
    assert cli.main(args(w, out="sheet.tifxyz") + ["--fallback-march"]) == 4
    err = capsys.readouterr().err
    assert "NOT accepted" in err and "fit_sheet" in err and "rests on the Offset fit alone" in err
    rep = json.loads((tmp_path / "sheet.tifxyz" / "snap_report.json").read_text())
    state = np.asarray(images.read(str(tmp_path / "sheet.tifxyz" / "snap_state.tif")))
    assert rep["delivery_basis"] == "fit_sheet" and rep["fallback_marched"] and rep["marched"] == 0 and rep["delivered"] == int((state > 0).sum()) > 0
    surf = tifxyz.read(tmp_path / "sheet.tifxyz", mask=False)
    assert np.abs(np.asarray(surf.xyz)[..., 2][state > 0] - TOUCH_Z).max() <= 4.5


def test_command_names_the_volume_and_refuses_another_one(world):
    assert cli.main(args(world, out="id.tifxyz")) == 0
    rep = json.loads((world["tmp"] / "id.tifxyz" / "snap_report.json").read_text()); ident = rep["ct_identity"]
    assert ident["shape_zyx"] == [384, 256, 256] and ident["chunks_zyx"] == [128, 128, 128] and ident["dtype"] == "uint8" and len(ident["crop_sha256"]) == 64
    assert rep["sources"] == {"start_uuid": "start.tifxyz", "fit_uuid": "fit.tifxyz"}
    w = dict(world); w["named"] = write_grid(world["tmp"] / "named.tifxyz", gap_fit())
    meta = json.loads((w["named"] / "meta.json").read_text())
    (w["named"] / "meta.json").write_text(json.dumps(dict(meta, base_shape_zyx=[384, 256, 256])))
    assert cli.main(args(w, fit="named", out="same_volume.tifxyz")) == 0                                    # the surface names this volume
    assert json.loads((world["tmp"] / "same_volume.tifxyz" / "snap_report.json").read_text())["ct_identity"]["checked_against"] == ["--fit"]
    (w["named"] / "meta.json").write_text(json.dumps(dict(meta, base_shape_zyx=[999, 256, 256])))
    assert cli.main(args(w, fit="named", out="other_volume.tifxyz")) == 3 and not (world["tmp"] / "other_volume.tifxyz").exists()
    (w["named"] / "meta.json").write_text(json.dumps(dict(meta, base_shape_zyx=[999, 256, 256], export_coordinate_scale=0.5)))
    assert cli.main(args(w, fit="named", out="other_scale.tifxyz")) == 3                                      # still checked at another export scale
    (w["named"] / "meta.json").write_text(json.dumps(dict(meta, base_shape_zyx=[384, 256, 256], export_coordinate_scale=0.5)))
    assert cli.main(args(w, fit="named", out="same_scale.tifxyz")) == 0
    assert json.loads((world["tmp"] / "same_scale.tifxyz" / "snap_report.json").read_text())["ct_identity"]["checked_against"] == ["--fit"]


def test_command_refuses_degenerate_requests(world):
    ZarrWriter.create(world["tmp"] / "masked.zarr", shape=(384, 256, 256), chunks=(128, 128, 128), dtype="uint8")[:] = np.zeros((384, 256, 256), np.uint8)
    base = args(world, out="never.tifxyz"); r = base.index("--roi")
    for k, a in enumerate((base[:r] + ["--roi", "250", "250", "380", "2", "2", "2"],                         # inverted box
                           base[:r] + ["--roi", "210", "210", "2", "250", "250", "100"],                      # box away from the surfaces
                           args(world, out="never.tifxyz", ct=str(world["tmp"] / "masked.zarr")),               # an all-masked volume
                           args(world, out="never.tifxyz", offset="2"))):
        assert cli.main(a) == 3, k
        assert not (world["tmp"] / "never.tifxyz").exists()


@pytest.mark.skipif(not os.environ.get("LASAGNA_OFFSET_CT_SNAP_REAL"), reason="set LASAGNA_OFFSET_CT_SNAP_REAL=1 to run on the published PHerc0139 scan")
def test_real_window_of_pherc0139(tmp_path):
    out = tmp_path / "snapped.tifxyz"
    ct_src = os.environ.get("LASAGNA_OFFSET_CT_SNAP_CT_OVERRIDE", REAL_CT)
    assert cli.main(["--start", str(DATA / "start.tifxyz"), "--fit", str(DATA / "fit.tifxyz"), "--offset", "1", "--ct", ct_src, "--roi", *REAL_ROI, "--out", str(out)]) == 0
    report = json.loads((out / "snap_report.json").read_text())
    assert {k: report[k] for k in REAL_REPORT} == REAL_REPORT
    for name, digest in REAL_SHA.items():
        assert hashlib.sha256(np.ascontiguousarray(np.asarray(images.read(str(out / name)))).tobytes()).hexdigest() == digest, name
