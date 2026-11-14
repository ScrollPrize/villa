"""Tests of ridge_correct_checkpoint.py on a synthetic fit and a synthetic CT.

No scroll data: the fit is a small random SpiralAndTransform and the CT puts
a bright ridge at W = k + d(theta) for a known d (tests/ridge_correction_fixture.py).
These tests check correctness of the plumbing; they are not evidence that the
correction helps on real scrolls.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import ridge_correction_fixture as fixture
import torch

import ridge_correct_checkpoint as cli
from ridge_correction import estimate as est
from ridge_correction import gapmap as WB
from ridge_correction import stage
from ridge_correction import writer as W3
from ridge_correction.ctstore import ZarrChunkStore


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("ridge")
    ckpt, umb = fixture.make_fit(root / "fit")
    ct = fixture.make_ct(root / "ct.zarr", ckpt, umb)
    return ckpt, umb, ct


@pytest.fixture(scope="module", autouse=True)
def short_taper():
    # The method's 150-voxel z taper is longer than the 96-slice synthetic volume.
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(stage.SETTINGS, "taper", 12.0)
        yield


@pytest.fixture(scope="module")
def base(synthetic, tmp_path_factory):
    """One default run of the command, shared by the tests that compare against it."""
    d = tmp_path_factory.mktemp("base")
    return d, _run(d, synthetic, name="base.ckpt")


def _run(tmp_path, synthetic, *extra, name="out.ckpt", slab_rows=4, stop_after_slabs=None):
    ckpt, _umb, ct = synthetic
    args = cli._parse_args([str(ckpt), str(tmp_path / name), "--ct", str(ct), *extra])
    return cli.run(args, log=lambda *a: None, slab_rows=slab_rows, stop_after_slabs=stop_after_slabs)


def _logits(path):
    return torch.load(path, weights_only=False)["spiral_and_transform"]["gap_expander_params.logits"][0, 0].numpy()


def _stacks(tmp_path, name):
    work = tmp_path / f"{name}.work"
    plan = json.loads((work / "plan.json").read_text())
    return np.concatenate([np.load(work / f"slab_{a:05d}_{b:05d}.npy") for a, b in plan["slabs"]])


def _node_angles(geo):
    idx = geo["idx"]
    wc = geo["Wc"]
    coord = np.arange(wc) * wc / (wc - 1)
    j = np.clip(np.searchsorted(idx, coord, "right") - 1, 0, len(idx) - 2)
    return (coord - idx[j]) / np.diff(idx)[j] * 2 * np.pi


def _planted_field(fit):
    th = _node_angles(fit.geo)
    idx = fit.geo["idx"].astype(int)
    U = np.zeros((fit.geo["Z"], fit.geo["Wc"]))
    U[:, idx[0]:idx[6]] = fixture.planted_offset(th[idx[0]:idx[6]])[None]
    return U


# ---------------------------------------------------------------------------
# Estimator and writer on their own (no checkpoint, no CT)
# ---------------------------------------------------------------------------

def test_estimator_recovers_known_shifts_of_synthetic_profiles():
    rng = np.random.default_rng(1)
    n = 4000
    s = rng.uniform(-0.3, 0.3, n)

    def ridge(x, w=0.07):
        return np.exp(-0.5 * (x / w) ** 2)

    U = est.U
    P = (ridge(U[None] - s[:, None]) + 0.5 * ridge(U[None] - s[:, None] - 1)
         + 0.5 * ridge(U[None] - s[:, None] + 1) + 0.15 * rng.standard_normal((n, len(U))))
    E = est.Est(tmax=0.40, win=0.55)
    T, _ = est.sheet_template(E, P)
    u, c = E.estimate(P, T)
    m = c >= 0.70
    err = u[m] - s[m]
    slope = float((s[m] * u[m]).sum() / (s[m] ** 2).sum())
    assert m.mean() > 0.5
    assert abs(np.median(err)) < 0.02 and np.abs(err).mean() < 0.03
    assert 0.95 < slope < 1.05


def _strip(saturated):
    """A gap strip with a steep softplus (like real fits); optionally ~15% of the cells
    saturated at the minimum gap, where the gap map is flat and gap -> logit does not invert."""
    rng = np.random.default_rng(3)
    gm = WB.GapMap.__new__(WB.GapMap)
    gm.dr, gm.min_gap, gm.bias, gm.k, gm.lr = 14.952, 2.5, 4.0, 969.895, 0.3
    gm.den = float(np.log1p(np.exp(-abs(gm.bias))) + max(gm.bias, 0.0))
    num = np.array([20, 22, 25, 27, 29, 31, 34, 36, 38])
    idx = np.concatenate([[0], np.cumsum(num)]).astype(float)
    wc = int(idx[-1])
    L0 = gm.gapinv(gm.dr * rng.uniform(0.6, 1.4, (6, wc)))
    sat = rng.uniform(0, 1, (6, wc)) < (0.15 if saturated else 0.0)
    L0 = np.where(sat, rng.uniform(-0.5, -0.02, (6, wc)), L0)
    L0[:, 0] = 0.0
    return gm, idx, num, L0.astype(np.float32).astype(np.float64), sat, rng


def _write_strip(gm, idx, L0, U):
    Ln, Ueff, info = W3.write_field(L0, gm, idx, gm.dr, U, gfloor=3.0)
    return np.where(np.abs(Ln - L0) < 1e-12, L0, Ln), Ueff, info


def _intended(gm, idx, num, L, U, k, t):
    Lp = L.copy()
    Lp[:, 0] = 0.0
    coord = idx[k - 1] + t * num[k - 1]
    wc = int(idx[-1])
    ub = WB.interp_cols(U, coord, wc)
    g_in = gm.gapf(WB.interp_cols(Lp, coord, wc))
    g_out = gm.gapf(WB.interp_cols(Lp, idx[k] + t * num[k], wc)) if k < len(num) else g_in
    return WB.radii_at(Lp, gm, idx, gm.dr, k, t) + np.where(ub > 0, ub * g_out, ub * g_in)


@pytest.mark.parametrize("saturated", [False, True])
def test_writer_zero_field_is_bit_identical(saturated):
    gm, idx, _num, L0, _sat, _ = _strip(saturated)
    Ln, info, _ = W3.solve_rows(L0, gm, idx, gm.dr, np.zeros_like(L0))
    assert np.array_equal(Ln, L0) and info["row_segments_solved"] == 0


def test_writer_keeps_untouched_columns_and_rows_bit_identical():
    gm, idx, num, L0, sat, rng = _strip(saturated=True)
    nw = len(num)
    U = np.zeros_like(L0)
    U[:3, int(idx[nw - 1]) + 1:int(idx[nw])] = rng.uniform(-0.25, 0.25, (3, int(num[-1]) - 1))
    Ln, _, _ = _write_strip(gm, idx, L0, U)
    keep = np.ones(L0.shape[1], bool)
    keep[int(idx[nw - 1]) - 2:] = False  # the last 2 columns of the inner winding share samples with the moved one
    assert sat[:, keep].sum() > 0
    assert np.array_equal(Ln[:, keep], L0[:, keep]) and np.array_equal(Ln[3:], L0[3:])


@pytest.mark.parametrize("saturated", [False, True])
def test_writer_hits_the_nodes_including_each_windings_last_node(saturated):
    gm, idx, num, L0, _sat, _rng = _strip(saturated)
    U = np.zeros_like(L0)
    th_all = WB.strip_geometry(idx)[3]
    for k in (2, 3, 4):
        cols = np.arange(int(idx[k - 1]), int(idx[k]))
        U[:, cols] = 0.15 * np.sin(2 * np.pi * th_all[cols])[None]
    Ln, Ueff, info = _write_strip(gm, idx, L0, U)
    assert info["floor_violations_left"] == 0
    Lp = Ln.copy()
    Lp[:, 0] = 0.0
    last, other = [], []
    for k in range(1, len(num) + 1):
        tn = th_all[int(idx[k - 1]):int(idx[k])]
        e = (WB.radii_at(Lp, gm, idx, gm.dr, k, tn) - _intended(gm, idx, num, L0, Ueff, k, tn)) / gm.dr
        other.append(e[:, :-1].ravel())
        last.append(e[:, -1])
    rms_last = float(np.sqrt((np.concatenate(last) ** 2).mean()))
    rms_other = float(np.sqrt((np.concatenate(other) ** 2).mean()))
    # saturated cells cannot follow a move (their gap map is flat), so only the plain strip is near exact
    assert rms_other < (0.05 if saturated else 0.01), rms_other
    assert rms_last < 2 * rms_other + 1e-6, (rms_last, rms_other)


def test_writer_scales_down_a_crossing_move_instead_of_breaking_the_gap_floor():
    gm, idx, _num, L0, _sat, _ = _strip(saturated=True)
    U = np.zeros_like(L0)
    c3, c4 = int(idx[2]) + 5, int(idx[3]) + 5
    U[:, c3:c3 + 3] = 0.9
    U[:, c4:c4 + 3] = -0.5
    _, Ueff, info = _write_strip(gm, idx, L0, U)
    assert info["floor_violations_left"] == 0 and np.abs(Ueff).max() < 0.9


# ---------------------------------------------------------------------------
# Checkpoint write and the fidelity check
# ---------------------------------------------------------------------------

def test_zero_field_keeps_the_model_state_and_its_digest(synthetic, tmp_path):
    ckpt, umb, _ = synthetic
    fit = stage.load_fit(ckpt, umb)
    U = np.zeros((fit.geo["Z"], fit.geo["Wc"]))
    rep, _ = stage.write(fit, U, tmp_path / "zero.ckpt", band=(0.0, 96.0), windings=(1, 6))
    before = torch.load(ckpt, weights_only=False)
    after = torch.load(tmp_path / "zero.ckpt", weights_only=False)
    assert rep["logits_changed"] == 0 and rep["input_digest_consistent"]
    assert after["model_state_sha256"] == before["model_state_sha256"]
    sa, sb = after["spiral_and_transform"], before["spiral_and_transform"]
    assert list(sa) == list(sb)
    for key in sb:
        assert sa[key].dtype == sb[key].dtype and torch.equal(sa[key], sb[key]), key
    assert set(after) == set(before) | {stage.PROVENANCE_KEY}
    for key in before:
        if key != "spiral_and_transform":
            assert repr(after[key]) == repr(before[key]), key


def test_a_known_field_is_realised_and_untouched_rows_keep_their_logits(synthetic, tmp_path):
    ckpt, umb, _ = synthetic
    fit = stage.load_fit(ckpt, umb)
    U = _planted_field(fit)
    U[:3] = 0.0
    U[10:] = 0.0
    rep, _Ueff = stage.write(fit, U, tmp_path / "known.ckpt", band=(0.0, 96.0), windings=(1, 6))
    assert rep["fidelity_failures"] == [] and rep["fidelity"]["corr"] > 0.99 and rep["fidelity"]["ratio"] < 0.1
    assert rep["verify_at_nodes"]["rms_err_w"] < 0.005 and abs(rep["verify_at_nodes"]["slope"] - 1) < 0.02
    changed = _logits(tmp_path / "known.ckpt").view(np.uint32) != _logits(ckpt).view(np.uint32)
    assert changed.any() and not changed[:3].any() and not changed[10:].any()
    after = torch.load(tmp_path / "known.ckpt", weights_only=False)
    from checkpoint_io import model_state_sha256

    assert after["model_state_sha256"] == model_state_sha256(after["spiral_and_transform"], (), 0, 96)
    for key, value in torch.load(ckpt, weights_only=False)["spiral_and_transform"].items():
        if key != "gap_expander_params.logits":
            assert torch.equal(after["spiral_and_transform"][key], value), key


def test_the_fidelity_bar():
    assert stage.fidelity_failures({"corr": 0.95, "ratio": 0.30}) == []
    assert "correlation" in stage.fidelity_failures({"corr": 0.85, "ratio": 0.30})[0]
    assert "error ratio" in stage.fidelity_failures({"corr": 0.95, "ratio": 0.60})[0]
    assert len(stage.fidelity_failures({"corr": None, "ratio": None})) == 2


def test_a_field_the_writer_must_scale_down_fails_the_error_ratio(synthetic, tmp_path):
    # windings 3 and 4 pushed toward each other by 0.9 and 0.6 winding: the writer halves the moves rather than
    # cross the gap floor, so the written shift is far below the asked one (correlation stays high)
    ckpt, umb, _ = synthetic
    fit = stage.load_fit(ckpt, umb)
    idx = fit.geo["idx"].astype(int)
    U = np.zeros((fit.geo["Z"], fit.geo["Wc"]))
    U[:, idx[2]:idx[3]] = 0.9
    U[:, idx[3]:idx[4]] = -0.6
    with pytest.raises(stage.FidelityError, match="error ratio between nodes") as err:
        stage.write(fit, U, tmp_path / "crossing.ckpt", band=(0.0, 96.0), windings=(1, 6))
    fid = err.value.report["fidelity"]
    assert fid["ratio"] > stage.FIDELITY["between_ratio_max"] and fid["corr"] >= stage.FIDELITY["between_corr_min"]
    assert not (tmp_path / "crossing.ckpt").exists() and not list(tmp_path.glob(".crossing.ckpt*"))


def test_a_field_below_the_correlation_bar_is_refused(synthetic, tmp_path, monkeypatch):
    # a node-scale checkerboard field is reproduced with correlation ~0.90 on this fit; a bar of 0.95 refuses it
    monkeypatch.setitem(stage.FIDELITY, "between_corr_min", 0.95)
    ckpt, umb, _ = synthetic
    fit = stage.load_fit(ckpt, umb)
    idx = fit.geo["idx"].astype(int)
    cols = np.arange(idx[0], idx[6])
    U = np.zeros((fit.geo["Z"], fit.geo["Wc"]))
    U[:, cols] = 0.3 * (-1.0) ** (cols[None] + np.arange(fit.geo["Z"])[:, None])
    with pytest.raises(stage.FidelityError, match="correlation between nodes") as err:
        stage.write(fit, U, tmp_path / "checker.ckpt", band=(0.0, 96.0), windings=(1, 6))
    assert "error ratio" not in str(err.value)
    assert not (tmp_path / "checker.ckpt").exists()


def test_a_band_shorter_than_the_taper_cannot_be_checked_and_is_refused(synthetic, tmp_path, monkeypatch):
    monkeypatch.setitem(stage.SETTINGS, "taper", 150.0)
    ckpt, umb, _ = synthetic
    fit = stage.load_fit(ckpt, umb)
    with pytest.raises(stage.FidelityError, match="shorter than twice"):
        stage.write(fit, _planted_field(fit), tmp_path / "short.ckpt", band=(0.0, 96.0), windings=(1, 6))
    assert not (tmp_path / "short.ckpt").exists()


def test_the_command_exits_non_zero_when_the_check_fails(synthetic, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(stage.FIDELITY, "between_ratio_max", 1e-9)  # an unreachable bar
    ckpt, _umb, ct = synthetic
    out = tmp_path / "refused.ckpt"
    assert cli.main([str(ckpt), str(out), "--ct", str(ct)]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not out.exists()
    report = json.loads(Path(str(out) + ".json").read_text())
    assert "no checkpoint written" in report["refused"]


# ---------------------------------------------------------------------------
# End to end on the synthetic CT
# ---------------------------------------------------------------------------

def test_end_to_end_on_the_synthetic_ct(synthetic, base):
    ckpt, umb, _ = synthetic
    d, rep = base
    fit = stage.load_fit(ckpt, umb)
    th = _node_angles(fit.geo)
    idx = fit.geo["idx"].astype(int)
    field = np.load(d / "base.ckpt.field.npy")
    zrow = fit.geo["zmin"] + np.arange(fit.geo["Z"]) * (fit.geo["zmax"] - fit.geo["zmin"]) / (fit.geo["Z"] - 1)
    # the planted ridge offset is recovered where the z taper is 1
    f = field[(zrow >= 24) & (zrow <= 72)][:, idx[0]:idx[6]]
    d_true = np.broadcast_to(fixture.planted_offset(th[idx[0]:idx[6]]), f.shape)
    moved = np.abs(f) > 0
    assert moved.mean() > 0.6 and np.sqrt(((f[moved] - d_true[moved]) ** 2).mean()) < 0.02
    assert rep["write"]["fidelity_failures"] == [] and rep["write"]["logits_changed"] > 0
    assert rep["write"]["verify_at_nodes"]["rms_err_w"] < 0.005
    # profiles that leave the CT are NaN and their nodes never move
    assert rep["extraction"]["nan_profiles"] > 0
    stacks = _stacks(d, "base.ckpt")
    r0, r1 = rep["rows"]
    zr = zrow[r0:r1]
    outside = (zr < -8) | (zr > 104)
    assert outside.any() and np.isnan(stacks[outside]).all()
    assert np.isfinite(stacks[(zr > 16) & (zr < 80)]).all()
    assert (field[r0:r1][outside] == 0).all()


def test_slab_split_equals_larger_slabs(synthetic, base, tmp_path):
    d, ref = base
    split = _run(tmp_path, synthetic, name="split.ckpt", slab_rows=1)
    assert ref["slabs"] > 1 and split["slabs"] == len(range(*split["rows"]))
    assert np.array_equal(_stacks(d, "base.ckpt"), _stacks(tmp_path, "split.ckpt"), equal_nan=True)
    assert ref["write"]["output_model_state_sha256"] == split["write"]["output_model_state_sha256"]


def test_chunks_missed_by_the_plan_are_fetched_on_demand(synthetic, base, tmp_path, monkeypatch):
    d, ref = base
    assert ref["extraction"]["on_demand_chunks"] == 0
    monkeypatch.setattr(ZarrChunkStore, "keys_near", lambda self, points, margin: np.zeros(0, np.int64))
    lazy = _run(tmp_path, synthetic, name="lazy.ckpt")
    assert lazy["extraction"]["on_demand_chunks"] > 0
    assert np.array_equal(_stacks(d, "base.ckpt"), _stacks(tmp_path, "lazy.ckpt"), equal_nan=True)
    assert np.array_equal(_logits(d / "base.ckpt"), _logits(tmp_path / "lazy.ckpt"))


def test_an_interrupted_run_resumes_to_the_same_result(synthetic, base, tmp_path):
    d, _ = base
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, synthetic, name="resumed.ckpt", slab_rows=2, stop_after_slabs=2)
    work = tmp_path / "resumed.ckpt.work"
    plan = json.loads((work / "plan.json").read_text())
    assert not (tmp_path / "resumed.ckpt").exists()
    # a slab that died after writing its data but before its marker, and a stray temporary file
    a, b = plan["slabs"][2]
    np.save(work / f"slab_{a:05d}_{b:05d}.npy", np.zeros((1, 1, 41), np.float32))
    (work / ".slab_x.npy.tmp-1").write_bytes(b"partial")
    resumed = _run(tmp_path, synthetic, name="resumed.ckpt", slab_rows=2)
    assert resumed["extraction"]["slabs_resumed"] == 2
    assert np.array_equal(_stacks(d, "base.ckpt"), _stacks(tmp_path, "resumed.ckpt"), equal_nan=True)
    assert np.array_equal(_logits(d / "base.ckpt"), _logits(tmp_path / "resumed.ckpt"))


def test_unfinished_work_of_a_different_run_is_refused(synthetic, tmp_path):
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, synthetic, "--windings", "1", "6", name="x.ckpt", stop_after_slabs=1)
    with pytest.raises(RuntimeError, match="different run"):
        _run(tmp_path, synthetic, "--windings", "2", "6", name="x.ckpt")


def test_refusals(synthetic, base, tmp_path):
    ckpt, umb, ct = synthetic
    (tmp_path / "exists.ckpt").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        _run(tmp_path, synthetic, name="exists.ckpt")
    wrong = tmp_path / "wrong.zarr"
    shutil.copytree(ct, wrong)
    meta = json.loads((wrong / ".zarray").read_text())
    meta["shape"] = [96, 224, 200]
    (wrong / ".zarray").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="base_shape_zyx"):
        cli.run(cli._parse_args([str(ckpt), str(tmp_path / "w.ckpt"), "--ct", str(wrong)]), log=lambda *a: None)
    with pytest.raises(ValueError, match="already corrected"):
        cli.run(cli._parse_args([str(base[0] / "base.ckpt"), str(tmp_path / "twice.ckpt"), "--ct", str(ct),
                                 "--umbilicus", str(umb)]), log=lambda *a: None)


def test_the_memory_plan_picks_a_slab_size_that_fits(synthetic, tmp_path):
    ckpt, umb, ct_path = synthetic
    fit = stage.load_fit(ckpt, umb)
    ct = ZarrChunkStore(ct_path)
    peak = {r: stage.plan_extraction(fit, ct, (0.0, 96.0), (1, 6), r, 1).memory["peak"] / 2**30 for r in (1, 2, 8)}
    assert peak[1] < peak[2] < peak[8]
    e = stage.estimate(fit, ct, (0.0, 96.0), windings=(1, 6), work_dir=tmp_path / "w", slab_rows=8,
                       max_memory_gb=(peak[1] + peak[2]) / 2, log=lambda *a: None)
    assert {b - a for a, b in e["plan"].slabs} == {1}
    with pytest.raises(MemoryError, match="even with 1-row slabs"):
        stage.estimate(fit, ct, (0.0, 96.0), windings=(1, 6), work_dir=tmp_path / "w2", slab_rows=8,
                       max_memory_gb=peak[1] / 2, log=lambda *a: None)


def test_parallel_planning_gives_the_serial_plan(synthetic):
    ckpt, umb, ct_path = synthetic
    fit = stage.load_fit(ckpt, umb)
    serial = stage.plan_extraction(fit, ZarrChunkStore(ct_path), (0.0, 96.0), (1, 6), 3, 1)
    parallel = stage.plan_extraction(fit, ZarrChunkStore(ct_path), (0.0, 96.0), (1, 6), 3, 3)
    assert serial.slabs == parallel.slabs and serial.fingerprint == parallel.fingerprint
    assert all(np.array_equal(a, b) for a, b in zip(serial.keys, parallel.keys, strict=True))
    ct = ZarrChunkStore(ct_path)
    stage._install(fit, ct)
    for (ra, rb), keys in zip(serial.slabs, serial.keys, strict=True):  # union of rows == probing the slab's rows
        assert np.array_equal(ct.keys_near(stage._probe_points(fit, ra, rb, 1, 6), stage.PROBE_MARGIN), keys)


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="worker processes use fork")
def test_worker_processes_give_the_single_process_result(synthetic, base, tmp_path):
    d, _ = base
    _run(tmp_path, synthetic, "--threads", "2", name="t2.ckpt")
    assert np.array_equal(_stacks(d, "base.ckpt"), _stacks(tmp_path, "t2.ckpt"), equal_nan=True)
    assert np.array_equal(_logits(d / "base.ckpt"), _logits(tmp_path / "t2.ckpt"))


def test_field_scale(synthetic, base, tmp_path):
    ckpt, _umb, _ = synthetic
    d, full = base
    _run(tmp_path, synthetic, "--field-scale", "1", name="s1b.ckpt")
    half = _run(tmp_path, synthetic, "--field-scale", "0.5", name="s05.ckpt")
    zero = _run(tmp_path, synthetic, "--field-scale", "0", name="s0.ckpt")
    assert np.array_equal(_logits(d / "base.ckpt").view(np.uint32), _logits(tmp_path / "s1b.ckpt").view(np.uint32))
    f1 = np.load(d / "base.ckpt.field.npy").astype(np.float64)
    f05 = np.load(str(tmp_path / "s05.ckpt") + ".field.npy").astype(np.float64)
    assert np.array_equal(f05, (0.5 * f1.astype(np.float32).astype(np.float64)).astype(np.float32).astype(np.float64))
    assert half["write"]["fidelity_failures"] == []
    ratio = half["write"]["fidelity"]["rms_intended"] / full["write"]["fidelity"]["rms_intended"]
    assert abs(ratio - 0.5) < 1e-3
    assert zero["write"]["logits_changed"] == 0
    assert torch.load(tmp_path / "s0.ckpt", weights_only=False)["model_state_sha256"] == \
        torch.load(ckpt, weights_only=False)["model_state_sha256"]


def test_default_windings_skip_unset_config_values(synthetic):
    fit = stage.load_fit(synthetic[0], synthetic[1])
    fit.checkpoint = dict(fit.checkpoint)
    fit.config = dict(fit.config, first_winding=None, output_first_winding=3, shell_outer_winding_idx=5)
    fit.checkpoint.pop("preview_first_winding", None)
    assert stage.default_windings(fit) == (3, 5)
    fit.checkpoint["preview_first_winding"] = 2
    assert stage.default_windings(fit) == (2, 5)


def test_progress_lines_reach_a_redirected_log_at_once(tmp_path):
    # os._exit skips the final flush: the line is in the file only if it was flushed when written
    log = tmp_path / "log.txt"
    with open(log, "w") as fh:
        code = subprocess.call([sys.executable, "-c", ("import os; from ridge_correction import stage; "
                                                       "stage.log_line('[ridge] slab rows 1-2'); os._exit(0)")],
                               stdout=fh, cwd=Path(stage.__file__).parent.parent)
    assert code == 0 and "[ridge] slab rows 1-2" in log.read_text()
