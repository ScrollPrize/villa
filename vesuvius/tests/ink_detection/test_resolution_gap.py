from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from vesuvius.ink_detection.eval import resolution_gap as rg
from vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input import prepare_isotropic_input

ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


def _texture(shape, lam_lo, lam_hi, seed):
    """Band-limited unit-variance noise: wavelengths between lam_lo and lam_hi px."""
    rng = np.random.default_rng(seed)
    f = np.fft.rfft2(rng.standard_normal(shape))
    k = np.sqrt(np.fft.fftfreq(shape[0])[:, None] ** 2 + np.fft.rfftfreq(shape[1])[None, :] ** 2)
    t = np.fft.irfft2(f * ((k >= 1.0 / lam_hi) & (k <= 1.0 / lam_lo)), s=shape)
    return (t / t.std()).astype(np.float32)


def _fine_ZYX(n_layers, shape, seed=0, ink=None):
    """A synthetic fine surface volume: textures that change with depth, a large-scale term and noise."""
    rng = np.random.default_rng(seed)
    terms = [_texture(shape, 10 + 6 * k, 60 + 30 * k, seed + 1 + k) for k in range(4)]
    side = min(shape)
    low = _texture(shape, min(150, side / 2), min(600, side), seed + 9)
    out = np.zeros((n_layers,) + tuple(shape), np.uint8)
    for z in range(n_layers):
        d = z - (n_layers - 1) / 2.0
        f = sum(np.float32(np.exp(-0.5 * ((d - c) / 7.0) ** 2)) * t for c, t in zip((-18, -6, 6, 18), terms))
        f = f + np.float32(1.5 * np.cos(d / 20.0)) * low
        if ink is not None:
            f = f + np.float32(1.6 * np.exp(-0.5 * (d / 9.0) ** 2)) * ink
        v = 110 + 28 * f + rng.normal(0, 5, shape).astype(np.float32)
        out[z] = np.clip(np.rint(v), 1, 255).astype(np.uint8)
    return out


def _halve_in_plane(a_ZYX):
    z, h, w = a_ZYX.shape
    return np.rint(a_ZYX.reshape(z, h // 2, 2, w // 2, 2).astype(np.float32).mean(axis=(2, 4))).astype(np.uint8)


def _write_group(path: Path, levels: dict, attrs: dict) -> None:
    kwargs = {"mode": "w"}
    if ZARR_V3:
        kwargs["zarr_format"] = 2
    group = zarr.open_group(path, **kwargs)
    group.attrs.update(attrs)
    for name, data in levels.items():
        chunks = tuple(min(64, s) for s in data.shape)
        if ZARR_V3:
            group.create_array(name, data=data, chunks=chunks)
        else:
            group.create_dataset(name, data=data, chunks=chunks)


def _multiscales(um):
    return {
        "multiscales": [
            {"datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [um, um, um]}]}]}
        ]
    }


# ---------------------------------------------------------------- geometry and registration


def test_site_geometry_centers_the_layers_and_checks_the_coarse_range():
    site = rg.site_geometry(109, 28, 2.399, 9.362)
    assert site["fine_layers"] == [24, 84]
    assert site["fine_c"] == 54.0 and site["real_c"] == 13.5
    assert site["check"]["verdict"] == "PASS"
    m = rg.real_layer_map(site)
    assert m.shape == (60,) and m.min() >= 0 and m.max() <= 27
    assert rg.site_geometry(50, 28, 2.399, 9.362)["check"]["verdict"] == "FAIL"
    assert rg.site_geometry(109, 12, 2.399, 9.362)["check"]["verdict"] == "FAIL"


def test_whole_voxel_registration_recovers_a_known_shift():
    site = rg.site_geometry(109, 28, 2.399, 9.362)
    scale = (0.25625, 0.25625)
    n, origin = 504, (1008, 1512)
    fine = np.repeat((120 + 30 * _texture((n, n), 40, 200, 3))[None], 60, axis=0)
    fine = np.clip(np.rint(fine), 1, 255).astype(np.uint8)
    box = rg.real_box(origin[0], origin[1], n, scale)
    grid, _ = rg.to_real_grid(fine.mean(axis=0), origin, box, scale)
    shifted = np.roll(np.nan_to_num(grid), (3, -2), axis=(0, 1))
    real = np.repeat(shifted[None].astype(np.float32), 28, axis=0)
    got = rg.register_real(site, fine, origin, real, box, scale)
    assert got["shift"] == [3, -2]
    assert got["r"] > 0.99


# ---------------------------------------------------------------- blur fits and gray levels


def _coarse_list(seed=0):
    vols = [_fine_ZYX(60, (128, 128), seed + 10 * i) for i in range(2)]
    return [rg.coarse4(v) for v in vols]


def test_blur_fit_recovers_the_blur_of_a_blurred_copy():
    cl = _coarse_list()
    target = rg.blur_ratio(cl, 0.8)
    assert target < rg.blur_ratio(cl, 0.0)
    sigma, ratio, trace = rg.match_blur(cl, target, hi=1.5)
    assert abs(sigma - 0.8) < 0.01
    assert abs(ratio / target - 1) < 0.01
    assert trace[0][0] == 0.0 and len(trace) == 14
    assert rg.match_blur(cl, 1.0)[0] == 0.0
    with pytest.raises(ValueError, match="still above the target"):
        rg.match_blur(cl, 1e-6, hi=0.5)


def test_gray_level_map_moves_the_histograms_onto_the_target():
    src = _fine_ZYX(60, (64, 64), 1)
    tgt = np.clip(np.rint(0.7 * src.astype(np.float32) + 40), 1, 255).astype(np.uint8)
    hs, ht = rg.gray_level_hist(src, tgt)
    maps = rg.gray_level_fit(hs, ht)
    assert maps.shape == (15, 256) and maps.dtype == np.uint8
    mapped = rg.gray_level_apply(src, maps)
    before = rg.w1_from_hists(hs, ht)[0]
    after = rg.w1_from_hists(*rg.gray_level_hist(mapped, tgt))[0]
    assert after < 0.1 * before
    assert rg.gray_level_check({"windows_w1_before": before, "windows_w1_after": after})["verdict"] == "PASS"


def test_directional_fit_recovers_depth_and_in_plane_blurs():
    vols = [_fine_ZYX(60, (96, 96), 20 + i) for i in range(2)]
    cl = [rg.coarse4(v) for v in vols]
    real = [rg.blurred_box_average_axes(v, 0.5, 0.7) for v in vols]
    ht = None
    items = []
    for (c, rend), rv in zip(cl, real):
        ht = rg.target_hist(rg.blur_coarse(c, rend, 0.0), rv, ht)
        items.append((c, rend, rg.cell_weights(rv)))
    rxy, rz, _, _ = rg.directional_ratios(items, ht, 0.5, 0.7)
    sz, sxy, got_xy, got_z, trace, maps, _ = rg.match_directional(items, ht, rxy, rz, hi_xy=1.5, hi_z=1.0, iters=10)
    assert abs(sz - 0.5) < 0.05 and abs(sxy - 0.7) < 0.05
    assert abs(got_xy / rxy - 1) < 0.02 and abs(got_z / rz - 1) < 0.02
    assert maps.shape == (15, 256)


# ---------------------------------------------------------------- villa's recipe


def test_villa_planes_match_prepare_9um_isotropic_input(tmp_path):
    level2 = _fine_ZYX(109, (32, 48), 5)
    source = tmp_path / "fine.zarr"
    _write_group(source, {"2": level2}, {})
    out = prepare_isotropic_input(str(source), tmp_path / "prepared.zarr", level="2", workers=2)
    prepared = zarr.open_group(out, mode="r")["0"][:]
    pooled, (z0, z1) = rg.villa_pool(level2)
    assert (z0, z1) == (13, 97)
    np.testing.assert_array_equal(pooled, prepared)

    grid = rg.villa_grid(109, 24)
    assert grid["offset"] == 3 and grid["phase_fine_layers"] == 3
    assert grid["villa_input_layers"] == [25, 85] and grid["villa_minus_box_layers"] == 1
    planes, _ = rg.villa_planes(level2, 24)
    np.testing.assert_array_equal(planes, prepared[3:18])
    fine_grid = rg.villa_input(level2, 24)
    assert fine_grid.shape == (60, 128, 192)
    np.testing.assert_array_equal(fine_grid[::4, ::4, ::4], planes)


def test_alignment_check_passes_on_the_pyramid_and_fails_off_grid():
    full = _fine_ZYX(109, (64, 64), 7)
    level2 = _halve_in_plane(_halve_in_plane(full))
    box = rg.box4_at(full, 24, rg.GROUPS)
    on_box = rg.villa_planes_on_box_layers(level2, 24)
    assert rg.alignment_check(on_box, box)["verdict"] == "PASS"
    shifted = rg.villa_planes_on_box_layers(np.roll(level2, 1, axis=2), 24)
    assert rg.alignment_check(shifted, box)["verdict"] == "FAIL"


# ---------------------------------------------------------------- windows and tiles


def test_window_rule_orders_by_ink_share_and_keeps_windows_apart():
    sup3 = np.full((378, 378), 255, np.uint8)
    lab3 = np.zeros_like(sup3)
    shares = {(0, 0): 0.30, (0, 126): 0.45, (126, 0): 0.10, (126, 126): 0.25, (252, 252): 0.31}
    for (r, c), s in shares.items():
        lab3[r : r + 126, c : c + int(round(126 * s))] = 255
    sup3[252:, :126] = 0
    wins, n = rg.pick_windows(lab3, sup3)
    assert [(w["row3"], w["col3"]) for w in wins] == [(0, 0), (252, 252), (126, 126), (0, 126)]
    assert n == 4 and all(w["size"] == 1008 and w["y0"] == 8 * w["row3"] for w in wins)
    tiles = rg.pick_tiles(sup3)
    assert len(tiles) == 32 and all(t["size"] == 504 for t in tiles)


# ---------------------------------------------------------------- scoring


def test_exact_auc_and_bootstrap_draws_agree_with_brute_force():
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.3
    s = rng.normal(size=300) + y
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]])
    assert rg.auc_exact(s, y) == pytest.approx(brute, abs=1e-12)
    w = np.ones((2, 300))
    np.testing.assert_allclose(rg.weighted_auc_draws(s, y, w), [brute, brute], atol=1e-12)
    assert rg.t_quantile(0.975, 5) == pytest.approx(2.570582, abs=1e-5)
    assert rg.t_quantile(0.975, 1) == pytest.approx(12.706205, abs=1e-4)


STRENGTH = {
    "fine": 2.0,
    "box_average": 1.8,
    "box_blur": 1.0,
    "box_blur_x1.25": 0.85,
    "box_blur_x1.5": 0.7,
    "box_blur_x1.75": 0.55,
    "box_blur_x2": 0.45,
    "box_blur_x2.5": 0.3,
    "box_blur_matched": 1.0,
    "box_blur_directional": 1.0,
    "real": 1.1,
    "real_registered": 1.15,
    "villa_input": 1.7,
}


def _stages():
    site = rg.site_geometry(109, 28, 2.399, 9.362)
    sm = {"ratio": 0.05}
    return {
        "site": site,
        "folds": [0, 1],
        "registration": {
            "median_shift": [1, 1],
            "dz": 0,
            "dz_r": 0.99,
            "windows": [{"fold": j, "r": 0.98, "shift": [1, 1]} for j in (0, 1)],
        },
        "registration_subvoxel": {
            "dz_sub": 0.0,
            "dz_sub_r": 0.99,
            "windows": [{"fold": j, "check": {"pass": True}} for j in (0, 1)],
            "check": {"verdict": "PASS"},
        },
        "blur": {"sigma": 1.0, "refused": None, "windows_real": sm, "windows_box_blur": sm},
        "matched_blur": {
            "sigma": 1.0,
            "refused": None,
            "shortcut_equal": True,
            "windows_w1_before": 5.0,
            "windows_w1_after": 0.5,
            "windows_real": sm,
            "windows_box_blur_matched": sm,
        },
        "directional_blur": {
            "sz": 1.1,
            "sxy": 0.9,
            "refused": None,
            "shortcut_equal": True,
            "registration_subvoxel": {"verdict": "PASS"},
            "windows_w1_before": 5.0,
            "windows_w1_after": 0.5,
            "windows_real": sm,
            "windows_real_depth": sm,
            "windows_box_blur_directional": sm,
            "windows_box_blur_directional_depth": sm,
        },
        "villa": {"grid": rg.villa_grid(109, 24), "windows": [{"alignment": {"verdict": "PASS"}} for _ in (0, 1)]},
        "inputs": {a: {"code": c, "window_sha256": ["w0", "w1"], "norm": [0.4, 0.1]} for a, c in rg.INPUTS},
    }


def _maps(seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    kept = rng.random((2, 84, 84)) < 0.9
    ink = kept & (rng.random((2, 84, 84)) < 0.3)
    maps = {}
    for a, s in STRENGTH.items():
        for sd in rg.SEEDS:
            fwd = scale * s * ink + rng.normal(size=ink.shape)
            maps[(a, sd)] = {
                "kept": kept,
                "ink": ink,
                "forward": fwd.astype(np.float32),
                "reversed": rng.normal(size=ink.shape).astype(np.float32),
            }
    return maps


def _records():
    return [
        {
            "input": a,
            "code": c,
            "seed": sd,
            "smoke": False,
            "steps": rg.STEPS,
            "norm": [0.4, 0.1],
            "windows": [{"sha256_input": "w0"}, {"sha256_input": "w1"}],
        }
        for a, c in rg.INPUTS
        for sd in rg.SEEDS
    ]


def test_score_reads_clear_differences_from_synthetic_maps():
    maps = _maps()
    donor = _maps(seed=1)[("fine", 1)]
    res = rg.score(maps, (donor["kept"], donor["ink"]), _stages(), _records(), "synthetic", draws=200)
    assert set(res["inputs"]) == set(STRENGTH)
    row = res["inputs"]["box_average"]
    assert row["reversed_depth"]["verdict"] == "PASS" and row["donor_null"]["verdict"] == "PASS"
    assert res["controls"]["input_check"]["box_average"]["verdict"] == "PASS"
    assert res["controls"]["villa"]["verdict"] == "PASS"
    assert res["differences"]["blur_cost"]["value"] > 0
    assert res["verdict"]["blur_cost"] == "CLEAR"
    assert res["verdict"]["dose"] == "CLEAR" and res["dose"]["slope"]["value"] < 0
    assert res["verdict"]["closeness"] == "BLUR CLOSER"
    assert res["plan"]["train"] == [a for a, _ in rg.INPUTS]

    weak = rg.score(_maps(scale=0.05), (donor["kept"], donor["ink"]), _stages(), _records(), "synthetic", draws=50)
    assert weak["controls"]["fine_floor"]["verdict"] == "FAIL"
    assert set(weak["verdict"].values()) == {"VOID"}


def test_input_check_fails_on_a_changed_input():
    recs = _records()
    recs[0]["windows"][1]["sha256_input"] = "other"
    out = rg.input_check_verdict(_stages(), recs)
    assert out[recs[0]["input"]]["verdict"] == "FAIL"
    assert all(v["verdict"] == "PASS" for k, v in out.items() if k != recs[0]["input"])


# ---------------------------------------------------------------- the command line, end to end (smoke layout)


def _segment(tmp_path: Path):
    """A synthetic segment: a 96-layer fine volume (levels 0 and 2), a real coarse volume of the same papyrus
    (shifted, blurrier), ink labels and a supervision mask, with one 504 px window and three training tiles."""
    H, W = 1008, 768
    ink = _texture((H, W), 40, 160, 99) > 0.55
    fine = _fine_ZYX(96, (H, W), 11, ink)
    um_f, um_r = 2.399, 9.362
    s = um_f / um_r
    rh, rw = int(round(H * s)), int(round(W * s))
    yy = (np.arange(rh)[:, None] - 1.0) / s
    xx = (np.arange(rw)[None, :] + 1.0) / s
    yi = np.clip(np.rint(yy), 0, H - 1).astype(int)
    xi = np.clip(np.rint(xx), 0, W - 1).astype(int)
    real = np.zeros((24, rh, rw), np.uint8)
    for z in range(24):
        fz = int(round((96 - 1) / 2.0 + (z - (24 - 1) / 2.0) / s))
        lo, hi = max(0, fz - 2), min(96, fz + 2)
        real[z] = np.rint(fine[lo:hi].astype(np.float32).mean(axis=0))[yi, xi] if hi > lo else 0
    level2 = _halve_in_plane(_halve_in_plane(fine))
    paths = {k: tmp_path / (k + ".zarr") for k in ("fine", "coarse", "labels", "mask")}
    _write_group(paths["fine"], {"0": fine, "2": level2}, {"canvas_size": [W, H], **_multiscales(um_f)})
    _write_group(paths["coarse"], {"0": real}, {"canvas_size": [rw, rh], **_multiscales(um_r)})
    _write_group(paths["labels"], {"0": np.where(ink, 255, 0).astype(np.uint8)}, {})
    _write_group(paths["mask"], {"0": np.full((H, W), 255, np.uint8)}, {})
    windows = tmp_path / "windows.json"
    tiles = tmp_path / "tiles.json"
    windows.write_text(json.dumps({"windows": [{"y0": 0, "x0": 0, "size": 504}]}))
    tiles.write_text(json.dumps({"tiles": [{"y0": 756, "x0": x0, "size": 252} for x0 in (0, 252, 504)]}))
    return {k: str(v) for k, v in paths.items()}, str(windows), str(tiles)


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """The label-free stages of a synthetic segment on the smoke layout, through the command line."""
    tmp = tmp_path_factory.mktemp("resolution_gap")
    paths, windows, tiles = _segment(tmp)
    vol_args = ["--fine", paths["fine"], "--coarse", paths["coarse"], "--windows", windows, "--tiles", tiles]
    out = tmp / "run"
    assert rg.main(["stages", *vol_args, "--out", str(out), "--name", "syn", "--smoke"]) == 0
    return {
        "paths": paths,
        "windows": windows,
        "tiles": tiles,
        "vol_args": vol_args,
        "out": out,
        "stages": out / "syn-stages-smoke.json",
    }


def test_stages_record_the_registration_fits_and_every_planned_input(smoke_run):
    rec = json.loads(smoke_run["stages"].read_text())
    assert rec["site"]["check"]["verdict"] == "PASS"
    assert rec["registration"]["median_shift"] is not None
    assert rec["villa"]["grid"] == rg.villa_grid(96, 18)
    assert {"fine", "box_average", "real", "villa_input"} <= set(rec["plan"]["train"])
    assert set(rec["inputs"]) == set(rec["plan"]["train"])
    for name in rec["plan"]["train"]:
        assert len(rec["inputs"][name]["window_sha256"]) == 1
        assert rec["inputs"][name]["code"] == rg.INPUT_CODE[name]


@pytest.mark.slow
def test_train_and_score_run_end_to_end_on_the_smoke_layout(smoke_run):
    pytest.importorskip("torch")
    paths, stages = smoke_run["paths"], smoke_run["stages"]
    lab_args = ["--labels", paths["labels"], "--mask", paths["mask"], "--stages", str(stages)]
    maps = smoke_run["out"] / "maps"
    names = ("box_average", "villa_input")
    for name in names:
        argv = ["train", *smoke_run["vol_args"], *lab_args, "--input", name, "--seed", "1", "--out", str(maps)]
        assert rg.main([*argv, "--name", "syn", "--smoke"]) == 0
        r = json.loads((maps / f"syn-{name}-s1-smoke.json").read_text())
        assert r["input_check"]["verdict"] == "PASS" and r["steps"] == rg.SMOKE_STEPS
        z = np.load(maps / f"syn-{name}-s1-smoke.npz")
        assert z["forward"].shape == z["reversed"].shape == (1, 42, 42)

    data = rg.Data(paths["fine"], paths["coarse"], paths["labels"], paths["mask"])
    with pytest.raises(SystemExit, match="refused: stage record sha256"):
        rg.run_train(
            data,
            "fine",
            [1],
            smoke_run["windows"],
            smoke_run["tiles"],
            str(stages),
            "0" * 64,
            str(maps),
            "syn",
            smoke=True,
        )

    score = smoke_run["out"] / "score.json"
    npz = sorted(str(p) for p in maps.glob("*.npz"))
    argv = ["score", *npz, "--stages", str(stages), "--donor", npz[0], "--out", str(score), "--seeds", "1"]
    assert rg.main([*argv, "--draws", "20", "--smoke"]) == 0
    res = json.loads(score.read_text())
    assert res["controls"]["maps"]["verdict"] == "PASS"
    assert set(res["verdict"].values()) == {"VOID"}
    assert set(res["inputs"]) == set(names)
