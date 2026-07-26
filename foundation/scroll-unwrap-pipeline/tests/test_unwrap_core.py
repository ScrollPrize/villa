"""M3 core math on the analytic cylinder: endpoints exact, transit area-preserving,
multi-turn branch unwrapping correct."""

import numpy as np
import pytest

from fixtures import cylinder_wrap, spiral_wrap
from scrollkit.unwrap.dg_interp import (
    branch_consistent_logs,
    build_unwrap_path,
    deformation_gradients,
    face_uv_tangents,
    polar_decompose,
    rotation_exp,
    rotation_log,
    winding_decomposition,
)
from scrollkit.unwrap.embedding import build_target_embedding
from scrollkit.unwrap.operators import face_areas
from scrollkit.unwrap.timeline import timeline_t


def test_rotation_log_exp_roundtrip():
    rng = np.random.default_rng(7)
    w = rng.normal(size=(500, 3))
    w *= (rng.uniform(0.01, np.pi - 0.01, size=500) / np.linalg.norm(w, axis=1))[:, None]
    R = rotation_exp(w)
    w2 = rotation_log(R)
    assert np.allclose(w, w2, atol=1e-9)
    # near-pi
    wpi = rng.normal(size=(100, 3))
    wpi *= ((np.pi - 1e-7) / np.linalg.norm(wpi, axis=1))[:, None]
    R = rotation_exp(wpi)
    w2 = rotation_log(R)
    assert np.allclose(rotation_exp(w2), R, atol=1e-6)


def test_polar_handles_reflection():
    Fm = np.tile(np.eye(3), (2, 1, 1))
    Fm[1, 2, 2] = -1.0  # reflection
    R, S, inv = polar_decompose(Fm)
    assert not inv[0] and inv[1]
    assert np.allclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ S, Fm, atol=1e-12)


@pytest.mark.parametrize("turns", [0.75, 2.5])
def test_branch_unwrapping_multi_turn(turns):
    c = cylinder_wrap(n_u=int(96 * max(1.0, turns)), n_v=9, turns=turns)
    emb = build_target_embedding(c["vertices"], c["faces"], c["uv_norm"])
    Fg = deformation_gradients(c["vertices"].astype(np.float64), emb["V1"], c["faces"])
    R, S, inv = polar_decompose(Fg)
    assert inv.sum() == 0
    w, info = branch_consistent_logs(R, c["faces"], seed_face=0)
    assert info["n_unreached"] == 0
    # total unwrapped rotation must reach ~ the full winding angle of the roll
    expected = turns * 2 * np.pi
    assert info["max_unwrapped_angle_rad"] > 0.75 * expected, info
    # and naive (wrapped) logs must NOT exceed pi — proving the unwrap did something real
    assert np.linalg.norm(rotation_log(R), axis=1).max() <= np.pi + 1e-6


def test_cylinder_unroll_endpoints_and_area():
    c = cylinder_wrap(n_u=128, n_v=13, turns=1.5)
    V0 = c["vertices"].astype(np.float64)
    F = c["faces"]
    emb = build_target_embedding(V0, F, c["uv_norm"])
    assert emb["mirror_check"] > 0, "embedding must not mirror the texture"
    # de-norm scales must recover the true UV extents (anisotropy = extent ratio, NOT 1)
    su_true, sv_true = c["scale_true"]
    assert abs(emb["denorm_fit"]["s_u"] - su_true) / su_true < 0.01
    assert abs(emb["denorm_fit"]["s_v"] - sv_true) / sv_true < 0.01
    assert emb["denorm_fit"]["rel_residual_rms"] < 0.005
    assert emb["axis"]["axis_uv"] == "v", "cylinder axis runs along v"
    assert emb["anchor_rms"] < 0.5, "anchor strip must fit near-exactly"

    path = build_unwrap_path(V0, emb["V1"], F, c["uv_norm"].astype(np.float64),
                             emb["anchor_idx"], solver_backend="scipy-splu")
    assert path.info["n_unreached"] == 0

    A0 = face_areas(V0, F)
    A1 = face_areas(emb["V1"], F)

    # endpoints (exact short-circuits)
    assert np.allclose(path.eval_frame(0.0), V0)
    assert np.allclose(path.eval_frame(1.0), emb["V1"])
    # ...and through the actual solve: reconstruction at t≈1 must reproduce the target
    V_near1 = path.eval_frame(1.0 - 1e-9)
    span = (V0.max(0) - V0.min(0)).max()
    assert np.abs(V_near1 - emb["V1"]).max() < 1e-4 * span, (
        f"Poisson solve at t≈1 deviates from target by {np.abs(V_near1 - emb['V1']).max():.4g}"
    )

    # transit: areas within tight bounds of the endpoint blend (isometric roll ⇒ ~constant)
    for t in (0.25, 0.5, 0.75):
        Vt = path.eval_frame(t)
        assert np.isfinite(Vt).all()
        At = face_areas(Vt, F)
        blend = (1 - t) * A0 + t * A1
        rel = np.abs(At - blend) / blend
        assert np.quantile(rel, 0.95) < 0.02, f"t={t}: P95 area deviation {np.quantile(rel, 0.95):.4f}"
        assert rel.max() < 0.05, f"t={t}: max area deviation {rel.max():.4f}"

    # solved midpoint must differ hugely from linear vertex blend (the morph actually unrolls)
    lerp_mid = 0.5 * (V0 + emb["V1"])
    Vmid = path.eval_frame(0.5)
    assert np.abs(Vmid - lerp_mid).max() > 0.05 * (V0.max(0) - V0.min(0)).max()


@pytest.mark.parametrize("inward", [True, False])
def test_embedding_natural_orientation(inward):
    """The flat target must sit in the surface's NATURAL frame at the anchor strip.

    An inward-parameterized spiral makes the anchor tangent point toward the roll
    centroid at both u-extremes; the legacy 'extend away from the body' heuristic then
    rotated the target 180° in-plane, which (a) reversed the anchor strip along the
    scroll axis and (b) injected a constant π rotation about an in-plane axis into the
    source→target map — unrepresentable by the axis-winding model, so ‖w_res‖≈π on most
    faces and the transit tumbled (observed on several damaged merge traces). The natural
    placement must hold for BOTH trace directions, and the residual must stay bump-scale.
    """
    c = spiral_wrap(inward=inward)
    V0, F, uvn = c["vertices"], c["faces"], c["uv_norm"]
    emb = build_target_embedding(V0, F, uvn)
    assert emb["mirror_check"] > 0
    if inward:
        # the legacy frame disagrees with the natural one here — correction must engage
        assert emb["orientation_corrected"], "legacy body-side frame should differ on inward trace"
    # anchor strip placed un-reversed: rms ≪ strip extent (~height)
    assert emb["anchor_rms"] < 2.0, emb["anchor_rms"]

    # source v-tangents must agree with the embedded sheet's v direction (no global flip)
    t_u, t_v = face_uv_tangents(V0, F, uvn.astype(np.float64))
    uvc = uvn - uvn.mean(0)
    V1c = emb["V1"] - emb["V1"].mean(0)
    v_dir = V1c.T @ uvc[:, 1].astype(np.float64)
    v_hat = v_dir / np.linalg.norm(v_dir)
    cos = (t_v / np.linalg.norm(t_v, axis=1, keepdims=True)) @ v_hat
    assert (cos > 0).mean() > 0.999, "embedded v-axis anti-parallel to source v-tangents"

    # winding decomposition: residual must be bump-scale, no π-residual population
    Fg = deformation_gradients(V0.astype(np.float64), emb["V1"], F)
    R, S, inv = polar_decompose(Fg)
    assert inv.sum() == 0
    anchor_c = V0[emb["anchor_idx"]].mean(0)
    cent = V0[F].mean(1)
    seed = int(np.argmin(((cent - anchor_c) ** 2).sum(1)))
    d = winding_decomposition(R, V0.astype(np.float64), F, uvn.astype(np.float64),
                              emb["V1"], seed_face=seed, axis_mode="global")
    assert d["n_big_residual"] == 0, d
    assert d["residual_norm_p99"] < 0.3, d
    assert d["phi_span_turns"] > 0.75 * 4.0, d


def test_phi_unwrap_branch_immune_to_junk_moat():
    """A junk moat with garbage raw φ must not shift the far side onto a wrong 2π branch.

    Regression guard: low-confidence anchor-strip regions can land
    ±1..3 full turns off, so they secretly orbited the axis mid-transit and the anchor
    Kabsch tumbled the whole sheet. Junk faces inherit the branch from confident
    ancestry; their own (untrusted) raw values only contribute mod 2π.
    """
    from scrollkit.unwrap.dg_interp import _scalar_graph_unwrap

    c = cylinder_wrap(n_u=160, n_v=7, turns=1.5)
    F = c["faces"]
    uvc = c["uv_norm"][F].mean(1)
    # true smooth field: 1.5 turns across u
    phi_true = uvc[:, 0] * 1.5 * 2 * np.pi
    raw = (phi_true + np.pi) % (2 * np.pi) - np.pi
    # junk moat in the middle of u with GARBAGE raw values
    moat = (uvc[:, 0] > 0.45) & (uvc[:, 0] < 0.55)
    rng = np.random.default_rng(3)
    raw_garbage = raw.copy()
    raw_garbage[moat] = rng.uniform(-np.pi, np.pi, moat.sum())
    conf = ~moat

    out, n_unreached = _scalar_graph_unwrap(raw_garbage, F, seed_face=0, conf=conf)
    assert n_unreached == 0
    err = out[conf] - phi_true[conf]
    err -= err[0]  # global offset irrelevant
    assert np.abs(err).max() < 1e-6, (
        f"confident faces landed on a wrong branch: max err {np.abs(err).max():.3f} rad"
    )
    # junk faces: garbage mod-2π values are expected (raw is noise), but the BRANCH must
    # follow the confident neighborhood — no junk face may orbit a full turn away
    err_j = out[moat] - phi_true[moat]
    assert np.abs(err_j).max() < 2 * np.pi, (
        f"junk face on a wrong branch: max err {np.abs(err_j).max():.3f} rad"
    )


def test_timeline_shape():
    t = timeline_t(240, 20, 25)
    assert len(t) == 240
    assert (t[:20] == 0).all() and (t[-25:] == 1).all()
    d = np.diff(t)
    assert (d >= -1e-12).all()
    assert d.max() < 0.02  # smooth, no jumps
