"""Convention traps: any silent flip / transpose / merge / reorder must fail loudly here."""

import numpy as np

from fixtures import asymmetric_quad, cylinder_wrap, write_binary_ply_wrapstyle
from scrollkit.io import MeshData, dedup_wedge_uv, read_obj, read_ply, write_obj
from scrollkit.unwrap import fit_uv_scale


def test_no_vertex_merge_on_duplicate_positions(tmp_path):
    q = asymmetric_quad()  # v3 duplicates v0's position
    p = tmp_path / "q.ply"
    write_binary_ply_wrapstyle(p, q["vertices"], q["normals"], q["vertex_uv"], q["faces"], q["wedge_uv"])
    m = read_ply(p)
    assert m.n_vertices == 4, "duplicate-position vertices must NOT merge"
    o = tmp_path / "q.obj"
    write_obj(o, m, texture_file="tex.png")
    r = read_obj(o)
    assert r.n_vertices == 4
    assert np.array_equal(r.faces, q["faces"]), "face order/indices must be preserved verbatim"


def test_uv_not_flipped_or_transposed(tmp_path):
    q = asymmetric_quad()
    m = MeshData(vertices=q["vertices"], faces=q["faces"], normals=q["normals"],
                 vertex_uv=q["vertex_uv"], wedge_uv=q["wedge_uv"])
    o = tmp_path / "q.obj"
    write_obj(o, m, texture_file="tex.png")
    r = read_obj(o)
    uv = r.vertex_uv
    # u of vertex1 is 0.9 (extreme), v of vertex2 is 0.93 (extreme): transpose or 1-v flip would move them
    assert np.float32(uv[1, 0]) == np.float32(0.9)
    assert np.float32(uv[2, 1]) == np.float32(0.93)
    assert not np.allclose(uv[:, 0], q["vertex_uv"][:, 1]), "UV transpose detected"
    assert not np.allclose(uv[:, 1], 1.0 - q["vertex_uv"][:, 1]), "V-flip detected"


def test_winding_preserved(tmp_path):
    q = asymmetric_quad()
    m = MeshData(vertices=q["vertices"], faces=q["faces"])
    o = tmp_path / "w.obj"
    write_obj(o, m)
    r = read_obj(o)
    e0 = m.vertices[m.faces[0, 1]] - m.vertices[m.faces[0, 0]]
    e1 = m.vertices[m.faces[0, 2]] - m.vertices[m.faces[0, 0]]
    n_src = np.cross(e0, e1)
    f0 = r.faces[0]
    n_out = np.cross(r.vertices[f0[1]] - r.vertices[f0[0]], r.vertices[f0[2]] - r.vertices[f0[0]])
    assert np.dot(n_src, n_out) > 0, "winding flipped"


def test_wedge_dedup_is_bit_exact():
    a = np.float32(0.5)
    b = np.nextafter(a, np.float32(1.0))  # 1 ulp away — must remain distinct
    nz = np.float32(-0.0)                  # -0.0 vs +0.0: equal as values, different bits
    pz = np.float32(0.0)
    w = np.array([[[a, a], [b, a], [a, b]],
                  [[nz, a], [pz, a], [a, a]]], dtype=np.float32)
    vt, idx = dedup_wedge_uv(w)
    # unique bit-rows: (a,a) [shared by both faces], (b,a), (a,b), (-0,a), (+0,a)
    assert vt.shape[0] == 5, f"bit-exact dedup expected 5 unique rows, got {vt.shape[0]}"
    vt_bits = vt.view(np.uint32)
    neg_zero_row = np.array([np.array(-0.0, np.float32).view(np.uint32),
                             np.array(0.5, np.float32).view(np.uint32)], dtype=np.uint32)
    assert (vt_bits == neg_zero_row).all(axis=1).any(), "-0.0 row must survive dedup distinct from +0.0"
    rec = vt[idx]
    assert np.array_equal(rec.view(np.uint32), w.view(np.uint32))


def test_denorm_scale_fit_recovers_cylinder_scales():
    c = cylinder_wrap()
    fit = fit_uv_scale(c["vertices"], c["faces"], c["uv_norm"])
    su_true, sv_true = c["scale_true"]
    # chord-vs-arc error at 64 segments/0.75 turns is ~0.1%; tolerance 1%
    assert abs(fit["s_u"] - su_true) / su_true < 0.01
    assert abs(fit["s_v"] - sv_true) / sv_true < 0.01
    assert fit["rel_residual_rms"] < 0.01


def test_denorm_scale_fit_anisotropic():
    c = cylinder_wrap()
    uv = c["uv_norm"].copy()
    # simulate a non-aspect-preserving normalization: stretch v by 3x in normalized space
    uv[:, 1] /= 3.0
    fit = fit_uv_scale(c["vertices"], c["faces"], uv)
    su_true, sv_true = c["scale_true"]
    assert abs(fit["s_v"] - 3.0 * sv_true) / (3.0 * sv_true) < 0.01
    assert abs(fit["anisotropy"] - su_true / (3.0 * sv_true)) / (su_true / (3.0 * sv_true)) < 0.02
