"""M2 decimation tests on the analytic cylinder fixture.

The cylinder PLY is written with the tests' own raw-struct writer (NOT scrollkit.io),
so the decimation path is exercised end-to-end from independent bytes.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fixtures import cylinder_wrap, write_binary_ply_wrapstyle

from scrollkit.decimate import metrics
from scrollkit.decimate.core import MeshDecimator
from scrollkit.io import read_obj


@pytest.fixture(scope="module")
def cyl_ply(tmp_path_factory):
    td = tmp_path_factory.mktemp("cyl")
    cyl = cylinder_wrap()  # R=50, 0.75 turns, 64x17 grid -> 2016 faces
    V, F, UV = cyl["vertices"], cyl["faces"], cyl["uv_norm"]
    normals = np.zeros_like(V)
    r = np.linalg.norm(V[:, :2], axis=1, keepdims=True)
    normals[:, :2] = V[:, :2] / r  # exact outward radial normals
    wedge = UV[F]
    ply = td / "cyl.ply"
    write_binary_ply_wrapstyle(ply, V, normals.astype(np.float32), UV, F, wedge, texture_file="tex.png")
    # real (tiny) texture so the pymeshlab load + copy_texture path runs cleanly
    img = np.zeros((8, 8, 4), np.uint8)
    img[..., 0] = 200
    img[..., 3] = 255
    Image.fromarray(img).save(td / "tex.png")
    return {"ply": ply, "cyl": cyl, "wedge": wedge}


def test_cylinder_decimation_keep_030(cyl_ply, tmp_path):
    cyl = cyl_ply["cyl"]
    radius = 50.0
    n_faces_in = cyl["faces"].shape[0]

    dec = MeshDecimator(cyl_ply["ply"])  # edge_len_mean computed from arrays
    att = dec.try_rung(0.30, 1.0)

    # ~30% of faces kept (VCG hits the target closely; allow 25-35%)
    keep = att["faces_out"] / n_faces_in
    assert 0.25 <= keep <= 0.35, f"keep_achieved {keep:.3f} not ~0.30"

    W = dec.last["W"]
    # UV preserved: every output wedge UV within the source UV bbox
    src_uv = cyl_ply["wedge"].reshape(-1, 2)
    lo, hi = src_uv.min(axis=0), src_uv.max(axis=0)
    out_uv = W.reshape(-1, 2)
    assert (out_uv >= lo - 1e-6).all() and (out_uv <= hi + 1e-6).all(), (
        f"output UV outside source bbox: [{out_uv.min(0)}, {out_uv.max(0)}] vs [{lo}, {hi}]"
    )
    # no flips: source orientation is uniformly positive in UV space
    areas = metrics.uv_signed_areas(W)
    assert (areas > 0).all(), f"{(areas <= 0).sum()} non-positive UV triangles after decimation"

    # Hausdorff small vs cylinder radius (two-sided, sampled by pymeshlab)
    hd = att["hausdorff"]["two_sided_max"]
    assert hd < 0.05 * radius, f"two-sided Hausdorff {hd:.3f} not small vs radius {radius}"

    # geometry gates as wired for production all hold on the analytic cylinder
    assert att["pass"], f"gates failed: {att['fail_reasons']}"

    # written OBJ round-trips through the strict reader with identical arrays
    out = dec.write_outputs(tmp_path, "cyl", "A")
    re = read_obj(tmp_path / "cyl_decimated.obj")
    assert re.n_faces == att["faces_out"]
    assert np.array_equal(re.vertices.view(np.uint32), dec.last["V"].view(np.uint32))
    re_wedge = re.vertex_uv[re.faces] if re.vertex_uv is not None else re.wedge_uv
    assert np.array_equal(re_wedge.view(np.uint32), W.view(np.uint32))
    assert out["uv_obj_mode"] in ("shared", "wedge")
    assert (tmp_path / "tex.png").exists() and out["texture_sha256"]


def test_wedge_to_vertex_uv_exact_detects_seams():
    # 2 faces sharing vertices 0,2 with identical UVs everywhere -> exact conversion
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)
    wedge = uv[faces]
    got = metrics.wedge_to_vertex_uv_exact(4, faces, wedge)
    assert got is not None and np.array_equal(got.view(np.uint32), uv.view(np.uint32))
    # introduce a seam: vertex 2 carries two distinct UVs -> must refuse
    wedge2 = wedge.copy()
    wedge2[1, 1] = [0.5, 0.5]
    assert metrics.wedge_to_vertex_uv_exact(4, faces, wedge2) is None


def test_uv_chart_count_splits():
    # two triangles sharing a mesh edge but with disagreeing UVs across it = 2 charts
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)
    assert metrics.uv_chart_count(faces, uv[faces]) == 1
    wedge = uv[faces].copy()
    wedge[1] += 2.0  # move face 1's chart elsewhere in UV space
    assert metrics.uv_chart_count(faces, wedge) == 2
