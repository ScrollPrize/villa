"""Bit-exactness of the strict IO paths."""

import numpy as np
import pytest

from fixtures import asymmetric_quad, write_binary_ply_scrollstyle, write_binary_ply_wrapstyle
from scrollkit.io import MeshData, read_obj, read_ply, write_obj


def bits(a):
    return a.view(np.uint32) if a is not None else None


def test_ply_reader_wrapstyle(tmp_path):
    q = asymmetric_quad()
    p = tmp_path / "quad.ply"
    write_binary_ply_wrapstyle(p, q["vertices"], q["normals"], q["vertex_uv"], q["faces"], q["wedge_uv"])
    m = read_ply(p)
    assert np.array_equal(bits(m.vertices), bits(q["vertices"]))
    assert np.array_equal(bits(m.normals), bits(q["normals"]))
    assert np.array_equal(bits(m.vertex_uv), bits(q["vertex_uv"]))
    assert np.array_equal(m.faces, q["faces"])
    assert np.array_equal(bits(m.wedge_uv), bits(q["wedge_uv"]))
    assert m.texture_files == ["tex.png"]
    assert m.wedge_equals_vertex_uv() is True


def test_ply_reader_scrollstyle_texnumber(tmp_path):
    q = asymmetric_quad()
    tn = np.array([0, 0], dtype=np.int32)
    p = tmp_path / "scroll.ply"
    write_binary_ply_scrollstyle(p, q["vertices"], q["faces"], q["wedge_uv"], texnumber=tn,
                                 texture_files=("a.png", "missing.png"))
    m = read_ply(p)
    assert m.normals is None and m.vertex_uv is None
    assert np.array_equal(m.face_texnumber, tn)
    assert m.texture_files == ["a.png", "missing.png"]


@pytest.mark.parametrize("with_normals", [True, False])
def test_obj_roundtrip_shared(tmp_path, with_normals):
    q = asymmetric_quad()
    mesh = MeshData(
        vertices=q["vertices"], faces=q["faces"],
        normals=q["normals"] if with_normals else None,
        vertex_uv=q["vertex_uv"], wedge_uv=q["wedge_uv"],
    )
    p = tmp_path / "quad.obj"
    write_obj(p, mesh, texture_file="tex.png", uv_mode="shared")
    r = read_obj(p)
    assert np.array_equal(bits(r.vertices), bits(mesh.vertices))
    assert np.array_equal(r.faces, mesh.faces)
    assert np.array_equal(bits(r.vertex_uv), bits(mesh.vertex_uv))
    if with_normals:
        assert np.array_equal(bits(r.normals), bits(mesh.normals))
    assert r.texture_files == ["tex.png"]


def test_obj_roundtrip_wedge(tmp_path):
    q = asymmetric_quad()
    # wedge UVs deliberately NOT equal to any per-vertex assignment: corner-unique values
    wuv = np.arange(12, dtype=np.float32).reshape(2, 3, 2) / 16.0 + 0.015625
    mesh = MeshData(vertices=q["vertices"], faces=q["faces"], wedge_uv=wuv)
    p = tmp_path / "wedge.obj"
    write_obj(p, mesh, texture_file="tex.png", uv_mode="wedge")
    r = read_obj(p)
    assert np.array_equal(bits(r.vertices), bits(mesh.vertices))
    assert np.array_equal(r.faces, mesh.faces)
    assert r.wedge_uv is not None
    assert np.array_equal(bits(r.wedge_uv), bits(wuv))


def test_format_round_trips_adversarial_float32(tmp_path):
    # denormals, near-max, ulp neighbours, negative zero, pi-ish values
    vals = np.array(
        [1e-39, -1e-39, 3.4028235e38, -3.4028235e38, 1.0, np.nextafter(np.float32(1.0), np.float32(2.0)),
         -0.0, 0.0, 3.14159265, 2.7182818, 1e-45, 12345.678],
        dtype=np.float32,
    )
    V = np.repeat(vals, 3)[: (len(vals) // 3) * 9].reshape(-1, 3)
    mesh = MeshData(vertices=V.astype(np.float32), faces=np.zeros((1, 3), np.int32))
    p = tmp_path / "adv.obj"
    write_obj(p, mesh)
    r = read_obj(p)
    assert np.array_equal(bits(r.vertices), bits(mesh.vertices)), (
        "%.9g failed to round-trip float32 bit-exactly"
    )
