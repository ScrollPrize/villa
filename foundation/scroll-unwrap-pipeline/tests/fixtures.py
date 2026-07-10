"""Synthetic fixtures: hand-written binary PLY bytes (independent of scrollkit.io writers,
so reader bugs can't self-confirm) and analytic meshes for unwrap math."""

from __future__ import annotations

import struct

import numpy as np


def asymmetric_quad() -> dict:
    """Two triangles with deliberately asymmetric geometry, normals, and UVs.

    Every value is unique so any silent reorder/merge/flip/transpose changes something
    detectable. Includes a duplicated *position* (v3 == v0 location) that must NOT merge.
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.125, 0.0],
            [1.75, 3.0, 0.5],
            [0.0, 0.0, 0.0],  # duplicate position of v0 — merge trap
        ],
        dtype=np.float32,
    )
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.99],
            [0.0, 0.1, 0.99],
            [0.0, 0.0, -1.0],  # deliberately different from v0's normal
        ],
        dtype=np.float32,
    )
    vertex_uv = np.array(
        [[0.03125, 0.0625], [0.9, 0.11], [0.77, 0.93], [0.13, 0.81]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    wedge_uv = vertex_uv[faces]  # consistent wedge==vertex case
    return dict(vertices=vertices, normals=normals, vertex_uv=vertex_uv, faces=faces, wedge_uv=wedge_uv)


def write_binary_ply_wrapstyle(path, vertices, normals, vertex_uv, faces, wedge_uv, texture_file="tex.png") -> None:
    """Write a Group-A/B-style PLY (vertex: x,y,z,nx,ny,nz,s,t; face: idx list + texcoord list)
    using raw struct packing — intentionally NOT scrollkit.io."""
    n, m = len(vertices), len(faces)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment TextureFile {texture_file}\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in ["x", "y", "z", "nx", "ny", "nz", "s", "t"])
        + f"element face {m}\n"
        "property list uchar int vertex_indices\n"
        "property list uchar float texcoord\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<8f", *vertices[i], *normals[i], *vertex_uv[i]))
        for t in range(m):
            f.write(struct.pack("<B3i", 3, *[int(x) for x in faces[t]]))
            f.write(struct.pack("<B6f", 6, *wedge_uv[t].reshape(-1)))


def write_binary_ply_scrollstyle(path, vertices, faces, wedge_uv, texnumber=None, texture_files=("tex.png",)) -> None:
    """Group-C-style PLY (vertex: x,y,z; face: idx + texcoord [+ texnumber])."""
    n, m = len(vertices), len(faces)
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment VCGLIB generated\n"
        + "".join(f"comment TextureFile {t}\n" for t in texture_files)
        + f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {m}\n"
        "property list uchar int vertex_indices\n"
        "property list uchar float texcoord\n"
        + ("property int texnumber\n" if texnumber is not None else "")
        + "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<3f", *vertices[i]))
        for t in range(m):
            f.write(struct.pack("<B3i", 3, *[int(x) for x in faces[t]]))
            f.write(struct.pack("<B6f", 6, *wedge_uv[t].reshape(-1)))
            if texnumber is not None:
                f.write(struct.pack("<i", int(texnumber[t])))


def spiral_wrap(n_u: int = 512, n_v: int = 13, r_outer: float = 60.0, r_inner: float = 25.0,
                height: float = 80.0, turns: float = 4.0, inward: bool = True) -> dict:
    """Archimedean spiral sheet with a near-isometric flattening (u = arc length).

    `inward=True` parameterizes the trace from the OUTER edge winding INWARD with
    increasing u — like the production meshes whose anchor-strip tangent points toward
    the roll centroid at both u-extremes (tangent·(centroid−strip) = +r·|dr/dθ| > 0
    everywhere for an inward trace). This is the geometry that made the legacy
    'extend away from the body' embedding heuristic rotate the flat target 180° in-plane.
    """
    theta = np.linspace(0.0, turns * 2 * np.pi, n_u)
    r = r_outer + (r_inner - r_outer) * theta / theta[-1]
    if not inward:
        r = r[::-1].copy()
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.concatenate([[0.0], np.cumsum(seg)])  # arc length: isometric u
    v = np.linspace(0.0, height, n_v)
    P = np.stack([np.repeat(x, n_v), np.repeat(y, n_v), np.tile(v, n_u)], axis=1)
    flat = np.stack([np.repeat(s, n_v), np.tile(v, n_u)], axis=1)
    uv_norm = flat / np.array([s[-1], v[-1]])
    faces = []
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            a = i * n_v + j
            b = (i + 1) * n_v + j
            faces.append([a, b, a + 1])
            faces.append([b, b + 1, a + 1])
    return dict(
        vertices=P.astype(np.float64),
        faces=np.array(faces, dtype=np.int32),
        uv_norm=uv_norm.astype(np.float64),
        flat=flat.astype(np.float64),
        scale_true=(float(s[-1]), float(v[-1])),
    )


def cylinder_wrap(n_u: int = 64, n_v: int = 17, radius: float = 50.0, height: float = 80.0, turns: float = 0.75) -> dict:
    """Open cylindrical sheet with its exact isometric flattening.

    3D: p(u,v) = (R cos(θ0 + u/R), R sin(θ0 + u/R), v) for u ∈ [0, turns*2πR], v ∈ [0, H].
    Exact flat coords: (u, v). Normalized UV divides by extents (the texturing convention).
    """
    u = np.linspace(0.0, turns * 2 * np.pi * radius, n_u)
    v = np.linspace(0.0, height, n_v)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    theta = uu / radius
    P = np.stack([radius * np.cos(theta), radius * np.sin(theta), vv], axis=-1).reshape(-1, 3)
    flat = np.stack([uu, vv], axis=-1).reshape(-1, 2)
    uv_norm = flat / np.array([u[-1], v[-1]])
    faces = []
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            a = i * n_v + j
            b = (i + 1) * n_v + j
            faces.append([a, b, a + 1])
            faces.append([b, b + 1, a + 1])
    return dict(
        vertices=P.astype(np.float32),
        faces=np.array(faces, dtype=np.int32),
        uv_norm=uv_norm.astype(np.float32),
        flat=flat.astype(np.float32),
        scale_true=(float(u[-1]), float(v[-1])),
    )
