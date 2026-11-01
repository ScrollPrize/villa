#!/usr/bin/env python
"""
Topology and size QA for OpenMVS PLY meshes, with optional canonical renders.

For each PLY: finite vertices, valid and non-degenerate faces, boundary and non-manifold edges
(both must be 0 for case generation), face-connected components, and extents along the
principal axes (world axes are arbitrary, so compare these with the object's measured size).
Reads the binary PLY indices directly, so texture seams do not split the mesh.

--render writes shaded views of the first mesh along +-each principal axis, plus a textured
row if --texture names the TextureMesh PLY (same geometry, texture image from its header).
--undistorted checks that OpenMVS's undistorted_images/ has one <stem>.mask.png per image;
densification silently ignores a mask that fails to load.

Usage:
    python mesh_qa.py MESH.ply [MORE.ply ...] [--render views.png [--texture TEXTURED.ply]]
                      [--undistorted <run>/mvs/undistorted_images]
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

TYPES = {"float32": "f4", "float": "f4", "float64": "f8", "double": "f8", "uint8": "u1", "uchar": "u1",
         "int8": "i1", "char": "i1", "int16": "i2", "short": "i2", "uint16": "u2", "ushort": "u2",
         "int32": "i4", "int": "i4", "uint32": "u4", "uint": "u4"}


def read_ply(path):
    """Binary little-endian PLY -> (vertices (n,3) float64, faces (m,3) int64, texcoords (m,3,2) or None, header)."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode(errors="replace")
    if "binary_little_endian" not in head:
        raise SystemExit(f"{path}: only binary_little_endian PLY is supported")
    elems, cur = [], None
    for line in head.splitlines():
        p = line.split()
        if p[:1] == ["element"]:
            cur = [p[1], int(p[2]), []]
            elems.append(cur)
        elif p[:1] == ["property"]:
            cur[2].append(p[1:])
    off, V, F, T = end, None, None, None
    for name, n, props in elems:
        fields = []
        for q in props:
            if q[0] == "list":  # fixed-length lists: take each list's length from the first record
                count = raw[off + sum(np.dtype(f[1]).itemsize * int(np.prod(f[2] if len(f) > 2 else 1)) for f in fields)]
                fields += [(q[-1] + "_n", "<" + TYPES[q[1]]), (q[-1], "<" + TYPES[q[2]], (count,))]
            else:
                fields.append((q[-1], "<" + TYPES[q[0]]))
        dt = np.dtype(fields)
        rec = np.frombuffer(raw, dt, n, off)
        off += n * dt.itemsize
        if name == "vertex":
            V = np.c_[rec["x"], rec["y"], rec["z"]].astype(np.float64)
        elif name == "face":
            key = "vertex_indices" if "vertex_indices" in dt.names else "vertex_index"
            if rec[key].shape[1] != 3 or (rec[key + "_n"] != 3).any():
                raise SystemExit(f"{path}: non-triangle faces")
            F = rec[key].astype(np.int64)
            if "texcoord" in dt.names:
                T = rec["texcoord"].reshape(n, 3, 2).astype(np.float64)
    return V, F, T, head


def face_components(n_vertices, F):
    """Component label per vertex (vectorised union-find over face edges)."""
    parent = np.arange(n_vertices)
    e = np.r_[F[:, [0, 1]], F[:, [1, 2]]]
    while True:
        a, b = parent[e[:, 0]], parent[e[:, 1]]
        if (a == b).all():
            return parent
        np.minimum.at(parent, np.maximum(a, b), np.minimum(a, b))
        while True:
            nxt = parent[parent]
            if (nxt == parent).all():
                break
            parent = nxt


def mesh_report(V, F):
    deg = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    E = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)
    _, cnt = np.unique(E[:, 0] * len(V) + E[:, 1], return_counts=True)
    flab = face_components(len(V), F)[F[:, 0]]
    labels, sizes = np.unique(flab, return_counts=True)
    main = flab == labels[np.argmax(sizes)]
    P = V[np.unique(F[main])]
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    tri = V[F[main]]
    return dict(vertices=len(V), faces=len(F), finite=bool(np.isfinite(V).all()),
                indices_valid=bool(F.min() >= 0 and F.max() < len(V)), degenerate=int(deg.sum()),
                zero_area=int((area < 1e-12).sum()), boundary_edges=int((cnt == 1).sum()),
                nonmanifold_edges=int((cnt > 2).sum()), components=len(labels),
                main_share=sizes.max() / len(F), unreferenced=len(V) - len(np.unique(F)),
                extents=np.ptp((P - c) @ Vt.T, axis=0),
                volume=np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6,
                area=area[main].sum(), edge_median=np.median(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)))


def sample_surface(V, F, n, rng):
    tri = V[F]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = np.linalg.norm(cross, axis=1)
    fi = rng.choice(len(F), n, p=a / a.sum())
    s, r2 = np.sqrt(rng.random(n)), rng.random(n)
    bary = np.c_[1 - s, s * (1 - r2), s * r2]
    return fi, bary, np.einsum("ij,ijk->ik", bary, tri[fi]), cross[fi] / a[fi, None]


def splat(P, colours, R, c, half, px=700):
    Q = (P - c) @ R.T
    u = ((Q[:, 0] / half + 1) * 0.5 * (px - 1)).astype(int)
    v = ((1 - (Q[:, 1] / half + 1) * 0.5) * (px - 1)).astype(int)
    ok = (u >= 0) & (u < px) & (v >= 0) & (v < px)
    order = np.argsort(Q[ok, 2])  # nearest (largest z) drawn last
    img = np.full((px, px, 3), 255, np.uint8)
    img[v[ok][order], u[ok][order]] = colours[ok][order]
    return img


def render(V, F, out, textured=None):
    rng = np.random.default_rng(0)
    c = V.mean(0)
    _, _, R0 = np.linalg.svd(V - c, full_matrices=False)
    half = 1.08 * np.abs((V - c) @ R0.T).max()
    views = [np.stack([np.cross(R0[up], s * R0[ax]), R0[up], s * R0[ax]]) for ax, up in ((2, 1), (0, 1), (1, 2)) for s in (1, -1)]
    light = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
    _, _, P, N = sample_surface(V, F, 3000000, rng)
    rows = []
    tiles = []
    for R in views:
        shade = (40 + 215 * np.clip(np.abs((N @ R.T) @ light), 0, 1)).astype(np.uint8)
        tiles.append(splat(P, np.repeat(shade[:, None], 3, 1), R, c, half))
    rows.append(np.concatenate(tiles, 1))
    if textured:
        Vt, Ft, T, head = read_ply(textured)
        if T is None:
            raise SystemExit(f"{textured}: no texcoord face property")
        tex_name = re.search(r"comment TextureFile (\S+)", head).group(1)
        tex = np.array(Image.open(os.path.join(os.path.dirname(textured), tex_name)).convert("RGB"))
        fi, bary, P, _ = sample_surface(Vt, Ft, 4000000, rng)
        uv = np.einsum("ij,ijk->ik", bary, T[fi])
        H, W = tex.shape[:2]
        col = tex[np.clip(((1 - uv[:, 1]) * H).astype(int), 0, H - 1), np.clip((uv[:, 0] * W).astype(int), 0, W - 1)]
        rows.append(np.concatenate([splat(P, col, R, c, half) for R in views], 1))
    img = np.concatenate(rows, 0)
    Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2), Image.LANCZOS).save(out)
    print("wrote", out)


def check_undistorted(d):
    files = os.listdir(d)
    images = {os.path.splitext(f)[0] for f in files if not f.endswith(".mask.png") and f.lower().endswith((".jpg", ".png"))}
    masks = {f[:-len(".mask.png")] for f in files if f.endswith(".mask.png")}
    missing = sorted(images - masks)
    print(f"{d}: {len(images)} images, {len(masks)} masks; images without a mask: {len(missing)}"
          + (f" (e.g. {', '.join(missing[:5])})" if missing else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("meshes", nargs="+", help="PLY meshes to check")
    parser.add_argument("--render", help="write canonical renders of the first mesh to this PNG")
    parser.add_argument("--texture", help="TextureMesh PLY of the same geometry, for a textured render row")
    parser.add_argument("--undistorted", help="OpenMVS undistorted_images/ directory to check for masks")
    args = parser.parse_args()
    if args.undistorted:
        check_undistorted(args.undistorted)
    for path in args.meshes:
        V, F, _, _ = read_ply(path)
        r = mesh_report(V, F)
        print(f"{path}\n  vertices {r['vertices']}, faces {r['faces']}; finite {r['finite']}, indices valid "
              f"{r['indices_valid']}, degenerate {r['degenerate']} (zero-area {r['zero_area']}), unreferenced vertices "
              f"{r['unreferenced']}\n  boundary edges {r['boundary_edges']}, non-manifold edges {r['nonmanifold_edges']}, "
              f"components {r['components']} (largest {r['main_share']:.2%} of faces)\n"
              f"  principal-axis extents {np.round(r['extents'], 2).tolist()}, volume {r['volume']:.1f}, area "
              f"{r['area']:.1f}, median edge {r['edge_median']:.3f} (mesh units, mm after scaling)")
    if args.render:
        V, F, _, _ = read_ply(args.meshes[0])
        render(V, F, args.render, args.texture)


if __name__ == "__main__":
    main()
