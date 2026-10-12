#!/usr/bin/env python
"""
QA for a scrollcase kit written by scrollcase/scripts/stl_generator.py.

Checks the kit is complete (the three STLs and a row in scroll_summary.csv) and, per STL:
finite geometry, winding consistency, boundary and non-manifold edges, and components, with
zero-volume slivers (left where the halves meet the divider) counted separately. Then the fit:
the angle between the cavity's wide direction and the split plane, and the clearance from the
scroll surface to each half; no scroll surface may lie inside case material.

--render writes cross-sections at five heights (scroll red, left half blue, right half green)
above shaded views of the halves and the scroll.

Run with the scrollcase venv (needs trimesh and meshlib):
    python scrollcase_qa.py OUTPUT_DIR SCROLL_ID [--render kit.png]
"""

import argparse
import csv
import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from meshlib import mrmeshpy

PARTS = ("scroll", "left", "right")
COLOURS = {"scroll": (200, 30, 30), "left": (40, 80, 200), "right": (30, 150, 60)}
PX = 520


def topology(m):
    vols = np.array([abs(c.volume) for c in m.split(only_watertight=False)])
    _, cnt = np.unique(np.sort(m.edges, axis=1), axis=0, return_counts=True)
    return (f"faces {len(m.faces)}, finite {np.isfinite(m.vertices).all()}, winding consistent {m.is_winding_consistent}, "
            f"boundary edges {(cnt == 1).sum()}, non-manifold edges {(cnt > 2).sum()}, components {len(vols)} "
            f"(zero-volume slivers {(vols < 1e-3).sum()}), volume {m.volume / 1000:.1f} cm3, extents {np.round(m.extents, 2).tolist()}")


def clearance(scroll, half_path, n=30000):
    """Signed distances from scroll surface samples to a case half (negative = inside its material)."""
    P, _ = trimesh.sample.sample_surface(scroll, n, seed=0)
    mesh = mrmeshpy.loadMesh(half_path)
    return np.array([mrmeshpy.findSignedDistance(mrmeshpy.Vector3f(*map(float, p)), mesh).dist for p in P])


def section_tile(parts, z, lo, hi):
    img = Image.new("RGB", (PX, PX), "white")
    draw = ImageDraw.Draw(img)
    half = 0.52 * max(hi[0] - lo[0], hi[1] - lo[1])
    c = (lo + hi) / 2
    to_px = lambda p: ((p[0] - c[0]) / half * PX / 2 + PX / 2, PX / 2 - (p[1] - c[1]) / half * PX / 2)
    for k in ("left", "right", "scroll"):
        lines = trimesh.intersections.mesh_plane(parts[k], [0, 0, 1], [0, 0, z])
        for a, b in lines:
            draw.line([to_px(a), to_px(b)], fill=COLOURS[k], width=2)
    draw.text((8, 8), f"z = {z:.1f} mm", fill="black")
    return img


def view_tile(meshes, R, n=1500000):
    light = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
    pts, cols = [], []
    for k, m in meshes:
        P, fi = trimesh.sample.sample_surface(m, n, seed=0)
        shade = 0.25 + 0.75 * np.clip((m.face_normals[fi] @ R.T) @ light, 0, 1)
        pts.append(P @ R.T)
        cols.append((0.3 * np.array(COLOURS[k]) + 0.7 * 255 * shade[:, None]) * shade[:, None] ** 0.2)
    Q, C = np.concatenate(pts), np.concatenate(cols).clip(0, 255).astype(np.uint8)
    ctr, half = (Q.max(0) + Q.min(0)) / 2, 0.55 * (Q.max(0) - Q.min(0))[:2].max()
    u = ((Q[:, 0] - ctr[0]) / half * PX / 2 + PX / 2).astype(int)
    v = (PX / 2 - (Q[:, 1] - ctr[1]) / half * PX / 2).astype(int)
    ok = (u >= 0) & (u < PX) & (v >= 0) & (v < PX)
    order = np.argsort(Q[ok, 2])
    img = np.full((PX, PX, 3), 255, np.uint8)
    img[v[ok][order], u[ok][order]] = C[ok][order]
    return Image.fromarray(img)


def look(z_axis, up=(0, 0, 1)):
    z = np.asarray(z_axis, float) / np.linalg.norm(z_axis)
    y = np.asarray(up, float) - (np.dot(up, z)) * z
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z])


def render(parts, out):
    lo = np.min([m.bounds[0] for m in parts.values()], 0)
    hi = np.max([m.bounds[1] for m in parts.values()], 0)
    sb = parts["scroll"].bounds
    sections = [section_tile(parts, z, lo, hi) for z in np.linspace(sb[0, 2] + 3, sb[1, 2] - 3, 5)]
    split = (parts["right"].centroid - parts["left"].centroid) * [1, 1, 0]
    side = np.cross(split, [0, 0, 1])
    views = [view_tile([(k, parts[k]) for k in ("left", "right")], look(side)),
             view_tile([(k, parts[k]) for k in ("left", "scroll")], look(split)),
             view_tile([(k, parts[k]) for k in ("right", "scroll")], look(-split)),
             view_tile([("scroll", parts["scroll"])], look(split)),
             view_tile([("scroll", parts["scroll"])], look(side))]
    sheet = Image.new("RGB", (PX * 5, PX * 2), "white")
    for i, (a, b) in enumerate(zip(sections, views)):
        sheet.paste(a, (i * PX, 0))
        sheet.paste(b, (i * PX, PX))
    sheet.save(out)
    print("wrote", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("output_dir", help="stl_generator.py --output directory (holds scroll_summary.csv)")
    parser.add_argument("scroll_id", help="padded scroll identifier, e.g. 0525")
    parser.add_argument("--render", help="write sections and views to this PNG")
    args = parser.parse_args()
    kit = os.path.join(args.output_dir, args.scroll_id)
    paths = {k: os.path.join(kit, f"{args.scroll_id}_{k}.stl") for k in PARTS}
    missing = [p for p in paths.values() if not os.path.exists(p) or os.path.getsize(p) == 0]
    if missing:
        raise SystemExit(f"missing or empty STLs: {missing}")
    with open(os.path.join(args.output_dir, "scroll_summary.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["Scroll ID"] == args.scroll_id]
    print(f"scroll_summary.csv: {rows[0] if rows else 'NO ROW for ' + args.scroll_id}")

    parts = {k: trimesh.load(p, process=True) for k, p in paths.items()}
    for k, m in parts.items():
        print(f"{k:6s} {topology(m)}")
    xy = parts["scroll"].vertices[:, :2]
    w, E = np.linalg.eigh(np.cov(xy - xy.mean(0), rowvar=False))
    print(f"cavity wide direction vs split plane (x): {np.degrees(np.arccos(abs(E[0, -1]))):.1f} deg")
    for k in ("left", "right"):
        s = clearance(parts["scroll"], paths[k])
        out = s[s > 0]
        print(f"{k:6s} clearance min {out.min():.2f}, 1st percentile {np.percentile(out, 1):.2f}, median "
              f"{np.median(out):.2f} mm; scroll samples inside case material {(s <= 0).sum()}/{len(s)}")
    if args.render:
        render(parts, args.render)


if __name__ == "__main__":
    main()
