#!/usr/bin/env python
"""
QA for a scrollcase kit written by scrollcase/scripts/stl_generator.py.

Checks the kit is complete (the three STLs and a row in scroll_summary.csv) and, per STL:
finite geometry, winding consistency, boundary and non-manifold edges, and components, with
zero-volume slivers (left where the halves meet the divider) counted separately. Then the fit:
the angle between the cavity's wide direction and the split plane, and the clearance from the
scroll surface to each half; no scroll surface may lie inside case material. The wiggle is the gap
from each scroll surface sample to the nearer half (near the lining offset everywhere for a snug fit).
Each half must lift off the scroll along its side of the y axis without hitting it, stay within the
caps' footprint, and not overlap the other; away from the flat divider joint the halves must keep
--sliding-mm apart (where one half protrudes past the divider into the other, their faces slide
past each other).

--render writes cross-sections at five heights (scroll red, left half blue, right half green)
above shaded views of the halves and the scroll.

Run with the scrollcase venv (needs scrollcase, trimesh and meshlib):
    python scrollcase_qa.py OUTPUT_DIR SCROLL_ID [--render kit.png]
"""

import argparse
import csv
import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw

# Import scrollcase (build123d/OCP) before meshlib: the reverse order aborts at import.
from scrollcase.curved_divider_wall import divider_y
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


def winding_number(mesh, x):
    """Generalized winding number of point x in a triangle mesh (1 inside a closed surface, 0 outside)."""
    a, b, c = (mesh.triangles - x).transpose(1, 0, 2)
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    num = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", b, c) * la
           + np.einsum("ij,ij->i", c, a) * lb)
    return 2 * np.arctan2(num, den).sum() / (4 * np.pi)


def signed_distances(P, mesh_path, mesh, max_confirm=500):
    """Signed distances from points to a case half, negative samples confirmed by winding number (see clearance).
    Returns the distances and the number of samples meshlib alone reported inside."""
    mm_mesh = mrmeshpy.loadMesh(mesh_path)
    d = np.array([mrmeshpy.findSignedDistance(mrmeshpy.Vector3f(*map(float, p)), mm_mesh).dist for p in P])
    flagged = np.flatnonzero(d <= 0)
    for i in flagged[:max_confirm]:
        if winding_number(mesh, P[i]) < 0.5:
            d[i] = abs(d[i])
    return d, len(flagged)


def clearance(scroll, half, half_path, n=30000):
    """Signed distances from scroll surface samples to a case half (negative = inside its material).

    meshlib's sign comes from the pseudo-normal at the closest feature and is occasionally wrong at a case vertex
    (1482, 1592, 0608: samples 2-16 mm from any material reported inside). Every negative sample is therefore
    checked with the exact winding number and made positive when it is outside. Returns the distances and the
    number of samples meshlib alone reported inside."""
    P, _ = trimesh.sample.sample_surface(scroll, n, seed=0)
    return signed_distances(P, half_path, half)


def pull_blocked(scroll, half_path, direction, n=30000):
    """Scroll surface samples from which a ray along `direction` (the way the scroll leaves this half) hits it."""
    P, _ = trimesh.sample.sample_surface(scroll, n, seed=1)
    part = mrmeshpy.MeshPart(mrmeshpy.loadMesh(half_path))
    d = mrmeshpy.Vector3f(*map(float, direction))
    return sum(bool(mrmeshpy.rayMeshIntersect(part, mrmeshpy.Line3f(mrmeshpy.Vector3f(*map(float, p)), d)))
               for p in P)


def half_contact(half, other, other_path, lining_outer_radius, z_range, sliding_mm, n=60000):
    """Samples of one half's surface inside the other half, and (beside the scroll, away from the flat divider
    joint) within sliding_mm of it, i.e. faces that must slide past each other with less than the required
    clearance. Outside z_range the caps slide by design (right_cap_buffer above the base disc). Flush contact
    along the joint gives distances near 0, so only samples deeper than 0.05 mm are confirmed by winding number."""
    P, _ = trimesh.sample.sample_surface(half, n, seed=2)
    mm_mesh = mrmeshpy.loadMesh(other_path)
    s = np.array([mrmeshpy.findSignedDistance(mrmeshpy.Vector3f(*map(float, p)), mm_mesh).dist for p in P])
    for i in np.flatnonzero(s < -0.05)[:500]:
        if winding_number(other, P[i]) < 0.5:
            s[i] = abs(s[i])
    off_joint = ((np.abs(P[:, 1] - divider_y(P[:, 0], lining_outer_radius, 2.0)) > 1.5)
                 & (P[:, 2] > z_range[0]) & (P[:, 2] < z_range[1]))
    return (s < -0.2).sum(), (off_joint & (s > -0.2) & (s < sliding_mm)).sum(), s[off_joint].min()


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
    parser.add_argument("--sliding-mm", type=float, default=1.0,
                        help="minimum clearance between the halves away from the divider joint (default 1)")
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
    sd = {}
    for k in ("left", "right"):
        s, flagged = sd[k] = clearance(parts["scroll"], parts[k], paths[k])
        out = s[s > 0]
        inside = int((s <= 0).sum())
        note = (f" ({flagged} reported by the signed distance, {flagged - inside} of them outside by winding number)"
                if flagged > inside else "")
        print(f"{k:6s} clearance min {out.min():.2f}, 1st percentile {np.percentile(out, 1):.2f}, median "
              f"{np.median(out):.2f} mm; scroll samples inside case material {(s <= 0).sum()}/{len(s)}{note}")
    gap = np.minimum(sd["left"][0], sd["right"][0])
    print(f"wiggle (gap to the nearer half) median {np.median(gap):.2f}, 95th percentile {np.percentile(gap, 95):.2f}, "
          f"99th {np.percentile(gap, 99):.2f}, max {gap.max():.2f} mm; samples over 3 mm {(gap > 3).mean():.1%}")

    # The right half lies on the +y side of the divider and lifts off along +y, the left along -y.
    up = 1.0 if parts["right"].centroid[1] > parts["left"].centroid[1] else -1.0
    for k, pull in (("right", up), ("left", -up)):
        blocked = pull_blocked(parts["scroll"], paths[k], (0, -pull, 0))
        print(f"{k:6s} lifts off along {'+' if pull > 0 else '-'}y: scroll samples blocked by it {blocked}/30000")
    radius = float(rows[0]["Diameter (mm)"]) / 2 if rows else None
    if radius:
        # The lining (offset 2 + wall 2 by default) around the scroll; the caps start beyond it.
        z_range = parts["scroll"].bounds[:, 2] + [-4, 4]
        # Square caps of half-width max(mount disc radius 56.25, lining radius) bound each half in y.
        cap = max(56.25, radius)
        for k in ("left", "right"):
            beyond = (np.abs(parts[k].vertices[:, 1]) > cap + 0.01).sum()
            print(f"{k:6s} vertices beyond the caps' footprint (|y| > {cap:.2f}) {beyond}")
        for k, other in (("left", "right"), ("right", "left")):
            inside, close, nearest = half_contact(parts[k], parts[other], paths[other], radius, z_range,
                                                  args.sliding_mm)
            print(f"{k:6s} surface samples inside {other} {inside}/60000; off the divider joint and within "
                  f"{args.sliding_mm} mm of {other} {close}/60000 (nearest {nearest:.2f} mm)")
    if args.render:
        render(parts, args.render)


if __name__ == "__main__":
    main()
