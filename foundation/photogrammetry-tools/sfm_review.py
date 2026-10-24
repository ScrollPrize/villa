#!/usr/bin/env python
"""
Review a PGS-Recon sparse reconstruction per acquisition block.

Reads from the PGS-Recon SfM output directory (the one containing mvg/):
  - mvg/recon_dir/SfMReconstruction_Report.html: posed views and reprojection residuals per block;
  - mvg/matches_dir/geometric_matches and mvg/sfm_data.json: geometrically verified image pairs
    between blocks of different orientations. OpenMVG reconstructs only the largest connected
    group of views, so an orientation without verified links to the others is dropped;
  - mvg/recon_dir/review_sfm_data.json, if present: camera trajectories. Each block should be a
    circle (one camera height, turntable rotation), and blocks of one orientation should share
    its axis; a broken circle suggests the object moved. Export it with
        openMVG_main_ConvertSfM_DataFormat -i sfm_data.bin -o review_sfm_data.json -V -I -E

Blocks are the subdirectories of BLOCKS, named <orientation>_camera_<height> as in
photogrammetry-docs/INSTRUCTIONS.md; image stems are taken from the RAW/JPEG files in each
(or its JPGEnhanced/).

Usage:
    python sfm_review.py SFM_DIR BLOCKS
"""

import argparse
import collections
import html
import json
import os
import re
from pathlib import Path

import numpy as np

IMAGE_EXT = (".arw", ".cr2", ".nef", ".dng", ".jpg", ".jpeg", ".tif", ".tiff", ".png")


def block_stems(root):
    blocks = {}
    for b in sorted(os.listdir(root)):
        d = os.path.join(root, b)
        if not os.path.isdir(d):
            continue
        stems = {os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(IMAGE_EXT)}
        if not stems and os.path.isdir(os.path.join(d, "JPGEnhanced")):
            stems = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(d, "JPGEnhanced"))}
        if stems:
            blocks[b] = sorted(stems)
    return blocks


def orientation(block):
    return block.split("_camera")[0]


def report_residuals(path):
    """Per posed view: (observations, median residual, mean residual) from the OpenMVG HTML report."""
    text = re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", " ", Path(path).read_text())))
    rows = re.finditer(r"\d+ (\S+?) (\d+) [\d.e+-]+ ([\d.e+-]+) ([\d.e+-]+) [\d.e+-]+", text)
    return {m.group(1): (int(m.group(2)), float(m.group(3)), float(m.group(4))) for m in rows}


def verified_pairs(mvg_dir):
    views = json.loads(Path(os.path.join(mvg_dir, "sfm_data.json")).read_text())["views"]
    name = {v["key"]: os.path.splitext(v["value"]["ptr_wrapper"]["data"]["filename"])[0] for v in views}
    text = Path(os.path.join(mvg_dir, "matches_dir", "geometric_matches")).read_text()
    return [(name[int(a)], name[int(b)]) for a, b in re.findall(r"n(\d+) --\s+n(\d+)", text)]


def fit_circle(P):
    """Plane + circle fit; returns centre, normal, radius and residuals relative to the radius."""
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    Q = (P - c) @ Vt[:2].T
    sol = np.linalg.lstsq(np.c_[2 * Q, np.ones(len(Q))], (Q ** 2).sum(1), rcond=None)[0]
    r = np.sqrt(sol[2] + (sol[:2] ** 2).sum())
    ang = np.sort(np.degrees(np.arctan2(Q[:, 1] - sol[1], Q[:, 0] - sol[0])) % 360)
    return dict(center=c + sol[:2] @ Vt[:2], normal=Vt[2], r=r,
                radial=(np.linalg.norm(Q - sol[:2], axis=1) - r).std() / r,
                planar=((P - c) @ Vt[2]).std() / r,
                max_gap=np.diff(np.r_[ang, ang[0] + 360]).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("sfm_dir", help="PGS-Recon SfM output directory (contains mvg/)")
    parser.add_argument("blocks", help="directory with one subdirectory per acquisition block")
    args = parser.parse_args()
    mvg = os.path.join(args.sfm_dir, "mvg")
    recon = os.path.join(mvg, "recon_dir")
    blocks = block_stems(args.blocks)
    block_of = {s: b for b, stems in blocks.items() for s in stems}

    res = report_residuals(os.path.join(recon, "SfMReconstruction_Report.html"))
    print("== Posed views and reprojection residuals (px) per block")
    total = 0
    for b, stems in blocks.items():
        r = np.array([res[s] for s in stems if s in res]).reshape(-1, 3)
        total += len(r)
        print(f"{b:30s} posed {len(r):4d}/{len(stems):<4d}" + (
            f" observations/view median {np.median(r[:, 0]):.0f}, residual median {np.median(r[:, 1]):.3f},"
            f" mean {r[:, 2].mean():.3f}" if len(r) else ""))
    print(f"total posed {total}/{len(block_of)}")

    print("\n== Geometrically verified image pairs between orientations")
    links = collections.Counter()
    for a, b in verified_pairs(mvg):
        if a in block_of and b in block_of and orientation(block_of[a]) != orientation(block_of[b]):
            links[tuple(sorted((block_of[a], block_of[b])))] += 1
    orientations = sorted({orientation(b) for b in blocks})
    for i, o1 in enumerate(orientations):
        for o2 in orientations[i + 1:]:
            pairs = {k: n for k, n in links.items() if {orientation(k[0]), orientation(k[1])} == {o1, o2}}
            detail = ", ".join(f"{a} / {b}: {n}" for (a, b), n in sorted(pairs.items(), key=lambda x: -x[1]))
            print(f"{o1} - {o2}: {sum(pairs.values())} pairs{' (' + detail + ')' if detail else ''}")

    review = os.path.join(recon, "review_sfm_data.json")
    if not os.path.exists(review):
        print(f"\n(no {review}; export it to review camera trajectories)")
        return
    d = json.loads(Path(review).read_text())
    pose_of = {os.path.splitext(v["value"]["ptr_wrapper"]["data"]["filename"])[0]: v["value"]["ptr_wrapper"]["data"]["id_pose"]
               for v in d["views"]}
    centres = {e["key"]: np.array(e["value"]["center"]) for e in d["extrinsics"]}
    print("\n== Camera trajectories (radius in SfM units; rms relative to radius)")
    fits = {}
    for b, stems in blocks.items():
        P = np.array([centres[pose_of[s]] for s in stems if s in pose_of and pose_of[s] in centres])
        if len(P) < 5:
            print(f"{b:30s} too few posed views")
            continue
        f = fits[b] = fit_circle(P)
        print(f"{b:30s} radius {f['r']:.3f}  radial rms {100 * f['radial']:.2f}%  out-of-plane rms "
              f"{100 * f['planar']:.2f}%  largest angular gap {f['max_gap']:.1f} deg")
    print("\n== Rotation axes")
    for o in orientations:
        grp = [b for b in fits if orientation(b) == o]
        for b in grp[1:]:
            ref, f = fits[grp[0]], fits[b]
            dd = f["center"] - ref["center"]
            off = np.linalg.norm(dd - (dd @ ref["normal"]) * ref["normal"]) / ref["r"]
            print(f"{grp[0]} vs {b}: axes {np.degrees(np.arccos(min(1, abs(ref['normal'] @ f['normal'])))):.2f} deg "
                  f"apart, centre {100 * off:.2f}% of radius off the shared axis")
    firsts = [next(b for b in fits if orientation(b) == o) for o in orientations if any(orientation(b) == o for b in fits)]
    for i, a in enumerate(firsts):
        for b in firsts[i + 1:]:
            ang = np.degrees(np.arccos(min(1, abs(fits[a]["normal"] @ fits[b]["normal"]))))
            print(f"{orientation(a)} vs {orientation(b)}: object axes {ang:.1f} deg apart")


if __name__ == "__main__":
    main()
