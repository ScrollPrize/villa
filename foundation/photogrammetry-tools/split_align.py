#!/usr/bin/env python
"""
Align two independently reconstructed, independently scaled orientations of one object and check the result.

`align` registers orientation B's dense cloud onto orientation A's (both OpenMVS scene_dense.ply in mm, from scenes
scaled by their own markers). Because each orientation sees mostly one face plus the rim, the clouds only overlap on
the rim; the registration is a trimmed point-to-plane ICP that only pairs points with compatible normals, started from
the sign/permutation hypotheses of the principal axes. It writes the transform B->A and the pose covariance
implied by the rim geometry, with the predicted uncertainty along the thinnest object axis (the thickness), and the
relative scale of B that would fit the rim best (an independent check of the two marker scales). --scale-a/--scale-b
correct a scene whose marker scale was wrong; they are stored with the alignment and applied by the other commands.

`silhouette` projects the merged cloud (A plus transformed B) into every photograph of both orientations, using the
scaled OpenMVG scenes (sfm_data_scaled.json) and the lens model, and compares the projected outline with the SAM2
mask. A thickness error shows up as an outline that over- or undershoots the mask in the edge-on (low) views,
including the parts reconstructed only from the other orientation. It also repeats the comparison with B shifted
along the thickness axis, to show how strongly the photographs constrain the alignment there.

`merge` writes one OpenMVG scene holding both orientations, B's cameras and landmarks moved into A's frame, for a
single OpenMVS run on all photographs (convert it to .bin with openMVG_main_ConvertSfM_DataFormat, which validates it,
then pgs-recon --resume-sfm). --thin-shift moves B along the thin axis first, e.g. by the silhouette optimum.

`check` reviews the merged scene and the mesh OpenMVS made from it: landmark reprojection per orientation (the merge
must not disturb either half), signed distance of each aligned half-cloud to the mesh per face (the single run must
reproduce both halves), and the mesh outline against the masks in the merged cameras. --sheet writes a contact sheet
of mesh outline (green) over mask outline (red).

`reference` (rehearsal only) registers a dense cloud of the same object from a joint reconstruction onto the merged
cloud and reports, face by face, how far its surface lies from the merged one.

Usage:
    python split_align.py align A_DENSE.ply B_DENSE.ply OUT_DIR [--scale-a K] [--scale-b K]
    python split_align.py silhouette OUT_DIR A_SCENE.json B_SCENE.json MASK_DIR_A MASK_DIR_B
    python split_align.py reference OUT_DIR A_SCENE.json REF_DENSE.ply REF_SCENE.json
    python split_align.py merge OUT_DIR A_SCENE.json B_SCENE.json MERGED_SCENE.json [--thin-shift MM]
    python split_align.py check OUT_DIR MERGED_SCENE.json MESH.ply MASK_DIR [--sheet BLOCKS sheet.jpg]

OUT_DIR holds alignment.json, clouds.npz and merge.json between the commands. Scenes are OpenMVG JSON
(openMVG_main_ConvertSfM_DataFormat -i sfm_data.bin -o sfm_data.json); masks are <stem>_mask.png as in the PGS-Recon
input directory.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

DOWN = 4  # silhouette images are compared at 1/DOWN resolution


# ---------------------------------------------------------------- I/O

def read_dense_ply(path):
    """Points and normals of an OpenMVS dense cloud (binary PLY with per-vertex view lists)."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        buf = f.read()
    lines = header.decode().splitlines()
    n = int(next(l for l in lines if l.startswith("element vertex")).split()[2])
    sizes = {"float32": 4, "float": 4, "uint8": 1, "uchar": 1, "int32": 4, "uint32": 4}
    scalars, lists, off = {}, [], 0
    for l in lines:
        p = l.split()
        if p[:1] != ["property"]:
            continue
        if p[1] == "list":
            lists.append(sizes[p[3]])
        elif lists:
            raise SystemExit("scalar property after a list property is not supported")
        else:
            scalars[p[2]] = (off, p[1])
            off += sizes[p[1]]
    fixed = off
    raw = np.frombuffer(buf, np.uint8)
    if lists:
        offs = np.empty(n, np.int64)
        o = 0
        for i in range(n):
            offs[i] = o
            o += fixed
            for item in lists:
                o += 1 + item * buf[o]
    else:
        offs = np.arange(n, dtype=np.int64) * fixed
    rec = raw[offs[:, None] + np.arange(fixed)]

    def col(name):
        o, t = scalars[name]
        return rec[:, o:o + 4].copy().view(np.float32)[:, 0]
    P = np.stack([col("x"), col("y"), col("z")], 1).astype(np.float64)
    N = np.stack([col("nx"), col("ny"), col("nz")], 1).astype(np.float64) if "nx" in scalars else None
    return P, N


def load_scene(path, scale=1.0):
    """filename -> (intrinsic dict, R, C) for posed views of an OpenMVG JSON scene, centres multiplied by scale."""
    d = json.load(open(path))
    intr = {i["key"]: i["value"]["ptr_wrapper"]["data"] for i in d["intrinsics"]}
    ext = {e["key"]: e["value"] for e in d["extrinsics"]}
    cams = {}
    for v in d["views"]:
        vd = v["value"]["ptr_wrapper"]["data"]
        if vd["id_pose"] in ext and vd["id_intrinsic"] in intr:
            e = ext[vd["id_pose"]]
            cams[vd["filename"]] = (intr[vd["id_intrinsic"]], np.array(e["rotation"]), scale * np.array(e["center"]))
    return cams


def project(X, cam):
    """Pixel coordinates (pinhole_radial_k3) and depth of world points X in one camera."""
    K, R, C = cam
    Xc = (X - C) @ R.T
    z = Xc[:, 2]
    u, v = Xc[:, 0] / z, Xc[:, 1] / z
    r2 = u * u + v * v
    k = K.get("disto_k3", [0, 0, 0])
    s = 1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3
    f, (cx, cy) = K["focal_length"], K["principal_point"]
    return np.stack([f * s * u + cx, f * s * v + cy], 1), z


def save_T(path, T, extra):
    json.dump({"T_B_to_A": T.tolist(), **extra}, open(path, "w"), indent=1)


def load_T(out_dir):
    return np.array(json.load(open(os.path.join(out_dir, "alignment.json")))["T_B_to_A"])


def apply(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------- cloud preparation

def clean(P, N, voxel):
    """Voxel-downsample, drop isolated points and keep the largest connected cluster."""
    key = np.floor(P / voxel).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    P = np.stack([np.bincount(inv, P[:, k]) for k in range(3)], 1) / cnt[:, None]  # voxel centroids
    N = np.stack([np.bincount(inv, N[:, k]) for k in range(3)], 1)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    tree = cKDTree(P)
    d, _ = tree.query(P, 9)
    keep = d[:, -1] < np.median(d[:, -1]) * 3
    P, N = P[keep], N[keep]
    tree = cKDTree(P)
    pairs = tree.query_pairs(voxel * 3, output_type="ndarray")
    import scipy.sparse as sp
    import scipy.sparse.csgraph as cg
    g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    _, lab = cg.connected_components(g, directed=False)
    big = np.argmax(np.bincount(lab))
    return P[lab == big], N[lab == big]


def principal_axes(P):
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    return c, V[:, ::-1]  # columns: longest, middle, shortest axis


# ---------------------------------------------------------------- ICP

def match(T, src, srcn, tree, dst_n, dmax, min_dot):
    S = apply(T, src)
    Sn = srcn @ T[:3, :3].T
    d, j = tree.query(S, distance_upper_bound=dmax)
    ok = np.isfinite(d)
    ok[ok] &= np.einsum("ij,ij->i", Sn[ok], dst_n[j[ok]]) > min_dot
    return S[ok], tree.data[j[ok]], dst_n[j[ok]]


# (max pair distance mm, min normal agreement): coarse stages converge from the axis hypotheses; the tight final
# stages remove the bias from points of one cloud lying just beyond the other's coverage, which otherwise snap onto
# its last points and pull the halves together (0.08 mm along the thin axis on a synthetic 0525 split, 0.001 mm
# with the final stages).
STAGES = ((4, 0.6), (2, 0.6), (1, 0.6), (0.6, 0.6), (0.4, 0.6), (0.3, 0.95), (0.2, 0.95), (0.15, 0.95))


def icp(src, srcn, dst, dstn, T, stages=STAGES, iters=15):
    """Point-to-plane ICP of src onto dst over pairs with compatible normals."""
    tree = cKDTree(dst)
    for dmax, min_dot in stages:
        for _ in range(iters):
            p, q, n = match(T, src, srcn, tree, dstn, dmax, min_dot)
            if len(p) < 100:
                return T, None
            A = np.hstack([np.cross(p, n), n])
            b = np.einsum("ij,ij->i", q - p, n)
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            th = np.linalg.norm(x[:3])
            Tk = np.eye(4)
            Tk[:3, :3], Tk[:3, 3] = (np.eye(3) if th == 0 else rodrigues(x[:3] / th, th)), x[3:]
            T = Tk @ T
            if np.linalg.norm(x) < 1e-6:
                break
    p, q, n = match(T, src, srcn, tree, dstn, *stages[-1])
    r = np.einsum("ij,ij->i", q - p, n)
    return T, {"pairs": len(p), "rms": float(np.sqrt(np.mean(r ** 2))), "p": p, "n": n, "r": r}


def rodrigues(k, th):
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def hypotheses(PA, PB):
    cA, VA = principal_axes(PA)
    cB, VB = principal_axes(PB)
    for sx in (1, -1):
        for sy in (1, -1):
            D = np.diag([sx, sy, sx * sy * np.linalg.det(VA) * np.linalg.det(VB)])
            R = VA @ D @ VB.T
            for shift in (-1, 0, 1):  # partial clouds: centroids differ along the thickness axis
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = cA - R @ cB + shift * VA[:, 2] * 0.5 * np.ptp((PA - cA) @ VA[:, 2])
                yield f"axes {sx:+d}{sy:+d} shift {shift:+d}", T


def align(args):
    os.makedirs(args.out_dir, exist_ok=True)
    PA, NA = read_dense_ply(args.a)
    PB, NB = read_dense_ply(args.b)
    PA, PB = PA * args.scale_a, PB * args.scale_b
    print(f"A {len(PA)} points, B {len(PB)} points (scene units multiplied by {args.scale_a} and {args.scale_b})")
    PA, NA = clean(PA, NA, args.voxel)
    PB, NB = clean(PB, NB, args.voxel)
    print(f"after cleaning at {args.voxel} mm: A {len(PA)}, B {len(PB)}")
    np.savez_compressed(os.path.join(args.out_dir, "clouds.npz"), PA=PA, NA=NA, PB=PB, NB=NB)
    results = []
    for name, T0 in hypotheses(PA, PB):
        T, st = icp(PB, NB, PA, NA, T0)
        if st is None:
            print(f"{name}: no convergence")
            continue
        frac = st["pairs"] / len(PB)
        print(f"{name}: overlap {frac:.1%} of B, rms {st['rms']:.3f} mm")
        results.append((frac / max(st["rms"], 1e-3), name, T, st))
    results.sort(key=lambda r: -r[0])
    _, name, T, st = results[0]
    # pose covariance from the point-to-plane normal equations, sigma = residual rms
    p, n, r = st["p"], st["n"], st["r"]
    c = p.mean(0)
    A = np.hstack([np.cross(p - c, n), n])
    cov = np.linalg.inv(A.T @ A) * np.mean(r ** 2)
    _, V = principal_axes(np.vstack([PA, apply(T, PB)]))
    t_sd = [float(np.sqrt(V[:, k] @ cov[3:, 3:] @ V[:, k])) for k in range(3)]
    # translation uncertainty per object axis if the rim were the only constraint; correlated noise and mesh bias
    # make the real error larger, which is why the silhouette and reference checks exist.
    print(f"best: {name}, {st['pairs']} rim pairs, rms {st['rms']:.3f} mm")
    print(f"formal translation sd along long/middle/thin axis: {t_sd[0]:.4f} / {t_sd[1]:.4f} / {t_sd[2]:.4f} mm")
    order = [f"{x[1]} score {x[0]:.1f}" for x in results[:4]]
    print("ranking:", "; ".join(order))
    save_T(os.path.join(args.out_dir, "alignment.json"), T,
           {"hypothesis": name, "scale_a": args.scale_a, "scale_b": args.scale_b, "pairs": st["pairs"], "rms_mm": st["rms"], "axes": V.T.tolist(),
            "translation_sd_mm_long_mid_thin": t_sd, "voxel_mm": args.voxel})
    write_overlap_ply(os.path.join(args.out_dir, "rim_residuals.ply"), p, r)
    k = rim_scale(PB, NB, PA, NA, T)
    print(f"scale of B relative to A that best fits the rim: {k:.5f} (not applied; 1 = the two marker scales agree "
          f"on the object's size)")
    merged = np.vstack([PA, apply(T, PB)])
    ext = np.ptp((merged - merged.mean(0)) @ V, axis=0)
    print(f"merged cloud extent along long/middle/thin axis: {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} mm")


# pairing stages for rim_scale: a scale mismatch of 1% moves the rim of a 60 mm object by 0.3 mm, beyond the final
# ICP gate, so the scale is first fitted with wider gates
SCALE_STAGES = ((1.0, 0.6), (0.4, 0.6)) + STAGES[-1:]


def rim_scale(src, srcn, dst, dstn, T, iters=30):
    """Similarity refinement of a rigid alignment: the scale of src (about its centroid) that best fits the overlap.
    NaN if too few rim pairs remain to measure it."""
    tree = cKDTree(dst)
    c = src.mean(0)
    k = 1.0
    for dmax, min_dot in SCALE_STAGES:
        for _ in range(iters):
            S = c + k * (src - c)
            p, q, n = match(T, S, srcn, tree, dstn, dmax, min_dot)
            if len(p) < 100:
                return float("nan")
            cw = apply(T, c[None])[0]
            A = np.hstack([np.cross(p, n), n, np.einsum("ij,ij->i", p - cw, n)[:, None]])
            x = np.linalg.lstsq(A, np.einsum("ij,ij->i", q - p, n), rcond=None)[0]
            th = np.linalg.norm(x[:3])
            Tk = np.eye(4)
            Tk[:3, :3], Tk[:3, 3] = (np.eye(3) if th == 0 else rodrigues(x[:3] / th, th)), x[3:6]
            T = Tk @ T
            k *= 1 + x[6]
            if abs(x[6]) < 1e-7 and np.linalg.norm(x[:6]) < 1e-6:
                break
    return k


def write_overlap_ply(path, p, r):
    a = np.clip(np.abs(r) / 0.5, 0, 1)
    col = np.stack([255 * a, 255 * (1 - a), 64 + 0 * a], 1).astype(np.uint8)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(p)}\nproperty float x\n"
                 "property float y\nproperty float z\nproperty uchar red\nproperty uchar green\n"
                 "property uchar blue\nend_header\n").encode())
        rec = np.zeros(len(p), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = p, col
        f.write(rec.tobytes())


# ---------------------------------------------------------------- silhouettes

def silhouette_errors(X, cam, mask_path, close_px):
    """Signed distances (px at full resolution) from the projected outline to the mask outline, and px->mm."""
    K = cam[0]
    W, H = K["width"] // DOWN, K["height"] // DOWN
    uv, z = project(X, cam)
    uv = np.round(uv / DOWN).astype(int)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    sil = np.zeros((H, W), bool)
    sil[uv[ok, 1], uv[ok, 0]] = True
    st = ndimage.generate_binary_structure(2, 1)
    sil = ndimage.binary_closing(sil, st, iterations=close_px, border_value=0)
    sil = ndimage.binary_fill_holes(sil)
    m = np.array(Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)) > 127
    m = ndimage.binary_fill_holes(m)
    lab, nl = ndimage.label(m)
    if nl > 1:
        m = lab == 1 + np.argmax(ndimage.sum(m, lab, range(1, nl + 1)))
    edge = sil & ~ndimage.binary_erosion(sil)
    inside = ndimage.distance_transform_edt(m)
    outside = ndimage.distance_transform_edt(~m)
    d = np.where(m[edge], -(inside[edge] - 0.5), outside[edge] - 0.5) * DOWN  # >0: outline beyond the mask
    mm_per_px = float(np.median(z[ok])) / K["focal_length"]
    return d, mm_per_px


def silhouette(args):
    T = load_T(args.out_dir)
    c = np.load(os.path.join(args.out_dir, "clouds.npz"))
    PA, PB = c["PA"], c["PB"]
    meta = json.load(open(os.path.join(args.out_dir, "alignment.json")))
    camsA = load_scene(args.scene_a, meta.get("scale_a", 1.0))
    camsB = load_scene(args.scene_b, meta.get("scale_b", 1.0))
    V = np.array(meta["axes"])
    thin = V[2]
    rows = []
    for shift in args.shifts:
        TB = T.copy()
        TB[:3, 3] += shift * thin
        merged_A = np.vstack([PA, apply(TB, PB)])                    # in A's frame
        merged_B = apply(np.linalg.inv(TB), merged_A)                # in B's frame
        for side, cams, X, mdir in (("A", camsA, merged_A, args.mask_a), ("B", camsB, merged_B, args.mask_b)):
            for k, (fn, cam) in enumerate(sorted(cams.items())):
                if k % args.every:
                    continue
                mp = os.path.join(mdir, os.path.splitext(fn)[0] + "_mask.png")
                d, mmpx = silhouette_errors(X, cam, mp, args.close_px)
                rows.append((shift, side, fn, np.median(d) * mmpx, np.percentile(np.abs(d), 95) * mmpx,
                             np.mean(d > 2 * DOWN), len(d)))
        sel = [r for r in rows if r[0] == shift]
        med = np.array([r[3] for r in sel])
        p95 = np.array([r[4] for r in sel])
        print(f"shift {shift:+.2f} mm along thin axis: median signed outline offset {np.median(med):+.3f} mm, "
              f"median p95 |offset| {np.median(p95):.3f} mm, worst view p95 {p95.max():.3f} mm")
    shifts = np.array(args.shifts)
    if len(shifts) >= 3:
        curve = np.array([np.median([r[4] for r in rows if r[0] == sh]) for sh in shifts])
        a, b, _ = np.polyfit(shifts, curve, 2)
        best = -b / (2 * a) if a > 0 else float("nan")
        print(f"outline error is smallest at a shift of {best:+.3f} mm along the thin axis "
              f"(0 = the alignment agrees with the photographs; fitted over {shifts.min():+.1f}..{shifts.max():+.1f} mm)")
    with open(os.path.join(args.out_dir, "silhouette.tsv"), "w") as f:
        f.write("shift_mm\tside\tview\tmedian_signed_mm\tp95_abs_mm\tfrac_outside_2px\tedge_px\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print("wrote", os.path.join(args.out_dir, "silhouette.tsv"))


# ---------------------------------------------------------------- reference transform

def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = np.trace(np.diag(sig) @ E) / S.var(0).sum()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, md - s * R @ ms
    return T, s


def reference(args):
    """Register a reference dense cloud (same object, e.g. a joint reconstruction) onto the merged cloud and report
    the normal offset of each face. Camera-centre similarities between independent SfM runs are not a usable
    reference: focal length trades against camera distance, moving the centres without changing the geometry."""
    T = load_T(args.out_dir)
    a = json.load(open(os.path.join(args.out_dir, "alignment.json")))
    V = np.array(a["axes"])
    c = np.load(os.path.join(args.out_dir, "clouds.npz"))
    M = np.vstack([c["PA"], apply(T, c["PB"])])
    MN = np.vstack([c["NA"], c["NB"] @ (T[:3, :3] / np.cbrt(np.linalg.det(T[:3, :3]))).T])
    side = np.r_[np.zeros(len(c["PA"]), int), np.ones(len(c["PB"]), int)]
    J, JN = read_dense_ply(args.ref_cloud)
    J, JN = clean(J, JN, a.get("voxel_mm", 0.25))
    camsA, camsJ = load_scene(args.scene_a, a.get("scale_a", 1.0)), load_scene(args.ref_scene)
    common = sorted(set(camsA) & set(camsJ))
    S, s = umeyama(np.array([camsJ[f][2] for f in common]), np.array([camsA[f][2] for f in common]))
    print(f"initial pose from {len(common)} shared A cameras, scale {s:.5f}")
    J, JN = apply(S, J), JN @ (S[:3, :3] / s).T
    Tj, st = icp(J, JN, M, MN, np.eye(4), stages=STAGES[2:])
    print(f"reference registered rigidly onto the merged cloud: {st['pairs']} pairs, rms {st['rms']:.3f} mm")
    P, PN = apply(Tj, J), JN @ Tj[:3, :3].T
    d, j = cKDTree(M).query(P, distance_upper_bound=2.0)
    ok = np.isfinite(d)
    r = np.einsum("ij,ij->i", P[ok] - M[j[ok]], MN[j[ok]])
    nt = PN[ok] @ V[2]
    face = np.where(nt > 0.7, 0, np.where(nt < -0.7, 1, 2))
    med = {}
    for f, fname in enumerate(("+thin face", "-thin face", "rim")):
        for w, wname in enumerate(("A", "B")):
            m = (face == f) & (side[j[ok]] == w)
            if m.sum() > 500:
                med[f, w] = float(np.median(r[m]))
                print(f"  {fname:10s} vs merged points from {wname}: n {m.sum():7d}, median offset {med[f, w]:+.3f} mm "
                      f"(>0: reference outside), p5..p95 {np.percentile(r[m], 5):+.3f}..{np.percentile(r[m], 95):+.3f}")
    print("reference thickness minus merged thickness, from the dominant source of each face: "
          f"{max(med.get((0, 0), 0), med.get((0, 1), 0), key=abs) + max(med.get((1, 0), 0), med.get((1, 1), 0), key=abs):+.3f} mm")


# ---------------------------------------------------------------- merged scene

def merge(args):
    """One OpenMVG scene with A's views as they are and B's views, poses and landmarks moved by the alignment."""
    T = load_T(args.out_dir)
    s = np.cbrt(np.linalg.det(T[:3, :3]))
    R = T[:3, :3] / s
    A, B = json.load(open(args.scene_a)), json.load(open(args.scene_b))
    meta = json.load(open(os.path.join(args.out_dir, "alignment.json")))
    T[:3, 3] += args.thin_shift * np.array(meta["axes"])[2]  # same convention as the silhouette shifts
    for scene, k in ((A, meta.get("scale_a", 1.0)), (B, meta.get("scale_b", 1.0))):
        for e in scene["extrinsics"]:
            e["value"]["center"] = [k * x for x in e["value"]["center"]]
        for l in scene["structure"]:
            l["value"]["X"] = [k * x for x in l["value"]["X"]]
    if A["root_path"] != B["root_path"]:
        raise SystemExit("scenes have different root paths")
    vo = 1 + max(v["key"] for v in A["views"])
    po = 1 + max([v["value"]["ptr_wrapper"]["data"]["id_pose"] for v in A["views"]])
    io = 1 + max(i["key"] for i in A["intrinsics"])
    lo = 1 + max([l["key"] for l in A["structure"]], default=-1)
    views = [v for v in A["views"]]
    for v in B["views"]:
        d = dict(v["value"]["ptr_wrapper"]["data"])
        d["id_view"] += vo
        d["id_pose"] += po
        d["id_intrinsic"] += io
        views.append({"key": v["key"] + vo, "value": {"polymorphic_id": v["value"]["polymorphic_id"],
                                                      "ptr_wrapper": {"id": 0, "data": d}}})
    names = [v["value"]["ptr_wrapper"]["data"]["filename"] for v in views]
    if len(set(names)) != len(names):
        raise SystemExit("the two scenes share photographs")
    intr = [i for i in A["intrinsics"]] + [{"key": i["key"] + io, "value": i["value"]} for i in B["intrinsics"]]
    ext = list(A["extrinsics"])
    for e in B["extrinsics"]:
        Rc, C = np.array(e["value"]["rotation"]), np.array(e["value"]["center"])
        ext.append({"key": e["key"] + po, "value": {"rotation": (Rc @ R.T).tolist(),
                                                    "center": apply(T, C[None])[0].tolist()}})
    struct = list(A["structure"])
    for l in B["structure"]:
        X = apply(T, np.array(l["value"]["X"])[None])[0].tolist()
        obs = [{"key": o["key"] + vo, "value": o["value"]} for o in l["value"]["observations"]]
        struct.append({"key": l["key"] + lo, "value": {"X": X, "observations": obs}})
    # cereal pointer ids: sequential over views then intrinsics; the first intrinsic carries the type name
    pid = 2147483649
    for v in views:
        v["value"]["ptr_wrapper"]["id"] = pid
        pid += 1
    named = next(i["value"] for i in A["intrinsics"] if "polymorphic_name" in i["value"])
    out_intr = []
    for k, i in enumerate(intr):
        val = {"polymorphic_id": 2147483649 if k == 0 else 1}
        if k == 0:
            val["polymorphic_name"] = named["polymorphic_name"]
        val["ptr_wrapper"] = {"id": pid, "data": i["value"]["ptr_wrapper"]["data"]}
        pid += 1
        out_intr.append({"key": i["key"], "value": val})
    types = {i["value"].get("polymorphic_name") for i in A["intrinsics"] + B["intrinsics"]} - {None}
    if len(types) != 1:
        raise SystemExit(f"mixed intrinsic types {types} are not supported")
    out = {"sfm_data_version": A["sfm_data_version"], "root_path": A["root_path"], "views": views,
           "intrinsics": out_intr, "extrinsics": ext, "structure": struct, "control_points": []}
    json.dump(out, open(args.out_scene, "w"), indent=1)
    json.dump({"thin_shift_mm": args.thin_shift, "view_offset_b": vo, "scene_a": os.path.abspath(args.scene_a),
               "scene_b": os.path.abspath(args.scene_b), "merged_scene": os.path.abspath(args.out_scene)},
              open(os.path.join(args.out_dir, "merge.json"), "w"), indent=1)
    print(f"wrote {args.out_scene}: {len(views)} views, {len(out_intr)} intrinsics, {len(ext)} poses, "
          f"{len(struct)} landmarks (B scale factor {s:.5f}, thin-axis shift {args.thin_shift:+.3f} mm)")


# ---------------------------------------------------------------- merged scene and mesh review

def load_views(path):
    """Merged scene -> (cams by filename, view key by filename, landmarks, intrinsics, extrinsics, views by key)."""
    d = json.load(open(path))
    intr = {i["key"]: i["value"]["ptr_wrapper"]["data"] for i in d["intrinsics"]}
    ext = {e["key"]: e["value"] for e in d["extrinsics"]}
    views = {v["key"]: v["value"]["ptr_wrapper"]["data"] for v in d["views"]}
    return d, intr, ext, views


def reprojection_by_half(d, intr, ext, views, offset, n=4000, seed=0):
    """Pixel residuals of a random subset of landmarks, split by orientation (view key below/above offset)."""
    err = {"A": [], "B": []}
    rng = np.random.default_rng(seed)
    S = d["structure"]
    for k in rng.permutation(len(S))[:n]:
        X = np.array(S[k]["value"]["X"])[None]
        for o in S[k]["value"]["observations"]:
            v = views[o["key"]]
            if v["id_pose"] not in ext:
                continue
            e = ext[v["id_pose"]]
            px, _ = project(X, (intr[v["id_intrinsic"]], np.array(e["rotation"]), np.array(e["center"])))
            err["A" if o["key"] < offset else "B"].append(np.linalg.norm(px[0] - np.array(o["value"]["x"])))
    return {k: np.array(v) for k, v in err.items()}


def vertex_normals(V, F):
    """Area-weighted vertex normals of a triangle mesh."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def face_offsets(P, N, V, VN, thin, dmax=1.0):
    """Signed distance (+ = point outside the mesh) of points to the nearest mesh vertex along its normal, grouped by
    the face the points lie on: normal along +thin, along -thin, or across it (rim)."""
    d, i = cKDTree(V).query(P)
    s = np.einsum("ij,ij->i", P - V[i], VN[i])
    nz = N @ thin
    out = {}
    for face, sel in (("+thin face", nz > 0.7), ("-thin face", nz < -0.7), ("rim", np.abs(nz) < 0.3)):
        out[face] = s[sel & (d < dmax)]
    return out


def outline_overlay(X, cam, img_path, mask_path, down=DOWN, close_px=3):
    """Photo at 1/down resolution with the mask outline in red and the projected outline of X in green."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size[0] // down, img.size[1] // down
    a = np.array(img.resize((W, H)))
    m = np.array(Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)) > 127
    uv, z = project(X, cam)
    uv = np.round(uv / down).astype(int)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    sil = np.zeros((H, W), bool)
    sil[uv[ok, 1], uv[ok, 0]] = True
    sil = ndimage.binary_fill_holes(ndimage.binary_closing(sil, iterations=close_px))
    a[m & ~ndimage.binary_erosion(m)] = (255, 0, 0)
    a[sil & ~ndimage.binary_erosion(sil)] = (0, 255, 0)
    ys, xs = np.nonzero(m | sil)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    r = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + 20
    return Image.fromarray(a).crop((cx - r, cy - r, cx + r, cy + r))


def check(args):
    from PIL import ImageDraw
    import trimesh
    meta = json.load(open(os.path.join(args.out_dir, "alignment.json")))
    mj = json.load(open(os.path.join(args.out_dir, "merge.json")))
    thin = np.array(meta["axes"])[2]
    d, intr, ext, views = load_views(args.merged_scene)
    print(f"== merged scene: {len(views)} views ({sum(v['id_pose'] in ext for v in views.values())} posed), "
          f"B moved by {mj['thin_shift_mm']:+.3f} mm along the thin axis")
    for k, e in reprojection_by_half(d, intr, ext, views, mj["view_offset_b"]).items():
        print(f"   landmark reprojection, orientation {k}: {len(e)} observations, median {np.median(e):.3f} px, "
              f"p95 {np.percentile(e, 95):.3f} px")

    m = trimesh.load(args.mesh, process=False)
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    VN = vertex_normals(V, F)
    T = load_T(args.out_dir)
    T[:3, 3] += mj["thin_shift_mm"] * thin
    c = np.load(os.path.join(args.out_dir, "clouds.npz"))
    print("== aligned half-clouds vs the mesh (signed, + = cloud outside the mesh; points within 1 mm)")
    for name, P, N in (("A", c["PA"], c["NA"]), ("B", apply(T, c["PB"]), c["NB"] @ T[:3, :3].T)):
        for face, s in face_offsets(P, N, V, VN, thin).items():
            if len(s):
                print(f"   {name} {face:10s}: {len(s):7d} points, median {np.median(s):+.3f} mm, IQR "
                      f"{np.percentile(s, 25):+.3f}..{np.percentile(s, 75):+.3f}, |d| > 0.5 mm {np.mean(np.abs(s) > 0.5):.1%}")

    key = {v["filename"]: k for k, v in views.items()}
    cams = load_scene(args.merged_scene)
    rows = []
    for k, (fn, cam) in enumerate(sorted(cams.items())):
        if k % args.every:
            continue
        e, mmpx = silhouette_errors(V, cam, os.path.join(args.mask_dir, os.path.splitext(fn)[0] + "_mask.png"),
                                    args.close_px)
        rows.append(("A" if key[fn] < mj["view_offset_b"] else "B", fn, np.median(e) * mmpx,
                     np.percentile(np.abs(e), 95) * mmpx))
    print(f"== mesh outline vs masks (every {args.every}th view; + = mesh beyond the mask)")
    for side in "AB":
        r = [x for x in rows if x[0] == side]
        if not r:
            continue
        med, p95 = np.array([x[2] for x in r]), np.array([x[3] for x in r])
        worst = r[int(np.argmax(p95))]
        print(f"   {side}: {len(r)} views, median signed offset {np.median(med):+.3f} mm, median p95 |offset| "
              f"{np.median(p95):.3f} mm, worst view {worst[1]} p95 {worst[3]:.3f} mm")

    if args.sheet:
        blocks, out = args.sheet
        tile, per = 300, 6
        grid = []
        for b in sorted(os.listdir(blocks)):
            src = os.path.join(blocks, b, "JPGEnhanced")
            if not os.path.isdir(src):
                continue
            stems = sorted(os.path.splitext(f)[0] for f in os.listdir(src) if f.lower().endswith(".jpg"))
            row = []
            for st in stems[::max(1, len(stems) // per)][:per]:
                if st + ".jpg" not in cams:
                    continue
                t = outline_overlay(V, cams[st + ".jpg"], os.path.join(args.mask_dir, st + ".jpg"),
                                    os.path.join(args.mask_dir, st + "_mask.png")).resize((tile, tile))
                ImageDraw.Draw(t).text((5, 5), f"{b} {st}", fill=(255, 255, 0))
                row.append(t)
            grid.append(row)
        sheet = Image.new("RGB", (tile * max(map(len, grid)), tile * len(grid)), "white")
        for i, row in enumerate(grid):
            for j, t in enumerate(row):
                sheet.paste(t, (tile * j, tile * i))
        sheet.save(out, quality=90)
        print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("align")
    a.add_argument("a")
    a.add_argument("b")
    a.add_argument("out_dir")
    a.add_argument("--voxel", type=float, default=0.25, help="downsampling voxel (mm)")
    a.add_argument("--scale-a", type=float, default=1.0, help="factor taking A's scene units to mm")
    a.add_argument("--scale-b", type=float, default=1.0, help="factor taking B's scene units to mm")
    s = sub.add_parser("silhouette")
    s.add_argument("out_dir")
    s.add_argument("scene_a")
    s.add_argument("scene_b")
    s.add_argument("mask_a")
    s.add_argument("mask_b")
    s.add_argument("--every", type=int, default=4, help="use every n-th view")
    s.add_argument("--close-px", type=int, default=3, help="closing iterations for the splatted outline")
    s.add_argument("--shifts", type=float, nargs="+", default=[-0.6, -0.3, 0, 0.3, 0.6])
    r = sub.add_parser("reference")
    r.add_argument("out_dir")
    r.add_argument("scene_a")
    r.add_argument("ref_cloud")
    r.add_argument("ref_scene")
    m = sub.add_parser("merge")
    m.add_argument("out_dir")
    m.add_argument("scene_a")
    m.add_argument("scene_b")
    m.add_argument("out_scene")
    m.add_argument("--thin-shift", type=float, default=0.0,
                   help="move B along the thin axis by this many mm before merging (e.g. the silhouette optimum)")
    c = sub.add_parser("check")
    c.add_argument("out_dir")
    c.add_argument("merged_scene")
    c.add_argument("mesh", help="OpenMVS mesh of the merged scene (scene_dense_mesh.ply or scene_dense_refine.ply)")
    c.add_argument("mask_dir", help="directory with every photograph and its <stem>_mask.png (the PGS-Recon input)")
    c.add_argument("--every", type=int, default=8, help="use every n-th view for the outline check")
    c.add_argument("--close-px", type=int, default=3, help="closing iterations for the splatted outline")
    c.add_argument("--sheet", nargs=2, metavar=("BLOCKS", "SHEET"),
                   help="acquisition tree (one <orientation>_camera_<height>/JPGEnhanced/ per block) and output .jpg")
    args = p.parse_args()
    {"align": align, "silhouette": silhouette, "reference": reference, "merge": merge,
     "check": check}[args.command](args)


if __name__ == "__main__":
    main()
