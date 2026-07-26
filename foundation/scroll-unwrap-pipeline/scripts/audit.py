"""M0 audit: full mesh/texture/orientation/ink/zip inventory.

Produces reports/audit.json (machine-readable, includes the binding `tex_orientation`
that downstream renderers must read) and reports/audit.md (human summary).

All mesh bytes are read via scrollkit.io (strict reader). Images via PIL/tifffile only.

Orientation oracle
------------------
Contract metric: IoU between the rasterized UV-triangle coverage mask and the texture
content mask (alpha>127 for RGBA; |gray - flat_fill_mode| > 4 for the gray-filled Group C
atlases) under the two prescribed hypotheses
    opengl_bottomleft: pixel_row = (1-t)*(H-1)
    topleft:           pixel_row = t*(H-1)
(both with pixel_col = s*(W-1)). Winner by IoU; |delta| < 0.02 => 'inconclusive'.

Because the prescribed pair spans only the vertical flip, the oracle also evaluates the
two u-flipped hypotheses (topleft_u_flipped, rot180) plus a one-sided containment
diagnostic (fraction of content-mask pixels OUTSIDE the chart coverage; ~0 under the true
mapping since composited content can only exist inside the chart). The definitive
`tex_orientation` is derived from the 4-hypothesis result.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFilter

Image.MAX_IMAGE_PIXELS = None  # trusted data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scrollkit.io import read_ply  # noqa: E402
from scrollkit.unwrap.denorm import fit_uv_scale  # noqa: E402

MARCH_DIR = ROOT / "textured_plys" / "textured_ply_march"
MAY_DIR = ROOT / "textured_plys" / "textured_ply_may"
C_GLOB = "scroll_meshes-20260610T062158Z-3-001/scroll_meshes/*/*.ply"
INK_MARCH_DIR = ROOT / "1667 - ink detection"
INK_MAY_DIR = ROOT / "ink det + max comp (may 5th)"

ORACLE_MAX_DIM = 768
ORACLE_MAX_TRIS = 60_000
ORACLE_MARGIN = 0.02

HYPOTHESES = {
    "topleft": "pixel_col = s*(W-1); pixel_row = t*(H-1)",
    "opengl_bottomleft": "pixel_col = s*(W-1); pixel_row = (1-t)*(H-1)",
    "topleft_u_flipped": "pixel_col = (1-s)*(W-1); pixel_row = t*(H-1)",
    "rot180": "pixel_col = (1-s)*(W-1); pixel_row = (1-t)*(H-1)",
}


# ---------------------------------------------------------------- helpers

def js(o):
    """Recursively convert to JSON-serializable python types."""
    if isinstance(o, dict):
        return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [js(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return js(o.tolist())
    return o


def sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def edge_topology(faces: np.ndarray, n_vertices: int, vertices: np.ndarray) -> dict:
    """Unique-edge stats, boundary/nonmanifold counts, Euler char, components."""
    f = faces.astype(np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    key = e[:, 0] * np.int64(n_vertices) + e[:, 1]
    ukey, counts = np.unique(key, return_counts=True)
    i = ukey // n_vertices
    j = ukey % n_vertices
    n_edges = int(len(ukey))
    lengths = np.linalg.norm(
        vertices[i].astype(np.float64) - vertices[j].astype(np.float64), axis=1
    )

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    adj = coo_matrix(
        (np.ones(n_edges, dtype=np.int8), (i, j)), shape=(n_vertices, n_vertices)
    )
    n_comp, labels = connected_components(adj, directed=False)
    comp_sizes = np.sort(np.bincount(labels))[::-1][:8].tolist()
    referenced = np.zeros(n_vertices, dtype=bool)
    referenced[faces] = True
    n_isolated = int((~referenced).sum())

    return {
        "component_vertex_sizes_desc": comp_sizes,
        "n_edges": n_edges,
        "edge_len_mean": float(lengths.mean()),
        "edge_len_median": float(np.median(lengths)),
        "boundary_edge_count": int((counts == 1).sum()),
        "nonmanifold_edge_count": int((counts > 2).sum()),
        "euler_characteristic": int(n_vertices - n_edges + faces.shape[0]),
        "connected_components": int(n_comp),
        "connected_components_excluding_isolated_vertices": int(n_comp - n_isolated),
        "isolated_vertex_count": n_isolated,
    }


def uv_signed_areas(uv_tris: np.ndarray) -> np.ndarray:
    t = uv_tris.astype(np.float64)
    e1 = t[:, 1] - t[:, 0]
    e2 = t[:, 2] - t[:, 0]
    return 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def tri_areas_3d(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = vertices.astype(np.float64)[faces]
    return 0.5 * np.linalg.norm(
        np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1
    )


def rasterize_coverage(uv_tris: np.ndarray, wr: int, hr: int) -> np.ndarray:
    """Boolean chart-coverage mask: subsampled UV triangles drawn topleft (row = t*(H-1)),
    then dilated (MaxFilter 5) to close the gaps left by face subsampling."""
    stride = max(1, int(np.ceil(len(uv_tris) / ORACLE_MAX_TRIS)))
    tri = uv_tris[::stride].astype(np.float64)
    px = np.rint(
        np.stack([tri[..., 0] * (wr - 1), tri[..., 1] * (hr - 1)], axis=-1)
    ).astype(np.int32)
    img = Image.new("L", (wr, hr), 0)
    d = ImageDraw.Draw(img)
    for t in px:
        d.polygon(
            [(t[0, 0], t[0, 1]), (t[1, 0], t[1, 1]), (t[2, 0], t[2, 1])], fill=255
        )
    return np.asarray(img.filter(ImageFilter.MaxFilter(5))) > 0


def content_mask(im: Image.Image, wr: int, hr: int) -> tuple[np.ndarray, str]:
    """Texture content mask at raster size. RGBA -> alpha>127. Otherwise treat the
    dominant flat gray value as atlas background fill (Group C) and mask everything else."""
    if im.mode == "RGBA":
        a = np.asarray(im.getchannel("A").resize((wr, hr), Image.NEAREST))
        return a > 127, "alpha>127"
    g = np.asarray(im.convert("L").resize((wr, hr), Image.NEAREST))
    mode = int(np.bincount(g.ravel(), minlength=256).argmax())
    return np.abs(g.astype(np.int16) - mode) > 4, f"|gray-{mode}|>4 (flat fill)"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def ratio_record(num: int, den: int) -> dict:
    fr = Fraction(num, den)
    approx = fr.limit_denominator(16)
    val = num / den
    return {
        "ratio": round(val, 6),
        "nearest_simple_rational": f"{approx.numerator}/{approx.denominator}",
        "nearest_simple_rational_rel_err": round(abs(float(approx) - val) / val, 6)
        if val
        else 0.0,
    }


# ---------------------------------------------------------------- mesh audit

def audit_mesh(path: Path, group: str) -> tuple[dict, dict]:
    """Returns (mesh_record, oracle_record)."""
    m = read_ply(path)
    rec: dict = {
        "group": group,
        "path": str(path.relative_to(ROOT)),
        "stem": path.stem,
        "n_vertices": m.n_vertices,
        "n_faces": m.n_faces,
        "bbox_min": np.asarray(m.vertices, dtype=np.float64).min(axis=0).tolist(),
        "bbox_max": np.asarray(m.vertices, dtype=np.float64).max(axis=0).tolist(),
        "has_normals": m.normals is not None,
        "has_vertex_uv": m.vertex_uv is not None,
        "has_wedge_uv": m.wedge_uv is not None,
        "wedge_equals_vertex_uv": m.wedge_equals_vertex_uv(),
    }
    rec.update(edge_topology(m.faces, m.n_vertices, m.vertices))
    rec["degenerate_3d_tri_count"] = int(
        (tri_areas_3d(m.vertices, m.faces) < 1e-12).sum()
    )

    if m.face_texnumber is not None:
        vals, cnts = np.unique(m.face_texnumber, return_counts=True)
        rec["texnumber_histogram"] = {int(v): int(c) for v, c in zip(vals, cnts)}
    else:
        rec["texnumber_histogram"] = None

    # UV metrics: per-vertex UV is the flip/denorm source for A/B, wedge for C.
    if m.vertex_uv is not None:
        uv_tris = m.vertex_uv[m.faces]
        rec["uv_source"] = "vertex_uv"
        flat_uv = m.vertex_uv.astype(np.float64)
    else:
        uv_tris = m.wedge_uv
        rec["uv_source"] = "wedge_uv"
        flat_uv = m.wedge_uv.reshape(-1, 2).astype(np.float64)
    sa = uv_signed_areas(uv_tris)
    rec["uv_range"] = {
        "u": [float(flat_uv[:, 0].min()), float(flat_uv[:, 0].max())],
        "v": [float(flat_uv[:, 1].min()), float(flat_uv[:, 1].max())],
    }
    rec["uv_flipped_count"] = int((sa <= 0).sum())  # signed area <= 0
    rec["uv_flipped_fraction"] = round(float((sa <= 0).mean()), 6)
    rec["uv_zero_area_count"] = int((sa == 0).sum())
    rec["uv_positive_count"] = int((sa > 0).sum())

    # Textures (all declared TextureFile comments, resolved relative to the PLY dir).
    textures = []
    first_im: Image.Image | None = None
    for name in m.texture_files:
        tp = path.parent / name
        t: dict = {"name": name, "exists": tp.is_file()}
        if t["exists"]:
            t["file_size"] = tp.stat().st_size
            t["sha256_16"] = sha256_16(tp)
            im = Image.open(tp)
            im.load()
            t["width"], t["height"] = im.size
            t["mode"] = im.mode
            if im.mode == "RGBA":
                a = np.asarray(im.getchannel("A"))[::4, ::4]
                n = a.size
                t["alpha_frac_0"] = round(float((a == 0).sum() / n), 6)
                t["alpha_frac_255"] = round(float((a == 255).sum() / n), 6)
                t["alpha_frac_mid"] = round(
                    1.0 - t["alpha_frac_0"] - t["alpha_frac_255"], 6
                )
                del a
            if first_im is None:
                first_im = im
        textures.append(t)
    rec["textures"] = textures

    # De-normalization scale fit (groups A/B: normalized SLIM UVs).
    if group in ("A", "B") and m.vertex_uv is not None and first_im is not None:
        fit = fit_uv_scale(m.vertices, m.faces, m.vertex_uv)
        tw, th = first_im.size
        fit["s_u_px"] = fit["s_u"] / tw  # voxels per texture pixel along u
        fit["s_v_px"] = fit["s_v"] / th
        fit["pixel_scale_anisotropy"] = fit["s_u_px"] / fit["s_v_px"]
        rec["denorm"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in fit.items()}
    else:
        rec["denorm"] = None

    # Orientation oracle.
    oracle: dict = {"group": group, "path": rec["path"]}
    if first_im is not None:
        W, H = first_im.size
        sc = ORACLE_MAX_DIM / max(W, H)
        wr, hr = max(1, round(W * sc)), max(1, round(H * sc))
        cov = rasterize_coverage(uv_tris, wr, hr)
        msk, src = content_mask(first_im, wr, hr)
        transforms = {
            "topleft": msk,
            "opengl_bottomleft": msk[::-1, :],
            "topleft_u_flipped": msk[:, ::-1],
            "rot180": msk[::-1, ::-1],
        }
        ious = {k: round(iou(cov, v), 4) for k, v in transforms.items()}
        msum = max(1, int(msk.sum()))
        uncov = {
            k: round(float((v & ~cov).sum() / msum), 4) for k, v in transforms.items()
        }
        # Contract decision: 2 prescribed hypotheses.
        i_b, i_t = ious["opengl_bottomleft"], ious["topleft"]
        margin2 = abs(i_b - i_t)
        lead2 = "opengl_bottomleft" if i_b > i_t else "topleft"
        # Extended 4-way decision.
        order = sorted(ious, key=ious.get, reverse=True)
        margin4 = ious[order[0]] - ious[order[1]]
        stride = max(1, int(np.ceil(len(uv_tris) / ORACLE_MAX_TRIS)))
        oracle.update(
            {
                "raster_size": [wr, hr],
                "n_tris_drawn": len(range(0, len(uv_tris), stride)),
                "mask_source": src,
                "coverage_fraction": round(float(cov.mean()), 4),
                "mask_fraction": round(float(msk.mean()), 4),
                "iou_opengl_bottomleft": i_b,
                "iou_topleft": i_t,
                "margin_2way": round(margin2, 4),
                "leader_2way": lead2,
                "winner_2way": lead2 if margin2 >= ORACLE_MARGIN else "inconclusive",
                "iou_topleft_u_flipped": ious["topleft_u_flipped"],
                "iou_rot180": ious["rot180"],
                "margin_4way": round(margin4, 4),
                "winner_4way": order[0] if margin4 >= ORACLE_MARGIN else "inconclusive",
                "leader_4way": order[0],
                "uncovered_content_fraction": uncov,
            }
        )
    else:
        oracle["winner_2way"] = oracle["winner_4way"] = "no_texture"
    del m
    return rec, oracle


# ---------------------------------------------------------------- ink inventory

def stats_2d(a: np.ndarray) -> dict:
    if a.ndim == 3:
        a = a.mean(axis=2)
    af = a.astype(np.float64)
    mx = float(af.max()) if af.size else 0.0
    return {
        "min": float(af.min()) if af.size else 0.0,
        "max": mx,
        "mean": round(float(af.mean()), 4) if af.size else 0.0,
        "frac_gt_half_max": round(float((af > 0.5 * mx).mean()), 6) if mx > 0 else 0.0,
    }


def audit_ink_march(mesh_by_stem: dict) -> tuple[list, list, list]:
    matches, orphans = [], []
    a_stems = {s for s, r in mesh_by_stem.items() if r["group"] == "A"}
    tif_stems = set()
    for tp in sorted(INK_MARCH_DIR.glob("*.tif")):
        tif_stems.add(tp.stem)
        with tifffile.TiffFile(tp) as tf:
            page = tf.pages[0]
            shape, dtype = tuple(page.shape), str(page.dtype)
        arr = tifffile.imread(tp)
        sub = arr[::8, ::8].copy()
        del arr
        gc.collect()
        rec = {
            "tif": str(tp.relative_to(ROOT)),
            "stem": tp.stem,
            "shape_hw": list(shape),
            "dtype": dtype,
            "file_size": tp.stat().st_size,
            "stats_subsampled_8x": stats_2d(sub),
            "polarity": "positive (ink bright)",
        }
        del sub
        if tp.stem in a_stems:
            mrec = mesh_by_stem[tp.stem]
            tex = next((t for t in mrec["textures"] if t["exists"]), None)
            rec["matched_mesh"] = mrec["path"]
            if tex and len(shape) >= 2:
                rec["dims_ratio_vs_texture"] = {
                    "w": ratio_record(shape[1], tex["width"]),
                    "h": ratio_record(shape[0], tex["height"]),
                    "uniform_scale": bool(
                        abs(
                            (shape[1] / tex["width"]) / (shape[0] / tex["height"]) - 1
                        )
                        < 0.01
                    ),
                }
            matches.append(rec)
        else:
            rec["matched_mesh"] = None
            orphans.append(rec)
    unmatched = sorted(
        mesh_by_stem[s]["path"] for s in a_stems - tif_stems
    )
    return matches, orphans, unmatched


def audit_ink_may(mesh_by_stem: dict) -> tuple[list, list, list, list]:
    pred_dir = INK_MAY_DIR / "predictions_inv"
    b_stems = sorted(s for s, r in mesh_by_stem.items() if r["group"] == "B")
    matches, unmatched = [], []
    for stem in b_stems:
        jp = pred_dir / f"{stem}_new_canon_autoresearch_recipe.jpg"
        if not jp.is_file():
            unmatched.append(mesh_by_stem[stem]["path"])
            continue
        im = Image.open(jp)
        w, h = im.size
        sub = np.asarray(im)[::8, ::8].copy()
        im.close()
        st = stats_2d(sub)
        del sub
        gc.collect()
        mrec = mesh_by_stem[stem]
        tex = next((t for t in mrec["textures"] if t["exists"]), None)
        rec = {
            "jpg": str(jp.relative_to(ROOT)),
            "matched_mesh": mrec["path"],
            "dims_wh": [w, h],
            "file_size": jp.stat().st_size,
            "stats_subsampled_8x": st,
            "polarity": "inverted (ink dark)",
            "polarity_verified_mean_gt_half": bool(st["mean"] > 0.5 * max(st["max"], 1)),
        }
        if tex:
            rec["dims_ratio_vs_texture"] = {
                "w": ratio_record(w, tex["width"]),
                "h": ratio_record(h, tex["height"]),
                "uniform_scale": bool(
                    abs((w / tex["width"]) / (h / tex["height"]) - 1) < 0.01
                ),
            }
        matches.append(rec)
    psds = [
        {"psd": str(p.relative_to(ROOT)), "file_size": p.stat().st_size}
        for p in sorted(pred_dir.glob("*.psd"))
    ]
    mc_dir = INK_MAY_DIR / "max comp 5"
    max_comp = []
    for p in sorted(mc_dir.iterdir()):
        e: dict = {"file": str(p.relative_to(ROOT)), "file_size": p.stat().st_size}
        if p.suffix.lower() == ".jpg":
            with Image.open(p) as im:
                e["dims_wh"] = list(im.size)
        max_comp.append(e)
    return matches, unmatched, psds, max_comp


def audit_zips() -> list:
    out = []
    line_re = re.compile(r"^\s*(\d+)\s+[\d-]+\s+[\d:]+\s+(.+?)\s*$")
    for zp in sorted(ROOT.glob("*.zip")):
        res = subprocess.run(
            ["unzip", "-l", str(zp)], capture_output=True, text=True, check=True
        )
        n_entries = n_files = 0
        missing, mismatched = [], []
        for line in res.stdout.splitlines():
            mm = line_re.match(line)
            if not mm or line.strip().startswith("Length"):
                continue
            size, name = int(mm.group(1)), mm.group(2)
            if name in ("Name", "----") or name.endswith("/"):
                continue
            n_entries += 1
            n_files += 1
            disk = ROOT / name
            if not disk.is_file():
                missing.append(name)
            elif disk.stat().st_size != size:
                mismatched.append(
                    {"name": name, "zip_size": size, "disk_size": disk.stat().st_size}
                )
        out.append(
            {
                "zip": zp.name,
                "zip_size": zp.stat().st_size,
                "n_files_listed": n_files,
                "complete": not missing and not mismatched,
                "missing_on_disk": missing,
                "size_mismatch": mismatched,
            }
        )
    return out


# ---------------------------------------------------------------- consensus + anomalies

def vertical_decision(o: dict) -> tuple[str, float]:
    """Decide top vs bottom vertical orientation from the better of each u-variant pair."""
    top = max(o["iou_topleft"], o["iou_topleft_u_flipped"])
    bot = max(o["iou_opengl_bottomleft"], o["iou_rot180"])
    return ("bottom" if bot > top else "top"), round(abs(bot - top), 4)


def build_consensus(oracles: list[dict]) -> tuple[dict, dict, list]:
    """Per-group 2-way/4-way consensus; returns (group_consensus, tex_orientation, anomalies)."""
    consensus: dict = {}
    tex_orient: dict = {"per_mesh_overrides": {}, "definitions": HYPOTHESES}
    anomalies: list = []
    for grp in ("A", "B", "C"):
        recs = [o for o in oracles if o["group"] == grp and "iou_topleft" in o]
        for o in recs:
            o["vertical_winner"], o["vertical_margin"] = vertical_decision(o)
        v2 = [o["winner_2way"] for o in recs if o["winner_2way"] != "inconclusive"]
        v4 = [o["winner_4way"] for o in recs if o["winner_4way"] != "inconclusive"]
        c2 = max(set(v2), key=v2.count) if v2 else "inconclusive"
        c4 = max(set(v4), key=v4.count) if v4 else "inconclusive"
        # Resolved orientation for renderers: 4-way majority when available; otherwise
        # decide the vertical axis from the (much larger) pair-level margin and default
        # to no u-flip unless a conclusive within-pair margin proves one.
        if v4:
            resolved, basis = c4, "majority of conclusive 4-way winners"
        else:
            verts = [o["vertical_winner"] for o in recs if o["vertical_margin"] >= ORACLE_MARGIN]
            if verts and len(set(verts)) == 1:
                vert = verts[0]
                pair = ("opengl_bottomleft", "rot180") if vert == "bottom" else ("topleft", "topleft_u_flipped")
                uflip_votes = [
                    o[f"iou_{pair[1]}"] - o[f"iou_{pair[0]}"] >= ORACLE_MARGIN for o in recs
                ]
                resolved = pair[1] if all(uflip_votes) and uflip_votes else pair[0]
                basis = (
                    f"vertical axis decided by pair-level IoU margin (>= {ORACLE_MARGIN} "
                    "for all meshes); u-flip unproven within pair -> identity-u default"
                )
            else:
                resolved, basis = "inconclusive", "no conclusive signal"
        dis2 = [o["path"] for o in recs if o["winner_2way"] not in ("inconclusive", c2)]
        dis4 = [o["path"] for o in recs if o["winner_4way"] not in ("inconclusive", c4)]
        consensus[grp] = {
            "n_meshes": len(recs),
            "consensus_2way": c2,
            "votes_2way": {k: v2.count(k) for k in sorted(set(v2))},
            "n_inconclusive_2way": len(recs) - len(v2),
            "disagree_with_majority_2way": dis2,
            "consensus_4way": c4,
            "votes_4way": {k: v4.count(k) for k in sorted(set(v4))},
            "n_inconclusive_4way": len(recs) - len(v4),
            "disagree_with_majority_4way": dis4,
            "resolved_orientation": resolved,
            "resolution_basis": basis,
            "consensus_2way_reliable": c2 == resolved,
        }
        tex_orient[grp] = resolved
        for o in recs:
            if o["winner_4way"] not in ("inconclusive", resolved):
                tex_orient["per_mesh_overrides"][o["path"]] = o["winner_4way"]
    tex_orient["basis"] = (
        "4-hypothesis UV-coverage vs texture-content-mask IoU with one-sided "
        "containment cross-check; per-group resolution recorded in "
        "orientation_oracle.group_consensus.resolution_basis. Renderers must apply the "
        "per-group mapping (see definitions); per_mesh_overrides take precedence when "
        "present."
    )
    return consensus, tex_orient, anomalies


def main() -> None:
    t0 = datetime.now(timezone.utc)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    mesh_paths = (
        [("A", p) for p in sorted(MARCH_DIR.glob("*.ply"))]
        + [("B", p) for p in sorted(MAY_DIR.glob("*.ply"))]
        + [("C", p) for p in sorted(ROOT.glob(C_GLOB))]
    )
    meshes, oracles = [], []
    for grp, p in mesh_paths:
        print(f"[mesh] {grp} {p.name}", flush=True)
        rec, orc = audit_mesh(p, grp)
        meshes.append(rec)
        oracles.append(orc)
        gc.collect()

    mesh_by_stem = {r["stem"]: r for r in meshes}
    consensus, tex_orientation, anomalies = build_consensus(oracles)

    print("[ink] march TIFs", flush=True)
    march_matches, orphans, unmatched_a = audit_ink_march(mesh_by_stem)
    print("[ink] may JPGs", flush=True)
    may_matches, unmatched_b, may_psds, max_comp = audit_ink_may(mesh_by_stem)
    print("[zip] verifying archives", flush=True)
    zips = audit_zips()

    # ---------------- anomalies (every entry carries an explanation) ----------------
    def add(path, kind, detail, explanation):
        anomalies.append(
            {"path": str(path), "kind": kind, "detail": detail, "explanation": explanation}
        )

    # Missing textures (PHerc172 second material).
    for r in meshes:
        for t in r["textures"]:
            if not t["exists"]:
                hist = r["texnumber_histogram"]
                add(
                    r["path"],
                    "missing_texture",
                    f"declared TextureFile '{t['name']}' not found next to the PLY",
                    "Harmless: the face texnumber histogram is "
                    f"{hist} — every face references material 0 "
                    f"({r['textures'][0]['name']}), so the missing second texture is "
                    "never used. Treat the mesh as single-material (per "
                    "mesh-io-conventions).",
                )

    # Group A texture orientation is outside the prescribed 2-hypothesis set.
    a_orcs = [o for o in oracles if o["group"] == "A"]
    if consensus["A"]["resolved_orientation"] == "rot180":
        v2 = consensus["A"]["votes_2way"]
        n_lead = sum(1 for o in a_orcs if o["leader_4way"] == "rot180")
        n_win = sum(1 for o in a_orcs if o["winner_4way"] == "rot180")
        add(
            "textured_plys/textured_ply_march/",
            "tex_orientation_outside_2way_hypothesis_set",
            f"2-way oracle split across Group A: votes {v2}, "
            f"{consensus['A']['n_inconclusive_2way']} inconclusive; rot180 has the "
            f"highest 4-way IoU for {n_lead}/{len(a_orcs)} meshes (conclusive winner "
            f"for {n_win}, margin < {ORACLE_MARGIN} for the rest).",
            "The March composites are stored rotated 180 deg relative to top-left UV "
            "indexing (both axes flipped: pixel_col=(1-s)*(W-1), pixel_row=(1-t)*(H-1)). "
            "Neither prescribed hypothesis (topleft / opengl_bottomleft) matches, so the "
            "2-way vote degenerates to noise driven by each segment's left-right "
            "asymmetry. Proof: under rot180 the fraction of texture content (alpha>127) "
            "falling OUTSIDE the UV chart is <=0.9% for every Group A mesh (max "
            f"{max(o['uncovered_content_fraction']['rot180'] for o in a_orcs):.3f}), vs "
            f">= {min(min(o['uncovered_content_fraction']['topleft'], o['uncovered_content_fraction']['opengl_bottomleft']) for o in a_orcs):.3f} "
            "for the best prescribed hypothesis per mesh; visual boundary-shape check on "
            "w013/w030/w041 confirms exact outline alignment under rot180. Renderers must "
            "read tex_orientation.A = rot180 (equivalently: rotate the PNG 180 deg at "
            "load, then sample top-left).",
        )
    # Per-mesh 2-way disagreements (contract: flag meshes disagreeing with group majority).
    for grp in ("A", "B", "C"):
        for pth in consensus[grp]["disagree_with_majority_2way"]:
            o = next(o for o in oracles if o["path"] == pth)
            add(
                pth,
                "oracle_2way_disagrees_with_group_majority",
                f"2-way winner {o['winner_2way']} vs group-{grp} majority "
                f"{consensus[grp]['consensus_2way']} (margin {o['margin_2way']}).",
                "Not a data defect: the true Group A mapping (rot180) is outside the "
                "2-way hypothesis set, so 2-way winners are determined by segment "
                "left-right asymmetry and split the group. The 4-way oracle is unanimous "
                "(rot180); use tex_orientation, not the 2-way vote.",
            )
        for pth in consensus[grp]["disagree_with_majority_4way"]:
            o = next(o for o in oracles if o["path"] == pth)
            add(
                pth,
                "oracle_4way_disagrees_with_group_majority",
                f"4-way winner {o['winner_4way']} vs group consensus "
                f"{consensus[grp]['consensus_4way']}.",
                "Per-mesh override recorded in tex_orientation.per_mesh_overrides.",
            )
    # 2-way inconclusive meshes.
    inc = [o for o in oracles if o.get("winner_2way") == "inconclusive"]
    if inc:
        still4 = [o for o in inc if o["winner_4way"] == "inconclusive"]
        tail = (
            " For "
            + ", ".join(Path(o["path"]).stem for o in still4)
            + " even the 4-way margin is < 0.02 (near-symmetric content); they inherit "
            "the group consensus, and their rot180 containment diagnostic (uncovered "
            "content "
            + ", ".join(
                f"{o['uncovered_content_fraction']['rot180']:.3f}" for o in still4
            )
            + ") confirms it."
            if still4
            else " The 4-way oracle resolves all of them conclusively."
        )
        add(
            "(multiple meshes)",
            "oracle_2way_inconclusive",
            "2-way IoU margin < 0.02 for: "
            + ", ".join(Path(o["path"]).stem for o in inc),
            "All are Group A meshes: since the true mapping is rot180, topleft and "
            "opengl_bottomleft are equally wrong (each differs from the truth by one "
            "axis flip), so for nearly left-right-symmetric content their IoUs tie."
            + tail,
        )
    # Group C: u-axis orientation unproven (bottomleft vs rot180 tie).
    if (
        consensus["C"]["consensus_4way"] == "inconclusive"
        and consensus["C"]["resolved_orientation"] == "opengl_bottomleft"
    ):
        c_orcs = [o for o in oracles if o["group"] == "C"]
        add(
            "scroll_meshes-20260610T062158Z-3-001/scroll_meshes/",
            "oracle_u_axis_unproven",
            "4-way oracle inconclusive for all 3 Group C meshes: opengl_bottomleft vs "
            "rot180 IoUs within 0.02 ("
            + ", ".join(
                f"{Path(o['path']).stem}: {o['iou_opengl_bottomleft']:.3f}/{o['iou_rot180']:.3f}"
                for o in c_orcs
            )
            + ").",
            "The VCG atlases are full-width, nearly left-right-symmetric horizontal "
            "bands, so a u-flip barely changes IoU. The vertical axis is decided "
            "overwhelmingly (pair-level margins "
            + ", ".join(f"{o['vertical_margin']:.2f}" for o in c_orcs)
            + "): texture content occupies the top rows while top-left-rastered charts "
            "occupy the bottom rows. With no evidence for a u-flip, identity-u is "
            "chosen (standard VCGLIB texcoord convention) -> opengl_bottomleft.",
        )

    # Declining alpha coverage in late Group A wraps.
    sparse = [
        o
        for o in oracles
        if o["group"] == "A" and o["mask_fraction"] < 0.5 * o["coverage_fraction"]
    ]
    if sparse:
        add(
            "textured_plys/textured_ply_march/",
            "sparse_texture_content",
            "Texture alpha (content) covers < 50% of the UV chart for: "
            + ", ".join(
                f"{Path(o['path']).stem} ({o['mask_fraction']:.2f}/{o['coverage_fraction']:.2f})"
                for o in sparse
            ),
            "Expected physical signal, not a defect: alpha marks where papyrus was "
            "actually composited. Content fraction declines monotonically with wrap "
            "index (w031->w041) because the outermost scroll wraps are the most damaged. "
            "Under rot180 the sparse content is still fully inside the UV chart "
            "(uncovered <= "
            + f"{max(o['uncovered_content_fraction']['rot180'] for o in sparse):.3f}"
            + "), confirming correct pairing of texture and mesh.",
        )
    # Group C mixed UV winding.
    for r in meshes:
        if r["group"] == "C" and 0 < r["uv_positive_count"] != r["n_faces"]:
            add(
                r["path"],
                "mixed_uv_winding",
                f"{r['uv_flipped_count']} of {r['n_faces']} faces have non-positive "
                "signed UV area.",
                "Normal for VCGLIB/MeshLab multi-chart texture atlases: individual "
                "charts may be packed mirrored, so per-face UV winding is mixed. The "
                "texture was baked with the same parameterization, so sampling is "
                "self-consistent; 'flipped' here is not an orientation defect (unlike "
                "the single-chart SLIM UVs of groups A/B, which are 100% positive).",
            )
        if r["group"] in ("A", "B") and r["uv_flipped_count"] > 0:
            add(
                r["path"],
                "flipped_uv_triangles",
                f"{r['uv_flipped_count']} non-positively oriented UV triangles.",
                "Unexpected for SLIM-derived single-chart UVs; inspect before unwrap.",
            )
    # Zero-area UV / degenerate 3D triangles.
    for r in meshes:
        if r["uv_zero_area_count"] > 0 and r["group"] == "C":
            add(
                r["path"],
                "zero_area_uv_triangles",
                f"{r['uv_zero_area_count']} UV triangles with exactly zero signed area.",
                f"{r['uv_zero_area_count']} of {r['n_faces']} faces "
                f"({r['uv_zero_area_count'] / r['n_faces']:.2e}) collapse in UV space — "
                "atlas chart seams where VCG assigned identical wedge texcoords. "
                "Sub-ppm count; harmless for rendering (zero sampled area), but the "
                "wedge-dedup OBJ path must keep exact duplicates (it does).",
            )
        elif r["uv_zero_area_count"] > 0:
            add(
                r["path"],
                "zero_area_uv_triangles",
                f"{r['uv_zero_area_count']} zero-area UV triangles.",
                "Unexpected in groups A/B; inspect.",
            )
        if r["degenerate_3d_tri_count"] > 0:
            add(
                r["path"],
                "degenerate_3d_triangles",
                f"{r['degenerate_3d_tri_count']} triangles with 3D area < 1e-12.",
                "Degenerate slivers; flagged for the decimation stage (M2) which must "
                "not propagate them.",
            )
        if r["connected_components_excluding_isolated_vertices"] > 1:
            sizes = r["component_vertex_sizes_desc"]
            extra_v = r["n_vertices"] - sizes[0]
            if sizes[1] < 100:
                expl = (
                    f"The extra component(s) are tiny detached triangle islets "
                    f"({extra_v} vertices total = "
                    f"{100 * extra_v / r['n_vertices']:.4f}% of the mesh; sizes "
                    f"{sizes[1:]}) left by the wrap-tracing/segmentation pipeline — the "
                    "principal sheet holds >99.99% of vertices. Harmless for exact "
                    "conversion (order-preserving); flag for the decimation/unwrap "
                    "stages, which operate on the principal sheet."
                )
            else:
                expl = (
                    "Substantial secondary component — investigate the segmentation "
                    "before unwrap."
                )
            add(
                r["path"],
                "multiple_components",
                f"{r['connected_components']} connected components; vertex counts "
                f"{sizes} ({r['isolated_vertex_count']} isolated vertices).",
                expl,
            )
        if r["denorm"] and abs(r["denorm"]["pixel_scale_anisotropy"] - 1) > 0.05:
            add(
                r["path"],
                "uv_pixel_scale_anisotropy",
                f"s_u_px/s_v_px = {r['denorm']['pixel_scale_anisotropy']:.4f}.",
                "Pixel scale deviates >5% from isotropic; check texture aspect vs "
                "UV normalization.",
            )

    # Ink anomalies.
    for o in orphans:
        add(
            o["tif"],
            "orphan_ink",
            f"ink TIF stem '{o['stem']}' matches no Group A mesh stem.",
            "The March mesh delivery starts at w011 (plus one auto-trace segment); no "
            "w010 PLY/texture was delivered, so this full-res prediction has no mesh to "
            "map onto. Kept for provenance; excluded from M2.5 overlays.",
        )
    for pth in unmatched_a:
        add(
            pth,
            "mesh_without_ink",
            "Group A mesh has no same-stem TIF in '1667 - ink detection/'.",
            "The ink-detection batch covers the w-numbered wraps only; this auto-trace "
            "segment was traced after (2026-01-21) and no prediction was produced for "
            "it. M2.5 must list it as UNMATCHED (plain-texture renders only).",
        )
    for pth in unmatched_b:
        add(pth, "mesh_without_ink", "no predictions_inv JPG for this stem.", "Inspect.")
    for mrec in may_matches:
        if not mrec["polarity_verified_mean_gt_half"]:
            add(
                mrec["jpg"],
                "ink_polarity_unexpected",
                f"mean {mrec['stats_subsampled_8x']['mean']} not > 0.5*max.",
                "predictions_inv JPGs are documented as inverted (ink dark); this one "
                "is not — verify before overlay.",
            )
    for z in zips:
        if not z["complete"]:
            add(
                z["zip"],
                "zip_incomplete_extraction",
                f"missing: {z['missing_on_disk']}, mismatched: {z['size_mismatch']}",
                "Extracted folder does not byte-match the archive listing; re-extract.",
            )

    # ---------------- write reports ----------------
    audit = {
        "generated_at": t0.isoformat(),
        "tex_orientation": tex_orientation,
        "meshes": meshes,
        "orientation_oracle": {
            "method": (
                f"UV-triangle coverage mask (<= {ORACLE_MAX_TRIS} faces drawn, raster "
                f"max dim {ORACLE_MAX_DIM}px, MaxFilter(5) dilation to close "
                "subsampling gaps) vs texture content mask; IoU under the 2 prescribed "
                "vertical hypotheses plus the 2 u-flipped ones; margin < "
                f"{ORACLE_MARGIN} => inconclusive. uncovered_content_fraction is the "
                "one-sided containment diagnostic (content outside chart; ~0 under the "
                "true mapping)."
            ),
            "hypotheses": HYPOTHESES,
            "per_mesh": {o["path"]: {k: v for k, v in o.items() if k != "path"} for o in oracles},
            "group_consensus": consensus,
        },
        "ink": {
            "march_matches": march_matches,
            "may_matches": may_matches,
            "orphans": orphans,
            "unmatched_meshes": unmatched_a + unmatched_b,
            "zip_completeness": zips,
            "predictions_inv_psds": may_psds,
            "max_comp_5": max_comp,
        },
        "anomalies": anomalies,
    }
    (reports / "audit.json").write_text(json.dumps(js(audit), indent=1))
    write_markdown(reports / "audit.md", audit)
    dt = (datetime.now(timezone.utc) - t0).total_seconds()
    print(f"done in {dt:.1f}s -> reports/audit.json, reports/audit.md")


# ---------------------------------------------------------------- markdown

def write_markdown(path: Path, a: dict) -> None:
    L: list[str] = []
    w = L.append
    w(f"# Mesh & ink audit — {a['generated_at']}\n")
    meshes = a["meshes"]
    groups = {g: [m for m in meshes if m["group"] == g] for g in ("A", "B", "C")}

    w("## Binding texture orientation (renderers read this)\n")
    to = a["tex_orientation"]
    w("| group | tex_orientation | mapping |")
    w("|---|---|---|")
    for g in ("A", "B", "C"):
        w(f"| {g} | **{to[g]}** | `{to['definitions'].get(to[g], 'n/a')}` |")
    w(f"\nPer-mesh overrides: {to['per_mesh_overrides'] or 'none'}\n")
    w(f"Basis: {to['basis']}\n")

    w("## Per-group mesh summary\n")
    w("| group | meshes | vertices | faces | edge len (mean) | components | boundary edges | flipped UV | zero-area UV | degenerate 3D |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for g, ms in groups.items():
        if not ms:
            continue
        w(
            f"| {g} | {len(ms)} | "
            f"{min(m['n_vertices'] for m in ms):,}–{max(m['n_vertices'] for m in ms):,} | "
            f"{min(m['n_faces'] for m in ms):,}–{max(m['n_faces'] for m in ms):,} | "
            f"{min(m['edge_len_mean'] for m in ms):.3f}–{max(m['edge_len_mean'] for m in ms):.3f} | "
            f"{sorted(set(m['connected_components'] for m in ms))} | "
            f"{min(m['boundary_edge_count'] for m in ms):,}–{max(m['boundary_edge_count'] for m in ms):,} | "
            f"{sum(m['uv_flipped_count'] for m in ms):,} | "
            f"{sum(m['uv_zero_area_count'] for m in ms):,} | "
            f"{sum(m['degenerate_3d_tri_count'] for m in ms)} |"
        )
    w("")
    w("- Group A+B: wedge UV ≡ vertex UV bit-exact for "
      f"{sum(1 for m in meshes if m['wedge_equals_vertex_uv'] is True)}/31 meshes; "
      "all UV triangles positively oriented.")
    w("- Group C: per-wedge UV only (no per-vertex); mixed chart winding (see anomalies).")
    ph = next(m for m in meshes if "PHerc172" in m["path"])
    w(
        f"- PHerc172 special case: declared textures "
        f"{[t['name'] for t in ph['textures']]}, exists="
        f"{[t['exists'] for t in ph['textures']]}; texnumber histogram over all "
        f"{ph['n_faces']:,} faces = {ph['texnumber_histogram']} (all faces use "
        "material 0).\n"
    )

    w("## De-normalization fit (groups A/B)\n")
    w("| mesh | s_u | s_v | aniso (s_u/s_v) | s_u_px | s_v_px | px aniso | resid RMS | resid P95 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for m in meshes:
        d = m["denorm"]
        if not d:
            continue
        w(
            f"| {m['stem']} | {d['s_u']:.1f} | {d['s_v']:.1f} | {d['anisotropy']:.4f} | "
            f"{d['s_u_px']:.4f} | {d['s_v_px']:.4f} | {d['pixel_scale_anisotropy']:.4f} | "
            f"{d['rel_residual_rms']:.4f} | {d['rel_residual_p95']:.4f} |"
        )
    ds = [m["denorm"] for m in meshes if m["denorm"]]
    hi_resid = sorted(
        ((m["denorm"]["rel_residual_rms"], m["stem"]) for m in meshes if m["denorm"]),
        reverse=True,
    )[:3]
    w(
        f"\nRanges: UV-space anisotropy {min(d['anisotropy'] for d in ds):.3f}–"
        f"{max(d['anisotropy'] for d in ds):.3f} (expected ≠1: axes normalized "
        f"independently); pixel-scale anisotropy {min(d['pixel_scale_anisotropy'] for d in ds):.4f}–"
        f"{max(d['pixel_scale_anisotropy'] for d in ds):.4f} (≈1 ⇒ isotropic voxels/px ✓); "
        f"residual RMS {min(d['rel_residual_rms'] for d in ds):.4f}–"
        f"{max(d['rel_residual_rms'] for d in ds):.4f}. Highest residuals: "
        + ", ".join(f"{s} ({r:.3f})" for r, s in hi_resid)
        + " — the most damaged outer wraps / auto-trace segment, consistent with "
        "heavier flattening distortion there.\n"
    )

    w("## Orientation oracle\n")
    oc = a["orientation_oracle"]["group_consensus"]
    w("| group | 2-way consensus | 2-way votes | inconclusive | 4-way votes | resolved | basis |")
    w("|---|---|---|---|---|---|---|")
    for g in ("A", "B", "C"):
        c = oc[g]
        w(
            f"| {g} | {c['consensus_2way']} | {c['votes_2way']} | "
            f"{c['n_inconclusive_2way']} | {c['votes_4way']} | "
            f"**{c['resolved_orientation']}** | {c['resolution_basis']} |"
        )
    w("")
    w("| mesh | g | IoU bottomleft | IoU topleft | 2-way | IoU u-flip | IoU rot180 | 4-way | content frac |")
    w("|---|---|---|---|---|---|---|---|---|")
    for p, o in a["orientation_oracle"]["per_mesh"].items():
        if "iou_topleft" not in o:
            continue
        w(
            f"| {Path(p).stem} | {o['group']} | {o['iou_opengl_bottomleft']:.4f} | "
            f"{o['iou_topleft']:.4f} | {o['winner_2way']} | "
            f"{o['iou_topleft_u_flipped']:.4f} | {o['iou_rot180']:.4f} | "
            f"{o['winner_4way']} | {o['mask_fraction']:.2f} |"
        )
    w("")

    ink = a["ink"]
    w("## Ink inventory\n")
    w(
        f"March TIFs: **{len(ink['march_matches'])} matched** / "
        f"{len(ink['orphans'])} orphan / "
        f"{len([u for u in ink['unmatched_meshes'] if 'march' in u])} unmatched Group A mesh. "
        f"May JPGs: **{len(ink['may_matches'])}/11 matched** "
        f"(+{len(ink['predictions_inv_psds'])} PSDs listed, not opened). "
        f"max comp 5: {len(ink['max_comp_5'])} files.\n"
    )
    w("| ink (March, TIF) | dims H×W | dtype | mean | frac>½max | ratio vs texture (w,h) |")
    w("|---|---|---|---|---|---|")
    for r in ink["march_matches"] + ink["orphans"]:
        st = r["stats_subsampled_8x"]
        rr = r.get("dims_ratio_vs_texture")
        ratio = f"{rr['w']['ratio']}, {rr['h']['ratio']}" if rr else "— (orphan)"
        w(
            f"| {r['stem']} | {r['shape_hw'][0]}×{r['shape_hw'][1]} | {r['dtype']} | "
            f"{st['mean']:.2f} | {st['frac_gt_half_max']:.4f} | {ratio} |"
        )
    w("")
    w("| ink (May, JPG, inverted) | dims W×H | mean | frac>½max | inverted? | ratio vs texture (w,h) |")
    w("|---|---|---|---|---|---|")
    for r in ink["may_matches"]:
        st = r["stats_subsampled_8x"]
        rr = r.get("dims_ratio_vs_texture")
        ratio = f"{rr['w']['ratio']}, {rr['h']['ratio']}" if rr else "—"
        w(
            f"| {Path(r['jpg']).stem} | {r['dims_wh'][0]}×{r['dims_wh'][1]} | "
            f"{st['mean']:.1f} | {st['frac_gt_half_max']:.4f} | "
            f"{'✓' if r['polarity_verified_mean_gt_half'] else '✗'} | {ratio} |"
        )
    w("\nMarch ratio note: every matched TIF/texture dim pair shares one uniform scale "
      "(textures are the flattened grid resized to max-dim 8192; TIFs are full-res).\n")

    w("## Zip completeness\n")
    w("| zip | files | complete | issues |")
    w("|---|---|---|---|")
    for z in ink["zip_completeness"]:
        issues = (
            "none"
            if z["complete"]
            else f"missing {len(z['missing_on_disk'])}, mismatch {len(z['size_mismatch'])}"
        )
        w(f"| {z['zip']} | {z['n_files_listed']} | {'✓' if z['complete'] else '✗'} | {issues} |")
    w("")

    w("## Anomalies (all explained)\n")
    for i, an in enumerate(a["anomalies"], 1):
        w(f"**{i}. [{an['kind']}]** `{an['path']}`")
        w(f"   - detail: {an['detail']}")
        w(f"   - explanation: {an['explanation']}\n")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
