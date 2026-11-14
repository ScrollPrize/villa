"""Per-site features for Parting, from one 128^3 CT window and base dinovol's token grid of it (pure numpy).

A site has three points, a, a2 (same layer as a) and b (next layer), each with a unit normal, in xyz voxels of the
volume read; its window is the 128^3 cube centered on the site's `centre` field. Per site this makes:

  tok      (3, 288)        the token grid sampled trilinearly at a, a2 and b
  ct_n     (3, 17)         the CT along each point's own normal, -8 to +8 voxels, after a per-window robust scale
  ct_seg   (2, 16)         the same scaled CT at 16 evenly spaced points from a to a2, and from a to b
  p_tok    (2, 2, 16)      along those 16 points, the token cosine of each sample with the sample at a and at x
  scores   dino_base       [same, cross]: token cosine of (a, a2) and of (a, b), base dinovol's own judgment
           ct_valley       [same, cross]: the darkest raw CT value on the segment over the mean of its two ends

Features are stored as float16 (`save_run`), as in the frozen run, and the model reads them back as float32.
"""
import json
import os
from pathlib import Path

import numpy as np

PROFILE = tuple(range(-8, 9))
SEG = 16
POINTS = ("a", "a2", "b")
GROUPS = {"tok": (3, 288), "ct_n": (3, len(PROFILE)), "ct_seg": (2, SEG), "p_tok": (2, 2, SEG)}
SCORES = ("dino_base", "ct_valley")
STRIDE = 8


def segment(p, q, size=128):
    """Voxel indices along the straight segment p to q at about one voxel steps, endpoints included."""
    n = max(2, int(np.ceil(np.linalg.norm(np.asarray(q) - np.asarray(p)))) + 1)
    t = np.linspace(0, 1, n)[:, None]
    return np.clip(np.rint(np.asarray(p)[None] * (1 - t) + np.asarray(q)[None] * t), 0, size - 1).astype(int)


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def trilinear(grid, p_zyx, stride=STRIDE):
    """(C, G, G, G) token grid sampled at a window voxel position, token i centered at stride * i + (stride - 1) / 2."""
    G = grid.shape[1]
    u = np.asarray(p_zyx, float)
    f = (u - (stride - 1) / 2) / stride
    f = np.clip(f, 0, G - 1 - 1e-6)
    i0 = np.floor(f).astype(int)
    d = f - i0
    out = np.zeros(grid.shape[0])
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (d[0] if dz else 1 - d[0]) * (d[1] if dy else 1 - d[1]) * (d[2] if dx else 1 - d[2])
                out += w * grid[:, i0[0] + dz, i0[1] + dy, i0[2] + dx]
    return out


def _line(vol, p, q, n):
    """vol sampled (nearest voxel) at n evenly spaced points from p to q, endpoints included."""
    t = np.linspace(0, 1, n)[:, None]
    ix = np.clip(np.rint(np.asarray(p)[None] * (1 - t) + np.asarray(q)[None] * t), 0, vol.shape[0] - 1).astype(int)
    return vol[ix[:, 0], ix[:, 1], ix[:, 2]].astype(np.float32)


def window_lo(site):
    """The window's low corner, zyx voxels."""
    return np.asarray(site["centre"])[::-1] - 64


def site_features(site, ct, tokens):
    """({group: array}, {score: [same, cross]}) for one site; ct (128, 128, 128) uint8, tokens (288, 16, 16, 16)."""
    lo = window_lo(site)
    size = ct.shape[0]
    pts = {k: np.asarray(site[k])[::-1] - lo for k in POINTS}
    nrm = {k: np.asarray(site["n_" + k])[::-1] for k in POINTS}
    out = {k: np.zeros(s[:1] + (tokens.shape[0],) if k == "tok" else s, np.float32) for k, s in GROUPS.items()}
    ctf = ct.astype(np.float32)
    lo_, hi_ = np.percentile(ctf, (1, 99))
    ctf = (ctf - np.median(ctf)) / (hi_ - lo_ + 1e-6)                  # per-window robust scale: scans differ
    for i, k in enumerate(POINTS):
        out["tok"][i] = trilinear(tokens, pts[k])
        prof = np.clip(np.rint(pts[k][None] + np.asarray(PROFILE, float)[:, None] * nrm[k][None]), 0,
                       size - 1).astype(int)
        out["ct_n"][i] = ctf[prof[:, 0], prof[:, 1], prof[:, 2]]
    scores = {"dino_base": [], "ct_valley": []}
    for j, k in enumerate(("a2", "b")):
        out["ct_seg"][j] = _line(ctf, pts["a"], pts[k], SEG)
        t = np.linspace(0, 1, SEG)[:, None]
        line = pts["a"][None] * (1 - t) + pts[k][None] * t
        tk = np.stack([trilinear(tokens, q) for q in line])
        tk = tk / (np.linalg.norm(tk, axis=1, keepdims=True) + 1e-8)
        out["p_tok"][j] = [tk @ tk[0], tk @ tk[-1]]
        s = segment(pts["a"], pts[k], size)
        ctv = ct[s[:, 0], s[:, 1], s[:, 2]].astype(float)
        scores["dino_base"].append(cosine(trilinear(tokens, pts["a"]), trilinear(tokens, pts[k])))
        scores["ct_valley"].append(float(ctv.min() / (0.5 * (ctv[0] + ctv[-1]) + 1e-6)))
    return out, scores


def save_run(folder, sites, feats, scores, meta):
    """A run folder: results.jsonl (one row per site: id, pair, stratum, scores), features.npz (ids plus one float16
    array per group) and run.json (meta). Written atomically; rows in site id order."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    by = {s["id"]: s for s in sites}
    ids = sorted(feats)
    rows = [json.dumps({"id": i, "pair": by[i]["pair"], "stratum": by[i]["stratum"], "scores": scores[i]})
            for i in ids]
    tmp = folder / "results.jsonl.tmp"
    tmp.write_text("\n".join(rows) + ("\n" if rows else ""))
    os.replace(tmp, folder / "results.jsonl")
    arrs = {k: np.stack([feats[i][k] for i in ids]).astype(np.float16) for k in GROUPS} if ids else {}
    tmp = folder / "features.tmp.npz"
    np.savez(tmp, ids=np.asarray(ids, np.int64), **arrs)
    os.replace(tmp, folder / "features.npz")
    (folder / "run.json").write_text(json.dumps(meta, indent=1))


def load_done(folder):
    """(features, scores) of a run folder written by save_run, for resuming; empty when there is none."""
    folder = Path(folder)
    if not (folder / "features.npz").exists() or not (folder / "results.jsonl").exists():
        return {}, {}
    z = np.load(folder / "features.npz")
    feats = {int(i): {k: z[k][n].astype(np.float32) for k in GROUPS} for n, i in enumerate(z["ids"])}
    scores = {}
    for line in (folder / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            scores[r["id"]] = {k: r["scores"][k] for k in SCORES}
    keep = set(feats) & set(scores)
    return {i: feats[i] for i in keep}, {i: scores[i] for i in keep}
