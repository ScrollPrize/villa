"""Face check: is a flat ink prediction one-sided, like writing, or two-sided, like texture?

Ink sits on one face of a papyrus sheet. Fibre texture, cracks and voids look much the same from both faces. When
two or more ink models are run on the same surface volume with ``--direction both``, writing shows up as models
agreeing on one face and not on the other. This module turns that into numbers:

``face_asymmetry``
    IoU of two models' top-3 % pixels on the better face divided by the same IoU on the other face.
``face_consistency``
    The surface is split into half-overlapping tiles. A tile is *strong* for a face when the pair's IoU there is
    above 0.1 and more than 3x the other face's. Score = (strong tiles on the majority face - on the other face) /
    tiles. ``consistent_face`` is ``"tie"`` when neither face wins.
``ink_on_dark`` (with ``--ct``)
    Share of the agreed-ink pixels on the darkest 10 % of the CT (chance = 0.10). Cracks and gaps are dark and
    can trigger ink models.

With ``--ct`` the scores use only pixels where the flattened CT shows the papyrus weave (a sheet check: a
12-orientation Gabor bank; crossing fibres keep orthogonal energy high, cross-cut traces do not).

Typical use, after ``vesuvius.ink_detection.inference.infer ... --direction both`` with two checkpoints::

    python -m vesuvius.ink_detection.face_check seg_seed42.tif seg_dense_native.tif \\
        --ct surface-volume.zarr --out face_check.json

Each positional TIFF is the forward map; its ``_reverse`` sibling (as written by ``infer --direction both``) is
found automatically. With seed42 (scrollprize/ink_9um) and dense_native (Nieuwlaar/ink9um-dense-native), the PHerc. 343
First Letters segment gives face_asymmetry 3.2 and face_consistency 0.52; over 78 traced patches across the 21 First
Letters scrolls the medians are 1.36 and 0.14.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

TOP_FRACTION = 0.03
TILE = 768
STRONG_IOU = 0.1
STRONG_RATIO = 3.0
CROSS_THRESHOLD = 0.31


def _norm(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    v = a[m] if m.any() else a.ravel()
    lo, hi = np.percentile(v, [1, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def _top(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    return (a >= np.quantile(a[m], 1 - TOP_FRACTION)) & m


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def sheet_mask(ct: np.ndarray, win: int = 64, lam: float = 8.0, smooth: int = 129) -> np.ndarray:
    """Pixels where the flattened CT shows crossing papyrus fibres.

    A 12-orientation Gabor bank measures oriented texture energy; crossness is the energy at the orthogonal
    orientation over the dominant one, averaged over ``smooth`` px. Papyrus seen face-on keeps it high (>= 0.31);
    a trace cutting across windings shows one swirling direction and it collapses."""
    import cv2
    img = _norm(ct.astype(np.float32), ct > 0).astype(np.float32)
    ths = np.arange(0, 180, 15)
    energy = []
    for t in ths:
        k = cv2.getGaborKernel((int(4 * lam) | 1,) * 2, sigma=lam * 0.6, theta=np.radians(t), lambd=lam, gamma=0.5, psi=0)
        k -= k.mean()
        r = cv2.filter2D(img, cv2.CV_32F, k)
        energy.append(cv2.blur(r * r, (win, win)))
    e = np.stack(energy)
    i = np.argmax(e, 0)
    emax = np.take_along_axis(e, i[None], 0)[0]
    eorth = np.take_along_axis(e, ((i + len(ths) // 2) % len(ths))[None], 0)[0]
    valid = cv2.blur((img > 0.02).astype(np.float32), (win, win)) > 0.9
    cross = np.where(valid, eorth / np.maximum(emax, 1e-9), 0).astype(np.float32)
    cross = cv2.blur(cross, (smooth, smooth)) / np.maximum(cv2.blur(valid.astype(np.float32), (smooth, smooth)), 1e-3)
    return (cross >= CROSS_THRESHOLD) & valid


def face_scores(fwd: dict[str, np.ndarray], rev: dict[str, np.ndarray], mask: np.ndarray,
                ct: np.ndarray | None = None) -> dict:
    """Scores for every model pair. fwd/rev map model name -> 2D probability map of the same shape as mask."""
    norm = {f: {k: _norm(a, mask) for k, a in d.items()} for f, d in (('forward', fwd), ('reverse', rev))}
    pairs = {}
    for a, b in itertools.combinations(sorted(fwd), 2):
        tops = {f: (_top(norm[f][a], mask), _top(norm[f][b], mask)) for f in norm}
        # whole-patch agreement: strictly above the 97th percentile of the masked pixels
        agree = {f: _iou(norm[f][a][mask] > np.percentile(norm[f][a][mask], 97),
                         norm[f][b][mask] > np.percentile(norm[f][b][mask], 97)) for f in norm}
        best = max(agree, key=agree.get)
        other = 'reverse' if best == 'forward' else 'forward'
        n = {'forward': 0, 'reverse': 0}
        tiles = 0
        H, W = mask.shape
        for y in range(0, H - TILE + 1, TILE // 2):
            for x in range(0, W - TILE + 1, TILE // 2):
                sl = (slice(y, y + TILE), slice(x, x + TILE))
                if mask[sl].mean() < 0.5:
                    continue
                tiles += 1
                t = {f: _iou(tops[f][0][sl], tops[f][1][sl]) for f in tops}
                for f, o in (('forward', 'reverse'), ('reverse', 'forward')):
                    if t[f] > STRONG_IOU and t[f] > STRONG_RATIO * t[o]:
                        n[f] += 1
        major = max(n, key=n.get)
        minor = 'reverse' if major == 'forward' else 'forward'
        r = {'best_face': best, 'agree_best_face': round(agree[best], 4), 'agree_other_face': round(agree[other], 4),
             'face_asymmetry': round(agree[best] / max(agree[other], 1e-3), 3),
             'face_consistency': round((n[major] - n[minor]) / max(tiles, 1), 3),
             'consistent_face': major if n[major] > n[minor] else 'tie',
             'tiles': tiles, 'strong_tiles': n[major], 'strong_tiles_other_face': n[minor]}
        if ct is not None:
            agreed = tops[best][0] & tops[best][1]
            z = (ct - ct[mask].mean()) / (ct[mask].std() + 1e-6)
            r['ink_on_dark'] = round(float(np.mean(z[agreed] < np.quantile(z[mask], 0.1))), 3) if agreed.any() else None
        pairs[f'{a}/{b}'] = r
    return pairs


def _load_map(path: Path) -> np.ndarray:
    import tifffile
    a = tifffile.imread(path).astype(np.float32)
    return a[..., 0] if a.ndim == 3 else a


def _reverse_path(path: Path) -> Path:
    return path.with_name(f'{path.stem}_reverse{path.suffix}')


def _load_ct(spec: str) -> np.ndarray:
    p = Path(spec)
    if p.suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        return tifffile.imread(p).astype(np.float32)
    import zarr
    z = zarr.open(str(p), mode='r')
    a = z if hasattr(z, 'shape') else z['0']
    return np.asarray(a[a.shape[0] // 2]).astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('maps', nargs='+', type=Path, help='forward ink maps (TIFF), one per model; _reverse siblings required')
    ap.add_argument('--ct', help='surface volume (zarr; middle layer used) or a 2D CT TIFF: enables the sheet check '
                                 'and ink_on_dark')
    ap.add_argument('--no-sheet-check', action='store_true', help='with --ct, use all non-zero CT pixels')
    ap.add_argument('--out', type=Path, help='write the JSON here (default: print)')
    a = ap.parse_args(argv)
    if len(a.maps) < 2:
        ap.error('need at least two models')
    fwd = {p.stem: _load_map(p) for p in a.maps}
    rev = {p.stem: _load_map(_reverse_path(p)) for p in a.maps}
    shapes = [m.shape for m in [*fwd.values(), *rev.values()]]
    ct = _load_ct(a.ct) if a.ct else None
    if ct is not None:
        shapes.append(ct.shape)
    h, w = min(s[0] for s in shapes), min(s[1] for s in shapes)
    fwd = {k: v[:h, :w] for k, v in fwd.items()}
    rev = {k: v[:h, :w] for k, v in rev.items()}
    if ct is not None:
        ct = ct[:h, :w]
        mask = (ct > 0) if a.no_sheet_check else sheet_mask(ct)
        if mask.mean() < 0.05:
            mask = ct > 0
    else:
        mask = np.ones((h, w), bool)
    res = {'maps': [str(p) for p in a.maps], 'pixels_used': int(mask.sum()), 'pairs': face_scores(fwd, rev, mask, ct)}
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
