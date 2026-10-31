"""JSON report and side-by-side PNG."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _stretch(a, m, lo=1, hi=99):
    v = a[m] if m.any() else a.ravel()
    l, h = np.percentile(v, [lo, hi])
    out = np.clip((a - l) / max(h - l, 1e-9), 0, 1)
    out[~m] = 0
    return out


def _to_json(o):
    if isinstance(o, dict):
        return {str(k): _to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(res, path, extra=None):
    d = {"verdict": res.verdict, "summary": res.summary, "warnings": res.warnings,
         "numbers": res.numbers, "config": res.config, "nulls": res.nulls}
    if extra:
        d["inputs"] = extra
    Path(path).write_text(json.dumps(_to_json(d), indent=2))


def write_png(res, path, title="", panel_max=900):
    A = res.arrays
    m = A["mask"]
    um = res.numbers["analysis_um"]
    p = _stretch(A["pred"], m)
    r = _stretch(A["ref"], m)
    H, W = p.shape
    s = min(1.0, panel_max / max(H, W))
    size = (max(1, int(W * s)), max(1, int(H * s)))

    def im(x):
        return Image.fromarray((x * 255).astype(np.uint8)).resize(size, Image.BOX).convert("RGB")

    ov = np.zeros((H, W, 3), np.float32)
    ov[..., 0] = r
    ov[..., 2] = r
    ov[..., 1] = p
    ovi = Image.fromarray((ov * 255).astype(np.uint8)).resize(size, Image.BOX)
    panels = [("prediction", im(p)), ("reference (registered)", im(r)), ("overlay: green=pred, magenta=ref", ovi)]
    pad, top, bottom = 10, 110, 190
    Wt = 3 * size[0] + 4 * pad
    canvas = Image.new("RGB", (Wt, top + size[1] + bottom), "white")
    d = ImageDraw.Draw(canvas)
    col = {"PASS": (0, 140, 0), "WEAK": (200, 120, 0), "FAIL": (190, 0, 0)}[res.verdict]
    d.text((pad, 8), title, fill="black", font=_font(20))
    d.text((pad, 36), f"{res.verdict}: {res.summary}", fill=col, font=_font(15))
    n = res.numbers
    d.text((pad, 62), (f"r_bandpass={n['r_bandpass']:+.3f} (null {n['r_bandpass_null_mean']:+.3f}±{n['r_bandpass_null_sd']:.3f}, "
                       f"max {n['r_bandpass_null_max']:+.3f}, z={n['r_bandpass_z']:.1f}, p={n['r_bandpass_p']:.3g})   "
                       f"raw r={n['r_raw']:+.3f}   AUC={n['auc_raw']:.3f}   breakeven F1={n['breakeven_f1']:.3f}   "
                       f"stroke recall={n['stroke_recall']:.2f} / precision={n['stroke_precision']:.2f}"),
           fill="black", font=_font(13))
    d.text((pad, 82), f"grid {um:.2f} um/px, overlap {n['overlap_cm2']:.2f} cm², shift {n['shift_um'][0]:.0f},{n['shift_um'][1]:.0f} um",
           fill="gray", font=_font(13))
    for i, (lab, img) in enumerate(panels):
        x = pad + i * (size[0] + pad)
        canvas.paste(img, (x, top))
        d.text((x, top - 18), lab, fill="black", font=_font(13))
    # 5 mm scale bar
    bar = 5000 / um * s
    y = top + size[1] + 12
    d.rectangle([pad, y, pad + bar, y + 6], fill="black")
    d.text((pad, y + 9), "5 mm", fill="black", font=_font(12))
    # null histogram
    nulls = np.asarray(res.nulls.get("r_bandpass_shift_nulls", []))
    if nulls.size:
        hx0, hy0, hw, hh = pad + 120, y, Wt - 2 * pad - 140, bottom - 40
        lo = min(nulls.min(), n["r_bandpass"], 0) - 0.02
        hi = max(nulls.max(), n["r_bandpass"]) + 0.02
        h, e = np.histogram(nulls, bins=40, range=(lo, hi))
        bw = hw / 40
        for j, c in enumerate(h):
            bh = (hh - 30) * c / max(h.max(), 1)
            d.rectangle([hx0 + j * bw, hy0 + hh - 20 - bh, hx0 + (j + 1) * bw - 1, hy0 + hh - 20], fill=(120, 120, 120))
        xo = hx0 + (n["r_bandpass"] - lo) / (hi - lo) * hw
        d.line([xo, hy0, xo, hy0 + hh - 20], fill=col, width=3)
        d.text((hx0, hy0 + hh - 16), f"{lo:+.2f}", fill="black", font=_font(11))
        d.text((hx0 + hw - 40, hy0 + hh - 16), f"{hi:+.2f}", fill="black", font=_font(11))
        d.text((xo + 4, hy0), "observed", fill=col, font=_font(12))
        d.text((hx0 + hw / 2 - 160, hy0 + hh - 16), "band-pass r of shifted references (search-matched null)",
               fill="gray", font=_font(11))
    canvas.save(path)
