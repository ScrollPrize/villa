"""Reconstruct the same slices with several methods; write float32 TIFFs (previews opt in).

    python evaluation/compare_methods.py --projections PROJ --cache_dir CAL --out OUT     # default_methods.json
    python evaluation/compare_methods.py my_methods.json --out OUT -n 3
    python evaluation/compare_methods.py --projections PROJ --out OUT --roi 1700 2700 1700 2700 --z 1300 1400

Config (JSON):
  projections, cache_dir, device       the scan pair (as for helirec-slices; or --projections / --cache_dir)
  params                               HelicalRecon parameters shared by all methods, e.g. {"filters": []}
                                       (filters: a list of dicts or chain text, e.g. "paganin(delta_beta=1000)")
  slices                               one of {"z": [...]}, {"heights": [...]}, {"fractions": [...]},
                                       {"n": N, "z_range": [Z0, Z1]} (N evenly spaced in [Z0, Z1)),
                                       {"n": N} (N evenly spaced over the middle 2/3 of the object)
                                       z: volume slice index (view height - first doable height)
  roi                                  [X0, X1, Y0, Y1] crop of each slice, or null for whole slices
  window                               "auto" (0.5 / 99.5 percentiles of the first method, per slice, shared by all
                                       methods) or [LO, HI]
  tile                                 longest side of a contact-sheet tile (pixels; slices are box-downsampled)
  methods                              [{"name", "backprojector", "desc"?, "params"?}, ...]; backprojector is a
                                       helirec spec (fbp(...), sirt(...), gridrec(...), ghbp); params override the
                                       shared ones for this method
                                       mode=legacy_reference on a method uses its pinned settings, ignoring shared
                                       corrected-mode parameters. TIFFs retain the method's native output units.

Output (OUT):
  slices/<method>/z<Z>.tif             float32 slice (pipeline units)
  With --previews: slices/<method>/z<Z>.png, sheets/z<Z>.png, contact_sheet.png
  config.json, manifest.json           the resolved config; per method / slice timing and statistics
Existing slice TIFFs are reused only when their provenance fingerprint matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from helirec.cli import DEFAULT_OBJECT_EXTENT, write_png_gray
from helirec.filters import parse_chain
from helirec.recon import HelicalRecon, DEFAULTS

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "default_methods.json")


def load_config(a):
    cfg = json.load(open(a.config))
    if a.n is not None:
        cfg["slices"] = {**{k: v for k, v in cfg.get("slices", {}).items() if k in ("n", "z_range")}, "n": a.n}
    if a.z is not None:
        cfg["slices"] = {"n": cfg.get("slices", {}).get("n", 5), "z_range": a.z}
    if a.roi is not None:
        cfg["roi"] = None if a.roi == [-1] else a.roi
    if a.methods:
        keep = set(a.methods)
        cfg["methods"] = [m for m in cfg["methods"] if m["name"] in keep]
        if not cfg["methods"]:
            sys.exit(f"--methods: none of {sorted(keep)} in the config")
    for k in ("projections", "cache_dir", "device"):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
    if not cfg.get("projections"):
        sys.exit("no projections folder: set it in the config or pass --projections")
    names = [m["name"] for m in cfg["methods"]]
    if len(set(names)) != len(names):
        sys.exit("method names must be unique")
    return cfg


def recon_params(p):
    p = dict(p)
    if isinstance(p.get("filters"), str):
        p["filters"] = parse_chain(p["filters"]) if p["filters"].strip() else []
    return p


def method_params(shared, method):
    overrides = recon_params(method.get("params", {}))
    mode = method.get("mode", overrides.get("mode", shared.get("mode", "corrected")))
    if mode == "legacy_reference":
        return {**overrides, "mode": mode, "backprojector": method["backprojector"]}
    return {**DEFAULTS, **shared, **overrides, "mode": mode, "backprojector": method["backprojector"]}


def display_scale(units):
    if units == "attenuation_cm-1":
        return 1.0
    if units == "reference_units (attenuation / 2)":
        return 2.0
    raise ValueError(f"unsupported output units: {units}")


def slice_heights(r, spec):
    """-> [(z, view height)]"""
    lo, hi = r.lo, r.hi
    if "z" in spec:
        zs = list(spec["z"])
    elif "heights" in spec:
        zs = [h - lo for h in spec["heights"]]
    elif "fractions" in spec:
        zs = [r.slice_info(f).view_height - lo for f in spec["fractions"]]
    else:
        n = int(spec.get("n", 5))
        if "z_range" in spec:
            z0, z1 = spec["z_range"]
            zs = [int(round(0.5 * (z0 + z1 - 1)))] if n == 1 else [int(round(z0 + (z1 - 1 - z0) * i / (n - 1))) for i in range(n)]
        else:   # as helirec-slices: the middle 2/3 of the object extent
            A, B = DEFAULT_OBJECT_EXTENT
            f0, f1 = A + (B - A) / 6, A + (B - A) * 5 / 6
            fr = [0.5 * (f0 + f1)] if n == 1 else [f0 + (f1 - f0) * i / (n - 1) for i in range(n)]
            zs = [r.slice_info(f).view_height - lo for f in fr]
    out = []
    for z in zs:
        if not 0 <= z < hi - lo:
            sys.exit(f"slice z {z} outside the doable range [0, {hi - lo})")
        out.append((int(z), int(z + lo)))
    return out


def window_of(ref, w):
    if w == "auto" or w is None:
        v = ref[np.isfinite(ref) & (ref != 0)]
        return tuple(float(x) for x in np.percentile(v, [0.5, 99.5])) if v.size else (0.0, 1.0)
    return float(w[0]), float(w[1])


def to_u8(img, win):
    lo, hi = win
    return (np.clip((np.nan_to_num(img) - lo) / (hi - lo), 0, 1) * 255 + 0.5).astype(np.uint8)


def thumb(img, tile):
    f = max(1, int(np.ceil(max(img.shape) / tile)))
    if f == 1:
        return img
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def fine_std(a):
    """std of a - its 3x3 mean (noise + finest detail)"""
    k = sum(np.roll(np.roll(a, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)) / 9
    return float(np.std((a - k)[2:-2, 2:-2]))


def _font(size):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def label_band(width, lines, size):
    """8-bit band with text lines, word-wrapped to the width (blank without Pillow)"""
    font = _font(size)
    if font is None:
        return np.zeros((int(size * 1.35) * len(lines) + size // 2, width), np.uint8)
    from PIL import Image, ImageDraw
    room = width - 2 * (size // 3)
    wrapped = []
    for text, shade in lines:
        cur = ""
        for word in text.split(" "):
            nxt = word if not cur else cur + " " + word
            if cur and font.getlength(nxt) > room:
                wrapped.append((cur, shade))
                cur = word
            else:
                cur = nxt
        wrapped.append((cur, shade))
    step = int(size * 1.35)
    im = Image.fromarray(np.zeros((step * len(wrapped) + size // 2, width), np.uint8))
    d = ImageDraw.Draw(im)
    for i, (text, shade) in enumerate(wrapped):
        d.text((size // 3, size // 4 + i * step), text, fill=shade, font=font)
    return np.asarray(im)


def grid(rows, gap=8):
    """rows of equally sized 8-bit tiles -> one image"""
    th = max(t.shape[0] for row in rows for t in row)
    tw = max(t.shape[1] for row in rows for t in row)
    nc = max(len(row) for row in rows)
    out = np.full((len(rows) * (th + gap) - gap, nc * (tw + gap) - gap), 40, np.uint8)
    for i, row in enumerate(rows):
        for j, t in enumerate(row):
            out[i * (th + gap):i * (th + gap) + t.shape[0], j * (tw + gap):j * (tw + gap) + t.shape[1]] = t
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help="methods JSON (default: %(default)s)")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("-n", type=int, default=None, help="number of slices (overrides the config)")
    p.add_argument("--z", nargs=2, type=int, default=None, metavar=("Z0", "Z1"), help="spread the slices over [Z0, Z1)")
    p.add_argument("--roi", nargs="+", type=int, default=None, metavar="X0 X1 Y0 Y1",
                   help="crop [X0, X1) x [Y0, Y1); -1: whole slices")
    p.add_argument("--methods", nargs="+", default=None, help="only these method names")
    p.add_argument("--projections", default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--overwrite", action="store_true", help="recompute slices that already exist")
    p.add_argument("--previews", action="store_true", help="also write PNG slices and contact sheets (default: TIFFs only)")
    p.add_argument("--run-name", default="", help="suffix for metadata and preview sheets, preserving earlier runs")
    a = p.parse_args(argv)
    if a.roi is not None and a.roi != [-1] and len(a.roi) != 4:
        sys.exit("--roi takes X0 X1 Y0 Y1 (or -1)")
    cfg = load_config(a)
    if a.run_name and not all(c.isalnum() or c in "_-" for c in a.run_name):
        p.error("--run-name must contain only letters, digits, underscores or hyphens")
    suffix = "_" + a.run_name if a.run_name else ""
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    if a.previews:
        os.makedirs(os.path.join(out, "sheets" + suffix), exist_ok=True)

    shared = recon_params(cfg.get("params", {}))
    r = HelicalRecon(cfg["projections"], cache_dir=cfg.get("cache_dir"), device=cfg.get("device", "cuda"),
                     params=shared, log=lambda *x: None)
    slices = slice_heights(r, cfg.get("slices", {"n": 5}))
    roi = cfg.get("roi")
    N = r.info["grid_size"]
    if roi is not None and not (0 <= roi[0] < roi[1] <= N and 0 <= roi[2] < roi[3] <= N):
        sys.exit(f"roi must lie within [0, {N}]")
    cfg["resolved_slices"] = [dict(z=z, view_height=h) for z, h in slices]
    json.dump(cfg, open(os.path.join(out, "config" + suffix + ".json"), "w"), indent=1)
    manifest_path = os.path.join(out, "manifest" + suffix + ".json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    print(f"{len(cfg['methods'])} methods x {len(slices)} slices (z {[z for z, _ in slices]}), "
          f"roi {roi or 'whole slice'} -> {out}", flush=True)

    import tifffile
    # 1. reconstruct (method-major: one reconstructor rebuild per method)
    for m in cfg["methods"]:
        d = os.path.join(out, "slices", m["name"])
        os.makedirs(d, exist_ok=True)
        # Reset per-method state: set_params otherwise retains previous overrides.
        r.set_params(**method_params(shared, m))
        r.log = lambda *x: None
        provenance = r.provenance()
        units = provenance["output_units"]
        display_scale(units)  # check known conventions; preserve native TIFF values
        keys = {str(z): fingerprint(dict(provenance=provenance, height=h, roi=roi)) for z, h in slices}
        todo = [(z, h) for z, h in slices if a.overwrite
                or not os.path.exists(os.path.join(d, f"z{z:05d}.tif"))
                or manifest.get(m["name"], {}).get(str(z), {}).get("fingerprint") != keys[str(z)]]
        for z, h in todo:
            t = time.time()
            img = r.reconstruct_heights(h, h + 1, roi=roi)[0].astype(np.float32)
            dt = time.time() - t
            image_path = os.path.join(d, f"z{z:05d}.tif")
            tifffile.imwrite(image_path + ".tmp", img)
            os.replace(image_path + ".tmp", image_path)
            manifest.setdefault(m["name"], {})[str(z)] = dict(
                seconds=round(dt, 2), view_height=h, mean=float(np.mean(img)), std=float(np.std(img)),
                fine_std=fine_std(img), fingerprint=keys[str(z)], provenance=provenance, output_units=units)
            json.dump(manifest, open(manifest_path, "w"), indent=1)
            print(f"  {m['name']:>12s}  z {z:5d}  {dt:7.1f}s", flush=True)

    if not a.previews:
        print("wrote TIFF slices under", os.path.join(out, "slices"))
        return

    # 2. optional images: shared window per slice; per-slice sheets; the overall contact sheet
    tile = int(cfg.get("tile", 1024))
    rows = []
    for z, h in slices:
        imgs = [tifffile.imread(os.path.join(out, "slices", m["name"], f"z{z:05d}.tif"))
                * display_scale(manifest[m["name"]][str(z)]["output_units"]) for m in cfg["methods"]]
        win = window_of(imgs[0], cfg.get("window", "auto"))
        tiles = []
        for m, img in zip(cfg["methods"], imgs):
            u8 = to_u8(img, win)
            write_png_gray(os.path.join(out, "slices", m["name"], f"z{z:05d}.png"), u8)
            th = to_u8(thumb(img, tile), win)
            size = max(14, th.shape[1] // 40)
            band = label_band(th.shape[1], [(f"{m['name']}   z {z}  (h {h})", 255),
                                            (m.get("desc", m["backprojector"]), 190)], size)
            tiles.append(np.concatenate([band, th], 0))
        write_png_gray(os.path.join(out, "sheets" + suffix, f"z{z:05d}.png"), grid([tiles]))
        rows.append(tiles)
        print(f"  z {z}: window [{win[0]:.4g}, {win[1]:.4g}]", flush=True)
    write_png_gray(os.path.join(out, "contact_sheet" + suffix + ".png"), grid(rows))
    print("wrote", os.path.join(out, "contact_sheet" + suffix + ".png"))


if __name__ == "__main__":
    main()
