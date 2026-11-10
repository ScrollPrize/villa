"""helirec-slices: reconstruct slices of an extended-FOV helical scan pair from the raw projection
folders with helirec (C++/CUDA).

Two modes:

* fractions (independent single slices, TIFF; PNG previews opt in with --previews):
    helirec-slices PROJ -n 8 --tag baseline
    helirec-slices PROJ --fractions 0.4 0.5 --set "filters=paganin(delta_beta=500)+unsharp(sigma=1.2,coeff=4)" --tag db500
    helirec-slices PROJ --fractions 0.5 --set "filters=paganin(delta_beta=2000)+median(radius=1,stage=log)"

* range (contiguous slices, reconstructed in chunks of --chunk slices; OME-Zarr by default):
    helirec-slices PROJ --z Z0 Z1 --roi X0 X1 Y0 Y1 --uint8 --tag block
    helirec-slices PROJ --heights H0 H1 --format tiff --tag block_tif

  --z uses volume slice indices (z = view height - first doable height, z = 0 at the top);
  --heights uses helirec view heights; both are [start, end). --roi X0 X1 Y0 Y1 crops each slice ([start, end)).
  Output: volume.zarr (float32 OME-Zarr, raw chunks), volume_uint8.zarr with --uint8 (window --u8_window,
  lossless vc-delta3d chunks, 0 outside the reconstruction disk); empty chunks are not written; or
  float32/ (and uint8/) TIFF stacks with --format tiff; manifest.json.

Units: attenuation in cm^-1 by default; --set mode=legacy_reference gives the reference workflow's units (half of
it) and conventions. The default uint8 window is the reference one in the run's units.
Note: in mode legacy_reference chunked slices carry the chunk-height dependence of the reference Paganin kernel (a
uniform ~+1.4e-4 offset vs single-slice reconstruction for chunks >= 16, as in the reference's full-volume runs); the
default mode's Paganin kernel does not depend on the chunk height.
Calibration (darks/flats, weights, rotation axis, HA/TA vector) is computed once and cached (--cache_dir).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import struct
import sys
import time
import zlib

import numpy as np

from .recon import REFERENCE_WINDOW as PREVIEW_WINDOW  # noqa: E402  (legacy reference units)
# object extent (fractions of the doable span) that -n spreads slices over; default: the whole span
DEFAULT_OBJECT_EXTENT = (0.0, 1.0)


def write_png_gray(path, img_u8):
    """Minimal 8-bit grayscale PNG writer (no Pillow dependency)."""
    h, w = img_u8.shape
    raw = b"".join(b"\x00" + img_u8[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)


def to_u8(img, window=PREVIEW_WINDOW):
    from .omezarr import to_uint8
    return to_uint8(img, *window)


def downsample(img, size):
    f = max(1, img.shape[0] // size)
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def parse_set(items):
    """--set KEY=VALUE items -> params; filters=CHAIN (text or JSON) and TYPE.param=VALUE address the filter chain."""
    from .filters import _value, is_filter_key, parse_chain
    from .recon import DEFAULTS
    out = {}
    for it in items:
        if "=" not in it:
            sys.exit(f"bad --set '{it}', expected KEY=VALUE")
        k, v = (x.strip() for x in it.split("=", 1))
        if is_filter_key(k):
            out[k] = parse_chain(v) if k == "filters" else _value(v)
            continue
        if k not in DEFAULTS:
            sys.exit(f"unknown parameter '{k}'; known: {', '.join(sorted(DEFAULTS))}")
        d = DEFAULTS[k]
        if isinstance(d, bool):
            out[k] = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(d, int):
            out[k] = int(v)
        elif isinstance(d, float):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _json_default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def contact_sheet(path, previews):
    cols = min(5, int(np.ceil(np.sqrt(len(previews)))))
    rows = int(np.ceil(len(previews) / cols))
    s = max(p.shape[0] for p in previews)
    w = max(p.shape[1] for p in previews)
    sheet = np.zeros((rows * s, cols * w), np.uint8)
    for k, im in enumerate(previews):
        sheet[(k // cols) * s:(k // cols) * s + im.shape[0], (k % cols) * w:(k % cols) * w + im.shape[1]] = im
    write_png_gray(path, sheet)


# ------------------------------------------------------------------------------------------ modes
def run_fractions(a, r, out, manifest):
    import tifffile
    if a.fractions:
        fractions = list(a.fractions)
    else:
        A, B = a.object_extent
        f0, f1 = (A + (B - A) * x for x in a.range)
        n = a.n_slices
        fractions = [0.5 * (f0 + f1)] if n == 1 else [f0 + (f1 - f0) * i / (n - 1) for i in range(n)]
    if not a.no_preview:
        os.makedirs(os.path.join(out, "preview"), exist_ok=True)
    previews = []
    for i, f in enumerate(fractions):
        si = r.slice_info(f)
        t = time.time()
        img = crop(r.reconstruct_heights(si.view_height, si.view_height + 1)[0], a.roi)
        dt = time.time() - t
        name = f"slice_f{f:.4f}"
        path = os.path.join(out, name + ".tiff")
        tifffile.imwrite(path, img)
        entry = dict(name=name, fraction=f, view_height=si.view_height, z=si.view_height - r.lo,
                     nabu_index=si.nabu_index, height_mm=si.height_mm, seconds=round(dt, 2), tiff=path)
        u8 = to_u8(img, a.u8_window)
        if a.uint8:
            entry["tiff_uint8"] = os.path.join(out, name + "_uint8.tif")
            tifffile.imwrite(entry["tiff_uint8"], u8)
        if not a.no_preview:
            entry["preview"] = os.path.join(out, "preview", name + ".png")
            write_png_gray(entry["preview"], u8)
            previews.append(to_u8(downsample(img, 512), a.u8_window))
        manifest["slices"].append(entry)
        print(f"[{i + 1}/{len(fractions)}] f={f:.4f} h={si.view_height} z={entry['z']} ({si.height_mm:.2f} mm) "
              f"{dt:.1f}s -> {path}")
        save_manifest(out, manifest)
    if previews:
        contact_sheet(os.path.join(out, "contact_sheet.png"), previews)


def crop(img, roi):
    if roi is None:
        return img
    x0, x1, y0, y1 = roi
    return np.ascontiguousarray(img[..., y0:y1, x0:x1])


def run_range(a, r, out, manifest):
    import tifffile
    from .omezarr import OmeZarrWriter, outside_disk, to_uint8
    if a.z is not None:
        h0, h1 = a.z[0] + r.lo, a.z[1] + r.lo
    else:
        h0, h1 = a.heights
    if not (r.lo <= h0 < h1 <= r.hi):
        sys.exit(f"range outside the doable view heights [{r.lo}, {r.hi}) (z [0, {r.hi - r.lo}))")
    N = r.info["grid_size"]
    x0, x1, y0, y1 = a.roi if a.roi else (0, N, 0, N)
    if not (0 <= x0 < x1 <= N and 0 <= y0 < y1 <= N):
        sys.exit(f"--roi must lie within [0, {N}]")
    n = h1 - h0
    shape = (n, y1 - y0, x1 - x0)
    z0 = h0 - r.lo
    outside = outside_disk(N, (x0, x1, y0, y1)) if r.params["clip_outer_circle"] else None
    manifest.update(view_heights=[h0, h1], z=[z0, z0 + n], roi_xyxy=[x0, x1, y0, y1], shape_zyx=list(shape),
                    chunk=a.chunk, format=a.format)
    writers = []
    provenance = {"helirec": {k: manifest[k] for k in ("created", "projections", "mode", "params", "provenance",
                                                     "view_heights", "z", "roi_xyxy", "chunk")},
                  "helirec_calibration": manifest["calibration"]}
    vox = r.main.pixel_size_m * 1e6
    if a.format == "zarr":
        chunks = tuple(a.zarr_chunks) if a.zarr_chunks else (128, 128, 128)  # 32-aligned for vc-delta3d presets
        writers.append(("float32", OmeZarrWriter(os.path.join(out, "volume.zarr"), shape, np.float32, chunks, vox,
                                                 (z0, y0, x0), a.levels, a.compression, name=f"{a.tag} float32",
                                                 attrs=provenance, overwrite=a.overwrite)))
        if a.uint8:
            writers.append(("uint8", OmeZarrWriter(os.path.join(out, "volume_uint8.zarr"), shape, np.uint8, chunks,
                                                   vox, (z0, y0, x0), a.levels, a.u8_compression,
                                                   delta3d_preset=a.u8_delta3d_preset,
                                                   name=f"{a.tag} uint8", attrs={**provenance,
                                                   "helirec_uint8_window": list(a.u8_window)},
                                                   overwrite=a.overwrite)))
        manifest["outputs"] = {k: w.path for k, w in writers}
    else:
        os.makedirs(os.path.join(out, "float32"), exist_ok=True)
        if a.uint8:
            os.makedirs(os.path.join(out, "uint8"), exist_ok=True)
        manifest["outputs"] = {"float32": os.path.join(out, "float32"),
                               **({"uint8": os.path.join(out, "uint8")} if a.uint8 else {})}
    save_manifest(out, manifest)
    t_all = time.time()
    done = 0
    mid_preview = None
    for c0 in range(h0, h1, a.chunk):
        c1 = min(h1, c0 + a.chunk)
        t = time.time()
        vol = crop(r.reconstruct_heights(c0, c1, next=(c1, min(h1, c1 + a.chunk)) if c1 < h1 else None), (x0, x1, y0, y1))
        t_rec = time.time() - t
        zc = c0 - h0  # index inside the output
        if a.format == "zarr":
            for kind, w in writers:
                w.write(zc, vol if kind == "float32" else to_uint8(vol, *a.u8_window, outside=outside))
        else:
            for k in range(vol.shape[0]):
                zz = c0 + k - r.lo
                tifffile.imwrite(os.path.join(out, "float32", f"z{zz:05d}.tiff"), vol[k])
                if a.uint8:
                    tifffile.imwrite(os.path.join(out, "uint8", f"z{zz:05d}.tif"),
                                     to_uint8(vol[k], *a.u8_window, outside=outside))
        if mid_preview is None and c0 <= (h0 + h1) // 2 < c1:
            mid_preview = vol[(h0 + h1) // 2 - c0]
        done += c1 - c0
        el = time.time() - t_all
        print(f"  z {c0 - r.lo}..{c1 - r.lo - 1} (h {c0}..{c1 - 1}): recon {t_rec:.1f}s, total {el:.0f}s, "
              f"{done}/{n} slices, ETA {el / done * (n - done):.0f}s", flush=True)
    for kind, w in writers:
        w.close()
        manifest.setdefault("chunks_written", {})[kind] = dict(written=w.n_written, skipped_empty=w.n_skipped)
    if mid_preview is not None and not a.no_preview:
        write_png_gray(os.path.join(out, "preview_mid.png"), to_u8(mid_preview, a.u8_window))
    manifest["seconds"] = round(time.time() - t_all, 1)
    save_manifest(out, manifest)


def save_manifest(out, manifest):
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=_json_default)


# ------------------------------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("projections", nargs="?", help="folder containing the raw Bliss scan folders (HA and TA)")
    p.add_argument("--config", help="JSON file of arguments (keys = option names, e.g. z, roi, uint8, set); "
                                     "command-line options override it. Each run saves its config.json")
    sel = p.add_argument_group("slice selection")
    sel.add_argument("-n", "--n_slices", type=int, default=5)
    sel.add_argument("--range", nargs=2, type=float, default=(1 / 6, 5 / 6), metavar=("R0", "R1"),
                     help="fraction mode: range within the object extent (default: middle 2/3)")
    sel.add_argument("--object_extent", nargs=2, type=float, default=DEFAULT_OBJECT_EXTENT, metavar=("A", "B"))
    sel.add_argument("--fractions", nargs="+", type=float, help="explicit fractions of the doable span")
    sel.add_argument("--z", nargs=2, type=int, metavar=("Z0", "Z1"),
                     help="range mode: volume slice indices [Z0, Z1)")
    sel.add_argument("--heights", nargs=2, type=int, metavar=("H0", "H1"), help="range mode: view heights [H0, H1)")
    sel.add_argument("--roi", nargs=4, type=int, metavar=("X0", "X1", "Y0", "Y1"), help="crop [X0, X1) x [Y0, Y1)")
    out = p.add_argument_group("output")
    out.add_argument("--format", choices=["zarr", "tiff"], default=None,
                     help="range mode default: zarr; fraction mode: tiff only")
    out.add_argument("--uint8", action="store_true", help="also write a uint8 version")
    out.add_argument("--u8_window", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                     help="float window mapped to 0..255 (default: the reference window, in the run's "
                          "units: [-0.06, 0.29] cm^-1, or [-0.03, 0.145] in mode legacy_reference)")
    out.add_argument("--chunk", type=int, default=32, help="range mode: slices reconstructed together (default 32)")
    out.add_argument("--zarr_chunks", nargs=3, type=int, metavar=("CZ", "CY", "CX"),
                     help="OME-Zarr chunk shape (default 128 128 128; keep multiples of 32 for vc-delta3d presets)")
    out.add_argument("--levels", type=int, default=1, help="OME-Zarr pyramid levels (2x mean pooling)")
    out.add_argument("--compression", choices=["none", "zlib"], default="none",
                     help="float32 OME-Zarr chunks (default: raw)")
    out.add_argument("--u8_compression", choices=["vc-delta3d", "none", "zlib"], default="vc-delta3d",
                     help="uint8 OME-Zarr chunks (default: lossless vc-delta3d)")
    out.add_argument("--u8_delta3d_preset", choices=["fast", "medium", "slow", "slower", "delta"], default="fast",
                     help="vc-delta3d preset (default fast, as vc_reader); 'delta' = legacy D3D1 frames for "
                          "vc_reader builds older than the preset codec")
    out.add_argument("--tag", default=None)
    out.add_argument("--out_root", default=os.path.abspath("helirec_slices"))
    previews = out.add_mutually_exclusive_group()
    previews.add_argument("--previews", dest="no_preview", action="store_false", help="also write PNG previews (default: off)")
    previews.add_argument("--no_preview", dest="no_preview", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(no_preview=True)
    out.add_argument("--overwrite", action="store_true")
    rc = p.add_argument_group("reconstruction")
    rc.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="parameter override (repeatable), e.g. backprojector=fbp, paganin.delta_beta=500, "
                         "unsharp.coeff=0, filters='paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)'; "
                         "mode=legacy_reference reproduces the reference workflow (every other parameter pinned)")
    rc.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    rc.add_argument("--cache_dir", default=None, help="calibration cache (default: OUT_ROOT/_calibration)")
    rc.add_argument("--cor", type=float, default=None, help="override the main scan's rotation axis (detector px)")
    rc.add_argument("--vector", nargs=2, type=float, default=None, metavar=("DY", "DX"),
                    help="override the main -> companion vector")
    a = p.parse_args(argv)
    if a.config:
        with open(a.config) as fh:
            cfg = {k.replace("-", "_"): v for k, v in json.load(fh).items() if not k.startswith("_")}
        unknown = set(cfg) - {act.dest for act in p._actions}
        if unknown:
            sys.exit(f"{a.config}: unknown keys {sorted(unknown)}")
        p.set_defaults(**cfg)
        a = p.parse_args(argv)  # command line wins over the file
    if not a.projections:
        sys.exit("projections folder missing (positional argument or 'projections' in --config)")

    range_mode = a.z is not None or a.heights is not None
    if a.z is not None and a.heights is not None:
        sys.exit("use either --z or --heights")
    if range_mode and a.fractions:
        sys.exit("--fractions cannot be combined with --z/--heights")
    a.format = a.format or ("zarr" if range_mode else "tiff")
    if not range_mode and a.format == "zarr":
        sys.exit("OME-Zarr output needs a contiguous range (--z or --heights)")
    if a.chunk < 1:
        sys.exit("--chunk must be >= 1")

    from .recon import HelicalRecon
    a.tag = a.tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(a.out_root, a.tag)
    if os.path.exists(outdir) and not a.overwrite:
        sys.exit(f"{outdir} exists (use another --tag or --overwrite)")
    os.makedirs(outdir, exist_ok=True)
    effective = {k: v for k, v in vars(a).items() if k != "config" and v is not None}
    effective["projections"] = os.path.abspath(a.projections)
    effective["out_root"] = os.path.abspath(a.out_root)
    with open(os.path.join(outdir, "config.json"), "w") as fh:  # rerun with: helirec-slices --config config.json
        json.dump(effective, fh, indent=2, default=_json_default)
    cache = a.cache_dir or os.path.join(a.out_root, "_calibration")
    t0 = time.time()
    r = HelicalRecon(a.projections, cache_dir=cache, device=a.device, params=parse_set(a.set),
                     cor_override=a.cor, vector_override=tuple(a.vector) if a.vector else None)
    if a.u8_window is None:
        from .recon import display_window
        a.u8_window = display_window(r.params["mode"])
    print(f"setup {time.time() - t0:.1f}s: doable view heights [{r.lo}, {r.hi}) (z 0..{r.hi - r.lo - 1}), "
          f"rotation axis {r.cal.rot_center}, grid {r.info['grid_size']}, margin {r.info['margin']}")
    manifest = dict(created=datetime.datetime.now().isoformat(timespec="seconds"),
                    projections=os.path.abspath(a.projections), device=a.device, mode=r.params["mode"], params=r.params,
                    provenance=r.provenance(), calibration=json.loads(r.cal.to_json()),
                    uint8_window=list(a.u8_window), slices=[])
    if range_mode:
        run_range(a, r, outdir, manifest)
    else:
        run_fractions(a, r, outdir, manifest)
    print(f"done in {time.time() - t0:.0f}s -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
