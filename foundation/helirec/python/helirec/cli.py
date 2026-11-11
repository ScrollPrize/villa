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
  float32/ (and uint8/) TIFF stacks with --format tiff; manifest.json. --uint8_only writes the uint8 version
  alone (no float32 output).

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
    from . import to_uint8
    return to_uint8(np.ascontiguousarray(img, np.float32), *window)


def reference_mask(a, r):
    """the --reference_volume mask on the reconstruction's grid (helirec.ReferenceMask), or None"""
    if not a.reference_volume:
        return None
    from ._core import ReferenceMask
    N = r.info["grid_size"]
    try:
        ref = ReferenceMask(a.reference_volume, a.reference_scale, (r.hi - r.lo, N, N), a.reference_cache or "")
    except RuntimeError as e:
        sys.exit(f"--reference_volume: {e}")
    print(f"reference volume {a.reference_volume}: uint8 output masked where its level {ref.scale} "
          f"({'x'.join(map(str, ref.level_shape))}, {'x'.join(map(str, ref.factors))} voxels each) is 0")
    if ref.slice_ratio != 1:
        print(f"  the reference has {ref.slice_ratio:.5f} slices per slice of this reconstruction (another helical pitch "
              f"calibration): its slices are spread evenly over the {r.hi - r.lo} of the reconstruction; the mask is "
              f"placed to within a few slices")
    return ref


def describe_reference(ref):
    return dict(location=ref.location, scale=ref.scale, factors=list(ref.factors), level_shape=list(ref.level_shape),
                slice_ratio=ref.slice_ratio)



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
    from . import ProgressBar
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
    with ProgressBar(len(fractions), "slices", "slice") as bar:
        for f in fractions:
            si = r.slice_info(f)
            t = time.time()
            N = r.info["grid_size"]
            z = si.view_height - r.lo
            mask = a.reference.read(z, z + 1, a.roi or (0, N, 0, N)) if a.reference else None
            # (single slices keep their values outside the reconstruction disk)
            img, u8 = r.reconstruct_heights(si.view_height, si.view_height + 1, roi=a.roi, progress=bar,
                                            uint8=(*a.u8_window, False), mask=mask, region=a.bbox)
            img, u8 = img[0], u8[0]
            dt = time.time() - t
            bar.set_postfix("write")
            name = f"slice_f{f:.4f}"
            entry = dict(name=name, fraction=f, view_height=si.view_height, z=si.view_height - r.lo,
                         nabu_index=si.nabu_index, height_mm=si.height_mm, seconds=round(dt, 2))
            if not a.uint8_only:
                entry["tiff"] = os.path.join(out, name + ".tiff")
                tifffile.imwrite(entry["tiff"], img)
            if a.uint8:
                entry["tiff_uint8"] = os.path.join(out, name + "_uint8.tif")
                tifffile.imwrite(entry["tiff_uint8"], u8)
            path = entry.get("tiff", entry.get("tiff_uint8"))
            if not a.no_preview:
                entry["preview"] = os.path.join(out, "preview", name + ".png")
                write_png_gray(entry["preview"], u8)
                previews.append(to_u8(downsample(img, 512), a.u8_window))
            manifest["slices"].append(entry)
            bar.write(f"f={f:.4f} h={si.view_height} z={entry['z']} ({si.height_mm:.2f} mm) {dt:.1f}s -> {path}")
            save_manifest(out, manifest)
    if previews:
        contact_sheet(os.path.join(out, "contact_sheet.png"), previews)


def run_range(a, r, out, manifest):
    import tifffile
    from . import ProgressBar
    from .omezarr import OmeZarrWriter
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
    manifest.update(view_heights=[h0, h1], z=[z0, z0 + n], roi_xyxy=[x0, x1, y0, y1], shape_zyx=list(shape),
                    chunk=a.chunk, format=a.format)
    if a.reference:
        manifest["reference_volume"] = describe_reference(a.reference)
    if a.bbox:
        manifest["bbox_xyxy"] = list(a.bbox)
    writers = []
    provenance = {"helirec": {k: manifest[k] for k in ("created", "projections", "mode", "params", "provenance",
                                                     "view_heights", "z", "roi_xyxy", "chunk")},
                  "helirec_calibration": manifest["calibration"]}
    u8_attrs = {"helirec_uint8_window": list(a.u8_window),
                **({"helirec_reference_volume": describe_reference(a.reference)} if a.reference else {})}
    vox = r.main.pixel_size_m * 1e6
    if a.format == "zarr":
        chunks = tuple(a.zarr_chunks) if a.zarr_chunks else (128, 128, 128)  # 32-aligned for vc-delta3d presets
        if not a.uint8_only:
            writers.append(("float32", OmeZarrWriter(os.path.join(out, "volume.zarr"), shape, np.float32, chunks, vox,
                                                     (z0, y0, x0), a.levels, a.compression, name=f"{a.tag} float32",
                                                     attrs=provenance, overwrite=a.overwrite)))
        if a.uint8:
            writers.append(("uint8", OmeZarrWriter(os.path.join(out, "volume_uint8.zarr"), shape, np.uint8, chunks,
                                                   vox, (z0, y0, x0), a.levels, a.u8_compression,
                                                   delta3d_preset=a.u8_delta3d_preset,
                                                   name=f"{a.tag} uint8", attrs={**provenance, **u8_attrs},
                                                   overwrite=a.overwrite)))
        manifest["outputs"] = {k: w.path for k, w in writers}
    else:
        manifest["outputs"] = {**({} if a.uint8_only else {"float32": os.path.join(out, "float32")}),
                               **({"uint8": os.path.join(out, "uint8")} if a.uint8 else {})}
        for d in manifest["outputs"].values():
            os.makedirs(d, exist_ok=True)
    save_manifest(out, manifest)
    t_all = time.time()
    mid_preview = None
    roi = (x0, x1, y0, y1) if a.roi else None
    # the reference volume's mask of a chunk is read while the chunk before it reconstructs (two buffers in turn)
    from concurrent.futures import ThreadPoolExecutor
    chunks = [(c0, min(h1, c0 + a.chunk)) for c0 in range(h0, h1, a.chunk)]
    reader = ThreadPoolExecutor(1) if a.reference else None

    def read_mask(i):
        c0, c1 = chunks[i]
        return reader.submit(a.reference.read, c0 - r.lo, c1 - r.lo, (x0, x1, y0, y1), i % 2)

    pending = read_mask(0) if reader else None
    with ProgressBar(n, "reconstruct", "slice") as bar:
        for i, (c0, c1) in enumerate(chunks):
            mask = None
            if reader:
                mask = pending.result()
                pending = read_mask(i + 1) if i + 1 < len(chunks) else None
            zc = c0 - h0  # index inside the output
            # float32 and / or uint8 slices (the uint8 ones converted on the device); OME-Zarr: the reconstructor
            # hands them to the writers itself, as the blocks it computed
            to = dict(writers)
            res = r.reconstruct_heights(c0, c1, roi=roi, next=chunks[i + 1] if i + 1 < len(chunks) else None,
                                        uint8=a.u8_window if a.uint8 else None, float32=not a.uint8_only, progress=bar,
                                        mask=mask, region=a.bbox, float32_to=to.get("float32"), uint8_to=to.get("uint8"),
                                        z_to=zc)
            vol, vol8 = res if a.uint8 and not a.uint8_only else (None, res) if a.uint8 else (res, None)
            bar.set_postfix("write")
            if a.format != "zarr":
                for k in range(c1 - c0):
                    zz = c0 + k - r.lo
                    if not a.uint8_only:
                        tifffile.imwrite(os.path.join(out, "float32", f"z{zz:05d}.tiff"), vol[k])
                    if a.uint8:
                        tifffile.imwrite(os.path.join(out, "uint8", f"z{zz:05d}.tif"), vol8[k])
            if a.format != "zarr" and c0 <= (h0 + h1) // 2 < c1 and not a.no_preview:
                k = (h0 + h1) // 2 - c0
                mid_preview = vol8[k] if a.uint8_only else to_u8(vol[k], a.u8_window)
    if reader:
        reader.shutdown()
    for kind, w in writers:
        bars = []   # one bar per pyramid level

        def level(l, slabs, kind=kind, bars=bars):
            for b in bars:
                b.close()
            bars.append(ProgressBar(slabs, f"{kind} level {l}", "slab"))
            return bars[-1].update

        w.close(progress=level)
        for b in bars:
            b.close()
        manifest.setdefault("chunks_written", {})[kind] = dict(written=w.n_written, skipped_empty=w.n_skipped)
    if writers and not a.no_preview:   # the middle slice, as stored
        k = (h0 + h1) // 2 - h0
        stored = {kind: w.read(0, k, k + 1)[0] for kind, w in writers}
        mid_preview = stored["uint8"] if "uint8" in stored else to_u8(stored["float32"], a.u8_window)
    if mid_preview is not None:
        write_png_gray(os.path.join(out, "preview_mid.png"), mid_preview)
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
    out.add_argument("--uint8_only", action="store_true", help="write the uint8 version alone (no float32 output)")
    out.add_argument("--u8_window", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                     help="float window mapped to 0..255 (default: the reference window, in the run's "
                          "units: [-0.06, 0.29] cm^-1, or [-0.03, 0.145] in mode legacy_reference)")
    sel.add_argument("--bbox", nargs=4, type=int, metavar=("X0", "X1", "Y0", "Y1"),
                     help="reconstruct only the pixels [X0, X1) x [Y0, Y1) of each slice; the output keeps its shape "
                          "(--roi crops it) and is 0 elsewhere")
    out.add_argument("--reference_volume", default=None, metavar="ZARR",
                     help="a masked uint8 volume on the reconstruction's grid (a local or http(s):// / s3:// Zarr array "
                          "or OME-Zarr): the uint8 output is 0 where it is 0, so its mask need not be redone. Its "
                          "number of slices may differ by up to 2 %% (a volume made with another helical pitch "
                          "calibration): its slices are then spread evenly over the reconstruction's")
    out.add_argument("--reference_scale", type=int, default=0, metavar="LEVEL",
                     help="pyramid level of --reference_volume that decides the mask (default 0, full resolution): a "
                          "coarser level is a much smaller read and marks the same area to within its voxel size")
    out.add_argument("--reference_cache", default=None, metavar="DIR",
                     help="vc_reader disk cache for a remote --reference_volume")
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
    a.uint8 = a.uint8 or a.uint8_only
    if a.reference_volume and not a.uint8:
        sys.exit("--reference_volume masks the uint8 output: add --uint8 or --uint8_only")

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
          f"rotation axis {r.rot_center():.3f}, grid {r.info['grid_size']}, margin {r.info['margin']}")
    a.reference = reference_mask(a, r)
    N = r.info["grid_size"]
    if a.bbox and not (0 <= a.bbox[0] < a.bbox[1] <= N and 0 <= a.bbox[2] < a.bbox[3] <= N):
        sys.exit(f"--bbox must lie within [0, {N}]")
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
