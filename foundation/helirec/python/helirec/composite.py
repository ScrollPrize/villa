"""helirec-composite: layer composites (max, mean) of a tifxyz surface, plus xy slices.

Two sources:
  * a reconstructed OME-Zarr block (helirec-slices output):
        helirec-composite VOLUME.zarr TIFXYZ_DIR --layers 15 --start -4 --tag seg
  * the raw projections directly (band reconstruction: exact backprojector at the pixels the layers need, no
    volume), optionally sweeping Paganin parameters:
        helirec-composite --projections PROJ --cache_dir CAL TIFXYZ_DIR --z Z0 Z1 --layers 15 --start -4
        helirec-composite --projections PROJ ... --set "filters=paganin(delta_beta=1000)" --sweep paganin.delta_beta=250,500,1000,2000
        helirec-composite --projections PROJ ... --set "filters=paganin(delta_beta=2000)+median(radius=1)"

Geometry and sampling follow vc_reader / VC3D: the surface is a bicubic Hermite patch grid, normals are
dP/du x dP/dv (u = increasing columns, v = increasing rows), layers are trilinear samples at offsets along the
unit normal, in voxels. Layer k (k = 0 .. layers-1) is at offset  direction * (start + k * step); with
start -4, step 1, 15 layers: 4 voxels away from the material, the surface, 10 voxels into it.
`--direction auto` steps toward the side with the higher mean intensity over offsets 1..5 (the material).

Output (one folder; per parameter set in a sweep, one subfolder each):
    max.tif, mean.tif (float32, NaN outside), max_u8.{tif,png}, mean_u8.{tif,png}, mask.png, composite.json
    layers/      the layers the composites are built from: NN_offset+O.tif (float32)
    xy_slices/   --xy_slices N slices spread over the z range: zZZZZZ.tif (float32) + zZZZZZ.png
    surface_volume.zarr  with --surface_zarr: the layers as a uint8 OME-Zarr (layers, H, W)
Pixels are full-resolution UV pixels of the surface (vc_render_tifxyz's grid; --out_shape renders the same extent
at another size); composite.json gives the crop's origin on that canvas.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

def volume_mode(path):
    """units mode of an OME-Zarr volume: the helirec mode in its provenance; volumes without one (written by the
    reference workflow) are in legacy reference units"""
    try:
        attrs = json.load(open(os.path.join(path, ".zattrs")))
    except (OSError, ValueError):
        return "legacy_reference"
    return (attrs.get("helirec") or {}).get("mode", "legacy_reference")


def _vcr():
    try:
        from vc_reader import grid, tifxyz
        from vc_reader.reader.zarr_reader import ZarrReader
    except ImportError as e:  # pragma: no cover
        raise ImportError("helirec-composite needs vc_reader (pip install villa/vc_reader)") from e
    return grid, tifxyz, ZarrReader


class Volume:
    """One level of an OME-Zarr block placed in the full volume (translation from the NGFF metadata)."""

    def __init__(self, path, level=0):
        _, _, ZarrReader = _vcr()
        self.path, self.level = os.path.abspath(path), level
        attrs = json.load(open(os.path.join(path, ".zattrs")))
        ms = attrs["multiscales"][0]
        ds = ms["datasets"][level]
        tf = ds["coordinateTransformations"]
        scale = next(t["scale"] for t in tf if t["type"] == "scale")
        tr = next((t["translation"] for t in tf if t["type"] == "translation"), [0.0, 0.0, 0.0])
        s = 2 ** level
        self.voxel_um = scale[0] / s                       # level-0 voxel
        # level-l index i covers level-0 voxels [o + s*i, o + s*i + s - 1], centre o + s*i + (s-1)/2
        self.origin0 = tuple(int(round(t / self.voxel_um - (s - 1) / 2)) for t in tr)   # ZYX, level-0 voxels
        self.s = s
        self.reader = ZarrReader.open(os.path.join(path, ds["path"]))
        self.shape = tuple(self.reader.shape)
        self.dtype = np.dtype(self.reader.dtype)

    def index(self, xyz):
        """Level-0 XYZ points -> ZYX indices of this level (float)."""
        o, s = self.origin0, self.s
        return np.stack([(xyz[..., 2 - k] - o[k] - (s - 1) / 2) / s for k in range(3)], -1)

    def map3x4(self):
        """Row-major 3x4 map from level-0 XYZ points to this level's ZYX indices (vc_reader GridLevel)."""
        o, s = self.origin0, self.s
        m = np.zeros((3, 4))
        for k in range(3):          # output axis k (Z, Y, X) reads input component 2 - k (Z, Y, X of XYZ)
            m[k, 2 - k] = 1 / s
            m[k, 3] = -(o[k] + (s - 1) / 2) / s
        return m

    def grid_level(self):
        return self.reader.grid_level(map=self.map3x4())

    def bounds0(self):
        """[lo, hi] of level-0 coordinates covered by sample centres, ZYX."""
        lo = np.array(self.origin0, float) + (self.s - 1) / 2
        return lo, lo + self.s * (np.array(self.shape) - 1)


def _inside(vol, xyz, margin=0.0):
    idx = vol.index(xyz)
    n = np.array(vol.shape, float)
    return np.all((idx >= margin) & (idx <= n - 1 - margin), -1)


def uv_window(surf, vol, reach, pixel_scale, box=None):
    """Output-pixel window [r0, r1) x [c0, c1) whose surface can reach the volume block (or the level-0 ZYX box
    (lo, hi)), from the stored grid."""
    xyz, valid = np.asarray(surf.xyz, np.float64), np.asarray(surf.valid)
    lo, hi = box if box is not None else vol.bounds0()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    zyx = xyz[..., ::-1]
    near = valid & np.all((zyx >= lo - reach - 2) & (zyx <= hi + reach + 2), -1)
    if not near.any():
        raise ValueError("the surface does not reach the volume block")
    rows, cols = np.nonzero(near.any(1))[0], np.nonzero(near.any(0))[0]
    sy, sx = surf.scale
    ky, kx = pixel_scale
    H, W = (int(round(n * k)) for n, k in zip(surf.full_shape, pixel_scale))
    r0 = max(0, int(np.floor((rows[0] - 1) / sy * ky)))
    r1 = min(H, int(np.ceil((rows[-1] + 1) / sy * ky)) + 1)
    c0 = max(0, int(np.floor((cols[0] - 1) / sx * kx)))
    c1 = min(W, int(np.ceil((cols[-1] + 1) / sx * kx)) + 1)
    return (r0, r1, c0, c1), (H, W)


def sample_layers(surf, vol, window, pixel_scale, offsets, interpolation="bicubic", tile_rows=256):
    """(L, H, W) float32 layers over the output window, NaN where a layer has no point or leaves the block."""
    grid, _, _ = _vcr()
    r0, r1, c0, c1 = window
    sy, sx = surf.scale
    ky, kx = pixel_scale
    level = vol.grid_level()
    out = np.full((len(offsets), r1 - r0, c1 - c0), np.nan, np.float32)
    for a in range(r0, r1, tile_rows):
        b = min(r1, a + tile_rows)
        raster = (c0 * sx / kx, a * sy / ky, sx / kx, sy / ky, c1 - c0, b - a)
        res = grid.sample_grid([level], surface=surf, raster=raster, interpolation=interpolation,
                               offsets=[float(o) for o in offsets])
        vals = np.asarray(res.values, np.float32)
        # geometry of the same pixels: keep a pixel only if every layer is inside the block (no fill-value reads)
        pos, ok, nrm = _points(surf, raster, interpolation)
        inside = ok.copy()
        for o in offsets:
            inside &= _inside(vol, pos + o * nrm)
        vals[:, ~inside] = np.nan
        out[:, a - r0:b - r0] = vals
    return out


def _points(surf, raster, interpolation):
    """Positions, validity, unit normals at raster pixels (vc_reader's tifxyz geometry)."""
    x0, y0, sx, sy, w, h = raster
    rr, cc = np.meshgrid(y0 + np.arange(h) * sy, x0 + np.arange(w) * sx, indexing="ij")
    pos, ok, nrm = surf.points_at(rr.astype(np.float32), cc.astype(np.float32), interpolation, True)
    pos, nrm = np.asarray(pos, np.float64), np.asarray(nrm, np.float64)
    ok = np.asarray(ok, bool) & np.isfinite(pos).all(-1) & np.isfinite(nrm).all(-1)
    return np.where(ok[..., None], pos, 0), ok, np.where(ok[..., None], nrm, 0)


def material_side(surf, vol, window, pixel_scale, interpolation="bicubic", depth=5, sub=4):
    """+1 or -1: the side of the normal with the higher mean intensity at offsets 1..depth (+ the profile)."""
    r0, r1, c0, c1 = window
    ky, kx = pixel_scale
    offs = list(range(-3 * depth, 3 * depth + 1))
    # every `sub`-th output pixel is enough for a mean profile
    sub_window = (r0 // sub, (r1 + sub - 1) // sub, c0 // sub, (c1 + sub - 1) // sub)
    layers = sample_layers(surf, vol, sub_window, (ky / sub, kx / sub), offs, interpolation)
    return side_from_layers(layers, offs, depth)


def side_from_layers(layers, offsets, depth=5):
    """+1 or -1 (the side of the normal with the higher mean intensity at offsets 1..depth) and the evidence;
    `layers` (L, H, W) must include offsets -depth..depth."""
    k = {float(o): i for i, o in enumerate(offsets)}
    need = [float(d) for d in range(-depth, depth + 1) if d]
    if any(d not in k for d in need):
        raise ValueError(f"side test needs offsets -{depth}..{depth}")
    sel = [k[d] for d in need]
    good = np.isfinite(layers[sel]).all(0)
    if good.sum() < 100:
        raise ValueError("too few surface pixels inside the volume to decide the material side")
    prof = {o: float(layers[i][good].mean()) for o, i in k.items()}
    plus = np.mean([prof[float(d)] for d in range(1, depth + 1)])
    minus = np.mean([prof[float(-d)] for d in range(1, depth + 1)])
    per_px = (np.mean([layers[k[float(d)]] for d in range(1, depth + 1)], 0)
              - np.mean([layers[k[float(-d)]] for d in range(1, depth + 1)], 0))
    return (1 if plus > minus else -1), dict(
        offsets=sorted(prof), mean=[prof[o] for o in sorted(prof)], plus_side_mean=float(plus),
        minus_side_mean=float(minus), frac_pixels_plus_brighter=float(np.mean(per_px[good] > 0)),
        n_pixels=int(good.sum()))


def to_u8(img, window):
    lo, hi = window
    u = np.rint(np.clip((np.nan_to_num(img, nan=lo) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    u[~np.isfinite(img)] = 0
    return u


def layer_offsets(direction, start, step, n):
    return [direction * (start + k * step) + 0.0 for k in range(n)]


def write_surface_zarr(path, layers, good, u8_window, voxel_um, origin_rc):
    """The layer stack as an OME-Zarr surface volume (layers, H, W) uint8 (u8_window; 0 outside): depth axis first,
    surface at the centre layer."""
    from .omezarr import OmeZarrWriter
    L, H, W = layers.shape
    u8 = np.stack([np.where(good, to_u8(l, u8_window), 0) for l in layers]).astype(np.uint8)
    w = OmeZarrWriter(path, (L, H, W), np.uint8, chunks=(L, 128, 128), voxel_size_um=voxel_um,
                      origin_zyx=(0, origin_rc[0], origin_rc[1]), levels=3, compression="none", overwrite=True,
                      name="surface volume", attrs={"helirec_surface_volume": dict(u8_window=list(u8_window))})
    w.write(0, u8)
    w.close()


def write_set(out, layers, offsets, good, meta, u8_window, slices=None, surface_zarr=False):
    """One output folder: composites at the root, layers/, xy_slices/ (and surface_volume.zarr)."""
    import tifffile
    from .cli import _json_default, write_png_gray
    os.makedirs(os.path.join(out, "layers"), exist_ok=True)
    filled = np.where(good, layers, 0)
    comp = {"max": np.where(good, filled.max(0), np.nan).astype(np.float32),
            "mean": np.where(good, filled.mean(0), np.nan).astype(np.float32)}
    for k, img in comp.items():
        tifffile.imwrite(os.path.join(out, f"{k}.tif"), img, compression="zlib")
        u8 = to_u8(img, u8_window)
        tifffile.imwrite(os.path.join(out, f"{k}_u8.tif"), u8, compression="zlib")
        write_png_gray(os.path.join(out, f"{k}_u8.png"), u8)
    write_png_gray(os.path.join(out, "mask.png"), (good * 255).astype(np.uint8))
    for i, (o, img) in enumerate(zip(offsets, layers)):
        tifffile.imwrite(os.path.join(out, "layers", f"{i:02d}_offset{o:+05.1f}.tif"),
                         np.where(good, img, np.nan).astype(np.float32), compression="zlib")
    if slices:
        os.makedirs(os.path.join(out, "xy_slices"), exist_ok=True)
        for z, img in slices:
            tifffile.imwrite(os.path.join(out, "xy_slices", f"z{z:05d}.tif"), img.astype(np.float32))
            write_png_gray(os.path.join(out, "xy_slices", f"z{z:05d}.png"), to_u8(img, u8_window))
        meta["xy_slices"] = [int(z) for z, _ in slices]
    if surface_zarr:
        write_surface_zarr(os.path.join(out, "surface_volume.zarr"), layers, good, u8_window,
                           meta.get("voxel_um", 8.64), (meta["crop"]["row0"], meta["crop"]["col0"]))
        meta["surface_volume"] = dict(path="surface_volume.zarr", layers=len(offsets), offsets=list(offsets),
                                      centre_layer=len(offsets) // 2, dtype="uint8", window=list(u8_window))
    meta["stats"] = {k: dict(p1=float(np.nanpercentile(v, 1)), p50=float(np.nanpercentile(v, 50)),
                             p99=float(np.nanpercentile(v, 99))) for k, v in comp.items()}
    json.dump(meta, open(os.path.join(out, "composite.json"), "w"), indent=2, default=_json_default)


def spread(z0, z1, n):
    """n slice indices spread over [z0, z1) (centres of n equal parts)."""
    if n <= 0:
        return []
    return sorted({int(z0 + (k + 0.5) * (z1 - z0) / n) for k in range(n)})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", help="[VOLUME.zarr] TIFXYZ_DIR (VOLUME omitted with --projections)")
    p.add_argument("--config", help="JSON file of arguments (keys = option names); command line overrides it")
    src = p.add_argument_group("source")
    src.add_argument("--volume", help="OME-Zarr written by helirec-slices (or the first positional argument)")
    src.add_argument("--surface", help="tifxyz directory (or the last positional argument)")
    src.add_argument("--projections", help="raw projection folder: reconstruct the band directly (no volume)")
    src.add_argument("--cache_dir", help="--projections: calibration cache (helirec-slices' OUT_ROOT/_calibration)")
    src.add_argument("--z", nargs=2, type=int, metavar=("Z0", "Z1"),
                     help="--projections: volume slices [Z0, Z1) to use (default: all the surface reaches)")
    src.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="--projections: recon parameter")
    src.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2,...",
                     help="--projections: filter sweep, e.g. paganin.delta_beta=250,500,1000 (keys TYPE[.k].param, "
                          "lists zipped; filters=CHAIN1;CHAIN2 sweeps whole chains); one subfolder per set")
    src.add_argument("--chunk", type=int, default=None, help="--projections: slices per chunk (default 64; sweeps 40)")
    src.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    src.add_argument("--level", type=int, default=0, help="volume: pyramid level to sample (default 0)")
    lay = p.add_argument_group("layers")
    lay.add_argument("--layers", type=int, default=5)
    lay.add_argument("--start", type=float, default=0.0, help="offset of the first layer from the surface, voxels")
    lay.add_argument("--step", type=float, default=1.0, help="layer spacing, voxels")
    lay.add_argument("--direction", default="auto", choices=["auto", "+1", "-1"],
                     help="normal sign toward the material (+1 = along dP/du x dP/dv); auto = denser side")
    lay.add_argument("--interpolation", default="bicubic", choices=["bicubic", "linear"], help="surface patches")
    lay.add_argument("--out_shape", nargs=2, type=int, metavar=("H", "W"),
                     help="full image size (default: the surface's full-resolution size)")
    out = p.add_argument_group("output")
    out.add_argument("--xy_slices", type=int, default=0, help="also write N xy slices spread over the z range")
    out.add_argument("--slice_margin", type=int, default=64,
                     help="xy slices: crop to the surface's xy extent plus this margin (-1: whole slices)")
    out.add_argument("--u8_window", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                     help="default: the reference window in the source's units (cm^-1: [-0.06, 0.29]; "
                          "legacy reference units: [-0.03, 0.145])")
    out.add_argument("--surface_zarr", action="store_true",
                     help="also write the layers as surface_volume.zarr (layers, H, W) uint8 OME-Zarr "
                          "(use symmetric layers, e.g. --layers 21 --start -10, so the surface is the centre layer)")
    out.add_argument("--save_layers", action="store_true", help=argparse.SUPPRESS)  # layers/ is always written
    out.add_argument("--tag", default=None)
    out.add_argument("--out_root", default=os.path.abspath("helirec_composites"))
    out.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if a.config:
        cfg = {k.replace("-", "_"): v for k, v in json.load(open(a.config)).items() if not k.startswith("_")}
        unknown = set(cfg) - {act.dest for act in p._actions}
        if unknown:
            sys.exit(f"{a.config}: unknown keys {sorted(unknown)}")
        p.set_defaults(**cfg)
        a = p.parse_args(argv)
    pos = list(a.inputs)
    if a.projections:
        a.surface = a.surface or (pos[-1] if pos else None)
    else:
        a.volume = a.volume or (pos[0] if len(pos) >= 2 else None)
        a.surface = a.surface or (pos[-1] if pos else None)
        if not a.volume:
            sys.exit("give a volume (OME-Zarr) or --projections")
    if not a.surface:
        sys.exit("surface (tifxyz) missing")
    if a.layers < 1:
        sys.exit("--layers must be >= 1")
    from .filters import set_name, sweep_sets
    sweep = sweep_sets(a.sweep) if a.sweep else None
    if sweep and not a.projections:
        sys.exit("--sweep needs --projections")

    _, tifxyz, _ = _vcr()
    t0 = time.time()
    a.tag = a.tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(a.out_root, a.tag)
    if os.path.exists(outdir) and not a.overwrite:
        sys.exit(f"{outdir} exists (use another --tag or --overwrite)")
    os.makedirs(outdir, exist_ok=True)
    eff = {k: v for k, v in vars(a).items() if k not in ("config", "inputs", "save_layers") and v is not None}
    for k in ("volume", "surface", "projections", "cache_dir", "out_root"):
        if eff.get(k):
            eff[k] = os.path.abspath(eff[k])
    json.dump(eff, open(os.path.join(outdir, "config.json"), "w"), indent=2)

    from .cli import parse_set
    from .recon import display_window
    params = parse_set(a.set) if a.projections else {}
    if a.u8_window is None:
        a.u8_window = display_window(params.get("mode", "corrected") if a.projections else volume_mode(a.volume))
    surf = tifxyz.read(a.surface)
    full = tuple(surf.full_shape)
    ps = (1.0, 1.0) if not a.out_shape else (a.out_shape[0] / full[0], a.out_shape[1] / full[1])
    canvas = tuple(int(round(n * k)) for n, k in zip(full, ps))
    span = [layer_offsets(+1, a.start, a.step, a.layers)]
    reach = max(abs(o) for o in span[0])
    common = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), surface=os.path.abspath(a.surface),
                  surface_uuid=surf.uuid, surface_full_shape=list(full), surface_scale_rows_cols=list(surf.scale),
                  interpolation=a.interpolation, normal="dP/du x dP/dv (u = columns, v = rows), vc_reader / VC3D",
                  canvas_shape=list(canvas), pixel_scale_vs_full_res=list(ps), u8_window=list(a.u8_window))

    if not a.projections:
        # ============================================================ from a reconstructed volume
        vol = Volume(a.volume, a.level)
        window, _ = uv_window(surf, vol, max(reach, 15.0), ps)
        side = None
        if a.direction == "auto":
            direction, side = material_side(surf, vol, window, ps, a.interpolation)
        else:
            direction = int(a.direction)
        offsets = layer_offsets(direction, a.start, a.step, a.layers)
        layers = sample_layers(surf, vol, window, ps, offsets, a.interpolation)
        good = np.isfinite(layers).all(0)
        if not good.any():
            sys.exit("no surface pixel has all its layers inside the volume")
        rows, cols = np.nonzero(good.any(1))[0], np.nonzero(good.any(0))[0]
        crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        layers, good = layers[:, crop[0], crop[1]], good[crop]
        r0, c0 = window[0] + int(rows[0]), window[2] + int(cols[0])
        slices = []
        if a.xy_slices:
            zs = spread(0, vol.shape[0], a.xy_slices)
            for z in zs:
                img = np.asarray(vol.reader.read((slice(z, z + 1), slice(0, vol.shape[1]), slice(0, vol.shape[2])))[0],
                                 np.float32)
                slices.append((z * vol.s + vol.origin0[0], img))
        meta = dict(common, source="volume", volume=os.path.abspath(a.volume), level=a.level,
                    volume_origin_zyx_level0=list(vol.origin0), volume_shape=list(vol.shape), voxel_um=vol.voxel_um,
                    direction=direction, direction_source=a.direction, side_test=side, layer_offsets_voxels=offsets,
                    crop=dict(row0=r0, col0=c0, rows=int(good.shape[0]), cols=int(good.shape[1])),
                    valid_pixels=int(good.sum()),
                    xy_slice_frame="slices of the volume block: zarr (y, x) of the stored crop")
        write_set(outdir, layers, offsets, good, meta, a.u8_window, slices, a.surface_zarr)
        print(f"{good.shape[0]} x {good.shape[1]} px ({good.sum()} valid), canvas origin (row {r0}, col {c0}), "
              f"direction {direction:+d}, offsets {offsets[0]:+g}..{offsets[-1]:+g} -> {outdir} ({time.time() - t0:.0f}s)")
        return 0

    # ================================================================ from the projections (band reconstruction)
    from .band import BandComposite
    from .recon import HelicalRecon
    if params.get("mode") != "legacy_reference":   # (the legacy mode pins ghbp)
        params.setdefault("backprojector", "fbp")  # composites: the exact backprojector at the band pixels (fast path)
    r = HelicalRecon(a.projections, cache_dir=a.cache_dir, device=a.device, params=params, log=lambda *x: None)
    N = r.info["grid_size"]
    z0, z1 = a.z if a.z else (0, r.hi - r.lo)
    z0, z1 = max(0, z0), min(r.hi - r.lo, z1)
    h0, h1 = z0 + r.lo, z1 + r.lo
    window, _ = uv_window(surf, None, max(reach, 15.0), ps, box=((z0, 0, 0), (z1 - 1, N - 1, N - 1)))
    sy, sx = surf.scale
    ky, kx = ps
    raster = (window[2] * sx / kx, window[0] * sy / ky, sx / kx, sy / ky, window[3] - window[2], window[1] - window[0])
    P, ok, nrm = _points(surf, raster, a.interpolation)
    # offsets to render: the layers for the given direction, or both directions + the side test's -5..5
    if a.direction == "auto":
        render = sorted(set(layer_offsets(+1, a.start, a.step, a.layers)) | set(layer_offsets(-1, a.start, a.step, a.layers))
                        | {float(d) for d in range(-5, 6)})
    else:
        render = layer_offsets(int(a.direction), a.start, a.step, a.layers)
    pts = np.stack([P + o * nrm for o in render]).astype(np.float32)
    K = len(sweep) if sweep else 1
    t = time.time()
    bc = BandComposite(r, pts, ok, h_range=(h0, h1), n_sets=K)
    del pts
    chunk = a.chunk or (40 if K > 1 else 64)
    print(f"surface {surf.uuid}: window rows {window[:2]} cols {window[2:]}, {bc.n_points} points, "
          f"z {z0}..{z1 - 1} (h {h0}..{h1 - 1}), {K} parameter set(s), chunk {chunk}", flush=True)
    sec = bc.run(chunk, sweep=sweep,
                 progress=lambda d, n, el: print(f"  {d}/{n} slices, {el:.0f}s", flush=True))
    print(f"band reconstruction {sec['wall']:.1f}s ({1e3 * sec['wall'] / (h1 - h0):.1f} ms/slice)", flush=True)
    good_all = bc.valid
    rows, cols = np.nonzero(good_all.any(1))[0], np.nonzero(good_all.any(0))[0]
    if not len(rows):
        sys.exit("no surface pixel has all its layers inside the z range")
    crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    r0, c0 = window[0] + int(rows[0]), window[2] + int(cols[0])
    bx0, bx1, by0, by1 = bc.sampler.box if bc.band else bc.box
    m = a.slice_margin
    slice_roi = (0, N, 0, N) if m < 0 else (max(0, bx0 - m), min(N, bx1 + m), max(0, by0 - m), min(N, by1 + m))
    zs = spread(z0, z1, a.xy_slices)
    sets = sweep or [None]
    for k, ps_k in enumerate(sets):
        lay_all = bc.layers(k)[:, crop[0], crop[1]]
        side = None
        if a.direction == "auto":
            direction, side = side_from_layers(lay_all, render)
        else:
            direction = int(a.direction)
        offsets = layer_offsets(direction, a.start, a.step, a.layers)
        idx = [render.index(o) for o in offsets]
        layers = lay_all[idx]
        good = good_all[crop]
        slices = []
        if zs:
            if ps_k is not None:
                r.set_params(**ps_k)
            for z in zs:
                slices.append((z, r.reconstruct_heights(z + r.lo, z + r.lo + 1, roi=slice_roi)[0]))
        out_k = outdir if ps_k is None else os.path.join(outdir, set_name(ps_k))
        os.makedirs(out_k, exist_ok=True)
        meta = dict(common, source="projections", projections=os.path.abspath(a.projections),
                    params=r.params, provenance=r.provenance(), sweep_set=ps_k,
                    filters=r.chains([ps_k or {}])[0], sweep_margin_rows=r.rec.sweep_margin(r.chains(sweep)) if sweep else None, z=[z0, z1], view_heights=[h0, h1],
                    backprojector=r.params["backprojector"],
                    composite_path="band (exact backprojector at the layer pixels)" if bc.band else "ROI slices + trilinear sampling",
                    direction=direction, direction_source=a.direction, side_test=side, layer_offsets_voxels=offsets,
                    crop=dict(row0=r0, col0=c0, rows=int(good.shape[0]), cols=int(good.shape[1])),
                    valid_pixels=int(good.sum()), band_box_x0x1y0y1=[bx0, bx1, by0, by1],
                    xy_slice_roi_x0x1y0y1=list(slice_roi), seconds_band=round(sec["wall"], 2))
        write_set(out_k, layers, offsets, good, meta, a.u8_window, slices, a.surface_zarr)
        print(f"{'set ' + set_name(ps_k) + ': ' if ps_k else ''}{good.shape[0]} x {good.shape[1]} px ({good.sum()} valid), "
              f"canvas origin (row {r0}, col {c0}), direction {direction:+d}, {len(slices)} xy slices -> {out_k}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
