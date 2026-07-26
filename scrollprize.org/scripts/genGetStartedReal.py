# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "tifffile", "zarr<3", "fsspec", "s3fs", "aiohttp"]
# ///
"""
Build the REAL /get_started demo assets from a tifxyz surface + volume +
vc_render_tifxyz output. Produces the exact same manifests/files as
scripts/genGetStartedMock.py, so the demo components need zero changes.

Typical use (seg demo), once you have a render stack for the chosen window:

  uv run scripts/genGetStartedReal.py seg \
    --tifxyz /path/to/w00_tifxyz \
    --window 2400 3400 1200 2000 \
    --renders /path/to/w00_fullres_render \
    --offsets 0 8 16 32 \
    --volume s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/ \
    --volume-level 2

Ink demo (papyrus/label/prediction must share the same (u,v) window):

  uv run scripts/genGetStartedReal.py ink \
    --papyrus render_center.tif --label label.png --prediction pred.png

Geometry check only (no volume needed — verifies the tifxyz window and
prints/plots the three slice polylines):

  uv run scripts/genGetStartedReal.py check --tifxyz /path/to/w00_tifxyz \
    --window 2400 3400 1200 2000

===========================================================================
How the pieces map (see also genGetStartedMock.py for the JSON contract)
===========================================================================
tifxyz: x/y/z.tif are float32 (V,U) grids; pixel (v,u) holds the volume
coordinate of that surface point; -1 marks invalid cells. meta.json "scale"
is the grid resolution relative to the volume (0.1 => one grid step ~ 10
full-res voxels along the surface).

Seg demo:
  * The user-facing cross-section for a z-plane z0 is the volume slice at
    z0, cropped to the polyline's xy bounding box (plus margin), downsampled
    to --slice-size.
  * The GT polyline is the intersection of the surface with z=z0: for each
    grid column u in the window we find the v where z(u,v) crosses z0
    (linear interp), then take (x,y) there -> slice pixel coords.
  * arc fraction t = (u - u0) / (u1 - u0) maps a dot to render x — the same
    u axis vc_render_tifxyz uses for its output width.
  * Renders: --renders points at the vc_render_tifxyz output for THE SAME
    tifxyz. Slices are numbered by normal offset (center slice = offset 0);
    we take max over a couple of slices around each requested offset,
    crop to the (u,v) window (scaled by --scale used at render time), and
    resize to --render-size.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# ---------------------------------------------------------------------------


def load_tifxyz(d: Path):
    x = tifffile.imread(d / "x.tif")
    y = tifffile.imread(d / "y.tif")
    z = tifffile.imread(d / "z.tif")
    meta = json.loads((d / "meta.json").read_text())
    return x, y, z, meta


def window_slice(arr, w):
    u0, u1, v0, v1 = w
    return arr[v0:v1, u0:u1]


def surface_polyline_at_z(xw, yw, zw, z0):
    """Intersect the (windowed) surface with the plane z=z0.

    Returns (pts_xy float array (N,2) in VOLUME coords, u_index array (N,),
    v_cross array (N,) — the fractional v row of the crossing).
    For each u column, find the first v where z crosses z0 and linearly
    interpolate x,y there. Columns that never cross are skipped.
    """
    V, U = zw.shape
    pts, us, vs = [], [], []
    valid = zw > 0
    for u in range(U):
        col = zw[:, u]
        ok = valid[:, u]
        # sign changes of (z - z0) between consecutive valid cells
        d = col - z0
        for v in range(V - 1):
            if not (ok[v] and ok[v + 1]):
                continue
            a, b = d[v], d[v + 1]
            if a == 0 or (a < 0) != (b < 0):
                t = 0.0 if a == 0 else a / (a - b)
                px = xw[v, u] * (1 - t) + xw[v + 1, u] * t
                py = yw[v, u] * (1 - t) + yw[v + 1, u] * t
                pts.append((px, py))
                us.append(u)
                vs.append(v + t)
                break
    return np.array(pts, dtype=np.float64), np.array(us), np.array(vs)


def fetch_plane_cached(vol, level, zi):
    """One full z-plane of a zarr level, cached to /tmp (S3 is slow)."""
    import tempfile

    cache = Path(tempfile.gettempdir()) / f"gs_plane_l{level}_z{zi}.npy"
    if cache.exists():
        return np.load(cache)
    print(f"fetching level-{level} plane z={zi} …")
    plane = np.asarray(vol[zi])
    np.save(cache, plane)
    return plane


def normalize_u8(a, lo_pct=1, hi_pct=99):
    a = a.astype(np.float32)
    finite = a[np.isfinite(a)]
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    a = np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)
    return (a * 255).astype(np.uint8)


def open_volume(path: str, level: int):
    """Open a (possibly multiscale) OME-Zarr volume, local or s3://."""
    import zarr

    if path.startswith("s3://"):
        import s3fs

        fs = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=path.replace("s3://", ""), s3=fs)
        g = zarr.open(store, mode="r")
    else:
        g = zarr.open(path, mode="r")
    # multiscale group -> pick level; plain array -> use as-is
    if hasattr(g, "shape"):
        return g, 1
    arr = g[str(level)]
    return arr, 2**level


# ---------------------------------------------------------------------------


def cmd_check(args):
    d = Path(args.tifxyz)
    x, y, z, meta = load_tifxyz(d)
    print(f"tifxyz {d.name}: grid (V,U)={z.shape}, scale={meta.get('scale')}")
    w = args.window or [0, z.shape[1], 0, z.shape[0]]
    xw, yw, zw = (window_slice(a, w) for a in (x, y, z))
    valid = zw > 0
    print(
        f"window u[{w[0]}:{w[1]}] v[{w[2]}:{w[3]}] -> {valid.mean() * 100:.1f}% valid"
    )
    if not valid.any():
        sys.exit("window has no valid surface — pick another")
    zmin, zmax = zw[valid].min(), zw[valid].max()
    print(f"z range in window: {zmin:.0f}..{zmax:.0f}")
    for frac, name in [(0.25, "top"), (0.5, "middle"), (0.75, "bottom")]:
        z0 = zmin + (zmax - zmin) * frac
        pts, us, _vs = surface_polyline_at_z(xw, yw, zw, z0)
        if len(pts) < 10:
            print(f"  {name}: z={z0:.0f} -> only {len(pts)} pts (too few!)")
            continue
        span = np.ptp(pts, axis=0)
        print(
            f"  {name}: z={z0:.0f} -> {len(pts)} pts, xy span {span[0]:.0f}x{span[1]:.0f} vx, u {us.min()}..{us.max()}"
        )


def cmd_seg(args):
    out = Path(args.out) / "seg"
    out.mkdir(parents=True, exist_ok=True)
    d = Path(args.tifxyz)
    x, y, z, meta = load_tifxyz(d)
    w = args.window
    xw, yw, zw = (window_slice(a, w) for a in (x, y, z))
    valid = zw > 0
    zmin, zmax = zw[valid].min(), zw[valid].max()

    vol = None
    ds = 1
    if args.volume:
        vol, ds = open_volume(args.volume, args.volume_level)
        print(f"volume open: shape={vol.shape}, downsample={ds}x")

    S = args.slice_size
    RW, RH = args.render_size
    # z DECREASES along +v for this segment, and the flattened reference
    # (where the text reads upright) has v increasing downward — so the
    # image-top band is HIGH z. Name the planes to match: "top" = high z.
    # The turntable projection flips dz the same way (text-up = screen-up).
    if args.plane_z:
        zs = [float(v) for v in args.plane_z]
    else:
        fr = args.plane_fracs
        zs = [float(zmin + (zmax - zmin) * f) for f in fr]
    assert zs[0] > zs[1] > zs[2], "planes are top,middle,bottom = high..low z"
    names = ["top", "middle", "bottom"]
    polys = []
    for z0 in zs:
        pts, us, vs = surface_polyline_at_z(xw, yw, zw, z0)
        assert len(pts) > 20, f"slice z={z0}: too few polyline points"
        polys.append((pts, us, vs))

    # crop box, shared by all three slices so the view doesn't jump when
    # the plane changes. --full-slice: the whole scroll cross-section
    # (nonzero bbox of the CT planes); otherwise the polyline bbox.
    slice_vol, sds = (None, 1)
    if args.volume:
        slice_vol, sds = open_volume(args.volume, args.slice_level)
        print(f"slice volume: shape={slice_vol.shape}, downsample={sds}x")
    planes_px = {}
    if args.full_slice:
        assert slice_vol is not None, "--full-slice needs --volume"
        bx0 = by0 = np.inf
        bx1 = by1 = -np.inf
        for z0 in zs:
            zi = int(round(z0 / sds))
            plane = fetch_plane_cached(slice_vol, args.slice_level, zi)
            planes_px[z0] = plane
            ys, xs = np.where(plane > 0)
            by0, by1 = min(by0, ys.min()), max(by1, ys.max())
            bx0, bx1 = min(bx0, xs.min()), max(bx1, xs.max())
        pad = 24  # slice-level px
        cx0, cy0 = float(bx0 - pad) * sds, float(by0 - pad) * sds
        side = float(max(bx1 - bx0, by1 - by0) + 2 * pad) * sds
        cx0 = (cx0 + float(bx1 + pad) * sds - side) / 2
        cy0 = (cy0 + float(by1 + pad) * sds - side) / 2
    else:
        allpts = np.concatenate([p for p, _, _ in polys])
        margin = args.margin
        cx0, cy0 = allpts.min(0) - margin
        cx1, cy1 = allpts.max(0) + margin
        side = max(cx1 - cx0, cy1 - cy0)
        cx0 = (cx0 + cx1 - side) / 2
        cy0 = (cy0 + cy1 - side) / 2
    vox_per_px = side / S
    print(f"slice crop: origin ({cx0:.0f},{cy0:.0f}) side {side:.0f} vox "
          f"({vox_per_px:.1f} vox/px)")

    # one tone curve for all slice/detail images (nonzero pixels only, so
    # the black outside doesn't skew the percentiles)
    lo = hi = None
    if planes_px:
        samp = np.concatenate(
            [p[p > 0][:: max(1, p.size // 2_000_000)] for p in planes_px.values()]
        )
        lo, hi = np.percentile(samp, [1, 99])

    detail_vol, dds = (None, 1)
    if args.volume and args.detail_size:
        detail_vol, dds = open_volume(args.volume, args.detail_level)

    slices_meta = []
    for i, (name, z0, (pts, us, vs)) in enumerate(zip(names, zs, polys)):
        name_img = f"slice_{name}.webp"
        entry = {
            "id": name,
            "label": name.capitalize(),
            "image": name_img,
            "width": S,
            "height": S,
        }
        if args.full_slice:
            zi = int(round(z0 / sds))
            x0i, y0i = int(cx0 / sds), int(cy0 / sds)
            n = int(round(side / sds))
            plane = planes_px[z0][y0i : y0i + n, x0i : x0i + n]
            a = np.clip((plane.astype(np.float32) - lo) / max(1e-6, hi - lo), 0, 1)
            img = Image.fromarray((a * 255).astype(np.uint8)).resize(
                (S, S), Image.LANCZOS
            )
            img.save(out / name_img, "WEBP", quality=args.slice_quality, method=6)
        elif vol is not None:
            zi = int(round(z0 / ds))
            y0i, y1i = int(cy0 / ds), int((cy0 + side) / ds)
            x0i, x1i = int(cx0 / ds), int((cx0 + side) / ds)
            plane = np.asarray(vol[zi, y0i:y1i, x0i:x1i])
            img = Image.fromarray(normalize_u8(plane)).resize((S, S), Image.LANCZOS)
            img.save(out / name_img, "WEBP", quality=80, method=6)
        else:
            print(f"  [no volume] skipping {name_img} — placeholder kept if present")

        # hi-res inset around the trace: crisp pixels where the user works,
        # while the full slice provides the context
        if detail_vol is not None and args.full_slice:
            dmargin = args.detail_margin
            dx0, dy0 = pts.min(0) - dmargin
            dx1, dy1 = pts.max(0) + dmargin
            dside = max(dx1 - dx0, dy1 - dy0)
            dx0 = (dx0 + dx1 - dside) / 2
            dy0 = (dy0 + dy1 - dside) / 2
            zi = int(round(z0 / dds))
            import tempfile, hashlib

            key = hashlib.md5(
                f"det|{args.detail_level}|{zi}|{dx0:.0f}|{dy0:.0f}|{dside:.0f}".encode()
            ).hexdigest()[:16]
            cache = Path(tempfile.gettempdir()) / f"gs_detail_{key}.npy"
            if cache.exists():
                det = np.load(cache)
            else:
                print(f"fetching detail {name} z={zi} …")
                det = np.asarray(
                    detail_vol[
                        zi,
                        int(dy0 / dds) : int((dy0 + dside) / dds),
                        int(dx0 / dds) : int((dx0 + dside) / dds),
                    ]
                )
                np.save(cache, det)
            a = np.clip((det.astype(np.float32) - lo) / max(1e-6, hi - lo), 0, 1)
            D = args.detail_size
            dimg = f"detail_{name}.webp"
            Image.fromarray((a * 255).astype(np.uint8)).resize(
                (D, D), Image.LANCZOS
            ).save(out / dimg, "WEBP", quality=args.detail_quality, method=6)
            entry["detail"] = {
                "image": dimg,
                "x": round((dx0 - cx0) / vox_per_px, 1),
                "y": round((dy0 - cy0) / vox_per_px, 1),
                "side": round(dside / vox_per_px, 1),
            }

        # polyline in slice pixel coords; subsample to ~140 points
        px = (pts[:, 0] - cx0) / vox_per_px
        py = (pts[:, 1] - cy0) / vox_per_px
        step = max(1, len(px) // 140)
        surface = [
            [round(float(a), 1), round(float(b), 1)]
            for a, b in zip(px[::step], py[::step])
        ]
        entry.update(
            {
                "surface": surface,
                "_vMean": round(float(vs.mean()), 1),
                # bookkeeping for reproducibility (ignored by the component):
                "_z0": z0,
                "_cropXY": [round(cx0, 1), round(cy0, 1), round(side, 1)],
                "_uRange": [int(us.min()) + w[0], int(us.max()) + w[0]],
            }
        )
        slices_meta.append(entry)

    # vBand: even thirds. A z plane's true crossing row varies by ±16 grid
    # rows across the window (the sheet undulates), so any exact mapping is
    # fuzzy anyway — even bands read best as trace-progress feedback.
    for i, sl in enumerate(slices_meta):
        sl["vBand"] = [i * RH // 3, (i + 1) * RH // 3]

    # renders: either the reference crop + volume-sampled normal offsets
    # (real data end-to-end), or crops out of a vc_render_tifxyz stack
    images = []
    if args.render_ref:
        # offset 0 = the real flattened surface render (reference crop,
        # full resolution); offsets > 0 = the volume sampled at
        # surface + offset * normal — what stepping off the sheet actually
        # returns (off-center papyrus -> air gap -> the neighboring wrap).
        # --render-window pins the renders to the reference crop's (u,v)
        # extent when the slice window is taller than it.
        ref = tifffile.imread(args.render_ref)
        Image.fromarray(normalize_u8(ref)).resize((RW, RH), Image.LANCZOS).save(
            out / "render_o0.webp", "WEBP", quality=84
        )
        images.append("render_o0.webp")
        offs = [o for o in args.offsets if o != 0]
        rvol, rds = open_volume(args.volume, args.render_level)
        print(f"render volume open: shape={rvol.shape}, downsample={rds}x")
        rw = args.render_window or w
        xr, yr, zr = (window_slice(a, rw) for a in (x, y, z))
        rmaps = render_offsets_from_volume(xr, yr, zr, offs, RW, RH, rvol, rds)
        lo, hi = np.percentile(np.concatenate([m.ravel() for m in rmaps.values()]), [1, 99])
        for o in offs:
            a = np.clip((rmaps[o] - lo) / max(1e-6, hi - lo), 0, 1)
            name_img = f"render_o{o}.webp"
            Image.fromarray((a * 255).astype(np.uint8)).save(
                out / name_img, "WEBP", quality=82
            )
            images.append(name_img)
        assert args.offsets[0] == 0, "offset 0 must come first (reference crop)"
    elif args.renders:
        rdir = Path(args.renders)
        tifs = sorted(rdir.glob("*.tif")) or sorted(rdir.glob("*.png"))
        assert tifs, f"no render slices found in {rdir}"
        center = len(tifs) // 2
        rs = args.render_scale  # scale used at render time vs tifxyz grid
        u0, u1, v0, v1 = (int(c * rs) for c in w)
        for off in args.offsets:
            picks = [center + off, center + off + 1] if off else [center - 1, center, center + 1]
            picks = [p for p in picks if 0 <= p < len(tifs)]
            stack = [tifffile.imread(tifs[p])[v0:v1, u0:u1] for p in picks]
            arr = np.maximum.reduce(stack)
            name_img = f"render_o{off}.webp"
            Image.fromarray(normalize_u8(arr)).resize((RW, RH), Image.LANCZOS).save(
                out / name_img, "WEBP", quality=84
            )
            images.append(name_img)
    else:
        images = [f"render_o{o}.webp" for o in args.offsets]
        print("[no renders] manifest will reference render_o*.webp — supply them")

    tol_px = (
        args.tolerance_vox / vox_per_px if args.tolerance_vox else args.tolerance
    )
    max_px = (
        args.max_dist_vox / vox_per_px if args.max_dist_vox else args.max_dist
    )
    manifest = {
        "version": 1,
        "dotRadius": 9,
        "tolerancePx": round(tol_px, 2),
        "maxDistPx": round(max_px, 2),
        "introZoom": args.intro_zoom,
        "maxZoom": args.max_zoom,
        "slices": slices_meta,
        "render": {"width": RW, "height": RH, "offsets": args.offsets, "images": images},
    }
    print(f"tolerance {tol_px:.1f}px / maxDist {max_px:.1f}px "
          f"({args.tolerance_vox or '?'} / {args.max_dist_vox or '?'} vox)")
    # keep an existing turntable block (generated separately) alive, but
    # recompute its plane screenYs — the sprite geometry doesn't change
    # when the plane z0s do
    mf_path = out / "manifest.json"
    if mf_path.exists():
        old = json.loads(mf_path.read_text())
        if "turntable" in old:
            tt = old["turntable"]
            cphi = np.cos(np.deg2rad(tt.get("tiltDeg", 0)))
            tt["planes"] = [
                {
                    "id": sl["id"],
                    "screenY": round(
                        tt["tileH"] / 2
                        - (sl["_z0"] - tt["center"][2])
                        * tt.get("zStretch", 1)
                        * cphi
                        * tt["scale"],
                        1,
                    ),
                }
                for sl in slices_meta
            ]
            manifest["turntable"] = tt
    mf_path.write_text(json.dumps(manifest, indent=2))
    print(f"seg manifest written -> {out}")


def render_offsets_from_volume(xw, yw, zw, offsets, RW, RH, vol, ds):
    """Sample the volume at surface + offset*normal on an (RH, RW) grid.

    One box fetch covers all offsets (cached), then nearest-neighbor
    gathers per offset. Offsets are in FULL-RES voxels.
    """
    def up(a):
        return np.asarray(
            Image.fromarray(a.astype(np.float32), "F").resize((RW, RH), Image.BICUBIC)
        )

    X, Y, Z = up(xw), up(yw), up(zw)
    tu = np.stack([np.gradient(a, axis=1) for a in (X, Y, Z)], -1)
    tv = np.stack([np.gradient(a, axis=0) for a in (X, Y, Z)], -1)
    n = np.cross(tu, tv)
    n /= np.maximum(1e-6, np.linalg.norm(n, axis=-1, keepdims=True))
    P0 = np.stack([X, Y, Z], -1)

    omax = max(abs(o) for o in offsets)
    lo3 = (P0.reshape(-1, 3).min(0) - omax) / ds
    hi3 = (P0.reshape(-1, 3).max(0) + omax) / ds
    x0, y0, z0 = (max(0, int(v) - 1) for v in lo3)
    x1 = min(vol.shape[2], int(hi3[0]) + 2)
    y1 = min(vol.shape[1], int(hi3[1]) + 2)
    z1 = min(vol.shape[0], int(hi3[2]) + 2)
    import hashlib, tempfile

    key = hashlib.md5(f"ro|{ds}|{z0}:{z1}|{y0}:{y1}|{x0}:{x1}".encode()).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / f"gs_rocrop_{key}.npy"
    if cache.exists():
        sub = np.load(cache)
        print(f"render box loaded from cache {cache}")
    else:
        print(f"fetching render box z {z0}:{z1} y {y0}:{y1} x {x0}:{x1} "
              f"({(z1 - z0) * (y1 - y0) * (x1 - x0) / 1e6:.0f} Mvox)")
        sub = np.asarray(vol[z0:z1, y0:y1, x0:x1])
        np.save(cache, sub)

    out = {}
    for o in offsets:
        P = P0 + o * n
        xs = np.clip((P[..., 0] / ds).round().astype(int) - x0, 0, sub.shape[2] - 1)
        ys = np.clip((P[..., 1] / ds).round().astype(int) - y0, 0, sub.shape[1] - 1)
        zs = np.clip((P[..., 2] / ds).round().astype(int) - z0, 0, sub.shape[0] - 1)
        out[o] = sub[zs, ys, xs].astype(np.float32)
        print(f"  offset {o}: mean {out[o].mean():.1f}")
    return out


def cmd_unroll(args):
    """Export a decimated 3D grid of the tifxyz surface for the seg demo's
    done-state unroll animation (seg/unroll.json). The texture is the ink
    demo's papyrus.webp — the same reference crop, already in cache when
    the visitor gets here. Points are absolute volume coordinates; the JS
    projects them with the same flipped-z convention as the turntable.
    """
    out = Path(args.out) / "seg"
    out.mkdir(parents=True, exist_ok=True)
    x, y, z, _meta = load_tifxyz(Path(args.tifxyz))
    w = args.window or [0, z.shape[1], 0, z.shape[0]]
    xw, yw, zw = (window_slice(a, w) for a in (x, y, z))
    assert (zw > 0).all(), "unroll window must be fully valid"
    R, C = args.grid
    vi = np.linspace(0, zw.shape[0] - 1, R).round().astype(int)
    ui = np.linspace(0, zw.shape[1] - 1, C).round().astype(int)
    P = np.stack([a[np.ix_(vi, ui)] for a in (xw, yw, zw)], -1)
    flat = P.reshape(-1, 3)
    center = (flat.min(0) + flat.max(0)) / 2
    # mean arc step (voxels) between column samples -> true flat aspect
    du = np.linalg.norm(np.diff(P, axis=1), axis=-1).mean()
    dv = np.linalg.norm(np.diff(P, axis=0), axis=-1).mean()
    data = {
        "rows": R,
        "cols": C,
        "center": [round(float(v), 1) for v in center],
        "flatAspect": round(float((du * (C - 1)) / (dv * (R - 1))), 3),
        "pts": [[round(float(v), 1) for v in p] for p in flat],
    }
    (out / "unroll.json").write_text(json.dumps(data, separators=(",", ":")))
    kb = (out / "unroll.json").stat().st_size / 1024
    print(f"unroll grid written -> {out}/unroll.json ({R}x{C}, {kb:.0f} KB)")


def cmd_turntable(args):
    """Render the windowed surface as a textured turntable sprite.

    Samples the volume at every (valid) surface grid point for texture,
    shades with a simple lambert term from grid normals, and z-buffers a
    point splat per frame while rotating about the volume z axis (the
    scroll axis, vertical on screen). Output: seg/turntable.webp (a
    cols x rows sprite grid, RGBA) + a "turntable" block merged into the
    existing seg/manifest.json:

      turntable: { sprite, frames, cols, tileW, tileH, thetaStepDeg,
                   center [cx,cy,cz] (volume coords), scale (px/voxel),
                   planes [{id, z0, screenY}] }

    The camera is tilted down by --tilt-deg so z-planes read as phantom
    ellipses (not lines); planeRx/planeRy in the manifest give their pixel
    radii. z is NEGATED in the projection: this segment's text-up direction
    is -z, so high z at screen top keeps the (eventual) render upright and
    the plane order matching the flattened image. JS projection of a volume
    point (X,Y,Z) at frame f (theta = f * step, phi = tilt):
      dx, dy, dz = X-cx, Y-cy, Z-cz
      depth = -dx*sin(th) + dy*cos(th)   # > 0 -> far side of the sheet
      sx = tileW/2 + (dx*cos(th) + dy*sin(th)) * scale
      sy = tileH/2 + (-dz*cos(phi) - depth*sin(phi)) * scale

    Render a WIDER --window than the demo's trace window (2+ windings) so
    the sheet reads as a spiral; dots still project correctly because
    everything is in absolute volume coordinates.
    """
    out = Path(args.out) / "seg"
    out.mkdir(parents=True, exist_ok=True)
    d = Path(args.tifxyz)
    x, y, z, _meta = load_tifxyz(d)
    w = args.window
    step = args.grid_step
    xw, yw, zw = (window_slice(a, w)[::step, ::step] for a in (x, y, z))

    # crop the v range to the z band first (before any upsampling): the
    # band is a thin ribbon of the full sheet, no point carrying the rest
    if args.z_band:
        zlo, zhi = args.z_band
        band = (zw >= zlo) & (zw <= zhi) & (zw > 0)
        rows = np.where(band.any(1))[0]
        v0b = max(0, int(rows.min()) - 2)
        v1b = min(zw.shape[0], int(rows.max()) + 3)
        xw, yw, zw = xw[v0b:v1b], yw[v0b:v1b], zw[v0b:v1b]
        print(f"z band v rows {v0b}..{v1b} of the window")
    vmask = (zw > 0).astype(np.float32)

    # a partly-valid parent grid + v upsampling need a clean validity mask:
    # upsample the mask alongside the coordinate grids and only keep cells
    # whose whole interpolation footprint was valid
    if args.upsample_v > 1:
        Vu = zw.shape[0] * args.upsample_v
        Uu = zw.shape[1]

        def upv(a):
            return np.asarray(
                Image.fromarray(a.astype(np.float32), "F").resize(
                    (Uu, Vu), Image.BILINEAR
                )
            )

        xw, yw, zw = upv(xw), upv(yw), upv(zw)
        vmask = upv(vmask)
    valid = vmask > 0.999
    print(f"grid {zw.shape}, {valid.mean() * 100:.1f}% valid")

    # z band: keep only the ribbon of windings around the demo's planes —
    # a thin slice of the roll that the projection then stretches tall
    if args.z_band:
        zlo, zhi = args.z_band
        valid &= (zw >= zlo) & (zw <= zhi)
        print(f"z band {zlo}..{zhi}: {valid.mean() * 100:.1f}% of grid kept")

    # normals from grid tangents (window is expected ~100% valid)
    tu = np.stack(
        [np.gradient(a, axis=1) for a in (xw, yw, zw)], axis=-1
    )
    tv = np.stack(
        [np.gradient(a, axis=0) for a in (xw, yw, zw)], axis=-1
    )
    n = np.cross(tu, tv)
    n /= np.maximum(1e-6, np.linalg.norm(n, axis=-1, keepdims=True))

    # texture: nearest-neighbor volume sample at each surface point
    vol, ds = open_volume(args.volume, args.volume_level)
    print(f"volume open: shape={vol.shape}, downsample={ds}x")
    xi = np.clip((xw / ds).round().astype(int), 0, vol.shape[2] - 1)
    yi = np.clip((yw / ds).round().astype(int), 0, vol.shape[1] - 1)
    zi = np.clip((zw / ds).round().astype(int), 0, vol.shape[0] - 1)
    z0, z1 = zi[valid].min(), zi[valid].max() + 1
    y0, y1 = yi[valid].min(), yi[valid].max() + 1
    x0, x1 = xi[valid].min(), xi[valid].max() + 1
    print(f"texture bounds: z {z0}:{z1}  y {y0}:{y1}  x {x0}:{x1}")
    import hashlib, tempfile
    key = hashlib.md5(
        (f"points-v2|{args.volume}|{args.volume_level}|{w}|{step}|"
         f"{args.z_band}|{args.upsample_v}").encode()
    ).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / f"gs_ttpoints_{key}.npy"
    if cache.exists():
        samples = np.load(cache)
        print(f"texture samples loaded from cache {cache}")
    else:
        # The surface winds through a large 3-D bounding box but touches only
        # a thin set of volume chunks. Reading the whole box is both slow and
        # needlessly memory hungry, especially at levels 2 and 3. Group the
        # requested points by chunk and fetch only chunks intersected by the
        # actual tifxyz surface.
        zv, yv, xv = zi[valid], yi[valid], xi[valid]
        chunks = getattr(vol, "chunks", None)
        if not chunks or len(chunks) != 3:
            samples = np.asarray(vol.vindex[zv, yv, xv], dtype=np.float32)
        else:
            cz, cy, cx = (int(v) for v in chunks)
            shape = tuple(int(v) for v in vol.shape)
            ncx = (shape[2] + cx - 1) // cx
            ncy = (shape[1] + cy - 1) // cy
            chunk_id = (zv // cz) * (ncy * ncx) + (yv // cy) * ncx + xv // cx
            order = np.argsort(chunk_id, kind="stable")
            ids = chunk_id[order]
            cuts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, len(ids)]
            samples = np.empty(len(zv), dtype=np.float32)
            print(f"texture sampling: {len(cuts) - 1} touched chunks of {chunks}")
            for j, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
                idx = order[a:b]
                zz, yy, xx = int(zv[idx[0]] // cz), int(yv[idx[0]] // cy), int(xv[idx[0]] // cx)
                zs, ys, xs = zz * cz, yy * cy, xx * cx
                block = np.asarray(vol[
                    zs:min(zs + cz, shape[0]),
                    ys:min(ys + cy, shape[1]),
                    xs:min(xs + cx, shape[2]),
                ])
                samples[idx] = block[zv[idx] - zs, yv[idx] - ys, xv[idx] - xs]
                if j % 100 == 0:
                    print(f"  texture chunks {j}/{len(cuts) - 1}")
        np.save(cache, samples)
    tex = np.zeros(zw.shape, np.float32)
    tex[valid] = samples
    lo, hi = np.percentile(tex[valid], [2, 98])
    tex = np.clip((tex - lo) / max(1e-6, hi - lo), 0, 1)

    # cutaway: cap the screen-top (HIGH z, with the flipped projection)
    # progressively with arc distance from the traced window, so the sheet
    # descends like a helical ramp as it spirals away — every winding's top
    # edge becomes visible and the roll reads as a spiral, with the traced
    # winding standing tallest. Applied AFTER texture sampling so the point
    # cache (keyed without cutaway args) stays valid for any cutaway.
    if args.cutaway > 0 and args.trace_window:
        keep = valid.copy()
        zmin_, zmax_ = zw[valid].min(), zw[valid].max()
        ug = np.arange(w[0], w[1], step, dtype=np.float64)
        t0, t1 = args.trace_window
        # each side descends to the full cutaway by its own end of the
        # spiral (the sides have different spans), capped by --cutaway-ramp
        lspan = min(float(t0 - ug.min()), args.cutaway_ramp)
        rspan = min(float(ug.max() - t1), args.cutaway_ramp)
        ramp = np.maximum(
            np.clip((t0 - ug) / max(1.0, lspan), 0, 1),
            np.clip((ug - t1) / max(1.0, rspan), 0, 1),
        )
        zfrac = (zw - zmin_) / max(1e-6, zmax_ - zmin_)
        valid &= zfrac <= 1 - args.cutaway * ramp[None, :]
        print(f"cutaway: {100 * (1 - valid.mean() / max(1e-9, keep.mean())):.1f}%"
              " of band points shaved")

    # flatten valid points
    P = np.stack([xw[valid], yw[valid], zw[valid]], axis=1)
    N = n[valid]
    T = tex[valid]
    # color: each winding gets its own tint so the wraps are unmistakable —
    # the traced winding keeps the warm real-papyrus tone and its neighbors
    # alternate lighter / darker cool greys along the spiral. Winding id
    # comes from the unwrapped per-column angle about the roll axis.
    colf = np.tile(np.array([[0.62, 0.62, 0.66]], np.float32), (len(T), 1))
    if args.trace_window:
        cxg, cyg = xw[valid].mean(), yw[valid].mean()
        zc = np.exp(1j * np.arctan2(yw - cyg, xw - cxg))
        zc[~valid] = 0
        colang = np.unwrap(np.angle(zc.sum(0)))
        # wraps counted from the sheet's OUTER edge so the whole outermost
        # wrap is one uniform color: find which u end sits at larger radius,
        # then accumulate turns monotonically inward (kills unwrap jitter on
        # sparse columns, which otherwise splits a wrap into stray ids)
        rr = np.hypot(xw - cxg, yw - cyg)[valid]
        un_pts = np.broadcast_to(
            np.linspace(0, 1, len(colang))[None, :], zw.shape
        )[valid]
        # same signal the winding spread uses: radius shrinking with u
        # means the u=0 end is the outer edge of the sheet
        outer_first = float(np.corrcoef(un_pts, rr)[0, 1]) < 0
        j_out = 0 if outer_first else len(colang) - 1
        j_in = len(colang) - 1 - j_out
        turns = (colang - colang[j_out]) * np.sign(
            colang[j_in] - colang[j_out]
        )
        if outer_first:
            turns = np.maximum.accumulate(turns)
        else:
            turns = np.maximum.accumulate(turns[::-1])[::-1]
        wid_col = np.floor(np.maximum(0, turns) / (2 * np.pi)).astype(int)
        wid = np.broadcast_to(wid_col[None, :], zw.shape)[valid]
        pap = np.array([1.0, 0.87, 0.78], np.float32)
        blue = np.array([0.42, 0.50, 0.72], np.float32)
        lightg = np.array([0.85, 0.87, 0.95], np.float32)
        colf[:] = lightg
        colf[wid == 1] = pap
        colf[wid >= 2] = blue
        print(f"winding ids: {np.unique(wid).tolist()}, outer_first={outer_first}")
    # stylized exploded spiral: push points radially in proportion to how
    # far along the wrap (u) they sit, so the gaps between windings widen
    # and the roll reads as a spiral instead of a solid block. Signed to
    # follow the sheet's natural outward direction; purely cosmetic (the
    # demo no longer projects volume points onto the sprite in JS).
    if args.winding_spread:
        ug_all = np.broadcast_to(
            np.arange(w[0], w[1], step, dtype=np.float64)[None, :], zw.shape
        )
        un = (ug_all[valid] - w[0]) / max(1.0, w[1] - w[0] - step)
        cxy = P[:, :2].mean(0)
        d = P[:, :2] - cxy
        r = np.hypot(d[:, 0], d[:, 1])
        sgn = float(np.sign(np.corrcoef(un, r)[0, 1])) or 1.0
        P[:, :2] = cxy + d * (
            1 + args.winding_spread * sgn * (un - 0.5)
        )[:, None]
        print(f"winding spread {args.winding_spread} (sgn {sgn:+.0f})")
    center = (P.min(0) + P.max(0)) / 2
    dxy = P[:, : 2] - center[:2]
    dz = P[:, 2] - center[2]
    r_xy = np.hypot(dxy[:, 0], dxy[:, 1]).max()  # true max radius
    phi = np.deg2rad(args.tilt_deg)
    cphi, sphi = np.cos(phi), np.sin(phi)
    TW, TH = args.tile
    # z stretch: a thin z band still reads as a tall scroll — "auto" picks
    # the factor that fills the tile height. The manifest records it so the
    # JS projection of dots/planes matches the sprite exactly.
    if args.z_stretch == "auto":
        s = (TW / 2 - 8) / r_xy
        zstretch = max(
            1.0, (TH / 2 - 10 - r_xy * s * sphi) / (np.abs(dz).max() * cphi * s)
        )
    else:
        zstretch = float(args.z_stretch)
        s = min(
            (TW / 2 - 8) / r_xy,
            (TH / 2 - 8) / (np.abs(dz).max() * zstretch * cphi + r_xy * sphi),
        )
    dz = dz * zstretch
    print(f"scale {s:.5f} px/vox, z stretch {zstretch:.2f}x")

    frames, cols = args.frames, args.cols
    rows = (frames + cols - 1) // cols
    sprite = np.zeros((rows * TH, cols * TW, 4), np.uint8)
    for f in range(frames):
        th = 2 * np.pi * f / frames
        c, si_ = np.cos(th), np.sin(th)
        depth = -dxy[:, 0] * si_ + dxy[:, 1] * c
        sx = np.clip(
            (TW / 2 + (dxy[:, 0] * c + dxy[:, 1] * si_) * s).round().astype(int),
            0, TW - 1,
        )
        sy = np.clip(
            (TH / 2 + (-dz * cphi - depth * sphi) * s).round().astype(int),
            0, TH - 1,
        )
        lam = np.abs(N[:, 0] * -si_ + N[:, 1] * c)
        # CT intensity should carry the texture. Strong point-normal shading
        # made the surface look like gravel and amplified grid noise; retain
        # just enough directional light to make rotation legible.
        shade = np.power(T, 0.88) * (0.72 + 0.28 * lam)
        if args.depth_fade > 0:
            # depth cue: the far side recedes, so front windings and the
            # gaps between them read clearly
            dn = (depth - depth.min()) / max(1e-6, depth.max() - depth.min())
            shade = shade * (1 - args.depth_fade * dn)
        # tint floor: the winding colors must survive dark CT texture, so
        # every pixel keeps a slice of its winding's hue
        rgb = (0.22 + 0.78 * shade[:, None]) * colf
        order = np.argsort(-depth)  # far first; nearest written last
        img = np.zeros((TH, TW, 3), np.float32)
        alpha = np.zeros((TH, TW), np.uint8)
        if args.winding_spread:
            # the radial spread (and big tiles) thin the splat: write each
            # point as a 2x2 block so the stretched sheet stays watertight
            sx1 = np.minimum(TW - 1, sx + 1)
            sy1 = np.minimum(TH - 1, sy + 1)
            for syy, sxx in ((sy, sx1), (sy1, sx), (sy1, sx1)):
                img[syy[order], sxx[order]] = rgb[order]
                alpha[syy[order], sxx[order]] = 1
        img[sy[order], sx[order]] = rgb[order]
        alpha[sy[order], sx[order]] = 1
        # fill pinholes (empty px with >=3 filled 4-neighbors)
        for _ in range(2):
            acc = np.zeros_like(img)
            cnt = np.zeros((TH, TW), np.int32)
            for dyy, dxx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                sa = np.roll(alpha, (dyy, dxx), (0, 1))
                acc += np.roll(img, (dyy, dxx), (0, 1)) * sa[..., None]
                cnt += sa
            fill = (alpha == 0) & (cnt >= 3)
            img[fill] = acc[fill] / np.maximum(1, cnt[fill])[:, None]
            alpha[fill] = 1
        r, ccol = divmod(f, cols)
        tile = sprite[r * TH:(r + 1) * TH, ccol * TW:(ccol + 1) * TW]
        tile[..., :3] = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        tile[..., 3] = alpha * 255
        if f % 10 == 0:
            print(f"frame {f}/{frames}")

    Image.fromarray(sprite).save(
        out / "turntable.webp", "WEBP", quality=args.quality, method=6
    )

    mf_path = out / "manifest.json"
    mf = json.loads(mf_path.read_text())
    # plane ring centers; optionally faked apart so all three stay visible
    # and tappable when the real z planes sit within a few px of each other
    ys = [
        TH / 2 - (sl["_z0"] - float(center[2])) * zstretch * cphi * s
        for sl in mf["slices"]
    ]
    if args.plane_min_gap_px > 0 and len(ys) > 2:
        # stylized: rings evenly spaced, middle ring on the roll's vertical
        # center, so top/middle/bottom read exactly as their names
        g = args.plane_min_gap_px
        o = sorted(range(len(ys)), key=lambda i: ys[i])
        for rank, i in enumerate(o):
            ys[i] = TH / 2 + (rank - (len(o) - 1) / 2) * g
        print("plane rings spread (stylized): "
              + ", ".join(f"{v:.0f}px" for v in ys))
    mf["turntable"] = {
        "sprite": "turntable.webp",
        "frames": frames,
        "cols": cols,
        "tileW": TW,
        "tileH": TH,
        "thetaStepDeg": 360 / frames,
        "tiltDeg": args.tilt_deg,
        "center": [round(float(v), 1) for v in center],
        "scale": round(float(s), 6),
        "zStretch": round(float(zstretch), 3),
        "planeRx": round(float(r_xy * s), 1),
        "planeRy": round(float(r_xy * s * sphi), 1),
        "planes": [
            {"id": sl["id"], "screenY": round(float(ys[k]), 1)}
            for k, sl in enumerate(mf["slices"])
        ],
    }
    mf_path.write_text(json.dumps(mf, indent=2))
    kb = (out / "turntable.webp").stat().st_size / 1024
    print(f"turntable sprite written -> {out} ({frames} frames, {kb:.0f} KB)")


def cmd_ink(args):
    out = Path(args.out) / "ink"
    out.mkdir(parents=True, exist_ok=True)

    def read_any(p):
        p = Path(p)
        if p.suffix.lower() in (".tif", ".tiff"):
            return tifffile.imread(p)
        return np.asarray(Image.open(p).convert("L"))

    if args.no_normalize:
        # the reference is already tone-mapped (baked contrast/gamma/white):
        # pass it through untouched
        papyrus = read_any(args.papyrus)
        assert papyrus.dtype == np.uint8, "--no-normalize expects a uint8 source"
    else:
        papyrus = normalize_u8(read_any(args.papyrus))
    H, W = papyrus.shape
    if args.label_from_pred is not None:
        # no hand label: the model output IS the scoring mask, thresholded
        # (never displayed as ground truth). Keep the prediction raw so the
        # threshold means what it says in original pixel values.
        pred = read_any(args.prediction).astype(np.uint8)
        label = None
    else:
        pred = normalize_u8(read_any(args.prediction))
        label = read_any(args.label)
        assert label.shape == papyrus.shape, "ink inputs must share dims"
    # differently-exported crops of the same window can disagree by a
    # pixel of rounding; trim to the common shape
    dh = abs(papyrus.shape[0] - pred.shape[0])
    dw = abs(papyrus.shape[1] - pred.shape[1])
    if (dh or dw) and dh <= 2 and dw <= 2:
        H2 = min(papyrus.shape[0], pred.shape[0])
        W2 = min(papyrus.shape[1], pred.shape[1])
        papyrus = papyrus[:H2, :W2]
        pred = pred[:H2, :W2]
        H, W = H2, W2
        print(f"trimmed inputs to common {W}x{H}")
    assert papyrus.shape == pred.shape, "ink inputs must share dims"
    if args.max_width and W > args.max_width:
        s = args.max_width / W
        W2, H2 = args.max_width, int(H * s)
        papyrus = np.asarray(Image.fromarray(papyrus).resize((W2, H2), Image.LANCZOS))
        pred = np.asarray(Image.fromarray(pred).resize((W2, H2), Image.LANCZOS))
        if label is not None:
            label = np.asarray(Image.fromarray((label > 0).astype(np.uint8) * 255).resize((W2, H2), Image.NEAREST))
        H, W = H2, W2

    Image.fromarray(papyrus).save(out / "papyrus.webp", "WEBP", quality=84)
    Image.fromarray(pred).save(out / "prediction.webp", "WEBP", quality=82)
    if args.label_from_pred is not None:
        lab = (pred >= args.label_from_pred).astype(np.uint8) * 255
        print(f"label from prediction >= {args.label_from_pred}: "
              f"{(lab > 0).mean() * 100:.1f}% of px")
    else:
        lab = (label > 127).astype(np.uint8) * 255
    Image.fromarray(lab).save(out / "label.png", "PNG", optimize=True)

    # letter bboxes = connected components of the label (8-connectivity)
    boxes = connected_boxes(lab > 0, min_area=args.min_letter_area)
    manifest = {
        "version": 1,
        "width": W,
        "height": H,
        "papyrus": "papyrus.webp",
        "label": "label.png",
        "prediction": "prediction.webp",
        "brushRadius": max(8, W // 80),
        "revealRadius": max(30, W // 22),
        "letters": boxes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"ink manifest written -> {out} ({len(boxes)} letters, {W}x{H})")


def connected_boxes(mask, min_area=120):
    """Two-pass union-find connected components -> bounding boxes."""
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nxt = 1
    for yy in range(H):
        for xx in range(W):
            if not mask[yy, xx]:
                continue
            up = labels[yy - 1, xx] if yy else 0
            left = labels[yy, xx - 1] if xx else 0
            if up and left:
                ru, rl = find(up), find(left)
                labels[yy, xx] = ru
                if ru != rl:
                    parent[rl] = ru
            elif up or left:
                labels[yy, xx] = up or left
            else:
                parent.append(nxt)
                labels[yy, xx] = nxt
                nxt += 1
    boxes = {}
    for yy in range(H):
        for xx in range(W):
            l = labels[yy, xx]
            if not l:
                continue
            r = find(l)
            b = boxes.setdefault(r, [xx, yy, xx, yy, 0])
            b[0] = min(b[0], xx)
            b[1] = min(b[1], yy)
            b[2] = max(b[2], xx)
            b[3] = max(b[3], yy)
            b[4] += 1
    return [
        {"x": b[0] - 4, "y": b[1] - 4, "w": b[2] - b[0] + 8, "h": b[3] - b[1] + 8}
        for b in boxes.values()
        if b[4] >= min_area
    ]


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--tifxyz", required=True)
        p.add_argument("--window", nargs=4, type=int, metavar=("U0", "U1", "V0", "V1"))
        p.add_argument("--out", default="static/get-started")

    pc = sub.add_parser("check", help="verify tifxyz window geometry (no outputs)")
    add_common(pc)

    ps = sub.add_parser("seg", help="build seg demo assets")
    add_common(ps)
    ps.add_argument("--volume", help="zarr path or s3:// url (optional)")
    ps.add_argument("--volume-level", type=int, default=2, help="multiscale level")
    ps.add_argument("--renders", help="vc_render_tifxyz output dir (optional)")
    ps.add_argument("--render-ref", help="flattened reference crop tif = the offset-0 render; offsets >0 sampled from --volume")
    ps.add_argument("--render-level", type=int, default=3, help="multiscale level for offset renders")
    ps.add_argument("--render-window", nargs=4, type=int, metavar=("U0", "U1", "V0", "V1"),
                    help="(u,v) window of the reference crop, when --window is taller")
    ps.add_argument("--plane-fracs", nargs=3, type=float, default=[0.75, 0.5, 0.25],
                    help="z fractions for the top,middle,bottom planes (descending)")
    ps.add_argument("--plane-z", nargs=3, type=float, metavar=("ZT", "ZM", "ZB"),
                    help="explicit z for the top,middle,bottom planes (descending)")
    ps.add_argument("--full-slice", action="store_true",
                    help="slice images = the whole scroll cross-section (shared bbox crop)")
    ps.add_argument("--slice-level", type=int, default=3, help="multiscale level for full slices")
    ps.add_argument("--slice-quality", type=int, default=70)
    ps.add_argument("--detail-size", type=int, default=1400,
                    help="px of the hi-res inset around the trace (0 = none)")
    ps.add_argument("--detail-level", type=int, default=2)
    ps.add_argument("--detail-quality", type=int, default=68)
    ps.add_argument("--detail-margin", type=float, default=300, help="inset margin, voxels")
    ps.add_argument("--intro-zoom", type=float, default=14)
    ps.add_argument("--max-zoom", type=float, default=26)
    ps.add_argument("--render-scale", type=float, default=1.0, help="render px per tifxyz grid px")
    ps.add_argument("--offsets", nargs="+", type=int, default=[0, 8, 16, 32])
    ps.add_argument("--slice-size", type=int, default=640)
    ps.add_argument("--render-size", nargs=2, type=int, default=[800, 480])
    ps.add_argument("--margin", type=float, default=120, help="crop margin, voxels")
    ps.add_argument("--tolerance", type=int, default=12)
    ps.add_argument("--max-dist", type=int, default=60)
    ps.add_argument("--tolerance-vox", type=float, help="tolerance in voxels (overrides --tolerance)")
    ps.add_argument("--max-dist-vox", type=float, help="max dist in voxels (overrides --max-dist)")

    pt = sub.add_parser("turntable", help="render textured turntable sprite for the seg demo")
    add_common(pt)
    pt.add_argument("--volume", required=True, help="zarr path or s3:// url")
    pt.add_argument("--volume-level", type=int, default=4, help="multiscale level for texture")
    pt.add_argument("--grid-step", type=int, default=1, help="tifxyz grid subsampling")
    pt.add_argument("--frames", type=int, default=48)
    pt.add_argument("--cols", type=int, default=8)
    pt.add_argument("--tile", nargs=2, type=int, default=[200, 360], metavar=("W", "H"))
    pt.add_argument("--tilt-deg", type=float, default=16, help="camera tilt (planes become ellipses)")
    pt.add_argument("--quality", type=int, default=70, help="webp quality for the sprite")
    pt.add_argument("--cutaway", type=float, default=0, help="shave outer windings' top by up to this z-fraction")
    pt.add_argument("--cutaway-ramp", type=float, default=1600,
                    help="grid columns from the trace window over which the cutaway reaches full depth")
    pt.add_argument("--trace-window", nargs=2, type=int, metavar=("U0", "U1"), help="demo trace window (kept at full height)")
    pt.add_argument("--z-band", nargs=2, type=float, metavar=("ZLO", "ZHI"),
                    help="keep only surface points with z in this band")
    pt.add_argument("--z-stretch", default="1", help="stretch dz by this factor, or 'auto' to fill the tile")
    pt.add_argument("--upsample-v", type=int, default=1, help="upsample grid rows (splat density for thin bands)")
    pt.add_argument("--winding-spread", type=float, default=0,
                    help="exaggerate the gap between windings: points move radially by up to this fraction of their radius across the u span (stylized exploded spiral)")
    pt.add_argument("--depth-fade", type=float, default=0,
                    help="dim points by up to this fraction with camera depth: the far side recedes and the winding gaps pop")
    pt.add_argument("--plane-min-gap-px", type=float, default=0,
                    help="minimum tile-px gap between plane ring centers in the manifest (the middle ring stays put; the others are faked outward so all three are visible and tappable)")

    pu = sub.add_parser("unroll", help="export decimated 3D grid for the unroll animation")
    add_common(pu)
    pu.add_argument("--grid", nargs=2, type=int, default=[26, 110], metavar=("ROWS", "COLS"))

    pi = sub.add_parser("ink", help="build ink demo assets")
    pi.add_argument("--papyrus", required=True, help="surface render (tif/png)")
    pi.add_argument("--label", help="binary ink label, same dims")
    pi.add_argument("--label-from-pred", type=int, metavar="T",
                    help="derive the label by thresholding the raw prediction at T (no --label needed)")
    pi.add_argument("--prediction", required=True, help="model output, same dims")
    pi.add_argument("--no-normalize", action="store_true",
                    help="papyrus is already tone-mapped uint8; pass through")
    pi.add_argument("--max-width", type=int, default=1024)
    pi.add_argument("--min-letter-area", type=int, default=120)
    pi.add_argument("--out", default="static/get-started")

    args = ap.parse_args()
    if args.cmd == "ink" and args.label is None and args.label_from_pred is None:
        ap.error("ink: pass --label or --label-from-pred")
    {"check": cmd_check, "seg": cmd_seg, "ink": cmd_ink,
     "turntable": cmd_turntable, "unroll": cmd_unroll}[args.cmd](args)
