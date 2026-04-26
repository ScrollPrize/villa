#!/usr/bin/env python3
"""Estimate memory savings from streamed sampling vs bbox loading.

Two scenarios, each compared to its streaming equivalent:
  A) Full scroll:    1 m² mesh (all windings), load entire 20k×20k×z scroll
  B) Crop cylinder:  mesh = outer cylinder surface from bbox, load = bbox + margin

Optionally reads a .lasagna.json to get real channel scaledowns and scroll Z.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _parse_bbox(s: str) -> tuple[int, int, int, int, int, int]:
    """Parse 'x0,y0,z0,w,h,d' string."""
    parts = [int(v) for v in s.split(",")]
    if len(parts) != 6:
        raise ValueError(f"bbox must be x0,y0,z0,w,h,d — got {len(parts)} values")
    return tuple(parts)


def _fmt(nbytes: int | float) -> str:
    b = float(nbytes)
    if b >= 2**30:
        return f"{b / 2**30:.2f} GiB"
    return f"{b / 2**20:.1f} MiB"


def _vol_bytes(lx: int, ly: int, lz: int, channel_ds: dict[str, int]) -> int:
    total = 0
    for ds in channel_ds.values():
        total += max(1, lx // ds) * max(1, ly // ds) * max(1, lz // ds)
    return total


CHUNK = 32  # zarr chunk edge length in zarr voxels


def _chunked_filled_bytes(diam: int, height: int, channel_ds: dict[str, int]) -> int:
    """Chunks touched by a filled cylinder (all windings inside)."""
    total = 0
    for ds in channel_ds.values():
        cb = CHUNK * ds  # chunk footprint in base voxels
        r_chunks = math.ceil(diam / 2.0 / cb)  # radius in chunk units
        z_chunks = math.ceil(height / cb)
        # Count chunks inside the circle at each z-level
        # For each row y in [-r, r], width = 2*sqrt(r²-y²)
        n_xy = 0
        for iy in range(-r_chunks, r_chunks + 1):
            half_w = math.sqrt(max(0, r_chunks * r_chunks - iy * iy))
            n_xy += 2 * int(math.ceil(half_w))  # chunks in this row
        total += n_xy * z_chunks * CHUNK**3
    return total


def _chunked_surface_bytes(diam: int, height: int, channel_ds: dict[str, int]) -> int:
    """Chunks touched by a cylinder surface (thin shell, ~1 chunk thick)."""
    total = 0
    for ds in channel_ds.values():
        cb = CHUNK * ds  # chunk footprint in base voxels
        # Circumference in chunk units — number of chunks around the ring
        circ_chunks = max(1, int(math.ceil(math.pi * diam / cb)))
        z_chunks = max(1, math.ceil(height / cb))
        total += circ_chunks * z_chunks * CHUNK**3
    return total


def _stream_bytes(N: int, sm: int, sw: int, N_ext: int,
                  lr_winding: bool = False) -> int:
    """Compute streamed memory for N LR vertices.

    lr_winding=True: winding density integral at LR step (not HR),
    grad_mag sampled with full 3×3 Jacobian (10 bytes) on CPU.
    """
    hr = sm * sw
    S = sm + 1
    bytes_diff = 4     # value + 3 partials (d/dx, d/dy, d/dz)
    bytes_nodiff = 1   # value only
    bytes_jac = 10     # value + 3×3 Jacobian

    if lr_winding:
        # winding_density: LR step, S strip samples, grad_mag with full Jacobian
        # ext_offset: LR step (already LR), S strip samples, grad_mag with full Jacobian
        calls = [
            (hr,            5, bytes_diff),    # fwd: data_s
            (hr,            1, bytes_nodiff),  # fwd: mask_hr
            (1,             1, bytes_nodiff),  # fwd: mask_lr
            (3,             1, bytes_nodiff),  # fwd: conn masks
            (N_ext,         1, bytes_nodiff),  # fwd: ext intersect
            (hr,            1, bytes_diff),    # data loss
            (1,             3, bytes_nodiff),  # normal (nx, ny, gm)
            (1,             1, bytes_diff),    # pred_dt
            (2 * S,         1, bytes_jac),     # winding_density: LR × S, full Jacobian
            (N_ext * S,     1, bytes_jac),     # ext_offset: LR × S, full Jacobian
        ]
    else:
        calls = [
            (hr,            5, bytes_diff),    # fwd: data_s
            (hr,            1, bytes_nodiff),  # fwd: mask_hr
            (1,             1, bytes_nodiff),  # fwd: mask_lr
            (3,             1, bytes_nodiff),  # fwd: conn masks
            (N_ext,         1, bytes_nodiff),  # fwd: ext intersect
            (hr,            1, bytes_diff),    # data loss
            (1,             3, bytes_nodiff),  # normal (nx, ny, gm)
            (1,             1, bytes_diff),    # pred_dt
            (2 * hr * S,    1, bytes_nodiff),  # winding_density
            (N_ext * S,     1, bytes_nodiff),  # ext_offset
        ]
    total = 0
    for factor, n_ch, bpc in calls:
        total += factor * N * n_ch * bpc
    return total


def _print_stream_detail(N: int, sm: int, sw: int, N_ext: int) -> None:
    hr = sm * sw
    S = sm + 1
    bytes_diff = 4
    bytes_nodiff = 1

    calls = [
        ("fwd: data_s",      hr,            5, True),
        ("fwd: mask_hr",     hr,            1, False),
        ("fwd: mask_lr",     1,             1, False),
        ("fwd: conn",        3,             1, False),
        ("fwd: ext",         N_ext,         1, False),
        ("data loss",        hr,            1, True),
        ("normal",           1,             3, False),
        ("pred_dt",          1,             1, True),
        ("winding_density",  2 * hr * S,    1, False),
        ("ext_offset",       N_ext * S,     1, False),
    ]

    total_factor = sum(f for _, f, _, _ in calls)
    print(f"  Sample factors (× N = {N:,d}):")
    for name, factor, n_ch, diff in calls:
        bpc = bytes_diff if diff else bytes_nodiff
        nb = factor * N * n_ch * bpc
        tag = "val+d/dxyz" if diff else "val"
        print(f"    {name:>20s}: {factor:6d}×N × {n_ch}ch × {bpc}B ({tag})"
              f" = {_fmt(nb):>10s}")
    print(f"    {'total factor':>20s}: {total_factor:6d}×N")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("lasagna_json", nargs="?", default=None,
                   help="Path to .lasagna.json (optional — overrides channel scaledowns and scroll Z)")
    p.add_argument("--bbox", type=_parse_bbox, default=None,
                   help="Crop bbox: x0,y0,z0,w,h,d in base-coord voxels (overrides --crop-xy/--crop-z)")
    p.add_argument("--mesh-step", type=int, default=50,
                   help="Mesh step in voxels (default: 50)")
    p.add_argument("--subsample-mesh", type=int, default=4,
                   help="HR subsample factor (default: 4)")
    p.add_argument("--subsample-winding", type=int, default=4,
                   help="HR subsample factor (default: 4)")
    p.add_argument("--n-ext", type=int, default=1,
                   help="Number of external surfaces (default: 1)")
    p.add_argument("--mesh-area-sqm", type=float, default=1.0,
                   help="Full-scroll mesh area in m² (default: 1.0)")
    p.add_argument("--voxel-um", type=float, default=2.0,
                   help="Voxel size in µm (default: 2.0, so step 50 = 0.1mm)")
    p.add_argument("--scroll-xy", type=int, default=20000,
                   help="Scroll XY extent in voxels (default: 20000)")
    p.add_argument("--scroll-z", type=int, default=None,
                   help="Scroll Z height in voxels (default: from json or 10000)")
    p.add_argument("--crop-xy", type=int, default=2000,
                   help="Crop cylinder XY diameter in voxels (default: 2000, overridden by --bbox)")
    p.add_argument("--crop-z", type=int, default=2000,
                   help="Crop cylinder Z height in voxels (default: 2000, overridden by --bbox)")
    p.add_argument("--margin-pct", type=float, default=100.0,
                   help="Bbox margin as percent padding per side (default: 100%%)")
    p.add_argument("--margin-vx", type=int, default=1000,
                   help="Bbox margin as max voxel padding per side (default: 1000)")
    args = p.parse_args(argv)

    mesh_step = args.mesh_step
    sm = args.subsample_mesh
    sw = args.subsample_winding
    N_ext = args.n_ext
    voxel_um = args.voxel_um
    step_mm = mesh_step * voxel_um / 1000.0
    margin_pct = args.margin_pct / 100.0
    margin_vx = args.margin_vx

    # --- Channel scaledowns: from json or hardcoded defaults ---
    channel_ds: dict[str, int] = {"cos": 2, "grad_mag": 4, "nx": 4, "ny": 4, "pred_dt": 4}
    scroll_z_from_json: int | None = None
    data_source = "synthetic defaults"

    if args.lasagna_json is not None:
        from lasagna_volume import LasagnaVolume
        import zarr

        lasagna_path = Path(args.lasagna_json)
        if not lasagna_path.exists():
            print(f"ERROR: {lasagna_path} not found", file=sys.stderr)
            return 1

        vol = LasagnaVolume.load(lasagna_path)
        s2b = vol.source_to_base

        # Read real scaledowns
        channel_ds = {}
        for ch_name in ["cos", "grad_mag", "nx", "ny", "pred_dt"]:
            if ch_name in vol.all_channels():
                group, _ = vol.channel_group(ch_name)
                channel_ds[ch_name] = int(round(group.sd_fac * s2b))

        # Read scroll Z from volume extent
        if vol.base_shape_zyx is not None:
            scroll_z_from_json = vol.base_shape_zyx[0]
        else:
            min_sd = min(g.sd_fac for g in vol.groups.values())
            for g in vol.groups.values():
                if g.sd_fac == min_sd:
                    zpath = str(vol.path.parent / g.zarr_path)
                    zsrc = zarr.open(zpath, mode="r")
                    shape = tuple(int(v) for v in zsrc.shape)
                    sZ = shape[0] if len(shape) == 3 else shape[1]
                    scroll_z_from_json = int(sZ * min_sd * s2b)
                    break

        data_source = str(lasagna_path)

    # Resolve scroll Z: explicit flag > json > default
    if args.scroll_z is not None:
        vol_Z = args.scroll_z
    elif scroll_z_from_json is not None:
        vol_Z = scroll_z_from_json
    else:
        vol_Z = 10000

    vol_X = args.scroll_xy
    vol_Y = args.scroll_xy

    # Resolve crop cylinder: --bbox overrides --crop-xy/--crop-z
    if args.bbox is not None:
        _x0, _y0, _z0, cyl_w, cyl_h, cyl_d = args.bbox
        cyl_diam = min(cyl_w, cyl_h)
        cyl_z = cyl_d
    else:
        cyl_diam = args.crop_xy
        cyl_z = args.crop_z

    # --- Print header ---
    print(f"=== Lasagna Stream Memory Estimator ===\n")
    print(f"Data source:       {data_source}")
    print(f"Channel scaledowns: {channel_ds}")
    print(f"Physical assumptions:")
    print(f"  Voxel size:      {voxel_um} µm")
    print(f"  Mesh step:       {mesh_step} voxels = {step_mm:.2f} mm")
    print(f"  Subsample:       mesh={sm} winding={sw}")
    print(f"  Ext surfaces:    {N_ext}")
    print(f"  Margin:          min({margin_pct*100:.0f}% per side, {margin_vx} vx per side)")

    # ==============================================================
    # A) Full scroll: 1 m² mesh, load entire scroll
    # ==============================================================
    area_mm2 = args.mesh_area_sqm * 1e6
    N_a = int(area_mm2 / (step_mm * step_mm))
    side = int(math.sqrt(N_a))
    N_a = side * side

    load_a = _vol_bytes(vol_X, vol_Y, vol_Z, channel_ds)
    chunked_a = _chunked_filled_bytes(vol_X, vol_Z, channel_ds)
    stream_a = _stream_bytes(N_a, sm, sw, N_ext)
    stream_a_lr = _stream_bytes(N_a, sm, sw, N_ext, lr_winding=True)

    print(f"\n{'='*60}")
    print(f"A) Full scroll")
    print(f"{'='*60}")
    print(f"  Mesh:   {args.mesh_area_sqm} m² = {side:,d}×{side:,d} = {N_a:,d} LR verts")
    print(f"  Volume: {vol_X:,d}×{vol_Y:,d}×{vol_Z:,d} voxels")
    _print_stream_detail(N_a, sm, sw, N_ext)
    print()
    print(f"  Dense bbox load:      {_fmt(load_a):>12s}")
    print(f"  Chunked (filled cyl): {_fmt(chunked_a):>12s}")
    print(f"  Streamed:             {_fmt(stream_a):>12s}")
    print(f"  Streamed LR-wind:     {_fmt(stream_a_lr):>12s}")

    # ==============================================================
    # B) Crop cylinder: mesh = outer surface, load = bbox + margin
    # ==============================================================
    circumference_verts = max(2, int(math.pi * cyl_diam / mesh_step) + 1)
    height_verts = max(2, int(cyl_z / mesh_step) + 1)
    N_b = circumference_verts * height_verts

    def _pad(extent: int) -> int:
        pad_pct = int(extent * margin_pct)
        pad = min(pad_pct, margin_vx)
        return extent + 2 * pad

    crop_w = _pad(cyl_diam)
    crop_h = _pad(cyl_diam)
    crop_d = _pad(cyl_z)

    # Clamp to scroll extent and error if clamped
    clamped = []
    if crop_w > vol_X: clamped.append(f"X: {crop_w} > {vol_X}")
    if crop_h > vol_Y: clamped.append(f"Y: {crop_h} > {vol_Y}")
    if crop_d > vol_Z: clamped.append(f"Z: {crop_d} > {vol_Z}")
    if clamped:
        print(f"\nWARNING: crop region exceeds volume bounds:", file=sys.stderr)
        for c in clamped:
            print(f"  {c}", file=sys.stderr)
    crop_w = min(crop_w, vol_X)
    crop_h = min(crop_h, vol_Y)
    crop_d = min(crop_d, vol_Z)

    load_b = _vol_bytes(crop_w, crop_h, crop_d, channel_ds)
    chunked_b = _chunked_surface_bytes(cyl_diam, cyl_z, channel_ds)
    stream_b = _stream_bytes(N_b, sm, sw, N_ext)
    stream_b_lr = _stream_bytes(N_b, sm, sw, N_ext, lr_winding=True)

    cyl_diam_mm = cyl_diam * voxel_um / 1000.0
    cyl_z_mm = cyl_z * voxel_um / 1000.0
    pad_x = (crop_w - cyl_diam) // 2
    pad_z = (crop_d - cyl_z) // 2

    print(f"\n{'='*60}")
    print(f"B) Crop cylinder")
    print(f"{'='*60}")
    print(f"  Cylinder:  diam={cyl_diam:,d} vx ({cyl_diam_mm:.1f} mm)"
          f"  h={cyl_z:,d} vx ({cyl_z_mm:.1f} mm)")
    print(f"  Mesh:      {circumference_verts:,d}×{height_verts:,d}"
          f" = {N_b:,d} LR verts")
    print(f"  Volume:    {crop_w:,d}×{crop_h:,d}×{crop_d:,d}"
          f" (+{pad_x} vx xy, +{pad_z} vx z per side)")
    _print_stream_detail(N_b, sm, sw, N_ext)
    print()
    print(f"  Dense bbox load:      {_fmt(load_b):>12s}")
    print(f"  Chunked (surface):    {_fmt(chunked_b):>12s}")
    print(f"  Streamed:             {_fmt(stream_b):>12s}")
    print(f"  Streamed LR-wind:     {_fmt(stream_b_lr):>12s}")

    # ==============================================================
    # Summary
    # ==============================================================
    # Per-cm² cost
    area_a_cm2 = args.mesh_area_sqm * 1e4
    area_b_cm2 = (math.pi * cyl_diam * cyl_z) * (voxel_um * 1e-4) ** 2

    def _ratio(a: float, b: float) -> str:
        r = a / max(1, b)
        if r >= 100:
            return f"{r:,.0f}×"
        return f"{r:.1f}×"

    col = f"{'Str LR-wind':>12s}"
    print(f"\n{'='*72}")
    print(f"Summary")
    print(f"{'='*72}")
    print(f"  {'':18s}  {'Dense':>12s}  {'Chunked':>12s}  {'Streamed':>12s}  {col}")
    print(f"  {'─'*70}")

    # Total memory
    print(f"  A) Full scroll  {_fmt(load_a):>12s}  {_fmt(chunked_a):>12s}  {_fmt(stream_a):>12s}  {_fmt(stream_a_lr):>12s}")
    print(f"  B) Crop cyl     {_fmt(load_b):>12s}  {_fmt(chunked_b):>12s}  {_fmt(stream_b):>12s}  {_fmt(stream_b_lr):>12s}")
    print()

    # Savings vs dense
    print(f"  Savings vs dense")
    print(f"  A) Full scroll  {'1×':>12s}  {_ratio(load_a, chunked_a):>12s}  {_ratio(load_a, stream_a):>12s}  {_ratio(load_a, stream_a_lr):>12s}")
    print(f"  B) Crop cyl     {'1×':>12s}  {_ratio(load_b, chunked_b):>12s}  {_ratio(load_b, stream_b):>12s}  {_ratio(load_b, stream_b_lr):>12s}")
    print()

    # Per cm²
    def _pcm2(b: float, a: float) -> str:
        return _fmt(b / a) + "/cm²"

    print(f"  Per cm² of mesh")
    print(f"  A) {area_a_cm2:>7,.0f} cm²  {_pcm2(load_a, area_a_cm2):>12s}  {_pcm2(chunked_a, area_a_cm2):>12s}  {_pcm2(stream_a, area_a_cm2):>12s}  {_pcm2(stream_a_lr, area_a_cm2):>12s}")
    print(f"  B) {area_b_cm2:>7,.1f} cm²  {_pcm2(load_b, area_b_cm2):>12s}  {_pcm2(chunked_b, area_b_cm2):>12s}  {_pcm2(stream_b, area_b_cm2):>12s}  {_pcm2(stream_b_lr, area_b_cm2):>12s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
