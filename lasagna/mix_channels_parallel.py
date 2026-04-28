"""Mix channels from two preprocessed prediction zarrs into one.

Cherry-pick channels from a secondary zarr (``--other``) while taking
the remaining channels from a primary zarr (``--base``).  Both inputs
must share the same ``scaledown`` and spatial shape.  When their
``crop_xyzwhd`` regions differ the output is restricted to the
intersection.

This variant bypasses zarr-python v3's async store layer for data I/O,
using direct file reads/writes (which release the GIL) so that threads
can saturate the filesystem without event-loop contention.

Example::

    python lasagna/mix_channels_parallel.py \
        --base pred_3d.zarr \
        --other pred_2d.zarr \
        --output mixed.zarr \
        --channels-other nx ny
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time

import numcodecs
import numpy as np
import zarr


CHUNK_SPATIAL = 32  # chunk size for all spatial dims


# ---------------------------------------------------------------------------
# Direct zarr v2 chunk I/O – bypasses zarr-python v3's async store layer so
# that plain file reads/writes (which release the GIL) can saturate the
# filesystem from many threads without event-loop contention.
# ---------------------------------------------------------------------------

def _load_zarr_meta(zarr_path: str) -> dict:
    """Read .zarray metadata for direct chunk I/O."""
    with open(os.path.join(zarr_path, '.zarray')) as f:
        meta = json.load(f)
    comp_cfg = meta.get('compressor')
    return {
        'path': zarr_path,
        'chunks': tuple(meta['chunks']),
        'dtype': np.dtype(meta['dtype']),
        'shape': tuple(meta['shape']),
        'fill_value': meta.get('fill_value', 0),
        'order': meta.get('order', 'C'),
        'dim_sep': meta.get('dimension_separator', '.'),
        'compressor': numcodecs.get_codec(comp_cfg) if comp_cfg else None,
        'filters': [numcodecs.get_codec(f) for f in (meta.get('filters') or [])],
    }


def _chunk_key(dim_sep: str, *indices: int) -> str:
    return dim_sep.join(str(i) for i in indices)


def _read_chunk_file(meta: dict, key: str, chunk_shape: tuple) -> np.ndarray:
    """Read and decode a single zarr v2 chunk file."""
    fpath = os.path.join(meta['path'], key.replace('/', os.sep))
    try:
        with open(fpath, 'rb') as f:
            raw = f.read()
    except FileNotFoundError:
        return np.full(chunk_shape, meta['fill_value'], dtype=meta['dtype'])
    if meta['compressor'] is not None:
        raw = meta['compressor'].decode(raw)
    for filt in reversed(meta['filters']):
        raw = filt.decode(raw)
    return np.frombuffer(raw, dtype=meta['dtype']).reshape(
        chunk_shape, order=meta['order']).copy()


def _copy_chunk(src_meta: dict, si: int,
                out_path: str, out_compressor, out_filters: list,
                out_order: str, out_dim_sep: str,
                ci: int, cz_idx: int, cy_idx: int, cx_idx: int,
                zs: int, ze: int, ys: int, ye: int, xs: int, xe: int):
    """Read one source chunk, write one output chunk via direct file I/O."""
    _, cs_z, cs_y, cs_x = src_meta['chunks']
    src_key = _chunk_key(src_meta['dim_sep'], si, cz_idx, cy_idx, cx_idx)
    chunk = _read_chunk_file(src_meta, src_key, (cs_z, cs_y, cs_x))

    # Crop to the active region within the chunk
    oz = zs - cz_idx * cs_z
    oy = ys - cy_idx * cs_y
    ox = xs - cx_idx * cs_x
    cdata = chunk[oz:oz + (ze - zs), oy:oy + (ye - ys), ox:ox + (xe - xs)]

    # Pad partial edge chunks to full output chunk size (zarr v2 spec)
    cs = CHUNK_SPATIAL
    if cdata.shape != (cs, cs, cs):
        padded = np.zeros((cs, cs, cs), dtype=cdata.dtype)
        padded[:cdata.shape[0], :cdata.shape[1], :cdata.shape[2]] = cdata
        cdata = padded

    raw = (np.ascontiguousarray(cdata) if out_order == 'C'
           else np.asfortranarray(cdata)).tobytes()
    for filt in out_filters:
        raw = filt.encode(raw)
    if out_compressor is not None:
        raw = out_compressor.encode(raw)

    out_key = _chunk_key(out_dim_sep, ci, cz_idx, cy_idx, cx_idx)
    fpath = os.path.join(out_path, out_key.replace('/', os.sep))
    if out_dim_sep == '/':
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'wb') as f:
        f.write(raw)


# ---------------------------------------------------------------------------
# Multiprocessing worker – each process gets its own GIL
# ---------------------------------------------------------------------------

_w_base_meta: dict | None = None
_w_other_meta: dict | None = None
_w_out_path: str = ""
_w_out_compressor = None
_w_out_filters: list = []
_w_out_order: str = "C"
_w_out_dim_sep: str = "."


def _init_worker(base_path: str, other_path: str, output_path: str):
    """Called once per worker process — load zarr metadata into globals."""
    global _w_base_meta, _w_other_meta
    global _w_out_path, _w_out_compressor, _w_out_filters
    global _w_out_order, _w_out_dim_sep
    _w_base_meta = _load_zarr_meta(base_path)
    _w_other_meta = _load_zarr_meta(other_path)
    out_meta = _load_zarr_meta(output_path)
    _w_out_path = output_path
    _w_out_compressor = out_meta['compressor']
    _w_out_filters = out_meta['filters']
    _w_out_order = out_meta['order']
    _w_out_dim_sep = out_meta['dim_sep']


def _worker_fn(item: tuple):
    """Process one chunk. item = (src_is_base, si, ci, cz, cy, cx, zs, ze, ys, ye, xs, xe)."""
    src_is_base, si, ci, cz_idx, cy_idx, cx_idx, zs, ze, ys, ye, xs, xe = item
    src_meta = _w_base_meta if src_is_base else _w_other_meta
    _copy_chunk(src_meta, si,
                _w_out_path, _w_out_compressor, _w_out_filters,
                _w_out_order, _w_out_dim_sep,
                ci, cz_idx, cy_idx, cx_idx,
                zs, ze, ys, ye, xs, xe)


# ---------------------------------------------------------------------------
# Metadata helpers (same as mix_channels.py)
# ---------------------------------------------------------------------------

def _parse_params(arr: zarr.Array) -> dict:
    params = dict(getattr(arr, "attrs", {}).get("preprocess_params", {}) or {})
    if not params:
        raise ValueError(f"zarr missing preprocess_params: shape={arr.shape}")
    return params


def _channels(params: dict) -> list[str]:
    ch = [str(v) for v in (params.get("channels", []) or [])]
    if not ch:
        ch = ["cos", "grad_mag", "nx", "ny"]
    return ch


def _crop_indices(params: dict, scaledown: int, shape_zyx: tuple[int, int, int]):
    """Return (z0, z1, y0, y1, x0, x1) zarr indices for the crop region."""
    crop = params.get("crop_xyzwhd", None)
    Z, Y, X = shape_zyx
    if crop is None:
        return 0, Z, 0, Y, 0, X
    x, y, z, w, h, d = (int(v) for v in crop)
    sd = max(1, int(scaledown))
    z0 = max(0, min(z // sd, Z))
    z1 = max(z0, min((z + d + sd - 1) // sd, Z))
    y0 = max(0, min(y // sd, Y))
    y1 = max(y0, min((y + h + sd - 1) // sd, Y))
    x0 = max(0, min(x // sd, X))
    x1 = max(x0, min((x + w + sd - 1) // sd, X))
    return z0, z1, y0, y1, x0, x1


def _intersect_crop(crop_a, crop_b, scaledown: int):
    """Intersect two crop_xyzwhd regions (fullres coords).

    Returns (crop_xyzwhd, is_valid).  If either is None the other is used.
    """
    if crop_a is None and crop_b is None:
        return None, True
    if crop_a is None:
        return list(int(v) for v in crop_b), True
    if crop_b is None:
        return list(int(v) for v in crop_a), True

    ax, ay, az, aw, ah, ad = (int(v) for v in crop_a)
    bx, by, bz, bw, bh, bd = (int(v) for v in crop_b)

    x0 = max(ax, bx)
    y0 = max(ay, by)
    z0 = max(az, bz)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    z1 = min(az + ad, bz + bd)

    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return None, False

    return [x0, y0, z0, x1 - x0, y1 - y0, z1 - z0], True


def main():
    ap = argparse.ArgumentParser(description="Mix channels from two prediction zarrs")
    ap.add_argument("--base", required=True, help="Primary zarr (default source)")
    ap.add_argument("--other", required=True, help="Secondary zarr")
    ap.add_argument("--output", required=True, help="Output zarr path")
    ap.add_argument(
        "--channels-other",
        nargs="+",
        required=True,
        help="Channel names to take from --other (rest come from --base)",
    )
    ap.add_argument(
        "--workers", type=int, default=os.cpu_count() or 4,
        help="Parallel copy threads (default: min(8, ncpu))",
    )
    args = ap.parse_args()

    # --- Open sources (zarr API for metadata only) ---
    base = zarr.open(args.base, mode="r")
    other = zarr.open(args.other, mode="r")
    if not isinstance(base, zarr.Array):
        raise ValueError(f"--base must point to a zarr Array, got {type(base)}")
    if not isinstance(other, zarr.Array):
        raise ValueError(f"--other must point to a zarr Array, got {type(other)}")

    base_params = _parse_params(base)
    other_params = _parse_params(other)

    base_sd = int(base_params["scaledown"])
    other_sd = int(other_params["scaledown"])
    if base_sd != other_sd:
        raise ValueError(f"scaledown mismatch: base={base_sd} other={other_sd}")
    scaledown = base_sd

    base_shape = tuple(int(v) for v in base.shape)
    other_shape = tuple(int(v) for v in other.shape)
    if len(base_shape) != 4 or len(other_shape) != 4:
        raise ValueError(f"expected 4D CZYX zarrs; base={base_shape} other={other_shape}")
    if base_shape[1:] != other_shape[1:]:
        raise ValueError(
            f"spatial shape mismatch: base={base_shape[1:]} other={other_shape[1:]}"
        )

    # --- Channels ---
    base_ch = _channels(base_params)
    other_ch = _channels(other_params)
    base_ci = {name: i for i, name in enumerate(base_ch)}
    other_ci = {name: i for i, name in enumerate(other_ch)}

    channels_from_other = set(args.channels_other)
    for name in channels_from_other:
        if name not in other_ci:
            raise ValueError(
                f"channel '{name}' requested from --other but not found; "
                f"available: {other_ch}"
            )

    # Output channel list = base channels (order preserved).
    # Any channel in channels_from_other that exists only in other is appended.
    out_channels: list[str] = list(base_ch)
    for name in other_ch:
        if name in channels_from_other and name not in base_ci:
            out_channels.append(name)

    # Build mapping: out_index -> (source_zarr, source_channel_index)
    channel_map: list[tuple[zarr.Array, int, str]] = []
    for name in out_channels:
        if name in channels_from_other:
            channel_map.append((other, other_ci[name], name))
        elif name in base_ci:
            channel_map.append((base, base_ci[name], name))
        else:
            raise ValueError(
                f"channel '{name}' not found in base ({base_ch}) or other ({other_ch})"
            )

    # --- Crop intersection ---
    base_crop = base_params.get("crop_xyzwhd", None)
    other_crop = other_params.get("crop_xyzwhd", None)
    out_crop, valid = _intersect_crop(base_crop, other_crop, scaledown)
    if not valid:
        raise ValueError("crop regions do not overlap — nothing to mix")

    # Compute zarr index bounds for the intersection region
    _, Z, Y, X = base_shape
    if out_crop is not None:
        # Use same index math as rest of codebase
        sd = max(1, scaledown)
        cx, cy, cz, cw, ch, cd = out_crop
        z0 = max(0, min(cz // sd, Z))
        z1 = max(z0, min((cz + cd + sd - 1) // sd, Z))
        y0 = max(0, min(cy // sd, Y))
        y1 = max(y0, min((cy + ch + sd - 1) // sd, Y))
        x0 = max(0, min(cx // sd, X))
        x1 = max(x0, min((cx + cw + sd - 1) // sd, X))
    else:
        z0, z1, y0, y1, x0, x1 = 0, Z, 0, Y, 0, X

    n_ch = len(out_channels)
    out_shape = (n_ch, Z, Y, X)
    CS = CHUNK_SPATIAL
    out_chunks = (1, min(Z, CS), min(Y, CS), min(X, CS))

    print(f"[mix_channels] base={args.base}  shape={base_shape}  channels={base_ch}")
    print(f"[mix_channels] other={args.other} shape={other_shape} channels={other_ch}")
    print(f"[mix_channels] output channels: {out_channels}")
    for ci, (src, si, name) in enumerate(channel_map):
        tag = "other" if src is other else "base"
        print(f"  ch[{ci}] {name:12s} <- {tag}[{si}]")
    print(f"[mix_channels] crop intersection z=[{z0},{z1}) y=[{y0},{y1}) x=[{x0},{x1})")
    n_workers = max(1, args.workers)
    print(f"[mix_channels] output {args.output} shape={out_shape} chunks={out_chunks} workers={n_workers}")

    # --- Create output (zarr API for metadata, then direct I/O for data) ---
    out = zarr.open(
        args.output,
        mode="w",
        shape=out_shape,
        chunks=out_chunks,
        dtype=np.uint8,
        fill_value=0,
        zarr_format=2,
    )

    # Provenance metadata
    gmag_scale = float(base_params.get("grad_mag_encode_scale",
                       other_params.get("grad_mag_encode_scale", 1000.0)))
    out_params = {
        "scaledown": scaledown,
        "grad_mag_encode_scale": gmag_scale,
        "channels": out_channels,
        "output_full_scaled": True,
        "source": "mix_channels",
        "base": args.base,
        "other": args.other,
        "channels_other": args.channels_other,
    }
    if out_crop is not None:
        out_params["crop_xyzwhd"] = out_crop
    out.attrs["preprocess_params"] = out_params

    # --- Build work units: one per 32³ output chunk (plain tuples for IPC) ---
    CS = CHUNK_SPATIAL
    work: list[tuple] = []
    for ci, (src, si, _name) in enumerate(channel_map):
        src_is_base = src is base
        for cz in range(z0 // CS * CS, z1, CS):
            cz_idx = cz // CS
            zs_c = max(cz, z0)
            ze_c = min(cz + CS, z1)
            for cy in range(y0 // CS * CS, y1, CS):
                cy_idx = cy // CS
                ys_c = max(cy, y0)
                ye_c = min(cy + CS, y1)
                for cx in range(x0 // CS * CS, x1, CS):
                    cx_idx = cx // CS
                    xs_c = max(cx, x0)
                    xe_c = min(cx + CS, x1)
                    work.append((src_is_base, si,
                                 ci, cz_idx, cy_idx, cx_idx,
                                 zs_c, ze_c, ys_c, ye_c, xs_c, xe_c))

    total = len(work)
    t0 = time.time()

    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(args.base, args.other, args.output),
    ) as pool:
        done_count = 0
        for _ in pool.imap_unordered(_worker_fn, work, chunksize=256):
            done_count += 1
            if done_count % 500 == 0 or done_count == total:
                elapsed = time.time() - t0
                eta = elapsed / done_count * (total - done_count) if done_count < total else 0
                print(
                    f"\r  chunk {done_count}/{total}  "
                    f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s   ",
                    end="", flush=True,
                )

    print()
    print(f"[mix_channels] done — {args.output}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
