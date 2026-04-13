"""Mix channels from two preprocessed prediction zarrs into one.

Cherry-pick channels from a secondary zarr (``--other``) while taking
the remaining channels from a primary zarr (``--base``).  Both inputs
must share the same ``scaledown`` and spatial shape.  When their
``crop_xyzwhd`` regions differ the output is restricted to the
intersection.

Example::

    python lasagna/mix_channels.py \
        --base pred_3d.zarr \
        --other pred_2d.zarr \
        --output mixed.zarr \
        --channels-other nx ny
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
import time

import numpy as np
import zarr


CHUNK_SPATIAL = 32  # chunk size for all spatial dims


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

    # --- Open sources ---
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

    # --- Create output ---
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

    # --- Build work units: one per output chunk (channel × z × y × x tile) ---
    # Iterate on chunk-aligned boundaries so each unit writes exactly one
    # zarr chunk.  Clamp to the crop region for partial border chunks.
    work: list[tuple[int, int, int, int, int, int, int, zarr.Array, int]] = []
    for ci, (src, si, _name) in enumerate(channel_map):
        for zs in range(z0 // CS * CS, z1, CS):
            ze = min(zs + CS, z1)
            zs_c = max(zs, z0)
            for ys in range(y0 // CS * CS, y1, CS):
                ye = min(ys + CS, y1)
                ys_c = max(ys, y0)
                for xs in range(x0 // CS * CS, x1, CS):
                    xe = min(xs + CS, x1)
                    xs_c = max(xs, x0)
                    work.append((ci, zs_c, ze, ys_c, ye, xs_c, xe, src, si))

    total = len(work)
    done_count = 0
    lock = threading.Lock()
    t0 = time.time()

    def _copy_chunk(ci: int, zs: int, ze: int,
                    ys: int, ye: int, xs: int, xe: int,
                    src: zarr.Array, si: int):
        nonlocal done_count
        # Materialize to numpy before writing to avoid zarr internal
        # executor conflicts between concurrent read and write paths.
        data = np.array(src[si, zs:ze, ys:ye, xs:xe])
        out[ci, zs:ze, ys:ye, xs:xe] = data
        with lock:
            done_count += 1
            n = done_count
        elapsed = time.time() - t0
        eta = elapsed / n * (total - n) if n < total else 0
        print(
            f"\r  chunk {n}/{total}  "
            f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s   ",
            end="", flush=True,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_copy_chunk, *w) for w in work]
        # Collect all results before pool shutdown to ensure zarr's
        # internal async tasks finish while the executor is alive.
        errors = []
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                errors.append(e)
        if errors:
            raise errors[0]

    print()
    print(f"[mix_channels] done — {args.output}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
