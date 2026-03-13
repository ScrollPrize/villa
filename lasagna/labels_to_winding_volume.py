"""Compute a winding-number volume from a label TIFF.

Label TIFF → connected components → greedy chain ordering → per-voxel
interpolated winding number → float32 zarr.

Usage
-----
  python labels_to_winding_volume.py \
      --input labels.tif \
      --output winding.zarr \
      --step 4 \
      --connectivity 6 \
      --min-voxels 1000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy.ndimage import distance_transform_edt


TAG = "[labels_to_winding_volume]"


# ---------------------------------------------------------------------------
# Connected components (copied from labels_to_cc_zarr.py)
# ---------------------------------------------------------------------------

def _connected_components(
    vol: np.ndarray,
    connectivity: int,
) -> tuple[np.ndarray, int]:
    """Label connected components in a binary 3D volume.

    Returns (labels, n_components).  *labels* dtype is int32/int64.
    *connectivity* is 6 (face) or 26 (full).
    """
    try:
        import cc3d
        labels = cc3d.connected_components(vol, connectivity=connectivity)
        n = int(labels.max())
        return labels, n
    except ImportError:
        pass

    from scipy.ndimage import label as scipy_label

    if connectivity == 6:
        struct3d = np.zeros((3, 3, 3), dtype=np.uint8)
        struct3d[1, 1, :] = 1
        struct3d[1, :, 1] = 1
        struct3d[:, 1, 1] = 1
    else:
        struct3d = np.ones((3, 3, 3), dtype=np.uint8)

    labels, n = scipy_label(vol, structure=struct3d)
    return labels, n


# ---------------------------------------------------------------------------
# Downsample
# ---------------------------------------------------------------------------

def _downsample_mean(vol: np.ndarray, step: int) -> np.ndarray:
    """Downsample a 3D float volume by *step* using mean-pooling."""
    if step == 1:
        return vol
    z, y, x = vol.shape
    pz = -z % step
    py = -y % step
    px = -x % step
    if pz or py or px:
        vol = np.pad(vol, ((0, pz), (0, py), (0, px)))
    z2, y2, x2 = vol.shape
    return (
        vol.reshape(z2 // step, step, y2 // step, step, x2 // step, step)
        .mean(axis=(1, 3, 5))
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Label TIFF → winding-number volume (float32 zarr)",
    )
    p.add_argument("--input", required=True,
                    help="Multi-layer label TIFF (ZYX): 0=bg, 1=pred, 2=ignore")
    p.add_argument("--output", required=True,
                    help="Output winding volume zarr path (float32)")
    p.add_argument("--step", type=int, default=4,
                    help="Downsample factor (default: 4)")
    p.add_argument("--connectivity", type=int, choices=[6, 26], default=6,
                    help="3D connectivity for CC (6=face, 26=full)")
    p.add_argument("--min-voxels", type=int, default=0,
                    help="Discard components with fewer voxels (0=keep all)")
    p.add_argument("--chunk-size", type=int, default=64,
                    help="Zarr chunk size per axis")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    step = args.step

    # -- Read TIFF ----------------------------------------------------------
    print(f"{TAG} reading {input_path}", flush=True)
    vol = tifffile.imread(str(input_path))
    if vol.ndim == 2:
        vol = vol[np.newaxis]
    assert vol.ndim == 3, f"expected 3D volume, got shape {vol.shape}"
    print(f"{TAG} volume shape={vol.shape}  dtype={vol.dtype}", flush=True)

    fg = (vol == 1).astype(np.uint8)
    fg_count = int(fg.sum())
    print(f"{TAG} foreground voxels: {fg_count} ({100*fg_count/fg.size:.1f}%)", flush=True)

    # -- Connected components -----------------------------------------------
    print(f"{TAG} running connected components (connectivity={args.connectivity})", flush=True)
    cc_labels, n_raw = _connected_components(fg, args.connectivity)
    print(f"{TAG} raw components: {n_raw}", flush=True)

    # -- Optional size filter + remap to contiguous 1..N --------------------
    if args.min_voxels > 0:
        counts = np.bincount(cc_labels.ravel())
        keep = np.where(counts >= args.min_voxels)[0]
        keep = keep[keep > 0]
        remap = np.zeros(n_raw + 1, dtype=np.int32)
        for new_id, old_id in enumerate(keep, start=1):
            remap[old_id] = new_id
        cc_labels = remap[cc_labels]
        N = int(len(keep))
        print(f"{TAG} after min_voxels={args.min_voxels}: {N} components "
              f"(removed {n_raw - N})", flush=True)
    else:
        N = n_raw

    if N < 2:
        print(f"{TAG} ERROR: need at least 2 components, got {N}", flush=True)
        return 1

    print(f"{TAG} processing {N} components", flush=True)
    shape = cc_labels.shape

    # -- Single pass through distance transforms ----------------------------
    # Track two nearest CCs per voxel + pairwise average distances
    dist_1 = np.full(shape, np.inf, dtype=np.float32)  # nearest CC distance
    dist_2 = np.full(shape, np.inf, dtype=np.float32)  # second-nearest
    idx_1 = np.zeros(shape, dtype=np.int32)             # CC index of nearest (0-based)
    idx_2 = np.zeros(shape, dtype=np.int32)             # CC index of second-nearest
    avg_dist = np.zeros((N, N), dtype=np.float64)       # pairwise avg distances

    for i in range(N):
        cc_id = i + 1  # 1-indexed CC label
        print(f"{TAG} DT {i+1}/{N} (cc_id={cc_id})", flush=True)
        dt_i = distance_transform_edt(cc_labels != cc_id).astype(np.float32)

        # Accumulate pairwise average distances
        for j in range(N):
            if i != j:
                cc_j = j + 1
                mask_j = cc_labels == cc_j
                n_j = int(mask_j.sum())
                if n_j > 0:
                    avg_dist[i, j] = float(dt_i[mask_j].mean())

        # Update two-nearest tracking
        closer = dt_i < dist_1
        second = (~closer) & (dt_i < dist_2)

        # Where dt_i is closer than current nearest: push nearest → second
        dist_2[closer] = dist_1[closer]
        idx_2[closer] = idx_1[closer]
        dist_1[closer] = dt_i[closer]
        idx_1[closer] = i

        # Where dt_i is between nearest and second-nearest
        dist_2[second] = dt_i[second]
        idx_2[second] = i

        del dt_i

    print(f"{TAG} distance transforms complete", flush=True)

    # -- Greedy chain winding assignment ------------------------------------
    # total_avg[i] = mean distance from CC i to all other CCs
    total_avg = np.zeros(N, dtype=np.float64)
    for i in range(N):
        vals = [avg_dist[i, j] for j in range(N) if i != j]
        total_avg[i] = np.mean(vals) if vals else 0.0

    # Start with most isolated CC
    used = set()
    chain = []
    wind_1 = int(np.argmax(total_avg))
    chain.append(wind_1)
    used.add(wind_1)

    # Greedily add closest unused CC
    for _ in range(N - 1):
        last = chain[-1]
        best_j = -1
        best_d = np.inf
        for j in range(N):
            if j not in used and avg_dist[last, j] < best_d:
                best_d = avg_dist[last, j]
                best_j = j
        if best_j < 0:
            break
        chain.append(best_j)
        used.add(best_j)

    # winding_map: 0-based CC index → 1-indexed winding number
    winding_map_arr = np.zeros(N, dtype=np.float32)
    for winding_num, cc_idx in enumerate(chain, start=1):
        winding_map_arr[cc_idx] = float(winding_num)

    winding_map_dict = {int(cc_idx): int(winding_num)
                        for winding_num, cc_idx in enumerate(chain, start=1)}

    print(f"{TAG} winding chain: {chain}", flush=True)
    print(f"{TAG} winding_map: {winding_map_dict}", flush=True)

    # -- Per-voxel winding interpolation ------------------------------------
    w1 = winding_map_arr[idx_1]  # (Z, Y, X) float32
    w2 = winding_map_arr[idx_2]  # (Z, Y, X) float32
    total = np.maximum(dist_1 + dist_2, 1e-8)
    winding_vol = (w1 * dist_2 + w2 * dist_1) / total

    # Free large arrays
    del dist_1, dist_2, idx_1, idx_2, w1, w2, total

    wv_min, wv_max = float(winding_vol.min()), float(winding_vol.max())
    print(f"{TAG} winding volume: min={wv_min:.3f} max={wv_max:.3f}", flush=True)

    # -- Downsample ---------------------------------------------------------
    winding_ds = _downsample_mean(winding_vol, step).astype(np.float32)
    del winding_vol
    print(f"{TAG} downsampled: {shape} → {winding_ds.shape} (step={step})", flush=True)

    # -- Write zarr ---------------------------------------------------------
    cs = args.chunk_size
    arr = zarr.open(
        str(output_path),
        mode="w",
        shape=winding_ds.shape,
        chunks=(cs, cs, cs),
        dtype=np.float32,
        fill_value=0.0,
        dimension_separator="/",
    )
    arr[:] = winding_ds
    arr.attrs["scaledown"] = step
    arr.attrs["n_components"] = N
    arr.attrs["winding_map"] = winding_map_dict
    arr.attrs["winding_chain"] = [int(v) for v in chain]
    print(f"{TAG} zarr written: {output_path}  shape={arr.shape}  "
          f"chunks={arr.chunks}", flush=True)

    print(f"{TAG} done — {N} components, {len(chain)} windings", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
