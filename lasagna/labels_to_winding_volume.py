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

    # -- Crop pure-background border -----------------------------------------
    # Label volumes often have a border of bg (0) voxels on all faces.  This
    # border connects non-ignore regions around ignore strips, defeating the
    # disconnected-segment detection below.  Detect and crop it off.
    non_bg = vol != 0  # fg or ignore
    crops = []  # (lo, hi) per axis — hi is from the end
    for ax in range(3):
        # Collapse other axes: True if any non-bg voxel in that slice
        other_axes = tuple(a for a in range(3) if a != ax)
        has_content = non_bg.any(axis=other_axes)  # (size_along_ax,)
        lo = 0
        while lo < len(has_content) and not has_content[lo]:
            lo += 1
        hi = 0
        while hi < len(has_content) and not has_content[len(has_content) - 1 - hi]:
            hi += 1
        crops.append((lo, hi))
    del non_bg
    axis_names = ["Z", "Y", "X"]
    crop_msg = "  ".join(f"{axis_names[i]}: -{crops[i][0]}...-{crops[i][1]}" for i in range(3))
    print(f"{TAG} bg border crop: {crop_msg}", flush=True)
    if any(lo > 0 or hi > 0 for lo, hi in crops):
        sz, sy, sx = vol.shape
        z0, z1 = crops[0][0], sz - crops[0][1]
        y0, y1 = crops[1][0], sy - crops[1][1]
        x0, x1 = crops[2][0], sx - crops[2][1]
        vol = vol[z0:z1, y0:y1, x0:x1]
        print(f"{TAG} cropped volume shape={vol.shape}", flush=True)

    fg = (vol == 1).astype(np.uint8)
    fg_count = int(fg.sum())
    print(f"{TAG} foreground voxels: {fg_count} ({100*fg_count/fg.size:.1f}%)", flush=True)

    # -- Filter disconnected non-ignore regions -----------------------------
    # If ignore labels split the volume into disconnected regions, keep only
    # the largest connected non-ignore region and treat the rest as ignore.
    # This prevents winding jumps across ignore boundaries.
    n_ignore = int((vol == 2).sum())
    print(f"{TAG} ignore voxels: {n_ignore} ({100*n_ignore/vol.size:.1f}%)", flush=True)
    non_ignore = (vol != 2).astype(np.uint8)
    ni_labels, ni_count = _connected_components(non_ignore, args.connectivity)
    del non_ignore
    ni_sizes = np.bincount(ni_labels.ravel())
    ni_sizes[0] = 0  # background of CC (i.e. ignore voxels) doesn't count
    # Print all detected non-ignore segments
    ni_order = np.argsort(-ni_sizes)  # largest first
    ni_order = ni_order[ni_sizes[ni_order] > 0]
    print(f"{TAG} non-ignore connected components: {ni_count}", flush=True)
    for rank, cc_id in enumerate(ni_order):
        n_fg_in = int((fg[ni_labels == cc_id] > 0).sum())
        print(f"{TAG}   segment {rank+1}: cc={cc_id}  size={int(ni_sizes[cc_id])}  fg={n_fg_in}", flush=True)
    if ni_count > 1:
        largest = int(ni_order[0])
        discard_mask = (ni_labels > 0) & (ni_labels != largest)
        n_discard_fg = int((fg[discard_mask] > 0).sum())
        print(f"{TAG} keeping largest segment (cc={largest}, {int(ni_sizes[largest])} voxels), "
              f"discarding {n_discard_fg} fg voxels from {ni_count - 1} smaller parts",
              flush=True)
        fg[discard_mask] = 0
        del discard_mask
        fg_count = int(fg.sum())
        print(f"{TAG} foreground voxels after filtering: {fg_count}", flush=True)
    del ni_labels

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

    # -- Pass 1: distance transforms for nearest CC + pairwise distances ----
    dist_1 = np.full(shape, np.inf, dtype=np.float32)  # nearest CC distance
    idx_1 = np.zeros(shape, dtype=np.int32)             # CC index of nearest (0-based)
    avg_dist = np.zeros((N, N), dtype=np.float64)       # pairwise avg distances

    for i in range(N):
        cc_id = i + 1  # 1-indexed CC label
        print(f"{TAG} DT pass 1: {i+1}/{N} (cc_id={cc_id})", flush=True)
        dt_i = distance_transform_edt(cc_labels != cc_id).astype(np.float32)

        # Accumulate pairwise average distances
        for j in range(N):
            if i != j:
                cc_j = j + 1
                mask_j = cc_labels == cc_j
                n_j = int(mask_j.sum())
                if n_j > 0:
                    avg_dist[i, j] = float(dt_i[mask_j].mean())

        # Update nearest tracking
        closer = dt_i < dist_1
        dist_1[closer] = dt_i[closer]
        idx_1[closer] = i

        del dt_i

    print(f"{TAG} pass 1 complete", flush=True)

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

    # -- Pass 2: chain-adjacent distances + dot-product side detection ------
    prev_in_chain = np.full(N, -1, dtype=np.int32)
    next_in_chain = np.full(N, -1, dtype=np.int32)
    for k in range(N):
        if k > 0:
            prev_in_chain[chain[k]] = chain[k - 1]
        if k < N - 1:
            next_in_chain[chain[k]] = chain[k + 1]

    need_prev = prev_in_chain[idx_1]
    need_next = next_in_chain[idx_1]

    dist_prev = np.full(shape, np.inf, dtype=np.float32)
    dist_next = np.full(shape, np.inf, dtype=np.float32)
    dot_prev = np.zeros(shape, dtype=np.float32)

    dt_cache = {}

    def _get_dt(cc_0idx):
        if cc_0idx not in dt_cache:
            dt_cache[cc_0idx] = distance_transform_edt(
                cc_labels != (cc_0idx + 1)).astype(np.float32)
        return dt_cache[cc_0idx]

    for k in range(N):
        cc_idx = chain[k]
        is_nearest = (idx_1 == cc_idx)
        if not np.any(is_nearest):
            continue
        print(f"{TAG} DT pass 2: chain[{k}]={cc_idx}", flush=True)

        dt_near = _get_dt(cc_idx)

        # Prev: distance + dot product of gradients
        if k > 0:
            prev_idx = chain[k - 1]
            dt_prev_cc = _get_dt(prev_idx)
            dist_prev[is_nearest] = dt_prev_cc[is_nearest]

            for ax in range(3):
                gn = np.gradient(dt_near, axis=ax)
                gp = np.gradient(dt_prev_cc, axis=ax)
                dot_prev[is_nearest] += gn[is_nearest] * gp[is_nearest]
                del gn, gp

        # Next: distance only
        if k < N - 1:
            next_idx = chain[k + 1]
            dt_next_cc = _get_dt(next_idx)
            dist_next[is_nearest] = dt_next_cc[is_nearest]

        # Evict DTs no longer needed
        if k >= 1 and chain[k - 1] in dt_cache:
            del dt_cache[chain[k - 1]]

    del dt_cache

    print(f"{TAG} pass 2 complete", flush=True)

    # -- Envelope mask (mark exterior voxels) -------------------------------
    first_cc_id = chain[0] + 1   # 1-indexed CC label
    last_cc_id = chain[-1] + 1
    print(f"{TAG} computing envelope mask (first_cc={first_cc_id}, last_cc={last_cc_id})", flush=True)

    dt_first = distance_transform_edt(cc_labels != first_cc_id).astype(np.float32)
    dt_last = distance_transform_edt(cc_labels != last_cc_id).astype(np.float32)

    # Dot product of gradients, one axis at a time to limit memory
    dot = np.zeros(shape, dtype=np.float32)
    for axis in range(3):
        g1 = np.gradient(dt_first, axis=axis)
        g2 = np.gradient(dt_last, axis=axis)
        dot += g1 * g2
        del g1, g2
    del dt_first, dt_last

    outside_mask = (dot > 0) & (fg == 0)   # never zero out foreground voxels
    del dot

    # Per-voxel fringe: outside voxels within half a local winding spacing.
    # Use chain-adjacent distance as the spacing reference.
    near_boundary = (idx_1 == chain[0]) | (idx_1 == chain[-1])
    dist_adj = np.minimum(dist_prev, dist_next)
    fringe_mask = outside_mask & (3 * dist_1 <= dist_adj) & near_boundary
    del dist_adj
    far_outside = outside_mask & ~fringe_mask
    n_fringe = int(fringe_mask.sum())
    print(f"{TAG} outside envelope: {int(outside_mask.sum())} voxels "
          f"({100 * outside_mask.sum() / outside_mask.size:.1f}%), "
          f"fringe kept: {n_fringe}", flush=True)

    # -- Per-voxel winding interpolation (chain-adjacent) -------------------
    w_near = winding_map_arr[idx_1]
    w_prev = winding_map_arr[np.clip(need_prev, 0, N - 1)]
    w_next = winding_map_arr[np.clip(need_next, 0, N - 1)]

    use_prev_side = dot_prev < 0
    use_prev_side[need_next < 0] = True   # last CC: no next, must use prev side
    # dot < 0 → gradients oppose → CC prev on opposite side → between prev and nearest
    # dot >= 0 → same side → between nearest and next
    # need_prev == -1 → dot_prev stays 0 → use_prev_side=False → always next side ✓
    del dot_prev

    # -- Debug slices (center YZ plane) ------------------------------------
    xc = shape[2] // 2
    dbg = {}
    dbg["d_lo"] = np.where(use_prev_side, dist_prev, dist_1)[:, :, xc].copy()
    dbg["d_hi"] = np.where(use_prev_side, dist_1, dist_next)[:, :, xc].copy()
    dbg["fringe"] = fringe_mask[:, :, xc].astype(np.float32)
    dbg["interior"] = (~outside_mask & ~fringe_mask)[:, :, xc].astype(np.float32)
    dbg["w_hi"] = np.where(use_prev_side, w_near, w_next)[:, :, xc].copy()
    dbg["w_lo"] = np.where(use_prev_side, w_prev, w_near)[:, :, xc].copy()

    # Case 1: between prev and nearest
    total_1 = np.maximum(dist_prev + dist_1, 1e-8)
    interp_1 = (w_prev * dist_1 + w_near * dist_prev) / total_1

    # Case 2: between nearest and next
    total_2 = np.maximum(dist_1 + dist_next, 1e-8)
    interp_2 = (w_near * dist_next + w_next * dist_1) / total_2

    winding_vol = np.where(use_prev_side, interp_1, interp_2)
    del total_1, total_2, interp_1, interp_2, use_prev_side

    # Fringe: extrapolate using chain-adjacent spacing
    is_first = idx_1[fringe_mask] == chain[0]
    fringe_d_adj = np.where(is_first, dist_next[fringe_mask], dist_prev[fringe_mask])
    fringe_w_adj = np.where(is_first, w_next[fringe_mask], w_prev[fringe_mask])
    fringe_spacing = np.maximum(fringe_d_adj - dist_1[fringe_mask], 1e-8)
    winding_vol[fringe_mask] = (
        w_near[fringe_mask]
        + (w_near[fringe_mask] - fringe_w_adj) * dist_1[fringe_mask] / fringe_spacing
    )
    del fringe_d_adj, fringe_w_adj, fringe_spacing, is_first, outside_mask

    winding_vol[far_outside] = 0.0
    valid_mask = (~far_outside).astype(np.float32)
    del far_outside, fringe_mask

    # Shift all non-zero values up by 1 so the minimum valid value is ~1.5
    # (truly invalid stays 0, well below the >= 1 threshold)
    winding_vol[winding_vol > 0] += 1.0

    # Debug: final winding slice
    dbg["winding"] = winding_vol[:, :, xc].copy()

    # Free large arrays
    del dist_1, dist_prev, dist_next, idx_1, w_near, w_prev, w_next
    del need_prev, need_next

    wv_min, wv_max = float(winding_vol.min()), float(winding_vol.max())
    print(f"{TAG} winding volume: min={wv_min:.3f} max={wv_max:.3f}", flush=True)

    # -- Debug TIFF (center YZ plane) --------------------------------------
    dbg_path = output_path.parent / (output_path.stem + "_dbg.tif")
    dbg_names = sorted(dbg)
    with tifffile.TiffWriter(str(dbg_path)) as tw:
        for name in dbg_names:
            tw.write(dbg[name].astype(np.float32),
                     extratags=[(285, 's', 0, name, False)])
    print(f"{TAG} debug TIFF: {dbg_path}  layers={dbg_names}", flush=True)
    del dbg

    # -- Downsample ---------------------------------------------------------
    valid_ds = _downsample_mean(valid_mask, step)
    del valid_mask
    winding_ds_raw = _downsample_mean(winding_vol, step)
    del winding_vol
    # Divide by valid fraction to get proper weighted average; zero where no valid voxels
    winding_ds = np.where(valid_ds > 0, winding_ds_raw / valid_ds, 0.0).astype(np.float32)
    del valid_ds, winding_ds_raw
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
        zarr_format=2,
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
