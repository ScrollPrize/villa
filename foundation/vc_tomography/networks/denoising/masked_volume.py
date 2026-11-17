"""A scan reconstructed with vc_tomography, denoised by a trained network (noise2inverse.py) and written as a uint8
OME-Zarr under a mask.

    python masked_volume.py DATA --cache DIR --geometry CAL.json --filters CHAIN [--set KEY=VALUE ..]
                            --window LO HI --voxel_um UM --mask MASK.npy --model model.pt
                            --zarr OUT.zarr [--zarr_not_denoised RAW.zarr] [--slices J0 J1]

Run with vc_tomography's environment, installed with the `denoise` extra (PyTorch). The reconstruction's parameters
are those of reconstruct_block.py (`pinned_params`: the same recipe as the network's training blocks).

The volume is vc_tomography's slice grid (grid x grid) over every height the scan can give: slice j is view height
(first doable height) + j. It is made in segments along the axis, each inside the box the mask keeps over the
segment (plus a margin for the network's halo), one after the other in this process:

    1. the segment's slices (with HALO_Z neighbours either side) reconstructed into memory by a reconstructor of their
       own, which is then destroyed: the reconstruction and the network do not share the device, and a reconstructor's
       buffers, which grow as it goes, last one segment;
    2. the network on those slices (noise2inverse.run_to_uint8), its memory given back to the device after it;
    3. the uint8 slices set to 0 outside the mask, put at their place in whole slices and given to the writer.

A segment whose reconstruction runs out of device memory starts again with a new reconstructor and half as many
slices a call, from the slice where it stopped. The host holds one segment's float slices: (segment + 2 HALO_Z) x the
box x 4 bytes (--segment makes it smaller). A run that stops is made again from the start (--slices makes a part).

The mask is a .npy (z, y, x) at a quarter of the volume's resolution, not 0 where the volume is kept: each of its
voxels decides the 4 x 4 x 4 voxels it covers (as vc_tomography's --reference_volume at --reference_scale 2 does).
"""
import argparse
import gc
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MARGIN = 48                                # pixels of real slice around the mask's box, for the network's halo
BLOCK = 16                                 # slices given to the writer at a time


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rb = load("reconstruct_block")


def reconstruct_box(a, pinned, a0, n, roi, log):
    """slices [a0, a0 + n) of the volume inside roi = (x0, x1, y0, y1), float32 per cm on the host; a reconstructor
    that runs out of device memory is replaced by one that takes half as many slices a call"""
    from vc_tomography.recon import HelicalRecon
    x0, x1, y0, y1 = roi
    v = np.empty((n, y1 - y0, x1 - x0), np.float32)
    k, chunk = 0, a.chunk
    while k < n:
        r = HelicalRecon(a.data, device="cuda", params=pinned, cache_dir=a.cache, log=lambda s: None)
        try:
            for q in range(k, n, chunk):
                e = min(q + chunk, n)
                v[q:e] = r.reconstruct_heights(r.lo + a0 + q, r.lo + a0 + e, roi=roi)
                k = e
        except Exception as err:
            if "out of memory" not in str(err) or chunk <= 2:
                raise
            chunk //= 2
            log("  the reconstruction ran out of device memory at slice %d; going on with %d slices a call" % (a0 + k, chunk))
        finally:
            del r
            gc.collect()           # (HelicalRecon holds a reference cycle: its end, which frees the device, waits for this)
    if not np.isfinite(v).all():
        raise SystemExit("slices %d..%d hold values that are not finite" % (a0, a0 + n - 1))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--filters", required=True)
    ap.add_argument("--backprojector", default="fbp")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE over the pinned parameters (reconstruct_block.pinned_params)")
    ap.add_argument("--window", type=float, nargs=2, required=True, help="the uint8 window (per cm)")
    ap.add_argument("--voxel_um", type=float, required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--zarr_not_denoised", default="", help="a second volume: the same slices without the network (same window, same mask)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mask", required=True, help="the volume's 4x level > 0, as .npy (z, y, x)")
    ap.add_argument("--slices", type=int, nargs=2, default=None, help="slices [J0, J1) of the volume (default: all)")
    ap.add_argument("--segment", type=int, default=480)
    ap.add_argument("--chunk", type=int, default=16, help="slices per reconstruction call")
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    import torch
    from vc_tomography.omezarr import OmeZarrWriter
    from vc_tomography.recon import HelicalRecon
    n2i = load("noise2inverse")
    pinned = rb.pinned_params(a.geometry, a.filters, a.backprojector, a.set)
    log = lambda text: print(text, flush=True)      # noqa: E731
    window = tuple(a.window)
    r = HelicalRecon(a.data, device="cpu", params=pinned, cache_dir=a.cache, log=lambda s: None)     # (for what it tells; nothing of the device)
    lo, hi = r.lo, r.hi
    n_all, grid = hi - lo, int(r.info["grid_size"])
    resolved = r.provenance()["resolved_sharpness_weighting"]
    del r
    gc.collect()
    mask = np.load(a.mask, mmap_mode="r")
    if abs(mask.shape[0] * 4 - n_all) > 4 or abs(mask.shape[1] * 4 - grid) > 4:
        raise SystemExit("the mask %s is not the 4x level of %d slices of %d px" % (mask.shape, n_all, grid))
    dev = torch.device("cuda")
    net = n2i.load_net(a.model, dev)
    j0, j1 = a.slices if a.slices else (0, n_all)
    attrs = dict(vc_tomography_denoised_volume=dict(
        scans=a.data, view_heights=[lo + j0, lo + j1], z_of_height="height - %d" % lo, parameters=pinned, resolved_sharpness_weighting=resolved,
        denoiser=str(a.model), uint8_window_per_cm=list(window), mask=str(a.mask), script=str(Path(__file__).resolve())))
    w = OmeZarrWriter(a.zarr, (n_all, grid, grid), np.uint8, chunks=(128, 128, 128), voxel_size_um=a.voxel_um, levels=a.levels, compression="vc-delta3d",
                      name=Path(a.zarr).stem, attrs=attrs, overwrite=a.overwrite)
    w_raw = None
    if a.zarr_not_denoised:
        attrs_raw = dict(vc_tomography_volume=dict(attrs["vc_tomography_denoised_volume"], denoiser="none (the reconstruction as it is; the twin of %s)" % a.zarr))
        w_raw = OmeZarrWriter(a.zarr_not_denoised, (n_all, grid, grid), np.uint8, chunks=(128, 128, 128), voxel_size_um=a.voxel_um, levels=a.levels,
                              compression="vc-delta3d", name=Path(a.zarr_not_denoised).stem, attrs=attrs_raw, overwrite=a.overwrite)
    log("slices [%d, %d) of %d (view heights [%d, %d)) on a grid of %d px; sharpness weighting %s; %s" % (j0, j1, n_all, lo + j0, lo + j1, grid, resolved, a.zarr))
    t0 = time.time()
    for s0 in range(j0, j1, a.segment):
        s1 = min(s0 + a.segment, j1)
        a0, a1 = max(s0 - n2i.HALO_Z, 0), min(s1 + n2i.HALO_Z, n_all)      # with the neighbours the network leans on
        if a1 - a0 < n2i.SLAB:
            a0 = max(a1 - n2i.SLAB, 0)
            a1 = min(a0 + n2i.SLAB, n_all)
        m = np.asarray(mask[a0 // 4: (a1 - 1) // 4 + 1])
        if not m.any():
            log("segment %d..%d: nothing kept by the mask" % (s0, s1 - 1))
            continue
        ys, xs = np.flatnonzero(m.any((0, 2))), np.flatnonzero(m.any((0, 1)))
        y0, y1 = max(4 * ys[0] - MARGIN, 0), min(4 * (ys[-1] + 1) + MARGIN, grid)
        x0, x1 = max(4 * xs[0] - MARGIN, 0), min(4 * (xs[-1] + 1) + MARGIN, grid)
        ny, nx = y1 - y0, x1 - x0
        t = time.time()
        v = reconstruct_box(a, pinned, a0, a1 - a0, (x0, x1, y0, y1), log)
        t_rec = time.time() - t
        raw = None
        if w_raw is not None:      # the reconstruction as it is, in the same window
            raw = np.clip(np.rint((v - window[0]) / (window[1] - window[0]) * 255), 0, 255).astype(np.uint8)
        t = time.time()
        den = n2i.run_to_uint8(net, v, window[0], window[1], dev)
        del v
        torch.cuda.empty_cache()   # (the device back to the next segment's reconstructor)
        t_den = time.time() - t
        for out, writer in ((den, w), (raw, w_raw)):
            if writer is None:
                continue
            for g0 in range(s0, s1, BLOCK):
                g1 = min(g0 + BLOCK, s1)
                block = np.zeros((g1 - g0, grid, grid), np.uint8)
                for i, z in enumerate(range(g0, g1)):
                    keep = np.repeat(np.repeat(np.asarray(mask[min(z // 4, mask.shape[0] - 1)]), 4, 0), 4, 1)[y0:y1, x0:x1]
                    if keep.shape != (ny, nx):          # (the grid's last pixels beyond the mask's 4 x 4 blocks)
                        keep = np.pad(keep, ((0, ny - keep.shape[0]), (0, nx - keep.shape[1])))
                    block[i, y0:y1, x0:x1] = np.where(keep, out[z - a0], 0)
                writer.write(g0, block)
        log("segment %d..%d: box %d x %d px at (%d, %d); reconstructed %d slices in %.0f s, denoised in %.0f s; %.1f min so far" %
            (s0, s1 - 1, ny, nx, y0, x0, a1 - a0, t_rec, t_den, (time.time() - t0) / 60))
    t = time.time()
    w.close()
    if w_raw is not None:
        w_raw.close()
    log("the Zarr closed (its levels made) in %.1f min; %d chunks stored, %d empty; all in %.1f min" % ((time.time() - t) / 60, w.n_written, w.n_skipped, (time.time() - t0) / 60))


if __name__ == "__main__":
    sys.exit(main())
