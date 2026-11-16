"""A scan reconstructed with helirec, denoised by a trained network (noise2inverse.py) and written as a uint8
OME-Zarr under a mask, where the reconstruction needs the device to itself.

    python masked_volume.py DATA --cache DIR --geometry CAL.json --filters CHAIN [--set KEY=VALUE ..]
                            --window LO HI --voxel_um UM --mask MASK.npy --model model.pt --torch_python PYTHON
                            --zarr OUT.zarr [--zarr_not_denoised RAW.zarr] --spool DIR [--slices J0 J1]

Run with helirec's environment; --torch_python is the Python of an environment that has PyTorch. The reconstruction's
parameters are those of reconstruct_block.py (`pinned_params`: the same recipe as the network's training blocks).

The volume is helirec's slice grid (grid x grid) over every height the scan can give: slice j is view height
(first doable height) + j. It is made in segments along the axis, each inside the box the mask keeps over the
segment (plus a margin for the network's halo):

    1. the segment's slices (with 8 neighbours either side) reconstructed into the spool folder by a process of their
       own (the device is clean for it, and after it: a reconstructor released inside a process leaves some 4 GB
       behind, which with the next one was too much on a device of 32 GB);
    2. denoise_spool.py on the spool, as a process of the PyTorch environment (the device to itself);
    3. what comes back (uint8) set to 0 outside the mask, put at its place in whole slices and given to the writer.

A segment whose reconstruction fails (the device's memory) goes on in a new process with half as many slices a call.
A finished segment's files are kept in the spool, so a run that stops is taken up where it was.

The mask is a .npy (z, y, x) at a quarter of the volume's resolution, not 0 where the volume is kept: each of its
voxels decides the 4 x 4 x 4 voxels it covers (as helirec's --reference_volume at --reference_scale 2 does).
"""
import argparse
import gc
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sp = load("denoise_spool")                 # (numpy alone at its top level)
rb = load("reconstruct_block")
MARGIN = 48                                # pixels of real slice around the mask's box, for the network's halo


def launcher():
    """the command this script was started with, up to the script itself (python, or python through a runner that
    sets up the build): a process started with it sees the same helirec"""
    me = Path(__file__).resolve()
    for i, v in enumerate(sys.orig_argv):
        if i and Path(v).resolve() == me:
            return [sys.executable] + sys.orig_argv[1: i + 1]
    return [sys.executable, str(me)]


def reconstruct_segment(a, pinned):
    """A process of its own (the device is clean when it starts, and whatever a failure leaves behind goes with it):
    slices [a0, a0 + n) of the volume inside a box, BLOCK to a spool file; files that are there already are kept."""
    from helirec.recon import HelicalRecon
    spool = Path(a.spool)
    x0, x1, y0, y1 = a.roi
    r = HelicalRecon(a.data, device="cuda", params=pinned, cache_dir=a.cache, log=lambda s: None)
    for k in range(0, a.n, sp.BLOCK):
        e = min(k + sp.BLOCK, a.n)
        f = spool / ("recon_j%05d_%05d.npy" % (k, e))
        if f.exists():
            continue
        v = np.concatenate([np.asarray(r.reconstruct_heights(r.lo + a.a0 + q, r.lo + a.a0 + min(q + a.chunk, e), roi=(x0, x1, y0, y1))) for q in range(k, e, a.chunk)]).astype(np.float32)
        if not np.isfinite(v).all():
            raise SystemExit("slices %d..%d hold values that are not finite" % (a.a0 + k, a.a0 + e - 1))
        sp.save(f, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--filters", required=True)
    ap.add_argument("--backprojector", default="fbp")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE over the pinned parameters (reconstruct_block.pinned_params)")
    ap.add_argument("--window", type=float, nargs=2, help="the uint8 window (per cm)")
    ap.add_argument("--voxel_um", type=float)
    ap.add_argument("--zarr")
    ap.add_argument("--zarr_not_denoised", default="", help="a second volume: the same slices without the network (same window, same mask)")
    ap.add_argument("--spool", required=True)
    ap.add_argument("--model")
    ap.add_argument("--torch_python", help="the Python of an environment with PyTorch")
    ap.add_argument("--mask", help="the volume's 4x level > 0, as .npy (z, y, x)")
    ap.add_argument("--slices", type=int, nargs=2, default=None, help="slices [J0, J1) of the volume (default: all)")
    ap.add_argument("--segment", type=int, default=480)
    ap.add_argument("--chunk", type=int, default=16, help="slices per reconstruction call")
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--reconstruct_segment", action="store_true", help="(used by the script itself) with --a0, --n, --roi")
    ap.add_argument("--a0", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument("--roi", type=int, nargs=4)
    a = ap.parse_args()
    pinned = rb.pinned_params(a.geometry, a.filters, a.backprojector, a.set)
    if a.reconstruct_segment:
        return reconstruct_segment(a, pinned)
    missing = [k for k in ("window", "voxel_um", "zarr", "model", "torch_python", "mask") if not getattr(a, k)]
    if missing:
        ap.error("needed: " + ", ".join("--" + k for k in missing))
    from helirec.omezarr import OmeZarrWriter
    from helirec.recon import HelicalRecon
    window = tuple(a.window)
    spool = Path(a.spool)
    spool.mkdir(parents=True, exist_ok=True)
    for f in list(spool.glob("*.npy")) + [spool / n for n in ("recon_done", "den_done", "den_failed")]:
        if f.exists():
            f.unlink()
    r = HelicalRecon(a.data, device="cpu", params=pinned, cache_dir=a.cache, log=lambda s: None)     # (for what it tells; nothing of the device)
    lo, hi = r.lo, r.hi
    n_all, grid = hi - lo, int(r.info["grid_size"])
    resolved = r.provenance()["resolved_sharpness_weighting"]
    del r
    gc.collect()
    mask = np.load(a.mask, mmap_mode="r")
    if abs(mask.shape[0] * 4 - n_all) > 4 or abs(mask.shape[1] * 4 - grid) > 4:
        raise SystemExit("the mask %s is not the 4x level of %d slices of %d px" % (mask.shape, n_all, grid))
    j0, j1 = a.slices if a.slices else (0, n_all)
    attrs = dict(helirec_denoised_volume=dict(
        scans=a.data, view_heights=[lo + j0, lo + j1], z_of_height="height - %d" % lo, parameters=pinned, resolved_sharpness_weighting=resolved,
        denoiser=str(a.model), uint8_window_per_cm=list(window), mask=str(a.mask), script=str(Path(__file__).resolve())))
    w = OmeZarrWriter(a.zarr, (n_all, grid, grid), np.uint8, chunks=(128, 128, 128), voxel_size_um=a.voxel_um, levels=a.levels, compression="vc-delta3d",
                      name=Path(a.zarr).stem, attrs=attrs, overwrite=a.overwrite)
    w_raw = None
    if a.zarr_not_denoised:
        attrs_raw = dict(helirec_volume=dict(attrs["helirec_denoised_volume"], denoiser="none (the reconstruction as it is; the twin of %s)" % a.zarr))
        w_raw = OmeZarrWriter(a.zarr_not_denoised, (n_all, grid, grid), np.uint8, chunks=(128, 128, 128), voxel_size_um=a.voxel_um, levels=a.levels,
                              compression="vc-delta3d", name=Path(a.zarr_not_denoised).stem, attrs=attrs_raw, overwrite=a.overwrite)
    print("slices [%d, %d) of %d (view heights [%d, %d)) on a grid of %d px; sharpness weighting %s; %s" % (j0, j1, n_all, lo + j0, lo + j1, grid, resolved, a.zarr), flush=True)
    t0 = time.time()
    recipe = [a.data, "--cache", a.cache, "--geometry", a.geometry, "--filters", a.filters, "--backprojector", a.backprojector]
    for item in a.set:
        recipe += ["--set", item]
    me = launcher() + recipe
    for s0 in range(j0, j1, a.segment):
        s1 = min(s0 + a.segment, j1)
        a0, a1 = max(s0 - sp.HALO_Z, 0), min(s1 + sp.HALO_Z, n_all)      # with the neighbours the network leans on
        if a1 - a0 < sp.SLAB:
            a0 = max(a1 - sp.SLAB, 0)
            a1 = min(a0 + sp.SLAB, n_all)
        m = np.asarray(mask[a0 // 4: (a1 - 1) // 4 + 1])
        if not m.any():
            print("segment %d..%d: nothing kept by the mask" % (s0, s1 - 1), flush=True)
            continue
        ys, xs = np.flatnonzero(m.any((0, 2))), np.flatnonzero(m.any((0, 1)))
        y0, y1 = max(4 * ys[0] - MARGIN, 0), min(4 * (ys[-1] + 1) + MARGIN, grid)
        x0, x1 = max(4 * xs[0] - MARGIN, 0), min(4 * (xs[-1] + 1) + MARGIN, grid)
        ny, nx = y1 - y0, x1 - x0
        n = a1 - a0
        done = spool / ("segment_%05d" % s0)           # a finished segment's uint8 files are kept: a later failure does not cost them
        t_rec = t_den = 0.0
        if not (done / "done").exists():
            t = time.time()
            chunk = a.chunk
            while True:
                # (on a failure the segment goes on, in a new process, with half as many slices a call: the
                # reconstructor's memory on the device rises with them)
                with open(spool / ("reconstruct_%05d.log" % s0), "a") as f:
                    code = subprocess.call(me + ["--reconstruct_segment", "--spool", str(spool), "--a0", str(a0), "--n", str(n), "--roi", str(x0), str(x1), str(y0), str(y1),
                                                 "--chunk", str(chunk)], stdout=f, stderr=subprocess.STDOUT)
                if code == 0:
                    break
                if chunk <= 2:
                    raise SystemExit("the reconstruction of segment %d failed (%s)" % (s0, spool / ("reconstruct_%05d.log" % s0)))
                chunk //= 2
                print("  segment %d..%d: the reconstruction stopped (%d files written); going on with %d slices a call" %
                      (s0, s1 - 1, len(sp.spool_files(spool, "recon")), chunk), flush=True)
            t_rec = time.time() - t
            done.mkdir(exist_ok=True)
            if w_raw is not None:      # the reconstruction as it is, in the same window (before the network takes the files)
                for ja, jb, f in sp.spool_files(spool, "recon"):
                    v = np.load(f)
                    sp.save(done / ("raw_j%05d_%05d.npy" % (ja, jb)), np.clip(np.rint((v - window[0]) / (window[1] - window[0]) * 255), 0, 255).astype(np.uint8))
            t = time.time()
            log = spool / ("denoise_%05d.log" % s0)
            with open(log, "w") as f:
                code = subprocess.call([a.torch_python, str(HERE / "denoise_spool.py"), "--model", a.model, "--spool", str(spool), "--slices", str(n),
                                        "--window", str(window[0]), str(window[1]), "--shape", str(ny), str(nx)], stdout=f, stderr=subprocess.STDOUT)
            if code or not (spool / "den_done").exists():
                raise SystemExit("the denoising of segment %d failed (%s)" % (s0, log))
            t_den = time.time() - t
            for ja, jb, f in sp.spool_files(spool, "den"):
                f.rename(done / f.name)
            (spool / "den_done").unlink()
            (done / "done").write_text("%d %d %d %d" % (x0, x1, y0, y1))
        elif (done / "done").read_text().split() != [str(v) for v in (x0, x1, y0, y1)]:
            raise SystemExit("%s was made for another box" % done)
        wrote = 0
        for kind, writer in (("den", w), ("raw", w_raw)):
            if writer is None:
                continue
            for ja, jb, f in sp.spool_files(done, kind):
                v = np.load(f)
                g0, g1 = max(a0 + ja, s0), min(a0 + jb, s1)        # of the volume's slices, those of this segment
                if g1 <= g0:
                    continue
                v = v[g0 - a0 - ja: g1 - a0 - ja]
                block = np.zeros((g1 - g0, grid, grid), np.uint8)
                for i, z in enumerate(range(g0, g1)):
                    keep = np.repeat(np.repeat(np.asarray(mask[min(z // 4, mask.shape[0] - 1)]), 4, 0), 4, 1)[y0:y1, x0:x1]
                    if keep.shape != (ny, nx):          # (the grid's last pixels beyond the mask's 4 x 4 blocks)
                        keep = np.pad(keep, ((0, ny - keep.shape[0]), (0, nx - keep.shape[1])))
                    block[i, y0:y1, x0:x1] = np.where(keep, v[i], 0)
                writer.write(g0, block)
                wrote += (g1 - g0) if kind == "den" else 0
        print("segment %d..%d: box %d x %d px at (%d, %d); reconstructed %d slices in %.0f s, denoised in %.0f s, %d written; %.1f min so far" %
              (s0, s1 - 1, ny, nx, y0, x0, n, t_rec, t_den, wrote, (time.time() - t0) / 60), flush=True)
    t = time.time()
    w.close()
    if w_raw is not None:
        w_raw.close()
    print("the Zarr closed (its levels made) in %.1f min; %d chunks stored, %d empty; all in %.1f min" % ((time.time() - t) / 60, w.n_written, w.n_skipped, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    sys.exit(main())
