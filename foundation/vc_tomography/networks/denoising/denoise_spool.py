"""The denoiser (noise2inverse.py) on a run of slices that another process reconstructs: the two pass blocks of slices
through a spool folder, because helirec and PyTorch live in two environments.

    python denoise_spool.py --model model.pt --spool DIR --slices N --window LO HI --shape ROWS COLUMNS

Run with an environment that has PyTorch. The spool:

    recon_jAAAAA_BBBBB.npy   slices [A, B) of the run, float32 per cm, BLOCK to a file, written by the reconstruction
    den_jAAAAA_BBBBB.npy     the same slices denoised, uint8 in the window, written here
    den_done / den_failed    written here at the end (den_failed holds the error)

Files are written under another name and renamed (`save`), so a file that is there is whole. The reconstruction's
files are removed as they are read.

The network sees slabs of SLAB slices of which the middle KEEP are kept (HALO_Z neighbours either side: against 16
the result differs by 2 % of what the network changes), the first and last slab to the run's ends; in the plane,
tiles of 256 px with a halo of 48, reflected at the extent's ends.

The module's top level needs numpy alone: the reconstruction's side imports `save`, `spool_files` and the sizes.
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BLOCK = 16                                       # slices per spool file
SLAB, KEEP, HALO_Z = 64, 48, 8


def save(path, array):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def spool_files(spool, kind):
    """the spool's files of a kind, (first slice, end, path), in order"""
    out = []
    for f in spool.glob(kind + "_j*.npy"):
        if ".tmp" in f.name:
            continue
        a, b = (int(v) for v in f.stem.split("_j")[1].split("_"))
        out.append((a, b, f))
    return sorted(out)


def load_noise2inverse():
    spec = importlib.util.spec_from_file_location("noise2inverse", HERE / "noise2inverse.py")
    n2i = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n2i)
    return n2i


def slab_to_uint8(net, vol, k0, k1, lo, hi, dev, n2i, torch, F, tile=256, halo=48):
    """the network on vol (a slab on the host), its slices [k0, k1) as uint8 in the window [lo, hi]; in tiles of the
    plane with a halo, reflected where the volume ends (the arithmetic of noise2inverse.run_tiled, a tile at a time
    so that the slab stays on the host)"""
    ny, nx = vol.shape[1:]
    out = np.empty((k1 - k0, ny, nx), np.uint8)
    with torch.no_grad(), torch.autocast("cuda", dtype=n2i.AMP[net.amp]):
        for y in range(0, ny, tile):
            for x in range(0, nx, tile):
                h, w = min(tile, ny - y), min(tile, nx - x)
                ya, yb, xa, xb = max(y - halo, 0), min(y + tile + halo, ny), max(x - halo, 0), min(x + tile + halo, nx)
                t = torch.from_numpy(vol[:, ya:yb, xa:xb]).to(dev)[None, None]
                top, left = halo - (y - ya), halo - (x - xa)
                bottom, right = min(halo, max(0, y + tile + halo - ny)), min(halo, max(0, x + tile + halo - nx))
                if top or left or bottom or right:
                    t = F.pad(t, (left, right, top, bottom, 0, 0), mode="reflect")
                py, px = tile + 2 * halo - t.shape[3], tile + 2 * halo - t.shape[4]
                if py or px:
                    t = F.pad(t, (0, px, 0, py, 0, 0), mode="replicate")
                r = net(t * n2i.SCALE)[0, 0].float() / n2i.SCALE
                r = r[k0:k1, halo: halo + h, halo: halo + w]
                out[:, y: y + h, x: x + w] = torch.round(torch.clamp((r - lo) / (hi - lo), 0, 1) * 255).to(torch.uint8).cpu().numpy()
    return out


def denoise(a):
    import torch
    import torch.nn.functional as F
    n2i = load_noise2inverse()
    spool = Path(a.spool)
    dev = torch.device("cuda")
    net = n2i.load_net(a.model, dev)
    lo, hi = a.window
    n = a.slices
    if n < SLAB:
        raise SystemExit("a run of %d slices is shorter than a slab of %d" % (n, SLAB))
    buf = np.empty((SLAB + KEEP, a.shape[0], a.shape[1]), np.float32)
    base = count = 0            # buf[:count] holds the slices [base, base + count); a slab starts at base
    done_to = 0                 # slices [0, done_to) are denoised
    t0 = time.time()

    def fill(to):
        """the slices up to `to` from the spool into buf"""
        nonlocal count
        while base + count < to:
            files = [f for f in spool_files(spool, "recon") if f[0] == base + count]
            if not files:
                time.sleep(1)
                continue
            ja, jb, f = files[0]
            buf[count: count + jb - ja] = np.load(f)
            count += jb - ja
            f.unlink()

    def emit(first, end, slab_first):
        """slices [first, end) from the slab that starts at slice slab_first"""
        nonlocal done_to
        t = time.time()
        out = slab_to_uint8(net, buf[slab_first - base: slab_first - base + SLAB], first - slab_first, end - slab_first, lo, hi, dev, n2i, torch, F)
        save(spool / ("den_j%05d_%05d.npy" % (first, end)), out)
        print("slices %d..%d from the slab %d..%d in %.0f s; %.1f min so far" % (first, end - 1, slab_first, slab_first + SLAB - 1, time.time() - t, (time.time() - t0) / 60), flush=True)
        done_to = end

    try:
        while done_to < n:
            fill(base + SLAB)
            emit(done_to, n if base + SLAB == n else base + SLAB - HALO_Z, base)
            if done_to == n:
                break
            if base + KEEP + SLAB > n:      # no further slab fits: the run's last SLAB slices give the rest
                fill(n)
                emit(done_to, n, n - SLAB)
                break
            buf[: count - KEEP] = buf[KEEP: count]
            base += KEEP
            count -= KEEP
        (spool / "den_done").write_text(str(n))
        print("denoised %d slices in %.1f min" % (n, (time.time() - t0) / 60), flush=True)
    except BaseException as e:
        (spool / "den_failed").write_text(repr(e))
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spool", required=True)
    ap.add_argument("--slices", type=int, required=True, help="slices in the run")
    ap.add_argument("--window", type=float, nargs=2, required=True, help="the uint8 window (per cm)")
    ap.add_argument("--shape", type=int, nargs=2, required=True, help="rows and columns of the spool's slices")
    return denoise(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
