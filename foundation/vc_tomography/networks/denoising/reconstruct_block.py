"""A block of slices reconstructed with vc_tomography, as the .npz the denoiser trains on and is tried on (noise2inverse.py).

    python reconstruct_block.py DATA --cache DIR --geometry CAL.json --out FILE.npz --heights H0 H1 --roi X0 X1 Y0 Y1
                                [--filters CHAIN] [--backprojector SPEC] [--set KEY=VALUE ..] [--subsets NAME=I/N ..]
                                [--info]

Run with vc_tomography's environment. The file holds one volume (float32, per cm) per subset of the angles, by default
`full` (every projection), `even` and `odd` (every second one), with `heights`, `roi`, `filters` and `backprojector`.

Two ways to a training pair, the same sample with independent noise:

    - `even` and `odd` of one file (Noise2Inverse as published). Each leaves out every second angle; on a scan with
      too few angles what that leaves in a slice has opposite signs in the two, and the network learns to remove it.
    - two runs with vc_tomography's `noise_split` (README, "Slices and volumes"), each from every angle:
          --subsets full=0/1 --set noise_split=A,gain=..,floor=..,smooth=..,flats=..
      once with the scale a and once with -1 / a. This is the way the trained networks were made.

Every parameter that decides the picture is pinned here (`pinned_params`), so that a block does not change with
vc_tomography's defaults; --set changes or adds one.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def value(text):
    """a --set value: true / false, a whole number, a number, or the text as it is"""
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def pinned_params(geometry, filters="none", backprojector="fbp", sets=()):
    """vc_tomography's parameters of a block or a volume: the pinned ones, then `sets` (KEY=VALUE ..) over them"""
    pinned = dict(geometry=str(Path(geometry).resolve()), sharpness_weighting="off", angle_interp="views", z_interp="lanczos8", x_interp="sinc",
                  backprojector=backprojector, filters=filters, luminosity_repartition=0.0, exposure_arc=0.0)
    for item in sets:
        k, v = item.split("=", 1)
        pinned[k] = value(v)
    return pinned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--out")
    ap.add_argument("--heights", type=int, nargs=2)
    ap.add_argument("--roi", type=int, nargs=4)
    ap.add_argument("--filters", default="none")
    ap.add_argument("--backprojector", default="fbp")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--subsets", nargs="+", default=["full=0/1", "even=0/2", "odd=1/2"], help="NAME=I/N ..")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from vc_tomography.recon import HelicalRecon
    pinned = pinned_params(a.geometry, a.filters, a.backprojector, a.set)
    t = time.time()
    r = HelicalRecon(a.data, device=a.device, params=pinned, cache_dir=a.cache, log=lambda s: None)
    print("set up in %.0f s; parameters: %s" % (time.time() - t, pinned), flush=True)
    if a.info:
        for k, v in r.info.items():
            print("  %s: %s" % (k, str(v)[:200]))
        return
    out = {}
    h0, h1 = a.heights
    for item in a.subsets:
        name, subset = item.split("=")
        t = time.time()
        r.set_params(angle_subset=subset)
        out[name] = np.concatenate([np.asarray(r.reconstruct_heights(h, min(h + 8, h1), roi=tuple(a.roi))) for h in range(h0, h1, 8)]).astype(np.float32)
        print("%s (angle_subset %s): %s in %.0f s; std %.4f" % (name, subset, out[name].shape, time.time() - t, out[name].std()), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, heights=np.array(a.heights), roi=np.array(a.roi), filters=a.filters, backprojector=a.backprojector, **out)
    print("written to", a.out)


if __name__ == "__main__":
    sys.exit(main())
