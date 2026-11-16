"""Noise2Inverse for vc_tomography's reconstructions: a 3D U-Net trained on pairs of reconstructions of the same slices
with independent noise, and applied to a reconstruction. Not in the vc_tomography package: it needs PyTorch.

Hendriksen, Pelt & Batenburg, "Noise2Inverse: self-supervised deep convolutional denoising for tomography" (2020).
The frames' noise is independent from frame to frame, so the reconstructions of the even and of the odd projections
hold the same sample and independent noise. A network that is given one and asked for the other cannot predict the
other's noise: the least squares answer is the sample as far as the input tells it. Here: a small 3D U-Net (three
scales, 32 / 64 / 128 channels by default: --width, --scales; no normalisation layers), which predicts what to take off its input; trained on
blocks of the two volumes of slabs (reconstruct_block.py), in both directions, with flips and the
transpose in the plane; squared error.

    noise2inverse.py train --slabs A.npz B.npz .. --out DIR [--minutes 10] [--watch TEST.npz]
    noise2inverse.py apply --model DIR/model.pt --block TEST.npz --out FILE.npz
    noise2inverse.py slices --model DIR/model.pt --block SLAB.npz --out DIR --name NAME --slices 16 .. --window LO HI

The halves leave out every second angle, and on a scan with too few angles what that leaves in a slice has opposite
signs in the two halves: the network learns it, and loses detail with it (measured on PHerc. 172). A slab given as
A.npz+B.npz is instead a pair of independent recordings of the same slices, each from all its angles (`full` of each
file: the two rows of scans where both record a height, each reconstructed alone): Noise2Noise, nothing left out.
A.npz>B.npz is a pair of which A is always the input: two runs of vc_tomography's noise_split on the same projections
(scales a and -1 / a: A with (1 + a^2) of the noise variance, close to the data's, B with (1 + 1 / a^2) of it).

apply writes, for the block's region: even, odd (the network on each half: neither has seen the other half), full
(their mean: the picture of all the projections) and full_direct (the network on the reconstruction of all the
projections, whose noise is half of what it was trained on). Run with an environment that has PyTorch (in this
repository: the vesuvius one); it reads and writes .npz files only. Whole volumes: denoise_spool.py, masked_volume.py.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE = 10.0     # the slices are per cm (standard deviation about 0.07): the network works on 10 times that
AMP = {"fp16": torch.float16, "bf16": torch.bfloat16}   # half precision of the convolutions (bf16: fp32's range, 8 bits)


def block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
                         nn.Conv3d(cout, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))


class UNet3d(nn.Module):
    """`scales` resolutions, each half the one before (width, 2 width, .. channels); 3 scales of 32: 1.41 million
    parameters, 4 scales of 48: 13.1 million"""

    def __init__(self, width=32, scales=3):
        super().__init__()
        ch = [width * 2 ** i for i in range(scales)]
        self.scales = scales
        self.amp = "fp16"   # (the precision the network was trained in: run_tiled uses the same)
        self.enc = nn.ModuleList([block(1 if i == 0 else ch[i - 1], ch[i]) for i in range(scales)])
        self.dec = nn.ModuleList([block(ch[i + 1] + ch[i], ch[i]) for i in range(scales - 1)])
        self.out = nn.Conv3d(width, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        feats = []
        h = x
        for i, e in enumerate(self.enc):
            h = e(h if i == 0 else F.avg_pool3d(h, 2))
            feats.append(h)
        for i in range(self.scales - 2, -1, -1):
            h = self.dec[i](torch.cat([F.interpolate(h, size=feats[i].shape[2:], mode="trilinear", align_corners=False), feats[i]], 1))
        return x - self.out(h)


def load_net(path, dev):
    """the network of a saved model (models saved before the scales could be chosen: 3 scales, their layers' names)"""
    saved = torch.load(path, map_location=dev)
    net = UNet3d(saved["width"], saved.get("scales", 3)).to(dev)
    net.amp = saved.get("amp", "fp16")
    state = saved["state"]
    if "e1.0.weight" in state:
        old = {"e1": "enc.0", "e2": "enc.1", "mid": "enc.2", "d1": "dec.0", "d2": "dec.1"}
        state = {(old[k.split(".")[0]] + k[k.index("."):] if k.split(".")[0] in old else k): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    return net


def run_tiled(net, v, tile=256, halo=48):
    """the network over a volume (z, y, x), in tiles of the plane with a halo (reflected at the ends); the volume may
    be on the host (a whole slab of slices): its tiles go to the network's device one by one"""
    dev = next(net.parameters()).device
    z, ny, nx = v.shape
    m = 2 ** (net.scales - 1)
    assert z % m == 0 and (tile + 2 * halo) % m == 0
    p = F.pad(v[None, None], (halo, halo, halo, halo, 0, 0), mode="reflect")[0, 0]
    out = torch.empty_like(v)
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP[net.amp]):
        for y in range(0, ny, tile):
            for x in range(0, nx, tile):
                h, w = min(tile, ny - y), min(tile, nx - x)
                t = p[:, y: y + tile + 2 * halo, x: x + tile + 2 * halo]
                py, px = tile + 2 * halo - t.shape[1], tile + 2 * halo - t.shape[2]
                if py or px:
                    t = F.pad(t[None, None], (0, px, 0, py, 0, 0), mode="replicate")[0, 0]   # (a last tile: filled to the size, cut away below)
                r = net(t[None, None].to(dev) * SCALE)[0, 0].float() / SCALE
                out[:, y: y + h, x: x + w] = r[:, halo: halo + h, halo: halo + w].to(v.device)
    return out


def train(a):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)
    slabs = []
    def pair(f):
        """the two volumes of a slab: the halves of FILE, or `full` of FILE_A and of FILE_B (FILE_A+FILE_B: either
        is the input; FILE_A>FILE_B: A is always the input, B the target)"""
        for mark in ("+", ">"):
            if mark in f:
                fa, fb = f.split(mark)
                za, zb = np.load(fa), np.load(fb)
                return za["full"], zb["full"], str(za["filters"]), str(za["backprojector"])
        z = np.load(f)
        return z["even"], z["odd"], str(z["filters"]), str(z["backprojector"])

    held = torch.device("cpu") if a.host else dev   # (--host: the slabs stay in the host's memory, the blocks go to the device)
    for f in a.slabs:
        u, v, filters, bp = pair(f)
        slabs.append((torch.from_numpy(u).to(held), torch.from_numpy(v).to(held), ">" in f))
        print("slab %s: %s, %s; %s; sd of one %.4f, the two differ by %.4f rms" % (f, tuple(u.shape), filters, bp, float(u.std()), float((u - v).std())), flush=True)
    watch = None
    if a.watch:
        u, v, _, _ = pair(a.watch)
        watch = tuple(torch.from_numpy(x[:, 384:640, 384:640].copy()).to(dev) for x in (u, v))
    net = UNet3d(a.width, a.scales).to(dev).to(memory_format=torch.channels_last_3d)
    net.amp = a.amp
    if a.init:   # start from a trained network of the same size (another scan's)
        net.load_state_dict(load_net(a.init, dev).state_dict())
        net.train()
        print("starting from %s" % a.init, flush=True)
    print("network: %d parameters" % sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp == "fp16")   # (bf16 has the range of single precision: nothing to scale)
    pz, pp, bs = a.patch_z, a.patch, a.batch
    budget = a.minutes * 60.0
    t0, step, run, skipped = time.time(), 0, None, 0
    rng = np.random.default_rng(0)
    while True:
        frac = (time.time() - t0) / budget
        if frac >= 1:
            break
        for g in opt.param_groups:   # (a warm-up over the first steps, then a cosine down to 2 %)
            g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * frac)) * 0.98 + 0.02) * min(1.0, (step + 1) / a.warmup)
        xs, ys = [], []
        for _ in range(bs):
            e, o, directed = slabs[rng.integers(len(slabs))]
            z0, y0, x0 = rng.integers(e.shape[0] - pz + 1), rng.integers(e.shape[1] - pp + 1), rng.integers(e.shape[2] - pp + 1)
            x, y = e[z0: z0 + pz, y0: y0 + pp, x0: x0 + pp], o[z0: z0 + pz, y0: y0 + pp, x0: x0 + pp]
            if not directed and rng.random() < 0.5:
                x, y = y, x
            flips = [d for d in range(3) if rng.random() < 0.5]
            if flips:
                x, y = torch.flip(x, flips), torch.flip(y, flips)
            if rng.random() < 0.5:
                x, y = x.transpose(1, 2), y.transpose(1, 2)
            xs.append(x)
            ys.append(y)
        x = (torch.stack(xs).to(dev)[:, None] * SCALE).contiguous(memory_format=torch.channels_last_3d)
        y = torch.stack(ys).to(dev)[:, None] * SCALE
        with torch.autocast("cuda", dtype=AMP[a.amp]):
            loss = F.mse_loss(net(x).float(), y)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
            scaler.step(opt)
            scaler.update()
        else:   # half precision overflowed: the step in single precision instead, counted
            skipped += 1
            loss = F.mse_loss(net(x), y)
            if not torch.isfinite(loss):
                raise SystemExit("step %d: the loss is not finite in single precision either (the network's weights: %s): stopped" %
                                 (step, "finite" if all(torch.isfinite(q).all() for q in net.parameters()) else "not finite"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
            opt.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        step += 1
        if step % a.report == 0:
            line = "step %6d, %4.1f min: loss %.5f (the input against the target: %.5f in this batch; %d steps in single precision)" % (step, (time.time() - t0) / 60, run if run is not None else float("nan"), F.mse_loss(x, y).item(), skipped)
            if watch is not None:
                net.eval()
                fe, fo = run_tiled(net, watch[0]), run_tiled(net, watch[1])
                net.train()
                before = ((watch[0] - watch[1]) ** 2).mean().item()
                after = (((fe - watch[1]) ** 2).mean().item() + ((fo - watch[0]) ** 2).mean().item()) / 2
                # (the two of equal noise: each one's variance is before / 2) what is left of a filtered one's error
                # against the sample, over its noise:
                line += "; watched block: error of a filtered one %.3f of its noise" % ((after - before / 2) / (before / 2))
            print(line, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=net.state_dict(), width=a.width, scales=a.scales, amp=a.amp, slabs=a.slabs, steps=step, minutes=a.minutes, patch=(pz, pp, pp), batch=bs), out / "model.pt")
    print("%d steps of %d blocks of %d x %d x %d in %.1f min; written to %s" % (step, bs, pz, pp, pp, (time.time() - t0) / 60, out / "model.pt"))


def apply(a):
    dev = torch.device("cuda")
    net = load_net(a.model, dev)
    z = np.load(a.block)
    t = time.time()
    keys = [k for k in ("even", "odd", "full") if k in z.files]     # (a block may hold `full` alone)
    res = {k: run_tiled(net, torch.from_numpy(z[k]).to(dev)).cpu().numpy() for k in keys}
    dt = time.time() - t
    out = dict(heights=z["heights"], roi=z["roi"])
    if "full" in res:
        out["full_direct"] = res["full"]
    if "even" in res and "odd" in res:
        out.update(even=res["even"], odd=res["odd"], full=(res["even"] + res["odd"]) / 2)
    np.savez(a.out, **out)
    print("%s through %s: %d volumes of %s in %.1f s; written to %s" % (a.block, a.model, len(keys), z[keys[0]].shape, dt, a.out))
    if "full" in out and "full_direct" in out:
        print("the mean of the two filtered halves against the network on the full reconstruction: %.5f rms (the full's sd %.4f)" %
              (float((out["full"] - out["full_direct"]).std()), float(z["full"].std())))


def slices(a):
    """the network on `full` of a slab, some of its slices written as TIFF (float32, and uint8 in the window given)"""
    import tifffile
    dev = torch.device("cuda")
    net = load_net(a.model, dev)
    z = np.load(a.block)
    v = torch.from_numpy(z[a.key])
    t = time.time()
    res = run_tiled(net, v).numpy()
    print("%s (%s) through %s: %s in %.0f s" % (a.block, a.key, a.model, tuple(v.shape), time.time() - t), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    h0 = int(z["heights"][0])
    lo, hi = a.window
    for i in a.slices:
        name = "h%+05d_%s" % (h0 + i, a.name)
        tifffile.imwrite(out / (name + ".tif"), res[i])
        tifffile.imwrite(out / (name + "_uint8.tif"), np.clip((res[i] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8))
        print("height %d: %s.tif; before sd %.4f, after %.4f; changed by %.4f rms" % (h0 + i, out / name, float(z[a.key][i].std()), float(res[i].std()), float((res[i] - z[a.key][i]).std())), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--slabs", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--watch", default="")
    t.add_argument("--minutes", type=float, default=10.0)
    t.add_argument("--width", type=int, default=32)
    t.add_argument("--scales", type=int, default=3)
    t.add_argument("--host", action="store_true", help="keep the slabs in the host's memory (more slabs than the device holds)")
    t.add_argument("--patch", type=int, default=96)
    t.add_argument("--patch_z", type=int, default=32)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--report", type=int, default=500)
    t.add_argument("--warmup", type=int, default=500, help="steps over which the learning rate rises to --lr")
    t.add_argument("--clip", type=float, default=1.0, help="the gradient's norm is cut to this")
    t.add_argument("--amp", default="fp16", choices=sorted(AMP), help="half precision of the convolutions")
    t.add_argument("--init", default="", help="a model.pt of the same --width and --scales to start from")
    p = sub.add_parser("apply")
    p.add_argument("--model", required=True)
    p.add_argument("--block", required=True)
    p.add_argument("--out", required=True)
    q = sub.add_parser("slices")
    q.add_argument("--model", required=True)
    q.add_argument("--block", required=True)
    q.add_argument("--key", default="full")
    q.add_argument("--out", required=True)
    q.add_argument("--name", required=True)
    q.add_argument("--slices", type=int, nargs="+", required=True, help="places in the slab (0: its first height)")
    q.add_argument("--window", type=float, nargs=2, required=True, help="the uint8 files' window (per cm)")
    a = ap.parse_args()
    return {"train": train, "apply": apply, "slices": slices}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
