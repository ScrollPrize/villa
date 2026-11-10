"""3D Fourier shell correlation of reconstructed volumes: against a registered reference volume of the same sample
(e.g. a finer scan), and / or between two reconstructions of one scan (the two halves of a projection split).

    python evaluation/shell_correlation.py --ref evaluation/343p_reference_2p215um.json \
        --volume base=RUN/base/volume.zarr --volume unsharp=RUN/unsharp/volume.zarr --out OUT
    python evaluation/shell_correlation.py --volume even=A/volume.zarr --volume odd=B/volume.zarr --pair even odd --out OUT

Volumes are OME-Zarr outputs of helirec-slices (float32 or uint8; their place in the scan's full volume is read from
their metadata) or any Zarr of the full volume (NAME=LOCATION@Z0,Y0,X0 gives an origin; remote ones use --cache).
Everything is computed on cubic blocks (--block, a lattice of step --step) of the region the volumes share, with
helirec.BlockCorrelator (C++ / CUDA): sums of |A|^2, |B|^2, Re(A conj B) in bins of |k| (shells) and of the angle of k
from the z axis. Reported per shell, for every direction ("shell"), within 30 degrees of the xy plane ("inplane")
and within 30 degrees of the z axis ("axial"):

  correlation      ab / sqrt(aa bb)
  matched          the correlation of the best linear estimate of the reference that weighs every (shell, direction)
                   bin on its own: no linear filter of the volume changes it, isotropic or not (the plain shell
                   correlation rises when a filter attenuates the directions of poorer signal-to-noise ratio, as a
                   blur along z does): the number to compare reconstructions with
  snr              the signal-to-noise power ratio it implies (reference: c^2 / (1 - c^2), a lower bound since the
                   reference has noise of its own; pair: 2 c / (1 - c) for the two halves together)
  filter           ab / aa: the least-squares linear filter from the volume to the reference
  power            the volume's own mean power per frequency

Reference (--ref JSON: location, affine_xyz from the volumes' grid to the reference's level 0, as the open-data
catalogue's volume_transforms): the reference is read at pyramid level --level, resampled --oversample times finer than
the volumes' grid and band-limited to it in the Fourier domain, then registered per block by a sub-voxel shift found
with the volume --register (default: the first) and used for all, so that every volume is compared under the same
registration. Blocks count where the reference is not masked (--coverage). Groups: "all", and "dense" / "sparse" = the
blocks of the upper / lower third of the reference's block means.

With --baseline NAME every other volume is also compared with it, per shell: power gain (its power over the baseline's),
signal gain squared ((ab / ab_baseline)^2: the gain on what the reference confirms) and their ratio, the unconfirmed
power index: 1 for any linear filter of the baseline, < 1 when noise was removed, > 1 when power was added that the
reference does not confirm.

With --ref and --pair A B (two halves of one scan) the three correlations give, per shell and without assuming
anything about the reference, each volume's correlation with the common signal: t_ref = sqrt(c_Ar c_Br / c_AB), t_half
= sqrt(c_AB c_Ar / c_Br), valid when the three noises are independent.

Output (OUT): curves.json (every curve, per group), blocks.npz (per-block tables, shifts, means: regroup at will),
and a printed summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import helirec

PARTS = {"shell": (0.0, 90.0), "inplane": (60.0, 90.0), "axial": (0.0, 30.0)}


def affine_zyx(m_xyz):
    """catalogue matrix ((x, y, z) of the target from (x, y, z, 1) of the source) -> the same in (z, y, x) order"""
    m = np.asarray(m_xyz, np.float64)
    if m.shape != (3, 4):
        raise ValueError("affine_xyz must be 3 x 4")
    out = np.empty((3, 4))
    out[:, :3] = m[::-1, 2::-1]
    out[:, 3] = m[::-1, 3]
    return np.ascontiguousarray(out)


class Volume:
    """a (z, y, x) Zarr array placed in the scan's full volume"""

    def __init__(self, spec, cache):
        name, _, loc = spec.partition("=")
        if not loc:
            sys.exit(f"--volume takes NAME=LOCATION[@Z0,Y0,X0], not '{spec}'")
        loc, _, origin = loc.partition("@")
        self.name, self.cache = name, cache or ""
        level = helirec.zarr_levels(loc, self.cache)[0]
        self.location, self.shape, self.dtype = level["location"], tuple(level["shape"]), level["dtype"]
        if origin:
            self.origin = tuple(int(v) for v in origin.split(","))
        else:
            self.origin = (0, 0, 0)
            attrs = os.path.join(loc, ".zattrs")
            if os.path.exists(attrs):   # helirec's writer: translation = origin * voxel size
                ct = json.load(open(attrs))["multiscales"][0]["datasets"][0]["coordinateTransformations"]
                scale = next(c["scale"] for c in ct if c["type"] == "scale")
                tr = next((c["translation"] for c in ct if c["type"] == "translation"), [0, 0, 0])
                self.origin = tuple(int(round(t / s)) for t, s in zip(tr, scale))
        self.data = None

    def bounds(self):
        return [(o, o + s) for o, s in zip(self.origin, self.shape)]

    def load(self, region):
        start = [lo - o for (lo, _), o in zip(region, self.origin)]
        shape = [hi - lo for lo, hi in region]
        self.data = np.ascontiguousarray(helirec.zarr_read(self.location, start, shape, self.cache), np.float32)
        self.region = region

    def block(self, z, y, x, n):
        (z0, _), (y0, _), (x0, _) = self.region
        return np.ascontiguousarray(self.data[z - z0:z - z0 + n, y - y0:y - y0 + n, x - x0:x - x0 + n])


def curves_of(corr, t):
    out = {}
    for part, (lo, hi) in PARTS.items():
        c = corr.curves(t[0], t[1], t[2], lo, hi)
        out[part] = {k: [None if not np.isfinite(x) else float(x) for x in v] for k, v in c.items()}
    return out


def fmt(v, digits=3):
    return " ".join("  nan " if x is None or not np.isfinite(x) else f"{x:6.{digits}f}" for x in v)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--volume", action="append", default=[], metavar="NAME=LOCATION", required=True)
    p.add_argument("--ref", help="reference JSON (location, affine_xyz[, cache_dir, level])")
    p.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"),
                   help="correlate two of the volumes with each other (repeatable)")
    p.add_argument("--out", required=True)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--step", type=int, default=None, help="block lattice step (default: half a block)")
    p.add_argument("--shells", type=int, default=16)
    p.add_argument("--thetas", type=int, default=6)
    p.add_argument("--oversample", type=int, default=2, help="reference samples per voxel of the volumes")
    p.add_argument("--level", type=int, default=None, help="reference pyramid level (default: the JSON's, else 1)")
    p.add_argument("--interp", default="lanczos3", choices=["linear", "cubic", "lanczos3"])
    p.add_argument("--coverage", type=float, default=0.98, help="least unmasked fraction of the reference in a block")
    p.add_argument("--region", nargs=6, type=int, metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
                   help="full-volume region (default: what the volumes share)")
    p.add_argument("--content_shift", nargs=3, type=float, default=[0, 0, 0], metavar=("DZ", "DY", "DX"),
                   help="the volumes' content at grid voxel p is the content of p + this on the grid the affine was "
                        "made for (e.g. a reconstruction with another helical pitch moves along z)")
    p.add_argument("--register", default=None, help="volume the per-block shifts are found with (default: the first)")
    p.add_argument("--no_align", action="store_true", help="catalogue affine only: no per-block shift")
    p.add_argument("--align_band", type=float, default=0.25)
    p.add_argument("--max_shift", type=int, default=4)
    p.add_argument("--baseline", default=None, help="volume the others are compared with (unconfirmed power index)")
    p.add_argument("--cache", default=None, help="disk cache for remote Zarr (default: the reference JSON's cache_dir)")
    p.add_argument("--device", default="cuda" if helirec.cuda_available() else "cpu", choices=["cuda", "cpu"])
    a = p.parse_args(argv)

    ref_cfg = json.load(open(a.ref)) if a.ref else None
    cache = a.cache or (ref_cfg or {}).get("cache_dir") or ""
    vols = [Volume(s, cache) for s in a.volume]
    names = [v.name for v in vols]
    if len(set(names)) != len(names):
        sys.exit("volume names must be unique")
    by_name = dict(zip(names, vols))
    for n in [x for pr in a.pair for x in pr] + [x for x in (a.register, a.baseline) if x]:
        if n not in by_name:
            sys.exit(f"unknown volume '{n}' (volumes: {', '.join(names)})")
    if not ref_cfg and not a.pair:
        sys.exit("nothing to do: give --ref and / or --pair")
    region = [(max(v.bounds()[d][0] for v in vols), min(v.bounds()[d][1] for v in vols)) for d in range(3)]
    if a.region:
        region = [(max(lo, a.region[2 * d]), min(hi, a.region[2 * d + 1])) for d, (lo, hi) in enumerate(region)]
    n, step = a.block, a.step or a.block // 2
    if any(hi - lo < n for lo, hi in region):
        sys.exit(f"the volumes share only {region}: smaller than a block of {n}")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for v in vols:
        v.load(region)
    print(f"{len(vols)} volumes, region z {region[0]} y {region[1]} x {region[2]}, loaded in {time.time() - t0:.1f}s", flush=True)

    lattice = [(z, y, x) for z in range(region[0][0], region[0][1] - n + 1, step)
               for y in range(region[1][0], region[1][1] - n + 1, step)
               for x in range(region[2][0], region[2][1] - n + 1, step)]
    ref = None
    if ref_cfg:
        level = a.level if a.level is not None else int(ref_cfg.get("level", 1))
        M = affine_zyx(ref_cfg["affine_xyz"])
        M[:, 3] += M[:, :3] @ np.asarray(a.content_shift, np.float64)
        ref = helirec.RegisteredVolume(ref_cfg["location"], level, M, cache)
        # the reference's mask, from a coarse level: which blocks lie inside it
        levels = helirec.zarr_levels(ref_cfg["location"], cache)
        scale0 = float(np.cbrt(abs(np.linalg.det(M[:, :3]))))   # reference level-0 voxels per voxel of the grid
        coarse = max((l["level"] for l in levels if l["scale"][0] <= 4 * scale0), default=level)
        ms = 4
        mshape = [-(-(hi - lo) // ms) for lo, hi in region]
        mask, _ = helirec.RegisteredVolume(ref_cfg["location"], coarse, M, cache).read(
            [float(lo) + (ms - 1) / 2 for lo, _ in region], float(ms), mshape, "linear", "cpu")
        mask = mask > 0

        def covered(z, y, x):
            i = [(c - lo) // ms for c, (lo, _) in zip((z, y, x), region)]
            m = mask[i[0]:i[0] + n // ms, i[1]:i[1] + n // ms, i[2]:i[2] + n // ms]
            return float(m.mean()) if m.size else 0.0

        lattice = [b for b in lattice if covered(*b) >= a.coverage]
        print(f"reference {ref_cfg['location']} level {level} ({ref.level_shape}), mask from level {coarse}: "
              f"{len(lattice)} blocks of {n} inside it", flush=True)
    if not lattice:
        sys.exit("no block to measure")

    spec = dict(shells=a.shells, thetas=a.thetas, device=a.device)
    corr_ref = helirec.BlockCorrelator(n, oversample=a.oversample, align=not a.no_align, align_band=a.align_band,
                                       max_shift=a.max_shift, **spec) if ref else None
    corr_pair = helirec.BlockCorrelator(n, oversample=1, align=False, **spec) if a.pair else None
    any_corr = corr_ref or corr_pair
    register = a.register or names[0]
    order = [register] + [x for x in names if x != register]
    shape = (len(lattice), any_corr.n_shells, any_corr.thetas)
    tables = {f"ref/{x}": np.zeros((3,) + shape) for x in names} if ref else {}
    tables.update({f"pair/{x}/{y}": np.zeros((3,) + shape) for x, y in a.pair})
    info = {k: np.zeros(len(lattice)) for k in ("coverage", "peak", "ref_mean", "ref_var")}
    info["shift"] = np.zeros((len(lattice), 3))
    info.update({f"mean/{x}": np.zeros(len(lattice)) for x in names})
    info.update({f"var/{x}": np.zeros(len(lattice)) for x in names})
    t0 = time.time()
    m = n * a.oversample
    for i, (z, y, x) in enumerate(lattice):
        blocks = {v.name: v.block(z, y, x, n) for v in vols}
        if ref:
            rb, info["coverage"][i] = ref.read([z, y, x], 1.0 / a.oversample, [m, m, m], a.interp, a.device)
            shift = None
            for name in order:
                r = corr_ref.compute(blocks[name], rb, shift)
                if shift is None:
                    shift = r["shift"]
                    info["shift"][i], info["peak"][i] = shift, r["peak"]
                    info["ref_mean"][i], info["ref_var"][i] = r["mean_b"], r["var_b"]
                tables[f"ref/{name}"][:, i] = r["aa"], r["bb"], r["ab"]
                info[f"mean/{name}"][i], info[f"var/{name}"][i] = r["mean_a"], r["var_a"]
        for xa, xb in a.pair:
            r = corr_pair.compute(blocks[xa], blocks[xb])
            tables[f"pair/{xa}/{xb}"][:, i] = r["aa"], r["bb"], r["ab"]
            for nm, k in ((xa, "a"), (xb, "b")):
                info[f"mean/{nm}"][i], info[f"var/{nm}"][i] = r["mean_" + k], r["var_" + k]
        if (i + 1) % 50 == 0 or i + 1 == len(lattice):
            print(f"  block {i + 1}/{len(lattice)}  {time.time() - t0:.0f}s", flush=True)

    # groups of blocks
    groups = {"all": np.ones(len(lattice), bool)}
    if ref:
        lo3, hi3 = np.percentile(info["ref_mean"], [100 / 3, 200 / 3])
        groups["dense"] = info["ref_mean"] >= hi3
        groups["sparse"] = info["ref_mean"] <= lo3
    result = dict(region=region, block=n, step=step, content_shift=list(a.content_shift), shells=a.shells, thetas=a.thetas, oversample=a.oversample,
                  n_blocks={g: int(s.sum()) for g, s in groups.items()}, volumes={v.name: v.location for v in vols},
                  reference=ref_cfg, register=register if ref else None, curves={}, comparison={}, triangulation={})
    for key, t in tables.items():
        result["curves"][key] = {g: curves_of(any_corr, t[:, sel].sum(1)) for g, sel in groups.items()}
    freq = np.array(result["curves"][next(iter(tables))]["all"]["shell"]["frequency"])
    use = freq < 0.5
    np.savez(os.path.join(a.out, "blocks.npz"), lattice=np.array(lattice), counts=any_corr.counts, frequency=freq,
             **{k.replace("/", "__"): v for k, v in {**tables, **info}.items()})

    print("\nshell centres (cycles/voxel): " + fmt(freq[use]))
    if ref:
        s = info["shift"]
        print(f"registration with '{register}': shift (z, y, x) mean {s.mean(0).round(3)} std {s.std(0).round(3)} "
              f"voxels, whitened correlation below {a.align_band} cycles/voxel: median {np.median(info['peak']):.3f}")
        for g in groups:
            print(f"\n== against the reference, {g} ({result['n_blocks'][g]} blocks)")
            for name in names:
                c = result["curves"][f"ref/{name}"][g]
                print(f"  {name}")
                for part in PARTS:
                    print(f"    correlation {part:8s} " + fmt(np.array(c[part]["correlation"], float)[use]))
                print("    matched     shell    " + fmt(np.array(c["shell"]["matched"], float)[use]))
                print("    snr (lower bound)    " + fmt(np.array(c["shell"]["snr_reference"], float)[use], 2))
                f = np.array(c["shell"]["filter"], float)[use]
                print("    filter / its shell 1 " + fmt(f / f[1]))
                pw = np.array(c["shell"]["power_a"], float)[use]
                print("    power / its shell 1  " + fmt(pw / pw[1], 4))
        if a.baseline:
            for g, sel in groups.items():
                base = tables[f"ref/{a.baseline}"][:, sel].sum(1)
                for name in names:
                    if name == a.baseline:
                        continue
                    t = tables[f"ref/{name}"][:, sel].sum(1)
                    comp = {}
                    for part, (lo, hi) in PARTS.items():
                        cb = any_corr.curves(base[0], base[1], base[2], lo, hi)
                        ct = any_corr.curves(t[0], t[1], t[2], lo, hi)
                        power = ct["power_a"] / cb["power_a"]
                        signal = (ct["filter"] * ct["power_a"] / (cb["filter"] * cb["power_a"])) ** 2
                        comp[part] = dict(power_gain=power.tolist(), signal_gain2=signal.tolist(),
                                          unconfirmed_power=(power / signal).tolist())
                    result["comparison"].setdefault(name, {})[g] = comp
            print(f"\n== against the baseline '{a.baseline}' (all blocks, every direction)")
            for name, comp in result["comparison"].items():
                c = comp["all"]["shell"]
                print(f"  {name}\n    power gain           " + fmt(np.array(c["power_gain"])[use]))
                print("    signal gain^2        " + fmt(np.array(c["signal_gain2"])[use]))
                print("    unconfirmed power    " + fmt(np.array(c["unconfirmed_power"])[use]))
    for xa, xb in a.pair:
        key = f"pair/{xa}/{xb}"
        print(f"\n== {xa} against {xb}")
        for g in groups:
            c = result["curves"][key][g]
            for part in PARTS:
                print(f"  {g:6s} correlation {part:8s} " + fmt(np.array(c[part]["correlation"], float)[use]))
        if ref:   # the three correlations -> each volume's correlation with the common signal
            for g, sel in groups.items():
                tri = {}
                for part, (lo, hi) in PARTS.items():
                    def corr_of(k):
                        t = tables[k][:, sel].sum(1)
                        return np.clip(any_corr.curves(t[0], t[1], t[2], lo, hi)["correlation"], 1e-12, None)
                    cab, car, cbr = corr_of(key), corr_of(f"ref/{xa}"), corr_of(f"ref/{xb}")
                    tri[part] = dict(reference=np.sqrt(car * cbr / cab).tolist(), half_a=np.sqrt(cab * car / cbr).tolist(),
                                     half_b=np.sqrt(cab * cbr / car).tolist(),
                                     whole_scan=np.sqrt(2 * cab / (1 + cab)).tolist())
                result["triangulation"].setdefault(key, {})[g] = tri
            t = result["triangulation"][key]["all"]["shell"]
            print("  correlation with the common signal (all blocks, every direction):")
            print("    reference            " + fmt(np.array(t["reference"])[use]))
            print(f"    {xa:20s} " + fmt(np.array(t["half_a"])[use]))
            print(f"    {xb:20s} " + fmt(np.array(t["half_b"])[use]))
            print("    both halves together " + fmt(np.array(t["whole_scan"])[use]))
    with open(os.path.join(a.out, "curves.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nwrote {os.path.join(a.out, 'curves.json')} and blocks.npz in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
