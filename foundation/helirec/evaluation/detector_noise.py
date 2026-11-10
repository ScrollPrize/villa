"""Detector noise of a scan from its own flats and darks: what a noise-aware (Wiener-type) projection filter and the
interpretation of shell correlations need.

    python evaluation/detector_noise.py --projections PROJ --out OUT [--frames 120] [--rows 300 900] [--cols 700 1724]

Per scan of the pair, from the first --frames frames of a flat series and of the darks, in a detector region:

  photon transfer        dark offset and temporal noise; flat signal; temporal variance / signal (ADU per quantum-
                         equivalent) and the signal-to-noise ratio of one frame
  noise power spectrum   of the temporal noise of the flats (each frame scaled to the series' mean intensity, the
                         per-pixel temporal mean removed), on Hann-windowed tiles: radial profile in bands of 1 / 32
                         cycles per pixel relative to the lowest band, the same along x and along y, and the spatial
                         autocorrelation at small lags. A detector blur before the noise is added colours it; noise
                         added after the blur (the white floor) does not: the spectrum says how much of each there is.
  temporal lag           autocorrelation of the noise between frames 1 .. 5 apart (afterglow), with the bias of the
                         estimator (-1 / (n - 1)) for comparison
  flat stability         what the flat-field division itself adds, which is the same in every projection (and so
                         traces helices through the volume instead of averaging out): the noise left in a series'
                         mean flat (temporal noise / sqrt(frames in the series)), and the change of the fixed pattern
                         between the first and the last series (their ratio, less its smooth part: standard deviation,
                         against what the noise of the two means explains)

Output: OUT/detector_noise.json and a printed summary. The radial noise spectrum ("nps": frequency, relative) is the
table a noise-aware filter takes.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from helirec import _core, bliss


def tiles(a, t):
    """(n, H, W) -> (n * tiles, t, t) half-overlapping tiles"""
    out = [a[:, y:y + t, x:x + t] for y in range(0, a.shape[1] - t + 1, t // 2) for x in range(0, a.shape[2] - t + 1, t // 2)]
    return np.concatenate(out)


def power_spectrum(res, t=128):
    """mean |FFT|^2 of Hann-windowed tiles, normalised to variance per frequency (white noise of variance v: v)"""
    w = np.hanning(t + 1)[:-1]
    w2 = w[:, None] * w[None, :]
    p = np.zeros((t, t))
    count = 0
    for k in range(0, len(res), 16):
        blk = tiles(res[k:k + 16], t)
        p += (np.abs(np.fft.fft2(blk * w2)) ** 2).sum(0)
        count += len(blk)
    return p / (count * (w2 ** 2).sum())


def radial(p, nb=16):
    t = p.shape[0]
    f = np.fft.fftfreq(t)
    fr = np.hypot(f[:, None], f[None, :])
    idx = np.minimum((fr * 2 * nb).astype(int), nb)
    prof = np.bincount(idx.ravel(), p.ravel(), nb + 1)[:nb] / np.bincount(idx.ravel(), minlength=nb + 1)[:nb]
    # along x (|fy| small) and along y (|fx| small): directional profiles
    ax = [np.array([p[np.abs(f) < 1.5 / t][:, (np.abs(f) >= b / (2 * nb)) & (np.abs(f) < (b + 1) / (2 * nb))].mean() for b in range(nb)]),
          np.array([p[(np.abs(f) >= b / (2 * nb)) & (np.abs(f) < (b + 1) / (2 * nb))][:, np.abs(f) < 1.5 / t].mean() for b in range(nb)])]
    return (np.arange(nb) + 0.5) / (2 * nb), prof, ax[0], ax[1]


def smooth(a, sigma):
    return _core.gaussian_filter1d(_core.gaussian_filter1d(np.ascontiguousarray(a, np.float64), 0, sigma, "reflect"), 1, sigma, "reflect")


def measure(scan, a):
    rows, cols = slice(*a.rows), slice(*a.cols)
    n = min(a.frames, len(scan.flats[0]), len(scan.darks))
    dark = scan.read_frames(scan.darks, slice(0, n), rows, cols).astype(np.float64)
    flat = scan.read_frames(scan.flats[0], slice(0, n), rows, cols).astype(np.float64)
    out = dict(scan=scan.name, frames=n, region=dict(rows=list(a.rows), cols=list(a.cols)))
    dmean = dark.mean(0)
    out["dark"] = dict(offset=float(dmean.mean()), temporal_std=float(np.sqrt(dark.var(0, ddof=1).mean())),
                       fixed_pattern_std=float((dmean - smooth(dmean, 8)).std()))
    sig = flat - dmean
    g = sig.mean((1, 2))
    scaled = sig * (g.mean() / g)[:, None, None]          # frame-to-frame intensity changes are not detector noise
    mean = scaled.mean(0)
    res = (scaled - mean) * np.sqrt(n / (n - 1))
    var = res.var(0).mean()
    out["flat"] = dict(signal=float(mean.mean()), signal_range=[float(mean.min()), float(mean.max())],
                       temporal_std=float(np.sqrt(var)), variance_over_signal=float(var / mean.mean()),
                       snr_per_frame=float(mean.mean() / np.sqrt(var)),
                       intensity_fluctuation_rms=float(g.std() / g.mean()))
    f, prof, px, py = radial(power_spectrum(res))
    out["nps"] = dict(frequency=f.tolist(), relative=(prof / prof[0]).tolist(), along_x=(px / px[0]).tolist(),
                      along_y=(py / py[0]).tolist(), absolute=prof.tolist(),
                      white_floor=float(prof[-3:].mean() / prof[0]))
    ac = lambda dy, dx: float((res[:, dy:, dx:] * res[:, :res.shape[1] - dy, :res.shape[2] - dx]).mean() / var)
    out["autocorrelation"] = {"0,1": ac(0, 1), "1,0": ac(1, 0), "1,1": ac(1, 1), "0,2": ac(0, 2), "2,0": ac(2, 0)}
    out["temporal_lag"] = dict(lags=[float((res[k:] * res[:-k]).mean() / var) for k in range(1, 6)],
                               estimator_bias=-1.0 / (n - 1))
    # flat stability: the fixed pattern of the first and of the last series
    first, last = scan.flats[0], scan.flats[-1]
    m = min(a.frames, len(first), len(last))
    fa = scan.read_frames(first, slice(0, m), rows, cols).astype(np.float64).mean(0) - dmean
    fb = scan.read_frames(last, slice(0, m), rows, cols).astype(np.float64).mean(0) - dmean
    ratio = fb / fa
    high = ratio / smooth(ratio, a.pattern_sigma) - 1
    rel = np.sqrt(var) / mean.mean()
    out["flat_stability"] = dict(
        series=[len(s) for s in scan.flats], frames_compared=m,
        mean_flat_noise_relative=float(rel / np.sqrt(len(first))),
        projection_noise_relative=float(rel),
        pattern_change_std=float(high.std()), pattern_change_expected_from_noise=float(rel * np.sqrt(2.0 / m)),
        pattern_sigma=a.pattern_sigma,
        fixed_pattern_std=float((fa / smooth(fa, a.pattern_sigma) - 1).std()))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--projections", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--rows", nargs=2, type=int, default=[300, 900])
    p.add_argument("--cols", nargs=2, type=int, default=[700, 1724])
    p.add_argument("--pattern_sigma", type=float, default=8.0, help="pixels: scale below which the flat pattern counts as fixed pattern")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pair = bliss.pair_companions(bliss.discover_scans(a.projections))
    results = [measure(s, a) for s in (pair["main"], pair["companion"])]
    with open(os.path.join(a.out, "detector_noise.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    np.set_printoptions(linewidth=200, precision=3, suppress=True)
    for r in results:
        print(f"\n{r['scan']}  ({r['frames']} frames, rows {r['region']['rows']}, cols {r['region']['cols']})")
        d, f = r["dark"], r["flat"]
        print(f"  dark {d['offset']:.1f} ADU, temporal std {d['temporal_std']:.2f}, fixed pattern {d['fixed_pattern_std']:.2f}")
        print(f"  flat {f['signal']:.0f} ADU ({f['signal_range'][0]:.0f}..{f['signal_range'][1]:.0f}), temporal std "
              f"{f['temporal_std']:.1f}, variance / signal {f['variance_over_signal']:.3f}, SNR per frame "
              f"{f['snr_per_frame']:.1f}, frame-to-frame intensity {100 * f['intensity_fluctuation_rms']:.2f} % rms")
        print("  noise power spectrum, relative to the lowest band (bands of 1/32 cycles/px):")
        for k in ("relative", "along_x", "along_y"):
            print(f"    {k:9s}", np.array(r["nps"][k]))
        print(f"    white floor (last three bands) {r['nps']['white_floor']:.3f}")
        print("  spatial autocorrelation", {k: round(v, 3) for k, v in r["autocorrelation"].items()})
        print("  temporal lag 1..5", np.array(r["temporal_lag"]["lags"]), "estimator bias", round(r["temporal_lag"]["estimator_bias"], 4))
        s = r["flat_stability"]
        print(f"  flats: series {s['series']}; noise of one projection {100 * s['projection_noise_relative']:.3f} %, of a "
              f"series' mean flat {100 * s['mean_flat_noise_relative']:.4f} %")
        print(f"  fixed pattern below {s['pattern_sigma']:.0f} px: {100 * s['fixed_pattern_std']:.3f} % std; its change first -> "
              f"last series: {100 * s['pattern_change_std']:.4f} % std (noise of the two {s['frames_compared']}-frame means "
              f"explains {100 * s['pattern_change_expected_from_noise']:.4f} %)")
    print(f"\nwrote {os.path.join(a.out, 'detector_noise.json')}")


if __name__ == "__main__":
    main()
