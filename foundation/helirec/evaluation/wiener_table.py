"""The table of the `wiener` projection filter (filters.hpp) from measurements of a scan.

    python evaluation/wiener_table.py OUT.json --transfer FILM.json [--subset radial] [--sharp NAME COL0 COL1]
        --curves RUN --pair half0 half1 [--filtered base] [--z_interp lanczos3]

The transfer T(nu, theta) of an UNFILTERED reconstruction (detector blur x phase contrast x what the gridding does):
  --transfer   its in-plane transfer G(nu), JSON with "frequency" and SUBSET: {"G", "valid"}, e.g. from a thin film
               of the sample (stage notes of "Fast MBIR for scroll tomography": film_transfer_table.py). A smooth
               model is fitted to it, G = (1 + A (2 pi nu)^2) (a exp(-2 pi^2 s1^2 nu^2) + (1 - a) exp(-2 pi^2 s2^2
               nu^2)), which carries it to nu = 0 and beyond the last measured frequency.
  --sharp      the same measured where the detector is sharpest: the bins of "by_column" / NAME of that file whose
               one column lies in [COL0, COL1]. In-plane the reconstruction is softer than that by S(nu) = G /
               G_sharp (a smear along the detector's columns; fitted as sinc(s nu)); along z it is not. With the
               detector's own blur the same in every direction (the noise of the flats says so where it is sharp):
                   T(f_u, f_z) = G(nu) / S(nu) x S(|f_u|),   f_u = nu sin theta, f_z = nu cos theta.
               Without --sharp: S = 1. --row_response restore adds the loss L(|f_z|) of the interpolation between
               detector rows (--z_interp) to T, so that the filter undoes it. Do not: on PHerc0343P that left a
               ridge of power at the slices' Nyquist frequency (alternating slices).
The signal-to-noise power ratio S(nu, theta):
  --curves     an output folder of evaluation/shell_correlation.py (blocks.npz) with
               --pair A B     the two half reconstructions of the scan (angle_subset=0/2, 1/2): 2 c / (1 - c) per
                              shell, table "snr". An upper bound: the halves share every error that all projections
                              share. By default no direction gets more than the in-plane value (--snr_direction):
                              near z the halves agree on much more than a reference confirms.
               --filtered V   a whole-scan volume measured against a reference volume in the same run: table
                              "snr_reference", c^2 / (1 - c^2), a lower bound (the reference's own noise and
                              registration), and "snr_mean", the geometric mean of the two.
               A direction dependence of the transfer is NOT taken from the reference: the ratio of the cross-powers
               mostly shows the residual registration error in each direction.

Output: OUT.json = {"nu", "theta", "transfer", "snr"[, "snr_reference", "snr_mean"], "model", "sources"}; theta in
degrees from the row axis of a frame (the scan's z): `--set "filters=wiener(table=OUT.json)"`.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


def model(p, nu):
    A, a, s1, s2 = p
    k2 = (2 * np.pi * nu) ** 2
    return (1 + A * k2) * (a * np.exp(-0.5 * s1 * s1 * k2) + (1 - a) * np.exp(-0.5 * s2 * s2 * k2))


def fit_transfer(nu, G):
    r = least_squares(lambda p: np.log(model(p, nu)) - np.log(G), [18.0, 0.5, 0.9, 1.9],
                      bounds=([1.0, 0.0, 0.2, 0.2], [100.0, 1.0, 4.0, 6.0]))
    return r.x, float(np.sqrt(np.mean(r.fun ** 2)))


def row_interpolation_response(name, nu, steps=64):
    """mean response of the interpolation between detector rows at a uniformly spread fractional position"""
    if name == "linear":
        taps, kern = np.arange(0, 2), lambda t: np.clip(1 - np.abs(t), 0, None)
    elif name == "cubic":
        taps = np.arange(-1, 3)

        def kern(t):
            t = np.abs(t)
            return np.where(t < 1, (1.5 * t - 2.5) * t * t + 1, np.where(t < 2, ((-0.5 * t + 2.5) * t - 4) * t + 2, 0.0))
    elif name == "lanczos3":
        taps, kern = np.arange(-2, 4), lambda t: np.where(np.abs(t) < 3, np.sinc(t) * np.sinc(t / 3), 0.0)
    else:
        raise SystemExit(f"--z_interp {name}: linear, cubic or lanczos3")
    d = (np.arange(steps) + 0.5) / steps                       # position between row 0 and row 1
    w = kern(taps[None, :] - d[:, None])
    w /= w.sum(1, keepdims=True)
    ph = np.exp(-2j * np.pi * nu[:, None, None] * (taps[None, None, :] - d[None, :, None]))
    return (w[None] * ph).sum(2).mean(1).real


def smooth_rows(t, counts, width=1.0):
    """tables (shells, thetas): a count-weighted Gaussian smoothing along the shells (width in shells)"""
    n = t.shape[0]
    i = np.arange(n)
    w = np.exp(-0.5 * ((i[:, None] - i[None, :]) / width) ** 2)
    return (w @ (t * counts)) / (w @ counts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--transfer", required=True)
    p.add_argument("--subset", default="radial")
    p.add_argument("--sharp", nargs=3, default=None, metavar=("NAME", "COL0", "COL1"))
    p.add_argument("--z_interp", default="lanczos3")
    p.add_argument("--row_response", choices=("keep", "restore"), default="keep",
                   help="the loss of the interpolation between detector rows: left as it is (default), or in the "
                        "transfer, so that the filter undoes it (a response that stays high up to the slices' Nyquist "
                        "frequency: slices alternate)")
    p.add_argument("--snr_direction", choices=("capped", "inplane", "measured"), default="capped",
                   help="capped (default): per direction, but nowhere above the in-plane value (near z the halves "
                        "agree on far more than the reference confirms; between z and the plane the ratio really "
                        "is lower); inplane: the in-plane value in every direction; measured: per direction")
    p.add_argument("--theta_step", type=float, default=7.5)
    p.add_argument("--curves", required=True)
    p.add_argument("--pair", nargs=2, required=True, metavar=("A", "B"))
    p.add_argument("--filtered", default=None)
    p.add_argument("--nu_max", type=float, default=0.75)
    p.add_argument("--step", type=float, default=0.005)
    p.add_argument("--snr_max", type=float, default=1e4)
    a = p.parse_args(argv)
    np.set_printoptions(linewidth=200, precision=3, suppress=True)

    # ---- in-plane transfer: smooth model through the measured points
    film = json.load(open(a.transfer))
    fg = np.array(film["frequency"])
    G = np.array(film[a.subset]["G"])
    ok = np.array(film[a.subset]["valid"], bool) & (G > 0)
    par, rms = fit_transfer(fg[ok], G[ok])
    print(f"transfer '{a.subset}': {ok.sum()} points; model A = {par[0]:.2f} px^2, {par[1]:.2f} of sigma {par[2]:.2f} px + "
          f"{1 - par[1]:.2f} of sigma {par[3]:.2f} px; rms misfit {100 * rms:.1f} %")
    nu = np.arange(1, int(round(a.nu_max / a.step)) + 1) * a.step
    smear = 0.0
    if a.sharp:
        bins = [b for b in film["by_column"][a.sharp[0]]["bins"]
                if len(b["columns"]) == 1 and float(a.sharp[1]) <= b["columns"][0] <= float(a.sharp[2])]
        if not bins:
            raise SystemExit("--sharp: no bin of one column in that range")
        Gs = np.exp(np.mean([np.log(b["G"]) for b in bins], 0))
        oks = ok & np.all([b["valid"] for b in bins], 0) & (fg >= 0.2)
        r = least_squares(lambda q: np.sinc(q[0] * fg[oks]) - G[oks] / Gs[oks], [1.0], bounds=([0.0], [2.0]))
        smear = float(r.x[0])
        print(f"sharp place ({len(bins)} bins, columns {[round(b['columns'][0]) for b in bins]}): G / G_sharp at "
              f"{fg[oks].round(3)} = {(G[oks] / Gs[oks]).round(2)}; as a smear of {smear:.2f} px: {np.sinc(smear * fg[oks]).round(2)}")

    # ---- tables of the correlation run
    z = np.load(os.path.join(a.curves, "blocks.npz"))
    freq, counts = z["frequency"], z["counts"].astype(float)
    nt = counts.shape[1]
    aa, bb, ab = z[f"pair__{a.pair[0]}__{a.pair[1]}"].sum(1)
    c = smooth_rows(ab, counts) / np.sqrt(smooth_rows(aa, counts) * smooth_rows(bb, counts))
    snr = {"snr": 2 * np.clip(c, 0, None) / np.clip(1 - c, 1 / a.snr_max, None)}
    if a.filtered:
        raa, rbb, rab = z[f"ref__{a.filtered}"].sum(1)
        cr = np.clip(smooth_rows(rab, counts) / np.sqrt(smooth_rows(raa, counts) * smooth_rows(rbb, counts)), 0, 0.99995)
        snr["snr_reference"] = cr ** 2 / (1 - cr ** 2)
        snr["snr_mean"] = np.sqrt(snr["snr"] * snr["snr_reference"])
    # a signal-to-noise ratio does not rise with the frequency: the running minimum from the lowest shell on
    for k in snr:
        snr[k] = np.minimum.accumulate(np.clip(snr[k], 0, a.snr_max), 0)
        if a.snr_direction == "inplane":
            snr[k] = np.repeat(snr[k][:, -1:], nt, 1)
        elif a.snr_direction == "capped":
            snr[k] = np.minimum(snr[k], snr[k][:, -1:])

    bin_theta = (np.arange(nt) + 0.5) * 90.0 / nt
    theta = np.arange(0, 90 + 1e-9, a.theta_step)

    def to_grid(t):
        """(shells, harness thetas) -> (nu, theta), in the logarithm"""
        v = np.log(np.maximum(t, 1e-6))
        v = np.stack([PchipInterpolator(freq, v[:, j], extrapolate=False)(np.clip(nu, freq[0], freq[-1]))
                      for j in range(nt)], 1)      # (smooth in nu: a kink of the table is a long kernel)
        return np.exp(np.stack([np.interp(theta, bin_theta, row) for row in v]))

    th = np.radians(theta)[None, :]
    # (a frame has no frequency beyond 0.5 along either axis; the tables go on to the corner of its spectrum)
    fu, fz = np.minimum(nu[:, None] * np.sin(th), 0.5), np.minimum(nu[:, None] * np.cos(th), 0.5)
    L = np.interp(fz, np.r_[0, nu], np.r_[1, row_interpolation_response(a.z_interp, nu)]) if a.row_response == "restore" else 1.0
    T = model(par, nu)[:, None] / np.sinc(smear * nu)[:, None] * np.sinc(smear * fu) * L
    result = dict(nu=nu.tolist(), theta=theta.tolist(), transfer=T.tolist(),
                  model=dict(A_px2=par[0], fraction=par[1], sigma1_px=par[2], sigma2_px=par[3], rms=rms, subset=a.subset,
                             smear_px=smear, z_interp=a.z_interp, row_response=a.row_response,
                             snr_direction=a.snr_direction),
                  sources=dict(transfer=a.transfer, curves=a.curves, pair=a.pair, filtered=a.filtered, sharp=a.sharp))
    for k, v in snr.items():
        result[k] = to_grid(v).tolist()
    with open(a.out, "w") as fh:
        json.dump(result, fh)

    show = [np.searchsorted(nu, f) for f in (0.1, 0.2, 0.25, 0.3, 0.36, 0.42, 0.45, 0.48, 0.55, 0.65)]
    print("nu                      ", nu[show])
    print("G measured (interp.)    ", np.interp(nu[show], fg[ok], G[ok]))
    print("G model                 ", model(par, nu[show]))
    print("row interpolation       ", row_interpolation_response(a.z_interp, nu[show]))
    print("T along z / T in plane  ", T[show, 0] / T[show, -1])
    print("T at 45 deg / T in plane", T[show, len(theta) // 2] / T[show, -1])
    for k in snr:
        g = np.array(result[k])
        print(f"{k}: gain S / (S + 1) in plane", (g / (g + 1))[show, -1], "\n" + " " * len(k) + "              along z ", (g / (g + 1))[show, 0])
    g = np.array(result["snr"])
    print("filter g / T, in plane  ", (g / (g + 1) / T)[show, -1])
    print("filter g / T, along z   ", (g / (g + 1) / T)[show, 0])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
