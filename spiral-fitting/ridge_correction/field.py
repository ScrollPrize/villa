"""Field construction and write verification.

field(): u = taper(z) * ramp(confidence; r0, r1) * clip(u*, -cap, cap) on
the gap-logit grid, zero outside the z band and outside the winding range;
no smoothing. verify(): at random (sheet, theta, z) in the band, the
winding the edited model realises versus the bilinear intended field.
"""


import numpy as np


def ramp(c, r0, r1):
    return np.clip((c - r0) / max(r1 - r0, 1e-9), 0, 1)


def taper_rows(zr, band, T):
    z0, z1 = band
    w = np.zeros_like(zr)
    inside = (zr >= z0) & (zr <= z1)
    d = np.minimum(zr - z0, z1 - zr)
    w[inside] = np.where(d[inside] >= T, 1.0, 0.5 - 0.5 * np.cos(np.pi * d[inside] / T))
    return w


def field(offset, confidence, geo, rows, band, sheets, *, r0, r1, cap, taper):
    """u = taper(z) * ramp(confidence; r0, r1) * clip(offset, -cap, cap) on the
    (Z, Wc) gap grid; ``offset`` / ``confidence`` cover gap rows ``rows``;
    windings outside ``sheets`` and the pinned column 0 stay 0."""
    idx = np.array(geo["idx"])
    Wc = int(geo["Wc"])
    Z = int(geo["Z"])
    r0w, r1w = rows
    zrow = geo["zmin"] + np.arange(Z) * (geo["zmax"] - geo["zmin"]) / (Z - 1)
    tw = taper_rows(zrow, band, taper)
    U = np.zeros((Z, Wc))
    u = np.asarray(offset).astype(np.float64)
    c = np.asarray(confidence).astype(np.float64)
    f = ramp(c, r0, r1) * np.clip(u, -cap, cap)
    U[r0w:r1w] = f * tw[r0w:r1w, None]
    U[:, 0] = 0.0
    U[:, : int(idx[sheets[0] - 1])] = 0.0
    U[:, int(idx[sheets[1]]) :] = 0.0
    st = {
        "mean_u": float(U[r0w:r1w].mean()),
        "mean_abs_u": float(np.abs(U[r0w:r1w]).mean()),
        "share_moved": float((np.abs(U[r0w:r1w]) > 1e-6).mean()),
        "share_cap": float((np.abs(U[r0w:r1w]) >= cap - 1e-9).mean()),
    }
    return U, st


def verify(model_old, model_new, U, geo, band, rng, n, sheets, at_nodes=False):
    import torch

    import sample_spiral

    tro = model_old.get_slice_to_spiral_transform()
    trn = model_new.get_slice_to_spiral_transform()
    dr = float(model_old.get_dr_per_winding().detach())
    idx = np.array(geo["idx"])
    num = np.diff(idx)
    Wc = int(geo["Wc"])
    Z = int(geo["Z"])
    k = rng.integers(sheets[0], sheets[1] + 1, n)
    th = rng.uniform(0, 1, n)
    z = rng.uniform(band[0], band[1], n)
    if at_nodes:  # snap to exact node positions (row r, column c of sheet k's segment): checks the write itself, no interpolation
        Zr = (z - geo["zmin"]) / (geo["zmax"] - geo["zmin"]) * (Z - 1)
        z = geo["zmin"] + np.round(Zr) * (geo["zmax"] - geo["zmin"]) / (Z - 1)
        cc = np.round(idx[k - 1] + th * num[k - 1])
        cc = np.clip(cc, idx[k - 1], idx[k] - 1)
        th = (cc * Wc / (Wc - 1) - idx[k - 1]) / num[k - 1]
        keep = (th >= 0) & (th < 1)
        k, th, z = k[keep], th[keep], z[keep]
    # intended u: bilinear on the node grid
    pr = (z - geo["zmin"]) / (geo["zmax"] - geo["zmin"]) * (Z - 1)
    r0 = np.floor(pr).astype(int)
    fr = pr - r0
    pc = np.clip((idx[k - 1] + th * num[k - 1]) * (Wc - 1) / Wc, 0, Wc - 1)
    c0 = np.minimum(np.floor(pc).astype(int), Wc - 2)
    fc = pc - c0
    ui = (
        U[r0, c0] * (1 - fr) * (1 - fc)
        + U[r0 + 1, c0] * fr * (1 - fc)
        + U[r0, c0 + 1] * (1 - fr) * fc
        + U[r0 + 1, c0 + 1] * fr * fc
    )
    TH = th * 2 * np.pi
    rr = (k + th) * dr
    sp = torch.tensor(np.stack([z, rr * np.sin(TH), rr * np.cos(TH)], 1), dtype=torch.float32)
    with torch.inference_mode():
        Xn = trn.inv(sp)
        Xo = tro.inv(sp)
        S = tro(Xn)
        t2, _ = sample_spiral.get_theta(S[:, 1:])
        r2 = torch.linalg.norm(S[:, 1:], dim=1)
    Wn = (r2.numpy() / dr - t2.numpy() / (2 * np.pi)).astype(np.float64)
    ur = Wn - k
    ur = ur - np.round(ur - ui)  # theta wrap guard (integer ambiguity only)
    d = (Xn - Xo).numpy().astype(np.float64)
    ok = np.isfinite(ur) & np.isfinite(ui)
    moved = ok & (np.abs(ui) > 1e-4)
    res = {
        "n": int(ok.sum()),
        "n_moved": int(moved.sum()),
        "corr": float(np.corrcoef(ui[moved], ur[moved])[0, 1]) if moved.sum() > 10 else None,
        "rms_err_w": float(np.sqrt(((ur - ui)[ok] ** 2).mean())),
        "rms_intended_w": float(np.sqrt((ui[ok] ** 2).mean())),
        "slope": float((ui[moved] * ur[moved]).sum() / (ui[moved] ** 2).sum())
        if moved.any()
        else None,
        "p99_abs_err_w": float(np.percentile(np.abs(ur - ui)[ok], 99)),
        "rms_err_unmoved_w": float(np.sqrt(((ur - ui)[ok & ~moved] ** 2).mean()))
        if (ok & ~moved).any()
        else None,
        "mean_disp_zyx_vx": [float(v) for v in d[ok].mean(0)],
        "rms_disp_vx": float(np.sqrt((d[ok] ** 2).sum(1).mean())),
    }
    return res


def _support(geo, k, th, z):
    """Gap-grid cell (row, column) and bilinear weights of spiral points (winding k, turn th, z)."""
    idx = np.array(geo["idx"])
    num = np.diff(idx)
    Wc, Z = int(geo["Wc"]), int(geo["Z"])
    pr = (z - geo["zmin"]) / (geo["zmax"] - geo["zmin"]) * (Z - 1)
    r0 = np.floor(pr).astype(int)
    pc = np.clip((idx[k - 1] + th * num[k - 1]) * (Wc - 1) / Wc, 0, Wc - 1)
    c0 = np.minimum(np.floor(pc).astype(int), Wc - 2)
    return r0, c0, pr - r0, pc - c0


def _bilinear(U, r0, c0, fr, fc):
    return (U[r0, c0] * (1 - fr) * (1 - fc) + U[r0 + 1, c0] * fr * (1 - fc)
            + U[r0, c0 + 1] * (1 - fr) * fc + U[r0 + 1, c0 + 1] * fr * fc)


def between_node_fidelity(model_old, model_new, U_asked, U_written, geo, band, sheets, rng, n=20000, pool=120000):
    """Write-back fidelity between nodes in the moved region.

    Random points (winding, turn, z) in the band whose bilinear support has at
    least one moved node (asked or written field non-zero); the first ``n`` of
    ``pool`` draws are used. Intended = the asked field, bilinear on the gap
    grid; realised = the winding the edited model gives the point, read back
    through the unedited model (as in verify()). Returns n, corr, slope,
    rms_intended, rms_err and ratio = rms_err / rms_intended (windings)."""
    import torch

    import sample_spiral

    moved_node = (U_asked != 0) | (U_written != 0)
    k = rng.integers(sheets[0], sheets[1] + 1, pool)
    th = rng.uniform(0, 1, pool)
    z = rng.uniform(band[0], band[1], pool)
    r0, c0, fr, fc = _support(geo, k, th, z)
    corner = moved_node[r0, c0] | moved_node[r0 + 1, c0] | moved_node[r0, c0 + 1] | moved_node[r0 + 1, c0 + 1]
    sel = np.flatnonzero(corner)[:n]
    if len(sel) < 3:
        return {"n": len(sel), "corr": None, "slope": None, "rms_intended": 0.0, "rms_err": None, "ratio": None}
    k, th, z, r0, c0, fr, fc = k[sel], th[sel], z[sel], r0[sel], c0[sel], fr[sel], fc[sel]
    ui = _bilinear(U_asked, r0, c0, fr, fc)
    guide = _bilinear(U_written, r0, c0, fr, fc)
    tro = model_old.get_slice_to_spiral_transform()
    trn = model_new.get_slice_to_spiral_transform()
    dr = float(model_old.get_dr_per_winding().detach())
    TH = th * 2 * np.pi
    rr = (k + th) * dr
    sp = torch.tensor(np.stack([z, rr * np.sin(TH), rr * np.cos(TH)], 1), dtype=torch.float32)
    with torch.inference_mode():
        S = tro(trn.inv(sp))
        t2, _ = sample_spiral.get_theta(S[:, 1:])
        r2 = torch.linalg.norm(S[:, 1:], dim=1)
    ur = (r2.numpy() / dr - t2.numpy() / (2 * np.pi)).astype(np.float64) - k
    ur = ur - np.round(ur - guide)  # theta wrap guard (integer ambiguity only)
    ok = np.isfinite(ui) & np.isfinite(ur)
    ui, ur = ui[ok], ur[ok]
    ri = float(np.sqrt((ui**2).mean()))
    re = float(np.sqrt(((ur - ui) ** 2).mean()))
    sd = ui.std() > 0 and ur.std() > 0
    return {
        "n": len(ui),
        "share_of_band_points": float(corner.mean()),
        "corr": float(np.corrcoef(ui, ur)[0, 1]) if sd else None,
        "slope": float((ui * ur).sum() / (ui**2).sum()) if (ui**2).sum() > 0 else None,
        "rms_intended": ri,
        "rms_err": re,
        "ratio": re / ri if ri > 0 else None,
    }
