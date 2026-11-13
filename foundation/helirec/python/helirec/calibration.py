"""Intensity calibration of an extended-field-of-view helical scan pair from the raw Bliss data, and the frame the
two scans are merged in.

  * darks / flats reduction (nx2etf), weight map + horizontal apodisation (prepare_weights_double), as the reference
    (nabu-helical / nabuxx) makes them,
  * the detector's blur field from the noise of the flats (the band-dependent redundancy weights),
  * the placement of both scans in the merged frame and the rotation axis there, from one axis and one vector.

The geometry is not measured here: the rotation axis, the second scan's place, the pitch, the rows' timing, the
detector map and the pose of every frame are the geometry calibration's (geometry.py, helirec-geometry), which
hands this module the merged frame's axis and vector (geometry.merged_layout).
Heavy kernels are C++ (helirec._core); the bookkeeping is numpy.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time
import types
from dataclasses import asdict, dataclass, field

import numpy as np

from . import _core
from .bliss import BlissScan, FrameRefs

NOMINAL_CURRENT_A = 0.2


# ============================================================================ raw access
class FrameSource:
    """Fast row-window access to one FrameRefs sequence (C++ pread reader)."""

    def __init__(self, refs: FrameRefs, shape):
        self.refs = refs
        self.rows, self.cols = shape
        self.reader = _core.RawReader(list(refs.files), refs.dataset, self.rows, self.cols)

    def frames(self, idx) -> np.ndarray:
        idx = np.asarray(idx)
        return self.reader.read_frames(self.refs.file_index[idx], self.refs.frame_index[idx])

    def rows_window(self, idx, row0, nrows) -> np.ndarray:
        idx = np.asarray(idx)
        return self.reader.read_rows(self.refs.file_index[idx], self.refs.frame_index[idx],
                                     np.ascontiguousarray(row0, np.int32), int(nrows))


def _mean_frames(src: FrameSource, transform=None, batch=40, progress=None) -> np.ndarray:
    """numpy.mean(stack.astype(float32), axis=0) semantics: sequential float32 sum, then / n.
    progress(k): called as k more frames are done."""
    n = len(src.refs)
    acc = None
    for s in range(0, n, batch):
        e = min(n, s + batch)
        fr = src.frames(np.arange(s, e)).astype(np.float32)
        for k in range(e - s):
            f = fr[k] if transform is None else transform(fr[k], s + k)
            if acc is None:
                acc = np.array(f, np.float32, copy=True)
            else:
                acc += f
        if progress:
            progress(e - s)
    return acc / n


def _bar(show, total, desc, unit):
    """a helirec.ProgressBar, or a stand-in that shows nothing"""
    if show:
        return _core.ProgressBar(int(total), desc, unit)
    return contextlib.nullcontext(types.SimpleNamespace(update=lambda n=1: None))


def reduce_darks_flats(scan: BlissScan, progress=None):
    """dark (rows, cols) and flats (n_series, rows, cols) exactly as nx2etf stores them.
    progress(k): called as k more of the len(scan.darks) + sum(len(series)) frames are done."""
    dark = _mean_frames(FrameSource(scan.darks, scan.frame_shape), progress=progress)
    flats = []
    for refs, ctl in zip(scan.flats, scan.flats_control_A):
        scale = np.float32(NOMINAL_CURRENT_A) / np.asarray(ctl, np.float32)

        def tr(F, i, scale=scale):
            return dark + (F - dark) * scale[i]
        flats.append(_mean_frames(FrameSource(refs, scan.frame_shape), tr, progress=progress))
    return dark.astype(np.float32), np.stack(flats).astype(np.float32)


def reduced_darks_flats(scan: BlissScan, cache_dir=None, log=print):
    """reduce_darks_flats(scan), kept in cache_dir: one file per scan, named after the scan and valid for the
    dark and flat files it was made from (path, size, time of change), the frames taken from them and the flats'
    ring current. Reducing a scan's 1200 frames takes most of a minute; reading the file a fraction of a second.
    cache_dir None: nothing is kept."""
    if not cache_dir:
        return reduce_darks_flats(scan)
    import hashlib
    h = hashlib.sha256(f"helirec-darks-flats-v{CALIBRATION_VERSION}|{list(scan.frame_shape)}".encode())
    for refs in (scan.darks, *scan.flats):
        for k in np.unique(refs.file_index):
            st = os.stat(refs.files[int(k)])
            h.update(f"{os.path.abspath(refs.files[int(k)])}|{st.st_size}|{st.st_mtime_ns}".encode())
        h.update(refs.file_index.tobytes())
        h.update(refs.frame_index.tobytes())
    for ctl in scan.flats_control_A:
        h.update(np.ascontiguousarray(ctl, np.float64).tobytes())
    fp = h.hexdigest()
    path = os.path.join(cache_dir, f"darks_flats_{scan.name}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            if "_fingerprint" in z.files and str(z["_fingerprint"]) == fp:
                return z["dark"], z["flats"]
        log(f"darks and flats in {path}: of other files, made again")
    dark, flats = reduce_darks_flats(scan)
    os.makedirs(cache_dir, exist_ok=True)

    def save(t):
        with open(t, "wb") as fh:
            np.savez(fh, _fingerprint=np.array(fp), dark=dark, flats=flats)
    _atomic_write(path, save)
    return dark, flats


# the low-pass of JPEG 2000's irreversible (9/7) wavelet, taps 0, +-1, ... +-4; unit gain
_LOW_97 = (0.6029490182363579, 0.2668641184428723, -0.07822326652898785, -0.01686411844287495, 0.02674875741080976)


def _low_band_97(x):
    """along the last axis: sample n is the low-pass about x[2 n] (whole-sample symmetric at both ends)"""
    n = x.shape[-1]
    idx = np.arange(0, n, 2)
    out = _LOW_97[0] * x[..., idx]
    for m in range(1, 5):
        left, right = np.abs(idx - m), idx + m
        right = np.where(right > n - 1, 2 * (n - 1) - right, right)
        out = out + _LOW_97[m] * (x[..., left] + x[..., right])
    return out


def at_resolution(image, reduce, chunk_rows):
    """An image of the detector (a dark, a flat, a weight: (..., rows, cols)) as RawReader(reduce=) gives the
    frames stored as JPEG 2000 in chunks of chunk_rows rows: `reduce` levels of the 9/7 wavelet's low band, each
    chunk of rows an image of its own, sample (j, k) at the stored pixel (j, k) 2^reduce. A flat reduced so
    divides a frame's fixed pattern out as the full-resolution flat does; a mean of 2^reduce pixels would not.
    A last chunk that reaches beyond the frame holds zeros there (as the detector software stores it), which
    change the frame's last two or three reduced rows, here as in the frames. On a Paris4 frame this differs
    from the reader's reduced frame by 0.3 counts rms (the rounding), at 1, 2 and 3 levels."""
    x = np.asarray(image, np.float64)
    rows = x.shape[-2]
    beyond = -rows % chunk_rows
    if beyond:
        x = np.concatenate([x, np.zeros(x.shape[:-2] + (beyond, x.shape[-1]))], -2)
    for level in range(int(reduce)):
        c = chunk_rows >> level
        x = np.concatenate([np.swapaxes(_low_band_97(np.swapaxes(x[..., a:a + c, :], -1, -2)), -1, -2) for a in range(0, x.shape[-2], c)], -2)
        x = _low_band_97(x)
    return x[..., :rows >> int(reduce), :].astype(np.float32)


# ============================================================================ weights
def weight_map(flat_stacks, transition_width_vertical=15.0, sigma=10.0, crop_top=0, crop_bottom=0) -> np.ndarray:
    """prepare_weights_double: gaussian_filter(mean(all flats) / max, sigma) * vertical erf border."""
    prof = 0
    n = 0
    for st in flat_stacks:
        for i in range(st.shape[0]):
            prof = prof + st[i]
        n += st.shape[0]
    prof = prof / n
    prof = (prof / prof.max()).astype(np.float32)
    g = _core.gaussian_filter1d(np.ascontiguousarray(prof), 0, sigma, "reflect")
    g = _core.gaussian_filter1d(g, 1, sigma, "reflect")
    border = _core.weight_border_v(prof.shape[0], transition_width_vertical, crop_top, crop_bottom)
    return (g * border[:, None]).astype(np.float32)


def apodisation_widths(n_cols, est_cor_main_offset, companion_dx, transition_width_horizontal=1000.0,
                       transition_width_borders=50.0):
    """(w_left, w_right) for (main, companion) as prepare_weights_double --is_tetra writes them.
    Main scan: 'pente' beyond the rotation axis (min(2 tw, distance axis->edge)) on the axis side and a
    border taper towards the companion; companion: border taper on the side facing the main scan."""
    r = (n_cols - 1) / 2 + est_cor_main_offset
    if r < (n_cols - 1) / 2:
        raise NotImplementedError("rotation axis in the left half of the main scan is not supported yet")
    pente = min(2 * transition_width_horizontal, n_cols - r)
    if companion_dx < 0:   # companion on the left of the main scan
        return (transition_width_borders, pente), (0.0, transition_width_borders)
    raise NotImplementedError("companion on the right of the main scan is not supported yet")


@dataclass
class ScanPlacement:
    shift_x: int
    x_readjustment: float
    z_readjustment: float


# bump when anything that changes the cached results changes (reduction, COR / overlap search, cache layout)
CALIBRATION_VERSION = 2

@dataclass
class Calibration:
    main: str
    companion: str
    cor_main_detector: float          # rotation axis in the main scan's own columns (the merged frame's: nominal)
    vector: tuple                     # (dy, dx) main -> companion
    rot_center: float                 # rotation axis in the merged frame
    merged_width: int
    placement: dict                   # name -> ScanPlacement (as dict)
    apodisation: dict                 # name -> (w_left, w_right)
    estimated_cor_from_motor_pix: dict
    details: dict = field(default_factory=dict)
    # The detector's blur field (measure_detector_blur; details["detector_blur"]): [a11, a12, a21, a22, c1, c2], a uniform blur of
    # A r + c pixels at r = (column, row) from the detector's centre in units of 1000 pixels, from the noise of the
    # flats. []: measured, nothing significant.
    detector_blur_field: list | None = None
    fingerprint: str = ""             # acquisition_fingerprint() of the scan pair it was computed from
    version: int = CALIBRATION_VERSION

    def to_json(self):
        return json.dumps(asdict(self), indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def acquisition_fingerprint(main: BlissScan, companion: BlissScan) -> str:
    """sha256 over everything the calibration depends on: calibration version, scan identities, metadata, frame
    geometry, trajectories and the referenced raw files (path, size, mtime). A replaced acquisition (even under the
    same name) or a different pair invalidates the cache."""
    import hashlib
    h = hashlib.sha256(f"helirec-calibration-v{CALIBRATION_VERSION}".encode())
    for sc in (main, companion):
        meta = dict(name=sc.name, master=os.path.abspath(sc.master), shape=list(sc.frame_shape), dtype=str(sc.dtype),
                    n_proj=len(sc.projections), n_darks=len(sc.darks), n_flats=[len(f) for f in sc.flats],
                    energy_kev=sc.energy_kev, distance_m=sc.distance_m, pixel_size_m=sc.pixel_size_m,
                    est_cor=sc.estimated_cor_from_motor_pix)
        h.update(json.dumps(meta, sort_keys=True).encode())
        for arr in (sc.unwrapped_angles_deg, sc.z_translation_m, sc.control_A, sc.flats_unwrapped_angles_deg):
            h.update(np.ascontiguousarray(arr, np.float64).tobytes())
        for fn in [sc.master, *sc.referenced_files()]:
            st = os.stat(fn)
            h.update(f"{os.path.abspath(fn)}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _atomic_write(path, write):
    """write(tmp_path) then rename over `path` (readers never see a partial file)."""
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def placement_from_vector(vector, n_cols):
    """nabuxx check_companion: outermost companion anchored at x = 0 of the merged frame."""
    dy, dx = vector
    cum = [(0.0, 0.0), (float(dy), float(dx))]
    outer_x = cum[-1][1]
    anchor = -outer_x if outer_x < 0 else 0.0
    out = []
    for j, (z, x) in enumerate(cum):
        d = anchor + x
        sx = int(math.floor(d))
        out.append(ScanPlacement(shift_x=sx, x_readjustment=float(np.float32(sx - d)),
                                 z_readjustment=float(z) if j > 0 else 0.0))
    width = n_cols + int(math.ceil(abs(outer_x)))
    return out, width, anchor


REDUCTION_KEYS = ("main_dark", "main_flats", "companion_dark", "companion_flats", "weight_map")


def _load_cache(cache_dir, fp, log):
    """(calibration dict or None, reduction arrays or None) from cache_dir, each only if its fingerprint matches."""
    js, npz = os.path.join(cache_dir, "calibration.json"), os.path.join(cache_dir, "calibration_arrays.npz")
    arrays = None
    if os.path.exists(npz):
        with np.load(npz) as z:
            if "_fingerprint" in z.files and str(z["_fingerprint"]) == fp and all(k in z.files for k in REDUCTION_KEYS):
                arrays = {k: z[k] for k in REDUCTION_KEYS}
            else:
                log(f"calibration cache {npz}: other acquisition / version, ignored")
    cal = None
    if arrays is not None and os.path.exists(js):
        d = json.load(open(js))
        if d.get("fingerprint") == fp and d.get("version") == CALIBRATION_VERSION:
            cal = d
        else:
            log(f"calibration cache {js}: other acquisition / version, ignored")
    return cal, arrays


def calibrate(main: BlissScan, companion: BlissScan, layout, cache_dir=None, log=print, progress=None):
    """The intensity calibration from raw data, in the merged frame of `layout`. Returns (Calibration, arrays dict).

    layout: dict(cor_main_detector, vector = (dy, dx) main -> companion): the rotation axis in the main scan's
    columns and the companion's place against the main scan, from the geometry calibration
    (geometry.merged_layout). The placement of both scans, the merged width, the axis in the merged frame and the
    horizontal apodisation follow from it.
    cache_dir: the darks / flats reduction and the detector's blur field are cached there, keyed by
    acquisition_fingerprint(); a cache of another acquisition or calibration version is ignored and replaced."""
    if progress is None:   # progress bars for the long steps: whenever the log is shown
        progress = log is print
    fp = acquisition_fingerprint(main, companion)
    cached_cal, arrays = _load_cache(cache_dir, fp, log) if cache_dir else (None, None)
    if cached_cal is not None:
        cached_cal = _current_names(cached_cal)
    if arrays is None:
        arrays = {}
        for tag, sc in (("main", main), ("companion", companion)):
            log(f"[{tag}] reducing darks/flats of {sc.name} ...")
            with _bar(progress, len(sc.darks) + sum(len(refs) for refs in sc.flats), f"darks/flats ({tag})", "frame") as bar:
                dark, flats = reduce_darks_flats(sc, bar.update)
            arrays[f"{tag}_dark"], arrays[f"{tag}_flats"] = dark, flats
        # weight map from all flats (companion first, as the reference --scan_files order)
        arrays["weight_map"] = weight_map([arrays["companion_flats"], arrays["main_flats"]])
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            def save(t):
                with open(t, "wb") as fh:
                    np.savez(fh, _fingerprint=np.array(fp), **arrays)
            _atomic_write(os.path.join(cache_dir, "calibration_arrays.npz"), save)
    W = main.frame_shape[1]
    cor_main = float(layout["cor_main_detector"])
    vector = tuple(float(v) for v in layout["vector"])
    pl, width, anchor = placement_from_vector(vector, W)
    (aw_m, aw_c) = apodisation_widths(W, main.estimated_cor_from_motor_pix, vector[1])
    cal = Calibration(main=main.name, companion=companion.name, cor_main_detector=cor_main, vector=vector,
                      rot_center=float(cor_main + anchor), merged_width=width,
                      placement={main.name: pl[0], companion.name: pl[1]},
                      apodisation={main.name: aw_m, companion.name: aw_c},
                      estimated_cor_from_motor_pix={main.name: main.estimated_cor_from_motor_pix,
                                                    companion.name: companion.estimated_cor_from_motor_pix},
                      details=dict(layout=dict(layout.get("measured") or {})), fingerprint=fp)
    if cached_cal is not None and cached_cal.get("detector_blur_field") is not None:
        # (the detector does not change with the frame the scans are merged in: the cached measurement)
        cal.detector_blur_field = list(cached_cal["detector_blur_field"])
        cal.details["detector_blur"] = (cached_cal.get("details") or {}).get("detector_blur")
        log(f"calibration loaded from {cache_dir}")
    else:
        _add_detector_blur(cal, main, arrays, log)
        if cache_dir:
            _save_calibration(cache_dir, cal)
    log(f"merged frame: rotation axis {cor_main:.3f} in the main scan's columns, companion at {vector} (rows, columns): "
        f"axis {cal.rot_center:.3f} of {width} merged columns")
    return cal, {**arrays, **_scan_weights(arrays["weight_map"], main, cal)}


def _save_calibration(cache_dir, cal):
    def save_json(t):
        with open(t, "w") as fh:
            fh.write(cal.to_json())
    _atomic_write(os.path.join(cache_dir, "calibration.json"), save_json)


def _current_names(cached: dict) -> dict:
    """a cached calibration under today's names (until 2026-11-11 the detector's blur field was "detector_smear",
    its details "smear" with "max_smear_px")"""
    d = dict(cached)
    if "detector_smear" in d:
        d.setdefault("detector_blur_field", d.pop("detector_smear"))
    details = dict(d.get("details") or {})
    if "smear" in details:
        blur = details.pop("smear")
        if isinstance(blur, dict) and "max_smear_px" in blur:
            blur = {("max_blur_px" if k == "max_smear_px" else k): v for k, v in blur.items()}
        details.setdefault("detector_blur", blur)
        d["details"] = details
    return d


def _add_detector_blur(cal, main, arrays, log):
    """measure the detector's blur field from the main scan's flats and record it in the calibration"""
    res = measure_detector_blur(main, arrays["main_dark"], log=log)
    cal.detector_blur_field = [float(v) for v in res["field"]]
    cal.details["detector_blur"] = res


# ============================================================================ detector blur from the flats' noise
BLUR_TILE = 128
BLUR_MIN_FLATS = 20      # fewer flats in the series do not carry the noise spectra
BLUR_MIN_PX = 0.5        # a field whose blur nowhere reaches this many pixels is not used
BLUR_MIN_GAIN = 0.97     # ... nor one that does not bring the residual of the noise spectra below this share


def flat_noise_spectra(scan: BlissScan, dark, n_flats=200, tile=BLUR_TILE):
    """Noise power spectra of the scan's first flat series on a lattice of tiles (_core.tile_power_spectra): of the
    differences of consecutive flats, each less the dark and scaled to its mean (what is the same in both, the beam's
    profile and the fixed pattern, drops out; the noise stays). Returns (power (ny, nx, tile, tile / 2 + 1), tile
    origins ys, xs, flats used)."""
    refs = scan.flats[0]
    n = min(n_flats, len(refs)) // 2 * 2
    if n < 2:
        raise ValueError("the detector's blur needs at least two flats")
    rows, cols = scan.frame_shape
    ys = list(range((rows % tile) // 2, rows - tile + 1, tile))
    xs = list(range((cols % tile) // 2, cols - tile + 1, tile))
    dark = np.asarray(dark, np.float32)
    power = np.zeros((len(ys), len(xs), tile, tile // 2 + 1))
    for k0 in range(0, n, 40):
        fl = scan.read_frames(refs, slice(k0, min(n, k0 + 40))).astype(np.float32) - dark
        fl /= fl.mean((1, 2), keepdims=True)
        power += _core.tile_power_spectra(np.ascontiguousarray(fl[0::2] - fl[1::2]), tile, ys, xs)
    return power, ys, xs, n


def measure_detector_blur(scan: BlissScan, dark, n_flats=200, log=print):
    """The detector's blur field of a scan, from the noise of its flats (_core.detector_blur_field_fit: calib.hpp). The field is kept
    only if it explains the noise spectra (residual below BLUR_MIN_GAIN of the residual without it) and reaches
    BLUR_MIN_PX somewhere on the detector; otherwise "field" is [] and nothing changes in the reconstruction."""
    t0 = time.time()
    rows, cols = scan.frame_shape
    n_have = min(n_flats, len(scan.flats[0])) if scan.flats else 0
    if n_have < BLUR_MIN_FLATS or (rows // BLUR_TILE) * (cols // BLUR_TILE) < 6:
        reason = (f"{n_have} flats in the first series" if n_have < BLUR_MIN_FLATS else
                  f"a detector of {rows} x {cols} holds too few tiles of {BLUR_TILE}")
        log(f"detector blur: not measured ({reason})")
        return dict(field=[], measured=None, reason=reason, scan=scan.name)
    power, ys, xs, n = flat_noise_spectra(scan, dark, n_flats)
    field, rms_none, rms_fit = _core.detector_blur_field_fit(power, ys, xs, rows, cols)
    th = np.array(field)
    try:
        zero = (-np.linalg.solve(th[:4].reshape(2, 2), th[4:]) * 1000 + [0.5 * (cols - 1), 0.5 * (rows - 1)]).tolist()
    except np.linalg.LinAlgError:
        zero = [float("nan"), float("nan")]
    res = dict(field=[float(v) for v in field], rms_none=float(rms_none), rms_fit=float(rms_fit), zero=zero,
               tiles=[len(ys), len(xs)])
    corners = [(x, y) for x in (-0.5 * cols / 1000, 0.5 * cols / 1000) for y in (-0.5 * rows / 1000, 0.5 * rows / 1000)]
    reach = max(abs(th[0] * x + th[1] * y + th[4]) for x, y in corners)
    res.update(n_flats=int(n), max_blur_px=float(reach), scan=scan.name)
    ok = res["rms_fit"] < BLUR_MIN_GAIN * res["rms_none"] and reach >= BLUR_MIN_PX
    res["measured"] = res["field"]
    if not ok:
        res["field"] = []
    log(f"detector blur from the noise of {n} flats of {scan.name}: "
        + (f"{reach:.1f} px along the columns at the worst corner, none at column {res['zero'][0]:.0f} "
           f"(residual of the noise spectra {res['rms_none']:.3f} -> {res['rms_fit']:.3f})" if ok else
           f"nothing significant (residual {res['rms_none']:.3f} -> {res['rms_fit']:.3f}, reach {reach:.2f} px)")
        + f"  [{time.time() - t0:.0f}s]")
    return res


def _scan_weights(wm, main, cal):
    """per-scan weight maps: the common weight map x each scan's horizontal apodisation"""
    W = main.frame_shape[1]
    out = {}
    for tag, name in (("main", cal.main), ("companion", cal.companion)):
        a = cal.apodisation[name]
        out[f"{tag}_weight"] = (wm * _core.apodisation_mask(W, a[0], a[1], 1e-8)[None, :]).astype(np.float32)
    return out
