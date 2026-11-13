"""Intensity calibration of an extended-field-of-view helical scan pair from the raw BLISS data, and the frame the two
scans are merged in: helirec::calibration::calibrate_intensity (helirec/calibration/intensity.hpp); from a shell
`helirec-calibration intensity MAIN COMPANION --geometry geometry.json --out DIR`.

  * darks / flats reduction (nx2etf), weight map + horizontal apodisation (prepare_weights_double), as the reference
    (nabu-helical / nabuxx) makes them,
  * the detector's blur field from the noise of the flats (the band-dependent redundancy weights),
  * the placement of both scans in the merged frame and the rotation axis there, from one axis and one vector.

The geometry is not measured here: the rotation axis, the second scan's place, the pitch, the rows' timing, the
detector map and the pose of every frame are the geometry calibration's (geometry.py, helirec-calibration geometry),
which hands this module the merged frame's axis and vector (geometry.merged_layout).

The caches (cache_dir) are helirec's array files (a first line of JSON naming the arrays, then float32 values):
calibration_arrays.bin (the reduction and the weight map), calibration.json (the rest, the blur field among it) and,
for a scan alone, darks_flats_<scan>.bin.
"""
from __future__ import annotations

import numpy as np

from . import _core

NOMINAL_CURRENT_A = _core.NOMINAL_CURRENT_A
CALIBRATION_VERSION = _core.INTENSITY_CALIBRATION_VERSION
Calibration = _core.IntensityCalibration
ScanPlacement = _core.ScanPlacement


def reduce_darks_flats(scan, progress=None):
    """dark (rows, cols) and flats (n_series, rows, cols) exactly as nx2etf stores them. progress(k): called as k more
    of the scan's dark and flat frames are done. (helirec::calibration::reduce_darks_flats)"""
    return _core.reduce_darks_flats(scan, progress)


def reduced_darks_flats(scan, cache_dir=None, log=print):
    """reduce_darks_flats(scan), kept in cache_dir as darks_flats_<scan>.bin, valid for the dark and flat files it was
    made from (path, size, time of change), the frames taken from them and the flats' ring current. Reducing a scan's
    1200 frames takes most of a minute; reading the file a fraction of a second. cache_dir None: nothing is kept.
    (helirec::calibration::reduced_darks_flats)"""
    return _core.reduced_darks_flats(scan, str(cache_dir or ""), log)


def at_resolution(image, reduce, chunk_rows):
    """An image of the detector (a dark, a flat, a weight: (..., rows, cols)) as RawReader(reduce=) gives the
    frames stored as JPEG 2000 in chunks of chunk_rows rows: `reduce` levels of the 9/7 wavelet's low band, each
    chunk of rows an image of its own, sample (j, k) at the stored pixel (j, k) 2^reduce. A flat reduced so
    divides a frame's fixed pattern out as the full-resolution flat does; a mean of 2^reduce pixels would not.
    A last chunk that reaches beyond the frame holds zeros there (as the detector software stores it), which
    change the frame's last two or three reduced rows, here as in the frames. On a Paris4 frame this differs
    from the reader's reduced frame by 0.3 counts rms (the rounding), at 1, 2 and 3 levels.
    (helirec::calibration::at_resolution)"""
    x = np.asarray(image, np.float64)
    out = _core.at_resolution(np.ascontiguousarray(x.reshape((-1,) + x.shape[-2:])), int(reduce), int(chunk_rows))
    return out.reshape(x.shape[:-2] + out.shape[-2:])


def weight_map(flat_stacks, transition_width_vertical=15.0, sigma=10.0, crop_top=0, crop_bottom=0) -> np.ndarray:
    """prepare_weights_double: gaussian_filter(mean(all flats) / max, sigma) * vertical erf border.
    (helirec::calibration::weight_map)"""
    return _core.weight_map([np.ascontiguousarray(st, np.float32) for st in flat_stacks], float(transition_width_vertical), float(sigma), int(crop_top), int(crop_bottom))


def apodisation_widths(n_cols, est_cor_main_offset, companion_dx, transition_width_horizontal=1000.0,
                       transition_width_borders=50.0):
    """(w_left, w_right) for (main, companion) as prepare_weights_double --is_tetra writes them.
    Main scan: 'pente' beyond the rotation axis (min(2 tw, distance axis->edge)) on the axis side and a
    border taper towards the companion; companion: border taper on the side facing the main scan.
    (helirec::calibration::apodisation_widths)"""
    try:
        return _core.apodisation_widths(int(n_cols), float(est_cor_main_offset), float(companion_dx), float(transition_width_horizontal), float(transition_width_borders))
    except RuntimeError as e:
        raise NotImplementedError(str(e)) from None


def placement_from_vector(vector, n_cols):
    """nabuxx check_companion: outermost companion anchored at x = 0 of the merged frame -> ([main, companion]
    ScanPlacement, merged width, anchor). (helirec::calibration::placement_from_vector)"""
    return _core.placement_from_vector(tuple(float(v) for v in vector), int(n_cols))


def acquisition_fingerprint(main, companion) -> str:
    """sha256 over everything the calibration depends on: calibration version, scan identities, metadata, frame
    geometry, trajectories and the referenced raw files (path, size, mtime). A replaced acquisition (even under the
    same name) or a different pair invalidates the cache. (helirec::calibration::acquisition_fingerprint)"""
    return _core.acquisition_fingerprint(main, companion)


def calibrate(main, companion, layout, cache_dir=None, log=print, progress=None):
    """The intensity calibration from raw data, in the merged frame of `layout`. Returns (Calibration, arrays dict:
    main_dark, main_flats, companion_dark, companion_flats, weight_map, main_weight, companion_weight).

    layout: dict(cor_main_detector, vector = (dy, dx) main -> companion[, measured]): the rotation axis in the main
    scan's columns and the companion's place against the main scan, from the geometry calibration
    (geometry.merged_layout). The placement of both scans, the merged width, the axis in the merged frame and the
    horizontal apodisation follow from it.
    cache_dir: the darks / flats reduction and the detector's blur field are cached there, keyed by
    acquisition_fingerprint(); a cache of another acquisition or calibration version is ignored and replaced.
    progress: progress bars for the reduction (default: whenever the log is print).
    (helirec::calibration::calibrate_intensity)"""
    if progress is None:
        progress = log is print
    cal = _core.calibrate_intensity(main, companion, float(layout["cor_main_detector"]), tuple(float(v) for v in layout["vector"]),
                                    dict(layout.get("measured") or {}), str(cache_dir or ""), log, bool(progress))
    return cal, cal.arrays()


BLUR_TILE = 128


def flat_noise_spectra(scan, dark, n_flats=200, tile=BLUR_TILE):
    """Noise power spectra of the scan's first flat series on a lattice of tiles: of the differences of consecutive
    flats, each less the dark and scaled to its mean. Returns (power (ny, nx, tile, tile / 2 + 1), tile origins ys, xs,
    flats used). (helirec::calibration::flat_noise_spectra)"""
    return _core.flat_noise_spectra(scan, np.ascontiguousarray(dark, np.float32), int(n_flats), int(tile))


def measure_detector_blur(scan, dark, n_flats=200, log=print):
    """The detector's blur field of a scan, from the noise of its flats: a dict whose "field" is [] when it does not
    explain the noise spectra or reaches less than half a pixel. (helirec::calibration::measure_detector_blur)"""
    return _core.measure_detector_blur(scan, np.ascontiguousarray(dark, np.float32), int(n_flats), log)
