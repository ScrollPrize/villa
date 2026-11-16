"""Intensity calibration of an extended-field-of-view helical scan pair from the raw BLISS data, and the frame the two
scans are merged in: vc_tomography::calibration::calibrate_intensity (vc_tomography/calibration/intensity.hpp); from a shell
`vc_tomography_calibration intensity MAIN COMPANION --geometry geometry.json --out DIR`.

  * darks / flats reduction (nx2etf), weight map + horizontal apodisation (prepare_weights_double), as the reference
    (nabu-helical / nabuxx) makes them,
  * the detector's blur field from the noise of the flats (the band-dependent redundancy weights),
  * the placement of both scans in the merged frame and the rotation axis there, from one axis and one vector.

The geometry is not measured here: the rotation axis, the second scan's place, the pitch, the rows' timing, the
detector map and the pose of every frame are the geometry calibration's (geometry.py, vc_tomography_calibration geometry),
which hands this module the merged frame's axis and vector (geometry.merged_layout).

The caches (cache_dir) are vc_tomography's array files (a first line of JSON naming the arrays, then float32 values):
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
    of the scan's dark and flat frames are done. (vc_tomography::calibration::reduce_darks_flats)"""
    return _core.reduce_darks_flats(scan, progress)


def reduced_darks_flats(scan, cache_dir=None, log=print):
    """reduce_darks_flats(scan), kept in cache_dir as darks_flats_<scan>.bin, valid for the dark and flat files it was
    made from (path, size, time of change), the frames taken from them and the flats' ring current. Reducing a scan's
    1200 frames takes most of a minute; reading the file a fraction of a second. cache_dir None: nothing is kept.
    (vc_tomography::calibration::reduced_darks_flats)"""
    return _core.reduced_darks_flats(scan, str(cache_dir or ""), log)


def at_resolution(image, reduce, chunk_rows):
    """An image of the detector (a dark, a flat, a weight: (..., rows, cols)) as RawReader(reduce=) gives the
    frames stored as JPEG 2000 in chunks of chunk_rows rows: `reduce` levels of the 9/7 wavelet's low band, each
    chunk of rows an image of its own, sample (j, k) at the stored pixel (j, k) 2^reduce. A flat reduced so
    divides a frame's fixed pattern out as the full-resolution flat does; a mean of 2^reduce pixels would not.
    A last chunk that reaches beyond the frame holds zeros there (as the detector software stores it), which
    change the frame's last two or three reduced rows, here as in the frames. On a Paris4 frame this differs
    from the reader's reduced frame by 0.3 counts rms (the rounding), at 1, 2 and 3 levels.
    (vc_tomography::calibration::at_resolution)"""
    x = np.asarray(image, np.float64)
    out = _core.at_resolution(np.ascontiguousarray(x.reshape((-1,) + x.shape[-2:])), int(reduce), int(chunk_rows))
    return out.reshape(x.shape[:-2] + out.shape[-2:])


def weight_map(flat_stacks, transition_width_vertical=15.0, sigma=10.0, crop_top=0, crop_bottom=0) -> np.ndarray:
    """prepare_weights_double: gaussian_filter(mean(all flats) / max, sigma) * vertical erf border.
    (vc_tomography::calibration::weight_map)"""
    return _core.weight_map([np.ascontiguousarray(st, np.float32) for st in flat_stacks], float(transition_width_vertical), float(sigma), int(crop_top), int(crop_bottom))


def apodisation_widths(n_cols, est_cor_main_offset, companion_dx, transition_width_horizontal=1000.0,
                       transition_width_borders=50.0):
    """(w_left, w_right) for (main, companion) as prepare_weights_double --is_tetra writes them.
    Main scan: 'pente' beyond the rotation axis (min(2 tw, distance axis->edge)) on the axis side and a
    border taper towards the companion; companion: border taper on the side facing the main scan.
    companion_dx below nothing: the companion on the left, the axis in the main scan's right half (the reference's);
    above: on the right, the axis in its left half, the same widths mirrored.
    (vc_tomography::calibration::apodisation_widths)"""
    try:
        return _core.apodisation_widths(int(n_cols), float(est_cor_main_offset), float(companion_dx), float(transition_width_horizontal), float(transition_width_borders))
    except RuntimeError as e:
        raise NotImplementedError(str(e)) from None


def placement_from_vectors(vectors, n_cols):
    """The placement of every scan in the merged frame (nabuxx check_companion for a pair): `vectors` holds (rows,
    columns) of every scan after the first, from the first -> (ScanPlacements, merged width, anchor).
    (vc_tomography::calibration::placement_from_vectors)"""
    return _core.placement_from_vectors([tuple(float(x) for x in v) for v in vectors], int(n_cols))


def acquisition_fingerprint(main, companion) -> str:
    """sha256 over everything the calibration depends on: calibration version, scan identities, metadata, frame
    geometry, trajectories and the referenced raw files (path, size, mtime). A replaced acquisition (even under the
    same name) or a different pair invalidates the cache. (vc_tomography::calibration::acquisition_fingerprint)"""
    return _core.acquisition_fingerprint(main, companion)


def calibrate(scans, layout, cache_dir=None, log=print, progress=None, projection_store="", reduce=0):
    """The intensity calibration of the scans, the main scan (the rotation axis in its field) first, in the merged
    frame of `layout` (geometry.merged_layout: cor_main_detector, vectors = [(rows, columns), ..] from the main scan
    to every scan after it, and what it measured). The other scans lie beside the main scan, all on one side of it,
    or at the columns of one of those at another height (a grid of circular scans). Returns (Calibration, arrays
    dict: main_*, companion_*, companion2_*, .., weight_map). Calibration.lit_rows: the rows the beam lights in
    every scan, which the weights are cropped to.
    cache_dir: the darks / flats reduction and the detector's blur field are cached there, keyed by
    acquisition_fingerprint(); a cache of another acquisition or calibration version is ignored and replaced.
    progress: progress bars for the reduction (default: whenever the log is print).
    (vc_tomography::calibration::calibrate_intensity)"""
    if progress is None:
        progress = log is print
    cal = _core.calibrate_intensity(list(scans), float(layout["cor_main_detector"]), [tuple(float(x) for x in v) for v in layout["vectors"]],
                                    str(cache_dir or ""), log, bool(progress), str(projection_store), int(reduce),
                                    float(layout.get("axis_first_scan", 0.0)), [tuple(float(x) for x in v) for v in layout.get("measured") or []],
                                    float(layout.get("quantum", 0.0)))
    return cal, cal.arrays()


def lit_rows(flat_less_dark, share=0.12):
    """The rows the beam lights, (first, end): around the brightest row of a flat less its dark (each row's mean over
    the columns), the rows that reach `share` of it. (vc_tomography::calibration::lit_rows)"""
    return _core.lit_rows(np.ascontiguousarray(flat_less_dark, np.float32), float(share))


BLUR_TILE = 128


def flat_noise_spectra(scan, dark, n_flats=200, tile=BLUR_TILE):
    """Noise power spectra of the scan's first flat series on a lattice of tiles: of the differences of consecutive
    flats, each less the dark and scaled to its mean. Returns (power (ny, nx, tile, tile / 2 + 1), tile origins ys, xs,
    flats used). (vc_tomography::calibration::flat_noise_spectra)"""
    return _core.flat_noise_spectra(scan, np.ascontiguousarray(dark, np.float32), int(n_flats), int(tile))


def measure_detector_blur(scan, dark, n_flats=200, log=print):
    """The detector's blur field of a scan, from the noise of its flats: a dict whose "field" is [] when it does not
    explain the noise spectra or reaches less than half a pixel. (vc_tomography::calibration::measure_detector_blur)"""
    return _core.measure_detector_blur(scan, np.ascontiguousarray(dark, np.float32), int(n_flats), log)
