"""A mosaic: one tile reconstructed alone, or all tiles as one slice.

A mosaic is scans of one sample at several places on the rotation stage (tiles), each a helix with the rotation
axis inside the detector. `tile_reconstructor` makes a vc_tomography reconstructor of one tile;
`canvas_reconstruction` reconstructs the whole mosaic as one slice by full-width filtered backprojection, from a geometry
calibration that holds the tiles' places.

`cal` is a geometry calibration of the tiles (`geometry.calibrate_geometry`, `vc_tomography_calibration geometry`:
"model": "joint") and `label` the tile as that calibration's layout names it: the tile is reconstructed under its
own parameters (axis column and its slope, axial scale, the roll, the rotation stage's motion within a turn, the
map, the rows' timing), its place, height and rotation against the anchor left out, and the slice h holds the
model's height z = h + row_origin of the sample as the tile holds it. The pixel at (x, y) from the slice's centre
is the model's (x, -y) from the tile's rotation axis.

The frames may be a partial copy (only the frames named are read) and stored as JPEG 2000; they can be read at a
half, a quarter or an eighth (`reduce`).
"""
from __future__ import annotations


def tile_reconstructor(master, cal, label, device="cuda", params=None, row_origin=None, weight=None, cache_gb=0.0, reduced=None, log=print, cache_dir=None, reduce=0, projections=None):
    """A reconstructor (vc_tomography SliceReconstructor) of one tile alone, on a geometry calibration `cal` of the tiles
    ("model": "joint"; label: the tile's name in its layout): the detector map, the pose and the angle of every frame,
    the rows' timing. The tile's own frame: the slice's centre is the tile's rotation axis; only the helix frames on
    disk are used. The first step towards reconstructing a mosaic; the intensity side is the pair pipeline's (the scan's own
    darks and flats by the ring current, a weight map from the flats with a taper at the detector's far edge and
    across the axis), and nothing knows yet that every ray passes through sample the tile does not see.

    params: reconstruction parameters over the defaults here (fbp, the ramp filter alone).
    row_origin: moves the slices' numbering (rows); None numbers the slices from the record's height of this tile's middle
    frame on disk (the slice (rows - 1) / 2 is then about what the middle row saw there). Tiles that are to be
    compared slice by slice must be given one value (the result's "row_origin" of the first).
    weight: a function (weight map (rows, cols)) -> weight map, to reconstruct from part of the detector.
    reduced: the "reduced" of an earlier call for this tile (its darks and flats as reduced then).
    cache_dir: the reduced darks and flats are kept there (calibration.reduced_darks_flats): reducing them is
    46 s of a tile's start, reading them back 0.3 s.
    reduce: 1, 2 or 3 reconstructs at a half, a quarter, an eighth: the frames are read at the codestream's own
    lower resolution (RawReader(reduce=): decoding 3.4, 12 and 44 times faster), the darks and flats taken
    through the same low-pass (calibration.at_resolution), and the whole geometry divided by 2^reduce, nothing
    added: slice h of a reduced reconstruction is slice h 2^reduce of the full one, and its pixel at (x, y) from
    the axis is the full one's at (x, y) 2^reduce from the axis. The result's rot_center, where the axis lies in
    the slice, is in reduced pixels. row_origin stays in stored rows.
    projections: the tile's group of a projection store (STORE.zarr/<tile>, made by vc_tomography_projections_zarr): the
    helix frames are read from its level `reduce` instead of the detector's files (values within one count of those). Darks
    and flats are read from the detector's files where those are on disk, and otherwise from the store's level.
    Returns (reconstructor, dict(rot_center, frames, scan, terms, row_origin, reduced, reduce, params)).
    (vc_tomography::tile_reconstructor, vc_tomography/reconstruction.hpp)"""
    import json

    from . import _core
    return _core.tile_reconstructor(str(master), cal if isinstance(cal, str) else json.dumps(cal, default=float), str(label), device,
                                    json.dumps(params or {}, default=float), row_origin, weight, float(cache_gb), reduced, str(cache_dir or ""),
                                    int(reduce), log, str(projections or ""))     # vc_tomography::tile_reconstructor


def canvas_reconstruction(projections, cal, device="cuda", params=None, reduce=0, projection_store=None, cache_dir=None, taper=200, levels=True, keep_tiles=False, log=print, rebin_angles="exposure"):
    """Reconstruct the whole mosaic with full-width filtered backprojection.
    Each tile's pipeline supplies measured rows in calibrated coordinates. The default uses all tiles
    for coarse structure and weights their fine detail by the exposure travel at each region's centre.

    cal: the joint geometry calibration of the mosaic's tiles (vc_tomography_calibration geometry on all of them:
    every scan's d and height against the anchor; a file with every place at nothing is refused), as text, a dict
    or a path.
    projections: the folder of the tiles' scan folders (a master is taken from there when the calibration names it
    elsewhere). params: reconstruction parameters (filters, backprojector="fbp(...)", z_interp, ...). reduce,
    projection_store: frames at a half, a quarter or an eighth, from a projection store. taper: stored columns at either
    end of a tile's rows over which its weight falls to nothing. levels: one additive level per tile and slice from the
    rays two tiles share. keep_tiles: every tile's reconstructor kept between chunks (faster; each holds device memory).

    Each tile uses angle_interp="nearest" even if params specifies "views". Repeated turns contribute to
    one row per angular bin before the canvas receives them. Standalone FBP can keep the turns separate.
    rebin_angles: interpolate measured rows and weights at fixed canvas column onto a common angular grid
    before backprojection. False selects the direct folded blend. True uses two linear passes. "linear" or "lanczos3" use one angular interpolation
    with the calibrated in-plane fan when beam_divergence is enabled. "measured" aligns tiles
    to the measured mean angle of each view and backprojects both halves of the turn. The default
    "exposure" uses this grid and preferentially weights views with short exposure travel for
    fine detail in each tile disc. It uses ordinary measured-angle FBP when gates are unavailable.

    Returns the reconstruction: .doable (slices [lo, hi)), .size (canvas pixels a side), .pixel_size_m, .info() (JSON
    text: the tiles' places on the canvas among it), .reconstruct(h_lo, h_hi, roi=None, debug=False) -> (n, ny, nx)
    float32 in 1 / cm with pixel (row, col) = (half + Y, half + X), and of the last chunk .levels and .angle_spread."""
    import json
    import os

    from . import _core
    if isinstance(cal, dict):
        text = json.dumps(cal, default=float)
    elif isinstance(cal, (str, os.PathLike)) and os.path.exists(cal):
        with open(cal) as f:
            text = f.read()
    else:
        text = str(cal)
    return _core.MosaicReconstruction(str(projections), text, device, json.dumps(params or {}, default=float), int(reduce), str(projection_store or ""), str(cache_dir or ""),
                                      int(taper), bool(levels), bool(keep_tiles), log, bool(rebin_angles),
                                      rebin_angles if isinstance(rebin_angles, str) else "two_pass")
