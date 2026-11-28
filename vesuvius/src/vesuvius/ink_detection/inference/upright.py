"""Rotate a surface volume upright before flat ink inference, and rotate the prediction back.

Hand-made segments follow the standard orientation (grid rows run down the scroll axis, +z, so text lines are
horizontal). Automatically grown surfaces (for example ``vc_grow_seg_from_seed``) come out at an arbitrary
in-plane angle. Training augments with flips and 90-degree turns only, so a surface at, say, 40 degrees shows
the models text at an angle they never saw. This tool finds the angle from the surface's own ``z.tif``, rotates
every layer of the rendered surface volume so z runs straight down the rows, and maps a prediction made on the
rotated volume back onto the original canvas, so masks and overlays made for the original render still line up.

    python -m vesuvius.ink_detection.inference.upright angle SEGMENT.tifxyz
    python -m vesuvius.ink_detection.inference.upright rotate SEGMENT.tifxyz RENDER.zarr UPRIGHT.zarr
    python -m vesuvius.ink_detection.inference.infer UPRIGHT.zarr CHECKPOINT.pth PRED_UPRIGHT.tif
    python -m vesuvius.ink_detection.inference.upright back UPRIGHT.zarr PRED_UPRIGHT.tif PRED.tif

Only the in-plane rotation is changed. Mirrored grids stay mirrored (training already includes flips), and the
depth order of the layers is untouched, so ``--direction`` means the same on the rotated volume.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy import special
from vc_reader import images, tifxyz
from vc_reader.reader import ZarrWriter

from vesuvius.ink_detection.volume_io import open_volume, open_volume_root

ATTR = "upright"
TILE = 256


def read_z(segment: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """The z grid of a tifxyz segment and the vertices the renderer uses (``mask.tif`` applied, z > 0)."""
    surface = tifxyz.read(segment)
    z = np.asarray(surface.xyz[..., 2], dtype=np.float64)
    return z, np.asarray(surface.valid) & (z > 0)


def find_angle(z: np.ndarray, valid: np.ndarray, step_deg: float = 1.0) -> tuple[float, float, float]:
    """Angle (degrees, for ``scipy.ndimage.rotate``) that makes z increase straight down the grid rows.

    The z map is rotated with the same function that will rotate the render, so the sign convention and
    mirrored grids are handled by construction. Returns (angle, median dz per row, median |dz| per column),
    measured after rotation, in voxels per grid cell.
    """
    zf = np.where(valid, z, 0.0)
    vf = valid.astype(np.float64)
    best = None
    for angle in np.arange(-180.0, 180.0, step_deg):
        zr = ndi.rotate(zf, angle, reshape=True, order=1)
        vr = ndi.rotate(vf, angle, reshape=True, order=1) > 0.99
        gy, gx = np.gradient(np.where(vr, zr, np.nan))
        ok = np.isfinite(gy) & np.isfinite(gx)
        if ok.sum() < 50:
            continue
        down, side = float(np.median(gy[ok])), float(np.median(np.abs(gx[ok])))
        score = down - 2.0 * side
        if best is None or score > best[0]:
            best = (score, float(angle), down, side)
    if best is None:
        raise ValueError("too few valid grid points to estimate the surface orientation")
    return best[1], best[2], best[3]


def rotate_volume(segment: str | Path, render: str | Path, out: str | Path, resolution: str = "0") -> dict:
    """Write ``out`` (a bare zarr array, layers first) with every layer of ``render`` rotated upright."""
    src = open_volume(render, resolution, root_array_is_requested_level=True)
    if len(src.shape) != 3 or int(np.argmin(src.shape)) != 0:
        raise ValueError(f"rotate needs a layers-first render (layers, height, width), got shape {tuple(src.shape)}")
    layers, height, width = src.shape
    z, valid = read_z(segment)
    angle, down, side = find_angle(z, valid)
    first = ndi.rotate(np.asarray(src[0]), angle, reshape=True, order=1)
    dst = ZarrWriter.create(out, shape=(layers,) + first.shape, chunks=(layers, 128, 128),
                            dtype=src.dtype, overwrite=True, auth=None)
    dst[0] = first
    for k in range(1, layers):
        dst[k] = ndi.rotate(np.asarray(src[k]), angle, reshape=True, order=1)
    meta = {"angle_deg": angle, "orig_shape": [height, width], "rot_shape": list(first.shape),
            "dz_per_row": down, "abs_dz_per_col": side, "segment": str(segment), "render": str(render)}
    dst.attrs[ATTR] = meta
    return meta


def rotate_back(upright: str | Path, prediction: str | Path, out: str | Path) -> tuple[int, int]:
    """Map a prediction made on the upright volume back onto the original render canvas."""
    meta = open_volume_root(upright).attrs[ATTR]
    pred = np.squeeze(np.asarray(images.read(Path(prediction))))
    dtype = pred.dtype
    rh, rw = meta["rot_shape"]
    if pred.shape != (rh, rw):
        raise ValueError(f"prediction is {pred.shape}, expected the full upright canvas {(rh, rw)} (--roi is not supported)")
    h, w = meta["orig_shape"]
    # Inverse of the forward ndi.rotate about the pixel centers; cosdg/sindg keep quarter turns exact, as there.
    c, s = special.cosdg(meta["angle_deg"]), special.sindg(meta["angle_deg"])
    inv = np.array([[c, -s], [s, c]])
    offset = (np.array([rh, rw]) - 1) / 2 - inv @ ((np.array([h, w]) - 1) / 2)
    b = ndi.affine_transform(pred.astype(np.float32), inv, offset, output_shape=(h, w), order=1)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        b = np.clip(np.rint(b), info.min, info.max)
    b = b.astype(dtype)
    with images.TiffWriter(Path(out), (h, w), dtype, compression="lzw", tile=(TILE, TILE)) as writer:
        for y0 in range(0, h, TILE):
            for x0 in range(0, w, TILE):
                writer.write_tile(y0, x0, b[y0:y0 + TILE, x0:x0 + TILE])
    return h, w


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("angle", help="print the upright angle of a tifxyz segment")
    p.add_argument("segment")
    p = sub.add_parser("rotate", help="rotate a rendered surface volume upright")
    p.add_argument("segment"); p.add_argument("render"); p.add_argument("out")
    p.add_argument("--resolution", default="0", help="pyramid key when RENDER is a group (default 0)")
    p = sub.add_parser("back", help="rotate a prediction made on the upright volume back")
    p.add_argument("upright"); p.add_argument("prediction"); p.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "angle":
        angle, down, side = find_angle(*read_z(a.segment))
        print(f"angle {angle:+.0f} deg; after rotation z changes {down:+.2f} vx per row, {side:.2f} vx per column")
    elif a.cmd == "rotate":
        m = rotate_volume(a.segment, a.render, a.out, a.resolution)
        print(f"rotated by {m['angle_deg']:+.0f} deg: {m['orig_shape'][0]}x{m['orig_shape'][1]} -> "
              f"{m['rot_shape'][0]}x{m['rot_shape'][1]} (z per row {m['dz_per_row']:+.2f}, per column {m['abs_dz_per_col']:.2f})")
    else:
        h, w = rotate_back(a.upright, a.prediction, a.out)
        print(f"wrote {a.out} ({h}x{w})")


if __name__ == "__main__":
    main()
