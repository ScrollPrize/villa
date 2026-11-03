"""Where in the scroll a patch is: overview images of a session's coarse volume.

A reviewer judging a 128^3 patch needs to know where it sits -- near an end, at the edge
of the field of view, out in the wrapping -- and does not need a streaming viewer for that.
Two projections along the scroll answer "where along it", and a cross-section at the
patch's height answers "where in the winding", which a projection along the scroll axis
cannot: it smears every wrap into one disc.

All of it comes from one read of a low pyramid level of the coarse volume, the same read
`dense_correlation.tissue_map` makes, sized so the longest axis is about `OVERVIEW_PIXELS`
across. That level is kept in the session, so a cross-section at overview resolution costs
nothing; a sharper one is read from a finer level on demand.

    <session>/overview/overview.json   the level, and how a pixel maps to coarse voxels
    <session>/overview/volume.npy      the level itself, uint8
    <session>/overview/{xz,yz}-{max,mean}.png

Image axes, top to bottom then left to right: `xz` is (z, x), `yz` is (z, y), and a
cross-section is (y, x). A pixel `i` of a level `s` times coarser is centred on level-0
voxel `s * i + (s - 1) / 2`, the half-voxel convention the pyramid levels are built with.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import dense_correlation as dense

# Roughly how many pixels the longest axis of an overview should have. Enough to place a
# patch along a scroll at a glance; a whole scroll at that size is a few megabytes.
OVERVIEW_PIXELS = 1024
# The same for a sharper cross-section, read on demand.
SECTION_PIXELS = 768

# Contrast is stretched between these percentiles of the voxels that hold anything. The
# masked volumes are exactly zero outside the object, and including those zeros would put
# every percentile at zero.
LOW, HIGH = 1.0, 99.5

AXES = {"z": 0, "y": 1, "x": 2}


def overview_dir(session):
    return Path(session.directory) / "overview"


def stretch(image, reference=None):
    """`image` as uint8, windowed on the non-zero values of `reference` (or itself)."""
    reference = image if reference is None else reference
    values = reference[reference > 0]
    if values.size == 0:
        return np.zeros(image.shape, np.uint8)
    low, high = np.percentile(values, [LOW, HIGH])
    scaled = (image.astype(np.float32) - low) / max(float(high - low), 1e-6)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def projections(volume):
    """Max and mean projections along y and along x, of a ZYX uint8 volume.

    The mean is over the voxels that hold anything, so a thin part of the object does not
    read as dark just because most of its column is air.
    """
    out = {}
    for name, axis in (("xz", 1), ("yz", 2)):
        out[f"{name}-max"] = stretch(volume.max(axis=axis))
        count = (volume > 0).sum(axis=axis)
        total = volume.sum(axis=axis, dtype=np.float64)
        mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
        out[f"{name}-mean"] = stretch(mean)
    return out


def build(session, pixels=OVERVIEW_PIXELS):
    """Read the overview level once and write the projections and their geometry."""
    coarse_levels, _fine_array, _scale = session.open()
    shape = np.asarray(coarse_levels[0][0].shape)
    where = dense.tissue_map(coarse_levels, np.zeros(3), shape - 1,
                             want=float(shape.max()) / pixels)
    if where is None:
        raise ValueError("the coarse store has no pyramid level to build an overview from")
    volume = np.asarray(where.data, np.uint8)

    directory = overview_dir(session)
    directory.mkdir(exist_ok=True)
    np.save(directory / "volume.npy", volume)
    images = projections(volume)
    for name, image in images.items():
        Image.fromarray(image).save(directory / f"{name}.png")
    geometry = {
        "scale": float(where.scale),
        "low_zyx": np.asarray(where.low, float).tolist(),
        "shape_zyx": list(volume.shape),
        "coarse_shape_zyx": shape.tolist(),
        "images": {name: f"{name}.png" for name in images},
        "axes": {"xz": ["z", "x"], "yz": ["z", "y"], "section": ["y", "x"]},
    }
    with open(directory / "overview.json", "w") as handle:
        json.dump(geometry, handle, indent=1)
    return geometry


def load(session):
    directory = overview_dir(session)
    with open(directory / "overview.json") as handle:
        geometry = json.load(handle)
    return geometry, np.load(directory / "volume.npy", mmap_mode="r")


def to_pixel(geometry, point_zyx):
    """A coarse level-0 point as fractional overview-level indices (z, y, x)."""
    scale = geometry["scale"]
    level = np.asarray(geometry["low_zyx"], float) / scale
    return (np.asarray(point_zyx, float) + 0.5) / scale - 0.5 - level


def to_voxel(geometry, pixel_zyx):
    """The reverse of `to_pixel`."""
    scale = geometry["scale"]
    level = np.asarray(geometry["low_zyx"], float) / scale
    return (np.asarray(pixel_zyx, float) + level + 0.5) * scale - 0.5


def overview_section(session, z):
    """The cross-section at coarse level-0 height `z`, from the stored overview level."""
    geometry, volume = load(session)
    row = int(round(to_pixel(geometry, [z, 0, 0])[0]))
    row = min(max(row, 0), volume.shape[0] - 1)
    plane = np.asarray(volume[row])
    return stretch(plane), {"scale": geometry["scale"], "low_yx": geometry["low_zyx"][1:],
                            "z": float(to_voxel(geometry, [row, 0, 0])[0])}


def section_level(levels, pixels=SECTION_PIXELS):
    """The finest level whose cross-section is at most about `pixels` across."""
    shape = np.asarray(levels[0][0].shape)
    want = float(max(shape[1], shape[2])) / pixels
    fitting = [level for level in levels if level[1] >= want]
    return (min(fitting, key=lambda level: level[1]) if fitting
            else max(levels, key=lambda level: level[1]))


def detailed_section(session, z, pixels=SECTION_PIXELS):
    """A sharper cross-section at height `z`, read from a finer level on demand.

    Reading one plane costs a whole row of chunks, 128 planes deep, so this is for the
    patch a person is looking at, not for every patch at once. Windowed on the overview's
    contrast so the two read alike.
    """
    coarse_levels, _fine_array, _scale = session.open()
    array, scale = section_level(coarse_levels, pixels)
    index = int(round((z + 0.5) / scale - 0.5))
    index = min(max(index, 0), array.shape[0] - 1)
    plane = dense.try_read(array, (index, 0, 0), (index + 1, array.shape[1], array.shape[2]))
    if plane is None:
        raise ValueError(f"the plane at z {z} could not be read")
    _geometry, volume = load(session)
    return stretch(plane[0], np.asarray(volume[::4])), {
        "scale": float(scale), "low_yx": [0.0, 0.0],
        "z": float(scale * index + (scale - 1) / 2)}
