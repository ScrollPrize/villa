"""The pictures a person judges, rendered once by a stage and stored, never on request.

A hold-out unit -- a patch, or one of its octants -- is drawn as the first runs' test page
drew it: the coarse block, and the fine volume through the affine the patch was measured
under, moved so the unit's centre lands on its own measurement. No fit enters. A blind
round's item is drawn through each option's transform. Each picture is three central
planes side by side (xy, xz, yz), from `review_server.py` of the first runs.

    work/images/holdout/<unit>/coarse.png, own.png
    work/images/holdout/<patch>/section.png, section.json
    work/images/blind/<round>/<item>/coarse.png, <option>.png
"""

import hashlib
import io
import json

import numpy as np
from PIL import Image
from scipy import ndimage

from .cores import dense_correlation as dense
from .cores import session_overview

# Space between the three planes of a strip, in pixels.
GAP = 4


def png(image):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def planes(block):
    """xy, xz and yz through the middle of a ZYX block, side by side."""
    middle = block.shape[0] // 2
    return lay_out([block[middle], block[:, middle, :], block[:, :, middle]])


def lay_out(cuts):
    height = max(cut.shape[0] for cut in cuts)
    width = sum(cut.shape[1] for cut in cuts) + GAP * (len(cuts) - 1)
    strip = np.zeros((height, width), np.uint8)
    column = 0
    for cut in cuts:
        strip[:cut.shape[0], column:column + cut.shape[1]] = cut
        column += cut.shape[1] + GAP
    return strip


def plane_points(origin, size):
    """Coarse level-0 points of the three central planes, in the order `planes` lays out."""
    middle = size // 2
    steps = np.arange(size, dtype=float)
    a, b = np.meshgrid(steps, steps, indexing="ij")
    flat = np.full(a.shape, float(middle))
    stacks = [np.stack([flat, a, b], -1), np.stack([a, flat, b], -1), np.stack([a, b, flat], -1)]
    return [np.asarray(origin, float) + stack.reshape(-1, 3) for stack in stacks]


def fine_planes(session, patch, to_fine, corner=(0, 0, 0), size=dense.BLOCK):
    """The fine volume on the three central planes of a patch (or its sub-block of `size`
    at `corner`) through the coarse-to-fine map `to_fine`; None if it falls outside."""
    _levels, fine_array, fine_scale = session.open()
    origin = np.asarray(patch["origin_zyx"]) + np.asarray(corner)
    lo, hi = dense.block_bounds(None, None, origin, size, to_fine, fine_scale)
    if (lo < 0).any() or (hi > np.array(fine_array.shape)).any():
        return None
    held = session._fine.get(patch["id"])
    if held is None or (held[0] > lo).any() or (hi > held[1]).any():
        read_lo = np.maximum(lo - dense.CACHE_PAD, 0)
        read_hi = np.minimum(hi + dense.CACHE_PAD, np.array(fine_array.shape))
        data = dense.try_read(fine_array, read_lo, read_hi)
        if data is None:
            return None
        held = (read_lo, read_hi, data)
        session._fine[patch["id"]] = held
    start, _end, data = held
    cuts = []
    for points in plane_points(origin, size):
        at = dense.level_coordinates(to_fine(points), fine_scale) - start
        values = ndimage.map_coordinates(data.astype(np.float32), at.T, order=1, mode="nearest")
        cuts.append(values.reshape(size, size))
    return cuts


def strip_of(block, cuts):
    return (np.zeros_like(planes(block)) if cuts is None
            else lay_out(list(session_overview.stretch(np.stack(cuts)))))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def holdout_unit(session, out, patch, unit_id, centre_zyx, fine_zyx, corner=(0, 0, 0),
                 size=dense.BLOCK):
    """Render one hold-out unit at its own measurement; returns the two pictures' digests."""
    measurement = patch["measurement"]
    block = session.coarse_block(patch["id"])
    block = block[tuple(slice(start, start + size) for start in corner)]
    affine = np.linalg.inv(session.fine_to_coarse(measurement["under"]))
    offset = np.asarray(fine_zyx, float) - dense.apply(affine, np.asarray(centre_zyx, float))
    cuts = fine_planes(session, patch, lambda points: dense.apply(affine, points) + offset,
                       corner=corner, size=size)
    folder = out / unit_id
    return {"coarse": write(folder / "coarse.png", png(planes(session_overview.stretch(block)))),
            "own": write(folder / "own.png", png(strip_of(block, cuts)))}


def section(session, out, patch):
    """The coarse cross-section through a patch's middle, from the stored overview, with
    what a page needs to box the patch on it."""
    z = patch["origin_zyx"][0] + dense.BLOCK / 2
    image, info = session_overview.overview_section(session, z)
    write(out / patch["id"] / "section.png", png(image))
    (out / patch["id"] / "section.json").write_text(json.dumps(info))


def blind_item(session, out, patch, options):
    """Render one blind item: the coarse block and the fine side through each option's
    first candidate (all of an option's candidates place the patch alike)."""
    block = session.coarse_block(patch["id"])
    digests = {"coarse": write(out / "coarse.png", png(planes(session_overview.stretch(block))))}
    for index, option in enumerate(options):
        cuts = fine_planes(session, patch, session.candidate_map(option[0]))
        digests[str(index)] = write(out / f"{index}.png", png(strip_of(block, cuts)))
    return digests
