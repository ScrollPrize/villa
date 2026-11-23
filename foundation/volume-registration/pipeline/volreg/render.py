"""The pictures a person judges, rendered once by a stage and stored, never on request.

A hold-out unit -- a patch, or one of its octants -- is drawn as the first runs' test page
drew it: the coarse block, and the fine volume through the affine the patch was measured
under, moved so the unit's centre lands on its own measurement. No fit enters. A blind
round's item is drawn through each option's transform. Each picture is three central
planes side by side (xy, xz, yz), from `review_server.py` of the first runs.

Grey levels: each side is mapped through two anchors of its own scan -- the gap-air peak
and the papyrus peak's 10 % upper edge -- onto the same two grey levels, so the overlay is
neutral where the scans agree. The anchors are measured on the histogram of all the pair's
blocks, without statistics over everything: bright debris (inclusions, often well above
papyrus) takes no part and saturates. The first runs stretched each picture to its own
0.5/99.5 % window; debris set that window, and the fine side read darker (a red cast of
+6 grey levels median, up to +34, on PHerc0814; under 1 with anchors). The anchor features
are the ones `investigations/sessions/20261022-1000-auto-intensity-normalization/edges.py`
found packing-invariant.

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
# Where the anchors land: gap air at 17.5 % of the u8 range, as the atlas exports place it;
# the papyrus 10 % edge at 190, leaving room above it before debris saturates.
AIR_GREY, PAPYRUS_GREY = 45.0, 190.0


def anchors(histogram):
    """Gap-air peak and papyrus 10 % upper edge of a u8 histogram, or None without both
    peaks. Smoothed (sigma 2 bins, 0 and 255 left out); peaks are local maxima above 5 % of
    the highest; air is the first, papyrus the highest after it; the edge is where the
    papyrus peak's upper flank falls to a tenth of its height, interpolated."""
    smooth = np.asarray(histogram, float).copy()
    smooth[0] = smooth[255] = 0
    smooth = ndimage.gaussian_filter1d(smooth, 2)
    peaks = [i for i in range(2, 253) if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]
             and smooth[i] > 0.05 * smooth.max()]
    if len(peaks) < 2:
        return None
    papyrus_guess = max(peaks[1:], key=lambda i: smooth[i])
    valley = peaks[0] + int(np.argmin(smooth[peaks[0]:papyrus_guess + 1]))
    air = int(np.argmax(smooth[1:valley])) + 1
    papyrus = valley + int(np.argmax(smooth[valley:254]))
    below = next((j for j in range(papyrus, 255) if smooth[j] < 0.1 * smooth[papyrus]), None)
    if below is None:
        return None
    step = (smooth[below - 1] - 0.1 * smooth[papyrus]) / (smooth[below - 1] - smooth[below])
    return {"air": float(air), "papyrus_mode": float(papyrus), "papyrus_e10": below - 1 + float(step)}


def pair_anchors(session):
    """Anchors of each side over every measured block of the session (coarse blocks, and the
    fine patches cached behind them), for the pictures of a whole pair."""
    histograms = {"coarse": np.zeros(256), "fine": np.zeros(256)}
    for patch in session.state["patches"]:
        if not patch["measurement"]:
            continue
        histograms["coarse"] += np.bincount(session.coarse_block(patch["id"]).ravel(), minlength=256)[:256]
        held = session._fine.get(patch["id"])
        if held is not None:
            histograms["fine"] += np.bincount(np.asarray(held[2]).ravel(), minlength=256)[:256]
    return {side: anchors(histogram) for side, histogram in histograms.items()}


def grey(values, anchor):
    """Values onto the shared grey levels through a side's anchors; the old per-picture
    stretch when its anchors could not be found."""
    if anchor is None:
        return session_overview.stretch(values)
    scale = (PAPYRUS_GREY - AIR_GREY) / (anchor["papyrus_e10"] - anchor["air"])
    return np.clip(AIR_GREY + (np.asarray(values, np.float32) - anchor["air"]) * scale,
                   0, 255).astype(np.uint8)


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


def strip_of(block, cuts, anchor):
    return (np.zeros_like(planes(block)) if cuts is None
            else lay_out(list(grey(np.stack(cuts), anchor))))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def holdout_unit(session, out, patch, unit_id, centre_zyx, fine_zyx, grey_anchors,
                 corner=(0, 0, 0), size=dense.BLOCK):
    """Render one hold-out unit at its own measurement; returns the two pictures' digests."""
    measurement = patch["measurement"]
    block = session.coarse_block(patch["id"])
    block = block[tuple(slice(start, start + size) for start in corner)]
    affine = np.linalg.inv(session.fine_to_coarse(measurement["under"]))
    offset = np.asarray(fine_zyx, float) - dense.apply(affine, np.asarray(centre_zyx, float))
    cuts = fine_planes(session, patch, lambda points: dense.apply(affine, points) + offset,
                       corner=corner, size=size)
    folder = out / unit_id
    return {"coarse": write(folder / "coarse.png", png(planes(grey(block, grey_anchors["coarse"])))),
            "own": write(folder / "own.png", png(strip_of(block, cuts, grey_anchors["fine"])))}


def section(session, out, patch):
    """The coarse cross-section through a patch's middle, from the stored overview, with
    what a page needs to box the patch on it."""
    z = patch["origin_zyx"][0] + dense.BLOCK / 2
    image, info = session_overview.overview_section(session, z)
    write(out / patch["id"] / "section.png", png(image))
    (out / patch["id"] / "section.json").write_text(json.dumps(info))


def blind_item(session, out, patch, options, grey_anchors):
    """Render one blind item: the coarse block and the fine side through each option's
    first candidate (all of an option's candidates place the patch alike)."""
    block = session.coarse_block(patch["id"])
    digests = {"coarse": write(out / "coarse.png", png(planes(grey(block, grey_anchors["coarse"]))))}
    for index, option in enumerate(options):
        cuts = fine_planes(session, patch, session.candidate_map(option[0]))
        digests[str(index)] = write(out / f"{index}.png", png(strip_of(block, cuts, grey_anchors["fine"])))
    return digests
