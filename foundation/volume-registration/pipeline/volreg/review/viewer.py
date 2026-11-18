"""A Neuroglancer link to a hold-out unit: the two whole volumes overlaid where it sits.

The coarse volume is drawn green and the fine one magenta, blended additively, so tissue
that lines up reads white and a fibre that does not appears twice -- the picture a person
judged, in a viewer that can be scrolled beyond the patch. The output space is the coarse
volume's level-0 voxels, ZYX, as in `research/viewer_links.py` of the first runs.

The fine layer is placed as the unit's picture was: through the affine its patch was
measured under, moved so the unit's centre lands on its own measurement. A second layer,
off when the link opens, places it through the current fit's affine part. A layer transform
is a matrix, so a fit's bend cannot travel in the link.
"""

import json
from urllib.parse import quote

import numpy as np

from ..cores import dense_correlation as dense

VIEWER = "https://neuroglancer-demo.appspot.com"
# The open-data bucket is also served by the chunk rewriter, in the 32^3 chunks a viewer
# likes; the bucket's own URL is what the pipeline reads from.
REWRITES = {"https://vesuvius-challenge-open-data.s3.amazonaws.com/":
            "https://zarr.aws.ash2txt.org/open-data/"}
CONCURRENT_DOWNLOADS = 8
UNITLESS = {"z": [1, ""], "y": [1, ""], "x": [1, ""]}
GREEN = "void main() {\n  emitRGB(vec3(0, toNormalized(getDataValue()), 0));\n}\n"
MAGENTA = ("void main() {\n  emitRGB(vec3(toNormalized(getDataValue()), 0,"
           " toNormalized(getDataValue())));\n}\n")


def viewable(url):
    for prefix, served in REWRITES.items():
        if url.startswith(prefix):
            return served + url[len(prefix):]
    return url


def layer(name, url, shader, matrix=None, visible=True):
    transform = {"outputDimensions": UNITLESS}
    if matrix is not None:
        transform["matrix"] = np.asarray(matrix, float)[:3].round(9).tolist()
    return {"type": "image", "source": [{"url": f"zarr://{viewable(url)}", "transform": transform}],
            "shader": shader, "blend": "additive", "opacity": 1.0, "name": name,
            "visible": visible}


def unit_link(session, held, unit_id):
    """The link for a hold-out unit, `<patch>` or `<patch>.<octant>`, from the record's
    hold-out set `held` and the pair's session (for the fits' matrices)."""
    patch_id, _dot, octant = unit_id.partition(".")
    patch = next(p for p in held["patches"] if p["id"] == patch_id)
    entry = patch if not octant else next(o for o in patch["octants"] if str(o["index"]) == octant)
    centre = np.asarray(entry["centre_zyx"], float)
    # As `render.holdout_unit`: coarse -> fine is inv(M) c + offset, so fine -> coarse is
    # M (f - offset).
    under = patch["measured_under"]
    fine_to_coarse = session.fine_to_coarse(under)
    offset = np.asarray(entry["fine_zyx"], float) - dense.apply(np.linalg.inv(fine_to_coarse), centre)
    shift = np.eye(4)
    shift[:3, 3] = -offset
    coarse_side = "fixed" if session.fixed_is_coarse else "moving"
    fine_side = "moving" if session.fixed_is_coarse else "fixed"
    coarse, fine = session.state[coarse_side], session.state[fine_side]
    current = session.state["current"]
    layers = [
        layer(f"coarse {coarse['voxel_um']:g} um", coarse["url"], GREEN),
        layer(f"fine {fine['voxel_um']:g} um, as judged ({under} + its own shift)",
              fine["url"], MAGENTA, fine_to_coarse @ shift),
    ]
    if current and current != "start":
        layers.append(layer(f"fine {fine['voxel_um']:g} um, current fit {current} (affine part)",
                            fine["url"], MAGENTA, session.fine_to_coarse(current), visible=False))
    state = {"dimensions": UNITLESS, "position": centre.round(1).tolist(),
             "crossSectionScale": 1.0, "projectionScale": 512, "layers": layers,
             "selectedLayer": {"layer": layers[1]["name"]}, "layout": "4panel",
             "concurrentDownloads": CONCURRENT_DOWNLOADS}
    fragment = quote(json.dumps(state, separators=(",", ":")), safe="~@#$&()*!+=:;,.?/'")
    return f"{VIEWER}/#!{fragment}"
