"""Build a Neuroglancer URL that overlays a trace polyline on the open-data zarr.

The PHercParis4 zarr is publicly accessible on S3 with OME-NGFF v0.4 metadata
(6 mip levels), so Neuroglancer can render it directly via the ``zarr://``
source scheme. This module turns a ``trace.npz`` (or any (N, 3) polyline of
volume-frame ``zyx`` voxel coordinates) into a Neuroglancer viewer state and
emits a ``https://neuroglancer-demo.appspot.com/#!<state>`` URL.

The state has two layers:

* an ``image`` layer pointing at the zarr,
* an ``annotation`` layer with the polyline encoded as inline line segments.

For long traces the encoded JSON state can exceed practical URL lengths. The
function supports two mitigations:

* ``--stride N`` keeps every Nth node (saves the rest), which keeps the trace
  shape but reduces segment count.
* If the URL would exceed ``--max-url-bytes`` (default 16 000, the
  conservative upper bound that fits in most browsers), the state is dumped
  to a sibling ``trace_neuroglancer_state.json`` and the function prints both
  the URL of an *empty* viewer and instructions for pasting the state via
  Neuroglancer's JSON state input.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ZARR_URL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
)
DEFAULT_NEUROGLANCER_BASE = "https://neuroglancer-demo.appspot.com"


def build_state(
    polyline_world_zyx: np.ndarray,
    *,
    zarr_url: str = DEFAULT_ZARR_URL,
    image_name: str = "PHercParis4",
    annotation_name: str = "autoreg_fiber_trace",
    annotation_color: str = "#ff8800",
    stride: int = 1,
    cross_section_scale: float = 8.0,
) -> dict[str, Any]:
    """Build a Neuroglancer viewer state dict for the image + polyline overlay.

    Coordinates are in voxels (the zarr's native frame). The polyline is encoded
    as line segments connecting consecutive nodes; ``stride > 1`` keeps every
    ``stride``-th node so the segment count stays manageable.
    """

    arr = np.asarray(polyline_world_zyx, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"polyline must have shape (N, 3); got {tuple(arr.shape)}")
    if arr.shape[0] < 2:
        raise ValueError("polyline must contain at least 2 nodes")
    if int(stride) < 1:
        raise ValueError("stride must be >= 1")
    sampled = arr[:: int(stride)]
    if sampled.shape[0] < 2:
        sampled = np.vstack([arr[0], arr[-1]])

    # Convert each zyx -> xyz (Neuroglancer's display order is x, y, z).
    points_xyz = sampled[:, ::-1].astype(np.float64)
    annotations: list[dict[str, Any]] = []
    for idx in range(int(points_xyz.shape[0]) - 1):
        annotations.append(
            {
                "type": "line",
                "id": f"seg_{idx}",
                "pointA": [float(v) for v in points_xyz[idx]],
                "pointB": [float(v) for v in points_xyz[idx + 1]],
            }
        )

    # Camera target: the polyline's centroid.
    centroid = points_xyz.mean(axis=0)
    # The PHercParis4 OME-NGFF has no units on its axes — Neuroglancer treats
    # the dimension space as voxel-indexed, so we use the same unitless space
    # for `position` and the inline annotations. (Setting dimensions to
    # micrometres would force a unit conversion that misplaces the polyline.)
    state: dict[str, Any] = {
        "dimensions": {
            "x": [1, ""],
            "y": [1, ""],
            "z": [1, ""],
        },
        "position": [float(v) for v in centroid],
        "crossSectionScale": float(cross_section_scale),
        "projectionScale": float(cross_section_scale) * 64.0,
        "layers": [
            {
                "type": "image",
                "source": f"zarr://{zarr_url}",
                "name": str(image_name),
                "tab": "rendering",
                "shader": "void main() { emitGrayscale(toNormalized(getDataValue())); }",
            },
            {
                "type": "annotation",
                "name": str(annotation_name),
                "annotationColor": str(annotation_color),
                "annotations": annotations,
            },
        ],
        "selectedLayer": {"layer": str(annotation_name), "visible": True},
        "layout": "xy-3d",
    }
    return state


def state_to_url(state: dict[str, Any], *, base_url: str = DEFAULT_NEUROGLANCER_BASE) -> str:
    """URL-encode a Neuroglancer state for the ``#!<state>`` fragment."""

    encoded = urllib.parse.quote(json.dumps(state, separators=(",", ":")), safe="")
    return f"{base_url}/#!{encoded}"


def emit(
    polyline_world_zyx: np.ndarray,
    out_dir: str | Path | None = None,
    *,
    zarr_url: str = DEFAULT_ZARR_URL,
    stride: int = 1,
    max_url_bytes: int = 16_000,
    annotation_name: str = "autoreg_fiber_trace",
) -> tuple[str, Path | None]:
    """Build the Neuroglancer state for ``polyline_world_zyx`` and return
    ``(url, state_json_path)``.

    The full state (with every annotation segment inline) is always written to
    ``out_dir/trace_neuroglancer_state.json`` when ``out_dir`` is given, so it
    can be pasted into Neuroglancer's JSON-state editor regardless of URL
    length. The returned URL embeds the state inline if it fits under
    ``max_url_bytes``; otherwise it points to an *image-only* viewer (so the
    user can open the volume and then paste the full state from disk).
    """

    state = build_state(
        polyline_world_zyx,
        zarr_url=zarr_url,
        stride=int(stride),
        annotation_name=annotation_name,
    )
    json_path: Path | None = None
    if out_dir is not None:
        target = Path(out_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "trace_neuroglancer_state.json"
        json_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    url = state_to_url(state)
    if len(url) > int(max_url_bytes):
        empty_state = dict(state)
        empty_state["layers"] = [state["layers"][0]]
        url = state_to_url(empty_state)
    return url, json_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Neuroglancer URL that overlays an autoreg_fiber trace on the "
            "public PHercParis4 zarr."
        ),
    )
    parser.add_argument("trace_npz", type=Path, help="Path to trace.npz (from vesuvius.trace_fiber).")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory to write trace_neuroglancer_state.json if the URL is too long.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth node along the polyline.")
    parser.add_argument(
        "--zarr-url",
        default=DEFAULT_ZARR_URL,
        help="Public zarr URL the image layer points at.",
    )
    parser.add_argument(
        "--max-url-bytes",
        type=int,
        default=16_000,
        help="URL length budget; if exceeded, state is dumped to JSON instead.",
    )
    parser.add_argument(
        "--annotation-name",
        default="autoreg_fiber_trace",
        help="Layer name for the polyline overlay.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    npz = np.load(args.trace_npz)
    if "polyline_world_zyx" not in npz.files:
        raise SystemExit(f"{args.trace_npz}: missing 'polyline_world_zyx' (this should be a trace.npz from vesuvius.trace_fiber)")
    polyline = np.asarray(npz["polyline_world_zyx"], dtype=np.float64)
    out_dir = args.out if args.out is not None else args.trace_npz.parent
    url, json_path = emit(
        polyline,
        out_dir=out_dir,
        zarr_url=args.zarr_url,
        stride=int(args.stride),
        max_url_bytes=int(args.max_url_bytes),
        annotation_name=str(args.annotation_name),
    )
    print(url)
    if json_path is not None:
        embedded = "fits in the URL above" if len(url) <= int(args.max_url_bytes) else "is too long to embed in a URL"
        print(
            f"\nState ({polyline.shape[0]} nodes, stride={int(args.stride)}) {embedded}; "
            f"the full state is also written to:\n  {json_path}\n\n"
            "To load the trace into Neuroglancer (works regardless of URL length):\n"
            "  1) Open the URL above.\n"
            "  2) In the Neuroglancer window, press the '{' '}' button on the top-right\n"
            "     toolbar to open the JSON state editor.\n"
            f"  3) Paste the contents of {json_path.name} and click 'Format'.\n"
            "  4) The trace appears as orange line segments overlaid on the volume.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
