#!/usr/bin/env python3
"""Generate fiber pseudo-labels directly from a CT volume (no manual annotation required).

Addresses `ScrollPrize/villa#193 <https://github.com/ScrollPrize/villa/issues/193>`_,
which calls out that current fiber label generation is **entirely manual**. The
existing scripts in this directory (``fibers-dataset-generator.py``,
``hz-vt-generator.py``) voxelize WebKnossos ``.nml`` skeletons that a human has
already drawn. That creates a catch-22 for compressed / highly-curved regions
where manual annotation is hardest and the labels are most needed.

This script takes the opposite approach: it runs the Frangi-style vesselness
filter already in ``tools.py`` directly on a CT volume and produces fiber
pseudo-labels without any manual input. Quality is lower than skilled
annotation but the labels are immediately available everywhere CT is, which
makes them useful as

* expanded supervision (mix as soft targets during training),
* fiber overlays for VC3D / Crackle-Viewer review,
* a starting point for human refinement (annotators correct the worst cases).

Usage
-----

::

    python generate_fiber_labels_from_ct.py \\
        --input /path/to/scroll.zarr \\
        --output /path/to/fiber_labels.zarr \\
        --bbox 18176 18240 4128 4192 4128 4192 \\
        --threshold 0.5

The bbox is given as ``z0 z1 y0 y1 x0 x1`` and is inclusive of ``z0``,
exclusive of ``z1`` (NumPy convention). The output zarr is created if it
doesn't exist, matching the input dtype's spatial shape, and the bbox region
is overwritten with the binary fiber labels (uint8, 0 or 1).

Limitations
-----------

* Single bbox per invocation. For full-scroll passes, drive this script from a
  worklist of bboxes (one per region of interest).
* No block-overlap handling. A ``--margin`` flag pads the read window so the
  ``gaussian_filter`` inside ``detect_vesselness`` has context on every face;
  the margin is cropped off before writing. Pick ``margin >= 3 * gauss_sigma``
  (default ``gauss_sigma=2``, so ``margin=8`` is safe).
* Writes binary labels above the threshold. Use ``--write-probability`` to
  also emit the floating-point probability volume (separate output path).

The script intentionally has no nnUNet / training-framework dependency: it is
a pure CT → label transform.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow ``from tools import detect_vesselness`` when invoked as a script from
# this directory. The hyphen in the parent directory name (``fibers-dataset``)
# precludes a normal package import.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tools import detect_vesselness  # noqa: E402


def _load_zarr_window(input_path: str, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Read a (z1-z0, y1-y0, x1-x0) CT window and return float32 in [0, 1]."""
    import zarr

    store = zarr.open(input_path, mode="r")
    arr = store if isinstance(store, zarr.core.Array) else store["0"]

    z1 = min(z1, arr.shape[0])
    y1 = min(y1, arr.shape[1])
    x1 = min(x1, arr.shape[2])
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        raise ValueError(f"empty bbox: ({z0},{z1},{y0},{y1},{x0},{x1}) outside shape {arr.shape}")

    raw = arr[z0:z1, y0:y1, x0:x1]
    if raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    return raw.astype(np.float32)


def _ensure_output_zarr(output_path: str, shape: tuple[int, int, int], chunks: tuple[int, int, int] = (128, 128, 128)):
    """Open an existing output zarr or create one matching the input spatial shape (uint8)."""
    import zarr

    if os.path.exists(output_path):
        return zarr.open(output_path, mode="a")
    return zarr.open(
        output_path,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype="uint8",
        fill_value=0,
    )


def generate_fiber_labels_from_ct(
    input_path: str,
    output_path: str,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    threshold: float = 0.5,
    margin: int = 8,
    probability_output: str | None = None,
) -> dict:
    """Run detect_vesselness on the bbox + margin, threshold, and write to output zarr.

    Returns a dict of run metadata: voxel count, fiber fraction, elapsed seconds,
    and the actual bbox written (which may be clipped to the input shape).
    """
    import zarr

    in_store = zarr.open(input_path, mode="r")
    in_arr = in_store if isinstance(in_store, zarr.core.Array) else in_store["0"]
    full_shape = tuple(int(s) for s in in_arr.shape[:3])

    z1 = min(z1, full_shape[0])
    y1 = min(y1, full_shape[1])
    x1 = min(x1, full_shape[2])

    # Read with margin to give gaussian_filter / gradient enough context.
    rz0 = max(0, z0 - margin)
    ry0 = max(0, y0 - margin)
    rx0 = max(0, x0 - margin)
    rz1 = min(full_shape[0], z1 + margin)
    ry1 = min(full_shape[1], y1 + margin)
    rx1 = min(full_shape[2], x1 + margin)

    start = time.perf_counter()
    ct_window = _load_zarr_window(input_path, rz0, rz1, ry0, ry1, rx0, rx1)
    vesselness_full = np.asarray(detect_vesselness(ct_window), dtype=np.float32)

    # Crop out the margin so the written region matches the user's bbox.
    cz0, cy0, cx0 = z0 - rz0, y0 - ry0, x0 - rx0
    cz1, cy1, cx1 = cz0 + (z1 - z0), cy0 + (y1 - y0), cx0 + (x1 - x0)
    vesselness = vesselness_full[cz0:cz1, cy0:cy1, cx0:cx1]
    fiber_label = (vesselness >= threshold).astype(np.uint8)

    out_arr = _ensure_output_zarr(output_path, full_shape)
    out_arr[z0:z1, y0:y1, x0:x1] = fiber_label

    if probability_output is not None:
        prob_arr = _ensure_output_zarr(probability_output, full_shape)
        if prob_arr.dtype != np.float32:
            # Re-open with float32 dtype.
            prob_arr = zarr.open(
                probability_output,
                mode="w",
                shape=full_shape,
                chunks=(128, 128, 128),
                dtype="float32",
                fill_value=0.0,
            )
        prob_arr[z0:z1, y0:y1, x0:x1] = vesselness

    elapsed = time.perf_counter() - start
    return {
        "bbox": (z0, z1, y0, y1, x0, x1),
        "voxels": int(fiber_label.size),
        "fiber_voxels": int(fiber_label.sum()),
        "fiber_fraction": float(fiber_label.mean()),
        "elapsed_s": elapsed,
        "input_shape": full_shape,
        "threshold": threshold,
        "margin": margin,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Path to the input CT zarr (or zarr group with /0).")
    parser.add_argument("--output", required=True, help="Path to the output fiber-label zarr (uint8).")
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=6,
        required=True,
        metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
        help="Inclusive-exclusive bounding box in voxel coordinates.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Vesselness probability threshold (default 0.5).")
    parser.add_argument(
        "--margin",
        type=int,
        default=8,
        help="Voxel margin around the bbox for gaussian_filter context (default 8, ~3*gauss_sigma).",
    )
    parser.add_argument(
        "--write-probability",
        type=str,
        default=None,
        help="Optional output zarr path for the float32 vesselness probability volume.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    z0, z1, y0, y1, x0, x1 = args.bbox
    print(
        f"# generate_fiber_labels_from_ct: input={args.input} output={args.output} "
        f"bbox=({z0},{z1},{y0},{y1},{x0},{x1}) threshold={args.threshold} margin={args.margin}"
    )
    result = generate_fiber_labels_from_ct(
        args.input,
        args.output,
        z0,
        z1,
        y0,
        y1,
        x0,
        x1,
        threshold=args.threshold,
        margin=args.margin,
        probability_output=args.write_probability,
    )
    print(
        f"# OK voxels={result['voxels']} fiber_voxels={result['fiber_voxels']} "
        f"fiber_fraction={result['fiber_fraction']:.6f} elapsed={result['elapsed_s']:.2f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
