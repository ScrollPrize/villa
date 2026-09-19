"""Depth-aware, human-reviewed ink labeling. Suggestions are never ground truth.

Run directly to avoid importing the optional Vesuvius ML stack.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy import ndimage

MAX_VOXELS = 16_777_216


def load_volume(path):
    path = Path(path)
    if path.is_dir():
        root = zarr.open(str(path), mode="r")
        array = root if isinstance(root, zarr.Array) else root["0"]
        if len(array.shape) != 3 or np.prod(array.shape) > MAX_VOXELS:
            raise ValueError("Provide a bounded 3D crop (at most 16,777,216 voxels).")
        data = np.asarray(array[:])
    else:
        with tifffile.TiffFile(path) as tif:
            if (
                len(tif.series[0].shape) != 3
                or np.prod(tif.series[0].shape) > MAX_VOXELS
            ):
                raise ValueError("Provide a bounded grayscale 3D TIFF crop.")
            if tif.series[0].axes not in ("ZYX", "QYX", "IYX"):
                raise ValueError(
                    "TIFF must have ZYX grayscale axes, not channel/sample axes."
                )
            data = tif.asarray()
    if data.dtype.kind not in "buif" or not np.isfinite(data).all():
        raise ValueError("Volume must contain finite numeric values.")
    return data


def fingerprint(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def suggest(image, prior, *, center=32, radius=8, threshold=2.5, min_size=8):
    """Rank local bright features inside a 2D ink prior; no ink-specific classifier.

    Background is a per-layer Gaussian trend, sigma=3 pixels. Residuals are scaled
    by each layer's robust MAD. Use only a user-declared depth interval. The prior
    limits search in YX and is never extruded into confirmed training labels.
    """
    if image.ndim != 3 or prior.shape != image.shape:
        raise ValueError("Image and prior must have identical 3D shapes.")
    if not np.isfinite(image).all() or not np.isfinite(prior).all():
        raise ValueError("Inputs must be finite.")
    if not np.isin(prior, [0, 1, 255]).all():
        raise ValueError("Prior must be binary (0, 1, or 255).")
    if not 0 <= center < image.shape[0] or radius < 0:
        raise ValueError("Invalid center or radius.")
    if threshold <= 0 or not np.isfinite(threshold) or min_size < 1:
        raise ValueError("Threshold must be finite and positive; min_size >= 1.")
    start = time.perf_counter()
    f = image.astype(np.float32)
    residual = f - ndimage.gaussian_filter(f, sigma=(0, 3, 3), mode="reflect")
    med = np.median(residual, axis=(1, 2), keepdims=True)
    mad = np.median(np.abs(residual - med), axis=(1, 2), keepdims=True) * 1.4826
    score = np.divide(residual - med, mad, out=np.zeros_like(f), where=mad > 1e-6)
    flat = np.any(prior > 0, axis=0)
    lo, hi = max(0, center - radius), min(image.shape[0], center + radius + 1)
    candidate = np.zeros_like(prior, dtype=bool)
    candidate[lo:hi] = (score[lo:hi] >= threshold) & flat
    labels, _ = ndimage.label(
        candidate
    )  # 6-connected; corner contact is not continuity
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    candidate = keep[labels]
    labels, _ = ndimage.label(candidate)
    components = []
    for i, bbox in enumerate(ndimage.find_objects(labels), 1):
        local = labels[bbox] == i
        values = score[bbox][local]
        indices = np.argwhere(local) + np.array([s.start for s in bbox])
        seed = indices[np.argmax(values)].tolist()
        components.append(
            {
                "id": i,
                "voxels": int(local.sum()),
                "seed_zyx": seed,
                "bbox_zyx": [s.start for s in bbox] + [s.stop for s in bbox],
                "max_score": float(values.max()),
            }
        )
    components.sort(key=lambda c: (-c["max_score"], c["id"]))
    report = {
        "shape": list(image.shape),
        "image_sha256": fingerprint(image),
        "prior_sha256": fingerprint(prior),
        "parameters": {
            "center": center,
            "radius": radius,
            "threshold": threshold,
            "min_size": min_size,
        },
        "prior_active_layers": np.flatnonzero(np.any(prior > 0, axis=(1, 2))).tolist(),
        "prior_positive_pixels_yx": int(flat.sum()),
        "candidate_voxels": int(candidate.sum()),
        "search_support_voxels": int(flat.sum()) * (hi - lo),
        "components": components,
        "elapsed_seconds": time.perf_counter() - start,
        "confirmed_ink_voxels": 0,
        "limitation": "Intensity proposals can be fibers or artifacts. No measured ink accuracy or 3D ground truth.",
        "depth_counts": np.count_nonzero(candidate, axis=(1, 2)).tolist(),
    }
    return candidate.astype(np.uint8), report


def export_review(image, review, output):
    """Export explicit reviewed classes only. Unknown voxels remain unsupervised."""
    if review.get("schema") != "ink-depth-review-v1" or review.get(
        "image_sha256"
    ) != fingerprint(image):
        raise ValueError("Review does not match this image/schema.")
    if review.get("shape") != list(image.shape):
        raise ValueError("Review shape mismatch.")
    coordinate_system = review.get("coordinate_system")
    if coordinate_system != "surface_dyx":
        raise ValueError("This release supports surface_dyx coordinates only.")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("A reviewer name is required.")
    # 0 unknown, 1 ink, 2 reviewed background. Do not infer background from silence.
    classes = np.zeros(image.size, np.uint8)
    seen = set()
    for entry in review.get("voxels", []):
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError("Invalid voxel entry.")
        idx, value = entry
        if (
            type(idx) is not int
            or not 0 <= idx < image.size
            or type(value) is not int
            or value not in (1, 2)
        ):
            raise ValueError("Invalid voxel index or class.")
        if idx in seen:
            raise ValueError("Duplicate voxel index.")
        seen.add(idx)
        classes[idx] = value
    if not seen:
        raise ValueError("No reviewed voxels; refusing an empty training export.")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    classes = classes.reshape(image.shape)
    for name, a in [
        ("image", image),
        ("inklabels", (classes == 1).astype(np.uint8)),
        ("supervision_mask", (classes > 0).astype(np.uint8)),
    ]:
        tifffile.imwrite(
            out / f"{name}.tif", a, photometric="minisblack", metadata={"axes": "ZYX"}
        )
    metadata = {
        **review,
        "counts": {
            "ink": int((classes == 1).sum()),
            "background": int((classes == 2).sum()),
            "unknown": int((classes == 0).sum()),
        },
    }
    (out / "review.json").write_text(json.dumps(metadata, indent=2))
    return metadata["counts"]


def html_review(image, prior, candidate, report, coordinate_system, source_metadata):
    # Display-only uint8 conversion; exports always use original intensities.
    lo, hi = np.percentile(image, [1, 99])
    display = np.clip(
        (image.astype(float) - lo) * 255 / max(float(hi - lo), 1), 0, 255
    ).astype(np.uint8)
    payload = {
        **report,
        "coordinate_system": coordinate_system,
        "source_metadata": source_metadata,
        "image": base64.b64encode(display.tobytes()).decode(),
        "prior": base64.b64encode((prior > 0).astype(np.uint8).tobytes()).decode(),
        "candidate": base64.b64encode(candidate.tobytes()).decode(),
    }
    template = Path(__file__).with_name("review_template.html").read_text()
    return template.replace("__DATA__", json.dumps(payload).replace("</", "<\\/"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--image", required=True)
    prep.add_argument("--prior", required=True)
    prep.add_argument("--manifest", required=True, help="Source/crop provenance JSON")
    prep.add_argument("--output", required=True)
    prep.add_argument("--coordinate-system", required=True, choices=["surface_dyx"])
    prep.add_argument("--center", type=int, default=32)
    prep.add_argument("--radius", type=int, default=8)
    prep.add_argument("--threshold", type=float, default=2.5)
    prep.add_argument("--min-size", type=int, default=8)
    ex = sub.add_parser("export")
    ex.add_argument("--image", required=True)
    ex.add_argument("--review", required=True)
    ex.add_argument("--output", required=True)
    args = p.parse_args()
    image = load_volume(args.image)
    if args.command == "export":
        print(
            json.dumps(
                export_review(
                    image, json.loads(Path(args.review).read_text()), args.output
                ),
                indent=2,
            )
        )
        return
    source_metadata = json.loads(Path(args.manifest).read_text())
    if not isinstance(source_metadata, dict) or source_metadata.get("shape") != list(
        image.shape
    ):
        raise ValueError("Source manifest must identify this crop shape.")
    prior = load_volume(args.prior)
    candidate, report = suggest(
        image,
        prior,
        center=args.center,
        radius=args.radius,
        threshold=args.threshold,
        min_size=args.min_size,
    )
    report["coordinate_system"] = args.coordinate_system
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    tifffile.imwrite(
        out / "candidates.tif",
        candidate,
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    (out / "review.html").write_text(
        html_review(
            image, prior, candidate, report, args.coordinate_system, source_metadata
        )
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("components", "depth_counts")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
