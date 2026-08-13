#!/usr/bin/env python3
"""Export the lasagna winding volume to a standalone OME-Zarr file.

The winding volume (produced by labels_to_winding_volume.py) assigns a
fractional winding position psi(x) to each voxel.  floor(psi(x)) = winding
number; psi(x) mod 1 = phase within a winding.

This script extracts the winding volume from a lasagna zarr store and
re-exports it in a self-contained OME-Zarr with metadata, for use by
downstream tools:
  - C2b (psi_prescreening.py): pre-screen straddler tracks before spiral fitting
  - C2c (L_winding_consistency): differentiable loss via grid_sample
  - C3 (auto_winding_annotations.py): automated relative winding annotations

Usage:
    python export_winding_field.py \\
        --input /path/to/winding.zarr \\
        --output /path/to/winding_field.zarr

    python export_winding_field.py \\
        --input /path/to/winding.zarr \\
        --output /path/to/winding_field.zarr \\
        --verify  # produces psi_verify.png cross-section plot
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import zarr

log = logging.getLogger(__name__)


def export_winding_field(
    input_path: str,
    output_path: str,
    resolution: int | None = None,
) -> None:
    """Export a winding volume zarr to a self-contained OME-Zarr.

    Args:
        input_path: Path to the source winding.zarr (from labels_to_winding_volume.py).
        output_path: Path for the output OME-Zarr.
        resolution: Override for the resolution factor (voxels per cell).
            If None, reads from source attrs or defaults to 4.
    """
    src = zarr.open(str(input_path), mode="r")

    # The source zarr stores the winding volume as the root array or as
    # a named dataset.  Handle both conventions.
    if isinstance(src, zarr.Array):
        psi_np = np.asarray(src)
        src_attrs = dict(src.attrs)
    elif "winding_position" in src:
        psi_np = np.asarray(src["winding_position"])
        src_attrs = dict(src.attrs)
    else:
        # Root group with the array as the direct dataset
        keys = list(src.array_keys()) if hasattr(src, 'array_keys') else list(src.keys())
        if len(keys) == 1:
            psi_np = np.asarray(src[keys[0]])
            src_attrs = dict(src.attrs)
        elif len(keys) == 0:
            # Try reading as direct array
            psi_np = np.asarray(src)
            src_attrs = dict(src.attrs) if hasattr(src, 'attrs') else {}
        else:
            raise KeyError(
                f"Cannot determine winding volume array. "
                f"Available keys: {keys}. "
                f"Expected a single root array or 'winding_position' dataset."
            )

    psi_np = np.asarray(psi_np, dtype=np.float32)

    # Sanity checks
    psi_max = float(np.nanmax(psi_np))
    psi_min = float(np.nanmin(psi_np))
    log.info(
        "Source winding volume: shape=%s, dtype=%s, range=[%.2f, %.2f]",
        psi_np.shape, psi_np.dtype, psi_min, psi_max,
    )

    if psi_max < 1.0:
        log.warning(
            "psi max = %.3f -- expected > 5.0 for a real scroll. "
            "Check that the input is a winding volume, not a binary mask.",
            psi_max,
        )

    res = resolution or int(src_attrs.get("scaledown", src_attrs.get("resolution_factor", 4)))

    # Write output
    store = zarr.open(output_path, mode="w")
    store.create_dataset(
        "winding_position",
        data=psi_np,
        chunks=(64, 64, 64),
        compressor=zarr.Blosc(cname="lz4", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
        dtype=np.float32,
        overwrite=True,
    )
    store.attrs.update({
        "resolution_factor": res,
        "field_name": "fractional_winding_position",
        "units": "winding_fraction_monotone_outward",
        "zarr_shape_DHW": list(psi_np.shape),
        "ct_voxels_per_psi_cell": res,
        "source_path": str(input_path),
        "min_winding": float(psi_min),
        "max_winding": float(psi_max),
    })

    log.info(
        "Exported winding field -> %s  shape=%s  range=[%.2f, %.2f]  resolution=%d",
        output_path, psi_np.shape, psi_min, psi_max, res,
    )


def verify_winding_field(zarr_path: str, output_image: str = "psi_verify.png") -> None:
    """Generate a cross-section plot of the exported winding field.

    Produces a mid-Z cross-section colored by winding position.
    If the field shows concentric rings increasing outward, it's correct.
    If it shows noise or a flat field, the source data is wrong.

    Args:
        zarr_path: Path to the exported winding_field.zarr.
        output_image: Output image path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available -- skipping verification plot.")
        return

    z = zarr.open(zarr_path, mode="r")
    psi = z["winding_position"]
    mid = psi.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Mid-Z cross-section
    slice_z = np.asarray(psi[mid])
    im0 = axes[0].imshow(slice_z, cmap="viridis")
    axes[0].set_title(f"Mid-Z (z={mid}) cross-section")
    plt.colorbar(im0, ax=axes[0], label="psi (winding position)")

    # Mid-Y cross-section
    mid_y = psi.shape[1] // 2
    slice_y = np.asarray(psi[:, mid_y, :])
    im1 = axes[1].imshow(slice_y, cmap="viridis", aspect="auto")
    axes[1].set_title(f"Mid-Y (y={mid_y}) cross-section")
    plt.colorbar(im1, ax=axes[1], label="psi")

    # Histogram of non-zero values
    flat = np.asarray(psi[:]).ravel()
    nonzero = flat[flat > 0]
    if len(nonzero) > 0:
        axes[2].hist(nonzero, bins=100, color="steelblue", edgecolor="none")
    axes[2].set_title("Distribution of psi > 0")
    axes[2].set_xlabel("psi")
    axes[2].set_ylabel("count")

    psi_min = float(np.nanmin(psi[:]))
    psi_max = float(np.nanmax(psi[:]))
    fig.suptitle(
        f"Winding Field Verification | shape={list(psi.shape)} | "
        f"range=[{psi_min:.2f}, {psi_max:.2f}]",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()

    log.info("Verification plot saved to %s", output_image)
    log.info("PASS if image shows concentric rings, increasing outward")
    log.info("FAIL (wrong source data) if image is noise or flat")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(
        description="Export lasagna winding volume to standalone OME-Zarr."
    )
    p.add_argument("--input", required=True, help="Source winding.zarr path")
    p.add_argument("--output", required=True, help="Output winding_field.zarr path")
    p.add_argument("--resolution", type=int, default=None,
                   help="Resolution factor override (voxels per cell)")
    p.add_argument("--verify", action="store_true",
                   help="Generate psi_verify.png verification plot")
    args = p.parse_args()

    export_winding_field(args.input, args.output, args.resolution)

    if args.verify:
        verify_winding_field(args.output)
