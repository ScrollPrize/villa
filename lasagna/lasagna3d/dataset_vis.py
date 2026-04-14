"""Dataset visualization: render 3-plane slices with surface-intersection overlays.

For each sample from TifxyzLasagnaDataset we cut the three mid-planes of the
CT patch (axial/coronal/sagittal), overlay the intersection of every voxelized
surface mask via skimage.measure.find_contours, and save a single jpeg per
sample. Per-patch wraps are grouped into chains using the same neighbor-
ordering logic as `dataset_rowcol_cond._build_triplet_neighbor_lookup`.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

# Ensure lasagna/ dir is on sys.path so we can import sibling modules
_LASAGNA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LASAGNA_DIR not in sys.path:
    sys.path.insert(0, _LASAGNA_DIR)

from vesuvius.neural_tracing.datasets.common import (
    _read_volume_crop_from_patch,
)


TAG = "[lasagna3d dataset vis]"

# Qualitatively distinct colors for chains; cycles after.
_CHAIN_COLORS = [
    "#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de",
    "#5ac8fa", "#ffcc00", "#ff2d92", "#30d158", "#64d2ff",
]


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    """Percentile-normalize a 2D slice to uint8 for display."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


# Chain building is shared with the training dataset (module-level import
# happens lazily inside run_dataset_vis because sibling modules need sys.path
# set up first).


# ---------------------------------------------------------------------------
# Per-sample extraction (wrap_idx-indexed, bypasses dataset[idx] filter)
# ---------------------------------------------------------------------------

def _extract_sample(dataset, idx: int) -> dict:
    """Return image + {wrap_idx: mask} for a patch, preserving wrap indices."""
    patch = dataset.patches[idx]
    z0, _, y0, _, x0, _ = patch.world_bbox
    crop_size = tuple(int(v) for v in dataset.patch_size_zyx)
    min_corner = np.array([z0, y0, x0], dtype=np.int64)
    max_corner = min_corner + np.array(crop_size, dtype=np.int64)

    vol_crop = _read_volume_crop_from_patch(
        patch, crop_size=crop_size,
        min_corner=min_corner, max_corner=max_corner,
    )

    wrap_masks: dict = {}
    for wrap_idx, wrap in enumerate(patch.wraps[: dataset.max_surfaces_per_patch]):
        mask, _, _ = dataset._extract_and_voxelize_wrap(
            patch, wrap, min_corner, crop_size,
        )
        if np.any(mask > 0):
            wrap_masks[wrap_idx] = mask

    return {
        "image": np.asarray(vol_crop, dtype=np.float32),
        "wrap_masks": wrap_masks,
        "patch": patch,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_sample_figure(
    image: np.ndarray,
    wrap_masks: dict,
    chain_info: dict,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure

    Z, Y, X = image.shape
    cz, cy, cx = Z // 2, Y // 2, X // 2

    planes = [
        ("axial z=%d" % cz,
         image[cz, :, :],
         {wi: m[cz, :, :] for wi, m in wrap_masks.items()}),
        ("coronal y=%d" % cy,
         image[:, cy, :],
         {wi: m[:, cy, :] for wi, m in wrap_masks.items()}),
        ("sagittal x=%d" % cx,
         image[:, :, cx],
         {wi: m[:, :, cx] for wi, m in wrap_masks.items()}),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, (name, slc, mask_slices) in zip(axes, planes):
        ax.imshow(_normalize_image(slc), cmap="gray", interpolation="nearest")
        for wi, mslice in mask_slices.items():
            if mslice.size == 0 or not np.any(mslice > 0.5):
                continue
            info = chain_info.get(wi, {"chain": 0, "label": "?",
                                       "has_prev": False, "has_next": False})
            complete = bool(info.get("has_prev", False)) and bool(info.get("has_next", False))
            color = _CHAIN_COLORS[info["chain"] % len(_CHAIN_COLORS)]
            contours = measure.find_contours(mslice.astype(np.float32), 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0],
                        color=color, linewidth=0.9, alpha=0.9)

            ys_nz, xs_nz = np.nonzero(mslice > 0.5)
            if ys_nz.size == 0:
                continue
            cx_lbl = float(xs_nz.mean())
            cy_lbl = float(ys_nz.mean())
            ax.text(
                cx_lbl, cy_lbl, info["label"],
                color=color,
                fontsize=11 if complete else 7,
                fontweight="bold" if complete else "normal",
                ha="center", va="center",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.0),
            )
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, format="jpeg")
    plt.close(fig)


def _dataset_display_name(dataset_cfg: dict, fallback_idx: int) -> str:
    """Derive a short human-readable name for a dataset entry."""
    segments_path = dataset_cfg.get("segments_path")
    if segments_path:
        parent = Path(segments_path).parent.name
        if parent:
            return parent
    volume_path = dataset_cfg.get("volume_path") or dataset_cfg.get("__volume_path")
    if volume_path:
        return Path(str(volume_path).rstrip("/")).name or f"dataset{fallback_idx}"
    return f"dataset{fallback_idx}"


def run_dataset_vis(
    train_config: str,
    vis_dir: str,
    num_samples: int = 10,
    seed: int = 0,
    patch_size: int | None = None,
) -> None:
    from tifxyz_lasagna_dataset import TifxyzLasagnaDataset, build_patch_chains

    with open(train_config, "r") as f:
        config = json.load(f)
    if patch_size is not None:
        config["patch_size"] = patch_size

    out_dir = Path(vis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets_cfg = config.get("datasets", [])
    if not datasets_cfg:
        print(f"{TAG} config has no datasets", flush=True)
        return

    total_rendered = 0
    for ds_idx, ds_entry in enumerate(datasets_cfg):
        if ds_entry.get("volume_path") is None:
            print(f"{TAG} [{ds_idx}] skipping (volume_path is null)", flush=True)
            continue

        ds_name = _dataset_display_name(ds_entry, ds_idx)
        print(f"{TAG} [{ds_idx}] building dataset '{ds_name}'", flush=True)

        sub_config = dict(config)
        sub_config["datasets"] = [ds_entry]
        dataset = TifxyzLasagnaDataset(sub_config, apply_augmentation=False)
        n_total = len(dataset)
        if n_total == 0:
            print(f"{TAG} [{ds_idx}] '{ds_name}' has 0 patches, skipping",
                  flush=True)
            continue

        indices = list(range(n_total))
        random.Random(seed + ds_idx).shuffle(indices)
        indices = indices[: min(num_samples, n_total)]

        print(
            f"{TAG} [{ds_idx}] '{ds_name}': rendering {len(indices)} / {n_total}",
            flush=True,
        )

        for i, idx in enumerate(indices):
            sample = _extract_sample(dataset, idx)
            patch = sample["patch"]
            chain_info = build_patch_chains(patch, dataset.max_surfaces_per_patch)

            n_wraps = len(sample["wrap_masks"])
            n_chains = len({c["chain"] for c in chain_info.values()}) if chain_info else 0
            title = (
                f"{ds_name}  idx={idx}  wraps={n_wraps}  chains={n_chains}\n"
                f"bbox={patch.world_bbox}"
            )
            out_path = (
                out_dir / f"{ds_name}_sample{i:03d}_idx{idx:06d}.jpg"
            )
            _render_sample_figure(
                sample["image"], sample["wrap_masks"], chain_info,
                out_path, title=title,
            )
            print(f"{TAG}   [{i + 1}/{len(indices)}] {out_path.name}",
                  flush=True)
            total_rendered += 1

    print(f"{TAG} done — rendered {total_rendered} samples", flush=True)
