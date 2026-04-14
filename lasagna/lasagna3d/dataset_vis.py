"""Dataset visualization: render 3-plane slices with supervision signals.

For each sample from ``TifxyzLasagnaDataset`` we cut the three mid-planes of
the CT patch (axial/coronal/sagittal) and stack several rows into a single
JPEG so training inputs, labels, and loss masks can be eyeballed together:

    row 1  CT + per-surface contours + chain labels
    row 2  CT + projected normal arrows
    row 3  cos supervision signal (masked by validity)
    row 4  grad_mag supervision signal (masked, auto-ranged)
    row 5  validity mask at full scale
    row 6  validity mask at scale 1 (scale-space pooled, erosion of invalid)
    row 7  validity mask at scale 2

Rows 3–7 require CUDA because ``compute_patch_labels`` uses CuPy EDT. If no
CUDA device is visible, only rows 1–2 are drawn.

Per-patch wraps are grouped into chains using
``tifxyz_lasagna_dataset.build_patch_chains`` (the same ordering fed to the
training loop), so the colours/labels you see here match what the network
actually sees.
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
    _trim_to_world_bbox,
    _upsample_world_triplet,
    voxelize_surface_grid_masked,
)


TAG = "[lasagna3d dataset vis]"

# Qualitatively distinct colors for chains; cycles after.
_CHAIN_COLORS = [
    "#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de",
    "#5ac8fa", "#ffcc00", "#ff2d92", "#30d158", "#64d2ff",
]

# Matches ScaleSpaceLoss3D default in train_tifxyz.py
_NUM_SCALES = 3
_ARROW_LEN_PX = 6.0
_MAX_ARROWS_PER_PLANE = 80


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    """Percentile-normalize a 2D slice to uint8 for display."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-wrap geometry extraction (mask + raw ZYX normals for arrows)
# ---------------------------------------------------------------------------

def _extract_wrap_geometry(wrap, world_bbox, min_corner, crop_size):
    """Return (mask, points_local_zyx, normals_local_zyx) for one wrap.

    This mirrors ``TifxyzLasagnaDataset._extract_and_voxelize_wrap`` but
    returns raw normals (not double-angle-encoded direction channels) so we
    can draw quivers. Returns ``None`` if the wrap doesn't produce anything
    usable inside the crop.
    """
    from tifxyz_lasagna_dataset import _estimate_grid_normals

    seg = wrap["segment"]
    r_min, r_max, c_min, c_max = wrap["bbox_2d"]
    seg_h, seg_w = seg._valid_mask.shape
    r_min = max(0, r_min)
    r_max = min(seg_h - 1, r_max)
    c_min = max(0, c_min)
    c_max = min(seg_w - 1, c_max)
    if r_max < r_min or c_max < c_min:
        return None

    seg.use_stored_resolution()
    scale_y, scale_x = seg._scale
    x_s, y_s, z_s, valid_s = seg[r_min:r_max + 1, c_min:c_max + 1]
    if x_s.size == 0:
        return None
    if valid_s is not None and not valid_s.all():
        return None
    try:
        x_full, y_full, z_full = _upsample_world_triplet(
            x_s, y_s, z_s, scale_y, scale_x,
        )
    except ValueError:
        return None

    trimmed = _trim_to_world_bbox(x_full, y_full, z_full, world_bbox)
    if trimmed is None:
        return None
    x_full, y_full, z_full = trimmed

    zyx_world = np.stack([z_full, y_full, x_full], axis=-1).astype(np.float32)
    min_corner_f = min_corner.astype(np.float32)
    zyx_local = zyx_world - min_corner_f

    crop = tuple(int(v) for v in crop_size)
    valid_grid = (
        np.isfinite(zyx_local).all(axis=-1)
        & (zyx_local[..., 0] >= -0.5) & (zyx_local[..., 0] < crop[0] - 0.5)
        & (zyx_local[..., 1] >= -0.5) & (zyx_local[..., 1] < crop[1] - 0.5)
        & (zyx_local[..., 2] >= -0.5) & (zyx_local[..., 2] < crop[2] - 0.5)
    )
    if not np.any(valid_grid):
        return None

    mask = voxelize_surface_grid_masked(zyx_local, crop, valid_grid)
    normals_full = _estimate_grid_normals(zyx_world)  # (H, W, 3) ZYX
    finite_normals = np.isfinite(normals_full).all(axis=-1)
    use = valid_grid & finite_normals
    if not np.any(use):
        return mask, np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    points_local = zyx_local[use].astype(np.float32)
    normals_local = normals_full[use].astype(np.float32)
    return mask, points_local, normals_local


# ---------------------------------------------------------------------------
# Per-sample extraction (wrap_idx-indexed, bypasses dataset[idx] filter)
# ---------------------------------------------------------------------------

def _extract_sample(dataset, idx: int) -> dict:
    """Read a patch and return image + per-wrap mask/points/normals (by wrap_idx)."""
    patch = dataset.patches[idx]
    z0, _, y0, _, x0, _ = patch.world_bbox
    crop_size = tuple(int(v) for v in dataset.patch_size_zyx)
    min_corner = np.array([z0, y0, x0], dtype=np.int64)
    max_corner = min_corner + np.array(crop_size, dtype=np.int64)

    vol_crop = _read_volume_crop_from_patch(
        patch, crop_size=crop_size,
        min_corner=min_corner, max_corner=max_corner,
    )

    wrap_data: dict = {}
    for wrap_idx, wrap in enumerate(patch.wraps[: dataset.max_surfaces_per_patch]):
        geom = _extract_wrap_geometry(wrap, patch.world_bbox, min_corner, crop_size)
        if geom is None:
            continue
        mask, points, normals = geom
        if not np.any(mask > 0):
            continue
        wrap_data[wrap_idx] = {
            "mask": mask,
            "points": points,
            "normals": normals,
        }

    return {
        "image": np.asarray(vol_crop, dtype=np.float32),
        "wrap_data": wrap_data,
        "patch": patch,
        "crop_size": crop_size,
    }


# ---------------------------------------------------------------------------
# Supervision computation (cos, grad_mag, validity + scale-space pyramid)
# ---------------------------------------------------------------------------

def _compute_supervision(
    image_shape: tuple,
    wrap_data: dict,
    chain_info_full: dict,
) -> dict | None:
    """Run ``compute_patch_labels`` on GPU to get cos, grad_mag, validity.

    Returns ``None`` when CUDA is unavailable (EDT backend is CuPy). Returns
    numpy arrays shaped ``(Z, Y, X)`` plus a scale-space pyramid of the
    validity mask matching ``ScaleSpaceLoss3D``'s pooling.
    """
    import torch

    if not torch.cuda.is_available():
        return None

    try:
        from tifxyz_labels import compute_patch_labels
    except Exception as exc:  # pragma: no cover - import guard
        print(f"{TAG} WARNING: tifxyz_labels import failed: {exc}", flush=True)
        return None

    device = torch.device("cuda")

    sorted_widx = sorted(wrap_data.keys())
    if not sorted_widx:
        return None

    surface_masks = [
        torch.as_tensor(wrap_data[wi]["mask"] > 0.5, device=device)
        for wi in sorted_widx
    ]

    surface_chain_info: list[dict] = []
    for wi in sorted_widx:
        entry = chain_info_full.get(wi)
        if entry is None:
            surface_chain_info.append(
                {"chain": -1, "pos": 0, "has_prev": False, "has_next": False},
            )
        else:
            surface_chain_info.append({
                "chain": int(entry["chain"]),
                "pos": int(entry["pos"]),
                "has_prev": bool(entry["has_prev"]),
                "has_next": bool(entry["has_next"]),
            })

    # Direction channels aren't needed for cos/grad_mag; pass zeros so the
    # direction loss channels don't distract the visualization.
    dir_ch = torch.zeros((6,) + image_shape, device=device)
    nv = torch.zeros(image_shape, dtype=torch.bool, device=device)

    try:
        result = compute_patch_labels(
            surface_masks=surface_masks,
            direction_channels=dir_ch,
            normals_valid=nv,
            surface_chain_info=surface_chain_info,
            device=device,
        )
    except Exception as exc:
        print(f"{TAG} WARNING: compute_patch_labels failed ({exc})", flush=True)
        return None

    targets = result["targets"].detach().cpu().numpy()       # (8, Z, Y, X)
    validity_np = result["validity"].detach().cpu().numpy().astype(np.float32)

    validity_pyramid = _validity_pyramid(validity_np, _NUM_SCALES)

    return {
        "cos": targets[0],
        "grad_mag": targets[1],
        "validity": validity_np,
        "validity_pyramid": validity_pyramid,
    }


def _validity_pyramid(validity: np.ndarray, num_scales: int) -> list[np.ndarray]:
    """Scale-space erosion of the validity mask — mirrors ScaleSpaceLoss3D.

    The loss pools by max over ``(1 - m)`` with kernel 2/stride 2 at each
    coarser level, which means a coarse voxel is valid only if *all* of its
    eight fine children were valid. We return the pyramid as full-res-sized
    arrays (upsampled nearest-neighbor) so panels align visually.
    """
    import torch
    import torch.nn.functional as F

    v = torch.as_tensor(validity, dtype=torch.float32)[None, None]  # (1,1,Z,Y,X)
    full = v
    out = [v[0, 0].numpy()]
    for _ in range(num_scales - 1):
        if v.shape[2] < 2 or v.shape[3] < 2 or v.shape[4] < 2:
            break
        invalid = 1.0 - v
        invalid_pooled = F.max_pool3d(invalid, kernel_size=2, stride=2)
        v = 1.0 - invalid_pooled
        up = F.interpolate(v, size=full.shape[2:], mode="nearest")
        out.append(up[0, 0].numpy())
    return out


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _auto_vmax(arr: np.ndarray, mask: np.ndarray | None, percentile: float = 99.0):
    """Percentile-based upper bound for display, ignoring masked-out voxels."""
    if mask is not None and np.any(mask):
        vals = arr[mask > 0]
    else:
        vals = arr
    if vals.size == 0:
        return 1.0
    vmax = float(np.percentile(vals, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(vals.max()) if vals.size else 1.0
    return max(vmax, 1e-6)


def _plane_slices(arr: np.ndarray, cz: int, cy: int, cx: int) -> list[np.ndarray]:
    return [arr[cz, :, :], arr[:, cy, :], arr[:, :, cx]]


def _draw_contours_and_labels(ax, mask_slices: dict, chain_info: dict):
    from skimage import measure
    for wi, mslice in mask_slices.items():
        if mslice.size == 0 or not np.any(mslice > 0.5):
            continue
        info = chain_info.get(wi, {
            "chain": 0, "label": "?", "has_prev": False, "has_next": False,
        })
        complete = bool(info.get("has_prev", False)) and bool(info.get("has_next", False))
        color = _CHAIN_COLORS[int(info.get("chain", 0)) % len(_CHAIN_COLORS)]
        for contour in measure.find_contours(mslice.astype(np.float32), 0.5):
            ax.plot(contour[:, 1], contour[:, 0],
                    color=color, linewidth=0.9, alpha=0.9)
        ys_nz, xs_nz = np.nonzero(mslice > 0.5)
        if ys_nz.size == 0:
            continue
        ax.text(
            float(xs_nz.mean()), float(ys_nz.mean()),
            info.get("label", "?"),
            color=color,
            fontsize=11 if complete else 7,
            fontweight="bold" if complete else "normal",
            ha="center", va="center",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.0),
        )


def _draw_normal_arrows(ax, wrap_data: dict, plane: str, plane_coord: int,
                        chain_info: dict, rng: np.random.Generator):
    """Project in-plane normals from surface points within ±0.5 of the slice."""
    for wi, wd in wrap_data.items():
        points = wd["points"]    # (N, 3) ZYX local
        normals = wd["normals"]  # (N, 3) ZYX
        if points.shape[0] == 0:
            continue

        if plane == "axial":
            coord_axis = 0
            hcol = 2  # x
            vcol = 1  # y
        elif plane == "coronal":
            coord_axis = 1
            hcol = 2  # x
            vcol = 0  # z
        else:  # sagittal
            coord_axis = 2
            hcol = 1  # y
            vcol = 0  # z

        m = np.abs(points[:, coord_axis] - plane_coord) < 0.6
        if not np.any(m):
            continue
        pts = points[m]
        nrm = normals[m]
        if pts.shape[0] > _MAX_ARROWS_PER_PLANE:
            pick = rng.choice(pts.shape[0], size=_MAX_ARROWS_PER_PLANE, replace=False)
            pts = pts[pick]
            nrm = nrm[pick]

        u = nrm[:, hcol]
        v = nrm[:, vcol]
        mag = np.sqrt(u * u + v * v) + 1e-8
        u = u / mag
        v = v / mag

        color = _CHAIN_COLORS[int(chain_info.get(wi, {}).get("chain", 0))
                              % len(_CHAIN_COLORS)]
        ax.quiver(
            pts[:, hcol], pts[:, vcol],
            u * _ARROW_LEN_PX, v * _ARROW_LEN_PX,
            angles="xy", scale_units="xy", scale=1.0,
            color=color, width=0.004, headwidth=3.0, headlength=4.0, alpha=0.9,
        )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def _render_sample_figure(
    image: np.ndarray,
    wrap_data: dict,
    chain_info: dict,
    supervision: dict | None,
    out_path: Path,
    title: str,
    arrow_seed: int = 0,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z, Y, X = image.shape
    cz, cy, cx = Z // 2, Y // 2, X // 2
    image_disp = [
        _normalize_image(image[cz, :, :]),
        _normalize_image(image[:, cy, :]),
        _normalize_image(image[:, :, cx]),
    ]
    plane_names = [f"axial z={cz}", f"coronal y={cy}", f"sagittal x={cx}"]
    plane_coords = [cz, cy, cx]
    plane_keys = ["axial", "coronal", "sagittal"]
    wrap_masks = {wi: wd["mask"] for wi, wd in wrap_data.items()}

    rows: list[tuple[str, callable]] = []

    # Row 1: CT + contours + chain labels
    def draw_row_contours(axes):
        for col, ax in enumerate(axes):
            ax.imshow(image_disp[col], cmap="gray", interpolation="nearest")
            mask_slices = {
                wi: _plane_slices(m, cz, cy, cx)[col] for wi, m in wrap_masks.items()
            }
            _draw_contours_and_labels(ax, mask_slices, chain_info)
            ax.set_title(plane_names[col], fontsize=9)
    rows.append(("CT + chain contours", draw_row_contours))

    # Row 2: CT + normal arrows
    def draw_row_arrows(axes):
        rng = np.random.default_rng(arrow_seed)
        for col, ax in enumerate(axes):
            ax.imshow(image_disp[col], cmap="gray", interpolation="nearest")
            _draw_normal_arrows(
                ax, wrap_data, plane_keys[col], plane_coords[col],
                chain_info, rng,
            )
            ax.set_title(plane_names[col], fontsize=9)
    rows.append(("CT + normals", draw_row_arrows))

    if supervision is not None:
        cos = supervision["cos"]
        grad_mag = supervision["grad_mag"]
        validity = supervision["validity"]
        pyramid = supervision["validity_pyramid"]
        cos_slices = _plane_slices(cos, cz, cy, cx)
        gm_slices = _plane_slices(grad_mag, cz, cy, cx)
        valid_slices_full = _plane_slices(validity, cz, cy, cx)
        gm_vmax = _auto_vmax(grad_mag, validity > 0.5, percentile=99.0)

        def draw_row_cos(axes):
            for col, ax in enumerate(axes):
                # Draw CT as faint background for context.
                ax.imshow(image_disp[col], cmap="gray", interpolation="nearest", alpha=0.35)
                disp = np.where(valid_slices_full[col] > 0.5, cos_slices[col], np.nan)
                ax.imshow(disp, cmap="twilight", interpolation="nearest",
                          vmin=0.0, vmax=1.0)
                ax.set_title(f"cos  {plane_names[col]}", fontsize=9)
        rows.append(("cos", draw_row_cos))

        def draw_row_gm(axes):
            for col, ax in enumerate(axes):
                ax.imshow(image_disp[col], cmap="gray", interpolation="nearest", alpha=0.35)
                disp = np.where(valid_slices_full[col] > 0.5, gm_slices[col], np.nan)
                ax.imshow(disp, cmap="viridis", interpolation="nearest",
                          vmin=0.0, vmax=gm_vmax)
                ax.set_title(f"grad_mag (vmax={gm_vmax:.3g})  {plane_names[col]}",
                             fontsize=9)
        rows.append(("grad_mag", draw_row_gm))

        for scale_idx, scale_vol in enumerate(pyramid):
            scale_slices = _plane_slices(scale_vol, cz, cy, cx)
            native_shape = validity.shape
            coarse_shape = tuple(s // (2 ** scale_idx) for s in native_shape)

            def _make_drawer(slices, scale_idx=scale_idx, coarse_shape=coarse_shape):
                def _draw(axes):
                    for col, ax in enumerate(axes):
                        ax.imshow(image_disp[col], cmap="gray",
                                  interpolation="nearest", alpha=0.35)
                        ax.imshow(slices[col], cmap="magma",
                                  interpolation="nearest", vmin=0.0, vmax=1.0)
                        ax.set_title(
                            f"validity s{scale_idx} "
                            f"(coarse {coarse_shape[0]}×{coarse_shape[1]}×{coarse_shape[2]}) "
                            f"{plane_names[col]}",
                            fontsize=8,
                        )
                return _draw
            rows.append((f"validity s{scale_idx}", _make_drawer(scale_slices)))
    else:
        rows.append((
            "supervision (skipped: no CUDA)",
            lambda axes: [ax.text(0.5, 0.5, "CUDA unavailable — EDT backend disabled",
                                  ha="center", va="center",
                                  transform=ax.transAxes, fontsize=9)
                          or ax.set_xticks([]) or ax.set_yticks([])
                          for ax in axes],
        ))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.0 * n_rows))
    if n_rows == 1:
        axes = np.asarray([axes])
    for row_idx, (_label, drawer) in enumerate(rows):
        drawer(axes[row_idx])
        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
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
            supervision = _compute_supervision(
                sample["image"].shape, sample["wrap_data"], chain_info,
            )

            n_wraps = len(sample["wrap_data"])
            n_chains = len({c["chain"] for c in chain_info.values()}) if chain_info else 0
            title = (
                f"{ds_name}  idx={idx}  wraps={n_wraps}  chains={n_chains}\n"
                f"bbox={patch.world_bbox}"
            )
            out_path = (
                out_dir / f"{ds_name}_sample{i:03d}_idx{idx:06d}.jpg"
            )
            _render_sample_figure(
                sample["image"], sample["wrap_data"], chain_info, supervision,
                out_path, title=title, arrow_seed=seed + idx,
            )
            print(f"{TAG}   [{i + 1}/{len(indices)}] {out_path.name}",
                  flush=True)
            total_rendered += 1

    print(f"{TAG} done — rendered {total_rendered} samples", flush=True)
