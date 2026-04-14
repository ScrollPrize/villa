"""Dataset visualization for TifxyzLasagnaDataset.

This tool **does not** re-implement any part of the training pipeline. It
calls ``dataset[idx]`` and ``compute_batch_targets`` exactly the way the
training loop does, then renders the resulting tensors. That means:

- The voxelized surface masks you see are the tensors the UNet is fed.
- The normal arrows are drawn from the raw normals the dataset computed
  to produce ``direction_channels`` (exposed via ``include_geometry=True``).
- The per-surface chain labels are the ``surface_chain_info`` metadata the
  loss sees.
- cos / grad_mag / validity come from ``train_tifxyz.compute_batch_targets``.
- The validity scale-space pyramid uses
  ``tifxyz_labels.scale_space_validity_pyramid``, which is the same helper
  ``ScaleSpaceLoss3D`` calls per step during training.

If any of those change, this vis follows automatically — there is no
parallel copy to keep in sync.

Per JPEG layout (rows × 3 planes: axial / coronal / sagittal):

    row 1  CT + per-chain contours + chain labels
    row 2  CT + projected normal arrows (from raw normals)
    row 3  cos supervision signal, masked by validity
    row 4  grad_mag supervision signal, masked, auto-ranged
    row 5  validity mask at full scale (scale 0)
    row 6  validity mask at scale 1 (ScaleSpaceLoss3D pooling)
    row 7  validity mask at scale 2

Rows 3–7 require CUDA (``edt_torch`` uses CuPy). Without CUDA those rows
are replaced with a "no CUDA" placeholder.
"""
from __future__ import annotations

import concurrent.futures
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

# Import matplotlib at module level with Agg backend. Importing inside a
# render worker thread would race on the backend setup.
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from skimage import measure  # noqa: E402


TAG = "[lasagna3d dataset vis]"

# Qualitatively distinct colors — indexed per surface (not per chain) in
# rows 1–2 so individual wraps within the same chain are distinguishable.
_SURFACE_COLORS = [
    "#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de",
    "#5ac8fa", "#ffcc00", "#ff2d92", "#30d158", "#64d2ff",
    "#ff6b6b", "#51cf66", "#339af0", "#fcc419", "#cc5de8",
    "#22b8cf", "#fd7e14", "#e64980", "#94d82d", "#15aabf",
]


def _surface_color(sample_seed: int, surface_idx: int) -> str:
    """Deterministic per-sample, per-surface color.

    We shuffle the palette once per sample so surface positions get
    different colors across samples (avoids always-red-for-surface-0).
    """
    palette = list(_SURFACE_COLORS)
    rng = random.Random(sample_seed)
    rng.shuffle(palette)
    return palette[surface_idx % len(palette)]

# Matches ScaleSpaceLoss3D default num_scales in train_tifxyz.py
_NUM_SCALES = 3
_ARROW_LEN_PX = 18.0
_MAX_ARROWS_PER_PLANE = 25
_GRID_NORMAL_STRIDE = 12

# Plane → (channel pair indices) for direction_channels (6, Z, Y, X).
# Encoding from tifxyz_lasagna_dataset.compute_direction_values:
#   axial    z-slice : channels (0,1) encode (nx, ny)
#   coronal  y-slice : channels (2,3) encode (nx, nz)
#   sagittal x-slice : channels (4,5) encode (ny, nz)
_PLANE_DIR_CHANNELS = {
    "axial": (0, 1),
    "coronal": (2, 3),
    "sagittal": (4, 5),
}


def _decode_dir_pair(d0, d1):
    """Inverse of `tifxyz_lasagna_dataset._encode_dir_np`.

    Returns a 2D unit-direction `(h, v)` for the in-plane normal
    component encoded by the (d0, d1) channel pair.
    """
    cos2t = 2.0 * d0 - 1.0
    sin2t = cos2t - np.sqrt(2.0) * (2.0 * d1 - 1.0)
    theta = np.arctan2(sin2t, cos2t) / 2.0
    return np.cos(theta), np.sin(theta)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _normalize_image(arr: np.ndarray) -> np.ndarray:
    """Percentile-normalize a 2D slice to uint8 for display."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _auto_vmax(arr: np.ndarray, mask: np.ndarray | None,
               percentile: float = 99.0) -> float:
    """Percentile-based upper bound, ignoring masked-out voxels."""
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


# ---------------------------------------------------------------------------
# Training-pipeline call (dataset[idx] + compute_batch_targets)
# ---------------------------------------------------------------------------

def _sample_from_batch(batch: dict) -> dict:
    """Unpack a single-item collated batch back into a per-sample dict.

    Mirrors the keys `TifxyzLasagnaDataset.__getitem__` produces. Used
    when we feed the renderer from a `DataLoader` — the loader hands us
    a batch (so workers can run in parallel), and we peel element 0 back
    out for the render thread.
    """
    return {
        "image": batch["image"][0],
        "surface_masks": batch["surface_masks"][0],
        "direction_channels": batch["direction_channels"][0],
        "normals_valid": batch["normals_valid"][0],
        "num_surfaces": batch["num_surfaces"][0],
        "padding_mask": batch["padding_mask"][0],
        "surface_chain_info": batch["surface_chain_info"][0],
        "surface_geometry": batch.get("surface_geometry", [[]])[0],
        "patch_info": batch["patch_info"][0],
    }


def _center_crop_torch(t, target: int):
    """Center-crop the trailing 3 spatial dims of a torch tensor to `target`.
    No-op on dims already <= target.
    """
    z, y, x = t.shape[-3:]
    def _slc(d):
        if d <= target:
            return slice(None)
        s = (d - target) // 2
        return slice(s, s + target)
    return t[..., _slc(z), _slc(y), _slc(x)]


def _center_crop_np(arr, target: int):
    """Center-crop the trailing 3 spatial dims of a numpy array to `target`."""
    *_, z, y, x = arr.shape
    def _slc(d):
        if d <= target:
            return slice(None)
        s = (d - target) // 2
        return slice(s, s + target)
    return arr[..., _slc(z), _slc(y), _slc(x)]


def _center_crop_or_pad_3d(arr, target: int, pad_value=np.nan):
    """Center each trailing 3 spatial axis of `arr` into a `target`-sized
    cubic box: crop axes that are larger, pad (with `pad_value`) axes
    that are smaller. Returns a new array of shape `(..., target, target, target)`.
    """
    *lead, z, y, x = arr.shape
    out = np.full((*lead, target, target, target), pad_value, dtype=arr.dtype)

    def _src_dst(d):
        if d >= target:
            s = (d - target) // 2
            return slice(s, s + target), slice(0, target)
        s = (target - d) // 2
        return slice(0, d), slice(s, s + d)

    sz, dz = _src_dst(z)
    sy, dy = _src_dst(y)
    sx, dx = _src_dst(x)
    out[..., dz, dy, dx] = arr[..., sz, sy, sx]
    return out


def _crop_offset(shape_zyx, target: int):
    """Per-axis offset subtracted from a coord to convert dataset-patch
    coordinates to comparison-tile coordinates.
    """
    return np.array(
        [(d - target) // 2 if d > target else 0 for d in shape_zyx],
        dtype=np.float32,
    )


_MODEL_CACHE: dict = {}


def _read_checkpoint_meta(model_path: str, patch_size_override: int | None = None) -> dict:
    """Read architecture metadata (patch_size, norm_type, upsample_mode)
    from a checkpoint. Falls back to the sibling ``config.json`` (the
    file ``train_tifxyz.train`` writes next to checkpoints) for old
    checkpoints that don't embed ``patch_size``.
    """
    import torch
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    meta = {}
    if isinstance(ckpt, dict):
        for k in ("patch_size", "norm_type", "upsample_mode",
                  "in_channels", "out_channels", "output_sigmoid"):
            if k in ckpt:
                meta[k] = ckpt[k]

    if "patch_size" not in meta:
        sibling = Path(model_path).parent / "config.json"
        if sibling.is_file():
            try:
                with open(sibling, "r") as f:
                    cfg = json.load(f)
                if "patch_size" in cfg:
                    meta["patch_size"] = int(cfg["patch_size"])
                for k in ("norm_type", "upsample_mode"):
                    if k in cfg and k not in meta:
                        meta[k] = cfg[k]
            except Exception:
                pass

    if patch_size_override is not None:
        meta["patch_size"] = int(patch_size_override)

    if "patch_size" not in meta:
        raise ValueError(
            f"Cannot determine patch_size for checkpoint {model_path}: "
            f"not in the checkpoint dict and no sibling config.json with "
            f"patch_size. Either pass --patch-size <N> (the patch size "
            f"the model was trained at) or re-train with the updated "
            f"train_tifxyz.py that embeds patch_size in the checkpoint."
        )
    return meta


def _load_model(model_path: str, patch_size: int, device):
    """Build + load a checkpoint once per run (strict), cache by path."""
    from train_tifxyz import build_model
    key = (model_path, patch_size, str(device))
    if key not in _MODEL_CACHE:
        model, _, _ = build_model(
            patch_size=patch_size, device=str(device), weights=model_path,
            strict=True,
        )
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _scale_space_residual_sum(pred, target, mask, num_scales: int, kind: str):
    """Per-voxel residual at every scale (matching ScaleSpaceLoss3D pooling),
    upsampled to full res and summed. Used to visualize where the
    multi-scale loss is being charged.
    """
    import torch
    import torch.nn.functional as F
    from tifxyz_labels import scale_space_pool_validity

    full_size = pred.shape[2:]
    x, y = pred, target
    m = (mask > 0.5).float()
    total = torch.zeros_like(pred)
    for scale in range(num_scales):
        if kind == "mse":
            r = (x - y) ** 2 * m
        elif kind == "abs":
            r = (x - y).abs() * m
        else:
            r = F.smooth_l1_loss(x, y, reduction="none") * m
        if r.shape[2:] != full_size:
            r = F.interpolate(r, size=full_size, mode="nearest")
        total = total + r
        if scale == num_scales - 1 or x.size(2) < 2 or x.size(3) < 2 or x.size(4) < 2:
            break
        eps = 1e-6
        m_count = F.avg_pool3d(m, kernel_size=2, stride=2)
        denom = m_count.clamp_min(eps)
        x = F.avg_pool3d(x * m, kernel_size=2, stride=2) / denom
        y = F.avg_pool3d(y * m, kernel_size=2, stride=2) / denom
        m = scale_space_pool_validity(m)
    return total


def _compute_inference_output(batch, training_output, model_path, device,
                              model_build_patch_size: int,
                              inference_size: int,
                              compare_size: int,
                              output_sigmoid: bool):
    """Run the model on a fresh CT crop centered on the patch's world
    bbox, then compute losses + residual maps cropped to `compare_size`.
    Returns ``None`` if CUDA / model load fail.
    """
    import torch
    import torch.nn.functional as F
    from train_tifxyz import MaskedMSE, MaskedSmoothL1, ScaleSpaceLoss3D
    from vesuvius.neural_tracing.datasets.common import (
        _read_volume_crop_from_patch,
    )

    if not torch.cuda.is_available() or training_output is None:
        return None

    try:
        model = _load_model(model_path, model_build_patch_size, device)

        # Read a fresh inference-sized CT crop centered on the patch's
        # world center. Independent of the dataset's image tensor —
        # works for any inference_size (smaller, equal, or larger than
        # the dataset patch). _read_volume_crop_from_patch zero-pads
        # outside the volume so edge patches still produce a clean
        # cubic crop.
        patches = batch.get("_patch")
        if not patches:
            print(f"{TAG} WARNING: batch missing '_patch' — cannot read "
                  f"inference crop", flush=True)
            return None
        patch = patches[0]
        z0, z1, y0, y1, x0, x1 = patch.world_bbox
        cz_w = (z0 + z1) // 2
        cy_w = (y0 + y1) // 2
        cx_w = (x0 + x1) // 2
        half = inference_size // 2
        min_corner = np.array([cz_w - half, cy_w - half, cx_w - half],
                              dtype=np.int64)
        max_corner = min_corner + np.array(
            [inference_size, inference_size, inference_size], dtype=np.int64,
        )
        inf_crop = _read_volume_crop_from_patch(
            patch,
            crop_size=(inference_size, inference_size, inference_size),
            min_corner=min_corner, max_corner=max_corner,
            image_normalization="unit",
        )
        img_for_model = (
            torch.from_numpy(inf_crop)
            .float()
            .unsqueeze(0).unsqueeze(0)
            .to(device, non_blocking=True)
        )
        print(
            f"{TAG} forward: input={tuple(img_for_model.shape)}  "
            f"world_min={tuple(int(v) for v in min_corner)}  "
            f"world_max={tuple(int(v) for v in max_corner)}",
            flush=True,
        )

        targets_np = training_output["targets"]
        validity_np = training_output["validity"]
        normals_valid_np = training_output["normals_valid"]
        targets_t = torch.from_numpy(targets_np).unsqueeze(0).to(device)
        validity_t = torch.from_numpy(validity_np).view(1, 1, *validity_np.shape).to(device)
        normals_valid_t = torch.from_numpy(normals_valid_np).view(
            1, 1, *normals_valid_np.shape).to(device)

        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=True,
        ):
            results = model(img_for_model)
            raw = results["output"].float()
            if output_sigmoid:
                pred_full = torch.sigmoid(raw)
            else:
                pred_full = raw.clamp(0.0, 1.0)
        print(
            f"{TAG} forward: output={tuple(pred_full.shape)}",
            flush=True,
        )

        # Loss + residuals at compare_size = min(inference, dataset).
        # pred_full is at inference_size; targets are at dataset_patch.
        # Both center-cropped to compare_size.
        pred = _center_crop_torch(pred_full, compare_size)
        targets_t = _center_crop_torch(targets_t, compare_size)
        validity_t = _center_crop_torch(validity_t, compare_size)
        normals_valid_t = _center_crop_torch(normals_valid_t, compare_size)

        cos_mask = validity_t
        dir_mask = (normals_valid_t > 0.5).float()

        sl_mse = ScaleSpaceLoss3D(MaskedMSE(), num_scales=_NUM_SCALES)
        sl_l1 = ScaleSpaceLoss3D(MaskedSmoothL1(), num_scales=_NUM_SCALES)

        with torch.no_grad():
            loss_cos = sl_mse(pred[:, 0:1], targets_t[:, 0:1], mask=cos_mask).item()
            loss_mag = sl_l1(pred[:, 1:2], targets_t[:, 1:2], mask=cos_mask).item()
            loss_dir = sl_mse(
                pred[:, 2:8], targets_t[:, 2:8], mask=dir_mask,
            ).item()

            # Use absolute residuals for visualization (intuitive scale)
            # — losses themselves above are still exact MSE/SmoothL1.
            cos_diff_full = ((pred[:, 0:1] - targets_t[:, 0:1]).abs() * cos_mask)
            mag_diff_full = ((pred[:, 1:2] - targets_t[:, 1:2]).abs() * cos_mask)
            cos_diff_ss = _scale_space_residual_sum(
                pred[:, 0:1], targets_t[:, 0:1], cos_mask, _NUM_SCALES, "abs",
            )
            mag_diff_ss = _scale_space_residual_sum(
                pred[:, 1:2], targets_t[:, 1:2], cos_mask, _NUM_SCALES, "abs",
            )

        # Full-size pred (at inference_size) for display; renderer will
        # crop or pad to dataset_patch (vis_size) as needed.
        return {
            "pred_cos": pred_full[0, 0].detach().cpu().numpy(),
            "pred_mag": pred_full[0, 1].detach().cpu().numpy(),
            "pred_dir": pred_full[0, 2:8].detach().cpu().numpy(),  # (6,Z,Y,X)
            # Residuals are at compare_size (loss size).
            "diff_cos_full": cos_diff_full[0, 0].detach().cpu().numpy(),
            "diff_mag_full": mag_diff_full[0, 0].detach().cpu().numpy(),
            "diff_cos_ss": cos_diff_ss[0, 0].detach().cpu().numpy(),
            "diff_mag_ss": mag_diff_ss[0, 0].detach().cpu().numpy(),
            "loss_cos": loss_cos,
            "loss_mag": loss_mag,
            "loss_dir": loss_dir,
            "loss_total": loss_cos + loss_mag + loss_dir,
            "inference_size": int(inference_size),
            "compare_size": int(compare_size),
        }
    except Exception as exc:
        print(f"{TAG} WARNING: inference failed ({exc})", flush=True)
        return None


def _compute_training_output(batch: dict):
    """Run ``compute_batch_targets`` on one batch and package numpy arrays.

    Returns ``None`` if CUDA is unavailable or the call fails, in which
    case the render falls back to the CT/arrow rows only.
    """
    from train_tifxyz import compute_batch_targets
    from tifxyz_labels import scale_space_validity_pyramid
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    try:
        targets, validity, normals_valid, _dir_weight = compute_batch_targets(
            batch, device,
        )
    except Exception as exc:
        print(f"{TAG} WARNING: compute_batch_targets failed ({exc})", flush=True)
        return None

    pyramid_tensors = scale_space_validity_pyramid(validity, _NUM_SCALES)
    full_size = validity.shape[2:]
    pyramid_upsampled = []
    for lvl in pyramid_tensors:
        if lvl.shape[2:] == full_size:
            pyramid_upsampled.append(lvl[0, 0].detach().cpu().numpy())
        else:
            up = F.interpolate(lvl, size=full_size, mode="nearest")
            pyramid_upsampled.append(up[0, 0].detach().cpu().numpy())

    return {
        "targets": targets[0].detach().cpu().numpy(),         # (8, Z, Y, X)
        "validity": validity[0, 0].detach().cpu().numpy(),    # (Z, Y, X)
        "normals_valid": normals_valid[0, 0].detach().cpu().numpy(),
        "validity_pyramid": pyramid_upsampled,                # list of (Z, Y, X)
    }


# ---------------------------------------------------------------------------
# Per-panel drawers
# ---------------------------------------------------------------------------

def _draw_contours_and_labels(ax, mask_slices: list, chain_info_list: list,
                              sample_seed: int):
    for si, mslice in enumerate(mask_slices):
        if mslice.size == 0 or not np.any(mslice > 0.5):
            continue
        info = chain_info_list[si]
        complete = bool(info.get("has_prev", False)) and bool(info.get("has_next", False))
        color = _surface_color(sample_seed, si)
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


def _draw_normal_arrows(ax, surface_geometry, chain_info_list,
                        plane: str, plane_coord: int, rng,
                        sample_seed: int):
    """Project in-plane normals from surface points within ±0.6 of the slice.

    ``surface_geometry`` is the dataset's ``sample['surface_geometry']``
    field — a list of {wrap_idx, points_local (M,3 ZYX), normals_zyx (M,3)}
    aligned with ``surface_masks`` / ``surface_chain_info``.
    """
    if plane == "axial":
        coord_axis, hcol, vcol = 0, 2, 1  # x, y
    elif plane == "coronal":
        coord_axis, hcol, vcol = 1, 2, 0  # x, z
    else:  # sagittal
        coord_axis, hcol, vcol = 2, 1, 0  # y, z

    for si, geom in enumerate(surface_geometry):
        points = np.asarray(geom["points_local"])
        normals = np.asarray(geom["normals_zyx"])
        if points.shape[0] == 0:
            continue

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

        color = _surface_color(sample_seed, si)
        ax.quiver(
            pts[:, hcol], pts[:, vcol],
            u * _ARROW_LEN_PX, v * _ARROW_LEN_PX,
            angles="xy", scale_units="xy", scale=1.0,
            color=color, width=0.004, headwidth=3.0, headlength=4.0, alpha=0.9,
        )


def _draw_grid_normal_arrows(ax, dir_channels, normals_valid_3d,
                             plane: str, plane_coord: int,
                             color: str = "#22c55e") -> None:
    """Decode direction_channels back to a 2D in-plane direction and draw
    arrows on a regular voxel grid (every `_GRID_NORMAL_STRIDE` voxels)
    where `normals_valid_3d` is set.
    """
    c0, c1 = _PLANE_DIR_CHANNELS[plane]
    if plane == "axial":
        d0 = dir_channels[c0, plane_coord, :, :]
        d1 = dir_channels[c1, plane_coord, :, :]
        valid = normals_valid_3d[plane_coord, :, :]
    elif plane == "coronal":
        d0 = dir_channels[c0, :, plane_coord, :]
        d1 = dir_channels[c1, :, plane_coord, :]
        valid = normals_valid_3d[:, plane_coord, :]
    else:
        d0 = dir_channels[c0, :, :, plane_coord]
        d1 = dir_channels[c1, :, :, plane_coord]
        valid = normals_valid_3d[:, :, plane_coord]

    rows, cols = d0.shape
    s = _GRID_NORMAL_STRIDE
    rr, cc = np.mgrid[s // 2:rows:s, s // 2:cols:s]
    rr = rr.flatten()
    cc = cc.flatten()
    keep = valid[rr, cc] > 0.5
    if not np.any(keep):
        return
    rr, cc = rr[keep], cc[keep]
    d0v = d0[rr, cc]
    d1v = d1[rr, cc]
    h, v = _decode_dir_pair(d0v, d1v)
    ax.quiver(
        cc, rr, h * _ARROW_LEN_PX, v * _ARROW_LEN_PX,
        angles="xy", scale_units="xy", scale=1.0,
        color=color, width=0.004, headwidth=3.0, headlength=4.0, alpha=0.9,
    )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def _render_sample_figure(
    sample: dict,
    training_output: dict | None,
    out_path: Path,
    title: str,
    arrow_seed: int = 0,
    inference_output: dict | None = None,
) -> None:
    image = sample["image"][0].numpy()                  # (Z, Y, X)
    surface_masks = sample["surface_masks"].numpy()      # (N, Z, Y, X)
    chain_info = sample["surface_chain_info"]            # list[dict]
    surface_geometry = sample.get("surface_geometry", [])  # list[dict]
    direction_channels_full = sample.get("direction_channels")
    if direction_channels_full is not None:
        direction_channels_full = direction_channels_full.numpy()  # (6,Z,Y,X)

    # Renderer always draws at the dataset patch size (vis_size).
    # Pred and residuals come from inference_output at potentially
    # different sizes — the per-row drawers center-crop or pad them
    # below as needed.
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

    has_inf = inference_output is not None
    direction_channels = direction_channels_full

    # Each row: (label, drawer, n_panels). drawer(subfigure) creates
    # its own subplot grid, so different rows can have different col
    # counts within the same figure.
    rows: list[tuple[str, callable, int]] = []

    # Row 1 — CT + contours + labels
    def draw_row_contours(sf):
        axes = sf.subplots(1, 3)
        for col, ax in enumerate(axes):
            ax.imshow(image_disp[col], cmap="gray", interpolation="nearest")
            mslices = [_plane_slices(m, cz, cy, cx)[col] for m in surface_masks]
            _draw_contours_and_labels(ax, mslices, chain_info, arrow_seed)
            ax.set_title(plane_names[col], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    rows.append(("CT + chain contours", draw_row_contours, 3))

    # Row 2 — CT + sparse normal arrows from raw points
    def draw_row_arrows(sf):
        axes = sf.subplots(1, 3)
        rng = np.random.default_rng(arrow_seed)
        for col, ax in enumerate(axes):
            ax.imshow(image_disp[col], cmap="gray", interpolation="nearest")
            _draw_normal_arrows(
                ax, surface_geometry, chain_info,
                plane_keys[col], plane_coords[col], rng, arrow_seed,
            )
            ax.set_title(plane_names[col], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    rows.append(("CT + sparse normals (raw)", draw_row_arrows, 3))

    # Row 3 — regular-grid normals decoded from direction_channels.
    # GT-only without --model; GT vs pred side-by-side with --model.
    if direction_channels is not None and training_output is not None:
        normals_valid = training_output["normals_valid"]  # (Z, Y, X)

        if has_inf:
            n_panels_norm = 6
            def draw_row_grid_normals(sf):
                axes = sf.subplots(1, 6)
                # Cols 0..2 GT; cols 3..5 pred.
                for col in range(3):
                    ax = axes[col]
                    ax.imshow(image_disp[col], cmap="gray",
                              interpolation="nearest")
                    _draw_grid_normal_arrows(
                        ax, direction_channels, normals_valid,
                        plane_keys[col], plane_coords[col],
                        color="#22c55e",
                    )
                    ax.set_title(f"GT  {plane_names[col]}", fontsize=9)
                    ax.set_xticks([]); ax.set_yticks([])

                # pred_dir is at inference_size; pad/crop to dataset
                # patch size so the grid arrows align with the GT panel.
                pred_dir = _center_crop_or_pad_3d(
                    inference_output["pred_dir"], normals_valid.shape[-1],
                    pad_value=0.0,
                )
                for col in range(3):
                    ax = axes[3 + col]
                    ax.imshow(image_disp[col], cmap="gray",
                              interpolation="nearest")
                    _draw_grid_normal_arrows(
                        ax, pred_dir, normals_valid,
                        plane_keys[col], plane_coords[col],
                        color="#f97316",
                    )
                    ax.set_title(f"pred  {plane_names[col]}", fontsize=9)
                    ax.set_xticks([]); ax.set_yticks([])
            rows.append(("normals grid (GT | pred)", draw_row_grid_normals, n_panels_norm))
        else:
            def draw_row_grid_normals(sf):
                axes = sf.subplots(1, 3)
                for col, ax in enumerate(axes):
                    ax.imshow(image_disp[col], cmap="gray",
                              interpolation="nearest")
                    _draw_grid_normal_arrows(
                        ax, direction_channels, normals_valid,
                        plane_keys[col], plane_coords[col],
                        color="#22c55e",
                    )
                    ax.set_title(f"GT normals  {plane_names[col]}", fontsize=9)
                    ax.set_xticks([]); ax.set_yticks([])
            rows.append(("normals grid (GT)", draw_row_grid_normals, 3))

    if training_output is not None:
        targets = training_output["targets"]              # (8, Z, Y, X)
        validity = training_output["validity"]            # (Z, Y, X)
        pyramid = training_output["validity_pyramid"]
        cos_gt = targets[0]
        grad_mag_gt = targets[1]
        cos_gt_slices = _plane_slices(cos_gt, cz, cy, cx)
        gm_gt_slices = _plane_slices(grad_mag_gt, cz, cy, cx)
        valid_slices_full = _plane_slices(validity, cz, cy, cx)
        # Shared vmax for grad_mag computed once on GT, reused for pred.
        gm_vmax = _auto_vmax(grad_mag_gt, validity > 0.5, percentile=99.0)

        if has_inf:
            # pred_* are at inference_size; residuals at compare_size.
            # Both are center-cropped/padded to dataset patch size (= Z)
            # so they line up with the GT panels in the same row.
            assert Z == Y == X, "vis assumes cubic dataset patches"
            pred_cos = _center_crop_or_pad_3d(
                inference_output["pred_cos"], Z)
            pred_mag = _center_crop_or_pad_3d(
                inference_output["pred_mag"], Z)
            diff_cos_full = _center_crop_or_pad_3d(
                inference_output["diff_cos_full"], Z)
            diff_mag_full = _center_crop_or_pad_3d(
                inference_output["diff_mag_full"], Z)
            diff_cos_ss = _center_crop_or_pad_3d(
                inference_output["diff_cos_ss"], Z)
            diff_mag_ss = _center_crop_or_pad_3d(
                inference_output["diff_mag_ss"], Z)
            pred_cos_slices = _plane_slices(pred_cos, cz, cy, cx)
            pred_mag_slices = _plane_slices(pred_mag, cz, cy, cx)
            dcos_full_slices = _plane_slices(diff_cos_full, cz, cy, cx)
            dmag_full_slices = _plane_slices(diff_mag_full, cz, cy, cx)
            dcos_ss_slices = _plane_slices(diff_cos_ss, cz, cy, cx)
            dmag_ss_slices = _plane_slices(diff_mag_ss, cz, cy, cx)
            # Shared diff vmax across full + ss for the same channel
            cos_diff_vmax = max(
                _auto_vmax(diff_cos_full, validity > 0.5, 99.0),
                _auto_vmax(diff_cos_ss, validity > 0.5, 99.0),
            )
            mag_diff_vmax = max(
                _auto_vmax(diff_mag_full, validity > 0.5, 99.0),
                _auto_vmax(diff_mag_ss, validity > 0.5, 99.0),
            )

        def _draw_cos_panel(ax, slc):
            ax.imshow(slc, cmap="gray", interpolation="nearest",
                      vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])

        def _draw_diff_panel(ax, slc, mask_slc, vmax):
            disp = np.where(mask_slc > 0.5, slc, np.nan)
            ax.imshow(disp, cmap="hot", interpolation="nearest",
                      vmin=0.0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])

        # cos row — GT only without --model, GT|pred|diff_full|diff_ss with --model
        if has_inf:
            def draw_row_cos(sf):
                axes = sf.subplots(1, 12)
                groups = [
                    ("GT", cos_gt_slices),
                    ("pred", pred_cos_slices),
                    (f"|p−t| (vmax={cos_diff_vmax:.3g})", dcos_full_slices),
                    (f"ss-sum (vmax={cos_diff_vmax:.3g})", dcos_ss_slices),
                ]
                for gi, (gname, slices) in enumerate(groups):
                    for col in range(3):
                        ax = axes[gi * 3 + col]
                        if gi < 2:  # GT / pred → grayscale 0..1
                            _draw_cos_panel(ax, slices[col])
                        else:
                            _draw_diff_panel(
                                ax, slices[col], valid_slices_full[col],
                                cos_diff_vmax,
                            )
                        ax.set_title(
                            f"cos {gname}  {plane_names[col]}", fontsize=8,
                        )
            rows.append(("cos", draw_row_cos, 12))
        else:
            def draw_row_cos(sf):
                axes = sf.subplots(1, 3)
                for col, ax in enumerate(axes):
                    _draw_cos_panel(ax, cos_gt_slices[col])
                    ax.set_title(f"cos  {plane_names[col]}", fontsize=9)
            rows.append(("cos", draw_row_cos, 3))

        # grad_mag row — same extension structure
        if has_inf:
            def draw_row_gm(sf):
                axes = sf.subplots(1, 12)
                groups = [
                    (f"GT (vmax={gm_vmax:.3g})", gm_gt_slices, "viridis"),
                    (f"pred", pred_mag_slices, "viridis"),
                    (f"|p−t|", dmag_full_slices, "hot"),
                    (f"ss-sum", dmag_ss_slices, "hot"),
                ]
                for gi, (gname, slices, cmap) in enumerate(groups):
                    for col in range(3):
                        ax = axes[gi * 3 + col]
                        if gi < 2:
                            disp = np.where(
                                valid_slices_full[col] > 0.5, slices[col], np.nan)
                            ax.imshow(disp, cmap=cmap,
                                      interpolation="nearest",
                                      vmin=0.0, vmax=gm_vmax)
                        else:
                            _draw_diff_panel(
                                ax, slices[col], valid_slices_full[col],
                                mag_diff_vmax,
                            )
                        ax.set_xticks([]); ax.set_yticks([])
                        ax.set_title(
                            f"grad_mag {gname}  {plane_names[col]}", fontsize=8,
                        )
            rows.append(("grad_mag", draw_row_gm, 12))
        else:
            def draw_row_gm(sf):
                axes = sf.subplots(1, 3)
                for col, ax in enumerate(axes):
                    disp = np.where(
                        valid_slices_full[col] > 0.5, gm_gt_slices[col], np.nan)
                    ax.imshow(disp, cmap="viridis",
                              interpolation="nearest", vmin=0.0, vmax=gm_vmax)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(
                        f"grad_mag (vmax={gm_vmax:.3g})  {plane_names[col]}",
                        fontsize=9,
                    )
            rows.append(("grad_mag", draw_row_gm, 3))

        # validity scale rows
        for scale_idx, scale_vol in enumerate(pyramid):
            scale_slices = _plane_slices(scale_vol, cz, cy, cx)
            coarse = tuple(s // (2 ** scale_idx) for s in validity.shape)

            def _make_validity_drawer(slices, scale_idx=scale_idx, coarse=coarse):
                def _draw(sf):
                    axes = sf.subplots(1, 3)
                    for col, ax in enumerate(axes):
                        ax.imshow(image_disp[col], cmap="gray",
                                  interpolation="nearest", alpha=0.35)
                        ax.imshow(slices[col], cmap="magma",
                                  interpolation="nearest", vmin=0.0, vmax=1.0)
                        ax.set_xticks([]); ax.set_yticks([])
                        ax.set_title(
                            f"validity s{scale_idx} "
                            f"({coarse[0]}×{coarse[1]}×{coarse[2]})  "
                            f"{plane_names[col]}", fontsize=8,
                        )
                return _draw
            rows.append(
                (f"validity s{scale_idx}",
                 _make_validity_drawer(scale_slices), 3))
    else:
        def draw_skip(sf):
            axes = sf.subplots(1, 3)
            for ax in axes:
                ax.text(0.5, 0.5, "CUDA unavailable — EDT backend disabled",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        rows.append(("supervision (skipped: no CUDA)", draw_skip, 3))

    n_rows = len(rows)
    max_panels = max(n for _, _, n in rows)
    fig = plt.figure(figsize=(2.4 * max_panels, 3.0 * n_rows))
    subfigs = fig.subfigures(nrows=n_rows, ncols=1)
    if n_rows == 1:
        subfigs = [subfigs]
    for (_label, drawer, _n), sf in zip(rows, subfigs):
        drawer(sf)

    full_title = title
    if has_inf:
        full_title = (
            f"{title}\n"
            f"loss_cos={inference_output['loss_cos']:.4f}  "
            f"loss_mag={inference_output['loss_mag']:.4f}  "
            f"loss_dir={inference_output['loss_dir']:.4f}  "
            f"total={inference_output['loss_total']:.4f}"
        )
    fig.suptitle(full_title, fontsize=9)
    fig.savefig(out_path, dpi=100, format="jpeg", bbox_inches="tight")
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
    num_workers: int | None = None,
    model_path: str | None = None,
    inference_tile_size: int | None = None,
) -> None:
    """Render visualization JPEGs for samples from each dataset.

    Parallelism:
        - ``num_workers`` DataLoader workers parallelize the per-sample
          extraction (zarr reads, voxelization, chain building).
        - The main thread runs ``compute_batch_targets`` on GPU (serial
          — one GPU, no benefit to concurrency).
        - A thread pool of size ``num_workers`` handles matplotlib
          render + JPEG save. Each figure is local to its task so Agg
          is thread-safe here.
    """
    from torch.utils.data import DataLoader, Subset
    from tifxyz_lasagna_dataset import (
        TifxyzLasagnaDataset,
        collate_variable_surfaces,
    )

    with open(train_config, "r") as f:
        config = json.load(f)
    # NOTE: --patch-size is intentionally NOT applied to config["patch_size"].
    # It is only the model-architecture patch size used as a fallback for old
    # checkpoints that don't embed `patch_size`. The dataset always uses the
    # training config's own `patch_size` (so GT, surface masks, validity,
    # cache, etc. match training exactly).

    # The dataset uses the training config patch size for everything:
    # patch finding, surface mask voxelization, GT computation. The
    # inference flag does NOT affect the dataset — we only feed the
    # model a separately-read CT crop of size `inference_size`.
    dataset_patch = int(config["patch_size"])
    inference_size_eff = (
        int(inference_tile_size) if inference_tile_size else dataset_patch
    )
    compare_size = min(dataset_patch, inference_size_eff)
    vis_size = dataset_patch

    # When --model is set, the checkpoint determines the architecture
    # (NetworkFromConfig.autoconfigure derives stage count from patch
    # size), independent of the dataset patch / inference patch.
    model_build_patch_size: int | None = None
    output_sigmoid: bool = True
    if model_path is not None:
        meta = _read_checkpoint_meta(model_path, patch_size_override=patch_size)
        model_build_patch_size = int(meta["patch_size"])
        if "output_sigmoid" in meta:
            output_sigmoid = bool(meta["output_sigmoid"])
            sig_src = "checkpoint"
        else:
            output_sigmoid = True
            sig_src = "default (no key in checkpoint)"
        print(
            f"{TAG} model_build={model_build_patch_size}  "
            f"dataset_patch={dataset_patch}  "
            f"inference={inference_size_eff}  "
            f"compare={compare_size}  "
            f"vis={vis_size}",
            flush=True,
        )
        print(
            f"{TAG} output_sigmoid={output_sigmoid} ({sig_src}) — "
            f"{'applying torch.sigmoid' if output_sigmoid else 'using clamp(0, 1)'} "
            f"to model output",
            flush=True,
        )

    out_dir = Path(vis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets_cfg = config.get("datasets", [])
    if not datasets_cfg:
        print(f"{TAG} config has no datasets", flush=True)
        return

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = int(num_workers)
    render_workers = max(1, num_workers)
    total_rendered = 0

    for ds_idx, ds_entry in enumerate(datasets_cfg):
        if ds_entry.get("volume_path") is None:
            print(f"{TAG} [{ds_idx}] skipping (volume_path is null)", flush=True)
            continue

        ds_name = _dataset_display_name(ds_entry, ds_idx)
        print(f"{TAG} [{ds_idx}] building dataset '{ds_name}'", flush=True)

        sub_config = dict(config)
        sub_config["datasets"] = [ds_entry]
        # include_geometry=True asks the dataset to emit raw per-wrap
        # points/normals alongside the training tensors. This is the only
        # extra work the vis requires over the training path.
        dataset = TifxyzLasagnaDataset(
            sub_config, apply_augmentation=False, include_geometry=True,
            include_patch_ref=(model_path is not None),
        )
        n_total = len(dataset)
        if n_total == 0:
            print(f"{TAG} [{ds_idx}] '{ds_name}' has 0 patches, skipping",
                  flush=True)
            continue

        indices = list(range(n_total))
        random.Random(seed + ds_idx).shuffle(indices)
        indices = indices[: min(num_samples, n_total)]

        print(
            f"{TAG} [{ds_idx}] '{ds_name}': rendering {len(indices)} / {n_total} "
            f"(num_workers={num_workers})",
            flush=True,
        )

        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset, batch_size=1, shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_variable_surfaces,
            persistent_workers=False,
        )

        render_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=render_workers,
        )
        pending: list[concurrent.futures.Future] = []

        try:
            for i, batch in enumerate(loader):
                idx = indices[i]
                sample = _sample_from_batch(batch)
                training_output = _compute_training_output(batch)

                inference_output = None
                if model_path is not None:
                    import torch
                    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                    inference_output = _compute_inference_output(
                        batch, training_output, model_path, device,
                        model_build_patch_size,
                        inference_size_eff,
                        compare_size,
                        output_sigmoid,
                    )

                n_wraps = int(sample["num_surfaces"])
                n_chains = (
                    len({c["chain"] for c in sample["surface_chain_info"]})
                    if sample["surface_chain_info"] else 0
                )
                title = (
                    f"{ds_name}  idx={idx}  wraps={n_wraps}  chains={n_chains}\n"
                    f"bbox={sample['patch_info']['world_bbox']}"
                )
                out_path = (
                    out_dir / f"{ds_name}_sample{i:03d}_idx{idx:06d}.jpg"
                )

                fut = render_pool.submit(
                    _render_sample_figure,
                    sample, training_output, out_path,
                    title, seed + idx, inference_output,
                )
                pending.append((i, idx, out_path, fut))

            # Drain renders in submission order and log each.
            for i, idx, out_path, fut in pending:
                fut.result()  # raises if the render thread raised
                print(
                    f"{TAG}   [{i + 1}/{len(indices)}] {out_path.name}",
                    flush=True,
                )
                total_rendered += 1
        finally:
            render_pool.shutdown(wait=True)

    print(f"{TAG} done — rendered {total_rendered} samples", flush=True)
