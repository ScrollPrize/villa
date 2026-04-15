"""Dataset visualization for TifxyzLasagnaDataset.

This tool **does not** re-implement any part of the training pipeline. It
calls ``dataset[idx]`` and ``compute_batch_targets`` exactly the way the
training loop does, then renders the resulting tensors. That means:

- The voxelized surface masks you see are the tensors the UNet is fed.
- The normal arrows are drawn from the raw normals stored in the
  per-sample ``surface_geometry`` list (exposed via
  ``include_geometry=True``). The direction-channel encoding itself
  is derived downstream inside ``tifxyz_labels.compute_patch_labels``
  via tensor-moment blending + ``encode_from_tensor`` — the vis
  pulls the 6-channel target from the post-compute ``targets[2:8]``,
  which is exactly what the loss sees.
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
import matplotlib.colors as mcolors  # noqa: E402
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


def _groups_from_merge_groups(merge_groups):
    """Rebuild ``list[list[int]]`` of original slots per merged slot."""
    groups: list[list[int]] = []
    slot_by_new: dict[int, int] = {}
    for original, new_slot in enumerate(merge_groups):
        new_slot = int(new_slot)
        if new_slot not in slot_by_new:
            slot_by_new[new_slot] = len(groups)
            groups.append([])
        groups[slot_by_new[new_slot]].append(original)
    return groups


def _merge_geometry_by_groups(orig_geom, orig_chain_info, merge_groups):
    """Collapse ``surface_geometry`` to one concatenated entry per merged slot.

    Each merged entry carries:
      - wrap_idx    — the representative's wrap_idx
      - points_local / normals_zyx — concatenation of every member's points
      - merged_from — tuple of original slot indices in this group
    """
    if not orig_geom:
        return []
    geom_by_slot: dict[int, dict] = {}
    for entry in orig_geom:
        wi = int(entry.get("wrap_idx", -1))
        for slot, ci in enumerate(orig_chain_info):
            if int(ci.get("wrap_idx", -2)) == wi:
                geom_by_slot[slot] = entry
                break
    merged: list[dict] = []
    for grp in _groups_from_merge_groups(merge_groups):
        pts_list, nrm_list = [], []
        for k in grp:
            g = geom_by_slot.get(k)
            if g is None:
                continue
            pts = np.asarray(
                g.get("points_local", np.zeros((0, 3), np.float32))
            )
            nrm = np.asarray(
                g.get("normals_zyx", np.zeros((0, 3), np.float32))
            )
            if pts.shape[0] > 0:
                pts_list.append(pts)
                nrm_list.append(nrm)
        merged.append({
            "wrap_idx": int(orig_chain_info[grp[0]].get("wrap_idx", grp[0])),
            "points_local": (np.concatenate(pts_list, axis=0)
                             if pts_list else np.zeros((0, 3), np.float32)),
            "normals_zyx": (np.concatenate(nrm_list, axis=0)
                            if nrm_list else np.zeros((0, 3), np.float32)),
            "merged_from": tuple(grp),
        })
    return merged

# Matches ScaleSpaceLoss3D default num_scales in train_tifxyz.py
_NUM_SCALES = 5
_ARROW_LEN_PX = 18.0
_MAX_ARROWS_PER_PLANE = 25
_GRID_NORMAL_STRIDE = 12
_SURFACE_ARROW_STRIDE = 4  # in-plane pixel grid for on-surface arrow rows


def _grid_bucket_indices(h_pos: np.ndarray, v_pos: np.ndarray,
                         stride: int) -> np.ndarray:
    """Keep at most one index per stride x stride in-plane cell."""
    if h_pos.size == 0:
        return np.empty(0, dtype=np.int64)
    hb = np.floor(h_pos / stride).astype(np.int64)
    vb = np.floor(v_pos / stride).astype(np.int64)
    v_span = int(vb.max() - vb.min() + 2)
    key = hb * v_span + vb
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def _brighten(color: str, factor: float = 1.25) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    return (min(1.0, r * factor), min(1.0, g * factor), min(1.0, b * factor))

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
        "tensor_moments": batch["tensor_moments"][0],
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
                  "in_channels", "out_channels", "output_sigmoid",
                  "w_cos", "w_mag", "w_dir"):
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


def _scale_space_dir_residual_sum(pred_d, target_d, mask, num_scales: int):
    """Multi-channel scale-space residual sum for direction channels.

    Per scale: takes the per-voxel mean abs diff over the 6 direction
    channels (masked), upsamples nearest to full res, and sums across
    scales. Mirrors `_scale_space_residual_sum` for scalar fields.
    """
    import torch
    import torch.nn.functional as F
    from tifxyz_labels import scale_space_pool_validity

    full_size = pred_d.shape[2:]
    x, y = pred_d, target_d
    m = (mask > 0.5).float()
    total = torch.zeros(
        (pred_d.shape[0], 1, *full_size),
        device=pred_d.device, dtype=pred_d.dtype,
    )
    for scale in range(num_scales):
        r = ((x - y).abs() * m).mean(dim=1, keepdim=True)
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
                              output_sigmoid: bool,
                              loss_weights: tuple = (1.0, 1.0, 1.0),
                              model=None,
                              inference_image_override=None):
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
        if model is None:
            model = _load_model(model_path, model_build_patch_size, device)

        if inference_image_override is not None:
            # Caller supplied the CT the model should see directly —
            # used by the training loop so the vis runs inference on
            # the same (augmented) batch image that's displayed, not
            # on a fresh world-frame read that would visually desync.
            img_for_model = inference_image_override[0:1].to(
                device, non_blocking=True,
            ).float()
            if img_for_model.ndim == 4:
                img_for_model = img_for_model.unsqueeze(0)
            inference_size = int(img_for_model.shape[-1])
            compare_size = min(int(compare_size), inference_size)
        else:
            # Read a fresh inference-sized CT crop centered on the
            # patch's world center. Independent of the dataset's image
            # tensor — works for any inference_size (smaller, equal,
            # or larger than the dataset patch).
            # _read_volume_crop_from_patch zero-pads outside the volume
            # so edge patches still produce a clean cubic crop.
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
            # Per-voxel mean abs residual over the 6 direction channels,
            # masked by the normals validity. Same idea as cos/grad_mag
            # but reduced over the channel axis so we can show one map.
            dir_diff_full = (
                (pred[:, 2:8] - targets_t[:, 2:8]).abs() * dir_mask
            ).mean(dim=1, keepdim=True)
            dir_diff_ss = _scale_space_dir_residual_sum(
                pred[:, 2:8], targets_t[:, 2:8], dir_mask, _NUM_SCALES,
            )

        # Full-size pred (at inference_size) for display; renderer will
        # crop or pad to dataset_patch (vis_size) as needed.
        return {
            "pred_cos": pred_full[0, 0].detach().cpu().numpy(),
            "pred_mag": pred_full[0, 1].detach().cpu().numpy(),
            "pred_dir": pred_full[0, 2:8].detach().cpu().numpy(),  # (6,Z,Y,X)
            # Uncropped pred at native inference_size, just for the
            # debug "full inference patch" row in the renderer.
            "pred_cos_native": pred_full[0, 0].detach().cpu().numpy(),
            # Residuals are at compare_size (loss size).
            "diff_cos_full": cos_diff_full[0, 0].detach().cpu().numpy(),
            "diff_mag_full": mag_diff_full[0, 0].detach().cpu().numpy(),
            "diff_cos_ss": cos_diff_ss[0, 0].detach().cpu().numpy(),
            "diff_mag_ss": mag_diff_ss[0, 0].detach().cpu().numpy(),
            "diff_dir_full": dir_diff_full[0, 0].detach().cpu().numpy(),
            "diff_dir_ss": dir_diff_ss[0, 0].detach().cpu().numpy(),
            "loss_cos": loss_cos,
            "loss_mag": loss_mag,
            "loss_dir": loss_dir,
            "loss_total": loss_cos + loss_mag + loss_dir,
            "w_cos": float(loss_weights[0]),
            "w_mag": float(loss_weights[1]),
            "w_dir": float(loss_weights[2]),
            "loss_weighted_total": (
                float(loss_weights[0]) * loss_cos
                + float(loss_weights[1]) * loss_mag
                + float(loss_weights[2]) * loss_dir
            ),
            "inference_size": int(inference_size),
            "compare_size": int(compare_size),
        }
    except Exception as exc:
        print(
            f"{TAG} WARNING: inference failed "
            f"({type(exc).__name__}: {exc})",
            flush=True,
        )
        return None


def _compute_training_output(
    batch: dict,
    same_surface_threshold: float | None = None,
    same_surface_groups: list[list[int]] | None = None,
):
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
    groups_batch = None
    if same_surface_groups is not None:
        groups_batch = [same_surface_groups]
    try:
        (
            targets, validity,
            dir_sparse_mask, dir_dense_mask, _dir_axis_weight,
            merge_groups_batch, merged_masks_batch, merged_chain_info_batch,
        ) = compute_batch_targets(
            batch, device,
            same_surface_threshold=same_surface_threshold,
            same_surface_groups_batch=groups_batch,
        )
    except Exception as exc:
        print(f"{TAG} WARNING: compute_batch_targets failed ({exc})", flush=True)
        return None

    # Direction supervision lives in the union of the sparse splat
    # mask and the densified slerp-blend mask. We keep it as a single
    # ``normals_valid`` field in the vis output for backward compat
    # with the downstream panels.
    normals_valid = (
        (dir_sparse_mask + dir_dense_mask) > 0.5
    ).float()

    full_size = validity.shape[2:]

    def _upsample_pyramid(tensors):
        out = []
        for lvl in tensors:
            if lvl.shape[2:] == full_size:
                out.append(lvl[0, 0].detach().cpu().numpy())
            else:
                up = F.interpolate(lvl, size=full_size, mode="nearest")
                out.append(up[0, 0].detach().cpu().numpy())
        return out

    pyramid_upsampled = _upsample_pyramid(
        scale_space_validity_pyramid(validity, _NUM_SCALES)
    )
    nv_pyramid_upsampled = _upsample_pyramid(
        scale_space_validity_pyramid(normals_valid, _NUM_SCALES)
    )

    def _pyramid_any(pyr):
        if not pyr:
            return np.zeros(tuple(int(s) for s in full_size), dtype=np.float32)
        out = np.asarray(pyr[0]).astype(np.float32)
        for lvl in pyr[1:]:
            out = np.maximum(out, np.asarray(lvl).astype(np.float32))
        return out

    validity_ss = _pyramid_any(pyramid_upsampled)
    normals_valid_ss = _pyramid_any(nv_pyramid_upsampled)

    merged_masks_np = [
        m.detach().cpu().numpy().astype(np.float32)
        for m in (merged_masks_batch[0] if merged_masks_batch else [])
    ]
    merged_chain_info = list(
        merged_chain_info_batch[0] if merged_chain_info_batch else []
    )

    return {
        "targets": targets[0].detach().cpu().numpy(),         # (8, Z, Y, X)
        "validity": validity[0, 0].detach().cpu().numpy(),    # (Z, Y, X)
        "normals_valid": normals_valid[0, 0].detach().cpu().numpy(),
        "validity_pyramid": pyramid_upsampled,                # list of (Z, Y, X)
        "validity_ss": validity_ss,                           # (Z, Y, X)
        "normals_valid_ss": normals_valid_ss,                 # (Z, Y, X)
        "merge_groups": list(merge_groups_batch[0]) if merge_groups_batch else [],
        "merged_surface_masks": merged_masks_np,              # list of (Z, Y, X)
        "merged_chain_info": merged_chain_info,               # list[dict]
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

        # Length reflects in-plane projection of the (unit 3D) normal,
        # so arrows shrink as the normal tilts out of the slice plane.
        u = nrm[:, hcol]
        v = nrm[:, vcol]

        color = _brighten(_surface_color(sample_seed, si))
        ax.quiver(
            pts[:, hcol], pts[:, vcol],
            u * _ARROW_LEN_PX, v * _ARROW_LEN_PX,
            angles="xy", scale_units="xy", scale=1.0,
            color=color, width=0.006, headwidth=3.5, headlength=4.5, alpha=1.0,
        )


def _fuse_normals_volume(direction_channels):
    """Apply the existing inference-side `_estimate_normal` (from
    `lasagna/preprocess_cos_omezarr.py`) voxel-wise to the (6, Z, Y, X)
    direction channels, returning the fused 3D normal as three (Z, Y, X)
    arrays `(nx, ny, nz)`. This is the same iterative observation-
    weighted fit used in cos/grad_mag fusion for production inference.
    """
    from preprocess_cos_omezarr import _estimate_normal
    d0_z = direction_channels[0]
    d1_z = direction_channels[1]
    d0_y = direction_channels[2]
    d1_y = direction_channels[3]
    d0_x = direction_channels[4]
    d1_x = direction_channels[5]
    _w_z, _w_y, _w_x, nx, ny, nz = _estimate_normal(
        d0_z, d1_z, d0_y, d1_y, d0_x, d1_x,
    )
    return nx.astype(np.float32), ny.astype(np.float32), nz.astype(np.float32)


def _plane_axes(plane: str):
    """Return (slice_axis, h_field_idx, v_field_idx) for the named plane.
    h/v are indices into the (z=0, y=1, x=2) order for ZYX volumes;
    used both for slicing direction-channel volumes and for picking
    in-plane components of a 3D normal.
    """
    if plane == "axial":
        return 0, 2, 1   # z fixed; horizontal=x, vertical=y
    if plane == "coronal":
        return 1, 2, 0   # y fixed; horizontal=x, vertical=z
    return 2, 1, 0       # sagittal: x fixed; horizontal=y, vertical=z


def _draw_grid_fused_arrows(ax, nx_vol, ny_vol, nz_vol, valid_3d,
                            plane: str, plane_coord: int,
                            color: str = "#22c55e") -> None:
    """Draw arrows of the fused 3D normal projected onto a plane, on a
    regular `_GRID_NORMAL_STRIDE` grid where `valid_3d` is set.
    """
    slice_axis, h_idx, v_idx = _plane_axes(plane)
    fields = (nz_vol, ny_vol, nx_vol)  # ZYX order

    def _slc(v):
        if slice_axis == 0:
            return v[plane_coord, :, :]
        if slice_axis == 1:
            return v[:, plane_coord, :]
        return v[:, :, plane_coord]

    h_field = _slc(fields[h_idx])
    v_field = _slc(fields[v_idx])
    valid = _slc(valid_3d)

    rows, cols = h_field.shape
    s = _GRID_NORMAL_STRIDE
    rr, cc = np.mgrid[s // 2:rows:s, s // 2:cols:s]
    rr = rr.flatten()
    cc = cc.flatten()
    keep = valid[rr, cc] > 0.5
    if not np.any(keep):
        return
    rr, cc = rr[keep], cc[keep]
    # Keep in-plane projection length so arrows shrink with out-of-plane tilt.
    h = h_field[rr, cc]
    v = v_field[rr, cc]
    ax.quiver(
        cc, rr, h * _ARROW_LEN_PX, v * _ARROW_LEN_PX,
        angles="xy", scale_units="xy", scale=1.0,
        color=_brighten(color), width=0.006,
        headwidth=3.5, headlength=4.5, alpha=1.0,
    )


def _draw_surface_axis_arrows(ax, dir_channels, surface_geometry,
                              plane: str, plane_coord: int,
                              sample_seed: int):
    """For each surface in `surface_geometry`, sample axis-decoded
    direction at the surface points and draw per-surface-colored arrows
    on the named plane.
    """
    c0_idx, c1_idx = _PLANE_DIR_CHANNELS[plane]
    slice_axis, h_idx, v_idx = _plane_axes(plane)
    Zd, Yd, Xd = dir_channels.shape[1], dir_channels.shape[2], dir_channels.shape[3]

    for si, geom in enumerate(surface_geometry):
        pts = np.asarray(geom["points_local"])
        if pts.shape[0] == 0:
            continue
        m = np.abs(pts[:, slice_axis] - plane_coord) < 0.6
        if not np.any(m):
            continue
        pts_p = pts[m]
        cols = pts_p[:, [0, 1, 2]]
        h_pos = cols[:, h_idx]
        v_pos = cols[:, v_idx]
        keep = _grid_bucket_indices(h_pos, v_pos, _SURFACE_ARROW_STRIDE)
        if keep.size == 0:
            continue
        pts_p = pts_p[keep]
        h_pos = h_pos[keep]
        v_pos = v_pos[keep]
        zi = np.clip(pts_p[:, 0].astype(np.int64), 0, Zd - 1)
        yi = np.clip(pts_p[:, 1].astype(np.int64), 0, Yd - 1)
        xi = np.clip(pts_p[:, 2].astype(np.int64), 0, Xd - 1)
        d0v = dir_channels[c0_idx, zi, yi, xi]
        d1v = dir_channels[c1_idx, zi, yi, xi]
        h, v = _decode_dir_pair(d0v, d1v)
        color = _brighten(_surface_color(sample_seed, si))
        ax.quiver(
            h_pos, v_pos, h * _ARROW_LEN_PX, v * _ARROW_LEN_PX,
            angles="xy", scale_units="xy", scale=1.0,
            color=color, width=0.006, headwidth=3.5, headlength=4.5, alpha=1.0,
        )


def _draw_surface_fused_arrows(ax, nx_vol, ny_vol, nz_vol, surface_geometry,
                               plane: str, plane_coord: int,
                               sample_seed: int):
    """For each surface in `surface_geometry`, sample the FUSED 3D normal
    at the surface points and draw the in-plane projection.
    """
    slice_axis, h_idx, v_idx = _plane_axes(plane)
    fields = (nz_vol, ny_vol, nx_vol)  # ZYX order
    Zd, Yd, Xd = nx_vol.shape

    for si, geom in enumerate(surface_geometry):
        pts = np.asarray(geom["points_local"])
        if pts.shape[0] == 0:
            continue
        m = np.abs(pts[:, slice_axis] - plane_coord) < 0.6
        if not np.any(m):
            continue
        pts_p = pts[m]
        cols = pts_p[:, [0, 1, 2]]
        h_pos = cols[:, h_idx]
        v_pos = cols[:, v_idx]
        keep = _grid_bucket_indices(h_pos, v_pos, _SURFACE_ARROW_STRIDE)
        if keep.size == 0:
            continue
        pts_p = pts_p[keep]
        h_pos = h_pos[keep]
        v_pos = v_pos[keep]
        zi = np.clip(pts_p[:, 0].astype(np.int64), 0, Zd - 1)
        yi = np.clip(pts_p[:, 1].astype(np.int64), 0, Yd - 1)
        xi = np.clip(pts_p[:, 2].astype(np.int64), 0, Xd - 1)
        h_field = fields[h_idx]
        v_field = fields[v_idx]
        # Keep in-plane projection length so arrows shrink with out-of-plane tilt.
        h = h_field[zi, yi, xi]
        v = v_field[zi, yi, xi]
        color = _brighten(_surface_color(sample_seed, si))
        ax.quiver(
            h_pos, v_pos, h * _ARROW_LEN_PX, v * _ARROW_LEN_PX,
            angles="xy", scale_units="xy", scale=1.0,
            color=color, width=0.006, headwidth=3.5, headlength=4.5, alpha=1.0,
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
        color=_brighten(color), width=0.006,
        headwidth=3.5, headlength=4.5, alpha=1.0,
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

    # Per-surface data from the dataset. We swap these three to the
    # merged views produced by compute_patch_labels when the merge
    # fires — the drawers then see one entry per merged group and
    # don't need any merge awareness. `surface_masks_np` is kept
    # around because the row-1 "ghost" outlines for discarded members
    # need the pre-merge tensors.
    surface_masks_np = sample["surface_masks"].numpy()   # (N_orig, Z, Y, X)
    original_chain_info = sample["surface_chain_info"]   # list[dict], len N_orig
    original_geometry = sample.get("surface_geometry", [])

    merge_groups: list[int] | None = None
    merged_masks = None
    merged_chain_info = None
    if training_output is not None:
        merged_masks = training_output.get("merged_surface_masks")
        merged_chain_info = training_output.get("merged_chain_info")
        mg = training_output.get("merge_groups")
        if mg and len(mg) == len(original_chain_info):
            merge_groups = list(mg)

    if (
        merge_groups is not None
        and merged_masks
        and merged_chain_info is not None
        and len(merged_chain_info) == len(set(merge_groups))
    ):
        surface_masks = np.stack(merged_masks, axis=0) \
            if merged_masks else np.zeros((0,) + surface_masks_np.shape[1:],
                                          dtype=np.float32)
        chain_info = list(merged_chain_info)
        surface_geometry = _merge_geometry_by_groups(
            original_geometry, original_chain_info, merge_groups,
        )
        n_orig = len(merge_groups)
        n_merged = len(chain_info)
        if n_merged < n_orig:
            groups_list = _groups_from_merge_groups(merge_groups)
            summary = ", ".join(
                f"{chain_info[new_slot].get('label', '?')}="
                f"{[original_chain_info[k].get('label', '?') for k in grp]}"
                for new_slot, grp in enumerate(groups_list)
                if len(grp) > 1
            )
            print(
                f"{TAG} merge: {n_orig}→{n_merged} surfaces ({summary})",
                flush=True,
            )
    else:
        surface_masks = surface_masks_np
        chain_info = list(original_chain_info)
        surface_geometry = original_geometry
        merge_groups = None

    # The dataset now carries raw normals (3, Z, Y, X) rather than
    # a pre-encoded 6-channel ``direction_channels`` — the
    # double-angle encoding is derived inside compute_patch_labels
    # via tensor-moment linear blending + ``encode_from_tensor`` at
    # the chain-adjacent bracket.
    # For the vis panels we prefer the post-compute encoded targets
    # (what the loss sees), which are already inside
    # ``training_output["targets"][2:8]``. Without training_output
    # (e.g. non-CUDA path) fall back to encoding the sparse tensor
    # splat directly.
    direction_channels_full = None
    if training_output is not None:
        direction_channels_full = training_output["targets"][2:8]
    else:
        tm_full = sample.get("tensor_moments")
        if tm_full is not None:
            import torch as _t
            from tifxyz_labels import encode_from_tensor as _enc_from_t
            tm_t = _t.as_tensor(tm_full, dtype=_t.float32)
            direction_channels_full = _enc_from_t(tm_t).numpy()

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

    # Row 1 — CT + contours + labels.
    # If merging discarded members of a group, draw the pre-merge
    # outlines as gray ghosts under the colored merged contours so
    # the absorbed wraps stay visible.
    discarded_slots: list[int] = []
    if merge_groups is not None:
        for grp in _groups_from_merge_groups(merge_groups):
            if len(grp) > 1:
                discarded_slots.extend(grp[1:])

    def draw_row_contours(sf):
        axes = sf.subplots(1, 3)
        for col, ax in enumerate(axes):
            ax.imshow(image_disp[col], cmap="gray", interpolation="nearest")

            # Gray ghosts of discarded (non-rep) members.
            for k in discarded_slots:
                mslice = _plane_slices(surface_masks_np[k], cz, cy, cx)[col]
                if mslice.size == 0 or not np.any(mslice > 0.5):
                    continue
                for contour in measure.find_contours(
                    mslice.astype(np.float32), 0.5,
                ):
                    ax.plot(
                        contour[:, 1], contour[:, 0],
                        color="#888888", linewidth=0.7, zorder=1,
                    )

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
            ax.imshow(image_disp[col], cmap="gray",
                      interpolation="nearest", vmin=0, vmax=600)
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
                # pred_dir is at inference_size; pad/crop to dataset
                # patch size so the grid arrows align with the GT panel.
                pred_dir = _center_crop_or_pad_3d(
                    inference_output["pred_dir"], normals_valid.shape[-1],
                    pad_value=0.0,
                )
                # Per axis: GT then pred.
                for col in range(3):
                    for gi, (gname, dc, qc) in enumerate([
                        ("GT", direction_channels, "#22c55e"),
                        ("pred", pred_dir, "#f97316"),
                    ]):
                        ax = axes[col * 2 + gi]
                        ax.imshow(image_disp[col], cmap="gray",
                                  interpolation="nearest",
                                  vmin=0, vmax=600)
                        _draw_grid_normal_arrows(
                            ax, dc, normals_valid,
                            plane_keys[col], plane_coords[col],
                            color=qc,
                        )
                        ax.set_title(
                            f"{gname}  {plane_names[col]}", fontsize=9)
                        ax.set_xticks([]); ax.set_yticks([])
            rows.append(("normals grid (GT | pred)", draw_row_grid_normals, n_panels_norm))
        else:
            def draw_row_grid_normals(sf):
                axes = sf.subplots(1, 3)
                for col, ax in enumerate(axes):
                    ax.imshow(image_disp[col], cmap="gray",
                              interpolation="nearest",
                              vmin=0, vmax=600)
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
        # Normals have their own (sparser) mask — normals_valid from the
        # splatting step, same mask the training dir loss uses.
        normals_valid_full = training_output["normals_valid"]  # (Z, Y, X)
        nv_slices_full = _plane_slices(normals_valid_full, cz, cy, cx)
        # Scale-space any-valid masks — used for the ss-sum panels so
        # the thicker (dilated) residual footprint from coarse scales
        # is actually drawn instead of being NaN'd out by the scale-0
        # mask.
        validity_ss_full = training_output.get("validity_ss")
        normals_valid_ss_full = training_output.get("normals_valid_ss")
        if validity_ss_full is None:
            validity_ss_full = validity
        if normals_valid_ss_full is None:
            normals_valid_ss_full = normals_valid_full
        valid_ss_slices = _plane_slices(validity_ss_full, cz, cy, cx)
        nv_ss_slices = _plane_slices(normals_valid_ss_full, cz, cy, cx)
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
            diff_dir_full = _center_crop_or_pad_3d(
                inference_output["diff_dir_full"], Z)
            diff_dir_ss = _center_crop_or_pad_3d(
                inference_output["diff_dir_ss"], Z)
            pred_cos_slices = _plane_slices(pred_cos, cz, cy, cx)
            pred_mag_slices = _plane_slices(pred_mag, cz, cy, cx)
            dcos_full_slices = _plane_slices(diff_cos_full, cz, cy, cx)
            dmag_full_slices = _plane_slices(diff_mag_full, cz, cy, cx)
            dcos_ss_slices = _plane_slices(diff_cos_ss, cz, cy, cx)
            dmag_ss_slices = _plane_slices(diff_mag_ss, cz, cy, cx)
            ddir_full_slices = _plane_slices(diff_dir_full, cz, cy, cx)
            ddir_ss_slices = _plane_slices(diff_dir_ss, cz, cy, cx)
            dir_diff_vmax = max(
                _auto_vmax(diff_dir_full, normals_valid_full > 0.5, 99.0),
                _auto_vmax(diff_dir_ss, normals_valid_ss_full > 0.5, 99.0),
            )
            # Pred direction channels padded/cropped to vis size (Z=Y=X)
            # for both decoded-axis and fused arrow rows.
            pred_dir_padded = _center_crop_or_pad_3d(
                inference_output["pred_dir"], Z, pad_value=0.0,
            )
            # Fused 3D normals (nx, ny, nz) for GT and pred. Same fit
            # as production cos/grad_mag fusion uses.
            try:
                nx_gt, ny_gt, nz_gt = _fuse_normals_volume(direction_channels)
                nx_pr, ny_pr, nz_pr = _fuse_normals_volume(pred_dir_padded)
                fused_ok = True
            except Exception as exc:
                print(f"{TAG} WARNING: normal fusion failed ({exc})", flush=True)
                fused_ok = False
            # Shared diff vmax across full + ss for the same channel
            cos_diff_vmax = max(
                _auto_vmax(diff_cos_full, validity > 0.5, 99.0),
                _auto_vmax(diff_cos_ss, validity_ss_full > 0.5, 99.0),
            )
            mag_diff_vmax = max(
                _auto_vmax(diff_mag_full, validity > 0.5, 99.0),
                _auto_vmax(diff_mag_ss, validity_ss_full > 0.5, 99.0),
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
                for col in range(3):
                    for gi, (gname, slices) in enumerate(groups):
                        ax = axes[col * 4 + gi]
                        if gi < 2:
                            _draw_cos_panel(ax, slices[col])
                        else:
                            mask_slices = (
                                valid_ss_slices if gi == 3
                                else valid_slices_full
                            )
                            _draw_diff_panel(
                                ax, slices[col], mask_slices[col],
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
                for col in range(3):
                    for gi, (gname, slices, cmap) in enumerate(groups):
                        ax = axes[col * 4 + gi]
                        if gi < 2:
                            disp = np.where(
                                valid_slices_full[col] > 0.5, slices[col], np.nan)
                            ax.imshow(disp, cmap=cmap,
                                      interpolation="nearest",
                                      vmin=0.0, vmax=gm_vmax)
                        else:
                            mask_slices = (
                                valid_ss_slices if gi == 3
                                else valid_slices_full
                            )
                            _draw_diff_panel(
                                ax, slices[col], mask_slices[col],
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

        # normals residual row — only when --model is set; mean abs
        # residual over the 6 direction channels under dir_mask, plus
        # scale-space sum.
        if has_inf:
            def draw_row_normals_residual(sf):
                axes = sf.subplots(1, 6)
                groups = [
                    (f"|p−t| (vmax={dir_diff_vmax:.3g})", ddir_full_slices),
                    (f"ss-sum (vmax={dir_diff_vmax:.3g})", ddir_ss_slices),
                ]
                for col in range(3):
                    for gi, (gname, slices) in enumerate(groups):
                        ax = axes[col * 2 + gi]
                        mask_slices = (
                            nv_ss_slices if gi == 1 else nv_slices_full
                        )
                        _draw_diff_panel(
                            ax, slices[col], mask_slices[col],
                            dir_diff_vmax,
                        )
                        ax.set_title(
                            f"normals {gname}  {plane_names[col]}",
                            fontsize=8,
                        )
            rows.append(("normals residual", draw_row_normals_residual, 6))

        # ---- Additional normal vis rows (only with --model) ----------
        if has_inf:
            # Row: dense regular-grid arrows of FUSED 3D normals
            # (production fit), GT|pred. Same fit as cos/grad_mag
            # fusion uses for inference.
            if fused_ok:
                def draw_row_grid_fused(sf):
                    axes = sf.subplots(1, 6)
                    entries = [
                        ("GT fused", nx_gt, ny_gt, nz_gt, "#22c55e"),
                        ("pred fused", nx_pr, ny_pr, nz_pr, "#f97316"),
                    ]
                    for col in range(3):
                        for gi, (gname, nx, ny, nz, qc) in enumerate(entries):
                            ax = axes[col * 2 + gi]
                            ax.imshow(image_disp[col], cmap="gray",
                                      interpolation="nearest",
                                      vmin=0, vmax=600)
                            _draw_grid_fused_arrows(
                                ax, nx, ny, nz, normals_valid,
                                plane_keys[col], plane_coords[col],
                                color=qc,
                            )
                            ax.set_title(
                                f"{gname}  {plane_names[col]}", fontsize=9)
                            ax.set_xticks([]); ax.set_yticks([])
                rows.append(
                    ("normals dense (fused)", draw_row_grid_fused, 6))

            # Row: on-surface arrows from axis-decoded direction
            # channels. Per-surface colors (matching row 1 contours).
            def draw_row_surface_axis(sf):
                axes = sf.subplots(1, 6)
                entries = [
                    ("GT axis on-surface", direction_channels),
                    ("pred axis on-surface", pred_dir_padded),
                ]
                for col in range(3):
                    for gi, (gname, dc) in enumerate(entries):
                        ax = axes[col * 2 + gi]
                        ax.imshow(image_disp[col], cmap="gray",
                                  interpolation="nearest",
                                  vmin=0, vmax=600)
                        _draw_surface_axis_arrows(
                            ax, dc, surface_geometry,
                            plane_keys[col], plane_coords[col], arrow_seed,
                        )
                        ax.set_title(
                            f"{gname}  {plane_names[col]}", fontsize=9)
                        ax.set_xticks([]); ax.set_yticks([])
            rows.append(
                ("normals on-surface (axis)", draw_row_surface_axis, 6))

            # Row: on-surface arrows from FUSED normals.
            if fused_ok:
                def draw_row_surface_fused(sf):
                    axes = sf.subplots(1, 6)
                    entries = [
                        ("GT fused on-surface", nx_gt, ny_gt, nz_gt),
                        ("pred fused on-surface", nx_pr, ny_pr, nz_pr),
                    ]
                    for col in range(3):
                        for gi, (gname, nx, ny, nz) in enumerate(entries):
                            ax = axes[col * 2 + gi]
                            ax.imshow(image_disp[col], cmap="gray",
                                      interpolation="nearest",
                                      vmin=0, vmax=600)
                            _draw_surface_fused_arrows(
                                ax, nx, ny, nz, surface_geometry,
                                plane_keys[col], plane_coords[col], arrow_seed,
                            )
                            ax.set_title(
                                f"{gname}  {plane_names[col]}", fontsize=9)
                            ax.set_xticks([]); ax.set_yticks([])
                rows.append(
                    ("normals on-surface (fused)",
                     draw_row_surface_fused, 6))

            # Row: 2-channel false color of direction channels.
            # Per plane, take the relevant axis pair and build an RGB
            # image with R = d0, G = d1, B = 0.5. GT vs pred.
            def draw_row_dir_falsecolor(sf):
                axes = sf.subplots(1, 6)

                def _falsecolor(d0_slice, d1_slice):
                    rgb = np.stack([
                        np.clip(d0_slice, 0.0, 1.0),
                        np.clip(d1_slice, 0.0, 1.0),
                        np.full_like(d0_slice, 0.5),
                    ], axis=-1)
                    return rgb

                for col in range(3):
                    plane = plane_keys[col]
                    pc = plane_coords[col]
                    c0i, c1i = _PLANE_DIR_CHANNELS[plane]
                    if plane == "axial":
                        gt_d0 = direction_channels[c0i, pc, :, :]
                        gt_d1 = direction_channels[c1i, pc, :, :]
                        pr_d0 = pred_dir_padded[c0i, pc, :, :]
                        pr_d1 = pred_dir_padded[c1i, pc, :, :]
                    elif plane == "coronal":
                        gt_d0 = direction_channels[c0i, :, pc, :]
                        gt_d1 = direction_channels[c1i, :, pc, :]
                        pr_d0 = pred_dir_padded[c0i, :, pc, :]
                        pr_d1 = pred_dir_padded[c1i, :, pc, :]
                    else:
                        gt_d0 = direction_channels[c0i, :, :, pc]
                        gt_d1 = direction_channels[c1i, :, :, pc]
                        pr_d0 = pred_dir_padded[c0i, :, :, pc]
                        pr_d1 = pred_dir_padded[c1i, :, :, pc]
                    ax_gt = axes[col * 2]
                    ax_pr = axes[col * 2 + 1]
                    ax_gt.imshow(_falsecolor(gt_d0, gt_d1),
                                 interpolation="nearest")
                    ax_gt.set_title(
                        f"GT (d{c0i},d{c1i})  {plane_names[col]}",
                        fontsize=8)
                    ax_gt.set_xticks([]); ax_gt.set_yticks([])

                    ax_pr.imshow(_falsecolor(pr_d0, pr_d1),
                                 interpolation="nearest")
                    ax_pr.set_title(
                        f"pred (d{c0i},d{c1i})  {plane_names[col]}",
                        fontsize=8)
                    ax_pr.set_xticks([]); ax_pr.set_yticks([])

            rows.append(
                ("direction channels (R=d0, G=d1)",
                 draw_row_dir_falsecolor, 6))

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

        # Debug row: pred_cos at the FULL native inference patch
        # size (no center-crop or pad) so it's visible whether the
        # model actually saw the larger context.
        if has_inf:
            pred_cos_native = inference_output["pred_cos_native"]
            inference_size = int(inference_output["inference_size"])
            ncz = pred_cos_native.shape[0] // 2
            ncy = pred_cos_native.shape[1] // 2
            ncx = pred_cos_native.shape[2] // 2
            native_slices = _plane_slices(pred_cos_native, ncz, ncy, ncx)

            def draw_row_pred_native(sf):
                axes = sf.subplots(1, 3)
                native_titles = [
                    f"axial z={ncz}",
                    f"coronal y={ncy}",
                    f"sagittal x={ncx}",
                ]
                for col, ax in enumerate(axes):
                    ax.imshow(native_slices[col], cmap="gray",
                              interpolation="nearest", vmin=0.0, vmax=1.0)
                    ax.set_title(
                        f"pred cos native ({inference_size}³)  "
                        f"{native_titles[col]}", fontsize=8,
                    )
                    ax.set_xticks([]); ax.set_yticks([])
            rows.append(("pred cos (full inference)", draw_row_pred_native, 3))
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
    fig = plt.figure(
        figsize=(2.4 * max_panels, 3.0 * n_rows),
        layout="constrained",
    )
    subfigs = fig.subfigures(nrows=n_rows, ncols=1)
    if n_rows == 1:
        subfigs = [subfigs]
    for (_label, drawer, _n), sf in zip(rows, subfigs):
        drawer(sf)

    full_title = title
    if has_inf:
        wc = inference_output["w_cos"]
        wm = inference_output["w_mag"]
        wd = inference_output["w_dir"]
        lc = inference_output["loss_cos"]
        lm = inference_output["loss_mag"]
        ld = inference_output["loss_dir"]
        lt = inference_output["loss_weighted_total"]
        full_title = (
            f"{title}\n"
            f"loss_cos={lc:.4f} (w={wc:g})  "
            f"loss_mag={lm:.4f} (w={wm:g})  "
            f"loss_dir={ld:.4f} (w={wd:g})\n"
            f"weighted total = {wc:g}·cos + {wm:g}·mag + {wd:g}·dir "
            f"= {lt:.4f}"
        )
    fig.suptitle(full_title, fontsize=10)
    fig.savefig(out_path, dpi=100, format="jpeg")
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


def default_vis_filename(ds_name: str, idx: int) -> str:
    """Canonical JPEG filename used by `dataset vis` — shared so other
    commands that render via `render_batch_figure` produce identically
    named files for the same (dataset, index). Keyed only on the
    stable dataset patch index, not on enumeration order."""
    return f"{ds_name}_idx{idx:06d}.jpg"


def default_vis_title(ds_name: str, idx: int, sample: dict) -> str:
    """Canonical figure title used by `dataset vis`."""
    n_wraps = int(sample["num_surfaces"])
    n_chains = (
        len({c["chain"] for c in sample["surface_chain_info"]})
        if sample["surface_chain_info"] else 0
    )
    return (
        f"{ds_name}  idx={idx}  wraps={n_wraps}  chains={n_chains}\n"
        f"bbox={sample['patch_info']['world_bbox']}"
    )


def build_inference_context(
    model_path: str | None,
    config: dict,
    patch_size: int | None = None,
    inference_tile_size: int | None = None,
    same_surface_threshold: float | None = None,
) -> dict:
    """Load checkpoint meta and derive all knobs `_compute_inference_output`
    needs. Returns a dict that's also safe when ``model_path is None``
    (the ``model_path`` field is then ``None`` and callers should skip
    inference entirely).

    This is the single source of truth for `dataset vis` and any other
    command (e.g. `dataset overlap --vis-dir`) that renders the same
    figure.
    """
    dataset_patch = int(config["patch_size"])
    inference_size_eff = (
        int(inference_tile_size) if inference_tile_size else dataset_patch
    )
    compare_size = min(dataset_patch, inference_size_eff)
    vis_size = dataset_patch

    # Resolution: explicit arg > training config field > None.
    resolved_same_surface_threshold: float | None
    if same_surface_threshold is not None:
        resolved_same_surface_threshold = float(same_surface_threshold)
    else:
        cfg_thr = config.get("same_surface_threshold")
        resolved_same_surface_threshold = (
            float(cfg_thr) if cfg_thr is not None else None
        )
    if resolved_same_surface_threshold is not None:
        print(
            f"{TAG} same_surface_threshold={resolved_same_surface_threshold} — "
            "vis will show merged duplicate wraps",
            flush=True,
        )

    ctx = {
        "model_path": model_path,
        "dataset_patch": dataset_patch,
        "inference_size_eff": inference_size_eff,
        "compare_size": compare_size,
        "vis_size": vis_size,
        "model_build_patch_size": None,
        "output_sigmoid": True,
        "loss_weights": (1.0, 1.0, 1.0),
        "same_surface_threshold": resolved_same_surface_threshold,
    }

    if model_path is None:
        return ctx

    meta = _read_checkpoint_meta(model_path, patch_size_override=patch_size)
    ctx["model_build_patch_size"] = int(meta["patch_size"])
    if "output_sigmoid" in meta:
        ctx["output_sigmoid"] = bool(meta["output_sigmoid"])
        sig_src = "checkpoint"
    else:
        ctx["output_sigmoid"] = True
        sig_src = "default (no key in checkpoint)"
    ctx["loss_weights"] = (
        float(meta.get("w_cos", 1.0)),
        float(meta.get("w_mag", 1.0)),
        float(meta.get("w_dir", 1.0)),
    )
    print(
        f"{TAG} model_build={ctx['model_build_patch_size']}  "
        f"dataset_patch={dataset_patch}  "
        f"inference={inference_size_eff}  "
        f"compare={compare_size}  "
        f"vis={vis_size}",
        flush=True,
    )
    print(
        f"{TAG} output_sigmoid={ctx['output_sigmoid']} ({sig_src}) — "
        f"{'applying torch.sigmoid' if ctx['output_sigmoid'] else 'using clamp(0, 1)'} "
        f"to model output",
        flush=True,
    )
    print(
        f"{TAG} loss weights: w_cos={ctx['loss_weights'][0]}  "
        f"w_mag={ctx['loss_weights'][1]}  w_dir={ctx['loss_weights'][2]}",
        flush=True,
    )
    return ctx


def render_batch_figure(
    batch: dict,
    out_path: Path,
    title: str,
    arrow_seed: int,
    inference_ctx: dict,
    same_surface_groups: list[list[int]] | None = None,
    model=None,
    inference_image_override=None,
) -> None:
    """End-to-end: compute training_output (+ optional inference_output)
    and write the JPEG. Mirrors what `run_dataset_vis` does for a single
    batch, so alternate commands render an identical figure.

    ``same_surface_groups`` (optional) overrides
    ``compute_patch_labels``'s own detection — used by
    `dataset overlap` to render exactly the pairs its analysis
    flagged. When ``None``, falls back to the threshold in
    ``inference_ctx``.
    """
    sample = _sample_from_batch(batch)
    training_output = _compute_training_output(
        batch,
        same_surface_threshold=inference_ctx.get("same_surface_threshold"),
        same_surface_groups=same_surface_groups,
    )

    inference_output = None
    if inference_ctx.get("model_path") is not None or model is not None:
        import torch
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        inference_output = _compute_inference_output(
            batch, training_output, inference_ctx.get("model_path"), device,
            inference_ctx["model_build_patch_size"],
            inference_ctx["inference_size_eff"],
            inference_ctx["compare_size"],
            inference_ctx["output_sigmoid"],
            inference_ctx["loss_weights"],
            model=model,
            inference_image_override=inference_image_override,
        )

    _render_sample_figure(
        sample, training_output, out_path, title, arrow_seed,
        inference_output,
    )


def run_dataset_vis(
    train_config: str,
    vis_dir: str,
    num_samples: int = 10,
    seed: int = 0,
    patch_size: int | None = None,
    num_workers: int | None = None,
    model_path: str | None = None,
    inference_tile_size: int | None = None,
    explicit_indices: list[int] | None = None,
    same_surface_threshold: float | None = None,
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
    inference_ctx = build_inference_context(
        model_path=model_path,
        config=config,
        patch_size=patch_size,
        inference_tile_size=inference_tile_size,
        same_surface_threshold=same_surface_threshold,
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

        if explicit_indices is not None:
            indices = [k for k in explicit_indices if 0 <= k < n_total]
            if len(indices) != len(explicit_indices):
                dropped = sorted(set(explicit_indices) - set(indices))
                print(
                    f"{TAG} [{ds_idx}] '{ds_name}': dropping out-of-range "
                    f"indices {dropped} (valid range 0..{n_total - 1})",
                    flush=True,
                )
        else:
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
                title = default_vis_title(ds_name, idx, sample)
                out_path = out_dir / default_vis_filename(ds_name, idx)

                fut = render_pool.submit(
                    render_batch_figure,
                    batch, out_path, title, seed + idx, inference_ctx,
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
