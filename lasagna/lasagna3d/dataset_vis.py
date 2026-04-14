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
                  "in_channels", "out_channels"):
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
                              model_build_patch_size: int):
    """Run the model, compute losses (same path as training), and produce
    per-channel diff maps (full-res masked residual + scale-space sum).
    Returns ``None`` if CUDA / model load fail.
    """
    import torch
    import torch.nn.functional as F
    from train_tifxyz import MaskedMSE, MaskedSmoothL1, ScaleSpaceLoss3D

    if not torch.cuda.is_available() or training_output is None:
        return None

    try:
        image = batch["image"].to(device, non_blocking=True)
        # The model is built once at the checkpoint's training patch
        # size (so NetworkFromConfig autoconfigures the right number
        # of stages), but the inference pass below runs on whatever
        # cubic patch the dataset emits — 3D UNets are size-agnostic
        # at forward time as long as the side is divisible by the
        # encoder strides.
        model = _load_model(model_path, model_build_patch_size, device)

        targets_np = training_output["targets"]
        validity_np = training_output["validity"]
        normals_valid_np = training_output["normals_valid"]
        targets_t = torch.from_numpy(targets_np).unsqueeze(0).to(device)
        validity_t = torch.from_numpy(validity_np).view(1, 1, *validity_np.shape).to(device)
        normals_valid_t = torch.from_numpy(normals_valid_np).view(
            1, 1, *normals_valid_np.shape).to(device)

        # Match training: bf16 autocast halves the activation memory
        # so inference fits in the same budget as the training step.
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=True,
        ):
            results = model(image)
            pred = torch.sigmoid(results["output"]).float()

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

            cos_diff_full = ((pred[:, 0:1] - targets_t[:, 0:1]) ** 2 * cos_mask)
            mag_diff_full = (
                F.smooth_l1_loss(pred[:, 1:2], targets_t[:, 1:2], reduction="none")
                * cos_mask
            )
            cos_diff_ss = _scale_space_residual_sum(
                pred[:, 0:1], targets_t[:, 0:1], cos_mask, _NUM_SCALES, "mse",
            )
            mag_diff_ss = _scale_space_residual_sum(
                pred[:, 1:2], targets_t[:, 1:2], cos_mask, _NUM_SCALES, "l1",
            )

        return {
            "pred_cos": pred[0, 0].detach().cpu().numpy(),
            "pred_mag": pred[0, 1].detach().cpu().numpy(),
            "diff_cos_full": cos_diff_full[0, 0].detach().cpu().numpy(),
            "diff_mag_full": mag_diff_full[0, 0].detach().cpu().numpy(),
            "diff_cos_ss": cos_diff_ss[0, 0].detach().cpu().numpy(),
            "diff_mag_ss": mag_diff_ss[0, 0].detach().cpu().numpy(),
            "loss_cos": loss_cos,
            "loss_mag": loss_mag,
            "loss_dir": loss_dir,
            "loss_total": loss_cos + loss_mag + loss_dir,
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

    rows: list[tuple[str, callable]] = []

    # Row 1 — CT + contours + labels
    def draw_row_contours(axes):
        for col, ax in enumerate(axes):
            ax.imshow(image_disp[col], cmap="gray", interpolation="nearest")
            mslices = [_plane_slices(m, cz, cy, cx)[col] for m in surface_masks]
            _draw_contours_and_labels(ax, mslices, chain_info, arrow_seed)
            ax.set_title(plane_names[col], fontsize=9)
    rows.append(("CT + chain contours", draw_row_contours))

    # Row 2 — CT + normal arrows
    def draw_row_arrows(axes):
        rng = np.random.default_rng(arrow_seed)
        for col, ax in enumerate(axes):
            ax.imshow(image_disp[col], cmap="gray", interpolation="nearest")
            _draw_normal_arrows(
                ax, surface_geometry, chain_info,
                plane_keys[col], plane_coords[col], rng,
                arrow_seed,
            )
            ax.set_title(plane_names[col], fontsize=9)
    rows.append(("CT + normals", draw_row_arrows))

    if training_output is not None:
        targets = training_output["targets"]              # (8, Z, Y, X)
        validity = training_output["validity"]            # (Z, Y, X)
        pyramid = training_output["validity_pyramid"]     # list[(Z,Y,X)]
        cos = targets[0]
        grad_mag = targets[1]
        cos_slices = _plane_slices(cos, cz, cy, cx)
        gm_slices = _plane_slices(grad_mag, cz, cy, cx)
        valid_slices_full = _plane_slices(validity, cz, cy, cx)
        gm_vmax = _auto_vmax(grad_mag, validity > 0.5, percentile=99.0)

        def draw_row_cos(axes):
            for col, ax in enumerate(axes):
                ax.imshow(cos_slices[col], cmap="gray",
                          interpolation="nearest", vmin=0.0, vmax=1.0)
                ax.set_title(f"cos  {plane_names[col]}", fontsize=9)
        rows.append(("cos", draw_row_cos))

        def draw_row_gm(axes):
            for col, ax in enumerate(axes):
                ax.imshow(image_disp[col], cmap="gray",
                          interpolation="nearest", alpha=0.35)
                disp = np.where(valid_slices_full[col] > 0.5, gm_slices[col], np.nan)
                ax.imshow(disp, cmap="viridis", interpolation="nearest",
                          vmin=0.0, vmax=gm_vmax)
                ax.set_title(f"grad_mag (vmax={gm_vmax:.3g})  {plane_names[col]}",
                             fontsize=9)
        rows.append(("grad_mag", draw_row_gm))

        for scale_idx, scale_vol in enumerate(pyramid):
            scale_slices = _plane_slices(scale_vol, cz, cy, cx)
            coarse = tuple(s // (2 ** scale_idx) for s in validity.shape)

            def _make_drawer(slices, scale_idx=scale_idx, coarse=coarse):
                def _draw(axes):
                    for col, ax in enumerate(axes):
                        ax.imshow(image_disp[col], cmap="gray",
                                  interpolation="nearest", alpha=0.35)
                        ax.imshow(slices[col], cmap="magma",
                                  interpolation="nearest", vmin=0.0, vmax=1.0)
                        ax.set_title(
                            f"validity s{scale_idx} "
                            f"(coarse {coarse[0]}×{coarse[1]}×{coarse[2]}) "
                            f"{plane_names[col]}",
                            fontsize=8,
                        )
                return _draw
            rows.append((f"validity s{scale_idx}", _make_drawer(scale_slices)))

        if inference_output is not None:
            pred_cos = inference_output["pred_cos"]
            pred_mag = inference_output["pred_mag"]
            diff_cos_full = inference_output["diff_cos_full"]
            diff_mag_full = inference_output["diff_mag_full"]
            diff_cos_ss = inference_output["diff_cos_ss"]
            diff_mag_ss = inference_output["diff_mag_ss"]

            pred_cos_slices = _plane_slices(pred_cos, cz, cy, cx)
            pred_mag_slices = _plane_slices(pred_mag, cz, cy, cx)
            dcos_full_slices = _plane_slices(diff_cos_full, cz, cy, cx)
            dmag_full_slices = _plane_slices(diff_mag_full, cz, cy, cx)
            dcos_ss_slices = _plane_slices(diff_cos_ss, cz, cy, cx)
            dmag_ss_slices = _plane_slices(diff_mag_ss, cz, cy, cx)

            pred_mag_vmax = _auto_vmax(pred_mag, validity > 0.5, percentile=99.0)
            dcos_full_vmax = _auto_vmax(diff_cos_full, validity > 0.5, 99.0)
            dmag_full_vmax = _auto_vmax(diff_mag_full, validity > 0.5, 99.0)
            dcos_ss_vmax = _auto_vmax(diff_cos_ss, validity > 0.5, 99.0)
            dmag_ss_vmax = _auto_vmax(diff_mag_ss, validity > 0.5, 99.0)

            def draw_pred_cos(axes):
                for col, ax in enumerate(axes):
                    ax.imshow(pred_cos_slices[col], cmap="gray",
                              interpolation="nearest", vmin=0.0, vmax=1.0)
                    ax.set_title(f"pred cos  {plane_names[col]}", fontsize=9)
            rows.append(("pred cos", draw_pred_cos))

            def draw_pred_mag(axes):
                for col, ax in enumerate(axes):
                    ax.imshow(image_disp[col], cmap="gray",
                              interpolation="nearest", alpha=0.35)
                    ax.imshow(pred_mag_slices[col], cmap="viridis",
                              interpolation="nearest",
                              vmin=0.0, vmax=pred_mag_vmax)
                    ax.set_title(
                        f"pred grad_mag (vmax={pred_mag_vmax:.3g})  "
                        f"{plane_names[col]}", fontsize=9)
            rows.append(("pred grad_mag", draw_pred_mag))

            def _make_diff_drawer(slices, vmax, label):
                def _draw(axes):
                    for col, ax in enumerate(axes):
                        ax.imshow(image_disp[col], cmap="gray",
                                  interpolation="nearest", alpha=0.35)
                        disp = np.where(
                            valid_slices_full[col] > 0.5, slices[col], np.nan)
                        ax.imshow(disp, cmap="hot",
                                  interpolation="nearest",
                                  vmin=0.0, vmax=vmax)
                        ax.set_title(
                            f"{label} (vmax={vmax:.3g})  "
                            f"{plane_names[col]}", fontsize=8)
                return _draw

            rows.append(("cos residual full",
                         _make_diff_drawer(dcos_full_slices, dcos_full_vmax,
                                           "cos resid full")))
            rows.append(("grad_mag residual full",
                         _make_diff_drawer(dmag_full_slices, dmag_full_vmax,
                                           "grad_mag resid full")))
            rows.append(("cos residual ss",
                         _make_diff_drawer(dcos_ss_slices, dcos_ss_vmax,
                                           "cos resid ss-sum")))
            rows.append(("grad_mag residual ss",
                         _make_diff_drawer(dmag_ss_slices, dmag_ss_vmax,
                                           "grad_mag resid ss-sum")))
    else:
        def _skip(axes):
            for ax in axes:
                ax.text(0.5, 0.5, "CUDA unavailable — EDT backend disabled",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
        rows.append(("supervision (skipped: no CUDA)", _skip))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.0 * n_rows))
    if n_rows == 1:
        axes = np.asarray([axes])
    for row_idx, (_label, drawer) in enumerate(rows):
        drawer(axes[row_idx])
        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    full_title = title
    if inference_output is not None:
        full_title = (
            f"{title}\n"
            f"loss_cos={inference_output['loss_cos']:.4f}  "
            f"loss_mag={inference_output['loss_mag']:.4f}  "
            f"loss_dir={inference_output['loss_dir']:.4f}  "
            f"total={inference_output['loss_total']:.4f}"
        )
    fig.suptitle(full_title, fontsize=9)
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
    num_workers: int | None = None,
    model_path: str | None = None,
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
    if patch_size is not None:
        config["patch_size"] = patch_size

    # When --model is set, the checkpoint dictates the architecture
    # (NetworkFromConfig.autoconfigure derives stage count from patch
    # size), so we MUST run the dataset at the same patch size the
    # checkpoint was trained on. Read the checkpoint metadata up front
    # and override config["patch_size"] before the dataset is built.
    model_build_patch_size: int | None = None
    if model_path is not None:
        meta = _read_checkpoint_meta(model_path, patch_size_override=patch_size)
        model_build_patch_size = int(meta["patch_size"])
        infer_patch = config.get("patch_size")
        print(
            f"{TAG} model built at patch_size={model_build_patch_size}, "
            f"inference at dataset patch_size={infer_patch}",
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
