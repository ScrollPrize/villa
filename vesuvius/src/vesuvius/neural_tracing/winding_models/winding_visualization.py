"""Training visualization for the winding phase model."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from vesuvius.neural_tracing.winding_models.winding_targets import (
    density_supervision_mask,
    extract_peaks,
    passage_kernels,
    phase_passages,
)


def make_winding_visualization(
    batch: dict,
    output: dict,
    save_path: str,
    *,
    spacing: float,
    peak_threshold: float = 0.3,
    peak_min_distance: int = 2,
    sample_idx: int = 0,
    density_min_gap_wv: float = 4.0,
    crossing_sigma_wv: float = 1.0,
) -> None:
    """Plot one slab: center slice, crossing maps, center-column curves.

    The map panels show the center transverse row of columns (all ray
    positions x one row of the column grid); the curve panels follow the
    single center column.
    """
    image = batch["slab_image"][sample_idx].detach().float().cpu().numpy()
    phase_pred = output["phase"][sample_idx].detach().float().cpu().numpy()
    phase_target = batch["phase_target"][sample_idx].cpu().numpy()
    phase_valid = batch["phase_valid"][sample_idx].cpu().numpy().astype(bool)
    increments = (
        output["phase_increments"][sample_idx].detach().float().cpu().numpy()
    )
    log_variance = output.get("density_log_variance")
    sigma = (
        (0.5 * log_variance[sample_idx].detach().float()).exp().cpu().numpy()
        if log_variance is not None
        else None
    )
    density_target = batch["density_target"][sample_idx].cpu().numpy()
    density_mask = (
        density_supervision_mask(
            batch["density_gap_wv"][sample_idx : sample_idx + 1].cpu(),
            batch["phase_valid"][sample_idx : sample_idx + 1].cpu(),
            min_gap_wv=density_min_gap_wv,
        )
        .numpy()
        .astype(bool)[0]
    )
    crossing_target = batch["crossing_target"][sample_idx].cpu().numpy()
    crossing_valid = batch["crossing_valid"][sample_idx].cpu().numpy().astype(bool)

    # One free offset per slab (matching the loss) so the center-column plot
    # also reveals cross-column coherence errors.
    slab_valid = phase_valid.reshape(-1)
    offset = (
        float(
            phase_target.reshape(-1)[slab_valid].mean()
            - phase_pred.reshape(-1)[slab_valid].mean()
        )
        if slab_valid.any()
        else 0.0
    )

    center_row = phase_pred.shape[0] // 2
    center_col = phase_pred.shape[1] // 2
    num_crossings = int(batch["num_crossings"][sample_idx, center_row, center_col])
    crossings = (
        batch["crossing_t"][sample_idx, center_row, center_col, :num_crossings]
        .cpu()
        .numpy()
        / spacing
    )
    logits = output.get("crossing_logits")
    if logits is not None:
        prob = torch.sigmoid(logits[sample_idx].detach().float()).cpu().numpy()
        prob_map = prob[center_row]
        center_prob = prob[center_row, center_col]
        peaks = extract_peaks(
            center_prob, threshold=peak_threshold, min_distance=peak_min_distance
        )
        map_label = "pred map"
    else:
        # Headless model: the crossing view renders where the registered
        # phase steps through integers — a unit-height Gaussian at each
        # passage — mirroring the old crossing-logits panels. Same renderer
        # as volume inference (winding_targets.passage_kernels).
        sigma_samples = max(crossing_sigma_wv / spacing, 1e-6)
        prob_map = passage_kernels(
            phase_pred[center_row] + offset, sigma_samples
        )
        center_prob = prob_map[center_col]
        peaks, _ = phase_passages(phase_pred[center_row, center_col] + offset)
        map_label = "pred map (phase steps)"

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(14, 15),
        sharex=True,
        constrained_layout=True,
    )
    ray_samples = np.arange(image.shape[-1])

    ax = axes[0]
    # The transverse slice containing the central ray.
    ax.imshow(image[image.shape[0] // 2], cmap="gray", aspect="auto", origin="lower")
    for t in crossings:
        ax.axvline(t, color="lime", linewidth=0.8)
    for t in peaks:
        ax.axvline(t, color="red", linewidth=0.8, linestyle="--")
    ax.set_ylabel("center slice")

    ax = axes[1]
    ax.imshow(
        crossing_target[center_row],
        cmap="viridis",
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_ylabel("target map")

    ax = axes[2]
    ax.imshow(
        prob_map,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_ylabel(map_label)

    column_valid = phase_valid[center_row, center_col]
    ax = axes[3]
    ax.plot(ray_samples, phase_target[center_row, center_col], color="gray", alpha=0.4)
    ax.plot(
        ray_samples,
        np.where(column_valid, phase_target[center_row, center_col], np.nan),
        color="tab:green",
        label="target (valid)",
    )
    ax.plot(
        ray_samples,
        phase_pred[center_row, center_col] + offset,
        color="tab:red",
        label="pred (aligned)",
    )
    ax.set_ylabel("phase")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[4]
    ax.plot(
        ray_samples,
        np.where(
            density_mask[center_row, center_col],
            density_target[center_row, center_col],
            np.nan,
        ),
        color="tab:green",
        label="target (supervised)",
    )
    ax.plot(
        ray_samples, increments[center_row, center_col], color="tab:red", label="pred"
    )
    if sigma is not None:
        ax.fill_between(
            ray_samples,
            increments[center_row, center_col] - sigma[center_row, center_col],
            increments[center_row, center_col] + sigma[center_row, center_col],
            color="tab:red",
            alpha=0.2,
            label="pred ± sigma",
        )
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[5]
    ax.plot(
        ray_samples,
        crossing_target[center_row, center_col],
        color="tab:green",
        label="target",
    )
    ax.plot(ray_samples, center_prob, color="tab:red", label="pred")
    ax.fill_between(
        ray_samples,
        0.0,
        1.0,
        where=~crossing_valid[center_row, center_col],
        color="gray",
        alpha=0.2,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("crossing")
    ax.set_xlabel("ray sample")
    ax.legend(loc="upper left", fontsize=8)

    fig.savefig(save_path, dpi=110)
    plt.close(fig)
