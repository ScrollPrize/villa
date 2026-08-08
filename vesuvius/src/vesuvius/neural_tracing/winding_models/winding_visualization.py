"""Training visualization for the winding phase model."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from vesuvius.neural_tracing.winding_models.winding_targets import extract_peaks


def make_winding_visualization(
    batch: dict,
    output: dict,
    save_path: str,
    *,
    spacing: float,
    peak_threshold: float = 0.3,
    peak_min_distance: int = 2,
    sample_idx: int = 0,
) -> None:
    """Plot one ray: planes with crossings, phase curves, crossing heatmap."""
    images = batch["plane_images"][sample_idx].detach().float().cpu().numpy()
    prob = (
        torch.sigmoid(output["crossing_logits"][sample_idx].detach().float())
        .cpu()
        .numpy()
    )
    phase_pred = output["phase"][sample_idx].detach().float().cpu().numpy()
    phase_target = batch["phase_target"][sample_idx].cpu().numpy()
    phase_valid = batch["phase_valid"][sample_idx].cpu().numpy().astype(bool)
    crossing_target = batch["crossing_target"][sample_idx].cpu().numpy()
    crossing_valid = batch["crossing_valid"][sample_idx].cpu().numpy().astype(bool)
    num_crossings = int(batch["num_crossings"][sample_idx])
    crossings = batch["crossing_t"][sample_idx, :num_crossings].cpu().numpy() / spacing
    peaks = extract_peaks(
        prob, threshold=peak_threshold, min_distance=peak_min_distance
    )

    plane_rows = min(2, images.shape[0])
    fig, axes = plt.subplots(
        plane_rows + 2,
        1,
        figsize=(14, 2.5 * (plane_rows + 2)),
        sharex=True,
        constrained_layout=True,
    )
    ray_samples = np.arange(images.shape[-1])

    for plane in range(plane_rows):
        ax = axes[plane]
        ax.imshow(images[plane], cmap="gray", aspect="auto", origin="lower")
        for t in crossings:
            ax.axvline(t, color="lime", linewidth=0.8)
        for t in peaks:
            ax.axvline(t, color="red", linewidth=0.8, linestyle="--")
        ax.set_ylabel(f"plane {plane}")

    ax = axes[plane_rows]
    offset = (
        float((phase_target[phase_valid] - phase_pred[phase_valid]).mean())
        if phase_valid.any()
        else 0.0
    )
    ax.plot(ray_samples, phase_target, color="gray", alpha=0.4)
    ax.plot(
        ray_samples,
        np.where(phase_valid, phase_target, np.nan),
        color="tab:green",
        label="target (valid)",
    )
    ax.plot(
        ray_samples, phase_pred + offset, color="tab:red", label="pred (aligned)"
    )
    ax.set_ylabel("phase")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[plane_rows + 1]
    ax.plot(ray_samples, crossing_target, color="tab:green", label="target")
    ax.plot(ray_samples, prob, color="tab:red", label="pred")
    ax.fill_between(
        ray_samples, 0.0, 1.0, where=~crossing_valid, color="gray", alpha=0.2
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("crossing")
    ax.set_xlabel("ray sample")
    ax.legend(loc="upper left", fontsize=8)

    fig.savefig(save_path, dpi=110)
    plt.close(fig)
