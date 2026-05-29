"""Self-distillation pseudo-label generator for v3 finetuning.

For each chunk in a training batch, run a primary UNet checkpoint (v2 ckpt_077000)
with TTA to produce a probability map. If the raw input chunk is both bright
(mean > mean_hi) and low-contrast (std < std_lo), additionally run an ensemble
checkpoint (v2 ckpt_064000), average the two probability maps, and use a slightly
lower binarization threshold. Otherwise threshold the primary's probability alone.
The final binary label is multiplied by the foreground mask emitted by the dataset
so background voxels are forced to 0.

This class is instantiated once per DDP rank and called from the trainer step
(NOT from dataset workers). It runs in inference_mode and uses bf16 by default.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from koine_machines.data.dino_guided_labels import _build_unet
from koine_machines.inference.infer_full3d_tifxyz import (
    TargetHeadWrapper,
    predict_batch,
    tta_variants,
)


_CHUNK_SIZE = 256


class SelfDistillLabelGenerator:
    """Frozen ckpt_077k (+ optional ckpt_064k ensemble) -> binary ink pseudo-labels."""

    def __init__(
        self,
        *,
        primary_ckpt: str | Path,
        ensemble_ckpt: str | Path,
        primary_threshold: float,
        ensemble_threshold: float,
        mean_hi: float,
        std_lo: float,
        tta: bool = True,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        # Accept and ignore unknown keys so callers can splat a JSON config.
        **_unused,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.thr_primary = float(primary_threshold)
        self.thr_ensemble = float(ensemble_threshold)
        self.mean_hi = float(mean_hi)
        self.std_lo = float(std_lo)
        self.tta = bool(tta)

        # _build_unet returns the raw multi-head UNet (model(x) -> dict of targets).
        # predict_batch expects a tensor-returning module, so wrap each in
        # TargetHeadWrapper to pull out the "ink" head's logits.
        self.primary = TargetHeadWrapper(
            _build_unet(Path(primary_ckpt), self.device, self.dtype),
            target_name="ink",
        ).to(self.device)
        self.ensemble = TargetHeadWrapper(
            _build_unet(Path(ensemble_ckpt), self.device, self.dtype),
            target_name="ink",
        ).to(self.device)
        self.primary.eval()
        self.ensemble.eval()

        self._variants = tta_variants(self.tta)
        # Process TTA variants in small chunks so peak activation memory is
        # bounded. Each forward through a 7-stage UNet at 256^3 in bf16 takes
        # ~3-5 GB of activations, so we want at most ~2 stacked at a time.
        self._tta_batch = max(1, min(2, len(self._variants)))
        self._patch_size = (_CHUNK_SIZE, _CHUNK_SIZE, _CHUNK_SIZE)

    @torch.inference_mode()
    def generate(
        self,
        image_b1zyx: torch.Tensor,
        mask_b1zyx: torch.Tensor | None = None,
        *,
        raw_mean: torch.Tensor,
        raw_std: torch.Tensor,
    ) -> torch.Tensor:
        binarized, _ = self._generate_internal(
            image_b1zyx, mask_b1zyx, raw_mean=raw_mean, raw_std=raw_std, return_debug=False
        )
        return binarized

    @torch.inference_mode()
    def generate_with_debug(
        self,
        image_b1zyx: torch.Tensor,
        mask_b1zyx: torch.Tensor | None = None,
        *,
        raw_mean: torch.Tensor,
        raw_std: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        return self._generate_internal(
            image_b1zyx, mask_b1zyx, raw_mean=raw_mean, raw_std=raw_std, return_debug=True
        )

    def _generate_internal(
        self,
        image_b1zyx: torch.Tensor,
        mask_b1zyx: torch.Tensor | None,
        *,
        raw_mean: torch.Tensor,
        raw_std: torch.Tensor,
        return_debug: bool,
    ):
        if image_b1zyx.shape[-3:] != (_CHUNK_SIZE,) * 3:
            raise ValueError(
                f"expected chunk size {_CHUNK_SIZE}^3, got {tuple(image_b1zyx.shape)}"
            )
        if raw_mean.shape[0] != image_b1zyx.shape[0] or raw_std.shape[0] != image_b1zyx.shape[0]:
            raise ValueError(
                f"raw_mean/raw_std must be [B] tensors matching batch size "
                f"{image_b1zyx.shape[0]}, got {tuple(raw_mean.shape)} / {tuple(raw_std.shape)}"
            )

        img = image_b1zyx.to(device=self.device, dtype=self.dtype, non_blocking=True)
        out = torch.empty_like(image_b1zyx, dtype=torch.float32, device=self.device)
        debug_records: list[dict] = []

        for b in range(img.shape[0]):
            sub = img[b : b + 1]
            use_ensemble = bool(
                float(raw_mean[b].item()) > self.mean_hi
                and float(raw_std[b].item()) < self.std_lo
            )

            prob_primary = predict_batch(
                self.primary,
                sub,
                variants=self._variants,
                tta_batch_size=self._tta_batch,
                patch_size_zyx=self._patch_size,
                foreground_channel=0,
            )  # float32 [1, 1, Z, Y, X] in [0, 1]

            if use_ensemble:
                prob_ensemble = predict_batch(
                    self.ensemble,
                    sub,
                    variants=self._variants,
                    tta_batch_size=self._tta_batch,
                    patch_size_zyx=self._patch_size,
                    foreground_channel=0,
                )
                prob = 0.5 * (prob_primary + prob_ensemble)
                threshold = self.thr_ensemble
            else:
                prob_ensemble = None
                prob = prob_primary
                threshold = self.thr_primary

            if mask_b1zyx is not None:
                mask_sub = mask_b1zyx[b : b + 1].to(device=self.device, dtype=prob.dtype)
                prob = prob * mask_sub
            else:
                mask_sub = None

            binary = (prob > threshold).float()
            out[b : b + 1] = binary

            if return_debug:
                debug_records.append({
                    "use_ensemble": use_ensemble,
                    "threshold": threshold,
                    "prob_primary": prob_primary,
                    "prob_ensemble": prob_ensemble,
                    "prob_combined": prob,
                    "mask": mask_sub,
                    "binarized": binary,
                    "raw_mean": float(raw_mean[b].item()),
                    "raw_std": float(raw_std[b].item()),
                })

        if not return_debug:
            return out, None
        return out, {"per_sample": debug_records}


def make_self_distill_debug_figure(
    *,
    image: np.ndarray,           # [Z, Y, X], normalized [0,1] image_for_label
    original_label: np.ndarray,  # [Z, Y, X] in {0,1}, the dataset's inklabels (pre-substitution)
    input_mask: np.ndarray,      # [Z, Y, X] in {0,1}, raw>50 foreground
    prob_primary: np.ndarray,    # [Z, Y, X] in [0,1]
    prob_combined: np.ndarray,   # [Z, Y, X] in [0,1] (= primary if no ensemble)
    binarized: np.ndarray,       # [Z, Y, X] in {0,1}
    use_ensemble: bool,
    threshold: float,
    raw_mean: float,
    raw_std: float,
    prob_ensemble: np.ndarray | None = None,
    z_index: int | None = None,
):
    """Build a v3 self-distill debug figure at the middle z-slice of a 3D chunk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = image.shape[0] // 2 if z_index is None else int(z_index)
    panels = [
        ("Image",          image[z],          "gray",  None),
        ("Original Label", original_label[z], "gray",  (0.0, 1.0)),
        ("Input Mask",     input_mask[z],     "gray",  (0.0, 1.0)),
        ("Primary prob",   prob_primary[z],   "magma", (0.0, 1.0)),
    ]
    if prob_ensemble is not None:
        panels.append(("Ensemble prob", prob_ensemble[z], "magma", (0.0, 1.0)))
        panels.append(("Mean prob",     prob_combined[z], "magma", (0.0, 1.0)))
    panels.append(("Binarized", binarized[z], "gray", (0.0, 1.0)))

    ncols = 4
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    for ax, (title, arr, cmap, vlim) in zip(np.atleast_1d(axes).flat, panels):
        if vlim is None:
            ax.imshow(arr, cmap=cmap)
        else:
            ax.imshow(arr, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in np.atleast_1d(axes).flat[len(panels):]:
        ax.axis("off")
    branch = "ensemble (077k+064k)" if use_ensemble else "primary (077k only)"
    fig.suptitle(
        f"v3 self-distill — z={z}  branch={branch}  thr={threshold:.4f}  "
        f"raw_mean={raw_mean:.1f}  raw_std={raw_std:.1f}",
        fontsize=10,
    )
    fig.tight_layout()
    return fig
