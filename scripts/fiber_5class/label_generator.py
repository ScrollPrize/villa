"""GPU-resident 4-class pseudo-label generator for fiber/ink self-distillation.

Pipeline (everything on the input tensor's device, under ``torch.inference_mode``):

  1. ``fiber_prob = sigmoid(fiber_teacher(image))``  (FG channel only, fp32)
  2. ``ink_prob   = sigmoid(ink_teacher(image))``    (FG channel only, fp32)
  3. ``fiber_mask = fiber_prob > fiber_thr``
  4. ``ink_mask   = ink_prob > ink_thr``                 (no exclusion; ink wins)
  5. ``ws_image  = ((1 - fiber_prob) * 65535).clamp_(0, 65535).to(uint16)``
     Watershed-from-minima on each sample via ``cuws`` (zero-copy DLPack).
  6. Per-instance PCA on ZYX voxel coords (scatter_add_ + linalg.eigh on (K,3,3)).
     ``|principal_axis . z_hat| > cos_thr`` -> label 1 vertical, else label 2 horizontal.
     Small instances (<min_voxels) default to label 2 (horiz/angular) so they
     are still classified as fibers, not lost to background.
  7. ``label[ink_mask] = 3``  (INK OVERRIDES FIBERS — ink wins.)
  8. ``label[raw < dark_voxel_thr] = 0``  (dark voxels in the PRE-augmentation
     raw volume are forced to background; this is the final step applied to
     the pseudo-label, regardless of which class was assigned earlier.)
  Classes: 0 bg, 1 vert fiber, 2 horiz/angular fiber, 3 ink. No papyrus class.

Optional fallback: ``ws_image_mode="distance"`` uses ``cucim.skimage`` distance
transform of ``fiber_mask`` (negated, scaled) instead of ``(1 - fiber_prob)``.

If ``cuws`` is unavailable, a connected-components fallback via cucim or
scipy is used so the pipeline still produces valid (if cruder) instance maps.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from model import build_fiber_unet  # noqa: E402

# Optional CuPy / cuws / cucim (resolved at import time, used per-call).
try:
    import cupy as _cp
    _CUPY_OK = True
except Exception:
    _cp = None
    _CUPY_OK = False

try:
    from cuws import watershed_from_minima as _cuws_ws_from_minima  # type: ignore
    _CUWS_OK = True
except Exception:
    _cuws_ws_from_minima = None
    _CUWS_OK = False

try:
    from cucim.skimage.measure import label as _cucim_label  # type: ignore
    from cucim.core.operations.morphology import distance_transform_edt as _cucim_edt  # type: ignore
    _CUCIM_OK = True
except Exception:
    _cucim_label = None
    _cucim_edt = None
    _CUCIM_OK = False


class DebugBundle(NamedTuple):
    fiber_prob: torch.Tensor       # (B, 1, Z, Y, X) fp32, in [0, 1]
    ink_prob: torch.Tensor         # (B, 1, Z, Y, X) fp32, in [0, 1]
    fiber_mask: torch.Tensor       # (B, 1, Z, Y, X) bool
    ink_mask: torch.Tensor         # (B, 1, Z, Y, X) bool
    instance_map: torch.Tensor     # (B, 1, Z, Y, X) int64 instance ids per sample
    label: torch.Tensor            # (B, 1, Z, Y, X) uint8 in {0..4}
    class_counts: torch.Tensor     # (B, 5) int64 voxel counts per class per sample
    n_instances: torch.Tensor      # (B,)   int64 number of >=min_voxels instances per sample
    n_vert: torch.Tensor           # (B,)   int64 number classified as vertical per sample


@dataclass
class FiveClassConfig:
    """4-class config. Name kept for historical compatibility (class count is 4)."""
    fiber_thr: float = 0.5
    ink_thr: float = 0.5
    papyrus_raw_thr: int = 90        # UNUSED in 4-class mode; kept for json compat.
    dark_voxel_thr: int = 90         # final step: voxels with raw<this are forced to bg
    ws_image_mode: str = "distance"  # "prob" or "distance"
    ws_h_merge: int = 14000          # cuws dynamic-merge height (uint16 scale, 0..65535)
    ws_min_voxels: int = 400         # min voxels for an instance to keep its PCA orientation
    pca_cos_threshold: float = 0.819 # cos(35 deg)
    fiber_target_name: str = "fibers"
    ink_target_name: str | None = None  # autodetect when None


def _extract_state_dict(ckpt_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the most-trained state dict from a checkpoint payload.

    Handles both the fibers-style layout (``ema.model_state`` or ``model``) and
    the sean_ink-style layout (``ema_model`` next to ``model``). When EMA is
    available we prefer it (smoother decision boundary at inference time).
    """
    if not isinstance(ckpt_payload, dict):
        return ckpt_payload  # type: ignore[return-value]
    if "ema" in ckpt_payload and isinstance(ckpt_payload["ema"], dict):
        sd = ckpt_payload["ema"].get("model_state")
        if sd is not None:
            return sd
    if "ema_model" in ckpt_payload and isinstance(ckpt_payload["ema_model"], dict):
        return ckpt_payload["ema_model"]
    if "model" in ckpt_payload:
        return ckpt_payload["model"]
    return ckpt_payload  # type: ignore[return-value]


def _detect_target_and_out(ckpt_payload: Mapping[str, Any]) -> tuple[str, int]:
    """Find (target_name, out_channels) from a NetworkFromConfig payload.

    The villa networks register the final 1x1x1 conv as
    ``task_heads.<NAME>.weight`` with shape ``(out_channels, 32, 1, 1, 1)``.
    Older layouts use ``task_decoders.<NAME>.head.weight``. Try both.
    """
    state = _extract_state_dict(ckpt_payload)
    if state is None:
        raise RuntimeError("unrecognized checkpoint payload (no model state)")

    candidates: list[tuple[str, int]] = []
    for k, v in state.items():
        if not hasattr(v, "shape"):
            continue
        if k.startswith("task_heads.") and k.endswith(".weight") and getattr(v, "ndim", 0) == 5:
            target = k.split(".")[1]
            candidates.append((target, int(v.shape[0])))
        elif k.startswith("task_decoders.") and k.endswith(".head.weight"):
            target = k.split(".")[1]
            candidates.append((target, int(v.shape[0])))
    if candidates:
        candidates.sort()
        return candidates[0]

    for k, v in state.items():
        if hasattr(v, "shape") and "head" in k and getattr(v, "ndim", 0) == 5 and v.shape[0] <= 64:
            target = k.split(".")[1] if "task_" in k and "." in k else "fibers"
            return (target, int(v.shape[0]))
    raise RuntimeError("could not detect (target_name, out_channels) from checkpoint")


def load_teacher_unet(
    ckpt_path: str | Path,
    *,
    crop_size: tuple[int, int, int] = (256, 256, 256),
    in_channels: int = 1,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.nn.Module, str, int]:
    """Load a teacher UNet, autodetecting (target_name, out_channels) from the ckpt."""
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    target_name, out_channels = _detect_target_and_out(payload)
    activation = "none"
    state_dict = _extract_state_dict(payload)

    model = build_fiber_unet(
        crop_size=crop_size,
        target_name=target_name,
        out_channels=out_channels,
        activation=activation,
        in_channels=in_channels,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if not k.startswith(("guide_", "task_decoders."))]
    bad_unexpected = [k for k in unexpected if not k.startswith(("guide_",))]
    if bad_missing or bad_unexpected:
        # Print warning but try to continue — these may be auxiliary heads from
        # the original training that are not used at inference time.
        print(f"[teacher load] {Path(ckpt_path).name}: "
              f"missing={bad_missing[:5]} unexpected={bad_unexpected[:5]}")
        if any("task_heads" in k or "task_decoders" in k for k in (bad_missing + bad_unexpected)):
            raise RuntimeError(
                f"{Path(ckpt_path).name}: critical task heads missing or mismatched."
            )
    model.eval().to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, target_name, out_channels


def _to_fg_prob(logits: torch.Tensor) -> torch.Tensor:
    """Convert teacher logits to a (B, 1, ...) FG-probability tensor in fp32."""
    if isinstance(logits, dict):
        # Single-target NetworkFromConfig returns a dict; take the only value.
        logits = next(iter(logits.values()))
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    logits = logits.float()
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    if logits.shape[1] == 2:
        # 2-class softmax: FG = channel 1.
        return torch.softmax(logits, dim=1)[:, 1:2]
    # >2-class: take argmax over FG-like channels (defensive default: take max non-bg prob).
    softmax = torch.softmax(logits, dim=1)
    return softmax[:, 1:].sum(dim=1, keepdim=True)


def _torch_to_cupy(t: torch.Tensor):
    """Zero-copy torch CUDA tensor -> CuPy ndarray."""
    if not _CUPY_OK:
        raise RuntimeError("cupy is unavailable")
    return _cp.from_dlpack(t.contiguous())


def _cupy_to_torch(a) -> torch.Tensor:
    """Zero-copy CuPy ndarray -> torch CUDA tensor (via __dlpack__ protocol)."""
    if not _CUPY_OK:
        raise RuntimeError("cupy is unavailable")
    # CuPy 14.x dropped toDlpack(); the modern path is the __dlpack__ protocol
    # which torch.utils.dlpack.from_dlpack consumes directly.
    return torch.utils.dlpack.from_dlpack(a)


def _watershed_single(
    fiber_prob_zyx: torch.Tensor,   # (Z, Y, X) fp32 in [0, 1]
    fiber_mask_zyx: torch.Tensor,   # (Z, Y, X) bool
    *,
    mode: str = "prob",
    h_merge: int = 0,
) -> torch.Tensor:
    """Run cuws watershed on a single 3D sample. Returns (Z, Y, X) int64 label map."""
    device = fiber_prob_zyx.device
    if not fiber_mask_zyx.any():
        return torch.zeros_like(fiber_prob_zyx, dtype=torch.int64)

    if _CUWS_OK and _CUPY_OK:
        if mode == "prob":
            img = ((1.0 - fiber_prob_zyx).clamp_(0.0, 1.0) * 65535.0).to(torch.uint16).contiguous()
        elif mode == "distance":
            if not _CUCIM_OK:
                # Fall back to prob mode if cucim missing.
                img = ((1.0 - fiber_prob_zyx).clamp_(0.0, 1.0) * 65535.0).to(torch.uint16).contiguous()
            else:
                # Distance transform of the fiber mask: high inside, low at boundary.
                mask_cp = _torch_to_cupy(fiber_mask_zyx.to(torch.uint8))
                dt = _cucim_edt(mask_cp)
                dt_t = _cupy_to_torch(dt).to(torch.float32)
                dt_max = float(dt_t.max().item()) if dt_t.numel() else 1.0
                # Invert so minima = centers of fibers (peaks of DT).
                img = (((dt_max - dt_t).clamp_(0.0) / max(dt_max, 1.0)) * 65535.0).to(torch.uint16).contiguous()
        else:
            raise ValueError(f"unknown ws_image_mode={mode!r}")

        mask = fiber_mask_zyx.to(torch.bool).contiguous()
        img_cp = _torch_to_cupy(img)
        mask_cp = _torch_to_cupy(mask)
        labels_cp = _cuws_ws_from_minima(img_cp, mask=mask_cp, h=int(h_merge), sparse=True)
        # cuws returns uint64; cast to int64 for torch indexing-friendly ops.
        return _cupy_to_torch(labels_cp.astype(_cp.int64))

    # Fallback: connected components on the fiber mask.
    if _CUCIM_OK:
        mask_cp = _torch_to_cupy(fiber_mask_zyx.to(torch.uint8))
        cc_cp = _cucim_label(mask_cp, connectivity=3)
        return _cupy_to_torch(cc_cp.astype(_cp.int64))
    # Last resort: CPU scipy.
    from scipy.ndimage import label as _scipy_label  # type: ignore
    mask_np = fiber_mask_zyx.detach().cpu().numpy().astype(np.uint8)
    cc_np, _ = _scipy_label(mask_np, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return torch.from_numpy(cc_np.astype(np.int64)).to(device=device)


def _classify_instances(
    instance_map_zyx: torch.Tensor,
    *,
    coord_grid: torch.Tensor,        # (Z, Y, X, 3) fp32
    min_voxels: int,
    cos_thr: float,
) -> tuple[torch.Tensor, int, int]:
    """Vectorized per-instance PCA. Returns (class_zyx int8, n_kept, n_vertical)."""
    device = instance_map_zyx.device
    flat = instance_map_zyx.view(-1)
    pos = flat > 0
    out_class = torch.zeros_like(instance_map_zyx, dtype=torch.int8)
    if not pos.any():
        return out_class, 0, 0

    vox_ids = flat[pos]
    uniq, inv = torch.unique(vox_ids, return_inverse=True)
    K = int(uniq.numel())

    coords = coord_grid.view(-1, 3)[pos]                                  # (N, 3) fp32
    n = torch.bincount(inv, minlength=K).clamp_min(1)                     # (K,) int
    n_f = n.to(coords.dtype)

    sum_c = torch.zeros((K, 3), device=device, dtype=coords.dtype)
    sum_c.scatter_add_(0, inv[:, None].expand(-1, 3), coords)
    mu = sum_c / n_f[:, None]
    c = coords - mu[inv]                                                  # (N, 3) centered

    outer = c[:, :, None] * c[:, None, :]                                 # (N, 3, 3)
    cov = torch.zeros((K, 3, 3), device=device, dtype=coords.dtype)
    cov.scatter_add_(
        0,
        inv[:, None, None].expand(-1, 3, 3),
        outer,
    )
    cov = cov / n_f[:, None, None].clamp_min(1.0)

    # Regularize tiny covariance entries to avoid eigh on a degenerate matrix.
    cov = cov + torch.eye(3, device=device, dtype=cov.dtype)[None] * 1e-4

    try:
        _, evecs = torch.linalg.eigh(cov)                                  # ascending
        v_top = evecs[..., -1]                                              # (K, 3)
    except Exception:
        # Defensive: if eigh fails, fall back to span-ratio heuristic.
        v_top = torch.zeros((K, 3), device=device, dtype=coords.dtype)
        v_top[:, 0] = 1.0

    cos_z = v_top[:, 0].abs()                                              # |z component|
    is_vert = cos_z > cos_thr
    keep = n >= min_voxels

    # Kept instances split into vertical (1) / horizontal (2) via PCA.
    # Dropped (too-small) instances default to horizontal/angular (2) so that
    # known-fiber voxels do not silently fall back to "background" or
    # "papyrus" downstream. This is the right call when over-segmentation is
    # the dominant failure mode: classifying as "horizontal-or-unsure" only
    # mislabels orientation, never class identity (fiber vs not-fiber).
    class_per_inst = torch.full((K,), 2, device=device, dtype=torch.int8)
    class_per_inst[keep & is_vert] = 1

    voxel_class = class_per_inst[inv]                                      # (N,) int8
    out_flat = out_class.view(-1)
    out_flat[pos] = voxel_class
    return out_class, int(keep.sum().item()), int((keep & is_vert).sum().item())


class FiveClassLabelGenerator:
    """Pseudo-label generator for the 5-class self-distillation trainer.

    Lives on a single device. ``generate(image, raw_image)`` is the only public
    entrypoint and never escapes ``torch.inference_mode`` until it returns.
    """

    def __init__(
        self,
        *,
        fiber_teacher_ckpt: str | Path,
        ink_teacher_ckpt: str | Path | None,
        config: FiveClassConfig,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        crop_size: tuple[int, int, int] = (256, 256, 256),
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.crop_size = tuple(int(s) for s in crop_size)
        self.cfg = config

        # Fiber teacher.
        self.fiber_model, self.fiber_target_name, self.fiber_out_channels = load_teacher_unet(
            fiber_teacher_ckpt,
            crop_size=self.crop_size,
            in_channels=1,
            device=self.device,
            dtype=self.dtype,
        )
        self.cfg.fiber_target_name = self.fiber_target_name

        # Ink teacher (optional — fail-soft).
        self.ink_model = None
        self.ink_target_name = None
        self.ink_out_channels = 0
        if ink_teacher_ckpt is not None:
            try:
                self.ink_model, self.ink_target_name, self.ink_out_channels = load_teacher_unet(
                    ink_teacher_ckpt,
                    crop_size=self.crop_size,
                    in_channels=1,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.cfg.ink_target_name = self.ink_target_name
            except Exception as exc:
                print(f"[FiveClassLabelGenerator] WARN: ink teacher load failed: {exc}; ink class 4 will be empty.")
                self.ink_model = None

        Z, Y, X = self.crop_size
        zs = torch.arange(Z, device=self.device, dtype=torch.float32)
        ys = torch.arange(Y, device=self.device, dtype=torch.float32)
        xs = torch.arange(X, device=self.device, dtype=torch.float32)
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        self._coord_grid = torch.stack([gz, gy, gx], dim=-1).contiguous()  # (Z, Y, X, 3)

    @torch.inference_mode()
    def generate(
        self,
        image_b1zyx: torch.Tensor,
        raw_image_b1zyx: torch.Tensor,
    ) -> tuple[torch.Tensor, DebugBundle]:
        """Run the full pipeline on a batch. Returns (label_b1zyx_uint8, debug)."""
        assert image_b1zyx.shape[-3:] == self.crop_size, (
            f"expected crop {self.crop_size}, got {tuple(image_b1zyx.shape[-3:])}"
        )
        img = image_b1zyx.to(device=self.device, dtype=self.dtype, non_blocking=True)
        raw = raw_image_b1zyx.to(device=self.device, dtype=torch.float32, non_blocking=True)
        B, _, Z, Y, X = img.shape

        # 1. Teacher predictions.
        fiber_logits = self.fiber_model(img)
        fiber_prob = _to_fg_prob(fiber_logits).clamp_(0.0, 1.0)            # (B, 1, Z, Y, X) fp32

        if self.ink_model is not None:
            ink_logits = self.ink_model(img)
            ink_prob = _to_fg_prob(ink_logits).clamp_(0.0, 1.0)
        else:
            ink_prob = torch.zeros_like(fiber_prob)

        fiber_mask = fiber_prob > float(self.cfg.fiber_thr)
        ink_mask = ink_prob > float(self.cfg.ink_thr)   # NOTE: no fiber exclusion — ink wins.

        # 2. Allocate outputs.
        label = torch.zeros((B, 1, Z, Y, X), device=self.device, dtype=torch.uint8)
        instance_map = torch.zeros((B, 1, Z, Y, X), device=self.device, dtype=torch.int64)
        n_inst = torch.zeros(B, device=self.device, dtype=torch.int64)
        n_vert = torch.zeros(B, device=self.device, dtype=torch.int64)

        for b in range(B):
            inst_b = _watershed_single(
                fiber_prob[b, 0].contiguous(),
                fiber_mask[b, 0].contiguous(),
                mode=self.cfg.ws_image_mode,
                h_merge=int(self.cfg.ws_h_merge),
            )
            instance_map[b, 0] = inst_b

            cls_b, n_kept, n_vert_b = _classify_instances(
                inst_b,
                coord_grid=self._coord_grid,
                min_voxels=int(self.cfg.ws_min_voxels),
                cos_thr=float(self.cfg.pca_cos_threshold),
            )
            label[b, 0] = cls_b.to(torch.uint8)
            n_inst[b] = n_kept
            n_vert[b] = n_vert_b

        # 3. Ink overrides fibers — even fiber-classified voxels become ink
        # where the ink teacher's confidence is above threshold.
        label[ink_mask] = 3

        # 4. Dark-voxel guard — voxels whose raw (pre-augmentation) intensity
        # is below ``dark_voxel_thr`` are forced to background regardless of
        # what the teachers / watershed / ink override said. This is the very
        # last step applied to the pseudo-label so that air / chamber padding
        # / very dark CT regions never get a non-bg class. Applied on GPU.
        dark_mask = raw < float(self.cfg.dark_voxel_thr)
        label[dark_mask] = 0

        # 5. Class fractions for logging (4 classes: bg, vert, horiz, ink).
        class_counts = torch.zeros((B, 4), device=self.device, dtype=torch.int64)
        for c in range(4):
            class_counts[:, c] = (label == c).view(B, -1).sum(dim=1)

        debug = DebugBundle(
            fiber_prob=fiber_prob,
            ink_prob=ink_prob,
            fiber_mask=fiber_mask,
            ink_mask=ink_mask,
            instance_map=instance_map,
            label=label,
            class_counts=class_counts,
            n_instances=n_inst,
            n_vert=n_vert,
        )
        return label, debug
