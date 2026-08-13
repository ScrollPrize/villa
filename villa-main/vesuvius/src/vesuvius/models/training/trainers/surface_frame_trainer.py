"""Trainer specialised for surface-frame (t_u, t_v, n) prediction."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Dict, Tuple, Union

import torch

from ..train import BaseTrainer
from ..loss.surface_frame import SurfaceFrameMultiTermLoss


logger = logging.getLogger(__name__)


class SurfaceFrameTrainer(BaseTrainer):
    """Trainer that learns to predict per-voxel surface frames."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Ensure deep supervision is disabled; surface-frame loss works on a single scale.
        self.mgr.enable_deep_supervision = False
        self.mgr.compute_loss_on_labeled_only = True

        # Retain only the surface-frame target.
        targets = getattr(self.mgr, "targets", {}) or {}
        surface_cfg = targets.get("surface_frame")
        if surface_cfg is None:
            raise ValueError("SurfaceFrameTrainer requires a 'surface_frame' target in the configuration.")

        weights = surface_cfg.get("loss_weights", {})
        self.loss_fn = SurfaceFrameMultiTermLoss(
            lambda_direction=weights.get("direction", 1.0),
            lambda_frame=weights.get("frame", 1.0),
            lambda_orthogonality=weights.get("orthogonality", 0.1),
        )

        surface_cfg.setdefault("out_channels", 9)
        surface_cfg.pop("losses", None)  # explicit losses handled by the trainer
        self.mgr.targets = {"surface_frame": surface_cfg}

        self._last_loss_components: Dict[str, float] = {}

    # --------------------------------------------------------------------------------------------
    # Dataset handling

    def _prepare_batch(self, batch: Dict[str, torch.Tensor], *, is_training: bool) -> Dict[str, torch.Tensor]:
        allowed_keys = {"image", "surface_frame", "surface_frame_mask", "patch_info", "is_unlabeled"}
        filtered = {k: v for k, v in batch.items() if k in allowed_keys}
        return filtered

    def _build_loss(self):  # noqa: D401 - base signature
        """Return an empty mapping; loss handled explicitly via mask-aware reduction."""
        return {"surface_frame": []}

    def _should_include_target_in_loss(self, target_name: str) -> bool:
        return target_name == "surface_frame"

    def _compute_train_loss(self, outputs, targets_dict, loss_fns):  # noqa: D401 - base signature
        loss, components = self._surface_frame_loss(outputs, targets_dict)
        self._last_loss_components = components
        return loss, {"surface_frame": loss.detach().cpu().item()}

    def _compute_validation_loss(self, outputs, targets_dict, loss_fns):  # noqa: D401 - base signature
        loss, components = self._surface_frame_loss(outputs, targets_dict)
        self._last_loss_components = components
        return {"surface_frame": loss.detach().cpu().item()}

    def _validation_step(self, model, data_dict, loss_fns, use_amp):  # noqa: D401 - base signature
        data_dict = self._prepare_batch(data_dict, is_training=False)
        inputs = data_dict["image"].to(self.device)
        targets_dict = {
            "surface_frame": data_dict["surface_frame"].to(self.device)
        }
        if "surface_frame_mask" in data_dict:
            targets_dict["surface_frame_mask"] = data_dict["surface_frame_mask"].to(self.device)

        if use_amp:
            if self.device.type == "cuda" and torch.cuda.is_available():
                autocast_ctx = torch.cuda.amp.autocast()
            elif hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(self.device.type)
            else:
                autocast_ctx = nullcontext()
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            outputs = model(inputs)
            loss_dict = self._compute_validation_loss(outputs, targets_dict, loss_fns)

        return loss_dict, inputs, targets_dict, outputs

    def _initialize_evaluation_metrics(self):  # noqa: D401 - base signature
        return {}

    # --------------------------------------------------------------------------------------------
    # Core loss logic

    def _surface_frame_loss(
        self,
        outputs: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        targets: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred = self._select_tensor(outputs.get("surface_frame"))
        target = self._select_tensor(targets.get("surface_frame"))
        if pred is None or target is None:
            raise ValueError("Surface frame tensors are required for loss computation.")

        if pred.shape != target.shape:
            raise ValueError(
                f"Surface frame prediction/target shape mismatch: pred={pred.shape} target={target.shape}"
            )

        mask_tensor = targets.get("surface_frame_mask")
        if isinstance(mask_tensor, (list, tuple)):
            mask_tensor = mask_tensor[0]

        mask_for_logging = (target.abs().sum(dim=1, keepdim=True) > 0).float()
        labeled = mask_for_logging.sum()
        if labeled <= 0:
            raise RuntimeError("Surface frame batch contains no labeled voxels; cannot compute loss.")

        loss, components = self.loss_fn(pred, target, mask=mask_tensor)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Surface frame loss computed: %s (labeled voxels: %s) components=%s",
                float(loss.detach().cpu()),
                int(labeled.item()),
                {k: float(v.detach().cpu()) for k, v in components.items()},
            )

        component_floats = {k: float(v.detach().cpu()) for k, v in components.items()}
        return loss, component_floats

    @staticmethod
    def _select_tensor(value: Union[torch.Tensor, Tuple[torch.Tensor, ...], None]) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value[0]
        return value


    def _prepare_metrics_for_logging(self, *args, **kwargs):  # noqa: D401
        metrics = super()._prepare_metrics_for_logging(*args, **kwargs)
        if self._last_loss_components:
            for name, value in self._last_loss_components.items():
                metrics[f"surface_frame_{name}"] = value
        return metrics


__all__ = ["SurfaceFrameTrainer"]
