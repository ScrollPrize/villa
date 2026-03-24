from __future__ import annotations

from dataclasses import dataclass

import torch

_BETTI_MATCHING_LOSS_CLASS = None
_BETTI_MATCHING_LOSS_CACHE = {}


def _get_betti_matching_loss_class():
    global _BETTI_MATCHING_LOSS_CLASS
    if _BETTI_MATCHING_LOSS_CLASS is not None:
        return _BETTI_MATCHING_LOSS_CLASS

    try:
        from topolosses.losses import BettiMatchingLoss
    except Exception:
        try:
            from topolosses.losses.betti_matching import BettiMatchingLoss
        except Exception as exc:
            raise ImportError(
                "Betti matching training loss requires `topolosses`. "
                "Install `topolosses==0.2.0` on the training environment."
            ) from exc

    _BETTI_MATCHING_LOSS_CLASS = BettiMatchingLoss
    return _BETTI_MATCHING_LOSS_CLASS


def _resolve_betti_matching_loss_module(*, filtration_type, num_processes):
    key = (str(filtration_type).strip().lower(), int(num_processes))
    module = _BETTI_MATCHING_LOSS_CACHE.get(key)
    if module is not None:
        return module

    betti_matching_loss_cls = _get_betti_matching_loss_class()
    common_kwargs = {
        "filtration_type": key[0],
        "num_processes": key[1],
        "sigmoid": False,
        "softmax": False,
        "include_background": False,
    }
    try:
        module = betti_matching_loss_cls(use_base_loss=False, **common_kwargs)
    except TypeError:
        module = betti_matching_loss_cls(use_base_component=False, **common_kwargs)

    _BETTI_MATCHING_LOSS_CACHE[key] = module
    return module


def compute_binary_betti_matching_loss(
    logits,
    targets,
    *,
    valid_mask=None,
    filtration_type="superlevel",
    num_processes=1,
):
    if tuple(logits.shape) != tuple(targets.shape):
        raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    if logits.ndim != 4:
        raise ValueError(f"binary topology helpers expect shape (N, C, H, W), got {tuple(logits.shape)}")

    probs = torch.sigmoid(logits)
    targets = targets.to(device=probs.device, dtype=probs.dtype)
    if valid_mask is not None:
        topology_mask = valid_mask.to(device=probs.device, dtype=probs.dtype)
        if tuple(topology_mask.shape) != tuple(probs.shape):
            raise ValueError(
                f"valid_mask shape must match logits shape, got {tuple(topology_mask.shape)} vs {tuple(probs.shape)}"
            )
        probs = probs * topology_mask
        targets = targets * topology_mask

    loss_module = _resolve_betti_matching_loss_module(
        filtration_type=filtration_type,
        num_processes=num_processes,
    )
    loss = loss_module(probs, targets)
    if not torch.is_tensor(loss):
        loss = torch.as_tensor(loss, device=probs.device, dtype=probs.dtype)
    if loss.ndim == 0:
        loss = loss.unsqueeze(0)
    return loss


@dataclass(frozen=True)
class StitchBettiMatchingLoss:
    weight: float = 1.0
    filtration_type: str = "superlevel"
    num_processes: int = 1

    def compute(self, batch):
        betti_matching_loss = compute_binary_betti_matching_loss(
            batch.logits[None, None],
            batch.targets[None, None],
            valid_mask=batch.valid_mask[None, None],
            filtration_type=self.filtration_type,
            num_processes=self.num_processes,
        )[0]
        return {
            "loss": float(self.weight) * betti_matching_loss,
            "betti_matching_loss": betti_matching_loss,
        }
