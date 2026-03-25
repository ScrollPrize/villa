from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from koine_machines.training.loss.betti_matching import BettiMatchingLoss
from vesuvius.models.training.loss.nnunet_losses import LabelSmoothedDCAndBCELoss


class WeightedLossTerm(nn.Module):
    def __init__(self, name: str, weight: float, module: nn.Module, metric_name: Optional[str] = None):
        super().__init__()
        self.name = str(name)
        self.weight = float(weight)
        self.module = module
        self.metric_name = str(metric_name or name)


def _sanitize_metric_name(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name))
    cleaned = cleaned.strip("_")
    return cleaned or "term"


def _split_loss_output(output) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(output, tuple):
        if len(output) != 2:
            raise ValueError(f"Expected (loss, aux_dict) tuple, got tuple of length {len(output)}")
        loss_value, aux = output
        if not isinstance(aux, dict):
            raise TypeError(f"Expected aux metrics dict, got {type(aux).__name__}")
        return loss_value.reshape(()), aux
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected loss tensor, got {type(output).__name__}")
    return output.reshape(()), {}


def _build_label_smoothed_dice_bce_term(term_cfg: dict, config: dict) -> nn.Module:
    return LabelSmoothedDCAndBCELoss(
        bce_kwargs=dict(term_cfg.get("bce_kwargs", {})),
        soft_dice_kwargs={
            "label_smoothing": float(
                term_cfg.get(
                    "dice_label_smoothing",
                    config.get("dice_label_smoothing", 0.0),
                )
            ),
        },
        weight_dice=float(term_cfg.get("weight_dice", term_cfg.get("dice_weight", 1.0))),
        weight_ce=float(term_cfg.get("weight_ce", term_cfg.get("ce_weight", 1.0))),
        use_ignore_label=True,
        bce_label_smoothing=float(
            term_cfg.get(
                "bce_label_smoothing",
                config.get("bce_label_smoothing", 0.0),
            )
        ),
    )


def _build_betti_matching_term(term_cfg: dict, _config: dict) -> nn.Module:
    return BettiMatchingLoss(
        filtration=str(term_cfg.get("filtration", "superlevel")),
        include_unmatched_target=bool(term_cfg.get("include_unmatched_target", False)),
        push_unmatched_to=str(term_cfg.get("push_unmatched_to", "diagonal")),
    )


LOSS_TERM_BUILDERS = {
    "LabelSmoothedDCAndBCELoss": _build_label_smoothed_dice_bce_term,
    "BettiMatchingLoss": _build_betti_matching_term,
}


class CompositeLoss(nn.Module):
    def __init__(self, terms: List[WeightedLossTerm]):
        super().__init__()
        if not terms:
            raise ValueError("CompositeLoss requires at least one term")
        self.terms = nn.ModuleList(terms)
        self.latest_metrics: Dict[str, float] = {}

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = net_output.new_zeros(())
        metrics: Dict[str, float] = {}
        used_metric_names = set()

        for index, term in enumerate(self.terms):
            raw_output = term.module(net_output, target)
            raw_loss, aux_metrics = _split_loss_output(raw_output)
            weighted_loss = raw_loss * term.weight
            total = total + weighted_loss

            metric_name = _sanitize_metric_name(term.metric_name)
            if metric_name in used_metric_names:
                metric_name = f"{metric_name}_{index}"
            used_metric_names.add(metric_name)

            metrics[f"loss_terms/{metric_name}_raw"] = float(raw_loss.detach().item())
            metrics[f"loss_terms/{metric_name}_weighted"] = float(weighted_loss.detach().item())

            for key, value in aux_metrics.items():
                aux_key = _sanitize_metric_name(key)
                metrics[f"loss_aux/{metric_name}/{aux_key}"] = float(value.detach().item())

        metrics["loss/total"] = float(total.detach().item())
        self.latest_metrics = metrics
        return total


ProjectedSegmentationLoss = CompositeLoss


def _default_terms(config: dict, loss_cfg: dict) -> List[dict]:
    return [
        {
            "name": "LabelSmoothedDCAndBCELoss",
            "metric_name": "base",
            "weight": 1.0,
            "weight_dice": float(loss_cfg.get("dice_weight", 0.25)),
            "weight_ce": float(loss_cfg.get("ce_weight", 1.0)),
            "dice_label_smoothing": float(
                loss_cfg.get("dice_label_smoothing", config.get("dice_label_smoothing", 0.0))
            ),
            "bce_label_smoothing": float(
                loss_cfg.get("bce_label_smoothing", config.get("bce_label_smoothing", 0.0))
            ),
        }
    ]


def _normalize_terms_config(config: dict) -> List[dict]:
    loss_cfg = dict(config.get("loss", {}) or {})
    raw_terms = loss_cfg.get("terms")
    if raw_terms is None:
        return _default_terms(config, loss_cfg)
    if not isinstance(raw_terms, list) or len(raw_terms) == 0:
        raise ValueError("loss.terms must be a non-empty list when provided")
    return [dict(term or {}) for term in raw_terms]


def create_loss_from_config(config: dict) -> CompositeLoss:
    terms_cfg = _normalize_terms_config(config)
    terms: List[WeightedLossTerm] = []

    for idx, term_cfg in enumerate(terms_cfg):
        name = term_cfg.get("name")
        if not name:
            raise ValueError(f"loss term at index {idx} is missing required key 'name'")
        weight = float(term_cfg.get("weight", 1.0))
        if weight == 0.0:
            continue
        builder = LOSS_TERM_BUILDERS.get(str(name))
        if builder is None:
            supported = ", ".join(sorted(LOSS_TERM_BUILDERS))
            raise ValueError(f"Unsupported loss term {name!r}. Supported terms: {supported}")

        module = builder(term_cfg, config)
        terms.append(
            WeightedLossTerm(
                name=str(name),
                weight=weight,
                module=module,
                metric_name=term_cfg.get("metric_name"),
            )
        )

    if not terms:
        raise ValueError("All configured loss terms have zero weight; at least one active term is required")

    return CompositeLoss(terms)
