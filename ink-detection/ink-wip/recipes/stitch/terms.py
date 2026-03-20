from __future__ import annotations

from typing import Any

import torch


def _stitch_loss_terms(recipe) -> tuple[object, ...]:
    if isinstance(recipe, tuple):
        return recipe
    if isinstance(recipe, list):
        return tuple(recipe)
    if recipe is None:
        return ()
    return (recipe,)


def _requires_boundary_dist_map(recipe) -> bool:
    return any(bool(getattr(term, "requires_boundary_dist_map", False)) for term in _stitch_loss_terms(recipe))


def _merge_stitch_term_metrics(out, term_metrics):
    loss = term_metrics.get("loss")
    if loss is None:
        raise TypeError("stitched component loss terms must return a 'loss' metric")
    out["loss"] = out["loss"] + loss

    for key, value in term_metrics.items():
        if key == "loss":
            continue
        out[key] = value


def compute_stitched_component_loss(
    recipe,
    stitched_logits,
    stitched_targets,
    *,
    valid_mask,
    boundary_dist_map=None,
):
    terms = _stitch_loss_terms(recipe)
    if len(terms) <= 0:
        raise TypeError("stitched component loss requires at least one stitched loss term")

    zero = torch.zeros((), device=stitched_logits.device, dtype=stitched_logits.dtype)
    out = {
        "loss": zero.clone(),
        "covered_px": int(valid_mask.sum().detach().item()),
    }

    for term in terms:
        compute = getattr(term, "compute", None)
        if not callable(compute):
            raise TypeError("stitched component loss terms must provide compute(...)")
        term_metrics = compute(
            stitched_logits,
            stitched_targets,
            valid_mask=valid_mask,
            boundary_dist_map=boundary_dist_map,
        )
        _merge_stitch_term_metrics(out, term_metrics)
    return out
