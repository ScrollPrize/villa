"""In-memory migrations for model/checkpoint numerical parameterisations."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from config import DEFAULT_GAP_EXPANDER_CAPACITY
from gap_parameterization import (
    DR_PARAMETER_SCALE,
    GAP_PARAMETERIZATION_VERSION,
    LEGACY_EXPONENT_SCALE,
    calibrated_gap_softplus_scale,
    inverse_softplus,
)


_GAP_LOGITS_KEY = "gap_expander_params.logits"
_GAP_INDEX_KEY = "gap_expander_params.winding_first_logit_idx"
_DR_LOGIT_KEY = "dr_per_winding_logit"
_CONFIG_FIELDS = ("cfg", "requested_config", "resolved_config")


def _config_value(config: Mapping, name: str, default=None):
    if name in config:
        return config[name]
    legacy = name.removeprefix("model_")
    return config.get(legacy, default)


def _updated_configs(checkpoint: dict, updates: Mapping) -> dict:
    updated = dict(checkpoint)
    fallback = checkpoint.get("cfg")
    for field in _CONFIG_FIELDS:
        source = checkpoint.get(field, fallback)
        if isinstance(source, Mapping):
            config = dict(source)
            config.update(updates)
            updated[field] = config
    return updated


def _reset_reparameterized_optimizer_state(optimiser_state):
    if not isinstance(optimiser_state, Mapping):
        return optimiser_state
    updated = dict(optimiser_state)
    state = dict(optimiser_state.get("state") or {})
    groups = [dict(group) for group in optimiser_state.get("param_groups") or ()]
    updated["state"] = state
    updated["param_groups"] = groups
    if not isinstance(state, dict) or not isinstance(groups, list):
        return updated
    # FitContext builds group 0 from the sole global-dr parameter and group 2
    # from the sole gap lattice.  Their old Adam moments are not meaningful
    # after a nonlinear change of coordinates, so restart just those moments.
    for group_index in (0, 2):
        if group_index >= len(groups):
            continue
        for parameter_id in groups[group_index].get("params", ()):
            state.pop(parameter_id, None)
    return updated


def migrate_legacy_gap_parameterization(checkpoint):
    """Convert an exponential-gap checkpoint to the stable softplus latent.

    Existing physical gaps above the new numerical floor are preserved.
    Gaps at or below the floor cannot be represented by the new transform and
    are projected to a small, trainable offset above it.  Gap/global-spacing
    Adam moments are reset because their coordinates changed nonlinearly.
    """
    if not isinstance(checkpoint, Mapping):
        return checkpoint
    version = int(checkpoint.get("gap_parameterization_version", 1) or 1)
    if version >= GAP_PARAMETERIZATION_VERSION:
        return checkpoint
    model_state = checkpoint.get("spiral_and_transform")
    config = checkpoint.get("cfg")
    if (not isinstance(model_state, Mapping)
            or _GAP_LOGITS_KEY not in model_state
            or _DR_LOGIT_KEY not in model_state
            or not isinstance(config, Mapping)):
        return checkpoint

    old_logits = model_state[_GAP_LOGITS_KEY]
    old_dr_logit = model_state[_DR_LOGIT_KEY]
    if not (isinstance(old_logits, torch.Tensor)
            and isinstance(old_dr_logit, torch.Tensor)):
        return checkpoint

    nominal_dr = float(_config_value(
        config, "model_initial_dr_per_winding", 16.0))
    lr_scale = float(_config_value(
        config, "model_gap_expander_lr_scale", 0.3))
    min_gap = float(_config_value(
        config, "model_gap_expander_min_gap", 1.0))
    bias = float(_config_value(
        config, "model_gap_expander_softplus_bias", 4.0))
    softplus_scale = calibrated_gap_softplus_scale(
        nominal_dr, min_gap, bias)

    old_dr = float(F.softplus(
        old_dr_logit.detach().to(torch.float64)
        * DR_PARAMETER_SCALE))
    projected_dr = max(old_dr, min_gap + 1.0e-3)
    residual = torch.tensor(
        projected_dr - min_gap, dtype=torch.float64,
        device=old_dr_logit.device)
    new_dr_logit = (
        inverse_softplus(residual) / DR_PARAMETER_SCALE
    ).to(dtype=old_dr_logit.dtype).reshape_as(old_dr_logit)

    denominator = F.softplus(torch.tensor(bias, dtype=torch.float64))
    projection_gap = min_gap + 1.0e-3
    new_logits = torch.empty_like(old_logits)
    projected_count = 0
    # A few z rows at a time avoids materialising several full double-precision
    # copies of production's ~200 MiB gap lattice during checkpoint loading.
    chunk_rows = 4
    for start in range(0, old_logits.shape[-2], chunk_rows):
        stop = min(start + chunk_rows, old_logits.shape[-2])
        old_chunk = old_logits[..., start:stop, :].to(torch.float64)
        old_gap = old_dr * torch.exp(
            old_chunk * (lr_scale * LEGACY_EXPONENT_SCALE))
        if not torch.isfinite(old_gap).all():
            raise ValueError(
                "legacy checkpoint contains exponential inter-winding gaps "
                "outside float64 range and cannot be migrated safely")
        projected_count += int((old_gap < projection_gap).sum())
        target_gap = old_gap.clamp_min(projection_gap)
        softplus_value = (
            (target_gap - min_gap) / (projected_dr - min_gap)
            * denominator)
        argument = inverse_softplus(softplus_value)
        migrated = (
            (argument - bias) / (softplus_scale * lr_scale)
        ).to(old_logits.dtype)
        new_logits[..., start:stop, :].copy_(migrated)

    saved_capacity = None
    saved_indices = model_state.get(_GAP_INDEX_KEY)
    if isinstance(saved_indices, torch.Tensor):
        saved_capacity = int(saved_indices.numel())
    if saved_capacity is None:
        saved_capacity = int(_config_value(
            config, "model_gap_expander_num_windings",
            old_logits.shape[-1]))
    target_capacity = max(
        saved_capacity,
        int(_config_value(
            config, "model_gap_expander_capacity_windings",
            DEFAULT_GAP_EXPANDER_CAPACITY)),
    )

    new_model_state = dict(model_state)
    new_model_state[_GAP_LOGITS_KEY] = new_logits
    new_model_state[_DR_LOGIT_KEY] = new_dr_logit
    updated = dict(checkpoint)
    updated["spiral_and_transform"] = new_model_state
    updated["optimiser"] = _reset_reparameterized_optimizer_state(
        checkpoint.get("optimiser"))
    updated = _updated_configs(updated, {
        "model_gap_expander_capacity_windings": saved_capacity,
        "model_gap_expander_min_gap": min_gap,
        "model_gap_expander_softplus_bias": bias,
    })
    updated["gap_parameterization_version"] = GAP_PARAMETERIZATION_VERSION
    updated["gap_parameterization_migration"] = {
        "source": "legacy_exponential",
        "projected_gap_logits": projected_count,
        "optimizer_moments_reset": True,
    }
    return expand_gap_checkpoint_capacity(updated, target_capacity)


def _capacity_geometry(config: Mapping, capacity: int):
    resolution = float(_config_value(
        config, "model_gap_expander_logit_resolution", 24.0))
    nominal_dr = float(_config_value(
        config, "model_initial_dr_per_winding", 16.0))
    num_by_winding = (
        2.0 * torch.pi * (torch.arange(1, capacity) + 0.5)
        * nominal_dr / resolution + 0.5
    ).to(torch.int64)
    indices = torch.cat([
        torch.zeros(1), torch.cumsum(num_by_winding, dim=0)
    ])
    return num_by_winding, indices


def expand_gap_checkpoint_capacity(checkpoint, target_capacity: int):
    """Append identity latents and zero Adam moments up to ``target_capacity``."""
    if not isinstance(checkpoint, Mapping):
        return checkpoint
    model_state = checkpoint.get("spiral_and_transform")
    config = checkpoint.get("cfg")
    if not isinstance(model_state, Mapping) or not isinstance(config, Mapping):
        return checkpoint
    old_indices = model_state.get(_GAP_INDEX_KEY)
    old_logits = model_state.get(_GAP_LOGITS_KEY)
    if not isinstance(old_indices, torch.Tensor) or not isinstance(old_logits, torch.Tensor):
        return checkpoint
    saved_capacity = int(old_indices.numel())
    target_capacity = int(target_capacity)
    if target_capacity < saved_capacity:
        raise ValueError(
            f"cannot shrink gap-expander capacity from {saved_capacity} to "
            f"{target_capacity} without discarding learned windings")
    if target_capacity == saved_capacity:
        return checkpoint

    _, new_indices_cpu = _capacity_geometry(config, target_capacity)
    old_width = old_logits.shape[-1]
    new_width = int(new_indices_cpu[-1])
    expanded_logits = F.pad(old_logits, (0, new_width - old_width))
    new_indices = new_indices_cpu.to(
        dtype=old_indices.dtype, device=old_indices.device)
    if not torch.equal(new_indices[:saved_capacity], old_indices):
        raise ValueError(
            "gap-expander capacity cannot be expanded because the existing "
            "winding lattice geometry is not a prefix of the requested one")

    new_model_state = dict(model_state)
    new_model_state[_GAP_LOGITS_KEY] = expanded_logits
    new_model_state[_GAP_INDEX_KEY] = new_indices
    updated = dict(checkpoint)
    updated["spiral_and_transform"] = new_model_state

    optimiser = checkpoint.get("optimiser")
    if isinstance(optimiser, Mapping):
        new_optimiser = dict(optimiser)
        new_optimiser["param_groups"] = [
            dict(group) for group in optimiser.get("param_groups") or ()]
        new_optimiser["state"] = {
            parameter_id: (dict(parameter_state)
                           if isinstance(parameter_state, Mapping)
                           else parameter_state)
            for parameter_id, parameter_state
            in (optimiser.get("state") or {}).items()
        }
        for parameter_state in new_optimiser["state"].values():
            if not isinstance(parameter_state, dict):
                continue
            for name, value in list(parameter_state.items()):
                if (isinstance(value, torch.Tensor)
                        and tuple(value.shape) == tuple(old_logits.shape)):
                    parameter_state[name] = F.pad(
                        value, (0, new_width - old_width))
        updated["optimiser"] = new_optimiser

    updated = _updated_configs(updated, {
        "model_gap_expander_capacity_windings": target_capacity,
    })
    return updated


_FLOW_FIELD_PREFIX = "flow_field."
_FLOW_LATTICE_KEYS = ("flow_field.flows.0", "flow_field.flows.1")
_EXTRA_FLOW_FIELD_PREFIX = "extra_flow_fields."
# FitContext's optimiser layout is [ungrouped, linear, gap, low-res flow,
# high-res flow]; the two flow groups held one parameter per stage.
_FLOW_LATTICE_GROUP_INDICES = (3, 4)


def merge_flow_stage_state(model_state):
    """Fold per-stage flow modules into the slab axis of one flow field.

    Multi-stage models written before the slab layout held stage 0 in
    ``flow_field.flows.{0,1}`` (each ``[1, 3, ...]``) and stages 1..N-1 in
    ``extra_flow_fields.{i}.flows.{0,1}``; the slab layout holds all N in
    ``flow_field.flows.{0,1}`` as ``[N, 3, ...]``, stage order preserved. The
    integrated map is unchanged. A state dict without extra stages (every
    single-stage model, whose layout never changed) is returned as is.
    """
    if not isinstance(model_state, Mapping):
        return model_state
    extra_keys = [key for key in model_state
                  if key.startswith(_EXTRA_FLOW_FIELD_PREFIX)]
    if not extra_keys:
        return model_state
    stage_indices = sorted({int(key.split(".")[1]) for key in extra_keys})
    if stage_indices != list(range(len(stage_indices))):
        raise ValueError(
            f"checkpoint flow stages are not contiguous: {stage_indices}")
    merged = {key: value for key, value in model_state.items()
              if not key.startswith(_EXTRA_FLOW_FIELD_PREFIX)}
    for lattice_key in _FLOW_LATTICE_KEYS:
        suffix = lattice_key[len(_FLOW_FIELD_PREFIX):]
        parts = [model_state[lattice_key]] + [
            model_state[f"{_EXTRA_FLOW_FIELD_PREFIX}{index}.{suffix}"]
            for index in stage_indices]
        if any(part.shape[0] != 1 for part in parts):
            raise ValueError(
                "checkpoint flow lattices carry a time axis longer than 1; "
                "time-varying flow fields are no longer supported")
        merged[lattice_key] = torch.cat(parts, dim=0)
    return merged


def _merge_flow_stage_optimizer(optimiser, num_stages):
    """Merge the per-stage flow parameters' optimiser state along the slab axis.

    Saved parameter ids enumerate parameters group by group; the two flow
    groups shrink from ``num_stages`` ids to one each and every later id is
    renumbered. Adam moments (``[1, 3, ...]`` per stage) concatenate in stage
    order; a stage without moments contributes zeros.
    """
    groups = [dict(group) for group in optimiser.get("param_groups") or ()]
    if len(groups) <= max(_FLOW_LATTICE_GROUP_INDICES):
        raise ValueError(
            "checkpoint optimiser has too few parameter groups to hold the "
            f"flow lattices ({len(groups)})")
    state = optimiser.get("state") or {}
    new_state = {}
    next_id = 0
    for index, group in enumerate(groups):
        params = list(group.get("params") or ())
        if index not in _FLOW_LATTICE_GROUP_INDICES:
            for pid in params:
                if pid in state:
                    new_state[next_id] = state[pid]
                next_id += 1
            group["params"] = list(range(next_id - len(params), next_id))
            continue
        if len(params) != num_stages:
            raise ValueError(
                f"checkpoint optimiser group {index} holds {len(params)} flow "
                f"parameters for {num_stages} flow stages")
        moments = [state.get(pid) for pid in params]
        present = [moment for moment in moments if isinstance(moment, Mapping)]
        if present:
            merged = dict(present[0])
            for name, value in present[0].items():
                if not (isinstance(value, torch.Tensor) and value.dim() >= 1
                        and value.shape[0] == 1):
                    continue
                merged[name] = torch.cat([
                    moment[name] if isinstance(moment, Mapping)
                    else torch.zeros_like(value)
                    for moment in moments], dim=0)
            new_state[next_id] = merged
        group["params"] = [next_id]
        next_id += 1
    updated = dict(optimiser)
    updated["param_groups"] = groups
    updated["state"] = new_state
    return updated


def merge_flow_stage_lattices(checkpoint):
    """Bring a checkpoint's flow stages into the slab layout.

    Applies merge_flow_stage_state to the model state and merges the
    optimiser's per-stage flow parameters (see _merge_flow_stage_optimizer).
    Checkpoints already in the
    slab layout are returned unchanged. A checkpoint that used the retired
    time-varying flow (``model_num_flow_timesteps > 1``) is refused: its
    lattices' leading axis meant linear interpolation in time, not stages.
    """
    if not isinstance(checkpoint, Mapping):
        return checkpoint
    config = checkpoint.get("cfg")
    if isinstance(config, Mapping):
        timesteps = _config_value(config, "model_num_flow_timesteps", 1)
        if int(timesteps or 1) != 1:
            raise ValueError(
                f"checkpoint was written with model_num_flow_timesteps="
                f"{timesteps!r}; time-varying flow fields are no longer "
                "supported")
    updated = None
    model_state = checkpoint.get("spiral_and_transform")
    merged_state = merge_flow_stage_state(model_state)
    if merged_state is not model_state:
        updated = dict(checkpoint)
        updated["spiral_and_transform"] = merged_state
        optimiser = checkpoint.get("optimiser")
        if isinstance(optimiser, Mapping):
            updated["optimiser"] = _merge_flow_stage_optimizer(
                optimiser, int(merged_state[_FLOW_LATTICE_KEYS[0]].shape[0]))
    return checkpoint if updated is None else updated
