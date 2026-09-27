"""In-memory migrations of checkpoint tensor layouts and configurations."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
import torch.nn.functional as F


_GAP_LOGITS_KEY = "gap_expander_params.logits"
_GAP_INDEX_KEY = "gap_expander_params.winding_first_logit_idx"
_CONFIG_FIELDS = ("cfg", "requested_config", "resolved_config")


def _config_value(config: Mapping, name: str, default=None):
    return config.get(name, default)


def tolerate_config(stored: Mapping, *, defaults: Mapping):
    """Bring a checkpoint's stored configuration onto the current schema.

    Keys the schema no longer has are dropped and keys the checkpoint
    predates are filled with their current defaults. Every such edit is
    described in the returned notes so the caller can report it. Values are
    then validated against the schema; an invalid one (a retired enum member,
    an out-of-range number) raises ValueError, since no default can say what
    the fit meant.

    Returns ``(config, notes)``. Nothing about the model tensors is inspected
    here; a checkpoint whose parameters do not fit the live model is still
    refused by the preflight's geometry checks.
    """
    from config import Config
    config = dict(stored)
    notes = []
    unknown = sorted(set(config) - set(defaults))
    for key in unknown:
        del config[key]
    if unknown:
        notes.append(
            "drops configuration keys the schema no longer has: "
            + ", ".join(unknown))
    missing = sorted(set(defaults) - set(config))
    for key in missing:
        config[key] = copy.deepcopy(defaults[key])
    if missing:
        notes.append(
            "predates configuration keys, which take their defaults: "
            + ", ".join(f"{key}={defaults[key]!r}" for key in missing))
    Config(config)  # value validation: raises ValueError on an invalid value
    return config, notes


def tolerate_checkpoint_config(checkpoint):
    """Apply tolerate_config to every configuration field of a checkpoint.

    Returns ``(checkpoint, notes)`` with the notes for the durable ``cfg``;
    the requested/resolved copies are normalised the same way. A checkpoint
    without a mapping ``cfg`` is returned unchanged with no notes, and the
    caller's own checks then refuse it.
    """
    if not isinstance(checkpoint, Mapping) or not isinstance(
            checkpoint.get("cfg"), Mapping):
        return checkpoint, []
    from config import Config
    defaults = Config().as_dict()
    updated = dict(checkpoint)
    notes = []
    for field in _CONFIG_FIELDS:
        source = checkpoint.get(field)
        if isinstance(source, Mapping):
            updated[field], field_notes = tolerate_config(
                source, defaults=defaults)
            if field == "cfg":
                notes = field_notes
    return updated, notes


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

