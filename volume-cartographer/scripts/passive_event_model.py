#!/usr/bin/env python3
"""Shared passive Callgrind event-cost model helpers."""

from __future__ import annotations

import numpy as np


LEGACY_FEATURE_NAMES = (
    "non_data_instructions",
    "data_writes",
    "l1_data_misses",
    "last_level_data_misses",
    "branch_misses",
    "branch_weighted_l1_misses",
)
FEATURE_NAMES = (*LEGACY_FEATURE_NAMES, "l1_miss_serial_pressure")
DATA_READ_FEATURE_NAMES = (
    "non_data_instructions",
    "data_reads",
    *LEGACY_FEATURE_NAMES[1:],
)
SPLIT_CACHE_FEATURE_NAMES = (
    "non_data_instructions",
    "data_writes",
    "l1_data_read_misses",
    "l1_data_write_misses",
    "last_level_data_read_misses",
    "last_level_data_write_misses",
    "branch_misses",
    "branch_weighted_l1_misses",
)
DATA_READ_SPLIT_CACHE_FEATURE_NAMES = (
    "non_data_instructions",
    "data_reads",
    *SPLIT_CACHE_FEATURE_NAMES[1:],
)


def profile_feature_values(events: dict[str, int]) -> dict[str, float]:
    l1_read_misses = events["D1mr"]
    l1_write_misses = events["D1mw"]
    last_level_read_misses = events["DLmr"]
    last_level_write_misses = events["DLmw"]
    l1_misses = l1_read_misses + l1_write_misses
    branch_misses = events["Bcm"] + events["Bim"]
    instructions = max(events["Ir"], 1)
    return {
        "non_data_instructions": max(0, events["Ir"] - events["Dr"] - events["Dw"]),
        "data_reads": events["Dr"],
        "data_writes": events["Dw"],
        "l1_data_misses": l1_misses,
        "l1_data_read_misses": l1_read_misses,
        "l1_data_write_misses": l1_write_misses,
        "last_level_data_misses": last_level_read_misses + last_level_write_misses,
        "last_level_data_read_misses": last_level_read_misses,
        "last_level_data_write_misses": last_level_write_misses,
        "branch_misses": branch_misses,
        "branch_weighted_l1_misses": l1_misses * branch_misses / instructions,
        "l1_miss_serial_pressure": l1_misses * l1_misses / instructions,
    }


def profile_features(events: dict[str, int]) -> np.ndarray:
    values = profile_feature_values(events)
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=float)


def modeled_feature_cost_ns(
    values: np.ndarray, coefficients: np.ndarray, overlap: float
) -> float:
    if not 0.0 <= overlap <= 1.0:
        raise RuntimeError("event-cost model has an invalid stall overlap")
    if values.shape != coefficients.shape or len(values) not in (6, 7, 8, 9):
        raise RuntimeError("event-cost model has an invalid feature count")
    if overlap == 0.0:
        return float(values @ coefficients)
    if len(values) != len(LEGACY_FEATURE_NAMES):
        raise RuntimeError("extended event-cost models require zero overlap")
    contributions = values * coefficients
    non_stall = contributions[0] + contributions[1]
    shared_interaction = contributions[5]
    branch_stall = contributions[4] + 0.5 * shared_interaction
    cache_stall = contributions[2] + contributions[3] + 0.5 * shared_interaction
    return float(
        non_stall
        + branch_stall
        + cache_stall
        - overlap * min(branch_stall, cache_stall)
    )


def features_for_model(
    events: dict[str, int], feature_names: tuple[str, ...]
) -> np.ndarray:
    supported = {
        LEGACY_FEATURE_NAMES,
        FEATURE_NAMES,
        DATA_READ_FEATURE_NAMES,
        SPLIT_CACHE_FEATURE_NAMES,
        DATA_READ_SPLIT_CACHE_FEATURE_NAMES,
    }
    if feature_names not in supported:
        raise RuntimeError("event-cost model has an unsupported feature basis")
    values = profile_feature_values(events)
    return np.asarray([values[name] for name in feature_names], dtype=float)


def modeled_profile_cost_ns(
    events: dict[str, int], model: dict[str, object]
) -> float:
    feature_names = tuple(model.get("feature_names", ()))
    values = features_for_model(events, feature_names)
    coefficients = np.asarray(model["coefficients_ns"], dtype=float)
    if coefficients.shape != values.shape:
        raise RuntimeError("event-cost model has an invalid coefficient count")
    return modeled_feature_cost_ns(
        values,
        coefficients,
        float(model.get("stall_overlap_fraction", 0.0)),
    )


def modeled_thread_costs_ns(
    profiles: dict[int, dict[str, int]], model: dict[str, object]
) -> dict[int, float]:
    return {
        thread: modeled_profile_cost_ns(events, model)
        for thread, events in profiles.items()
    }
