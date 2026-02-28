from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _normalize_sample_key(sample_key):
    if sample_key is None:
        return None
    patch_idx, wrap_idx = sample_key
    if wrap_idx is None:
        return (int(patch_idx), None)
    return (int(patch_idx), int(wrap_idx))


@dataclass
class TripletResampleTracker:
    """Track triplet target failures and choose non-repeating resamples."""

    failed_target_keys: set[tuple[int, int | None]] = field(default_factory=set)
    last_target_failure_reason: str | None = None

    def reset_last_target_failure_reason(self) -> None:
        self.last_target_failure_reason = None

    def set_last_target_failure_reason(self, reason: str) -> None:
        self.last_target_failure_reason = str(reason)

    def mark_failed_target_key(self, sample_key) -> None:
        key = _normalize_sample_key(sample_key)
        if key is None:
            return
        self.failed_target_keys.add(key)

    def is_failed_target_key(self, sample_key) -> bool:
        key = _normalize_sample_key(sample_key)
        if key is None:
            return False
        return key in self.failed_target_keys


def choose_replacement_index(
    sample_index,
    *,
    attempted_indices=None,
    failed_target_keys=None,
):
    """Pick an untried replacement index, avoiding known-failed triplet targets first."""
    total = len(sample_index)
    if total <= 0:
        return None

    attempted = {int(i) for i in (attempted_indices or ())}
    blocked = {_normalize_sample_key(k) for k in (failed_target_keys or ())}
    blocked.discard(None)

    preferred = [
        i for i, key in enumerate(sample_index)
        if i not in attempted and _normalize_sample_key(key) not in blocked
    ]
    if preferred:
        return int(preferred[int(np.random.randint(len(preferred)))])

    fallback = [i for i in range(total) if i not in attempted]
    if fallback:
        return int(fallback[int(np.random.randint(len(fallback)))])
    return None
