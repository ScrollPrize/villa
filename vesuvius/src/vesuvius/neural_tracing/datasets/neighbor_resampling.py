from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _normalize_sample_key(sample_key):
    if sample_key is None:
        return None

    if isinstance(sample_key, Mapping):
        patch_idx = sample_key.get("patch_idx")
        wrap_idx = sample_key.get("target_wrap_idx")
        if wrap_idx is None:
            wrap_idx = sample_key.get("wrap_idx")
        if wrap_idx is None:
            wrap_idx = sample_key.get("source_wrap_idx")
        if patch_idx is None or wrap_idx is None:
            return None
    else:
        patch_idx, wrap_idx = sample_key

    if wrap_idx is None:
        return (int(patch_idx), None)
    return (int(patch_idx), int(wrap_idx))


def choose_replacement_index(
    sample_index,
    *,
    attempted_indices=None,
    failed_target_keys=None,
):
    """Pick an untried replacement index, avoiding known failed targets first."""
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
