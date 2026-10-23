"""Forgiving sample-name lookup.

Sample ids look like ``PHercParis4``, ``PHerc0332`` or ``PHerc1667Cr1Fr3``.
People write them as ``PHerc. Paris 4``, ``pherc paris 4``, ``paris4``,
``PHerc 332`` or ``Scroll 1``. All of those resolve to the same key here.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable

# Community names, as listed in metadata.schema.json (Sample.id).
COMMUNITY_ALIASES: dict[str, str] = {
    "Scroll 1": "PHercParis4",
    "Scroll 2": "PHercParis3",
    "Scroll 3": "PHerc0332",
    "Scroll 4": "PHerc1667",
    "Scroll 5": "PHerc0172",
}

_PREFIX = "pherc"


def name_key(name: str) -> str:
    """Case-, punctuation- and zero-padding-insensitive key for a sample name."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", key)


class SampleNameIndex:
    """Maps name keys to sample ids."""

    def __init__(self) -> None:
        self._keys: dict[str, set[str]] = {}

    def add(self, sample_id: str, names: Iterable[str | None]) -> None:
        for name in (sample_id, *names):
            if name:
                self._keys.setdefault(name_key(name), set()).add(sample_id)

    def resolve(self, query: str) -> set[str]:
        """Sample ids whose names match ``query`` exactly (after normalisation)."""
        key = name_key(query)
        hits = self._keys.get(key)
        if not hits and not key.startswith(_PREFIX):
            hits = self._keys.get(_PREFIX + key)
        return set(hits or ())

    def search(self, query: str) -> set[str]:
        """Sample ids with a name containing ``query`` (after normalisation)."""
        key = name_key(query)
        return {sid for k, ids in self._keys.items() if key in k for sid in ids}

    def suggest(self, query: str, n: int = 5) -> list[str]:
        close = difflib.get_close_matches(name_key(query), list(self._keys), n=n, cutoff=0.6)
        seen: list[str] = []
        for k in close:
            for sid in sorted(self._keys[k]):
                if sid not in seen:
                    seen.append(sid)
        return seen
