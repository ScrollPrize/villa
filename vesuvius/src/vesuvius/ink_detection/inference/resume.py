"""Durable, bit-identical resume support for flat inference.

Flat inference accumulates weighted probabilities chunk by chunk, in a fixed
block order, and only writes the TIFF at the very end. On hosts whose
sessions die mid-run (e.g. Colab), that means a multi-hour job either
finishes in one go or loses everything. This module snapshots the
accumulation state every few thousand blocks into a durable directory and
restores it on the next run, so the job continues where it stopped.

Resuming is bit-identical to an uninterrupted run: completed chunks are
stored exactly as flushed, the still-open chunk buffers are restored exactly
(float32, not re-derived), and inference continues from the next scheduled
block, so every tile is added to every chunk in the same order as before.

Directory layout (all writes are write-to-temp + rename):

- ``fingerprint.json``: identity of the run; resuming with any different
  input, checkpoint or geometry is refused instead of mixing results.
- ``part_<k>.npz``: chunks flushed between save ``k-1`` and save ``k``.
  Parts are append-only.
- ``open_<k>.npz``: the open chunk buffers and seen-counts at save ``k``.
- ``state_<k>.json``: ``{"next_block": ..., "parts": k}``, written last.

The directory may live on a lazily-uploading network mount (rclone/Drive
VFS), where a newer state can reach durable storage before the files it
references. Loading therefore picks the newest state whose referenced files
all exist and load, rather than trusting the newest state blindly.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

LOGGER = logging.getLogger(__name__)

# Open buffers from older saves are only needed as fallbacks if the newest
# save is incomplete on durable storage; keep a couple, delete the rest.
_KEEP_OPEN_SAVES = 2


def _atomic_write_bytes(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        write(stream)
    os.replace(tmp, path)


def _chunk_name(prefix: str, row: int, col: int) -> str:
    return f"{prefix}_{row}_{col}"


def _parse_chunk_name(name: str) -> tuple[str, int, int]:
    prefix, row, col = name.split("_")
    return prefix, int(row), int(col)


class ResumeStore:
    """Save and restore ChunkAccumulator progress in a durable directory."""

    def __init__(self, directory: Path, fingerprint: Mapping[str, Any]) -> None:
        self.directory = Path(directory)
        self.fingerprint = json.loads(json.dumps(dict(fingerprint), sort_keys=True))
        self.saves = 0

    # -- loading -----------------------------------------------------------

    def _state_indices(self) -> list[int]:
        indices = []
        for path in self.directory.glob("state_*.json"):
            try:
                indices.append(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(indices)

    def _check_fingerprint(self) -> None:
        path = self.directory / "fingerprint.json"
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored != self.fingerprint:
                raise ValueError(
                    f"Resume directory {self.directory} belongs to a different "
                    f"inference run; refusing to mix results.\n"
                    f"  stored:  {stored}\n  current: {self.fingerprint}\n"
                    "Use a different --resume-dir or delete this one."
                )
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(
                path,
                lambda stream: stream.write(
                    json.dumps(self.fingerprint, indent=2, sort_keys=True).encode()
                ),
            )

    def _load_open(self, index: int) -> tuple[dict, dict] | None:
        path = self.directory / f"open_{index:05d}.npz"
        try:
            with np.load(path) as data:
                buffers: dict[tuple[int, int], list[np.ndarray]] = {}
                seen: dict[tuple[int, int], int] = {}
                for name in data.files:
                    prefix, row, col = _parse_chunk_name(name)
                    key = (row, col)
                    if prefix == "s":
                        seen[key] = int(data[name])
                    else:
                        slot = buffers.setdefault(key, [None, None])
                        slot[0 if prefix == "p" else 1] = np.array(
                            data[name], dtype=np.float32
                        )
            return {key: (p, w) for key, (p, w) in buffers.items()}, seen
        except (OSError, ValueError, KeyError):
            return None

    def load(self) -> tuple[int, dict, dict]:
        """Return (next_block, open_buffers, seen_counts) to continue from.

        A fresh directory returns (0, {}, {}).
        """

        self._check_fingerprint()
        for index in reversed(self._state_indices()):
            state_path = self.directory / f"state_{index:05d}.json"
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            parts_ok = all(
                (self.directory / f"part_{k:05d}.npz").exists()
                for k in range(1, int(state["parts"]) + 1)
            )
            opened = self._load_open(index) if parts_ok else None
            if opened is None:
                LOGGER.warning(
                    "Resume state %s is incomplete on disk; trying an older one",
                    state_path.name,
                )
                continue
            # Anything newer than the state we resume from is from an
            # abandoned timeline; it would be rewritten identically anyway.
            for newer in self._state_indices():
                if newer > index:
                    for name in (
                        f"state_{newer:05d}.json",
                        f"open_{newer:05d}.npz",
                        f"part_{newer:05d}.npz",
                    ):
                        (self.directory / name).unlink(missing_ok=True)
            self.saves = index
            buffers, seen = opened
            LOGGER.info(
                "Resuming from %s: next_block=%d open_chunks=%d",
                state_path.name,
                int(state["next_block"]),
                len(buffers),
            )
            return int(state["next_block"]), buffers, seen
        # No usable state: start over, clearing any leftovers so a stale
        # higher-numbered state can never be picked up later.
        for pattern in ("state_*.json", "open_*.npz", "part_*.npz"):
            for path in self.directory.glob(pattern):
                path.unlink(missing_ok=True)
        return 0, {}, {}

    # -- saving ------------------------------------------------------------

    def save(
        self,
        *,
        next_block: int,
        flushed: Mapping[tuple[int, int], tuple[np.ndarray, np.ndarray]],
        open_buffers: Mapping[tuple[int, int], tuple[np.ndarray, np.ndarray]],
        seen_counts: Mapping[tuple[int, int], int],
    ) -> None:
        """Persist newly flushed chunks plus the open state, then the state file."""

        index = self.saves + 1
        part_arrays: dict[str, np.ndarray] = {}
        for (row, col), (probability, weight) in flushed.items():
            part_arrays[_chunk_name("p", row, col)] = probability
            part_arrays[_chunk_name("w", row, col)] = weight
        _atomic_write_bytes(
            self.directory / f"part_{index:05d}.npz",
            lambda stream: np.savez_compressed(stream, **part_arrays),
        )
        open_arrays: dict[str, np.ndarray] = {}
        for (row, col), (probability, weight) in open_buffers.items():
            open_arrays[_chunk_name("p", row, col)] = probability
            open_arrays[_chunk_name("w", row, col)] = weight
            open_arrays[_chunk_name("s", row, col)] = np.asarray(
                seen_counts.get((row, col), 0), dtype=np.int64
            )
        _atomic_write_bytes(
            self.directory / f"open_{index:05d}.npz",
            lambda stream: np.savez_compressed(stream, **open_arrays),
        )
        _atomic_write_bytes(
            self.directory / f"state_{index:05d}.json",
            lambda stream: stream.write(
                json.dumps({"next_block": int(next_block), "parts": index}).encode()
            ),
        )
        self.saves = index
        stale = index - _KEEP_OPEN_SAVES
        if stale >= 1:
            (self.directory / f"open_{stale:05d}.npz").unlink(missing_ok=True)
            (self.directory / f"state_{stale:05d}.json").unlink(missing_ok=True)
        LOGGER.info(
            "Saved resume state %d: next_block=%d flushed_chunks=%d open_chunks=%d",
            index,
            int(next_block),
            len(flushed),
            len(open_buffers),
        )

    def iter_parts(self):
        """Yield ((row, col), probability, weight) for every saved chunk."""

        for k in range(1, self.saves + 1):
            with np.load(self.directory / f"part_{k:05d}.npz") as data:
                names = sorted(
                    {_parse_chunk_name(name)[1:] for name in data.files}
                )
                for row, col in names:
                    yield (
                        (row, col),
                        np.asarray(data[_chunk_name("p", row, col)], dtype=np.float32),
                        np.asarray(data[_chunk_name("w", row, col)], dtype=np.float32),
                    )
