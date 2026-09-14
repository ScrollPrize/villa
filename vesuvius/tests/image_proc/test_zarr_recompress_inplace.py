"""In-place recompression: source chunks must actually be freed, and a failure
must never destroy the only remaining copy of the data.

Both behaviours below were broken in ways that are invisible at runtime:

* ``delete_chunk_file`` assumed zarr v2 *nested* storage (``0/0/0``). For zarr
  v3 (``c/0/0/0``) and for v2's own default separator (``0.0.0``) it matched
  nothing and returned silently, so in-place recompression never freed the
  source. The whole point of deleting as it goes is to avoid needing twice the
  disk for the array; without it a large in-place recompress can run out of
  space on exactly the volumes the tool exists for.

* Once deletion does work, the original is incomplete from the first finished
  work item onward, which makes the temp tree the only full copy. The failure
  handler used to delete it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from vesuvius.image_proc.run.zarr_tasks import utils as zarr_utils
from vesuvius.image_proc.run.zarr_tasks.tasks import recompress as recompress_mod
from vesuvius.image_proc.run.zarr_tasks.tasks.recompress import (
    RecompressConfig,
    RecompressTask,
)

SHAPE = (4, 4, 4)
CHUNKS = (2, 2, 2)
TOTAL_CHUNKS = 8  # (4/2)**3
FIRST_CHUNK_COORDS = ((0, 2), (0, 2), (0, 2))


# --------------------------------------------------------------------------
# chunk key layouts
# --------------------------------------------------------------------------

# The layouts this package can encounter, given zarr>=2.18.7,<4.
_LAYOUTS = {
    "zarr_v3_default": "c/0/0/0",
    "zarr_v2_nested_separator": "0/0/0",
    "zarr_v2_dot_separator": "0.0.0",
}


@pytest.mark.parametrize(
    "relative_key", _LAYOUTS.values(), ids=list(_LAYOUTS)
)
def test_delete_chunk_file_removes_the_chunk_in_every_layout(
    tmp_path: Path, relative_key: str
) -> None:
    # Written by hand rather than through zarr so the case is pinned to the key
    # layout itself and does not depend on which zarr major version is
    # installed in the environment running the tests.
    chunk = tmp_path / relative_key
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(b"chunk-payload")

    removed = zarr_utils.delete_chunk_file(str(tmp_path), FIRST_CHUNK_COORDS, CHUNKS)

    # The behaviour that regressed. Only the nested layout worked before; the
    # v3 and dot-separator cases matched nothing and returned silently.
    assert not chunk.exists()
    # The reported outcome, so a silent miss cannot masquerade as a deletion.
    assert removed is True


def test_delete_chunk_file_reports_when_nothing_matched(tmp_path: Path) -> None:
    assert (
        zarr_utils.delete_chunk_file(str(tmp_path), FIRST_CHUNK_COORDS, CHUNKS) is False
    )


# --------------------------------------------------------------------------
# failure handling
# --------------------------------------------------------------------------


class _PoolFailingMidway:
    """Runs work items in-process and then raises, as a dying worker would.

    The items that do run perform real copies and real source deletions, which
    is precisely the half-destroyed state the failure handler must cope with.
    """

    def __init__(self, processes: int | None = None) -> None:
        self.processes = processes

    def __enter__(self) -> "_PoolFailingMidway":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def imap_unordered(self, func, items):
        for index, item in enumerate(items):
            if index >= 1:
                raise RuntimeError("worker died")
            yield func(item)


_ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


def _chunk_files(root: Path) -> list[Path]:
    """Every file under `root` that is a chunk rather than metadata.

    Metadata is ``.zarray``/``.zattrs`` on v2 and ``zarr.json`` on v3; chunk
    names are pure index strings, so filtering on a leading dot plus the v3
    metadata filename is sufficient and does not depend on the separator.
    """
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.name != "zarr.json"
    ]


def _make_source(path: Path) -> None:
    kwargs: dict = {"shape": SHAPE, "chunks": CHUNKS, "dtype": "uint8"}
    if _ZARR_V3:
        kwargs["zarr_format"] = 2
    array = zarr.open(str(path), mode="w", **kwargs)
    # Non-zero: zarr elides chunks that equal the fill value, and writing zeros
    # would leave no chunk files on disk to delete.
    array[:] = np.arange(int(np.prod(SHAPE)), dtype="uint8").reshape(SHAPE) + 1


@pytest.mark.skipif(
    _ZARR_V3,
    reason=(
        "RecompressTask cannot run under zarr 3.x at all: it builds its temp "
        "array with zarr.open(..., compressor=<numcodecs Blosc>), and zarr 3 "
        "creates new arrays as v3 by default, which rejects `compressor` "
        "outright ('compressor cannot be used for arrays with zarr_format 3'). "
        "That happens for every input, v2 sources included, so the task is "
        "usable only on zarr 2.x today. Tracked separately; this test covers "
        "the failure-handling contract on the versions where the task runs."
    ),
)
def test_inplace_failure_keeps_the_partially_recompressed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "volume.zarr"
    temp_tree = tmp_path / "volume.zarr_tmp"
    _make_source(source)

    monkeypatch.setattr(recompress_mod, "Pool", _PoolFailingMidway)

    task = RecompressTask(
        RecompressConfig(
            input_zarr=str(source),
            output_zarr=None,
            num_workers=1,
            inplace=True,
        )
    )
    task.prepare()

    with pytest.raises(RuntimeError) as failure:
        task._run_inplace()

    # The run got far enough to start destroying the original ...
    surviving = _chunk_files(source)
    assert len(surviving) < TOTAL_CHUNKS, (
        "expected the worker to have deleted at least one source chunk before "
        "the failure; otherwise this test is not exercising the dangerous state"
    )

    # ... so the recompressed copy is the only complete one and must survive.
    assert temp_tree.exists(), (
        "the temp tree was deleted on failure, destroying the only copy of the "
        "chunks already removed from the original"
    )
    assert _chunk_files(temp_tree), "temp tree survived but holds no chunks"
    assert str(temp_tree) in str(failure.value)
