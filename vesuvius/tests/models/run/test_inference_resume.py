"""An interrupted inference run should be worth something afterwards.

A whole-scroll pass streams for hours, so interruption is ordinary rather than exceptional.
Before --resume the only outcome was to start over: _create_output_stores opened the logits
store with mode='w', which recreates the array and discards every patch already written.

Two things carry the behaviour and both are tested here:

  * completion is recorded in a manifest, NOT by listing the chunks present in the store.
    write_empty_chunks=False means a patch that produced all zeros writes no chunk at all,
    and on a masked scroll volume those are the majority - so a listing-based resume would
    rerun every empty patch on every restart, forever.

  * a resume whose settings differ from the previous run is refused rather than merged. One
    store holding patches computed two different ways, with nothing recording the split, is
    the failure that made a published prediction reproducible only under a TTA setting no
    artifact mentioned.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from vesuvius.models.run.inference import Inferer


def _inferer(tmp_path, **overrides) -> Inferer:
    """An Inferer with just enough state for the resume helpers and store creation.

    __init__ builds a model and a dataset, neither of which this behaviour depends on, so
    the instance is assembled directly (same approach as test_inference_run_config_attrs).
    """
    inf = Inferer.__new__(Inferer)
    inf.num_classes = 2
    inf.patch_size = (4, 4, 4)
    inf.num_total_patches = 8
    inf.output_dir = str(tmp_path)
    inf.part_id = 0
    inf.num_parts = 1
    inf.overlap = 0.5
    inf.verbose = False
    # 'none' short-circuits _get_zarr_compressor, which reaches for zarr.Blosc - absent in
    # zarr 3. Compression is irrelevant to everything under test.
    inf.compressor_name = "none"
    inf.compression_level = 1
    inf.bbox = None
    inf.input_dir = "https://example.invalid/scroll.zarr/0"
    inf.model_path = "/models/m7"
    inf.disable_tta = True
    inf.tta_type = "mirroring"
    inf.do_tta = False
    inf.normalization_scheme = "instance_zscore"
    inf.is_multi_task = False
    inf.target_info = None
    inf.resume = False
    inf.resume_completed = None
    inf.dataset = SimpleNamespace(input_shape=(16, 16, 16))
    inf.patch_start_coords_list = [(i, 0, 0) for i in range(inf.num_total_patches)]
    for k, v in overrides.items():
        setattr(inf, k, v)
    return inf


def test_manifest_round_trips(tmp_path):
    inf = _inferer(tmp_path)
    inf._write_manifest({3, 1, 2})
    assert inf._load_completed() == {1, 2, 3}


def test_no_manifest_reads_as_nothing_to_resume(tmp_path):
    # None rather than set(): the caller must be able to tell "no previous run" apart from
    # "a previous run that completed zero patches".
    assert _inferer(tmp_path)._load_completed() is None


def test_unreadable_manifest_does_not_raise(tmp_path):
    inf = _inferer(tmp_path)
    with open(inf._manifest_path(), "w") as fh:
        fh.write("{not json")
    assert inf._load_completed() is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("bbox", (0, 4, 0, 4, 0, 4)),
        ("patch_size", (8, 8, 8)),
        ("overlap", 0.25),
        ("model_path", "/models/other"),
        ("disable_tta", False),
        ("num_total_patches", 9),
        ("num_parts", 2),
        ("input_dir", "https://example.invalid/other.zarr/0"),
    ],
)
def test_resume_refuses_a_different_run(tmp_path, field, value):
    _inferer(tmp_path)._write_manifest({0, 1})
    changed = _inferer(tmp_path, **{field: value})
    with pytest.raises(RuntimeError, match="resume refused"):
        changed._load_completed()


def test_each_part_resumes_from_its_own_manifest(tmp_path):
    """A different --part_id is not a mismatch, it is a different job.

    The manifest is per part, so part 1 finds nothing to resume rather than being refused.
    Re-sharding is the dangerous case, and that shows up as a changed num_parts, which is
    covered above.
    """
    _inferer(tmp_path, part_id=0)._write_manifest({0, 1})
    assert _inferer(tmp_path, part_id=1)._load_completed() is None
    assert _inferer(tmp_path, part_id=0)._load_completed() == {0, 1}


def test_refusal_names_the_field_that_differs(tmp_path):
    _inferer(tmp_path)._write_manifest({0})
    changed = _inferer(tmp_path, overlap=0.25)
    with pytest.raises(RuntimeError, match="overlap"):
        changed._load_completed()


def test_resume_reopens_the_store_instead_of_recreating_it(tmp_path):
    """The point of the whole feature: previously written patches must survive."""
    first = _inferer(tmp_path)
    first._create_output_stores()
    first.output_store[2] = np.full((2, 4, 4, 4), 7, dtype=np.float16)
    assert first.output_store[2].max() == 7

    again = _inferer(tmp_path, resume=True, resume_completed={2})
    again._create_output_stores()
    assert again.output_store[2].max() == 7, "resume recreated the store and lost the patch"


def test_without_resume_the_store_is_recreated(tmp_path):
    """The behaviour --resume exists to avoid, pinned so it cannot regress silently."""
    first = _inferer(tmp_path)
    first._create_output_stores()
    first.output_store[2] = np.full((2, 4, 4, 4), 7, dtype=np.float16)

    fresh = _inferer(tmp_path)
    fresh._create_output_stores()
    assert fresh.output_store[2].max() == 0


def test_resume_refuses_a_store_of_the_wrong_shape(tmp_path):
    first = _inferer(tmp_path)
    first._create_output_stores()

    grown = _inferer(tmp_path, num_total_patches=16, resume=True, resume_completed={0})
    with pytest.raises(RuntimeError, match="resume refused"):
        grown._create_output_stores()


def test_manifest_records_patches_that_wrote_no_chunk(tmp_path):
    """An all-zero patch writes no chunk (write_empty_chunks=False) but IS complete.

    This is why completion is tracked explicitly rather than by listing the store: the
    store cannot distinguish "done and empty" from "not done".
    """
    inf = _inferer(tmp_path)
    inf._create_output_stores()
    inf.output_store[5] = np.zeros((2, 4, 4, 4), dtype=np.float16)

    store_path = tmp_path / "logits_part_0.zarr"
    chunk_files = [p for p in store_path.rglob("*") if p.is_file() and not p.name.startswith(".")]
    assert chunk_files == [], "an all-zero patch should not have written a chunk"

    inf._write_manifest({5})
    assert inf._load_completed() == {5}
