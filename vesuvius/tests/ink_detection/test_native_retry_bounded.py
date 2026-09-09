"""Native patch resampling must terminate (#1482).

The replacement for an inadmissible native patch is a deterministic function
of the current index. With two patches the only legal successor of 0 is 1 and
of 1 is 0, so if both are inadmissible the old loop ran forever inside a
DataLoader worker. The loop now tracks visited indices and raises after a
bounded number of distinct patches.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from vesuvius.ink_detection.data import dataset as dataset_module
from vesuvius.ink_detection.data.dataset import InkDataset


def _dataset(n_patches: int, admissible: set[int], seed: int = 17) -> InkDataset:
    ds = InkDataset.__new__(InkDataset)
    ds.mode = "full_3d"
    ds.config = SimpleNamespace(seed=seed)
    ds.patches = [SimpleNamespace(index=i) for i in range(n_patches)]
    ds.calls = []

    def _native_sample(patch):
        ds.calls.append(patch.index)
        return {"ok": patch.index} if patch.index in admissible else None

    ds._native_sample = _native_sample
    return ds


def test_two_inadmissible_patches_raise_instead_of_cycling():
    ds = _dataset(2, admissible=set())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(RuntimeError, match="No admissible native crop after trying 2 distinct"):
            ds[0]
    assert sorted(ds.calls) == [0, 1]  # each patch tried once, no repeats


def test_reachable_admissible_patch_is_found_without_revisits():
    ds = _dataset(10, admissible={7})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert ds[0] == {"ok": 7}
    assert len(ds.calls) == len(set(ds.calls))  # never revisited a patch


def test_attempt_bound_caps_large_datasets(monkeypatch):
    monkeypatch.setattr(dataset_module, "MAX_NATIVE_RESAMPLE_ATTEMPTS", 5)
    ds = _dataset(1000, admissible=set())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(RuntimeError, match="trying 5 distinct"):
            ds[3]
    assert len(ds.calls) == 5


def test_single_patch_keeps_its_own_error():
    ds = _dataset(1, admissible=set())
    with pytest.raises(RuntimeError, match="with one patch"):
        ds[0]
