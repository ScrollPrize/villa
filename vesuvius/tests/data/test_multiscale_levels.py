"""Resolution levels must be discoverable without listing the group.

Over HTTP a zarr store has no LIST operation, so ``keys()`` and ``len()`` come
back empty even though every array is readable by name. Levels therefore have
to come from the OME-Zarr ``multiscales`` metadata, with listing as a fallback.
"""

from __future__ import annotations

import pytest
import zarr

from vesuvius.data.volume import Volume, _multiscale_level_paths


# --------------------------------------------------------------------------
# Reading level paths out of the metadata
# --------------------------------------------------------------------------

class _FakeGroup:
    """Minimal stand-in: attrs plus optional member listing."""

    def __init__(self, attrs, members=None):
        self.attrs = attrs
        self._members = members or {}

    def keys(self):
        return self._members.keys()

    def __getitem__(self, key):
        return self._members[key]


def _multiscales(paths):
    return {"multiscales": [{"datasets": [{"path": p} for p in paths]}]}


def test_paths_come_from_multiscales() -> None:
    g = _FakeGroup(_multiscales(["0", "1", "2"]))
    assert _multiscale_level_paths(g) == ["0", "1", "2"]


def test_order_is_preserved_not_sorted() -> None:
    """Level order is declared by the metadata; it must not be re-sorted."""
    g = _FakeGroup(_multiscales(["0", "1", "10", "2"]))
    assert _multiscale_level_paths(g) == ["0", "1", "10", "2"]


@pytest.mark.parametrize("attrs", [
    {},                                             # no metadata at all
    {"multiscales": []},                            # present but empty
    {"multiscales": [{}]},                          # no datasets key
    {"multiscales": [{"datasets": []}]},            # empty datasets
    {"multiscales": [{"datasets": [{}]}]},          # dataset without a path
    {"multiscales": "not-a-list"},                  # malformed
])
def test_missing_or_malformed_metadata_returns_none(attrs) -> None:
    """Anything unusable must yield None so the caller can fall back."""
    assert _multiscale_level_paths(_FakeGroup(attrs)) is None


def test_attrs_that_raise_do_not_propagate() -> None:
    class _Hostile:
        @property
        def attrs(self):
            raise RuntimeError("store unavailable")

    assert _multiscale_level_paths(_Hostile()) is None


# --------------------------------------------------------------------------
# How Volume uses them
# --------------------------------------------------------------------------

def _volume_with(data):
    vol = Volume.__new__(Volume)
    vol.data = data
    return vol


def _real_group(level_names, with_metadata):
    """A genuine in-memory zarr Group, optionally carrying multiscales."""
    g = zarr.group()
    # zarr 3 renamed Group.create_dataset to create_array; the repo supports
    # both (pyproject allows zarr>=2.18.7,<4), so branch as the other tests do.
    create = getattr(g, "create_array", None) or g.create_dataset
    for name in level_names:
        create(name=name, shape=(2, 2, 2), dtype="u1")
    if with_metadata:
        g.attrs["multiscales"] = _multiscales(level_names)["multiscales"]
    return g


def test_levels_come_from_metadata_when_present() -> None:
    """Metadata wins, and its declared order is kept."""
    g = _real_group(["0", "1", "2", "3"], with_metadata=True)
    vol = _volume_with(g)

    assert vol._level_keys() == ["0", "1", "2", "3"]
    assert vol._num_levels() == 4


def test_listing_used_when_no_metadata() -> None:
    g = _real_group(["0", "1"], with_metadata=False)
    vol = _volume_with(g)

    assert sorted(vol._level_keys()) == ["0", "1"]
    assert vol._num_levels() == 2


def test_metadata_survives_an_unlistable_store() -> None:
    """The HTTP case: keys() is empty, metadata still names the levels."""
    g = _real_group(["0", "1", "2"], with_metadata=True)
    vol = _volume_with(g)
    object.__setattr__(vol, "_level_keys_cache", None)

    # emulate a store that cannot enumerate itself
    import unittest.mock as mock
    with mock.patch.object(type(g), "keys", return_value=iter(())):
        assert vol._level_keys() == ["0", "1", "2"]


def test_plain_array_reports_a_single_level() -> None:
    vol = _volume_with(zarr.zeros((4, 4, 4)))
    assert vol._num_levels() == 1


def test_legacy_list_of_levels_still_counted() -> None:
    """The older form, where data is a plain list, must keep working."""
    vol = _volume_with([zarr.zeros((4, 4, 4)), zarr.zeros((2, 2, 2))])
    assert vol._num_levels() == 2


def test_level_keys_are_computed_once() -> None:
    """Each lookup would otherwise be a round trip on a remote store."""
    g = _real_group(["0", "1"], with_metadata=True)
    vol = _volume_with(g)

    first = vol._level_keys()
    assert vol._level_keys() is first
    assert vol._level_keys() is first


# --------------------------------------------------------------------------
# Reads must use the declared keys, not the level's position
# --------------------------------------------------------------------------
# Discovering the level names was only half the fix: every read path still
# indexed ``self.data`` directly, which works for the legacy list of levels but
# not for a Group, where a level is addressed by key. These tests pin the read
# paths themselves, and deliberately use non-numeric, non-sorted paths so that
# indexing by position or by ``str(idx)`` cannot pass by accident.

_NAMED = ["full", "half", "quarter"]


def _named_group():
    g = zarr.group()
    create = getattr(g, "create_array", None) or g.create_dataset
    for n, name in enumerate(_NAMED):
        arr = create(name=name, shape=(4 - n, 6 - n, 8 - n), dtype="u1")
        arr[...] = n + 1
    g.attrs["multiscales"] = _multiscales(_NAMED)["multiscales"]
    return g


def test_shape_uses_declared_paths() -> None:
    vol = _volume_with(_named_group())
    assert vol.shape(0) == (4, 6, 8)
    assert vol.shape(1) == (3, 5, 7)
    assert vol.shape(2) == (2, 4, 6)


def test_ndim_uses_declared_paths() -> None:
    vol = _volume_with(_named_group())
    assert vol.ndim == 3


def test_level_rejects_out_of_range() -> None:
    vol = _volume_with(_named_group())
    with pytest.raises(IndexError):
        vol.shape(len(_NAMED))
    with pytest.raises(IndexError):
        vol.shape(-1)


def test_single_array_store_reports_one_level() -> None:
    arr = zarr.zeros((2, 3, 4), dtype="u1")
    vol = _volume_with(arr)
    assert vol.shape() == (2, 3, 4)
    assert vol.ndim == 3
    with pytest.raises(IndexError):
        vol.shape(1)


def test_legacy_list_of_levels_still_works() -> None:
    """The pre-existing form must keep working: levels addressed by position."""
    levels = [zarr.zeros((4, 4, 4), dtype="u1"), zarr.zeros((2, 2, 2), dtype="u1")]
    vol = _volume_with(levels)
    assert vol.shape(0) == (4, 4, 4)
    assert vol.shape(1) == (2, 2, 2)
    assert vol.ndim == 3


def test_mixed_numeric_and_named_paths_do_not_break_sorting(capsys) -> None:
    """``["0", "scale1"]`` must not raise when levels are listed.

    Sorting with a key returning sometimes int and sometimes str compares the
    two and raises TypeError on any store that mixes them.
    """
    g = zarr.group()
    create = getattr(g, "create_array", None) or g.create_dataset
    for name in ["0", "scale1"]:
        create(name=name, shape=(2, 2, 2), dtype="u1")
    g.attrs["multiscales"] = _multiscales(["0", "scale1"])["multiscales"]

    vol = _volume_with(g)
    vol.type = "zarr"
    vol.scroll_id = None
    vol.energy = None
    vol.segment_id = None
    vol.resolution = None
    vol.url = "memory://"
    vol.dtype = "u1"
    vol.normalization_scheme = "none"
    vol.return_as_type = "none"
    vol.return_as_tensor = False
    vol.inklabel = None
    vol.meta()
    out = capsys.readouterr().out
    assert "Number of Resolution Levels: 2" in out
