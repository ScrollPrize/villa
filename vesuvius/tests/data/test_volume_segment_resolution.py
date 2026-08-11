"""A segment id is enough to reach a segment: the catalog knows its scroll.

The usage block Volume prints on failure suggests
``Volume(type="segment", segment_id=20230827161847)``, so that call has to work
without a scroll_id. These drive the real constructor as far as the URL lookup,
which is all local: the catalog is packaged YAML, so none of this touches the
network.
"""

from __future__ import annotations

import pytest

from vesuvius.data import volume as volume_module
from vesuvius.data.volume import Volume, _find_segment_locations

# Listed in install/configs/scrolls.yaml under scroll 1, 54 keV, 7.91 um. This is
# the id the printed usage block names.
SEGMENT_IN_CATALOG = 20230827161847
SEGMENT_NOT_IN_CATALOG = 19990101000000

_SYNTHETIC_SEGMENT = "20240102030405"


@pytest.fixture(autouse=True)
def _no_inklabel_download(monkeypatch):
    """download_only still fetches the ink label PNG for segments."""
    monkeypatch.setattr(Volume, "download_inklabel",
                        lambda self, save_path=None: None)


def _resolve(segment_id, scroll_id=None):
    """Init far enough to see the resolved fields, without opening any store."""
    return Volume(type="segment", segment_id=segment_id, scroll_id=scroll_id,
                  domain="dl.ash2txt", download_only=True)


def _catalog(entries):
    """Build a scrolls.yaml-shaped catalog holding the given segment locations."""
    catalog: dict = {}
    for scroll, energy, resolution in entries:
        level = catalog.setdefault(scroll, {}).setdefault(energy, {}).setdefault(
            resolution, {})
        level["volume"] = f"https://example.invalid/{scroll}/volume.zarr/"
        level.setdefault("segments", {})[_SYNTHETIC_SEGMENT] = (
            f"https://example.invalid/{scroll}/{energy}/{resolution}/seg.zarr/")
    return catalog


def _use_catalog(monkeypatch, catalog):
    monkeypatch.setattr(volume_module, "list_files", lambda: catalog)


def test_segment_id_alone_resolves_its_scroll() -> None:
    vol = _resolve(SEGMENT_IN_CATALOG)
    assert vol.scroll_id == 1
    assert vol.energy == 54
    assert vol.resolution == 7.91
    assert vol.url.endswith(f"{SEGMENT_IN_CATALOG}.zarr/")


def test_segment_timestamp_as_type_resolves_the_same_way() -> None:
    """Volume(type="<timestamp>") and type="segment" are the same request."""
    vol = Volume(type=str(SEGMENT_IN_CATALOG), domain="dl.ash2txt",
                 download_only=True)
    by_segment_id = _resolve(SEGMENT_IN_CATALOG)
    assert (vol.scroll_id, vol.energy, vol.resolution, vol.url) == (
        by_segment_id.scroll_id, by_segment_id.energy,
        by_segment_id.resolution, by_segment_id.url)


def test_explicit_scroll_id_is_used_as_passed() -> None:
    """An explicit scroll_id wins: the string is kept, not the inferred int 1."""
    vol = _resolve(SEGMENT_IN_CATALOG, scroll_id="1")
    assert vol.scroll_id == "1"
    assert vol.url.endswith(f"{SEGMENT_IN_CATALOG}.zarr/")


def test_explicit_energy_and_resolution_win_over_the_catalog() -> None:
    # 88 keV at 3.24 um is not where this segment lives, so the URL lookup fails
    # and names every key it looked under. That is the check: the passed values
    # reached it rather than the catalog's 54 and 7.91 overwriting them, while the
    # scroll was still resolved from the catalog.
    with pytest.raises(ValueError) as excinfo:
        Volume(type="segment", segment_id=SEGMENT_IN_CATALOG, energy=88,
               resolution=3.24, domain="dl.ash2txt", download_only=True)
    message = str(excinfo.value)
    assert "scroll=1" in message
    assert "energy=88" in message
    assert "resolution=3.24" in message


def test_unknown_segment_says_it_is_not_in_the_catalog() -> None:
    with pytest.raises(ValueError, match="not listed in the scroll catalog"):
        _resolve(SEGMENT_NOT_IN_CATALOG)


def test_segment_under_two_scrolls_refuses_to_choose(monkeypatch) -> None:
    _use_catalog(monkeypatch, _catalog([("1", "54", "7.91"),
                                        ("2", "88", "3.24")]))
    with pytest.raises(ValueError) as excinfo:
        _resolve(int(_SYNTHETIC_SEGMENT))
    message = str(excinfo.value)
    assert "more than one scroll" in message
    assert "scroll 1 at energy 54, resolution 7.91" in message
    assert "scroll 2 at energy 88, resolution 3.24" in message
    assert "Pass scroll_id explicitly" in message


def test_scroll_id_settles_a_segment_listed_under_two_scrolls(monkeypatch) -> None:
    _use_catalog(monkeypatch, _catalog([("1", "54", "7.91"),
                                        ("2", "88", "3.24")]))
    scroll, energy, resolution = volume_module._resolve_segment_location(
        _SYNTHETIC_SEGMENT, scroll_id="2")
    assert (scroll, energy, resolution) == ("2", 88, 3.24)


def test_one_scroll_at_two_energies_keeps_the_scroll_and_defers_the_rest(
        monkeypatch) -> None:
    # The scroll is settled, so it is returned. Energy and resolution are not, so
    # they are left None for the scroll's canonical defaults to fill in.
    _use_catalog(monkeypatch, _catalog([("1", "54", "7.91"),
                                        ("1", "88", "3.24")]))
    assert volume_module._resolve_segment_location(_SYNTHETIC_SEGMENT) == (
        1, None, None)


def test_all_catalog_locations_are_collected(monkeypatch) -> None:
    catalog = _catalog([("2", "88", "3.24"), ("1", "54", "7.91")])
    assert [match[:3] for match in
            _find_segment_locations(catalog, _SYNTHETIC_SEGMENT)] == [
        ("1", "54", "7.91"), ("2", "88", "3.24")]


def test_find_segment_details_still_reports_the_catalog_entry() -> None:
    scroll, energy, resolution, entry = Volume.__new__(
        Volume).find_segment_details(str(SEGMENT_IN_CATALOG))
    assert (scroll, energy, resolution) == ("1", "54", "7.91")
    assert entry.endswith(f"{SEGMENT_IN_CATALOG}.zarr/")


def test_find_segment_details_returns_nones_for_an_unknown_segment() -> None:
    assert Volume.__new__(Volume).find_segment_details(
        str(SEGMENT_NOT_IN_CATALOG)) == (None, None, None, None)
