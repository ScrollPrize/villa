"""The shipped scroll catalogue and the loader's canonical scans agree.

Offline: these read the bundled YAML, never the server.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vesuvius.data import volume as volume_module
from vesuvius.data.volume import Volume

CONFIG_PATH = (Path(volume_module.__file__).resolve().parents[1]
               / "install" / "configs" / "scrolls.yaml")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _canonical_scan(scroll_id: str) -> tuple:
    vol = Volume.__new__(Volume)
    vol.scroll_id = scroll_id
    return vol.grab_canonical_energy(), vol.grab_canonical_resolution()


def test_every_listed_scroll_is_reachable_by_its_canonical_scan(config) -> None:
    unreachable = []
    for scroll_id, scans in config.items():
        energy, resolution = _canonical_scan(scroll_id)
        entry = scans.get(str(energy), {}).get(str(resolution), {})
        if not entry.get("volume"):
            unreachable.append((scroll_id, energy, resolution))
    assert unreachable == []


@pytest.mark.parametrize("scroll_id, energy, resolution, filename", [
    ("1b", "54", "7.91", "54keV_7.91um_Scroll1B.zarr/"),
    ("4", "88", "3.24", "20231107190228.zarr/"),
])
def test_published_volume_is_listed(config, scroll_id, energy, resolution,
                                    filename) -> None:
    url = config[scroll_id][energy][resolution]["volume"]
    assert url.startswith("https://dl.ash2txt.org/")
    assert url.endswith(filename)


def test_scroll_1a_and_1b_are_different_volumes(config) -> None:
    assert (config["1"]["54"]["7.91"]["volume"]
            != config["1b"]["54"]["7.91"]["volume"])


def test_scroll_4_keeps_its_53_kev_entry(config) -> None:
    assert config["4"]["53"]["7.91"]["volume"].endswith("20231117161658.zarr/")


def test_segment_lookup_survives_the_new_scroll_key(config, monkeypatch) -> None:
    monkeypatch.setattr(volume_module, "list_files", lambda: config)
    vol = Volume.__new__(Volume)
    scroll_id, energy, resolution, url = vol.find_segment_details("20230827161847")
    assert (scroll_id, energy, resolution) == ("1", "54", "7.91")
    assert url.endswith("20230827161847.zarr/")
