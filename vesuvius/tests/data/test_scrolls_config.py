"""scrolls.yaml has to list every scroll volume the package claims to know (#1652).

Scroll 1B and the Scroll 4 88 keV / 3.24 um scan are published next to the listed ones but had no
entry, so opening them by scroll identity raised "URL not found in config". No network is used here.
"""
from pathlib import Path

import pytest
import yaml

import vesuvius
from vesuvius.data.volume import Volume

CONFIG = Path(vesuvius.__file__).parent / "install" / "configs" / "scrolls.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _lookup(scroll_id, energy, resolution):
    """Resolve the URL exactly the way Volume.get_url_from_yaml walks the config."""
    vol = Volume.__new__(Volume)
    vol.type = "scroll"
    vol.scroll_id = scroll_id
    vol.energy = energy
    vol.resolution = resolution
    vol.configs = str(CONFIG)
    return vol.get_url_from_yaml()


@pytest.mark.parametrize(
    "scroll_id, energy, resolution, expected",
    [
        ("1b", 54, 7.91, "54keV_7.91um_Scroll1B.zarr/"),
        (4, 88, 3.24, "Scroll4/PHerc1667.volpkg/volumes_zarr/20231107190228.zarr/"),
        # already listed: must keep resolving to the same volumes
        (1, 54, 7.91, "54keV_7.91um_Scroll1A.zarr/"),
        (4, 53, 7.91, "Scroll4/PHerc1667.volpkg/volumes_zarr/20231117161658.zarr/"),
    ],
)
def test_volume_resolves_by_scroll_identity(scroll_id, energy, resolution, expected):
    assert _lookup(scroll_id, energy, resolution).endswith(expected)


def test_canonical_defaults_resolve_to_a_listed_volume():
    """Volume(type="scroll1b") and Volume(type="scroll", scroll_id=4) use the canonical maps."""
    for scroll_id in ("1b", 4):
        vol = Volume.__new__(Volume)
        vol.scroll_id = scroll_id
        assert _lookup(scroll_id, vol.grab_canonical_energy(), vol.grab_canonical_resolution())


def test_every_entry_is_a_well_formed_https_volume_url():
    for scroll_id, energies in _config().items():
        for energy, resolutions in energies.items():
            for resolution, entry in resolutions.items():
                url = entry["volume"]
                assert url.startswith("https://") and url.endswith(".zarr/"), (scroll_id, energy, resolution, url)
