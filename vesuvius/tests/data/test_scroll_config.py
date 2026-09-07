"""The canonical energy/resolution defaults must key into the shipped scrolls.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vesuvius.data import volume as volume_module
from vesuvius.data.volume import Volume

CONFIG = Path(volume_module.__file__).resolve().parents[1] / "install" / "configs" / "scrolls.yaml"
SCROLLS = yaml.safe_load(CONFIG.read_text())


@pytest.mark.parametrize("scroll_id", sorted(SCROLLS))
def test_canonical_defaults_resolve_to_a_volume(scroll_id: str) -> None:
    """Volume(type="scroll", scroll_id=X) leaves energy/resolution to the canonical
    maps, so every scroll the config lists must be reachable through them."""
    vol = Volume.__new__(Volume)
    vol.scroll_id = scroll_id
    energy = vol.grab_canonical_energy()
    resolution = vol.grab_canonical_resolution()
    assert energy is not None and resolution is not None, f"no canonical scan for scroll {scroll_id}"

    url = SCROLLS[scroll_id].get(str(energy), {}).get(str(resolution), {}).get("volume")
    assert url, f"scroll {scroll_id}: canonical {energy} keV / {resolution} um is not in scrolls.yaml"
