"""Volume resolves a segment's scroll from the catalog and says what the catalog holds.

Volume(type="segment", segment_id=...) without scroll_id failed with "Could not
determine energy/resolution for scroll None", although it is the example Volume
prints on every error and the timestamp form Volume(type="<id>") already worked.
"""

import pytest

from vesuvius.data.volume import Volume

SEGMENT = 20230827161847  # listed in the packaged scrolls.yaml under scroll 1


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(Volume, "download_inklabel", lambda self, save_path=None: None)
    monkeypatch.setattr("vesuvius.data.volume.is_aws_ec2_instance", lambda: False)


def test_segment_id_alone_resolves_scroll_energy_and_resolution():
    vol = Volume(type="segment", segment_id=SEGMENT, download_only=True)
    assert (str(vol.scroll_id), int(vol.energy), float(vol.resolution)) == ("1", 54, 7.91)
    assert vol.url.endswith(f"{SEGMENT}.zarr/")


def test_unknown_segment_lists_catalog_segments():
    with pytest.raises(ValueError, match=f"Segments it lists: .*{SEGMENT}"):
        Volume(type="segment", segment_id=123, download_only=True)


def test_unknown_scroll_lists_catalog_scrolls():
    with pytest.raises(ValueError, match="not in the packaged catalog; it lists scrolls '1'"):
        Volume(type="scroll", scroll_id=6, download_only=True)


def test_missing_scan_lists_the_scroll_scans():
    with pytest.raises(ValueError, match="Scroll 1 is available at: energy=54, resolution=7.91"):
        Volume(type="scroll", scroll_id=1, energy=88, download_only=True)
