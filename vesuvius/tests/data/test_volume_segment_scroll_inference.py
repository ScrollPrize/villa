"""A segment given without a scroll should find its own scroll in the config.

Regression: ``Volume(type="segment", segment_id=...)`` is the form printed in
the class docstring and in the library's own error message, but energy and
resolution are keyed by scroll id. With no ``scroll_id`` the lookup ran against
None and init failed with "Could not determine energy/resolution for scroll
None", so the documented call could never succeed.
"""

import textwrap

import pytest

from vesuvius.data.volume import Volume


CONFIG = textwrap.dedent(
    """
    "1":
      "54":
        "7.91":
          volume: "https://example.invalid/scroll1.zarr/"
          segments:
            "20230827161847": "https://example.invalid/seg1.zarr/"
    "3":
      "53":
        "3.24":
          volume: "https://example.invalid/scroll3.zarr/"
          segments:
            "20231111135340": "https://example.invalid/seg3.zarr/"
    """
)


def _probe(segment_id, config_path):
    """A Volume shell carrying only what the helper reads."""
    vol = Volume.__new__(Volume)
    vol.segment_id = segment_id
    vol.configs = str(config_path)
    return vol


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "scrolls.yaml"
    path.write_text(CONFIG)
    return path


@pytest.mark.unit
@pytest.mark.parametrize(
    "segment_id,expected",
    [(20230827161847, ("1", 54, 7.91)), (20231111135340, ("3", 53, 3.24))],
)
def test_infers_scroll_energy_and_resolution(config_file, segment_id, expected):
    assert _probe(segment_id, config_file)._infer_scroll_from_segment() == expected


@pytest.mark.unit
def test_returns_none_for_unknown_segment(config_file):
    assert _probe(99999999999999, config_file)._infer_scroll_from_segment() is None


@pytest.mark.unit
def test_returns_none_when_config_missing(tmp_path):
    probe = _probe(20230827161847, tmp_path / "does_not_exist.yaml")
    assert probe._infer_scroll_from_segment() is None


@pytest.mark.unit
def test_shipped_config_covers_the_documented_example():
    """The segment named in the docstring must resolve from the shipped config."""
    import os
    import vesuvius

    base = os.path.dirname(os.path.abspath(vesuvius.data.volume.__file__))
    shipped = os.path.join(os.path.dirname(base), "install", "configs", "scrolls.yaml")
    if not os.path.exists(shipped):
        pytest.skip("shipped scrolls.yaml not present in this install")
    assert _probe(20230827161847, shipped)._infer_scroll_from_segment() is not None
