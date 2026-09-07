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


class _FakeArray:
    """Minimal stand-in for the zarr array Volume expects from load_data."""

    import numpy as _np

    shape = (65, 16, 16)
    ndim = 3
    dtype = _np.dtype("uint8")

    def __getitem__(self, idx):
        import numpy as np

        return np.zeros((1, 1, 1), dtype="uint8")


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
def test_malformed_config_raises_rather_than_reporting_a_missing_segment(tmp_path):
    """A config that exists but will not parse is a config problem, not a
    missing segment, and must not be flattened into the latter."""
    import yaml

    path = tmp_path / "scrolls.yaml"
    path.write_text('"1":\n  "54":\n   bad: [unclosed\n')
    with pytest.raises(yaml.YAMLError):
        _probe(20230827161847, path)._infer_scroll_from_segment()


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


@pytest.mark.unit
def test_documented_segment_call_initializes_without_scroll_id(monkeypatch):
    """The documented one-argument form must actually construct a Volume.

    The inference runs inside ``Volume.__init__``, so the helper tests above do
    not on their own show that ``Volume(type="segment", segment_id=...)`` works.
    Network-dependent steps are stubbed; everything before them is the real
    code path, including reading the shipped config.
    """
    import numpy as np

    monkeypatch.setattr(Volume, "load_ome_metadata", lambda self: {})
    monkeypatch.setattr(Volume, "load_data", lambda self: _FakeArray())
    monkeypatch.setattr(Volume, "download_inklabel", lambda self, save_path=None: None)

    vol = Volume(type="segment", segment_id=20230827161847, normalization_scheme="none")

    assert vol.scroll_id == "1"
    assert vol.energy == 54
    assert vol.resolution == 7.91
    assert vol.url.endswith("20230827161847.zarr/")


@pytest.mark.unit
def test_explicit_scroll_id_is_not_overridden_by_inference(monkeypatch):
    monkeypatch.setattr(Volume, "load_ome_metadata", lambda self: {})
    monkeypatch.setattr(Volume, "load_data", lambda self: _FakeArray())
    monkeypatch.setattr(Volume, "download_inklabel", lambda self, save_path=None: None)

    vol = Volume(type="segment", segment_id=20230827161847, scroll_id=1,
                 normalization_scheme="none")

    assert vol.scroll_id == 1
