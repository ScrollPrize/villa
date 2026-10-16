"""Parse the live metadata.json end to end. Run with ``pytest -m network``."""

from __future__ import annotations

import pytest

from vc_reader import Catalog

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    return Catalog.load(cache_dir=tmp_path_factory.mktemp("cache"))


def test_every_entity_parses_and_resolves(live):
    assert live.sample("Scroll 1").id == "PHercParis4"
    for sample in live.samples.values():
        assert live.sample(sample.id) is sample
        for vol in sample.volume_list():
            assert vol.zarr is not None and vol.url() is not None
            for lvl in vol.levels:
                lvl.predictions()
            for m in vol.ink_labels():
                assert m.labels.url() is not None
        for seg in sample.segment_list():
            for a in seg.artifacts:
                assert a.url() is not None, a


def test_headline_query(live):
    vol = live.sample("pherc paris 4").volume(resolution=2.399, energy_kev=78)
    lvl = vol.level(um=9.6)
    assert lvl.level == 2
    assert vol.ink_labels()
