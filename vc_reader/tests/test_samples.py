from __future__ import annotations

import pytest

from vc_reader import NotFoundError
from vc_reader.names import name_key


@pytest.mark.parametrize(
    "query",
    ["PHercParis4", "phercparis4", "PHerc. Paris 4", "pherc paris 4", "paris4", "Paris 4", "Scroll 1", "scroll1"],
)
def test_paris4_spellings(catalog, query):
    assert catalog.sample(query).id == "PHercParis4"


@pytest.mark.parametrize("query", ["PHerc0332", "PHerc 332", "pherc332", "332", "Scroll 3"])
def test_zero_padding_and_bare_number(catalog, query):
    assert catalog.sample(query).id == "PHerc0332"


def test_display_name(catalog):
    s = catalog.sample("PHerc. Paris 1 Fr 39")
    assert s.id == "PHercParis1Fr39"
    assert s.type == "fragment"
    assert s.display_name == "PHerc. Paris 1 Fr 39"


def test_unknown_sample_suggests(catalog):
    with pytest.raises(NotFoundError, match="did you mean PHercParis4"):
        catalog.sample("pherc paris 9")


def test_find_samples(catalog):
    assert [s.id for s in catalog.find_samples("paris")] == ["PHercParis1Fr39", "PHercParis4"]
    assert [s.id for s in catalog.find_samples(type="fragment")] == ["PHercParis1Fr39"]
    assert len(catalog.find_samples()) == len(catalog.samples)


def test_name_key_keeps_internal_zeros():
    assert name_key("PHerc0500P2") == "pherc500p2"
    assert name_key("PHerc1667Cr1Fr3") == "pherc1667cr1fr3"
    assert name_key("PHerc 0009 B") == "pherc9b"


def test_sample_contents(paris4):
    assert paris4.type == "scroll"
    assert paris4.aliases == ("Scroll 1",)
    assert len(paris4.volumes) == 7 and len(paris4.scans) == 7
    assert paris4.photo.url().endswith("/PHercParis4/photos/PHercParis4_photo.jpg")
    assert paris4.photo.url().startswith("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/")


def test_models(catalog):
    m = catalog.model("20260428123845")
    assert m.task == "ink-detection-3d"
    assert m.name == "v3-78k-fullsup"
    with pytest.raises(NotFoundError):
        catalog.model("19700101000000")
