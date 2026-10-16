from __future__ import annotations

from pathlib import Path

import pytest

from vc_reader import Catalog

FIXTURE = Path(__file__).parent / "data" / "metadata_subset.json"


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return Catalog.from_file(FIXTURE)


@pytest.fixture(scope="session")
def paris4(catalog):
    return catalog.sample("PHercParis4")


@pytest.fixture(scope="session")
def pherc139(catalog):
    return catalog.sample("PHerc0139")
