"""Select an explicit build tree before importing any vc extension modules."""
import os
from pathlib import Path
import sys

if build_python := os.environ.get("VC_OBJECT_STORE_BUILD_PYTHON_PATH"):
    sys.path.insert(0, str(Path(build_python).resolve()))


import pytest


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "objects.vcobj")


@pytest.fixture()
def db(db_path):
    from vc_object_store import ObjectStore
    with ObjectStore.create(db_path, author="pytest", hostname="testhost") as store:
        yield store
