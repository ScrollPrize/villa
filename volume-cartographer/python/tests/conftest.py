"""Select an explicit build tree before importing any vc extension modules."""
import os
from pathlib import Path
import sys

if build_python := os.environ.get("VC_BUILD_PYTHON"):
    # An older editable wheel can redirect vc ahead of sys.path and eagerly
    # load an incompatible libvc_core. Use only the requested build's package.
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if type(finder).__module__ != "_editable_skbc_volume_cartographer"
    ]
    sys.path.insert(0, str(Path(build_python).resolve()))


import pytest


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "objects.vcobj")


@pytest.fixture()
def db(db_path):
    from vc.objects import ObjectStore
    with ObjectStore.create(db_path, author="pytest", hostname="testhost") as store:
        yield store
