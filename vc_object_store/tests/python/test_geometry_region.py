"""geometry_region on a store opened in this process.

geometry_region reads through withRead<true>, which gives the GIL up once. A second release by withRead (the GIL not held any more)
crashed the interpreter (vesuvius/tests/fiber_spline_generator/test_store.py:64)."""
import pytest
from vc_object_store import ObjectStore, RegionOptions, RegionSlice

from object_helpers import VOLUME, framed_patch


def slab():
    region = RegionSlice()
    region.local_bounds = [0, 0, 0, 40, 40, 40]
    return region


def options(**changes):
    value = RegionOptions()
    value.volume = VOLUME
    value.target_scale = 1.0
    for name, setting in changes.items():
        setattr(value, name, setting)
    return value


def test_geometry_region_on_a_local_store(db):
    framed_patch(db)
    region = db.geometry_region(slab(), options())
    assert isinstance(region, dict) and "complete" in region and "runs" in region


def test_geometry_region_on_a_store_with_one_reader(db_path):
    with ObjectStore.create(db_path, author="pytest", hostname="testhost", reader_pool_size=1) as store:
        framed_patch(store)
        for _ in range(3):  # the reader goes back to the pool each time; a second release would crash or hang
            assert "complete" in store.geometry_region(slab(), options())
            assert store.objects_by_kind("surface")


def test_geometry_region_after_a_subinterpreter_exists(db):
    """Once a sub-interpreter exists CPython makes PyGILState_Check() answer 1 always, so a release guarded by it is not safe.
    Keep this the last test of the file: the process stays in that state."""
    interpreters = pytest.importorskip("concurrent.interpreters")
    framed_patch(db)
    sub = interpreters.create()
    try:
        assert "complete" in db.geometry_region(slab(), options())
    finally:
        sub.close()
