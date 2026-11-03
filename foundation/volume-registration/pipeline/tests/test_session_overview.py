"""The overview has to put a patch where it is.

A marker drawn one pixel off on an overview a few hundred pixels long is a few millimetres
off in the scroll, and nothing about the picture says so. So a bright cube is planted at a
known place in a store with a real pyramid, and must come back at that place through the
projections, the stored level and the cross-sections.
"""

import numpy as np
import pytest
import zarr

from volreg.cores import session_overview
from volreg.cores.fit_session import Session

SHAPE = (256, 128, 192)
CUBE = (slice(160, 176), slice(32, 48), slice(96, 112))
CENTRE = np.array([167.5, 39.5, 103.5])


def write_pyramid(path, data, levels=4):
    group = zarr.open_group(str(path), mode="w")
    datasets = []
    for level in range(levels):
        group.create_dataset(str(level), data=data, chunks=(64, 64, 64), compressor=None)
        datasets.append({"path": str(level), "coordinateTransformations": [
            {"type": "scale", "scale": [2.0 ** level] * 3}]})
        shape = tuple(size // 2 * 2 for size in data.shape)
        data = data[:shape[0], :shape[1], :shape[2]].reshape(
            shape[0] // 2, 2, shape[1] // 2, 2, shape[2] // 2, 2).mean((1, 3, 5)).astype(np.uint8)
    group.attrs["multiscales"] = [{"datasets": datasets}]
    return str(path)


@pytest.fixture()
def session(tmp_path):
    data = np.full(SHAPE, 40, np.uint8)
    data[CUBE] = 250
    store = write_pyramid(tmp_path / "coarse.zarr", data)
    matrix = np.hstack([np.eye(3), np.zeros((3, 1))])
    return Session.create(tmp_path / "s", store, store, 1.0, 1.0, matrix, "identity")


def test_the_projections_and_the_stored_level_place_the_cube(session):
    geometry = session_overview.build(session, pixels=64)
    assert geometry["scale"] == 4.0
    _geometry, volume = session_overview.load(session)
    assert volume.shape == tuple(size // 4 for size in SHAPE)

    centre = np.argwhere(np.asarray(volume) == volume.max()).mean(0)
    np.testing.assert_allclose(session_overview.to_voxel(geometry, centre), CENTRE, atol=0.01)
    np.testing.assert_allclose(session_overview.to_pixel(geometry, CENTRE), centre, atol=0.01)

    from PIL import Image
    xz = np.asarray(Image.open(session_overview.overview_dir(session) / "xz-max.png"))
    assert xz.shape == (SHAPE[0] // 4, SHAPE[2] // 4)
    row, column = np.argwhere(xz == xz.max()).mean(0)
    np.testing.assert_allclose([row, column], centre[[0, 2]], atol=0.01)


def test_both_cross_sections_cut_through_the_cube(session):
    session_overview.build(session, pixels=64)
    coarse, info = session_overview.overview_section(session, CENTRE[0])
    assert abs(info["z"] - CENTRE[0]) <= 2.0
    row, column = np.argwhere(coarse == coarse.max()).mean(0)
    np.testing.assert_allclose((np.array([row, column]) + 0.5) * info["scale"] - 0.5,
                               CENTRE[1:], atol=0.01)

    sharp, info = session_overview.detailed_section(session, CENTRE[0], pixels=96)
    assert info["scale"] == 2.0
    row, column = np.argwhere(sharp == sharp.max()).mean(0)
    np.testing.assert_allclose((np.array([row, column]) + 0.5) * info["scale"] - 0.5,
                               CENTRE[1:], atol=0.01)
