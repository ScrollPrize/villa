import numpy as np
import pytest
import zarr
from scipy import ndimage

from vesuvius.neural_tracing.winding_models.volume_slab_extractor import VolumeSlabExtractor


@pytest.fixture
def volume(tmp_path):
    data = np.random.default_rng(0).integers(0, 255, size=(20, 24, 28), dtype=np.uint8)
    path = tmp_path / "v.zarr" / "0"
    zarr.open_array(str(path), mode="w", shape=data.shape, chunks=(8, 8, 8), dtype="u1",
                    fill_value=0, zarr_format=2)[:] = data
    return path, data


def _transform():
    transform = np.eye(4)
    transform[:3, :3] *= 0.5
    transform[:3, 3] = (1.25, -2.0, 0.25)
    return transform


def _reference_points_zyx(extractor, frame):
    size, length = extractor.transverse_size, extractor.ray_length
    ijk = np.stack(np.meshgrid(np.arange(size), np.arange(size), np.arange(length), indexing="ij"), -1)
    transform = extractor.segment_to_volume_xyz[0]
    xyz = frame.to_world(ijk.reshape(-1, 3)) @ transform[:3, :3].T + transform[:3, 3]
    return xyz[:, ::-1].reshape(size, size, length, 3)


@pytest.mark.parametrize("sampling", ["trilinear", "nearest"])
def test_extract_samples_the_slab_frame(volume, sampling):
    path, data = volume
    extractor = VolumeSlabExtractor([VolumeSlabExtractor.scaled_volume_path(path.parent, 0)],
                                    transverse_size=6, ray_length=40, spacing=1.5, sampling=sampling,
                                    cache_bytes=1 << 20, io_threads=2, segment_to_volume_xyz=[_transform()])
    image, valid, frame = extractor.extract(0, np.array([0.3, 1.0, 0.2]), np.array([10.0, 12.0, 4.0]))

    points = _reference_points_zyx(extractor, frame)
    assert image.shape == valid.shape == points.shape[:3]
    np.testing.assert_array_equal(valid, np.all((points >= 0) & (points < data.shape), axis=-1))
    assert valid.any() and not valid.all()
    order = 1 if sampling == "trilinear" else 0
    expected = ndimage.map_coordinates(data.astype(np.float64), np.moveaxis(points, -1, 0), order=order,
                                       mode="grid-constant", cval=0.0)
    if order:
        assert image.dtype == np.float32
        np.testing.assert_allclose(image, expected, atol=1e-2)
    else:
        assert image.dtype == np.uint8
        np.testing.assert_array_equal(image[valid], expected[valid])


def test_sample_points_maps_segment_xyz_into_the_volume(volume):
    path, data = volume
    extractor = VolumeSlabExtractor([path], transverse_size=4, ray_length=4, sampling="nearest",
                                    segment_to_volume_xyz=[_transform()])
    points_xyz = np.array([[4.0, 10.0, 6.0], [20.0, 30.0, 12.0]])
    volume_xyz = points_xyz * 0.5 + (1.25, -2.0, 0.25)
    index_zyx = np.rint(volume_xyz[:, ::-1]).astype(int)
    np.testing.assert_array_equal(extractor.sample_points(0, points_xyz), data[tuple(index_zyx.T)])
