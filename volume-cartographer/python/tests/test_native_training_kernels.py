"""Native CPU preparation APIs; select an isolated build with --vc-build-python."""
import gc

import numpy as np
import pytest
import vc.volume as native


def _geometry_args(mode="none", shape=(7, 9, 11)):
    rng = np.random.default_rng(20260902)
    controls = []
    if mode == "1d":
        controls = [rng.uniform(-1, 1, 5).astype(np.float32)]
    elif mode in ("2d", "3d"):
        controls = [rng.uniform(-1, 1, (5, 6)).astype(np.float32)
                    for _ in range(1 if mode == "2d" else 3)]
    return dict(
        output_shape_zyx=shape,
        cp_local_zyx=np.array([2.25, 3.5, 4.75], np.float32),
        cp_volume_zyx=np.array([3.5, 2.75, 4.25], np.float32),
        output_to_source_zyx=np.array([[.9, .1, -.2], [-.1, 1.1, .2],
                                      [.2, -.1, .8]], np.float32),
        smooth_mode=mode, smooth_controls=controls,
        smooth_origin_zyx=np.array([-2, -3, -4], np.float32),
        smooth_spacing_zyx=np.array([1.25, 1.5, 1.75], np.float32),
        volume_spacing_base=2.0, base_shape_zyx=(40, 40, 40),
    )


def _control_sample(control, coordinates, origins, spacings):
    lo, hi, weights = [], [], []
    for axis, coordinate in enumerate(coordinates):
        u = np.clip((coordinate - origins[axis]) / spacings[axis],
                    0, control.shape[axis] - 1)
        lower = np.floor(u).astype(np.int64)
        lo.append(lower)
        hi.append(np.minimum(lower + 1, control.shape[axis] - 1))
        weights.append(u - lower.astype(np.float32))
    if control.ndim == 1:
        return control[lo[0]] * (1 - weights[0]) + control[hi[0]] * weights[0]
    a, b = weights
    top = control[lo[0], lo[1]] * (1 - b) + control[lo[0], hi[1]] * b
    bottom = control[hi[0], lo[1]] * (1 - b) + control[hi[0], hi[1]] * b
    return top * (1 - a) + bottom * a


def _geometry_reference(args):
    relative = np.indices(args["output_shape_zyx"], dtype=np.float32)
    relative -= args["cp_local_zyx"][:, None, None, None]
    matrix = args["output_to_source_zyx"]
    source = np.stack([row[0] * relative[0] + row[1] * relative[1] + row[2] * relative[2]
                       for row in matrix])
    mode, controls = args["smooth_mode"], args["smooth_controls"]
    origin, spacing = args["smooth_origin_zyx"], args["smooth_spacing_zyx"]
    if mode == "1d":
        source[0] -= _control_sample(controls[0], [source[2]], [origin[2]], [spacing[2]])
    elif mode == "2d":
        source[0] -= _control_sample(controls[0], source[[1, 2]], origin[[1, 2]], spacing[[1, 2]])
    elif mode == "3d":
        # Inverse shears compose sequentially: later lookups use the updated
        # coordinates, not three displacements from the original grid.
        for target, axes in ((2, [0, 1]), (1, [0, 2]), (0, [1, 2])):
            source[target] -= _control_sample(controls[target], source[axes],
                                               origin[axes], spacing[axes])
    coords = np.moveaxis((source + args["cp_volume_zyx"][:, None, None, None])
                         * args["volume_spacing_base"], 0, -1)
    valid = np.isfinite(coords).all(axis=-1)
    valid &= (coords >= 0).all(axis=-1)
    valid &= (coords <= np.array(args["base_shape_zyx"]) - 1).all(axis=-1)
    return coords, valid


@pytest.mark.parametrize("mode", ["none", "1d", "2d", "3d"])
def test_coordinates_match_independent_inverse_warp(mode):
    args = _geometry_args(mode)
    expected, valid = _geometry_reference(args)
    actual, mask = native.generate_fiber3d_coords(**args)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=8e-6)
    np.testing.assert_array_equal(mask, valid)
    repeated = native.generate_fiber3d_coords(**args)
    np.testing.assert_array_equal(actual, repeated[0])
    np.testing.assert_array_equal(mask, repeated[1])
    assert actual.shape == (*args["output_shape_zyx"], 3)
    assert actual.dtype == np.float32 and mask.dtype == np.uint8
    assert actual.flags.c_contiguous and mask.flags.c_contiguous


@pytest.mark.parametrize("mode", ["1d", "2d", "3d"])
def test_singleton_controls_and_clamped_control_boundaries(mode):
    args = _geometry_args(mode, (1, 3, 5))
    rank = 1 if mode == "1d" else 2
    args["smooth_controls"] = [np.full((1,) * rank, .75, np.float32)
                               for _ in args["smooth_controls"]]
    expected, valid = _geometry_reference(args)
    actual, mask = native.generate_fiber3d_coords(**args)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=8e-6)
    np.testing.assert_array_equal(mask, valid)


def test_coordinate_validity_includes_exact_volume_faces():
    args = _geometry_args(shape=(3, 3, 3))
    args.update(cp_local_zyx=np.ones(3, np.float32),
                cp_volume_zyx=np.zeros(3, np.float32),
                output_to_source_zyx=np.eye(3, dtype=np.float32),
                volume_spacing_base=1., base_shape_zyx=(2, 2, 2))
    coords, valid = native.generate_fiber3d_coords(**args)
    expected = np.zeros((3, 3, 3), np.uint8)
    expected[1:, 1:, 1:] = 1
    np.testing.assert_array_equal(valid, expected)
    np.testing.assert_array_equal(coords[1, 1, 1], [0, 0, 0])
    np.testing.assert_array_equal(coords[2, 2, 2], [1, 1, 1])


@pytest.mark.parametrize("changes,match", [
    ({"output_shape_zyx": (0, 2, 3)}, "positive"),
    ({"output_shape_zyx": (2147483647,) * 3}, "overflows"),
    ({"base_shape_zyx": (0, 2, 3)}, "positive"),
    ({"cp_local_zyx": np.zeros(2, np.float32)}, "shape"),
    ({"cp_volume_zyx": np.full(3, np.nan, np.float32)}, "finite"),
    ({"output_to_source_zyx": np.full((3, 3), np.inf, np.float32)}, "finite"),
    ({"smooth_spacing_zyx": np.zeros(3, np.float32)}, "positive"),
    ({"volume_spacing_base": 0.}, "positive"),
    ({"volume_spacing_base": float("nan")}, "finite"),
    ({"smooth_mode": "other"}, "smooth_mode"),
    ({"smooth_mode": "3d"}, "count"),
    ({"smooth_mode": "1d", "smooth_controls": [np.zeros((1, 2), np.float32)]}, "rank"),
    ({"smooth_mode": "1d", "smooth_controls": [np.zeros(0, np.float32)]}, "empty"),
    ({"smooth_mode": "1d", "smooth_controls": [np.full(2, np.nan, np.float32)]}, "finite"),
])
def test_coordinate_validation(changes, match):
    with pytest.raises(ValueError, match=match):
        native.generate_fiber3d_coords(**(_geometry_args() | changes))


def _blur_reference(volume, direction, along, across):
    """Independent trilinear sampling of each oriented 1D Gaussian pass."""
    direction = np.array(direction, np.float32, copy=True)
    direction /= np.linalg.norm(direction)
    reference = [0, 1, 0] if abs(direction[0]) > .9 else [1, 0, 0]
    a = np.cross(direction, reference).astype(np.float32)
    a /= np.linalg.norm(a)
    b = np.cross(direction, a)
    b /= np.linalg.norm(b)
    result = volume.copy()
    grid = np.indices(volume.shape, dtype=np.float32)
    for axis, sigma in zip((direction, a, b), (along, across, across), strict=True):
        if not sigma:
            continue
        radius = max(1, int(np.ceil(3 * np.float32(sigma))))
        taps = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-.5 * (taps / np.float32(sigma)) ** 2)
        kernel /= kernel.sum()
        output = np.zeros_like(result)
        for tap, weight in zip(taps, kernel, strict=True):
            coordinates = grid + (tap * axis)[:, None, None, None]
            lower = np.floor(coordinates).astype(np.int64)
            fraction = coordinates - lower
            for corner in np.ndindex(2, 2, 2):
                index = tuple(np.clip(lower[d] + corner[d], 0, volume.shape[d] - 1)
                              for d in range(3))
                interpolation = np.prod([fraction[d] if corner[d] else 1 - fraction[d]
                                         for d in range(3)], axis=0)
                output += (result[index] * interpolation * weight).astype(np.float32)
        result = output
    return result


@pytest.mark.parametrize("axis", [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
                                  [0., 0., 1.], [1., 2., -3.]])
@pytest.mark.parametrize("sigmas", [(0., 0.), (.8, 0.), (0., .4), (.8, .4)])
def test_blur_matches_oriented_trilinear_reference(axis, sigmas):
    source = np.random.default_rng(20260903).random((7, 9, 11), dtype=np.float32)
    original = source.copy()
    direction = np.array(axis, np.float32)
    actual = native.anisotropic_blur_3d(source, direction, *sigmas)
    expected = _blur_reference(source, direction, *sigmas)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(actual, native.anisotropic_blur_3d(source, direction, *sigmas))
    assert not np.shares_memory(source, actual)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous


def test_blur_preserves_constant_volume_and_replicates_edges():
    constant = np.full((1, 2, 3), .75, np.float32)
    actual = native.anisotropic_blur_3d(constant, np.array([1, 2, 3], np.float32), 2., 1.)
    np.testing.assert_allclose(actual, constant, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("changes,match", [
    ({"volume": np.zeros((2, 3), np.float32)}, "shape"),
    ({"volume": np.zeros((0, 2, 3), np.float32)}, "positive"),
    ({"axis_zyx": np.zeros(3, np.float32)}, "norm"),
    ({"axis_zyx": np.full(3, np.nan, np.float32)}, "finite"),
    ({"axis_zyx": np.zeros(2, np.float32)}, "shape"),
    ({"sigma_along": -.1}, "nonnegative"),
    ({"sigma_across": float("inf")}, "finite"),
])
def test_blur_validation(changes, match):
    args = dict(volume=np.zeros((2, 3, 4), np.float32),
                axis_zyx=np.array([1, 0, 0], np.float32), sigma_along=.8, sigma_across=.4)
    with pytest.raises(ValueError, match=match):
        native.anisotropic_blur_3d(**(args | changes))


def _value_reference(source, valid, mode, contrast, brightness, gamma):
    values = source[valid] if valid.any() else source.reshape(-1)
    mean = np.float32(values.mean(dtype=np.float64))
    normalized, center = source.copy(), mean
    if valid.any() and mode == "zscore":
        std = max(np.float32(np.sqrt(np.mean((values.astype(np.float64) - mean) ** 2))), 1e-6)
        normalized, center = (source - mean) / std, np.float32(0)
    elif valid.any() and mode == "minmax":
        divisor = max(values.max() - values.min(), 1e-6)
        normalized, center = (source - values.min()) / divisor, (mean - values.min()) / divisor
    result = (normalized - center) * np.float32(contrast) + center + np.float32(brightness)
    selected = result[valid] if valid.any() else result.reshape(-1)
    lo, hi = selected.min(), selected.max()
    if gamma != 1:
        result = np.clip((result - lo) / max(hi - lo, 1e-6), 0, 1) ** np.float32(gamma) * (hi - lo) + lo
    return result


@pytest.mark.parametrize("mode", ["none", "raw", "identity", "zscore", "minmax"])
@pytest.mark.parametrize("mask_kind", ["all", "partial", "none", "constant"])
def test_value_transform_matches_reference(mode, mask_kind):
    rng = np.random.default_rng(20260904)
    source = rng.normal(size=(7, 9, 11)).astype(np.float32)
    valid = rng.random(source.shape) > .2
    if mask_kind in ("all", "none"):
        valid[:] = mask_kind == "all"
    if mask_kind == "constant":
        source[:] = 3.25
    original, original_valid = source.copy(), valid.copy()
    args = (source, valid, mode, 1.7, .13, .71)
    actual = native.fiber3d_value_transform(*args)
    expected = _value_reference(*args)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual, native.fiber3d_value_transform(*args))
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(valid, original_valid)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous


@pytest.mark.parametrize("mode", ["none", "raw", "identity"])
def test_value_transform_identity_owns_output(mode):
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    actual = native.fiber3d_value_transform(source, np.ones(source.shape, bool), mode, 1., 0., 1.)
    np.testing.assert_array_equal(actual, source)
    assert not np.shares_memory(actual, source)
    del source
    gc.collect()
    assert actual.sum() == 276


@pytest.mark.parametrize("changes,match", [
    ({"volume": np.zeros((2, 3), np.float32)}, "shape"),
    ({"valid_mask": np.ones((1, 3, 4), bool)}, "shape must match"),
    ({"volume": np.zeros((0, 3, 4), np.float32), "valid_mask": np.ones((0, 3, 4), bool)}, "positive"),
    ({"normalization": "other"}, "unsupported"),
    ({"contrast": 0.}, "positive"),
    ({"brightness": float("nan")}, "finite"),
    ({"gamma": -.5}, "positive"),
])
def test_value_transform_validation(changes, match):
    args = dict(volume=np.zeros((2, 3, 4), np.float32), valid_mask=np.ones((2, 3, 4), bool),
                normalization="zscore", contrast=1., brightness=0., gamma=1.)
    with pytest.raises(ValueError, match=match):
        native.fiber3d_value_transform(**(args | changes))


def test_real_ct_patch_end_to_end(request):
    path = request.config.getoption("--vc-real-volume")
    if not path:
        pytest.skip("supply --vc-real-volume for real CT preparation validation")
    xyz_origin = tuple(float(v) for v in request.config.getoption("--vc-origin-xyz").split(","))
    volume = native.Volume.open(path)
    native.initialize_chunk_cache(256 * 1024**2, 1, 1, 1)
    args = _geometry_args("3d", (32, 32, 32))
    args.update(cp_local_zyx=np.full(3, 15.5, np.float32),
                cp_volume_zyx=np.array(xyz_origin[::-1], np.float32) + 15.5,
                volume_spacing_base=1., base_shape_zyx=(100000, 100000, 100000))
    coords, valid = native.generate_fiber3d_coords(**args)
    expected, expected_valid = _geometry_reference(args)
    np.testing.assert_allclose(coords, expected, rtol=0, atol=.002)
    np.testing.assert_array_equal(valid, expected_valid)

    def sample(grid, mask):
        xyz = np.ascontiguousarray(grid[..., ::-1].reshape(32 * 32, 32, 3))
        return volume.sample_coords(xyz, np.ascontiguousarray(mask.reshape(32 * 32, 32), bool),
                                    sampler_threads=1, level=request.config.getoption("--vc-level"))[0].reshape(32, 32, 32)

    sampled = sample(coords, valid)
    reference = sample(expected, expected_valid)
    assert np.count_nonzero(sampled), "CT crop is empty"
    # At real base-coordinate magnitudes, float32 reassociation can round an
    # interpolated uint8 intensity by one. This does not relax native sampling.
    assert np.abs(sampled.astype(np.int16) - reference.astype(np.int16)).max() <= 1
    image = sampled.astype(np.float32) / 255
    direction = np.array([1, 2, -3], np.float32)
    blurred = native.anisotropic_blur_3d(image, direction, .8, .4)
    np.testing.assert_allclose(blurred, _blur_reference(image, direction, .8, .4), rtol=2e-6, atol=2e-6)
    transformed = native.fiber3d_value_transform(blurred, valid.astype(bool), "zscore", 1.7, .13, .71)
    expected_values = _value_reference(blurred, valid.astype(bool), "zscore", 1.7, .13, .71)
    error = np.abs(transformed - expected_values)
    scale = max(1., float(np.max(np.abs(expected_values))))
    print(f"value reference: max_abs={error.max()}, mean_abs={error.mean()}, scale={scale}")
    # Use the original native-value regression's scale-normalized error bounds,
    # rather than a relative tolerance at values crossing through zero.
    assert error.max() <= 1e-6 * scale
    assert error.mean() <= 1e-7 * scale
    print(f"real CT: {path}, origin XYZ={xyz_origin}, shape={sampled.shape}, nonzero={np.count_nonzero(sampled)}")
