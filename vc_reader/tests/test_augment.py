"""Native augmentation contracts and independent numerical references."""
import pickle
import pytest

np = pytest.importorskip("numpy")
A = pytest.importorskip("vc_reader.augment")


def example(shape=(5, 7), dtype=np.float32):
    image = np.linspace(-1, 2, np.prod(shape), dtype=dtype).reshape(1, *shape)
    return {"image": image, "mask": (image > 0).astype(np.int16), "patch_info": {"id": "patch-1"}}


def transforms():
    return [
        A.BrightnessMultiply(), A.BrightnessAdd(), A.Contrast(), A.Gamma(), A.Invert(),
        A.GaussianNoise(), A.RicianNoise(), A.ClipOutliers(), A.GaussianBlur(),
        A.MedianFilter(size=3), A.Sharpen(), A.BrightnessGradient(scale=3),
        A.LocalContrast(scale=3), A.LocalGamma(scale=3), A.LocalSmoothing(scale=3),
        A.BlankRectangle(size=2), A.SliceIllumination(axis="y"), A.Smear(axis="y"),
        A.Flip(axes="yx"), A.Rotate90(planes=("yx",), turns=(2,)), A.Transpose(),
        A.Affine(degrees=(-10, 10)), A.Elastic(sigma=2, magnitude=0.2),
        A.SheetCompression(axes="y"), A.LowResolution(), A.ThickSlice(axes="y"),
        A.Morphology(radius=1), A.RandomBinaryOperator(channels=(0,), radius=1),
        A.RemoveComponents(channels=(0,)), A.ReplaceLabel(value=1, replacement=2),
        A.Regions(labels=[0, [1, 2]]), A.OneHot(labels=[0, 1]), A.MaskImage(),
        A.MedialSurface(tube=False), A.DeepSupervision(scales=[1, .5]),
        A.OneOf([A.Gamma(), A.GaussianNoise()]),
    ]


@pytest.mark.parametrize("transform_index", range(len(transforms())), ids=[repr(t) for t in transforms()])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(5, 7), (3, 5, 7)])
def test_catalog_replay_and_ownership(transform_index, dtype, shape):
    transform = transforms()[transform_index]
    sample = example(shape=shape, dtype=dtype)
    original = {k: v.copy() for k, v in sample.items() if isinstance(v, np.ndarray)}
    pipeline = A.Pipeline([transform])
    expected = pipeline(sample, seed=123)
    plan = pipeline.sample(sample, seed=123)
    replay = pipeline.apply(sample, plan)
    restored = pickle.loads(pickle.dumps(pipeline))(sample, seed=123)
    restored_plan = pickle.loads(pickle.dumps(plan)).apply(sample)
    for key, array in expected.items():
        if isinstance(array, np.ndarray):
            for result in (replay, restored, restored_plan):
                np.testing.assert_array_equal(array, result[key])
            assert np.isfinite(array).all()
            if key in original:
                assert not np.shares_memory(array, sample[key])
        elif isinstance(array, list):
            for a, b in zip(array, replay[key]):
                np.testing.assert_array_equal(a, b)
    for key, before in original.items():
        np.testing.assert_array_equal(sample[key], before)
    assert expected["patch_info"] is sample["patch_info"]


def test_flip_preserves_uint64_and_points_normals():
    shape = (3, 4, 5)
    sample = {
        "image": np.arange(60, dtype=np.float32).reshape(1, *shape),
        "labels": (np.arange(60, dtype=np.uint64) + 2**63).reshape(1, *shape),
        "points": np.array([[0, 1, 2], [2, 3, 4]], dtype=np.float64),
        "normals": np.zeros((3, *shape), dtype=np.float32),
        "vectors": np.array([[1, 2, 3]], dtype=np.float32),
    }
    sample["normals"][0] = 1
    targets = {"image": A.Image(), "labels": A.Label(), "points": A.Points(),
               "normals": A.NormalField(), "vectors": A.Vectors()}
    p = A.Pipeline([A.Flip(axes="zx", p_axis=1)], targets=targets)
    out = p(sample, seed=3)
    np.testing.assert_array_equal(out["labels"], sample["labels"][:, ::-1, :, ::-1])
    np.testing.assert_array_equal(out["points"], [[2, 1, 2], [0, 3, 0]])
    np.testing.assert_array_equal(out["vectors"], [[-1, 2, -3]])
    np.testing.assert_array_equal(out["normals"][0], -1)


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_rot90_alignment(turns):
    sample = example((3, 5))
    sample["points"] = np.array([[1, 2]], dtype=np.float32)
    p = A.Pipeline([A.Rotate90(planes=("yx",), turns=(turns,), resize=True)],
                   targets={"image": A.Image(), "mask": A.Label(), "points": A.Points(order="yx")})
    out = p(sample, seed=1)
    np.testing.assert_array_equal(out["image"], np.rot90(sample["image"], turns, axes=(1, 2)))
    y, x = out["points"][0].astype(int)
    assert out["image"][0, y, x] == sample["image"][0, 1, 2]


def test_padding_and_ignore_labels():
    sample = example((3, 3))
    sample["padding_mask"] = np.ones_like(sample["image"])
    p = A.Pipeline([A.CropPad(shape=(5, 5)), A.BrightnessAdd(shift=3)],
                   targets={"image": A.Image(), "mask": A.Label(ignore_value=-1), "padding_mask": A.ValidMask()})
    out = p(sample, seed=0)
    valid = out["padding_mask"].astype(bool)
    assert valid.sum() == 9
    assert (out["image"][~valid] == 0).all()
    assert (out["mask"][~valid] == -1).all()
    np.testing.assert_array_equal(out["image"][valid], sample["image"].ravel() + 3)


def test_affine_normals_use_inverse_transpose():
    shape = (5, 5)
    sample = {"image": np.ones((1, *shape), np.float64), "normals": np.zeros((2, *shape)), "vectors": np.array([[1., 0.]])}
    sample["normals"][0] = 1
    p = A.Pipeline([A.Affine(shear=.5, border=True)], targets={
        "image": A.Image(), "normals": A.NormalField(components="yx"), "vectors": A.Vectors(components="yx")})
    out = p(sample, seed=0)
    expected = np.array([1., -.5]) / np.sqrt(1.25)
    np.testing.assert_allclose(out["normals"][:, 2, 2], expected)
    np.testing.assert_allclose(out["vectors"], [[1, 0]])


def test_pseudo2d_roundtrip():
    sample = example((3, 5, 7))
    p = A.Pipeline([A.ToPseudo2D(), A.Flip(axes="x", p_axis=1), A.FromPseudo2D()])
    out = p(sample, seed=1)
    np.testing.assert_array_equal(out["image"], sample["image"][..., ::-1])


def test_data_dependent_geometry_replays_original_field():
    sample = example((5, 7))
    p = A.Pipeline([A.SheetCompression(axes="y", intensity_sigma=0, smoothing=0)])
    plan = p.sample(sample, seed=12)
    changed = dict(sample, image=np.zeros_like(sample["image"]))
    np.testing.assert_array_equal(p.apply(changed, plan)["mask"], p.apply(sample, plan)["mask"])


def test_plan_schema_validation_and_corruption():
    p = A.Pipeline([A.Gamma()])
    plan = p.sample(example(), seed=1)
    with pytest.raises(ValueError, match="identical"):
        p.apply(example((6, 7)), plan)
    payload = bytearray(plan._native.dump())
    payload[10] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        p._native.load_plan(bytes(payload))


def test_fail_loudly_for_undeclared_arrays_and_invalid_inputs():
    p = A.Pipeline([])
    with pytest.raises(ValueError, match="declaration"):
        p(dict(example(), new_label=np.ones((1, 5, 7))), seed=1)
    with pytest.raises(TypeError):
        A.Gamma(misspelled_argument=True)
    with pytest.raises(ValueError, match="probability"):
        A.Gamma(p=1.1)
    with pytest.raises((TypeError, ValueError)):
        p({"image": np.ones((1, 5, 7), dtype=np.uint8)}, seed=1)
    with pytest.raises(ValueError, match="nonfinite"):
        p({"image": np.full((1, 5, 7), np.nan)}, seed=1)


def test_seed_types_global_rng_and_optional_fields():
    assert A.seed(1, "2") == A.seed(1, "2")
    assert A.seed(1, "2") != A.seed("1", 2)
    np.random.seed(123)
    state = np.random.get_state()
    p = A.Pipeline([A.GaussianNoise(std=.1)])
    x = p(example(), seed=9)
    y = p({"image": example()["image"], "mask": None}, seed=9)
    np.testing.assert_array_equal(x["image"], y["image"])
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    assert state[2:] == np.random.get_state()[2:]


def test_shared_parameters_and_noise_are_separate():
    sample = {"image": np.zeros((2, 10, 10), np.float32)}
    independent = A.Pipeline([A.GaussianNoise(std=.1, shared=True)])(sample, seed=1)["image"]
    shared = A.Pipeline([A.GaussianNoise(std=.1, shared=True, shared_noise=True)])(sample, seed=1)["image"]
    assert not np.array_equal(independent[0], independent[1])
    np.testing.assert_array_equal(shared[0], shared[1])


def test_points_drop_and_empty():
    p = A.Pipeline([A.CropPad(shape=(2, 2))], targets={"image": A.Image(), "points": A.Points(order="yx", outside="drop")})
    out = p({"image": np.ones((1, 4, 4), np.float32), "points": np.array([[0, 0]], np.float32)}, seed=0)
    assert out["points"].shape == (0, 2)
    p(out, seed=0)


@pytest.mark.parametrize("shape", [(9, 11), (5, 7, 9)])
def test_scipy_filter_references(shape):
    ndimage = pytest.importorskip("scipy.ndimage")
    image = np.random.default_rng(0).normal(size=(1, *shape))
    for transform, reference in [
        (A.GaussianBlur(sigma=.8, truncate=8, boundary="reflect"), ndimage.gaussian_filter(image[0], .8, truncate=4)),
        (A.MedianFilter(size=4), ndimage.median_filter(image[0], size=4)),
    ]:
        result = A.Pipeline([transform])({"image": image}, seed=1)["image"][0]
        np.testing.assert_allclose(result, reference, atol=1e-12, rtol=1e-12)


def test_torch_cpu_conversion_and_resize_reference():
    torch = pytest.importorskip("torch")
    image = torch.arange(105, dtype=torch.float32).reshape(1, 5, 7, 3)
    sample = {"image": image}
    out = A.Pipeline([A.LowResolution(scale=.5)])(sample, seed=1)["image"]
    reference = torch.nn.functional.interpolate(image[None], size=(2, 4, 2), mode="nearest-exact")
    reference = torch.nn.functional.interpolate(reference, size=(5, 7, 3), mode="trilinear")[0]
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, reference, rtol=1e-6, atol=1e-5)
    assert image.data_ptr() != out.data_ptr()
    with pytest.raises(ValueError, match="autograd"):
        A.Pipeline([])({"image": image.requires_grad_()}, seed=1)


@pytest.mark.parametrize("shape", [(9, 11), (7, 9, 11)])
@pytest.mark.parametrize("sigma", [.3, .8, 1.25])
def test_gaussian_blur_torch_reference(shape, sigma):
    torch = pytest.importorskip("torch")
    image = torch.tensor(np.random.default_rng(1).normal(size=(1, *shape)), dtype=torch.float64)
    # Existing blur uses an odd full width near 6*sigma, with reflection that
    # excludes the edge pixel (Torch reflect / SciPy mirror).
    radius = int((6 * sigma + .5) // 2)
    x = torch.arange(-radius, radius + 1, dtype=image.dtype)
    kernel = torch.exp(-.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    reference = image[None]
    conv = torch.nn.functional.conv2d if len(shape) == 2 else torch.nn.functional.conv3d
    for axis in range(len(shape)):
        kernel_shape = [1, 1] + [1] * len(shape)
        kernel_shape[axis + 2] = len(kernel)
        pad = [0] * (2 * len(shape))
        pad[2 * (len(shape) - axis - 1):2 * (len(shape) - axis)] = [radius, radius]
        reference = conv(torch.nn.functional.pad(reference, pad, mode="reflect"), kernel.reshape(kernel_shape))
    out = A.Pipeline([A.GaussianBlur(sigma=sigma)])({"image": image}, seed=1)["image"]
    torch.testing.assert_close(out, reference[0], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("size", [1, 2, 3])
def test_closing_repeated_cross_reference(size):
    ndimage = pytest.importorskip("scipy.ndimage")
    labels = np.random.default_rng(0).integers(0, 2, (1, 15, 17, 19), dtype=np.uint8) * 3
    footprint = ndimage.generate_binary_structure(3, 1)
    for axis in range(3):
        footprint = np.repeat(footprint, size, axis=axis)
    reference = ndimage.binary_closing(labels[0] > 0, structure=footprint) * 3
    out = A.Pipeline([A.Morphology(radius=size, footprint="repeated_cross")])({"mask": labels}, seed=1)
    np.testing.assert_array_equal(out["mask"][0], reference)


@pytest.mark.parametrize("transform", ["ReplaceLabel", "Morphology", "Regions", "OneHot"])
def test_value_based_labels_reject_inexact_ids(transform):
    options = {"Regions": {"labels": [1]}, "OneHot": {"labels": [1]}}.get(transform, {})
    pipeline = A.Pipeline([getattr(A, transform)(**options)])
    with pytest.raises(ValueError, match="integer IDs"):
        pipeline({"mask": np.full((1, 3, 3), 2**63 + 1, dtype=np.uint64)}, seed=1)


def test_teacher_student_shared_geometry():
    sample = example((7, 7))
    geometry = A.Pipeline([A.Rotate90(), A.Flip(axes="yx")])
    aligned = geometry.apply(sample, geometry.sample(sample, seed=4))
    student = A.Pipeline([A.GaussianNoise(std=.5)])(aligned, seed=5)
    np.testing.assert_array_equal(student["mask"], aligned["mask"])
    assert not np.array_equal(student["image"], aligned["image"])


def test_strip_geometry_and_surface_perturb():
    shape = (9, 11)
    yy, xx = np.indices(shape, dtype=np.float64)
    surface = np.stack([np.zeros(shape), yy, xx])
    sample = {"image": (yy + xx)[None], "surface": surface,
              "points": np.array([[4., 5.]])}
    targets = {"image": A.Image(), "surface": A.CoordinateField(), "points": A.Points(order="yx")}
    pipeline = A.Pipeline([A.StripGeometry(smooth_offset=.2, stride=3),
                           A.SurfacePerturb(amplitude=.25, p=1)], targets=targets)
    plan = pipeline.sample(sample, seed=12)
    out = plan.apply(sample)
    assert np.any(out["surface"][0] != 0)
    assert np.max(np.abs(out["surface"][0])) <= 1.5
    np.testing.assert_array_equal(out["surface"], pickle.loads(pickle.dumps(plan)).apply(sample)["surface"])
    assert np.isfinite(out["points"]).all()


def test_crop_translation_keeps_supervision():
    points = np.array([[50., 50., 50.], [50., 60., 60.]])
    box = (0, 0, 0, 100, 100, 100)
    out = A.translate_crop_box(box, points, seed=1, p=1, minimum=10, maximum=10, keep=1)
    assert out != box
    assert ((points >= out[:3]) & (points < out[3:])).all()
    assert A.translate_crop_box(box, np.empty((0, 3)), seed=1) == box


def test_empty_and_malformed_inputs_raise():
    for value in [np.array(1.), np.ones((3,)), np.ones((1, 0, 3))]:
        with pytest.raises(ValueError):
            A.Pipeline([])({"image": value}, seed=1)
    with pytest.raises(ValueError):
        A.Pipeline([])({}, seed=1)


@pytest.mark.parametrize("fill", [.5, float(2**64)])
def test_integer_label_fill_must_be_representable(fill):
    pipeline = A.Pipeline([A.CropPad((5, 5))], targets={"mask": A.Label(fill=fill)})
    with pytest.raises(ValueError, match="representable"):
        pipeline({"mask": np.zeros((1, 3, 3), np.uint64)}, seed=1)


def test_multichannel_pseudo2d_validity():
    image = np.ones((2, 3, 5, 7), np.float32)
    valid = np.ones((1, 3, 5, 7), np.float32)
    valid[:, :, 0] = 0
    pipeline = A.Pipeline([A.ToPseudo2D(), A.BrightnessAdd(shift=1), A.FromPseudo2D()])
    out = pipeline({"image": image, "padding_mask": valid}, seed=1)
    np.testing.assert_array_equal(out["image"], np.broadcast_to(valid, image.shape) * 2)


@pytest.mark.parametrize("transform_index", range(len(transforms())), ids=[repr(t) for t in transforms()])
def test_inplace_matches_safe_pipeline(transform_index):
    transform = transforms()[transform_index]
    sample = example((3, 5, 7))
    pipeline = A.Pipeline([transform])
    expected = pipeline(sample, seed=123)
    actual = pipeline(sample, seed=123, inplace=True)
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[key], value)
        elif isinstance(value, list):
            for a, b in zip(actual[key], value):
                np.testing.assert_array_equal(a, b)


def test_inplace_ownership_and_reshape():
    import gc
    image = np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5)
    pipeline = A.Pipeline([A.ToPseudo2D(), A.BrightnessAdd(.5)])
    out = pipeline({"image": image}, seed=1, inplace=True)["image"]
    assert out.shape == (3, 4, 5)
    assert np.shares_memory(out, image)
    expected = np.arange(60, dtype=np.float32).reshape(3, 4, 5) + .5
    del image, pipeline
    gc.collect()
    np.testing.assert_array_equal(out, expected)
    out[:] = 2
    assert np.all(out == 2)


def test_inplace_rejects_readonly_strides_and_aliasing():
    pipeline = A.Pipeline([A.BrightnessAdd(1)])
    image = np.ones((1, 3, 5), dtype=np.float32)
    image.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        pipeline({"image": image}, seed=1, inplace=True)
    with pytest.raises(ValueError, match="writable"):
        pipeline._native.run({"image": image}, 1, False, False, True)
    # The default continues accepting readonly and strided inputs.
    np.testing.assert_array_equal(pipeline({"image": image[:, :, ::-1]}, seed=1)["image"], 2)
    image.flags.writeable = True
    with pytest.raises(ValueError, match="contiguous"):
        pipeline({"image": image[:, :, ::-1]}, seed=1, inplace=True)
    with pytest.raises(ValueError, match="nonoverlapping"):
        pipeline({"image": image, "mask": image}, seed=1, inplace=True)
    np.testing.assert_array_equal(image, 1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(1, 5), (4, 7), (1, 3, 5), (3, 4, 7)])
def test_resize_identity_and_half_up_nearest(dtype, shape):
    image = np.arange(np.prod(shape), dtype=dtype).reshape(1, *shape)
    pipeline = A.Pipeline([A.DeepSupervision([1, .6])])
    out = pipeline({"mask": image}, seed=5)["mask"]
    np.testing.assert_array_equal(out[0], image)
    # Deep supervision uses half-up coordinates, unlike ThickSlice's legacy floor.
    expected = image
    for axis, extent in enumerate(shape, start=1):
        size = max(1, round(extent * .6))
        indices = np.floor((np.arange(size) + .5) * extent / size).astype(int)
        expected = np.take(expected, indices, axis=axis)
    np.testing.assert_array_equal(out[1], expected)


def test_image_without_validity_mask():
    pipeline = A.Pipeline([A.BrightnessAdd(1)], targets={"image": A.Image(valid=None)})
    np.testing.assert_array_equal(pipeline({"image": np.zeros((1, 2, 3), np.float32)}, seed=0)["image"], 1)


def test_inplace_reader_buffer_survives_close_and_reuse(tmp_path):
    import gc
    import json
    from vc_reader.reader import ZarrReader

    shape = (3, 4, 5)
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    (tmp_path / '.zarray').write_text(json.dumps({
        "zarr_format": 2, "shape": shape, "chunks": shape, "dtype": "<f4",
        "compressor": None, "filters": None, "fill_value": 0, "order": "C",
    }))
    (tmp_path / '0.0.0').write_bytes(expected.tobytes())
    pipeline = A.Pipeline([A.BrightnessAdd(.5)])
    with ZarrReader.open(str(tmp_path)) as reader:
        image = reader.read((0, 0, 0), shape)[None]
        out = pipeline({"image": image}, seed=1, inplace=True)["image"]
        assert np.shares_memory(out, image)
        del image
        gc.collect()
        # Retained augmented outputs must not return their buffers to the pool.
        for _ in range(3):
            np.testing.assert_array_equal(reader.read((0, 0, 0), shape), expected)
    gc.collect()
    np.testing.assert_array_equal(out[0], expected + .5)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vectorized_validation_rejects_nonfinite(value, dtype):
    image = np.zeros((1, 3, 17), dtype=dtype)
    image.ravel()[35] = value
    with pytest.raises(ValueError, match="nonfinite"):
        A.Pipeline([])({"image": image}, seed=1)


def test_unaligned_input_is_copied_or_rejected():
    image = np.ndarray((1, 3, 5), dtype=np.float32, buffer=bytearray(61), offset=1)
    image[:] = 2
    pipeline = A.Pipeline([A.BrightnessAdd(1)])
    np.testing.assert_array_equal(pipeline({"image": image}, seed=1)["image"], 3)
    with pytest.raises(ValueError, match="aligned"):
        pipeline({"image": image}, seed=1, inplace=True)
    np.testing.assert_array_equal(image, 2)


@pytest.mark.parametrize("shape", [(1, 5), (7, 11), (2, 3, 5), (5, 9, 13)])
@pytest.mark.parametrize("size", [2, 3, 4, 5, 7])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_median_exact_palette_and_continuous_reference(shape, size, dtype):
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(37)
    # Nonuniform, noninteger values: the histogram must preserve the values,
    # including adjacent representable numbers, without quantizing them.
    palette = np.array([-10, -.1, 0, .2, np.nextafter(dtype(.2), dtype(1)), 100], dtype=dtype)
    for image in (rng.choice(palette, size=(2, *shape)), rng.normal(size=(2, *shape)).astype(dtype)):
        expected = np.stack([ndimage.median_filter(channel, size=size, mode="reflect") for channel in image])
        actual = A.Pipeline([A.MedianFilter(size)])({"image": image}, seed=7)["image"]
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("boundary", ["mirror", "reflect"])
@pytest.mark.parametrize("shape", [(1, 3), (5, 17), (1, 3, 9), (5, 7, 19)])
@pytest.mark.parametrize("sigma", [0., .3, 1., 3.])
def test_gaussian_register_blocks_and_wide_boundaries(shape, sigma, boundary):
    ndimage = pytest.importorskip("scipy.ndimage")
    image = np.random.default_rng(8).normal(size=(1, *shape))
    actual = A.Pipeline([A.GaussianBlur(sigma, boundary=boundary)])({"image": image}, seed=3)["image"][0]
    radius = int((6 * sigma + .5) // 2)
    expected = ndimage.gaussian_filter(image[0], sigma, mode=boundary, radius=radius)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("value", [-0., .123456789])
def test_median_constant_preserves_bits(value):
    image = np.full((1, 3, 5, 7), value, dtype=np.float64)
    actual = A.Pipeline([A.MedianFilter(7)])({"image": image}, seed=3)["image"]
    assert actual.tobytes() == image.tobytes()


def test_median_palette_overflow_after_probe():
    ndimage = pytest.importorskip("scipy.ndimage")
    image = np.zeros((1, 32, 32, 32), dtype=np.float64)
    # The evenly spaced probe sees zeros; the full encoding pass must still
    # detect the large palette and leave the input intact for the fallback.
    image.ravel()[1::8] = np.arange(image.size // 8) / 10000
    expected = ndimage.median_filter(image[0], size=3, mode="reflect")
    actual = A.Pipeline([A.MedianFilter(3)])({"image": image}, seed=3)["image"][0]
    np.testing.assert_array_equal(actual, expected)
