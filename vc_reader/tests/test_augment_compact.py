"""Production compact execution versus the ordinary native path."""
import pickle

import numpy as np
import pytest

from vc_reader import augment as A


def image(shape=(2, 9, 11, 13), dtype=np.float32):
    return (np.random.default_rng(37).integers(0, 256, shape) / 255).astype(dtype)


def chain():
    return [A.MedianFilter(2), A.Flip(axes="z", p_axis=1),
            A.Rotate90(planes=("yx",), turns=(1,), resize=True), A.Transpose(order="xyz"),
            A.BrightnessMultiply(-1.1), A.BrightnessAdd(.05), A.Contrast(1.2),
            A.Gamma(.8, p_retain_stats=1), A.Invert(), A.MedianFilter(3),
            A.Gamma(1.3, p_invert=1, p_retain_stats=1)]


@pytest.mark.parametrize("shape", [(2, 17, 19), (2, 9, 11, 13)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("retain", [0, 1])
@pytest.mark.parametrize("invert", [0, 1])
def test_compact_gamma_and_intensity_chain(shape, dtype, retain, invert):
    source = image(shape, dtype)
    transforms = [A.Gamma(.8, p_retain_stats=retain, p_invert=invert),
                  A.BrightnessMultiply(-1.1), A.BrightnessAdd(.07),
                  A.Contrast(1.3), A.Invert(), A.Gamma(1.7, p_retain_stats=1)]
    expected = A.Pipeline(transforms, compact=False)({"image": source}, seed=17)["image"]
    result = A.Pipeline(transforms)({"image": source}, seed=17)["image"]
    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=1e-6)
    assert result.dtype == source.dtype
    np.testing.assert_array_equal(source, image(shape, dtype))


def test_compact_geometry_median_replay_pickle_and_inplace():
    source = image()
    sample = {"image": source, "mask": (source[:1] > .5).astype(np.uint64) + 2**63}
    pipeline = A.Pipeline(chain())
    expected = A.Pipeline(chain(), compact=False)(sample, seed=3)
    actual = pipeline(sample, seed=3, profile=True)
    np.testing.assert_allclose(actual["image"], expected["image"], atol=2e-6, rtol=1e-6)
    np.testing.assert_array_equal(actual["mask"], expected["mask"])
    assert set(actual["_aug_perf"]) == {str(i) for i in range(len(chain()))}
    plan = pipeline.sample(sample, seed=3)
    for result in (plan.apply(sample), pickle.loads(pickle.dumps(plan)).apply(sample),
                   pickle.loads(pickle.dumps(pipeline))(sample, seed=3),
                   pipeline({k: v.copy() for k, v in sample.items()}, seed=3, inplace=True)):
        for key in sample:
            np.testing.assert_array_equal(result[key], actual[key])
    assert pickle.loads(pickle.dumps(A.Pipeline(chain(), compact=False))).compact is False
    np.testing.assert_array_equal(source, image())


@pytest.mark.parametrize("barrier", [A.GaussianBlur(1), A.GaussianNoise(.02), A.SliceIllumination(),
                                     A.CropPad((7, 9, 11)), A.ThickSlice(.5),
                                     A.SheetCompression(strength=.1, smoothing=2)])
def test_compact_decode_before_ordinary_sampling_and_execution(barrier):
    sample = {"image": image((1, 9, 11, 13))}
    transforms = [A.MedianFilter(3), A.BrightnessAdd(.1), A.Invert(), barrier, A.Gamma(.8)]
    expected = A.Pipeline(transforms, compact=False)(sample, seed=9)["image"]
    pipeline = A.Pipeline(transforms)
    result = pipeline(sample, seed=9)["image"]
    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=1e-6)
    np.testing.assert_array_equal(pipeline.sample(sample, seed=9).apply(sample)["image"], result)


@pytest.mark.parametrize("masked", [False, True])
def test_compact_target_channel_selection_and_validity(masked):
    source = image()
    sample = {"image": source, "other": source * 2 - .5}
    targets = {"image": A.Image(fill=-7), "other": A.Image(), "padding_mask": A.ValidMask()}
    if masked:
        sample["padding_mask"] = np.ones_like(source[:1], dtype=np.uint8)
        sample["padding_mask"][..., 0] = 0
    transforms = [A.Gamma(.8, on="image", channels=(1,)),
                  A.BrightnessAdd(.2, on="other", channels=(0,)),
                  A.MedianFilter(3, on="image", channels=(1,)), A.Invert(on="image", channels=(1,))]
    expected = A.Pipeline(transforms, targets=targets, compact=False)(sample, seed=1)
    result = A.Pipeline(transforms, targets=targets)(sample, seed=1)
    for key in sample:
        np.testing.assert_array_equal(result[key], expected[key])
    if not masked:
        np.testing.assert_array_equal(result["image"][0], source[0])


@pytest.mark.parametrize("kind", ["continuous", "257_values", "constant", "zeros", "signed_zeros"])
def test_compact_fallback_and_degenerate_inputs(kind):
    source = image((1, 17, 19))
    if kind == "continuous": source = np.random.default_rng(1).normal(size=source.shape).astype(np.float32)
    if kind == "257_values": source.ravel()[:] = np.arange(source.size) % 257
    if kind == "constant": source.fill(.123)
    if kind in ("zeros", "signed_zeros"): source.fill(0)
    if kind == "signed_zeros": source.ravel()[::2] = -0.
    transforms = [A.Gamma(.8, p_retain_stats=1), A.Invert(), A.MedianFilter(2)]
    expected = A.Pipeline(transforms, compact=False)({"image": source}, seed=1)["image"]
    result = A.Pipeline(transforms)({"image": source}, seed=1)["image"]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("selected", [(0,), (1,), (0, 1)])
def test_compact_quantized_median_does_not_quantize_unselected_channels(selected):
    source = image()
    source[0] = np.random.default_rng(4).uniform(-1, 2, source.shape[1:])
    transforms = [A.MedianFilter(3, quantize=True, channels=selected), A.Gamma(.8, channels=selected)]
    expected = A.Pipeline(transforms, compact=False)({"image": source}, seed=1)["image"]
    result = A.Pipeline(transforms)({"image": source}, seed=1)["image"]
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=2e-6)
    for channel in set(range(2)) - set(selected):
        np.testing.assert_array_equal(result[channel], source[channel])


def test_compact_nested_oneof_and_skipped_boundary():
    transforms = [A.MedianFilter(3), A.GaussianBlur(1, p=0),
                  A.OneOf([A.OneOf([A.Gamma(.8, p_retain_stats=1)]), A.Invert()]),
                  A.Flip(p_axis=1), A.BrightnessAdd(.03)]
    source = {"image": image()}
    for seed in range(5):
        pipeline = A.Pipeline(transforms)
        actual = pipeline(source, seed=seed)["image"]
        expected = A.Pipeline(transforms, compact=False)(source, seed=seed)["image"]
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-6)
        np.testing.assert_array_equal(pipeline.sample(source, seed=seed).apply(source)["image"], actual)


def test_compact_median_after_signed_zero_generation():
    source = image((1, 9, 11, 13)) * 2 - 1
    transforms = [A.Gamma(1), A.BrightnessMultiply(0), A.MedianFilter(2)]
    expected = A.Pipeline(transforms, compact=False)({"image": source}, seed=1)["image"]
    actual = A.Pipeline(transforms)({"image": source}, seed=1)["image"]
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_compact_late_palette_overflow_falls_back_without_input_changes():
    source = (np.arange(8193) % 256).astype(np.float32).reshape(1, 3, 2731)
    source.ravel()[-2] = 256  # outside the evenly spaced initial probes
    before = source.copy()
    transforms = [A.Gamma(.8, p_retain_stats=1), A.Invert()]
    expected = A.Pipeline(transforms, compact=False)({"image": source}, seed=1)["image"]
    actual = A.Pipeline(transforms)({"image": source}, seed=1)["image"]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, before)


def test_compact_torch_output_and_reentrant_pipeline():
    from concurrent.futures import ThreadPoolExecutor
    torch = pytest.importorskip("torch")
    source = torch.from_numpy(image())
    pipeline = A.Pipeline(chain())
    expected = pipeline({"image": source}, seed=5)["image"]
    assert isinstance(expected, torch.Tensor) and expected.dtype == source.dtype
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: pipeline({"image": source}, seed=5)["image"], range(8)))
    for result in results:
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    np.testing.assert_array_equal(source.numpy(), image())
