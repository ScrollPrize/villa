"""GPU augmentation source regressions plus queued deferred integration."""
from dataclasses import replace
import math
import os

import numpy as np
import pytest
import torch

from tests.neural_tracing.test_fiber_trace_3d import _loader_with_mapping
from tests.neural_tracing.test_sample_pipeline import _await, _real_paris4_loader
from tests.neural_tracing.test_cuda_batch_pipeline import _same_tensors
from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    _Augment3DParams, _BlurKernels3D, _DirectGaussianKernel3D,
    _apply_blur_kernels_3d, _build_blur_kernels_3d,
    _directional_gaussian_kernel_3d, _directional_kernel_groups, _gaussian_kernel_1d,
    apply_complete_value_augmentation_3d, apply_value_augmentation_batch_3d,
    apply_value_augmentation_suffix_batch_3d, prepare_fiber_trace_3d_cuda_suffix,
    collate_fiber_trace_3d_samples,
    apply_value_augmentation_3d,
)
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import (
    FiberTrace3DSampleQueue, SampleQueueSource, apply_deferred_batch_augmentation,
)
from vesuvius.neural_tracing.fiber_trace_3d.queued_batches import QueuedBatchView
from vesuvius.neural_tracing.fiber_trace_3d.cuda_batch_pipeline import FiberTrace3DCudaBatchPipeline
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets


def _value_augmentation_params(
    stream_index: int,
    *,
    blur_sigma: float,
    axis_zyx: np.ndarray | None,
) -> _Augment3DParams:
    vector = np.zeros((3,), dtype=np.float32)
    matrix = np.eye(3, dtype=np.float32)
    return _Augment3DParams(
        stream_index=stream_index,
        cp_local_zyx=vector,
        cp_volume_zyx=vector,
        source_to_output_zyx=matrix,
        output_to_source_zyx=matrix,
        brightness=0.0,
        contrast=1.0,
        gamma=1.0,
        noise_std=0.0,
        blur_sigma=blur_sigma,
        smooth=None,
        anisotropic_blur_sigma_along=1.1 if axis_zyx is not None else 0.0,
        anisotropic_blur_sigma_across=0.35 if axis_zyx is not None else 0.0,
        anisotropic_blur_axis_zyx=axis_zyx,
    )


def test_direct_gaussian_kernel_is_normalized_and_oriented() -> None:
    params = _value_augmentation_params(
        0,
        blur_sigma=0.0,
        axis_zyx=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )
    kernel = _directional_gaussian_kernel_3d(params)
    assert kernel is not None
    dense = kernel.dense_zyx
    assert dense.shape == (9, 5, 5)
    assert dense.dtype == np.float32
    assert np.isclose(float(dense.sum()), 1.0, atol=1.0e-6)
    assert bool((dense >= 0.0).all())
    center = tuple(value // 2 for value in dense.shape)
    assert dense[center[0] + 1, center[1], center[2]] > dense[
        center[0], center[1] + 1, center[2]
    ]
    expected_along_ratio = math.exp(-0.5 / (1.1 * 1.1))
    assert np.isclose(
        float(dense[center[0] + 1, center[1], center[2]] / dense[center]),
        expected_along_ratio,
        rtol=2.0e-6,
    )


def test_direct_gaussian_kernel_oblique_support_and_identity() -> None:
    identity = _value_augmentation_params(0, blur_sigma=0.0, axis_zyx=None)
    assert _directional_gaussian_kernel_3d(identity) is None
    oblique = _value_augmentation_params(
        1,
        blur_sigma=0.5,
        axis_zyx=np.asarray([0.31, 0.83, -0.46], dtype=np.float32),
    )
    kernel = _directional_gaussian_kernel_3d(oblique)
    assert kernel is not None
    dense = kernel.dense_zyx
    assert all(value % 2 == 1 for value in dense.shape)
    assert np.isclose(float(dense.sum()), 1.0, atol=1.0e-6)
    center = tuple(value // 2 for value in dense.shape)
    assert dense[center] == dense.max()


def test_axis_directional_kernel_and_replicate_border_match_manual_formula() -> None:
    params = replace(
        _value_augmentation_params(
            0,
            blur_sigma=0.0,
            axis_zyx=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        ),
        anisotropic_blur_sigma_along=1.1,
        anisotropic_blur_sigma_across=0.0,
    )
    kernel = _directional_gaussian_kernel_3d(params)
    assert kernel is not None
    assert kernel.dense_zyx.shape == (9, 1, 1)
    source = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(1, 1, 3, 2, 2)
    actual = _apply_blur_kernels_3d(
        source,
        (_BlurKernels3D(isotropic_1d=None, directional_zyx=kernel),),
    )
    expected = torch.zeros_like(source)
    radius = kernel.dense_zyx.shape[0] // 2
    for z in range(3):
        for offset, weight in enumerate(kernel.dense_zyx[:, 0, 0]):
            source_z = min(max(z + offset - radius, 0), 2)
            expected[0, 0, z] += source[0, 0, source_z] * float(weight)
    assert torch.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_isotropic_gaussian_kernel_uses_independent_radius() -> None:
    assert _gaussian_kernel_1d(0.0) is None
    small = _gaussian_kernel_1d(0.2)
    large = _gaussian_kernel_1d(0.5)
    assert small is not None and small.shape == (3,)
    assert large is not None and large.shape == (5,)
    assert np.isclose(float(small.sum()), 1.0, atol=1.0e-7)
    assert np.isclose(float(large.sum()), 1.0, atol=1.0e-7)


def test_directional_kernel_grouping_preserves_rectangular_shapes() -> None:
    def kernel(shape: tuple[int, int, int]) -> object:
        dense = np.ones(shape, dtype=np.float32)
        dense /= float(dense.size)
        return _DirectGaussianKernel3D(dense)

    kernels = (
        kernel((13, 3, 3)),
        kernel((3, 13, 3)),
        kernel((13, 3, 3)),
        None,
    )
    assert _directional_kernel_groups(kernels, max_padding_ratio=1.25) == [[0, 2], [1]]


@pytest.mark.parametrize("shape", ((9, 11, 13), (1, 7, 9)))
def test_direct_gaussian_batch_matches_individual_convolutions(
    shape: tuple[int, int, int],
) -> None:
    generator = torch.Generator().manual_seed(83)
    volumes = torch.randn((3, 1, *shape), generator=generator)
    params = (
        _value_augmentation_params(0, blur_sigma=0.0, axis_zyx=None),
        _value_augmentation_params(
            1,
            blur_sigma=0.5,
            axis_zyx=np.asarray([0.31, 0.83, -0.46], dtype=np.float32),
        ),
        _value_augmentation_params(2, blur_sigma=0.9, axis_zyx=None),
    )
    kernels = tuple(_build_blur_kernels_3d(item) for item in params)
    expected = torch.cat(
        [
            _apply_blur_kernels_3d(volumes[index : index + 1], (kernels[index],))
            for index in range(3)
        ],
        dim=0,
    )
    actual = apply_value_augmentation_suffix_batch_3d(volumes, kernels)
    assert torch.allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)
    assert torch.equal(actual[0], volumes[0])


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires a manager-reserved CUDA process")
def test_direct_gaussian_cuda_is_repeatable_and_matches_cpu() -> None:
    source = torch.randn((2, 1, 9, 11, 13), generator=torch.Generator().manual_seed(89))
    params = (
        _value_augmentation_params(0, blur_sigma=0.5, axis_zyx=None),
        _value_augmentation_params(
            1,
            blur_sigma=0.9,
            axis_zyx=np.asarray([0.31, 0.83, -0.46], dtype=np.float32),
        ),
    )
    kernels = tuple(_build_blur_kernels_3d(item) for item in params)
    expected = apply_value_augmentation_suffix_batch_3d(source, kernels)
    source_cuda = source.cuda()
    first = apply_value_augmentation_suffix_batch_3d(source_cuda, kernels)
    second = apply_value_augmentation_suffix_batch_3d(source_cuda, kernels)
    torch.cuda.synchronize()
    assert torch.equal(first, second)
    assert torch.allclose(first.cpu(), expected, rtol=2.0e-5, atol=2.0e-5)


def test_deferred_suffix_preserves_full_chain_and_kernel() -> None:
    loader = _loader_with_mapping(
        {
            "patch_shape_zyx": [9, 11, 13],
            "image_normalization": "zscore",
            "augment_brightness": 0.2,
            "augment_contrast_min": 0.7,
            "augment_contrast_max": 1.3,
            "augment_gamma_min": 0.8,
            "augment_gamma_max": 1.2,
            "augment_noise_std": 0.1,
            "augment_blur_sigma": 0.5,
            "augment_anisotropic_blur_probability": 1.0,
            "augment_anisotropic_blur_sigma_along": 1.1,
            "augment_anisotropic_blur_sigma_across": 0.35,
            "value_augmentation_backend": "torch",
        }
    )
    raw = loader.load_raw_sample(7)
    expected = replace(raw.sample, volume=apply_complete_value_augmentation_3d(
        loader.config, raw.sample.volume[0], raw.augmentation).unsqueeze(0))
    deferred = prepare_fiber_trace_3d_cuda_suffix(loader.config, raw)
    assert deferred.augmentation is raw.augmentation
    assert deferred.blur_kernels is not None
    assert not torch.equal(deferred.sample.volume, expected.volume)
    actual = apply_value_augmentation_suffix_batch_3d(
        deferred.sample.volume.unsqueeze(0),
        (deferred.blur_kernels,),
    )[0]
    assert torch.allclose(actual, expected.volume, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires a manager-reserved CUDA process")
def test_batched_cuda_value_augmentation_matches_cpu_without_noise() -> None:
    loader = _loader_with_mapping(
        {
            "patch_shape_zyx": [9, 11, 13],
            "image_normalization": "zscore",
            "augment_blur_sigma": 0.5,
            "value_augmentation_backend": "torch",
        }
    )
    base = _value_augmentation_params(11, blur_sigma=0.5, axis_zyx=None)
    params = (replace(base, brightness=0.13, contrast=0.81, gamma=1.17),)
    source = torch.randn((1, 1, 9, 11, 13), generator=torch.Generator().manual_seed(101))
    kernels = tuple(_build_blur_kernels_3d(item) for item in params)
    expected = apply_complete_value_augmentation_3d(loader.config, source[0, 0], params[0])
    actual = apply_value_augmentation_batch_3d(
        loader.config,
        source.cuda(),
        params,
        kernels,
    )[0, 0]
    torch.cuda.synchronize()
    assert torch.allclose(actual.cpu(), expected, rtol=3.0e-5, atol=3.0e-5)


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires a manager-reserved CUDA process")
def test_batched_cuda_value_augmentation_noise_is_partition_invariant() -> None:
    loader = _loader_with_mapping(
        {
            "patch_shape_zyx": [9, 11, 13],
            "image_normalization": "none",
            "augment_noise_std": 0.1,
            "value_augmentation_backend": "torch",
        }
    )
    params = tuple(
        replace(
            _value_augmentation_params(index + 20, blur_sigma=0.0, axis_zyx=None),
            noise_std=0.1,
        )
        for index in range(3)
    )
    kernels = tuple(_build_blur_kernels_3d(item) for item in params)
    source = torch.randn((3, 1, 9, 11, 13), generator=torch.Generator().manual_seed(103)).cuda()
    together = apply_value_augmentation_batch_3d(loader.config, source, params, kernels)
    permutation = (2, 0, 1)
    reordered = apply_value_augmentation_batch_3d(
        loader.config,
        source[list(permutation)],
        tuple(params[index] for index in permutation),
        tuple(kernels[index] for index in permutation),
    )
    restored = reordered[[1, 2, 0]]
    separate = torch.cat(
        [
            apply_value_augmentation_batch_3d(
                loader.config,
                source[index : index + 1],
                (params[index],),
                (kernels[index],),
            )
            for index in range(3)
        ],
        dim=0,
    )
    torch.cuda.synchronize()
    assert torch.equal(together, restored)
    assert torch.equal(together, separate)


@pytest.mark.parametrize("mode", ["blur", "all"])
def test_shared_deferred_queue_round_trip_keeps_parameters_and_masks(mode):
    loader = _loader_with_mapping({"batch_size": 2, "patch_shape_zyx": [8, 8, 8],
        "augment_blur_sigma": .4, "augment_noise_std": .03,
        "augment_anisotropic_blur_probability": 1.,
        "augment_anisotropic_blur_sigma_along": .7,
        "augment_anisotropic_blur_sigma_across": .2})
    owner = FiberTrace3DSampleQueue(sources={
        "train": SampleQueueSource(loader.config, defer_cuda_value_augmentation=mode)},
        workers=2, slot_count=3, initial_in_flight=2)
    try:
        for indices in ((3, 4), (5,), (8, 9)):
            ticket = owner.submit("train", indices)
            batch = _await(owner, lambda: owner.batch(ticket))
            expected_dtype = torch.uint8 if mode == "all" else torch.float32
            assert batch.volume.dtype == expected_dtype
            assert batch.deferred_augmentation_params is not None
            assert batch.deferred_blur_kernels is not None
            expected_samples = []
            for index in indices:
                raw = loader.load_raw_sample(index)
                if mode == "all":
                    raw = replace(raw, sample=replace(raw.sample,
                        volume=raw.sample.volume.to(torch.uint8)))
                prepared = prepare_fiber_trace_3d_cuda_suffix(loader.config, raw, mode=mode)
                assert prepared.blur_kernels is not None
                if mode == "blur":
                    volume = apply_value_augmentation_suffix_batch_3d(
                        prepared.sample.volume.float().unsqueeze(0),
                        (prepared.blur_kernels,))[0]
                else:
                    volume = apply_value_augmentation_batch_3d(
                        loader.config, prepared.sample.volume.unsqueeze(0),
                        (prepared.augmentation,), (prepared.blur_kernels,))[0]
                expected_samples.append(replace(prepared.sample, volume=volume))
            actual = apply_deferred_batch_augmentation(batch, loader.config, mode=mode)
            expected = collate_fiber_trace_3d_samples(expected_samples)
            torch.testing.assert_close(actual.volume, expected.volume, rtol=3e-5, atol=3e-5)
            assert torch.equal(actual.valid_mask, expected.valid_mask)
            assert actual.stream_indices.tolist() == list(indices)
            assert actual.deferred_augmentation_params is None
            assert actual.deferred_blur_kernels is None
            owner.release(ticket)
    finally:
        owner.close()


def test_complete_patch_recipe_does_not_replace_legacy_masked_statistics():
    loader = _loader_with_mapping({"augment_enabled": False, "image_normalization": "zscore",
                                  "value_augmentation_backend": "torch"})
    params = _value_augmentation_params(0, blur_sigma=0., axis_zyx=None)
    source = torch.arange(8., dtype=torch.float32).reshape(2, 2, 2)
    mask = source < 4.
    legacy = apply_value_augmentation_3d(loader.config, source, mask, params)
    complete = apply_complete_value_augmentation_3d(loader.config, source, params)
    assert abs(float(legacy[mask].mean())) < 1e-6
    assert abs(float(complete.mean())) < 1e-6
    assert not torch.allclose(legacy, complete)


@pytest.mark.parametrize("mode", ["blur", "all"])
def test_cuda_real_paris4_deferred_train_test_train_matches_direct(tmp_path, mode):
    if os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1":
        pytest.skip("requires a manager-reserved CUDA process")
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "a manager GPU lease is required"
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    original = _real_paris4_loader(tmp_path)
    loader = type(original)(replace(original.config,
        augment_anisotropic_blur_probability=1.,
        augment_anisotropic_blur_sigma_along=1.1,
        augment_anisotropic_blur_sigma_across=.35))
    test_loader = type(loader)(replace(loader.config, augment_enabled=False))
    owner = FiberTrace3DSampleQueue(sources={
        "train": SampleQueueSource(loader.config, sample_mode="flat",
            defer_cuda_value_augmentation=mode),
        "test": SampleQueueSource(test_loader.config, sample_mode="flat",
            defer_cuda_value_augmentation=mode)},
        workers=2, slot_count=3, initial_in_flight=2)
    pids = [process.pid for process in owner.pipeline.processes]
    try:
        for phase in ("train", "test", "train"):
            serial = loader if phase == "train" else test_loader
            view = QueuedBatchView(owner, phase, ((0, 1), (2,)))
            pipeline = FiberTrace3DCudaBatchPipeline(view, config=serial.config,
                device=device, output_stride=2, depth=2, value_augmentation_suffix=mode)
            try:
                for indices in ((0, 1), (2,)):
                    samples, params, kernels = [], [], []
                    for index in indices:
                        raw = serial.load_raw_sample(index, sample_mode="flat")
                        if mode == "all":
                            raw = replace(raw, sample=replace(raw.sample,
                                volume=raw.sample.volume.to(torch.uint8)))
                        prepared = prepare_fiber_trace_3d_cuda_suffix(serial.config, raw, mode=mode)
                        samples.append(prepared.sample)
                        params.append(prepared.augmentation)
                        kernels.append(prepared.blur_kernels)
                    staged = collate_fiber_trace_3d_samples(samples,
                        deferred_augmentation_params=tuple(params),
                        deferred_blur_kernels=tuple(kernels)).to(device)
                    # Same exact GPU algorithm/settings, but on the default
                    # stream without the asynchronous wrapper or shared slots.
                    with torch.autocast(device_type="cuda", enabled=False):
                        expected = apply_deferred_batch_augmentation(staged, serial.config, mode=mode)
                    expected = materialize_targets(expected, serial.config, output_stride=2)
                    actual = next(pipeline)
                    _same_tensors(actual, expected)
                    assert actual.volume.dtype == torch.float32
                    assert actual.profile_timings_ms["cuda_pipeline_value_augmentation_ms"] >= 0.
                with pytest.raises(StopIteration):
                    next(pipeline)
            finally:
                pipeline.close()
            assert not owner.pipeline.closed
            assert [process.pid for process in owner.pipeline.processes] == pids
    finally:
        owner.close()
    print(f"Paris4 deferred {mode}: exact direct/async images, masks and targets; persistent train/test/train workers")
