"""CPU ``edt`` fallback for positive full_3d label dilation."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy import ndimage
import torch

import vesuvius.ink_detection.training.dilation as dilation_module
from vesuvius.ink_detection.training.dilation import (
    CucimUnavailableError,
    apply_label_dilation,
    dilate_label_batch,
    dilate_label_batch_with_cucim,
    dilate_label_batch_with_edt,
)


def _reference_dilation(
    labels_BCZYX: np.ndarray, valid_B1ZYX: np.ndarray, distance: float
) -> np.ndarray:
    """Spell out the cuCIM path with SciPy: fill label-0 voxels within reach."""

    output = labels_BCZYX.copy()
    for batch_index in range(output.shape[0]):
        valid_ZYX = valid_B1ZYX[batch_index, 0] > 0
        for channel_index in range(output.shape[1]):
            label_ZYX = output[batch_index, channel_index]
            source_ZYX = (label_ZYX == 1) & valid_ZYX
            distances_ZYX = ndimage.distance_transform_edt(~source_ZYX)
            fill_ZYX = (label_ZYX == 0) & valid_ZYX & (distances_ZYX <= distance)
            label_ZYX[fill_ZYX] = 1
    return output


@pytest.fixture(autouse=True)
def _reset_fallback_memo(monkeypatch):
    monkeypatch.setattr(dilation_module, "_cucim_unavailable", False)
    monkeypatch.setattr(dilation_module, "_fallback_announced", False)


@pytest.mark.parametrize("distance", [1.0, 1.5, 2.0, 3.7])
@pytest.mark.parametrize("shape", [(2, 1, 9, 11, 13), (1, 2, 1, 17, 19)])
def test_edt_fallback_matches_scipy_reference(distance, shape):
    generator = np.random.default_rng(seed=int(distance * 10) + sum(shape))
    labels = (generator.random(shape) < 0.04).astype(np.float32)
    valid = (generator.random((shape[0], 1, *shape[2:])) < 0.85).astype(
        np.float32
    )

    expected = _reference_dilation(labels, valid, distance)
    output = dilate_label_batch_with_edt(
        torch.from_numpy(labels), torch.from_numpy(valid), distance
    )

    np.testing.assert_array_equal(output.numpy(), expected)
    assert output.dtype == torch.float32
    assert output.shape == labels.shape
    assert (output.numpy() >= labels).all()


def test_edt_fallback_respects_validity_mask_and_leaves_input_untouched():
    labels = torch.zeros(1, 1, 1, 1, 7)
    labels[..., 3] = 1
    valid = torch.ones(1, 1, 1, 1, 7)
    valid[..., 0] = 0
    valid[..., 5] = 0
    original = labels.clone()

    output = dilate_label_batch_with_edt(labels, valid, 2.0)

    torch.testing.assert_close(labels, original)
    torch.testing.assert_close(
        output, torch.tensor([[[[[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]]]]])
    )

    # Ink outside the validity mask must not seed growth.
    labels_outside = torch.zeros(1, 1, 1, 1, 7)
    labels_outside[..., 0] = 1
    torch.testing.assert_close(
        dilate_label_batch_with_edt(labels_outside, valid, 2.0), labels_outside
    )


def test_edt_fallback_handles_empty_and_full_labels_and_valid_rank():
    empty = torch.zeros(1, 1, 3, 4, 5)
    full = torch.ones(1, 1, 3, 4, 5)
    valid_BZYX = torch.ones(1, 3, 4, 5)

    torch.testing.assert_close(dilate_label_batch_with_edt(empty, valid_BZYX, 2.0), empty)
    torch.testing.assert_close(dilate_label_batch_with_edt(full, valid_BZYX, 2.0), full)
    assert dilate_label_batch_with_edt(empty, valid_BZYX, 0) is empty
    assert dilate_label_batch_with_edt(empty, valid_BZYX, None) is empty


@pytest.mark.parametrize("dtype", [torch.uint8, torch.bool, torch.bfloat16])
def test_edt_fallback_preserves_integer_bool_and_half_dtypes(dtype):
    labels = torch.zeros(1, 1, 1, 3, 3, dtype=dtype)
    labels[..., 1, 1] = 1
    valid = torch.ones(1, 1, 1, 3, 3, dtype=torch.uint8)

    output = dilate_label_batch_with_edt(labels, valid, 1.0)

    assert output.dtype == dtype
    expected = torch.tensor(
        [[[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]]], dtype=dtype
    )
    assert torch.equal(output, expected)


def test_dispatcher_uses_edt_on_cpu_and_warns_once(caplog):
    labels = torch.zeros(1, 1, 1, 1, 5)
    labels[..., 2] = 1
    valid = torch.ones(1, 1, 1, 1, 5)

    with caplog.at_level(logging.WARNING, logger=dilation_module.__name__):
        first = dilate_label_batch(labels, valid, 1.0)
        second = dilate_label_batch(labels, valid, 1.0)

    torch.testing.assert_close(first, torch.tensor([[[[[0.0, 1.0, 1.0, 1.0, 0.0]]]]]))
    torch.testing.assert_close(second, first)
    fallback_messages = [
        record for record in caplog.records if "edt fallback" in record.getMessage()
    ]
    assert len(fallback_messages) == 1
    assert dilate_label_batch(labels, valid, 0) is labels


def test_dispatcher_falls_back_when_cucim_raises_and_stops_retrying(monkeypatch):
    calls: list[int] = []

    def _unavailable(labels, valid, distance):
        calls.append(1)
        raise CucimUnavailableError("nope")

    monkeypatch.setattr(dilation_module, "dilate_label_batch_with_cucim", _unavailable)
    monkeypatch.setattr(
        dilation_module,
        "dilate_label_batch_with_edt",
        lambda labels, valid, distance: labels + 1,
    )

    class _CudaLike(torch.Tensor):
        pass

    labels = torch.zeros(1, 1, 1, 1, 3)
    valid = torch.ones_like(labels)
    fake_cuda = labels.as_subclass(_CudaLike)
    monkeypatch.setattr(
        _CudaLike, "device", property(lambda self: torch.device("cuda", 0))
    )

    torch.testing.assert_close(
        torch.Tensor(dilate_label_batch(fake_cuda, valid, 1.0)), labels + 1
    )
    torch.testing.assert_close(
        torch.Tensor(dilate_label_batch(fake_cuda, valid, 1.0)), labels + 1
    )
    assert len(calls) == 1
    assert dilation_module._cucim_unavailable is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("distance", [1.0, 2.5, 4.0])
def test_edt_fallback_matches_cucim_on_cuda(distance):
    pytest.importorskip("cupy")
    pytest.importorskip("cucim")
    generator = torch.Generator().manual_seed(int(distance * 100))
    labels = (torch.rand(2, 1, 17, 33, 29, generator=generator) < 0.03).float()
    valid = (torch.rand(2, 1, 17, 33, 29, generator=generator) < 0.9).float()

    with_cucim = dilate_label_batch_with_cucim(labels.cuda(), valid.cuda(), distance)
    with_edt = dilate_label_batch_with_edt(labels, valid, distance)

    assert torch.equal(with_cucim.cpu(), with_edt)


def test_cucim_path_still_refuses_cpu_tensors_with_the_typed_error():
    labels = torch.zeros(1, 1, 2, 2, 2)
    valid = torch.ones_like(labels)

    with pytest.raises(CucimUnavailableError, match="CUDA.*CuPy.*cuCIM"):
        dilate_label_batch_with_cucim(labels, valid, 1)


def test_apply_label_dilation_runs_end_to_end_on_cpu():
    labels = torch.zeros(1, 1, 1, 1, 5)
    labels[..., 2] = 1
    supervision = torch.zeros_like(labels)
    supervision[..., 0] = 1

    output = apply_label_dilation(
        {"inklabels": labels, "supervision_mask": supervision}, 1.0, 1.0
    )

    torch.testing.assert_close(
        output["inklabels"], torch.tensor([[[[[0.0, 1.0, 1.0, 1.0, 0.0]]]]])
    )
    torch.testing.assert_close(
        output["supervision_mask"],
        torch.tensor([[[[[1.0, 1.0, 1.0, 1.0, 0.0]]]]]),
    )
