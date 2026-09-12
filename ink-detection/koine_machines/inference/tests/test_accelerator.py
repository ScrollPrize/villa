"""Tests for inference accelerator selection."""

import pytest
import torch

from koine_machines.common.accelerator import (
    AUTOCAST_DEVICE_TYPES,
    DEVICE_CHOICES,
    autocast_supported,
    available_device_types,
    log_device,
    mps_available,
    select_device,
)


HAVE_CUDA = torch.cuda.is_available()
HAVE_MPS = mps_available()


def test_auto_prefers_cuda_then_mps_then_cpu():
    expected = "cuda" if HAVE_CUDA else ("mps" if HAVE_MPS else "cpu")
    assert select_device("auto").type == expected


def test_cpu_is_always_reachable():
    assert select_device("cpu").type == "cpu"


def test_available_device_types_always_ends_with_cpu():
    types = available_device_types()
    assert types[-1] == "cpu"
    assert ("cuda" in types) == HAVE_CUDA
    assert ("mps" in types) == HAVE_MPS


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unsupported device"):
        select_device("tpu")


def test_device_choices_match_the_resolver():
    assert DEVICE_CHOICES == ("auto", "cuda", "mps", "cpu")
    for choice in DEVICE_CHOICES:
        if choice == "auto":
            continue
        try:
            assert select_device(choice).type == choice
        except ValueError as exc:
            assert "not available" in str(exc)


def test_autocast_set_excludes_cpu():
    assert AUTOCAST_DEVICE_TYPES == {"cuda", "mps"}
    assert not autocast_supported(torch.device("cpu"))
    assert autocast_supported(torch.device("cuda"))
    assert autocast_supported(torch.device("mps"))


def test_log_device_never_raises():
    log_device(select_device("auto"))
    log_device(torch.device("cpu"))


@pytest.mark.skipif(HAVE_CUDA, reason="needs a machine without CUDA")
def test_gpu_ids_without_cuda_raise():
    with pytest.raises(ValueError, match="CUDA is not available"):
        select_device("auto", gpu_ids=(0,))


def test_gpu_ids_with_a_non_cuda_device_raise():
    with pytest.raises(ValueError, match="CUDA ordinals"):
        select_device("mps", gpu_ids=(0,))
    with pytest.raises(ValueError, match="CUDA ordinals"):
        select_device("cpu", gpu_ids=(0,))


@pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA")
def test_out_of_range_gpu_id_raises():
    too_high = torch.cuda.device_count()
    with pytest.raises(ValueError, match="unavailable"):
        select_device("auto", gpu_ids=(too_high,))


def _fake_cuda(monkeypatch, count):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


def test_auto_prefers_cuda_over_mps(monkeypatch):
    _fake_cuda(monkeypatch, 2)
    assert select_device("auto").type == "cuda"


def test_gpu_ids_select_the_first_ordinal(monkeypatch):
    _fake_cuda(monkeypatch, 4)
    device = select_device("auto", gpu_ids=(2, 3))
    assert (device.type, device.index) == ("cuda", 2)


def test_out_of_range_ordinal_reports_the_visible_count(monkeypatch):
    _fake_cuda(monkeypatch, 2)
    with pytest.raises(ValueError) as excinfo:
        select_device("auto", gpu_ids=(0, 7))
    message = str(excinfo.value)
    assert "[7]" in message and "visible device count is 2" in message


def test_explicit_cpu_still_wins_over_available_cuda(monkeypatch):
    _fake_cuda(monkeypatch, 1)
    assert select_device("cpu").type == "cpu"


def test_missing_cuda_message_is_unchanged(monkeypatch):
    _fake_cuda(monkeypatch, 0)
    with pytest.raises(ValueError) as excinfo:
        select_device("auto", gpu_ids=(0,))
    assert str(excinfo.value) == (
        "--gpus was provided, but CUDA is not available in this environment."
    )
