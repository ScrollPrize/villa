import warnings

import pytest
import torch

from vesuvius.models.utilities import get_accelerator as ga

CUDA13_ARCHS = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]  # torch 2.14.1+cu130
CUDA126_ARCHS = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]  # torch 2.14.1+cu126


def fake_cuda(monkeypatch, capability, archs, available=True):
    if not hasattr(torch.version, "hip") or torch.version.hip is None:
        monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(archs))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "Test GPU")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.mark.parametrize(
    "capability, archs, expected",
    [
        ((6, 1), CUDA13_ARCHS, False),   # GTX 1060/1080: no kernels in CUDA 13 wheels
        ((6, 0), CUDA13_ARCHS, False),   # Tesla P100
        ((7, 0), CUDA13_ARCHS, False),   # V100 / Titan V
        ((7, 5), CUDA13_ARCHS, True),    # T4 / RTX 20xx
        ((8, 9), CUDA13_ARCHS, True),    # RTX 40xx runs the sm_86 binary (same major, higher minor)
        ((6, 1), CUDA126_ARCHS, True),
        ((7, 0), CUDA126_ARCHS, True),
        ((10, 0), CUDA126_ARCHS, False), # no sm_10x and no PTX in the list
        ((10, 0), CUDA126_ARCHS + ["compute_90"], True),  # PTX JIT for newer devices
        ((8, 0), ["compute_86"], False),  # PTX only runs on equal or newer devices
    ],
)
def test_cuda_device_supported(monkeypatch, capability, archs, expected):
    fake_cuda(monkeypatch, capability, archs)
    supported, message = ga.cuda_device_supported(0)
    assert supported is expected
    assert (message == "") is expected
    if not expected:
        assert "cu126" in message and f"{capability[0]}.{capability[1]}" in message


def test_get_accelerator_falls_back_to_cpu_on_unsupported_gpu(monkeypatch):
    fake_cuda(monkeypatch, (6, 1), CUDA13_ARCHS)
    with pytest.warns(RuntimeWarning, match="not supported by the installed torch"):
        assert ga.get_accelerator() == torch.device("cpu")


def test_get_accelerator_keeps_cuda_on_supported_gpu(monkeypatch):
    fake_cuda(monkeypatch, (8, 6), CUDA13_ARCHS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ga.get_accelerator() == torch.device("cuda")


def test_get_accelerator_without_cuda_is_unchanged(monkeypatch):
    fake_cuda(monkeypatch, (0, 0), [], available=False)
    assert ga.get_accelerator() == torch.device("cpu")


def test_explicit_cuda_device_on_unsupported_gpu_raises_with_fix(monkeypatch):
    from vesuvius.ink_detection.inference import inference_runtime as rt

    fake_cuda(monkeypatch, (6, 1), CUDA13_ARCHS)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    model = torch.nn.Conv2d(1, 1, 1)
    for kwargs in ({"gpu_ids": (), "device_type": "cuda"}, {"gpu_ids": (0,), "device_type": "auto"}):
        with pytest.raises(ValueError, match="cu126"):
            rt.prepare_model_for_inference(model, compile_model=False, compile_mode="default", **kwargs)


def test_rocm_builds_are_not_second_guessed(monkeypatch):
    # ROCm torch reports gfx targets from get_arch_list(); the sm_XY check must not reject AMD GPUs.
    fake_cuda(monkeypatch, (9, 0), ["gfx90a", "gfx942", "gfx1100"])
    monkeypatch.setattr(torch.version, "hip", "6.2.41133", raising=False)
    assert ga.cuda_device_supported(0) == (True, "")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ga.get_accelerator() == torch.device("cuda")


def test_arch_suffixes_are_understood(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    fake_cuda(monkeypatch, (9, 0), ["sm_80", "sm_90a"])
    assert ga.cuda_device_supported(0)[0] is True
    fake_cuda(monkeypatch, (10, 0), ["sm_90a", "sm_100a"])
    assert ga.cuda_device_supported(0)[0] is True
