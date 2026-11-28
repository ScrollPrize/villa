import platform
import warnings

import pytest
import torch

from vesuvius.models.utilities import get_accelerator as ga

CUDA13_ARCHS = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]  # torch 2.12-2.15 CUDA 13 wheels
CUDA126_ARCHS = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]  # torch <=2.14 cu126


def fake_backend(
    monkeypatch,
    backend="nvidia",
    capability=(0, 0),
    archs=(),
    *,
    cuda_version="13.0",
    torch_version="2.12.1+cu130",
    machine="AMD64",
    torch_rule=None,
):
    """Simulate one backend completely, whatever torch build runs the tests.

    torch's private compatibility table differs between releases, so it is removed
    unless a test passes ``torch_rule`` explicitly.
    """
    nvidia, rocm = backend == "nvidia", backend == "rocm"
    monkeypatch.setattr(torch.version, "hip", "6.2.41133" if rocm else None, raising=False)
    monkeypatch.setattr(torch.version, "cuda", cuda_version if nvidia else None, raising=False)
    monkeypatch.setattr(torch, "__version__", torch_version)
    if torch_rule is None:
        monkeypatch.delattr(torch.cuda, "_code_compatible_with_device", raising=False)
    else:
        monkeypatch.setattr(torch.cuda, "_code_compatible_with_device", torch_rule, raising=False)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: nvidia or rocm)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1 if nvidia or rocm else 0)
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
        ((12, 0), CUDA126_ARCHS, False), # Blackwell RTX 50xx: no sm_12x and no PTX in cu126
        ((10, 0), CUDA126_ARCHS + ["compute_90"], True),  # PTX JIT for newer devices
        ((8, 0), ["compute_86"], False),  # PTX only runs on equal or newer devices
    ],
)
def test_cuda_device_supported(monkeypatch, capability, archs, expected):
    fake_backend(monkeypatch, "nvidia", capability, archs)
    supported, message = ga.cuda_device_supported(0)
    assert supported is expected
    assert (message == "") is expected
    if not expected:
        assert f"{capability[0]}.{capability[1]}" in message


@pytest.mark.parametrize(
    "torch_version, target",
    [
        ("2.12.1+cu130", "2.12.1"),  # villa's lock: same release, CUDA 12.6 build
        ("2.14.1+cu130", "2.14.1"),
        ("2.15.0+cu132", "2.14.1"),  # 2.15 has no CUDA 12.6 wheels: newest release that does
        ("2.16.0.dev20261201+cu132", "2.14.1"),
    ],
)
def test_pascal_volta_hint_names_an_available_cu126_build(monkeypatch, torch_version, target):
    fake_backend(monkeypatch, "nvidia", (6, 1), CUDA13_ARCHS, torch_version=torch_version)
    message = ga.cuda_device_supported(0)[1]
    assert f"torch=={target}+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126" in message
    # Reinstalling only torch and torchvision keeps the locked NumPy/Numba pair intact.
    assert "--reinstall-package torch --reinstall-package torchvision" in message
    assert "--reinstall " not in message and "numpy" not in message
    # On Linux the CUDA 12 libraries overwrite the locked CUDA 13 files, so not into .venv.
    assert "uv pip install --python .venv-cu126" in message
    # A plain `uv run` would re-sync the lock and restore the CUDA 13 build.
    assert "uv run --no-sync" in message


def test_newer_gpu_on_cuda12_build_is_sent_back_to_the_locked_build(monkeypatch):
    fake_backend(monkeypatch, "nvidia", (12, 0), CUDA126_ARCHS, cuda_version="12.6", torch_version="2.14.1+cu126")
    message = ga.cuda_device_supported(0)[1]
    # Only a full reinstall restores CUDA 13 files that CUDA 12 wheels overwrote in place.
    assert "uv sync --reinstall --extra models" in message
    assert "whl/cu126" not in message


@pytest.mark.parametrize(
    "capability, archs, machine",
    [
        ((3, 7), CUDA126_ARCHS, "AMD64"),   # Kepler: no current wheel supports it
        ((6, 1), CUDA13_ARCHS, "aarch64"),  # Arm cu126 wheels only carry sm_80/sm_90
        ((6, 2), CUDA13_ARCHS, "x86_64"),   # Jetson TX2
        ((5, 3), CUDA13_ARCHS, "aarch64"),  # Jetson Nano
        ((8, 7), CUDA126_ARCHS, "aarch64"),  # Jetson Orin: torch's table excludes 8.7 from sm_80/sm_86
    ],
)
def test_gpus_without_a_standard_wheel_get_a_generic_hint(monkeypatch, capability, archs, machine):
    def desktop_table(device_cc, code_cc):  # torch's rule: same major, minus integrated GPUs
        return device_cc not in (53, 62, 72, 87) and device_cc // 10 == code_cc // 10 and device_cc >= code_cc

    fake_backend(monkeypatch, "nvidia", capability, archs, cuda_version="12.6", machine=machine, torch_rule=desktop_table)
    message = ga.cuda_device_supported(0)[1]
    assert message and "pytorch.org/get-started/locally" in message and "--device cpu" in message
    assert "+cu126" not in message and "uv sync" not in message


def test_get_accelerator_falls_back_to_cpu_on_unsupported_gpu(monkeypatch):
    fake_backend(monkeypatch, "nvidia", (6, 1), CUDA13_ARCHS)
    with pytest.warns(RuntimeWarning, match="not supported by the installed torch"):
        assert ga.get_accelerator() == torch.device("cpu")


def test_get_accelerator_keeps_cuda_on_supported_gpu(monkeypatch):
    fake_backend(monkeypatch, "nvidia", (8, 6), CUDA13_ARCHS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ga.get_accelerator() == torch.device("cuda")


def test_get_accelerator_without_cuda_is_unchanged(monkeypatch):
    fake_backend(monkeypatch, "cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ga.get_accelerator() == torch.device("cpu")


def test_explicit_cuda_device_on_unsupported_gpu_raises_with_fix(monkeypatch):
    from vesuvius.ink_detection.inference import inference_runtime as rt

    fake_backend(monkeypatch, "nvidia", (6, 1), CUDA13_ARCHS)
    model = torch.nn.Conv2d(1, 1, 1)
    for kwargs in ({"gpu_ids": (), "device_type": "cuda"}, {"gpu_ids": (0,), "device_type": "auto"}):
        with pytest.raises(ValueError, match=r"torch==2\.12\.1\+cu126"):
            rt.prepare_model_for_inference(model, compile_model=False, compile_mode="default", **kwargs)


def test_rocm_builds_are_not_second_guessed(monkeypatch):
    # ROCm torch reports gfx targets from get_arch_list(); the sm_XY check must not reject AMD GPUs.
    fake_backend(monkeypatch, "rocm", (9, 0), ["gfx90a", "gfx942", "gfx1100"])
    assert ga.cuda_device_supported(0) == (True, "")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ga.get_accelerator() == torch.device("cuda")


@pytest.mark.parametrize(
    "capability, archs, expected",
    [
        ((9, 0), ["sm_80", "sm_90a"], True),     # sm_90a runs on exactly 9.0
        ((10, 0), ["sm_90a", "sm_100a"], True),  # sm_100a runs on exactly 10.0
        ((10, 3), ["sm_100a"], False),           # 'a' cubins are not forward compatible, even within 10.x
        ((10, 0), ["compute_90a"], False),       # 'a' PTX is not forward compatible either
        ((9, 0), ["compute_90a"], True),
        ((10, 3), ["sm_100f"], True),            # 'f' (family) cubins run on the same major, equal or higher minor
        ((12, 1), ["compute_120f"], True),
        ((12, 0), ["sm_100f"], False),           # a family never crosses a major version
        ((10, 0), ["sm_103f"], False),           # nor runs on a lower minor
    ],
)
def test_arch_suffixes_are_understood(monkeypatch, capability, archs, expected):
    fake_backend(monkeypatch, "nvidia", capability, archs)
    assert ga.cuda_device_supported(0)[0] is expected


def test_torchs_table_decides_cubin_compatibility_when_present(monkeypatch):
    # Desktop sm_80/sm_86 cubins don't run on an 8.7 Jetson Orin before CUDA 13.2; torch's table
    # knows this, the plain same-major rule does not.
    calls = []

    def jetson_table(device_cc, code_cc):
        calls.append((device_cc, code_cc))
        return device_cc != 87 and device_cc // 10 == code_cc // 10 and device_cc >= code_cc

    fake_backend(monkeypatch, "nvidia", (8, 7), ["sm_80", "sm_86"], torch_rule=jetson_table)
    assert ga.cuda_device_supported(0)[0] is False
    assert calls == [(87, 80), (87, 86)]

    fake_backend(monkeypatch, "nvidia", (8, 7), ["sm_80", "sm_86"], torch_rule=lambda device_cc, code_cc: True)
    assert ga.cuda_device_supported(0)[0] is True


def test_agrees_with_the_installed_torch_table(monkeypatch):
    real_rule = getattr(torch.cuda, "_code_compatible_with_device", None)
    if real_rule is None:
        pytest.skip("this torch has no compatibility table")
    fake_backend(monkeypatch, "nvidia", (8, 7), ["sm_80", "sm_86"], torch_rule=real_rule)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = any(real_rule(87, code) for code in (80, 86))
    assert ga.cuda_device_supported(0)[0] is expected


def test_without_torchs_table_the_documented_rule_applies(monkeypatch):
    fake_backend(monkeypatch, "nvidia", (6, 1), ["sm_60"])
    assert ga.cuda_device_supported(0)[0] is True
    fake_backend(monkeypatch, "nvidia", (6, 1), ["sm_75", "sm_80"])
    assert ga.cuda_device_supported(0)[0] is False
