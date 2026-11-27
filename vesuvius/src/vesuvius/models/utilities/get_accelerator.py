import platform
import warnings

import torch


def _cubin_runs_on(arch_major: int, arch_minor: int, major: int, minor: int) -> bool:
    """Whether an sm_XY / sm_XYf cubin runs on a device of compute capability major.minor."""
    # torch's own table (the one behind its "not compatible" warning) knows the
    # Jetson/integrated-GPU exceptions, e.g. desktop sm_80/sm_86 code on an 8.7 Orin.
    torch_rule = getattr(torch.cuda, "_code_compatible_with_device", None)
    if torch_rule is not None and getattr(torch.version, "cuda", None):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return bool(torch_rule(10 * major + minor, 10 * arch_major + arch_minor))
        except Exception:
            pass
    return arch_major == major and arch_minor <= minor


def cuda_device_supported(device_index: int = 0) -> tuple[bool, str]:
    """Whether the installed torch build has kernels for a visible CUDA device.

    ``torch.cuda.is_available()`` is True on any NVIDIA GPU the driver sees, even
    when the wheel was built without kernels for it. CUDA 13 wheels (PyPI's
    Linux default and the cu130 Windows index this project pins) dropped
    compute capability below 7.5: GTX 9xx/10xx, Titan Xp, P100, V100, Titan V.
    On those GPUs the first convolution fails with "no kernel image is
    available" or "unable to find an engine". This checks the device against
    ``torch.cuda.get_arch_list()`` using NVIDIA's compatibility rules: an ``sm_XY``
    (or family-specific ``sm_XYf``) binary runs on the same major version with an
    equal or higher minor version, minus the Jetson/integrated-GPU exceptions in
    torch's own table; ``compute_XY`` PTX can be JIT-compiled for any equal or
    newer device; architecture-specific ``sm_XYa``/``compute_XYa`` code runs only on
    exactly that compute capability.

    Returns ``(supported, message)``; the message explains the fix when not.
    """
    if getattr(torch.version, "hip", None):
        # ROCm builds report gfx targets (e.g. gfx90a), not sm_XY; this check is CUDA-only.
        return True, ""
    major, minor = torch.cuda.get_device_capability(device_index)
    arch_list = torch.cuda.get_arch_list()
    for arch in arch_list:
        kind, _, number = arch.partition("_")
        suffix = number[-1] if number[-1:].isalpha() else ""
        number = number[:-1] if suffix else number
        if kind not in ("sm", "compute") or not number.isdigit() or len(number) < 2:
            continue
        arch_major, arch_minor = int(number[:-1]), int(number[-1])
        if suffix == "a":
            # Architecture-specific (sm_90a, compute_100a): that exact capability only,
            # no forward compatibility for cubin or PTX.
            if (arch_major, arch_minor) == (major, minor):
                return True, ""
        elif suffix == "f" or kind == "sm":
            # Family-specific (sm_100f, compute_100f) and plain sm_XY cubins: same major,
            # equal or higher minor.
            if _cubin_runs_on(arch_major, arch_minor, major, minor):
                return True, ""
        elif kind == "compute" and not suffix:
            # Plain compute_XY PTX can be JIT-compiled for any equal or newer device.
            if (arch_major, arch_minor) <= (major, minor):
                return True, ""
    name = torch.cuda.get_device_name(device_index)
    message = (
        f"{name} (compute capability {major}.{minor}) is not supported by the "
        f"installed torch {torch.__version__}, which was built for "
        f"{', '.join(arch_list) or 'no CUDA architectures'}. {_recovery_hint(major, minor)}"
    )
    return False, message


# CUDA 12.6 wheels, the last with kernels below compute capability 7.5, exist for torch
# 2.6 through 2.14; 2.15 dropped them (https://github.com/pytorch/pytorch/blob/main/RELEASE.md).
# This is a fixed fact about published wheels; it only needs revisiting if pyproject.toml's torch
# floor rises above 2.14, when these GPUs would have no supported GPU build left.
_CU126_TORCH_RELEASES = ((2, 6), (2, 14))
_LAST_CU126_TORCH = "2.14.1"
# Jetson/integrated GPUs: desktop wheels exclude them (see torch.cuda.DEVICE_REQUIREMENT).
_INTEGRATED_GPUS = ((5, 3), (6, 2), (7, 2), (8, 7))


def _recovery_hint(major: int, minor: int) -> str:
    """How to get a torch build with kernels for this GPU, keeping the other installed packages."""
    capability = (major, minor)
    integrated = capability in _INTEGRATED_GPUS
    x86_64 = platform.machine().lower() in ("x86_64", "amd64")
    if (5, 0) <= capability < (7, 5) and x86_64 and not integrated:
        version = torch.__version__.split("+")[0]
        try:
            release = tuple(int(part) for part in version.split(".")[:2])
        except ValueError:
            release = None
        first, last = _CU126_TORCH_RELEASES
        if release is None or not first <= release <= last or not version.replace(".", "").isdigit():
            version = _LAST_CU126_TORCH
        return (
            "CUDA 13 builds of torch dropped GPUs older than compute capability 7.5; the CUDA 12.6 "
            f"builds of torch {last[0]}.{last[1]} and earlier still support them. From villa/vesuvius, "
            f"run `uv pip install torch=={version}+cu126 torchvision --index-url "
            "https://download.pytorch.org/whl/cu126 --reinstall-package torch "
            "--reinstall-package torchvision` (already-installed packages such as NumPy are kept), "
            "then start commands with `uv run --no-sync`: a plain `uv run` re-syncs uv.lock and puts "
            "the CUDA 13 build back."
        )
    cuda_major = (getattr(torch.version, "cuda", None) or "").split(".")[0]
    if capability >= (7, 5) and not integrated and cuda_major.isdigit() and int(cuda_major) < 13:
        return (
            "This GPU needs a newer CUDA build of torch. From villa/vesuvius, restore the locked "
            "CUDA 13 build with `uv sync --inexact --extra models --reinstall-package torch "
            "--reinstall-package torchvision` (--inexact keeps packages from other extras)."
        )
    return (
        f"Install a torch build that includes compute capability {major}.{minor} "
        "(https://pytorch.org/get-started/locally/; Jetson boards need NVIDIA's Jetson builds), "
        "or run on the CPU with `--device cpu`."
    )


def get_accelerator():

    if torch.cuda.is_available():
        supported, message = cuda_device_supported(torch.cuda.current_device())
        if supported:
            device = torch.device('cuda')
            print("Using CUDA device")
            return device
        warnings.warn(f"{message} Falling back to another device.", RuntimeWarning, stacklevel=2)
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU device")

    return device
