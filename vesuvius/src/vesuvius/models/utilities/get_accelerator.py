import warnings

import torch


def cuda_device_supported(device_index: int = 0) -> tuple[bool, str]:
    """Whether the installed torch build has kernels for a visible CUDA device.

    ``torch.cuda.is_available()`` is True on any NVIDIA GPU the driver sees, even
    when the wheel was built without kernels for it. CUDA 13 wheels (PyPI's
    Linux default and the cu130 Windows index this project pins) dropped
    compute capability below 7.5: GTX 9xx/10xx, Titan Xp, P100, V100, Titan V.
    On those GPUs the first convolution fails with "no kernel image is
    available" or "unable to find an engine". This checks the device against
    ``torch.cuda.get_arch_list()``: an ``sm_XY`` binary runs on the same major
    version with an equal or higher minor version, and ``compute_XY`` PTX can be
    JIT-compiled for any equal or newer device.

    Returns ``(supported, message)``; the message explains the fix when not.
    """
    major, minor = torch.cuda.get_device_capability(device_index)
    arch_list = torch.cuda.get_arch_list()
    for arch in arch_list:
        kind, _, number = arch.partition("_")
        if not number.isdigit():
            continue
        arch_major, arch_minor = int(number[:-1]), int(number[-1])
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            return True, ""
        if kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return True, ""
    name = torch.cuda.get_device_name(device_index)
    version = torch.__version__.split("+")[0]
    message = (
        f"{name} (compute capability {major}.{minor}) is not supported by the "
        f"installed torch {torch.__version__}, which was built for "
        f"{', '.join(arch_list) or 'no CUDA architectures'}. CUDA 13 builds of torch "
        f"dropped GPUs older than compute capability 7.5. To use this GPU, install "
        f"the same torch version from the CUDA 12.6 wheels, e.g. "
        f"`uv pip install --reinstall torch=={version}+cu126 torchvision --index-url "
        f"https://download.pytorch.org/whl/cu126` (match torchvision's version to torch)."
    )
    return False, message


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
