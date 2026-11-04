"""Compilation capability checks shared by training and inference."""


def has_triton() -> bool:
    """Check Triton support, deferring to the compiler if detection is unavailable."""
    try:
        from torch.utils._triton import has_triton as torch_has_triton
    except ImportError:
        return True
    return bool(torch_has_triton())
