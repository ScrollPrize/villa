"""Device selection and accelerator helpers.

Kept in one place so a new backend is added once rather than at every call
site. Before this existed, the device was chosen as
``cuda if torch.cuda.is_available() else cpu``, which silently sent every
Apple Silicon machine down the CPU path, and the autocast device type was
chosen as ``"cuda" if device.type == "cuda" else "cpu"``, which makes
autocast a no-op when the tensors are on MPS.
"""
import os
import torch

__all__ = ["select_device", "amp_device_type", "autocast_enabled", "sync", "empty_cache"]


def select_device(prefer: str | None = None) -> torch.device:
    """cuda, then mps, then cpu. ``INK_DEVICE`` or *prefer* overrides."""
    want = (prefer or os.getenv("INK_DEVICE") or "").strip().lower()
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_device_type(device: torch.device) -> str:
    """The device_type torch.autocast should be given for *device*.

    Must follow the device the tensors actually live on. Passing "cpu" while
    the tensors are on MPS does not error, it silently disables autocast.
    """
    t = device.type if isinstance(device, torch.device) else str(device)
    return t if t in ("cuda", "mps", "cpu", "xpu") else "cpu"


def sync(device: torch.device) -> None:
    """Block until queued work on *device* has finished, for honest timing."""
    t = device.type if isinstance(device, torch.device) else str(device)
    if t == "cuda":
        torch.cuda.synchronize()
    elif t == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device | None = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _native_cpu_bf16() -> bool:
    """True when the CPU itself has BF16 instructions (x86 AVX512_BF16 or AMX).

    Torch 2.11+ has torch.cpu.get_capabilities(). Torch 2.5-2.10 (the ink
    image uses 2.10) has only private probes and no AMX_BF16 one, so AMX_TILE
    stands in there. Both report the hardware and ignore ONEDNN_MAX_CPU_ISA.
    Torch before 2.5 has neither, so it gets False (CPU FP32).
    """
    get_capabilities = getattr(torch.cpu, "get_capabilities", None)
    if get_capabilities is not None:
        caps = get_capabilities()
        return bool(caps.get("avx512_bf16", False) or caps.get("amx_bf16", False))
    probes = ("_is_avx512_bf16_supported", "_is_amx_tile_supported")
    return any(bool(getattr(torch.cpu, name, lambda: False)()) for name in probes)


# oneDNN ISA caps below AVX512_CORE_BF16: BF16 there is emulated and slower than FP32.
_ONEDNN_CAPS_WITHOUT_BF16 = frozenset({
    "SSE41", "AVX", "AVX2", "AVX2_VNNI", "AVX2_VNNI_2", "AVX512_CORE", "AVX512_CORE_VNNI",
})


def _onednn_cap_allows_bf16() -> bool:
    """False when ONEDNN_MAX_CPU_ISA (or DNNL_MAX_CPU_ISA) caps oneDNN below native BF16."""
    cap = os.getenv("ONEDNN_MAX_CPU_ISA") or os.getenv("DNNL_MAX_CPU_ISA") or ""
    return cap.strip().upper() not in _ONEDNN_CAPS_WITHOUT_BF16


def autocast_enabled(device: torch.device) -> bool:
    """Keep accelerator AMP; use CPU BF16 only with native BF16 hardware.

    Both _native_cpu_bf16() and the oneDNN backend query must agree, and an
    ONEDNN_MAX_CPU_ISA cap must not exclude native BF16 (the hardware probe
    ignores the cap, and the oneDNN query only reflects AVX2-level caps).
    These are partly private APIs, so a failed or missing query uses CPU FP32.
    """
    if device.type != "cpu":
        return True
    if not _onednn_cap_allows_bf16():
        return False
    if not torch.backends.mkldnn.is_available() or not torch.backends.mkldnn.enabled:
        return False
    try:
        return _native_cpu_bf16() and bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())
    except (AttributeError, RuntimeError):
        return False
