"""Minimal NVRTC + CUDA driver API wrapper (no nvcc needed).

Compiles CUDA C++ at runtime and launches kernels into the context PyTorch
already created, using torch tensors for all device memory.
"""
import ctypes, glob, os
import torch

_nvrtc = None
_cuda = None


def _libs():
    global _nvrtc, _cuda
    if _nvrtc is None:
        sp = os.path.dirname(os.path.dirname(torch.__file__))
        cands = [c for c in glob.glob(os.path.join(sp, 'nvidia', 'cuda_nvrtc', 'lib', 'libnvrtc*.so*'))
                 if 'builtins' not in c]
        if not cands:
            raise RuntimeError("libnvrtc not found")
        # Sorted so the plain (non-`.alt.`) library is tried first -- deterministic either
        # way, but the REAL fault this traced to (2026-09-23, .142, every V100, every
        # render) was version, not which of the two: `nvidia-cuda-nvrtc-cu12` resolved to
        # 12.9.86 alongside torch==2.13.0+cu126 (the package's own version drifts loose of
        # torch's build tag), and NVRTC 12.9 emits PTX the 560.35.03 driver's JIT rejects
        # with `cuModuleLoadData failed with 222` (CUDA_ERROR_UNSUPPORTED_PTX_VERSION) on
        # BOTH the alt and non-alt library -- confirmed by reproducing the failure on each
        # standalone before finding the version mismatch. Nothing else in this codebase
        # calls nvrtcCompileProgram, so no other test caught it. `setup.py`'s grandprize
        # env build now pins `nvidia-cuda-nvrtc-cu12==12.6.*` for exactly this reason; this
        # sort is a smaller, secondary hedge in case a future box's toolchain differs
        # between the two builds too.
        cands.sort(key=lambda c: (".alt." in c, c))
        _nvrtc = ctypes.CDLL(cands[0])
        _cuda = ctypes.CDLL("libcuda.so.1")
    return _nvrtc, _cuda


def _chk(r, what):
    if r != 0:
        raise RuntimeError(f"{what} failed with {r}")


class Module:
    def __init__(self, src: str, device: int = 0, opts=None):
        nvrtc, cuda = _libs()
        torch.zeros(1, device=f'cuda:{device}')  # ensure context
        p = torch.cuda.get_device_properties(device)
        prog = ctypes.c_void_p()
        _chk(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"k.cu", 0, None, None),
             "nvrtcCreateProgram")
        o = [f"--gpu-architecture=compute_{p.major}{p.minor}".encode(),
             b"--fmad=false",          # never auto-contract; FMA is written explicitly
             b"-default-device"]
        if opts:
            o += [x.encode() for x in opts]
        arr = (ctypes.c_char_p * len(o))(*o)
        r = nvrtc.nvrtcCompileProgram(prog, len(o), arr)
        sz = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
        log = ctypes.create_string_buffer(sz.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        if r != 0:
            raise RuntimeError("NVRTC compile error:\n" + log.value.decode())
        nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(sz))
        ptx = ctypes.create_string_buffer(sz.value)
        nvrtc.nvrtcGetPTX(prog, ptx)
        self._mod = ctypes.c_void_p()
        _chk(cuda.cuModuleLoadData(ctypes.byref(self._mod), ptx), "cuModuleLoadData")
        self._fns = {}
        self._cuda = cuda

    def fn(self, name: str):
        if name not in self._fns:
            f = ctypes.c_void_p()
            _chk(self._cuda.cuModuleGetFunction(ctypes.byref(f), self._mod, name.encode()),
                 f"cuModuleGetFunction({name})")
            self._fns[name] = f
        return self._fns[name]

    def launch(self, name, grid, block, args, shared=0, stream=None):
        """args: list of (ctypes value | torch.Tensor)."""
        boxes, ptrs = [], []
        for a in args:
            if isinstance(a, torch.Tensor):
                b = ctypes.c_void_p(a.data_ptr())
            else:
                b = a
            boxes.append(b)
            ptrs.append(ctypes.cast(ctypes.byref(b), ctypes.c_void_p))
        argv = (ctypes.c_void_p * len(ptrs))(*ptrs)
        gx, gy, gz = (list(grid) + [1, 1])[:3]
        bx, by, bz = (list(block) + [1, 1])[:3]
        s = ctypes.c_void_p(stream) if stream else None
        _chk(self._cuda.cuLaunchKernel(self.fn(name), gx, gy, gz, bx, by, bz,
                                       shared, s, argv, None), f"launch({name})")
