"""Revisioned geometry, metadata and relationships in a single object store."""
import os as _os
import sys as _sys


def _load_torch_openmp_first():
    """On macOS, load torch's libomp before _native and return its path (None if there is none).

    _native links Homebrew's OpenCV, which links OpenBLAS, which loads Homebrew's libomp.dylib;
    torch's macOS wheels bring their own copy. The two copies export the same weak C++ symbols,
    and dyld binds both to the copy that loaded first. If Homebrew's loads first, torch's
    OpenMP worker threads run its code and crash in __kmp_suspend_initialize_thread, so torch's
    copy is loaded first here (the library only, torch itself is not imported).
    """
    import ctypes
    import importlib.util

    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):  # a broken or stubbed torch
        return None
    for root in (spec.submodule_search_locations or []) if spec else []:
        path = _os.path.join(root, "lib", "libomp.dylib")
        if _os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=_os.RTLD_GLOBAL)
            except OSError:
                return None
            return _os.path.realpath(path)
    return None


def _loaded_images():
    import ctypes

    libc = ctypes.CDLL(None)
    libc._dyld_image_count.restype = ctypes.c_uint32
    libc._dyld_get_image_name.restype = ctypes.c_char_p
    libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    names = (libc._dyld_get_image_name(index) for index in range(libc._dyld_image_count()))
    return {name.decode() for name in names if name}


def _hold_to_one_thread(runtimes):
    """Keep the libomp copies that _native brought to one thread.

    With torch's copy loaded first, a parallel region in another copy would run torch's copy of
    the shared code on its own threads and hang or crash. _native calls nothing in OpenCV that
    reaches OpenBLAS today, so this is a guard: one thread as the runtime's default and for the
    importing thread keeps any such call serial.
    """
    import ctypes

    for path in runtimes:
        try:
            runtime = ctypes.CDLL(path, mode=_os.RTLD_NOLOAD)
            runtime.kmp_set_defaults(b"OMP_NUM_THREADS=1")
            runtime.omp_set_num_threads(1)
        except (OSError, AttributeError):
            pass


_torch_libomp = None
_images_before = None
if _sys.platform == "darwin":
    _torch_libomp = _load_torch_openmp_first()
    if _torch_libomp is not None:
        _images_before = _loaded_images()
        # Two OpenMP runtimes stop the process with OMP Error #15 unless this is set. It is set
        # only here, where the load order is arranged. A value the user set wins.
        _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ._native import *  # noqa: E402,F403

if _images_before is not None:
    _hold_to_one_thread(sorted(
        name for name in _loaded_images() - _images_before
        if name.endswith("/libomp.dylib") and _os.path.realpath(name) != _torch_libomp))
