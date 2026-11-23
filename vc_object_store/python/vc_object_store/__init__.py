"""Revisioned geometry, metadata and relationships in a single object store."""
import os as _os
import sys as _sys


def _torch_openmp_runtime():
    """On macOS, the path of the libomp.dylib that torch's wheel brings, or None. torch is not imported."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):  # a broken or stubbed torch
        return None
    for root in (spec.submodule_search_locations or []) if spec else []:
        path = _os.path.join(root, "lib", "libomp.dylib")
        if _os.path.exists(path):
            return path
    return None


def _start_torch_openmp_first(path):
    """Load and start torch's libomp before _native loads; return its resolved path (None if it does not load).

    _native links Homebrew's OpenCV, which links OpenBLAS, which loads Homebrew's libomp.dylib;
    torch's macOS wheels bring their own copy. The two copies export the same weak C++ symbols,
    and dyld binds both to the copy that loaded first. If Homebrew's loads first, torch's
    OpenMP worker threads run its code and crash in __kmp_suspend_initialize_thread, so torch's
    copy is loaded first here (the library only, torch itself is not imported).

    A copy checks for another one (OMP Error #15) when it starts, at its first OpenMP call, not
    when it loads: Homebrew's starts while _native loads (OpenBLAS's initializer calls
    omp_get_max_threads), torch's at `import torch` (its module init calls omp_set_num_threads).
    So torch's copy is started here, and both checks happen during this import.
    """
    import ctypes

    try:
        ctypes.CDLL(path, mode=_os.RTLD_GLOBAL).omp_get_max_threads()
    except (OSError, AttributeError):  # not loadable, or not an OpenMP runtime
        return None
    return _os.path.realpath(path)


def _loaded_images():
    import ctypes

    libc = ctypes.CDLL(None)
    libc._dyld_image_count.restype = ctypes.c_uint32
    libc._dyld_get_image_name.restype = ctypes.c_char_p
    libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    names = (libc._dyld_get_image_name(index) for index in range(libc._dyld_image_count()))
    return [name.decode() for name in names if name]  # in load order


def _hold_to_one_thread(runtimes):
    """Keep the libomp copies that _native brought to one thread.

    With torch's copy loaded first, a parallel region in another copy would run torch's copy of
    the shared code on its own threads and hang or crash. _native calls nothing in OpenCV that
    reaches OpenBLAS today, so this is a guard: one thread as the runtime's default and for the
    importing thread keeps any such call serial. Only copies that load during this import are
    held; a copy that another module loaded first is not held at all, for new threads either.
    In a forked process each copy starts again at its first call, outside this import, so a
    first call into Homebrew's copy there stops with OMP Error #15; nothing in _native makes one
    today.
    """
    import ctypes

    for path in runtimes:
        try:
            runtime = ctypes.CDLL(path, mode=_os.RTLD_NOLOAD)
            runtime.kmp_set_defaults(b"OMP_NUM_THREADS=1")
            runtime.omp_set_num_threads(1)
        except (OSError, AttributeError):
            pass


def _openmp_reads_true(value):
    """True for the values libomp reads as true (__kmp_str_match_true in its kmp_str.cpp)."""
    value = value.lower()

    def match(word, least):
        common = min(len(word), len(value))
        return word[:common] == value[:common] and (common >= least if least else value == word)

    return any(match(word, least) for word, least in (
        ("true", 1), ("on", 2), ("1", 1), (".true.", 2), (".t.", 2), ("yes", 1), ("enabled", 0)))


def _another_openmp_runtime_first(torch_libomp):
    """True when another libomp copy was loaded before torch's (the first one in load order is not torch's).

    That copy serves torch's threads whatever is done here, so the import arranges nothing and
    OpenMP's own check (OMP Error #15) reports it.
    """
    first = next((name for name in _loaded_images() if name.endswith("/libomp.dylib")), None)
    return first is not None and _os.path.realpath(first) != _os.path.realpath(torch_libomp)


_torch_libomp = _torch_openmp_runtime() if _sys.platform == "darwin" else None
_user_value = _os.environ.get("KMP_DUPLICATE_LIB_OK")
if _torch_libomp is not None and (
        (_user_value is not None and not _openmp_reads_true(_user_value))
        or _another_openmp_runtime_first(_torch_libomp)):
    # The user asked OpenMP to refuse a second runtime, or another libomp copy was loaded before torch's:
    # arrange nothing.
    _torch_libomp = None
if _torch_libomp is None:
    from ._native import *  # noqa: F403
else:
    # Two started OpenMP runtimes stop the process with OMP Error #15 unless KMP_DUPLICATE_LIB_OK
    # is set. It is set only while the two start here, where the load order is arranged, and
    # removed again even if the import fails, so later loads and child processes keep the check.
    # It is in the process environment for the length of this import, so a child process that
    # another thread starts in that moment inherits it. A value the user set is left alone.
    _allowed_here = _user_value is None
    if _allowed_here:
        _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    try:
        _torch_libomp = _start_torch_openmp_first(_torch_libomp)
        _images_before = set(_loaded_images()) if _torch_libomp is not None else None
        from ._native import *  # noqa: F403
        if _images_before is not None:
            _hold_to_one_thread(sorted(
                name for name in set(_loaded_images()) - _images_before
                if name.endswith("/libomp.dylib") and _os.path.realpath(name) != _torch_libomp))
    finally:
        if _allowed_here:
            _os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
