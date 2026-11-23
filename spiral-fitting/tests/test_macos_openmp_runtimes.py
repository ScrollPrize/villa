"""On macOS, vc_object_store (through Homebrew's OpenCV and OpenBLAS) and torch each load an OpenMP
runtime. Whatever is imported first, torch's threads must keep working and the other runtime must stay
on one thread. KMP_DUPLICATE_LIB_OK must be as it was before the import, even when the import fails,
so that later loads and child processes keep OpenMP's check for a second runtime. When that check is
the user's choice, or another libomp copy was loaded before torch's, the import arranges nothing. Each
case runs in a fresh interpreter, because the import order is the point."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="the two OpenMP runtimes only meet on macOS")

# In the child, an abort (OMP Error #15) or a crash exits with the signal number instead of leaving a
# macOS crash report.
PRELUDE = r"""
import ctypes, signal
libc = ctypes.CDLL(None)
libc.signal.restype = ctypes.c_void_p
libc.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
for number in (signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS):
    libc.signal(number, ctypes.cast(libc._exit, ctypes.c_void_p))
"""

SCRIPT = PRELUDE + r"""
import json, os
import {first}, {second}
import torch

torch.set_num_threads(4)
x = torch.randn(2_000_000)
for _ in range(50):
    float((x.exp() * 1.0001).sum())

torch_omp = os.path.realpath(os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib"))
libc._dyld_image_count.restype = ctypes.c_uint32
libc._dyld_get_image_name.restype = ctypes.c_char_p
libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
others = []
for index in range(libc._dyld_image_count()):
    name = libc._dyld_get_image_name(index).decode()
    if name.endswith("/libomp.dylib") and os.path.realpath(name) != torch_omp:
        others.append(ctypes.CDLL(name, mode=os.RTLD_NOLOAD).omp_get_max_threads())
print(json.dumps({{"others": others, "allowed": os.environ.get("KMP_DUPLICATE_LIB_OK")}}))
"""

PLAIN_IMPORT = PRELUDE + r"""
import importlib.util, json, os
import vc_object_store

torch_omp = os.path.join(importlib.util.find_spec("torch").submodule_search_locations[0], "lib", "libomp.dylib")
try:
    ctypes.CDLL(torch_omp, mode=os.RTLD_NOLOAD)
    loaded = True
except OSError:
    loaded = False
print(json.dumps({"allowed": os.environ.get("KMP_DUPLICATE_LIB_OK"), "torch_libomp_loaded": loaded}))
"""

FAILED_IMPORT = PRELUDE + r"""
import json, os, sys
sys.modules["vc_object_store._native"] = None  # the compiled module fails to import
try:
    import vc_object_store
    failed = False
except ImportError:
    failed = True
print(json.dumps({"failed": failed, "allowed": os.environ.get("KMP_DUPLICATE_LIB_OK")}))
"""

ANOTHER_RUNTIME_FIRST = PRELUDE + r"""
import sys
ctypes.CDLL(sys.argv[1]).omp_get_max_threads()  # another OpenMP runtime, loaded and started first
import vc_object_store
import torch
print("no OMP Error #15")
"""

LOADED_BEFORE_TORCH = PRELUDE + r"""
import sys
ctypes.CDLL(sys.argv[1])  # another OpenMP runtime, loaded before torch's and never started
import torch
libc._dyld_image_count.restype = ctypes.c_uint32
libc._dyld_get_image_name.restype = ctypes.c_char_p
libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
def runtimes():
    names = (libc._dyld_get_image_name(index) for index in range(libc._dyld_image_count()))
    return {name for name in names if name.endswith(b"/libomp.dylib")}
before = runtimes()
import vc_object_store
print("no OMP Error #15" if runtimes() - before else "no second runtime")
"""


def child(script, *args, **environment):
    torch = pytest.importorskip("torch")
    if not os.path.exists(os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib")):
        pytest.skip("this torch does not bundle its own libomp.dylib")
    env = {key: value for key, value in os.environ.items() if key != "KMP_DUPLICATE_LIB_OK"}
    env.update(environment)
    return subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True, timeout=300,
                          env=env)


def run(script, **environment):
    result = child(script, **environment)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("first, second", [("vc_object_store", "torch"), ("torch", "vc_object_store")])
def test_torch_threads_run_beside_the_object_store(first, second):
    result = run(SCRIPT.format(first=first, second=second))
    assert result["allowed"] is None
    if not result["others"]:
        pytest.skip("vc_object_store brought no second OpenMP runtime on this Mac (its OpenCV links no "
                    "OpenMP build of OpenBLAS), so there is nothing to hold to one thread")
    assert all(count == 1 for count in result["others"]), result["others"]


def test_a_true_value_the_user_set_is_kept():
    result = run(SCRIPT.format(first="vc_object_store", second="torch"), KMP_DUPLICATE_LIB_OK="TRUE")
    assert result["allowed"] == "TRUE"


@pytest.mark.parametrize("value", ["FALSE", "0", ""])
def test_a_false_value_the_user_set_leaves_the_import_as_it_was(value):
    assert run(PLAIN_IMPORT, KMP_DUPLICATE_LIB_OK=value) == {"allowed": value, "torch_libomp_loaded": False}


def test_a_failed_import_leaves_the_environment_as_it_was():
    assert run(FAILED_IMPORT) == {"failed": True, "allowed": None}


def another_libomp():
    """pykdtree (through kimimaro) bundles its own libomp.dylib."""
    spec = importlib.util.find_spec("pykdtree")
    other = os.path.join(spec.submodule_search_locations[0], ".dylibs", "libomp.dylib") if spec else ""
    if not os.path.exists(other):
        pytest.skip("no pykdtree with its own libomp.dylib here")
    return other


@pytest.mark.parametrize("script", [ANOTHER_RUNTIME_FIRST, LOADED_BEFORE_TORCH],
                         ids=["started-first", "loaded-before-torch"])
def test_a_runtime_loaded_before_torchs_is_left_to_openmp(script):
    """When another libomp copy was loaded before torch's, the import arranges nothing and OpenMP's own
    check stops the process, as it does without this module: whether that copy already runs, or was only
    loaded and then torch was imported."""
    result = child(script, another_libomp())
    if result.returncode == 0 and "no second runtime" in result.stdout:
        pytest.skip("vc_object_store brought no second OpenMP runtime on this Mac")
    assert result.returncode == 6 and "OMP: Error #15" in result.stderr, (result.returncode, result.stderr[-3000:])
