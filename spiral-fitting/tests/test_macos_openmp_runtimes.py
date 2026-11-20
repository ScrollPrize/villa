"""On macOS, vc_object_store (through Homebrew's OpenCV and OpenBLAS) and torch each load an OpenMP
runtime. Whatever is imported first, torch's threads must keep working and the other runtime must stay
on one thread. Each case runs in a fresh interpreter, because the import order is the point."""
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="the two OpenMP runtimes only meet on macOS")

SCRIPT = r"""
import ctypes, json, os
import {first}, {second}
import torch

torch.set_num_threads(4)
x = torch.randn(2_000_000)
for _ in range(50):
    float((x.exp() * 1.0001).sum())

torch_omp = os.path.realpath(os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib"))
libc = ctypes.CDLL(None)
libc._dyld_image_count.restype = ctypes.c_uint32
libc._dyld_get_image_name.restype = ctypes.c_char_p
libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
others = []
for index in range(libc._dyld_image_count()):
    name = libc._dyld_get_image_name(index).decode()
    if name.endswith("/libomp.dylib") and os.path.realpath(name) != torch_omp:
        others.append(ctypes.CDLL(name, mode=os.RTLD_NOLOAD).omp_get_max_threads())
print(json.dumps(others))
"""


@pytest.mark.parametrize("first, second", [("vc_object_store", "torch"), ("torch", "vc_object_store")])
def test_torch_threads_run_beside_the_object_store(first, second):
    torch = pytest.importorskip("torch")
    if not os.path.exists(os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib")):
        pytest.skip("this torch does not bundle its own libomp.dylib")
    env = {key: value for key, value in os.environ.items() if key != "KMP_DUPLICATE_LIB_OK"}
    result = subprocess.run([sys.executable, "-c", SCRIPT.format(first=first, second=second)],
                            capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stderr[-3000:]
    other_threads = json.loads(result.stdout.strip().splitlines()[-1])
    assert all(count == 1 for count in other_threads), other_threads
