"""One OpenMP runtime per process: import torch and the given native modules (in the given order), run torch's
OpenMP-parallel CPU kernels on every core, and list the libomp images dyld loaded. Fails unless exactly one libomp is
loaded and it is torch's. Run it without KMP_DUPLICATE_LIB_OK (it refuses to run with it set): a second runtime would
abort with OMP Error #15 when it starts.
  python omp_check.py MODULE...   e.g. torch vc.volume vc.fiber_trace   (order = import order)
"""
import ctypes
import importlib
import os
import sys
import time

if "KMP_DUPLICATE_LIB_OK" in os.environ:
	raise SystemExit("unset KMP_DUPLICATE_LIB_OK: it would hide a second OpenMP runtime")

for name in sys.argv[1:]:
	importlib.import_module(name)
import torch

x = torch.randn(64 << 20)


def elementwise_pass():
	# elementwise kernels split the tensor with at::parallel_for, which is OpenMP in torch's macOS builds. Returns the
	# wall time and the process's CPU time, which counts every thread
	start, cpu = time.perf_counter(), time.process_time()
	for _ in range(20):
		torch.exp(x * 0.5 + 1.0).sqrt_()
	return time.perf_counter() - start, time.process_time() - cpu


torch.set_num_threads(1)
serial, _ = elementwise_pass()
torch.set_num_threads(os.cpu_count())
parallel, parallel_cpu = elementwise_pass()
# how many cores the parallel pass kept busy on average. The speedup itself is no proof: this pass is bound by memory
# bandwidth, which a shared CI runner may not scale with its cores
busy = parallel_cpu / parallel

dyld = ctypes.CDLL(None)
dyld._dyld_image_count.restype = ctypes.c_uint32
dyld._dyld_get_image_name.restype = ctypes.c_char_p
images = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
omp = [path for path in images if os.path.basename(path).startswith(("libomp", "libiomp", "libgomp"))]
torch_omp = os.path.realpath(os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib"))
print(f"imported {' '.join(sys.argv[1:])} then torch {torch.__version__}; {os.cpu_count()} threads: "
	f"elementwise pass {serial:.2f}s on 1 thread, {parallel:.2f}s on {torch.get_num_threads()} ({serial / parallel:.1f}x, "
	f"{busy:.1f} cores busy)")
print("OpenMP runtimes loaded:", *omp, sep="\n  ")
assert len(omp) == 1 and os.path.realpath(omp[0]) == torch_omp, "expected exactly torch's libomp"
# 2 cores, or half of them on a small machine such as a 3-core CI runner
assert busy > min(2, os.cpu_count() / 2), "the OpenMP pass did not run in parallel"
print("OK: one OpenMP runtime (torch's)")
