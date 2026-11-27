#pragma once
// DLPack device type of vc_reader's GPU arrays: CUDA, or ROCm in builds that
// compile the GPU backends with HIP (VC_READER_ENABLE_HIP). PyTorch and other
// DLPack consumers on AMD GPUs expect ROCm-typed arrays.
#include <nanobind/ndarray.h>

namespace vcr {
#ifdef VC_READER_HIP
using GpuDevice = nanobind::device::rocm;
#else
using GpuDevice = nanobind::device::cuda;
#endif
}
