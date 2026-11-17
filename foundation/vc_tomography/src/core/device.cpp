#include "vc_tomography/common.hpp"

#ifdef VC_TOMOGRAPHY_WITH_CUDA
#include <cuda_runtime.h>

#include "../cuda/cuda_utils.cuh"
#endif

namespace vc_tomography {

bool cuda_available() {
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
    return false;
#endif
}

Device parse_device(const std::string& name, bool allow_auto) {
    if (name == "cpu") return Device::CPU;
    if (name == "cuda") {
        if (!cuda_available()) throw Error("device 'cuda' requested but CUDA is not available");
        return Device::CUDA;
    }
    if (allow_auto && name == "auto") return cuda_available() ? Device::CUDA : Device::CPU;
    throw Error("unknown device '" + name + (allow_auto ? "' (expected cpu, cuda or auto)" : "' (expected cpu or cuda)"));
}

void release_pinned_memory() {
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    cuda::PinnedPool::get().trim();
#endif
}

void release_device_scratch() {
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    cudaDeviceSynchronize();   // (work in flight may still use the scratch; an error here is the caller's to see later)
    cuda::ThreadScratch::release();
#endif
}

}  // namespace vc_tomography
