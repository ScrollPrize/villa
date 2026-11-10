#include "helirec/common.hpp"

#ifdef HELIREC_WITH_CUDA
#include <cuda_runtime.h>

#include "../cuda/cuda_utils.cuh"
#endif

namespace helirec {

bool cuda_available() {
#ifdef HELIREC_WITH_CUDA
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
    return false;
#endif
}

void release_pinned_memory() {
#ifdef HELIREC_WITH_CUDA
    cuda::PinnedPool::get().trim();
#endif
}

}  // namespace helirec
