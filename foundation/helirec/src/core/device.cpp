#include "helirec/common.hpp"

#ifdef HELIREC_WITH_CUDA
#include <cuda_runtime.h>
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

}  // namespace helirec
