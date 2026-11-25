#ifndef VC_READER_HIP
#error "The installed HIP package must propagate VC_READER_HIP"
#endif

#include <cuda_runtime.h>

int main() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? 0 : 1;
}
