// Whether this build can use a CUDA device: false without VC_READER_CUDA, otherwise
// whether the runtime sees at least one device. For every vc_reader component.
#pragma once
#ifdef VC_READER_CUDA
#include "gpu_runtime.hpp"
#endif

namespace vcr::cuda {

inline bool available()
{
#ifdef VC_READER_CUDA
    int count = 0;
    return gpu::get_device_count(&count) == gpu::success && count > 0;
#else
    return false;
#endif
}

}  // namespace vcr::cuda
