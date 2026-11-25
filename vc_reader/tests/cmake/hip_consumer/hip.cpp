#ifndef VC_READER_HIP
#error "The installed HIP package must propagate VC_READER_HIP"
#endif

#include <vc_reader/cuda.hpp>
#include <vc_reader/cuda_available.hpp>
#include <type_traits>

static_assert(std::is_same_v<vcr::gpu::Stream, hipStream_t>);

bool hip_available() { return vcr::cuda::available(); }
