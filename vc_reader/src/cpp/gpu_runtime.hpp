#pragma once

// Host runtime calls used by installed headers. Keep CUDA's header names and
// global API available to native consumers that also link the NVIDIA runtime.
#ifdef VC_READER_HIP
#include <hip/hip_runtime_api.h>
#define VCR_GPU_SYMBOL(hip, cuda) hip
#else
#include <cuda_runtime.h>
#define VCR_GPU_SYMBOL(hip, cuda) cuda
#endif
#include <cstddef>

namespace vcr::gpu {
using Error = VCR_GPU_SYMBOL(hipError_t, cudaError_t);
using Event = VCR_GPU_SYMBOL(hipEvent_t, cudaEvent_t);
using MemoryPool = VCR_GPU_SYMBOL(hipMemPool_t, cudaMemPool_t);
using Stream = VCR_GPU_SYMBOL(hipStream_t, cudaStream_t);

inline constexpr auto success = VCR_GPU_SYMBOL(hipSuccess, cudaSuccess);
inline constexpr auto event_disable_timing = VCR_GPU_SYMBOL(hipEventDisableTiming, cudaEventDisableTiming);
inline constexpr auto host_alloc_default = VCR_GPU_SYMBOL(hipHostMallocDefault, cudaHostAllocDefault);
inline constexpr auto host_to_device = VCR_GPU_SYMBOL(hipMemcpyHostToDevice, cudaMemcpyHostToDevice);
inline constexpr auto device_to_host = VCR_GPU_SYMBOL(hipMemcpyDeviceToHost, cudaMemcpyDeviceToHost);
inline constexpr auto memory_pools_supported = VCR_GPU_SYMBOL(hipDeviceAttributeMemoryPoolsSupported, cudaDevAttrMemoryPoolsSupported);
inline constexpr auto release_threshold = VCR_GPU_SYMBOL(hipMemPoolAttrReleaseThreshold, cudaMemPoolAttrReleaseThreshold);

// These stream selectors are pointer-valued macros, not constant expressions.
inline Stream legacy_stream() {
#if defined(VC_READER_HIP) && !defined(hipStreamLegacy)
    return nullptr;
#else
    return VCR_GPU_SYMBOL(hipStreamLegacy, cudaStreamLegacy);
#endif
}
inline Stream per_thread_stream() { return VCR_GPU_SYMBOL(hipStreamPerThread, cudaStreamPerThread); }

inline constexpr auto get_error_string = VCR_GPU_SYMBOL(hipGetErrorString, cudaGetErrorString);
inline constexpr auto get_device = VCR_GPU_SYMBOL(hipGetDevice, cudaGetDevice);
inline constexpr auto set_device = VCR_GPU_SYMBOL(hipSetDevice, cudaSetDevice);
inline constexpr auto get_device_count = VCR_GPU_SYMBOL(hipGetDeviceCount, cudaGetDeviceCount);
inline constexpr auto device_synchronize = VCR_GPU_SYMBOL(hipDeviceSynchronize, cudaDeviceSynchronize);
inline constexpr auto get_device_attribute = VCR_GPU_SYMBOL(hipDeviceGetAttribute, cudaDeviceGetAttribute);
inline constexpr auto get_default_pool = VCR_GPU_SYMBOL(hipDeviceGetDefaultMemPool, cudaDeviceGetDefaultMemPool);
inline constexpr auto get_pool_attribute = VCR_GPU_SYMBOL(hipMemPoolGetAttribute, cudaMemPoolGetAttribute);
inline constexpr auto set_pool_attribute = VCR_GPU_SYMBOL(hipMemPoolSetAttribute, cudaMemPoolSetAttribute);
inline constexpr auto free = VCR_GPU_SYMBOL(hipFree, cudaFree);
inline constexpr auto free_host = VCR_GPU_SYMBOL(hipHostFree, cudaFreeHost);
inline constexpr auto memcpy_async = VCR_GPU_SYMBOL(hipMemcpyAsync, cudaMemcpyAsync);
inline constexpr auto stream_synchronize = VCR_GPU_SYMBOL(hipStreamSynchronize, cudaStreamSynchronize);
inline constexpr auto stream_wait_event = VCR_GPU_SYMBOL(hipStreamWaitEvent, cudaStreamWaitEvent);
inline constexpr auto event_create = VCR_GPU_SYMBOL(hipEventCreateWithFlags, cudaEventCreateWithFlags);
inline constexpr auto event_record = VCR_GPU_SYMBOL(hipEventRecord, cudaEventRecord);
inline constexpr auto event_synchronize = VCR_GPU_SYMBOL(hipEventSynchronize, cudaEventSynchronize);
inline constexpr auto event_destroy = VCR_GPU_SYMBOL(hipEventDestroy, cudaEventDestroy);

// Use the void** overloads even when compiling with a device compiler.
inline Error malloc(void** ptr, std::size_t size) { return VCR_GPU_SYMBOL(hipMalloc, cudaMalloc)(ptr, size); }
inline Error host_alloc(void** ptr, std::size_t size, unsigned flags) {
    return VCR_GPU_SYMBOL(hipHostAlloc, cudaHostAlloc)(ptr, size, flags);
}
#if defined(VC_READER_HIP) || CUDART_VERSION >= 11020
inline Error malloc_async(void** ptr, std::size_t size, Stream stream) {
    return VCR_GPU_SYMBOL(hipMallocAsync, cudaMallocAsync)(ptr, size, stream);
}
inline constexpr auto free_async = VCR_GPU_SYMBOL(hipFreeAsync, cudaFreeAsync);
#endif
}  // namespace vcr::gpu

#undef VCR_GPU_SYMBOL
