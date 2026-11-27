// HIP stand-in for <cuda_runtime.h>, used only by VC_READER_ENABLE_HIP builds.
//
// vc_reader's GPU backends are written against the CUDA runtime. For AMD GPUs the
// same sources are compiled by the HIP compiler, and this header maps the CUDA
// names they use to their HIP equivalents, so the CUDA code stays unchanged.
// The list covers the names vc_reader uses (from hipify-perl -examine); add a
// line here when new code uses another CUDA runtime name.
#pragma once

#if defined(__HIP__)
#include <hip/hip_runtime.h>
#else
// Host translation units built by the C++ compiler only need the runtime API.
#include <hip/hip_runtime_api.h>
#endif

#if HIP_VERSION_MAJOR < 7
#error "vc_reader requires ROCm 7.0 or newer"
#endif

// Version checks in vc_reader gate stream-ordered allocation (11.2) and
// __grid_constant__ kernel parameters (11.7); HIP provides the APIs used on those
// paths.
#ifndef CUDART_VERSION
#define CUDART_VERSION 12000
#endif

// CUDA's __grid_constant__ keeps a const kernel parameter in parameter memory
// when it is taken by reference. HIP has no such qualifier; kernel arguments are
// already read in place, so it expands to nothing.
#ifndef __grid_constant__
#define __grid_constant__
#endif

// Types and constants.
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaMemPool_t hipMemPool_t
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStreamLegacy hipStreamLegacy
#define cudaStreamPerThread hipStreamPerThread
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define cudaDevAttrL2CacheSize hipDeviceAttributeL2CacheSize
#define cudaDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported

// Errors and devices.
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceSynchronize hipDeviceSynchronize

// Memory.
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaHostAlloc hipHostAlloc
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#define cudaMemPoolGetAttribute hipMemPoolGetAttribute
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute

// Streams and events.
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventDestroy hipEventDestroy

// Warp-level functions. CUDA masks are 32-bit; HIP's *_sync functions take a
// 64-bit mask and reject narrower types with a static_assert (ROCm 7). These
// overloads accept the 32-bit masks of the CUDA code and pass them on widened.
// On wave32 GPUs (RDNA, the only targets cmake/Hip.cmake accepts) HIP ignores the
// upper half of the mask, so the lanes are the same as in CUDA. Overload
// resolution prefers them for unsigned masks; 64-bit masks reach HIP's own.
#if defined(__HIP__)
__device__ inline unsigned long long __ballot_sync(unsigned int mask, int predicate) {
    return __ballot_sync(static_cast<unsigned long long>(mask), predicate);
}
__device__ inline int __any_sync(unsigned int mask, int predicate) {
    return __any_sync(static_cast<unsigned long long>(mask), predicate);
}
__device__ inline int __all_sync(unsigned int mask, int predicate) {
    return __all_sync(static_cast<unsigned long long>(mask), predicate);
}
template<class T> __device__ inline unsigned long long __match_any_sync(unsigned int mask, T value) {
    return __match_any_sync(static_cast<unsigned long long>(mask), value);
}
__device__ inline void __syncwarp(unsigned int mask) {
    __syncwarp<unsigned long long>(mask);
}
template<class T> __device__ inline T __shfl_sync(unsigned int mask, T var, int src_lane, int width = warpSize) {
    return __shfl_sync(static_cast<unsigned long long>(mask), var, src_lane, width);
}
template<class T> __device__ inline T __shfl_up_sync(unsigned int mask, T var, unsigned int delta, int width = warpSize) {
    return __shfl_up_sync(static_cast<unsigned long long>(mask), var, delta, width);
}
template<class T> __device__ inline T __shfl_down_sync(unsigned int mask, T var, unsigned int delta, int width = warpSize) {
    return __shfl_down_sync(static_cast<unsigned long long>(mask), var, delta, width);
}
template<class T> __device__ inline T __shfl_xor_sync(unsigned int mask, T var, int lane_mask, int width = warpSize) {
    return __shfl_xor_sync(static_cast<unsigned long long>(mask), var, lane_mask, width);
}
#endif
