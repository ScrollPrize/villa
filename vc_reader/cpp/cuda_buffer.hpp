#pragma once
// Shared ownership for reader outputs and external augmentation inputs. CUDA
// operations complete before returning to Python; consumers may use any stream.
#include "cuda_stream.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>
namespace vcr::device {
inline void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(e));
}
struct Guard {
    int previous;
    explicit Guard(int device) { check(cudaGetDevice(&previous)); if (device != previous) check(cudaSetDevice(device)); }
    ~Guard() { cudaSetDevice(previous); }
};
struct Buffer {
    void* ptr = nullptr;
    std::size_t bytes;
    int device;
    pid_t pid = getpid();
    bool pooled = false;
    bool exported = false;
    cudaStream_t stream = vcr::device::stream();
    Buffer(std::size_t n, int d) : bytes(n),device(d) {
        Guard guard(device);
        if (!n) return;
#if CUDART_VERSION >= 11020
        int supported = 0; check(cudaDeviceGetAttribute(&supported,cudaDevAttrMemoryPoolsSupported,device));
        pooled = supported;
        if (pooled) { check(cudaMallocAsync(&ptr,n,stream)); return; }
#endif
        check(cudaMalloc(&ptr,n));
    }
    ~Buffer() {
        if (!ptr || pid != getpid()) return;
        int previous = 0; if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (cudaSetDevice(device) != cudaSuccess) return;
        // DLPack consumers may enqueue work on any stream and then drop their
        // last reference. Their allocator cannot track our external allocation.
        if (exported) cudaDeviceSynchronize();
#if CUDART_VERSION >= 11020
        if (pooled) cudaFreeAsync(ptr,stream); else
#endif
        cudaFree(ptr);
        cudaSetDevice(previous);
    }
    Buffer(const Buffer&) = delete;
};
using Storage = std::shared_ptr<Buffer>;
inline Storage allocate(std::size_t bytes, int device) { return std::make_shared<Buffer>(bytes,device); }
inline void complete() { check(cudaStreamSynchronize(stream())); }

// Reusable page-locked staging. The caller must finish all transfers before
// resizing or destroying it; CPU-only builds never include this header.
class PinnedBuffer {
    std::uint8_t* ptr_ = nullptr;
    std::size_t capacity_ = 0;
    pid_t pid_ = getpid();
public:
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    ~PinnedBuffer() { if (ptr_ && pid_ == getpid()) cudaFreeHost(ptr_); }
    void reserve(std::size_t bytes) {
        if (bytes <= capacity_) return;
        // Free first to keep peak locked memory bounded when growing.
        if (ptr_) { check(cudaFreeHost(ptr_)); ptr_ = nullptr; capacity_ = 0; }
        check(cudaHostAlloc(reinterpret_cast<void**>(&ptr_), bytes, cudaHostAllocDefault));
        capacity_ = bytes;
    }
    std::uint8_t* data() const { return ptr_; }
};
}
