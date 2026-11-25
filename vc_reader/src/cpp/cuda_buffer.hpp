#pragma once
// Shared ownership for reader outputs and external transform inputs. CUDA
// operations complete before returning to Python; consumers may use any stream.
#include "cuda_stream.hpp"
#include "cuda_pool.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>
namespace vcr::device {
inline void check(gpu::Error e) {
    if (e != gpu::success) throw std::runtime_error(std::string("CUDA: ")+gpu::get_error_string(e));
}
struct Guard {
    int previous;
    explicit Guard(int device) { check(gpu::get_device(&previous)); if (device != previous) check(gpu::set_device(device)); }
    ~Guard() { gpu::set_device(previous); }
};
struct Buffer {
    void* ptr = nullptr;
    std::size_t bytes;
    int device;
    pid_t pid = getpid();
    bool pooled = false;
    bool exported = false;
    bool borrowed = false;
    gpu::Stream stream = vcr::device::stream();
    // The synchronous caller retains external storage until all writes finish.
    Buffer(void* p, std::size_t n, int d) : ptr(p),bytes(n),device(d),borrowed(true) {}
    Buffer(std::size_t n, int d) : bytes(n),device(d) {
        Guard guard(device);
        if (!n) return;
#if defined(VC_READER_HIP) || CUDART_VERSION >= 11020
        int supported = 0; check(gpu::get_device_attribute(&supported,gpu::memory_pools_supported,device));
        pooled = supported;
        if (pooled) { prepare_pool(device); check(gpu::malloc_async(&ptr,n,stream)); return; }
#endif
        check(gpu::malloc(&ptr,n));
    }
    ~Buffer() {
        if (!ptr || borrowed || pid != getpid()) return;
        int previous = 0; if (gpu::get_device(&previous) != gpu::success) return;
        if (gpu::set_device(device) != gpu::success) return;
        // DLPack consumers may enqueue work on any stream and then drop their
        // last reference. Their allocator cannot track our external allocation.
        if (exported) gpu::device_synchronize();
#if defined(VC_READER_HIP) || CUDART_VERSION >= 11020
        if (pooled) gpu::free_async(ptr,stream); else
#endif
        gpu::free(ptr);
        gpu::set_device(previous);
    }
    Buffer(const Buffer&) = delete;
};
using Storage = std::shared_ptr<Buffer>;
inline Storage allocate(std::size_t bytes, int device) { return std::make_shared<Buffer>(bytes,device); }
inline Storage borrow(void* ptr, std::size_t bytes, int device) { return std::make_shared<Buffer>(ptr,bytes,device); }
inline void complete() { check(gpu::stream_synchronize(stream())); }

// Reusable page-locked staging. The caller must finish all transfers before
// resizing or destroying it; CPU-only builds never include this header.
class PinnedBuffer {
    std::uint8_t* ptr_ = nullptr;
    std::size_t capacity_ = 0;
    pid_t pid_ = getpid();
public:
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    ~PinnedBuffer() { if (ptr_ && pid_ == getpid()) gpu::free_host(ptr_); }
    void reserve(std::size_t bytes) {
        if (bytes <= capacity_) return;
        // Free first to keep peak locked memory bounded when growing.
        if (ptr_) { check(gpu::free_host(ptr_)); ptr_ = nullptr; capacity_ = 0; }
        check(gpu::host_alloc(reinterpret_cast<void**>(&ptr_), bytes, gpu::host_alloc_default));
        capacity_ = bytes;
    }
    std::uint8_t* data() const { return ptr_; }
};
}
