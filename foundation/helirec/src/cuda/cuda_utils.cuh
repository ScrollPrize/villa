#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>

#include "helirec/common.hpp"

namespace helirec::cuda {

inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw Error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}
#define HELIREC_CUDA_CHECK(x) ::helirec::cuda::check((x), #x)

inline cudaStream_t as_stream(void* s) { return static_cast<cudaStream_t>(s); }

inline unsigned grid_1d(std::size_t n, unsigned block) {
    return static_cast<unsigned>((n + block - 1) / block);
}

// Minimal RAII device buffer.
template <class T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { resize(n); }
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) { o.ptr_ = nullptr; o.n_ = 0; }

    void resize(std::size_t n) {
        if (n == n_) return;
        reset();
        if (n) HELIREC_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        n_ = n;
    }
    // grow-only: keeps the allocation when it is large enough (size() is then the capacity)
    void ensure(std::size_t n) {
        if (n > n_) resize(n);
    }
    void reset() {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        n_ = 0;
    }
    void upload(const T* host, std::size_t n, cudaStream_t s = nullptr) {
        resize(n);
        HELIREC_CUDA_CHECK(cudaMemcpyAsync(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice, s));
    }
    void download(T* host, cudaStream_t s = nullptr) const {
        HELIREC_CUDA_CHECK(cudaMemcpyAsync(host, ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost, s));
        HELIREC_CUDA_CHECK(cudaStreamSynchronize(s));
    }
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return n_; }

  private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};

// Grow-only page-locked host buffer (fast, asynchronous host <-> device copies). Contents are not initialised.
template <class T>
class PinnedBuffer {
  public:
    PinnedBuffer() = default;
    ~PinnedBuffer() { reset(); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) { o.ptr_ = nullptr; o.n_ = 0; }
    void ensure(std::size_t n) {
        if (n <= n_) return;
        n = std::max(n, n_ + n_ / 4);  // geometric growth: page-locking is slow (~0.2 s/GB)
        reset();
        HELIREC_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T), cudaHostAllocDefault));
        n_ = n;
    }
    void reset() {
        if (ptr_) cudaFreeHost(ptr_);
        ptr_ = nullptr;
        n_ = 0;
    }
    T* data() { return ptr_; }
    std::size_t size() const { return n_; }

  private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};

// RAII stream
class Stream {
  public:
    Stream() { HELIREC_CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking)); }
    ~Stream() { if (s_) cudaStreamDestroy(s_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return s_; }
    void sync() const { HELIREC_CUDA_CHECK(cudaStreamSynchronize(s_)); }

  private:
    cudaStream_t s_ = nullptr;
};

}  // namespace helirec::cuda
