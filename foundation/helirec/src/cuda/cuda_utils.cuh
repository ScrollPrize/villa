#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

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

// Process-wide cache of page-locked allocations. Locking pages costs ~0.2 s/GB, so the buffers a reconstructor
// releases (its raw windows: several GB) serve the next one (e.g. after a parameter change). At most kKeepBytes are
// kept (the rest is freed at once); trim() frees them all.
class PinnedPool {
  public:
    static constexpr std::size_t kKeepBytes = std::size_t(24) << 30;
    static PinnedPool& get() {
        static PinnedPool* pool = new PinnedPool;   // (never destroyed: no CUDA calls at exit)
        return *pool;
    }
    // an allocation of at least `bytes` (and not much more), or a new one; `got`: its size
    void* take(std::size_t bytes, std::size_t& got) {
        {
            std::lock_guard<std::mutex> lock(m_);
            std::size_t best = free_.size();
            for (std::size_t i = 0; i < free_.size(); ++i)
                if (free_[i].second >= bytes && free_[i].second <= bytes + bytes / 2 + (std::size_t(64) << 20) &&
                    (best == free_.size() || free_[i].second < free_[best].second))
                    best = i;
            if (best < free_.size()) {
                void* p = free_[best].first;
                got = free_[best].second;
                kept_ -= got;
                free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(best));
                return p;
            }
        }
        void* p = nullptr;
        HELIREC_CUDA_CHECK(cudaHostAlloc(&p, bytes, cudaHostAllocDefault));
        got = bytes;
        return p;
    }
    void give(void* p, std::size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(m_);
            if (kept_ + bytes <= kKeepBytes) {
                free_.push_back({p, bytes});
                kept_ += bytes;
                return;
            }
        }
        cudaFreeHost(p);
    }
    // free an allocation outright (a block its buffer has outgrown: keeping it would hold both sizes at once)
    void drop(void* p) { cudaFreeHost(p); }
    void trim() {
        std::vector<std::pair<void*, std::size_t>> all;
        {
            std::lock_guard<std::mutex> lock(m_);
            all.swap(free_);
            kept_ = 0;
        }
        for (auto& e : all) cudaFreeHost(e.first);
    }

  private:
    std::mutex m_;
    std::vector<std::pair<void*, std::size_t>> free_;
    std::size_t kept_ = 0;
};

// Grow-only page-locked host buffer (fast, asynchronous host <-> device copies), from the PinnedPool. Contents are
// not initialised.
template <class T>
class PinnedBuffer {
  public:
    PinnedBuffer() = default;
    ~PinnedBuffer() { reset(); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_), bytes_(o.bytes_) { o.ptr_ = nullptr; o.n_ = 0; o.bytes_ = 0; }
    void ensure(std::size_t n) {
        if (n <= n_) return;
        n = std::max(n, n_ + n_ / 4);  // geometric growth: page-locking is slow (~0.2 s/GB)
        // the outgrown block is freed, not pooled: the pool could only hand it to a smaller buffer, and meanwhile
        // both sizes would stay locked (raw windows are tens of GB where chunks see different numbers of projections)
        if (ptr_) PinnedPool::get().drop(ptr_);
        ptr_ = nullptr;
        n_ = 0;
        bytes_ = 0;
        ptr_ = static_cast<T*>(PinnedPool::get().take(n * sizeof(T), bytes_));
        n_ = bytes_ / sizeof(T);
    }
    void reset() {
        if (ptr_) PinnedPool::get().give(ptr_, bytes_);
        ptr_ = nullptr;
        n_ = 0;
        bytes_ = 0;
    }
    T* data() { return ptr_; }
    std::size_t size() const { return n_; }

  private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0, bytes_ = 0;
};

// Device rows -> pageable host memory (e.g. a new NumPy array) through a pinned staging buffer: `rows` rows of
// `width` elements at device pitch `src_pitch` elements, to `dst` at a pitch of `dst_pitch` elements (0: packed). The
// host copy runs on several threads (first-touch page faults of a fresh destination dominate a direct pageable copy).
template <class T>
void download_rows(T* dst, const T* src, std::size_t src_pitch, std::size_t width, std::size_t rows, PinnedBuffer<T>& stage,
                   std::size_t dst_pitch = 0) {
    const std::size_t n = width * rows;
    stage.ensure(n);
    HELIREC_CUDA_CHECK(cudaMemcpy2D(stage.data(), width * sizeof(T), src, src_pitch * sizeof(T), width * sizeof(T), rows,
                                    cudaMemcpyDeviceToHost));
    const std::size_t step = std::size_t(1) << 20;
    const T* s = stage.data();
    if (dst_pitch && dst_pitch != width) {
#pragma omp parallel for schedule(static) num_threads(8)
        for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r)
            std::copy(s + r * width, s + (r + 1) * width, dst + r * dst_pitch);
        return;
    }
#pragma omp parallel for schedule(static) num_threads(8)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>((n + step - 1) / step); ++i) {
        const std::size_t a = static_cast<std::size_t>(i) * step;
        std::copy(s + a, s + std::min(n, a + step), dst + a);
    }
}

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
