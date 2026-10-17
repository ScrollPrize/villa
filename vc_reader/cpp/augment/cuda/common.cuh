#pragma once
#include "../core.hpp"
#include <cuda_runtime.h>
#include <unistd.h>
namespace vcr::augment::cuda {
inline void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA augmentation: ") + cudaGetErrorString(status));
}
// Stream-ordered allocations reuse CUDA's device pool. Buffers remain owned by
// the invocation; no cached sample pointers or shared mutable scratch exist.
template<class T> struct Buffer {
    T* ptr = nullptr;
    std::size_t size;
    bool pooled = false;
    bool borrowed = false;
    bool exported = false;
    int device = 0;
    pid_t pid = getpid();
    explicit Buffer(std::size_t n) : size(n) {
        check(cudaGetDevice(&device));
        if (!n) return;
#if CUDART_VERSION >= 11020
        int supported = 0;
        check(cudaDeviceGetAttribute(&supported,cudaDevAttrMemoryPoolsSupported,device));
        pooled = supported;
        if (pooled) { check(cudaMallocAsync(&ptr,n*sizeof(T),0)); return; }
#endif
        check(cudaMalloc(&ptr,n*sizeof(T)));
    }
    explicit Buffer(const std::vector<T>& v) : Buffer(v.size()) { upload(v.data()); }
    Buffer(T* data, std::size_t n) : ptr(data),size(n),borrowed(true) { check(cudaGetDevice(&device)); }
    Buffer(Buffer&& other) noexcept : ptr(std::exchange(other.ptr,nullptr)),size(other.size),pooled(other.pooled),borrowed(other.borrowed),exported(other.exported),device(other.device),pid(other.pid) {}
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        if (!ptr || borrowed || pid != getpid()) return;
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) return;
        if (exported) cudaDeviceSynchronize();
#if CUDART_VERSION >= 11020
        if (pooled) cudaFreeAsync(ptr,0); else
#endif
        cudaFree(ptr);
        cudaSetDevice(previous);
    }
    void upload(const T* p) { if (size) check(cudaMemcpy(ptr,p,size*sizeof(T),cudaMemcpyHostToDevice)); }
    void download(T* p) const { if (size) check(cudaMemcpy(p,ptr,size*sizeof(T),cudaMemcpyDeviceToHost)); }
};
struct Layout {
    std::size_t shape[3]{1,1,1}, stride[3]{1,1,1}, size = 1;
    int dim;
    explicit Layout(const Grid& g) : size(g.size), dim(int(g.shape.size())) {
        for (int a = 0; a < dim; ++a) { shape[a] = g.shape[a]; stride[a] = g.stride[a]; }
    }
    __device__ std::size_t coord(std::size_t i, int a) const {
        // Patch indices normally fit in 32 bits. Avoid the much more expensive
        // 64-bit integer division in that case; retain the full-width path for
        // large volumes. Every caller supplies a spatial index below size.
        if (size <= 0xffffffffull) return unsigned(i) / unsigned(stride[a]) % unsigned(shape[a]);
        return i / stride[a] % shape[a];
    }
};
inline unsigned blocks(std::size_t n) { return unsigned(std::min<std::size_t>((n + 255) / 256, 65535)); }
#define EACH_INDEX(n) for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < (n); i += std::size_t(blockDim.x) * gridDim.x)
__device__ inline double clamp_value(double x, double low, double high) { return x < low ? low : x > high ? high : x; }
__device__ inline std::size_t boundary(long long x, std::size_t n, bool mirror) {
    if (n == 1) return 0;
    const auto period = static_cast<long long>(mirror ? 2 * (n - 1) : 2 * n);
    x = (x % period + period) % period;
    return x < static_cast<long long>(n) ? x : period - x - (mirror ? 0 : 1);
}
template<class T> __device__ double power(double x, double exponent) {
    if constexpr (std::is_same_v<T, float>) {
        if ((x == 0 || x >= 0x1p-126) && exponent >= 0x1p-126 && exponent <= 16)
            return powf(float(x), float(exponent));
    }
    return pow(x, exponent);
}

} // namespace vcr::augment::cuda
