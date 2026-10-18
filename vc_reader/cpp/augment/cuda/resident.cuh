#pragma once
#include "common.cuh"
#include "../local_field.hpp"
#include "../spatial.hpp"

namespace vcr::augment::cuda {
template<class T> struct View {
    using value_type = T;
    T* ptr;
    std::size_t size;
};
template<class To, class From> __global__ void convert_kernel(To* out, const From* in, std::size_t n) {
    EACH_INDEX(n) out[i] = To(in[i]);
}
template<class To, class From> void convert(View<To> to, View<From> from) {
    require(to.size == from.size,"CUDA conversion size mismatch");
    convert_kernel<<<blocks(to.size),256,0,vcr::device::stream()>>>(to.ptr,from.ptr,to.size); check(cudaGetLastError());
}
template<class T> void copy_device(View<T> to, View<T> from) {
    require(to.size == from.size,"CUDA copy size mismatch");
    check(cudaMemcpyAsync(to.ptr,from.ptr,to.size*sizeof(T),cudaMemcpyDeviceToDevice,vcr::device::stream()));
}
struct DeviceArray {
    Shape shape;
    char kind;
    std::size_t bytes;
    std::shared_ptr<Buffer<std::byte>> buffer;
    DeviceArray(Shape s, char k, std::size_t b) : shape(std::move(s)),kind(k),bytes(b),
        buffer(std::make_shared<Buffer<std::byte>>(product(shape)*bytes)) {}
    DeviceArray(Shape s, char k, std::size_t b, void* data) : shape(std::move(s)),kind(k),bytes(b),
        buffer(std::make_shared<Buffer<std::byte>>(static_cast<std::byte*>(data),product(shape)*bytes)) {}
    Shape spatial() const { return {shape.begin()+1,shape.end()}; }
    std::size_t size() const { return product(shape); }
    template<class T> View<T> view() const { return {reinterpret_cast<T*>(buffer->ptr),size()}; }
    template<class T> View<T> channel(std::size_t c) const {
        auto n = product(spatial()); return {reinterpret_cast<T*>(buffer->ptr)+c*n,n};
    }
    template<class F> decltype(auto) typed(F&& fn) const {
        if (kind == 'f') { if (bytes == 4) return fn(view<float>()); return fn(view<double>()); }
        if (kind == 'u' || kind == 'b') {
            if (bytes == 1) return fn(view<std::uint8_t>());
            if (bytes == 2) return fn(view<std::uint16_t>());
            if (bytes == 4) return fn(view<std::uint32_t>());
            return fn(view<std::uint64_t>());
        }
        if (bytes == 1) return fn(view<std::int8_t>());
        if (bytes == 2) return fn(view<std::int16_t>());
        if (bytes == 4) return fn(view<std::int32_t>());
        return fn(view<std::int64_t>());
    }
};
struct Session {
    std::map<std::string,DeviceArray> arrays;
    std::map<std::string,std::vector<DeviceArray>> pyramids;
    std::unique_ptr<Buffer<int>> nonfinite, geometry, checks;
    // Asynchronous sessions never read device data back to decide anything:
    // checks become device flags, raised by check_status() when consumed.
    bool async = false;
    int* finite_status();
    // Deferred directional/point failures, checked when the pipeline finishes.
    int* geometry_status();
    // Input validation and float32 overflow-protection flags (async only).
    int* check_flags();
    void check_status();
    Session() = default;
    explicit Session(const Sample&);
    void finish(Sample&);
    void restore_valid(const Sample&);
};
// Fields are already resident for newly sampled CUDA operations. Legacy/loaded
// replay plans upload their recorded field once for the operation.
inline std::shared_ptr<Buffer<float>> field_buffer(const std::vector<float>& field, const std::shared_ptr<void>& device) {
    return device ? std::static_pointer_cast<Buffer<float>>(device) : std::make_shared<Buffer<float>>(field);
}
struct Statistics { double mean, low, high, deviation; };
Buffer<Statistics> statistics(View<float>, bool deviation = false, const float* weights = nullptr);
// Sets `bit` in flags when max(|low|, |high|) exceeds limit.
void flag_magnitude(const Buffer<Statistics>&, double limit, int* flags, int bit);
Buffer<Statistics> statistics(View<double>, bool deviation = false, const float* weights = nullptr);
Buffer<Statistics> local_statistics(View<float>, LocalFieldView);
Buffer<Statistics> local_statistics(View<double>, LocalFieldView);
} // namespace vcr::augment::cuda
