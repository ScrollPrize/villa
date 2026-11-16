#pragma once
// Host-level calls of the CPU / CUDA array functions: the arrays are given in host memory; on the CPU the function
// takes them as they are, on CUDA it takes device copies, the inputs uploaded before and the outputs downloaded after.
//
//   staged(device, [&](const float* a, float* b) { VC_TOMOGRAPHY_ON(device, f, a, b, n); },
//          to_device(a_host, n), from_device(b_host, n));
//
// to_device(p, n): read; from_device(p, n): written; through_device(p, n): both. Every pointer the function gets is in the device's memory.
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "vc_tomography/common.hpp"
#ifdef VC_TOMOGRAPHY_WITH_CUDA
#include "../cuda/cuda_utils.cuh"
#endif

namespace vc_tomography {

template <class T, bool Read, bool Write>
struct Staged {
    using value_type = std::remove_const_t<T>;
    static constexpr bool read = Read, write = Write;
    T* host;
    std::size_t n;
};
template <class T>
Staged<const T, true, false> to_device(const T* p, std::size_t n) { return {p, n}; }
template <class T>
Staged<T, false, true> from_device(T* p, std::size_t n) { return {p, n}; }
template <class T>
Staged<T, true, true> through_device(T* p, std::size_t n) { return {p, n}; }

// f(args...) as cpu::f or cuda::f, by the device
#ifdef VC_TOMOGRAPHY_WITH_CUDA
#define VC_TOMOGRAPHY_ON(device, f, ...) ((device) == ::vc_tomography::Device::CPU ? cpu::f(__VA_ARGS__) : cuda::f(__VA_ARGS__))
#else
#define VC_TOMOGRAPHY_ON(device, f, ...) cpu::f(__VA_ARGS__)
#endif

template <class Run, class... Bufs>
void staged(Device device, Run&& run, Bufs... bufs) {
    if (device == Device::CPU) {
        run(bufs.host...);
        return;
    }
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    std::tuple<cuda::DeviceBuffer<typename Bufs::value_type>...> dev;
    const auto stage = [](const auto& b, auto& d) {
        d.resize(std::max<std::size_t>(b.n, 1));
        if constexpr (std::decay_t<decltype(b)>::read)
            if (b.n) d.upload(b.host, b.n);
    };
    std::apply([&](auto&... d) { (stage(bufs, d), ...); }, dev);
    std::apply([&](auto&... d) { run(d.data()...); }, dev);
    const auto back = [](const auto& b, auto& d) {
        if constexpr (std::decay_t<decltype(b)>::write)
            if (b.n) VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpy(b.host, d.data(), b.n * sizeof(*b.host), cudaMemcpyDeviceToHost));
    };
    std::apply([&](auto&... d) { (back(bufs, d), ...); }, dev);
#else
    throw Error("built without CUDA");
#endif
}

}  // namespace vc_tomography
