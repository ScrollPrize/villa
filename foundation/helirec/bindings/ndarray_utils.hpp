#pragma once
// Helpers to move between NumPy (host) arrays and helirec kernels.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include "helirec/common.hpp"

namespace helirec::py {

namespace nb = nanobind;

using HostIn = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using HostOut = nb::ndarray<nb::numpy, float>;

// New C-contiguous float32 NumPy array owning its memory.
inline HostOut new_array(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (auto s : shape) n *= s;
    float* data = new float[n ? n : 1];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return HostOut(data, shape.size(), shape.data(), owner);
}

inline float* data(HostOut& a) { return static_cast<float*>(a.data()); }

template <class T>
nb::ndarray<nb::numpy, T> new_array_t(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (auto s : shape) n *= s;
    T* p = new T[n ? n : 1];
    nb::capsule owner(p, [](void* q) noexcept { delete[] static_cast<T*>(q); });
    return nb::ndarray<nb::numpy, T>(p, shape.size(), shape.data(), owner);
}

using IntIn = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

inline std::vector<std::size_t> shape_of(const HostIn& a) {
    std::vector<std::size_t> s(a.ndim());
    for (std::size_t i = 0; i < a.ndim(); ++i) s[i] = a.shape(i);
    return s;
}

inline Device parse_device(const std::string& d) {
    if (d == "cpu") return Device::CPU;
    if (d == "cuda") {
        if (!cuda_available()) throw Error("device='cuda' requested but CUDA is not available");
        return Device::CUDA;
    }
    throw Error("unknown device '" + d + "' (expected 'cpu' or 'cuda')");
}

inline void require(bool ok, const std::string& msg) {
    if (!ok) throw Error(msg);
}

}  // namespace helirec::py
