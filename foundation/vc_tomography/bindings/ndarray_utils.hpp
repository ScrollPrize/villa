#pragma once
// Helpers to move between NumPy (host) arrays and vc_tomography kernels.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "vc_tomography/common.hpp"

namespace vc_tomography::py {

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

// A NumPy copy of a container of numbers (shape: its length when not given), owning its memory
template <class V, class T = std::decay_t<decltype(*std::begin(std::declval<const V&>()))>>
nb::ndarray<nb::numpy, T> to_numpy(const V& v, std::vector<std::size_t> shape = {}) {
    const std::size_t n = static_cast<std::size_t>(std::distance(std::begin(v), std::end(v)));
    if (shape.empty()) shape = {n};
    std::size_t total = 1;
    for (std::size_t s : shape) total *= s;
    if (total != n) throw Error("an array's values do not fill its shape");
    auto a = new_array_t<T>(shape);
    std::copy(std::begin(v), std::end(v), static_cast<T*>(a.data()));
    return a;
}
// the same as an object: a property's reference_internal policy does not apply to an array that owns its memory
template <class V>
nb::object to_numpy_object(const V& v, std::vector<std::size_t> shape = {}) {
    return nb::cast(to_numpy(v, std::move(shape)), nb::rv_policy::move);
}

inline nb::object json_loads(const std::string& text) { return nb::module_::import_("json").attr("loads")(text); }

// a Python callable (None: nothing) to be called from C++ while the GIL is released: each call takes the GIL
template <class... A>
std::function<void(A...)> py_callback(const std::optional<std::function<void(A...)>>& f) {
    if (!f) return nullptr;
    return [f](A... a) {
        nb::gil_scoped_acquire gil;
        (*f)(a...);
    };
}
using LogFn = std::function<void(const std::string&)>;
// a log line handed to a Python callable (None: silent), with the GIL taken for it
inline LogFn python_log(const std::optional<LogFn>& log) { return py_callback(log); }

using IntIn = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

inline std::vector<std::size_t> shape_of(const HostIn& a) {
    std::vector<std::size_t> s(a.ndim());
    for (std::size_t i = 0; i < a.ndim(); ++i) s[i] = a.shape(i);
    return s;
}

using vc_tomography::parse_device;

inline void require(bool ok, const std::string& msg) {
    if (!ok) throw Error(msg);
}

}  // namespace vc_tomography::py
