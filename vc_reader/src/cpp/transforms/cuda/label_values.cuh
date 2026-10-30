#pragma once
#include "../cpu/labels.hpp"
#include "resident.cuh"
namespace vcr::transforms::cuda {
template<class T> __device__ void check_label_ids(KernelIndex vcr_launch, View<T> data, int* invalid) {
    EACH_INDEX(data.size) if (data.ptr[i] < -9007199254740991. || data.ptr[i] > 9007199254740991.) atomicOr(invalid,8);
}
struct check_label_ids_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { check_label_ids(index,args...); }
};
inline void require_exact_labels(const DeviceArray& a, Session& session) {
    if ((a.kind == 'i' || a.kind == 'u') && a.bytes == 8) {
        a.typed([&](auto data) { launch(blocks(data.size),256,0,vcr::device::stream(), check_label_ids_launch{}, data,session.check_flags()); });
        check(cudaGetLastError());
    }
}
template<class T> __device__ T checked_label_cast(double value, int* status) {
    if constexpr (std::is_integral_v<T>) {
        const double upper = ldexp(1.,std::numeric_limits<T>::digits), lower = std::is_signed_v<T> ? -upper : 0;
        if (!isfinite(value) || value < lower || value >= upper || trunc(value) != value) { atomicOr(status,16); return T(0); }
    }
    return T(value);
}
inline Array array_metadata(const DeviceArray& a) { Array out; out.shape=a.shape; out.kind=a.kind; out.bytes=a.bytes; return out; }
inline void require_label_value(double value) { require(std::isfinite(value) && std::abs(value)<=9007199254740991.,"label value out of exact range"); }
}
