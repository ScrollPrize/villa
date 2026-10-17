#pragma once
#include "../median.hpp"

namespace vcr::augment::cuda {
// Both the feasibility benchmark and public backend compile the CPU selector's
// schedule into registers. There is only one source of comparator ordering.
template<std::size_t I = 0, class T> __device__ __forceinline__ void select27(T* v) {
    using namespace median_detail;
    if constexpr (I < network.count) {
        constexpr auto c = network.comparisons[I];
        if constexpr (c.minimum || c.maximum) {
            const auto a = v[c.low], b = v[c.high];
            if constexpr (c.minimum) v[c.low] = b < a ? b : a;
            if constexpr (c.maximum) v[c.high] = a < b ? b : a;
        }
        select27<I + 1>(v);
    }
}
}
