#pragma once
#include "../shared/median.hpp"

namespace vcr::transforms::cuda {
// Compile the shared rank-selection schedule into register operations.
template<const auto& Schedule, std::size_t I = 0, class T>
__device__ __forceinline__ void select_median(T* v) {
    if constexpr (I < Schedule.count) {
        constexpr auto c = Schedule.comparisons[I];
        const auto a = v[c.low], b = v[c.high];
        if constexpr (c.minimum) v[c.low] = b < a ? b : a;
        if constexpr (c.maximum) v[c.high] = a < b ? b : a;
        select_median<Schedule, I + 1>(v);
    }
}
}
