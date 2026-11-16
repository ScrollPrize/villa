#pragma once
// numpy's pairwise summation of float32 values (what numpy.sum and numpy.mean give for a contiguous run of values,
// including a reduction over axes numpy coalesces into one): for reproducing numpy results bit for bit.
#include <cstddef>

namespace helirec {

inline float pairwise_sum(const float* a, std::size_t n) {
    if (n < 8) {
        float r = 0.f;
        for (std::size_t i = 0; i < n; ++i) r += a[i];
        return r;
    }
    if (n <= 128) {
        float r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

}  // namespace helirec
