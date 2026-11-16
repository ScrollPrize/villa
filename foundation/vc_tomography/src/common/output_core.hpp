#pragma once
// float32 -> uint8 output shared by the CPU and CUDA kernels (kernels.hpp to_uint8).
#include <cstdint>

#include "hd_math.hpp"

namespace helirec::output_core {

// rint(clamp((v - lo) / range, 0, 1) * 255): every operation rounded once, ties to even; NaN -> 0
HR_HD std::uint8_t window_u8(float v, float lo, float range) {
    float t = hd::div(hd::sub(v, lo), range);
    t = t > 0.f ? (t < 1.f ? t : 1.f) : 0.f;
#if defined(__CUDA_ARCH__)
    return static_cast<std::uint8_t>(rintf(hd::mul(t, 255.f)));
#else
    return static_cast<std::uint8_t>(std::rint(t * 255.f));
#endif
}

// Pixel (X, Y) of an (n, n) slice outside its reconstruction disk: (X - c)^2 + (Y - c)^2 > (n / 2)^2 with
// c = (n - 1) / 2, in exact integers (doubled coordinates).
HR_HD bool outside_disk(int X, int Y, int n) {
    const long long a = 2LL * X - (n - 1), b = 2LL * Y - (n - 1), r = 2LL * (n / 2);
    return a * a + b * b > r * r;
}

// The mask element that output index v reads along one axis of a mask with n elements, each covering f output
// elements, shifted by o: (v + o) / f, clamped to the mask.
HR_HD int mask_index(int v, int o, int f, int n) { return hd::imin(hd::imax(v + o, 0) / f, n - 1); }

}  // namespace helirec::output_core
