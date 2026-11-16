#pragma once
// Bilinear sample of a low-resolution glare map (sld.hpp) at a detector pixel, shared by CPU and CUDA.
// Map cell (i, j) covers detector rows [i bin, (i + 1) bin): its centre is row (i + 1/2) bin - 1/2.
#include <cstddef>

#include "hd_math.hpp"

namespace helirec::sld_core {

using namespace helirec::hd;

HR_HD float glare_sample(const float* g, int rows, int cols, float inv_bin, int r, int c) {
    float y = sub(mul(add(static_cast<float>(r), 0.5f), inv_bin), 0.5f);
    float x = sub(mul(add(static_cast<float>(c), 0.5f), inv_bin), 0.5f);
    const float ymax = static_cast<float>(rows - 1), xmax = static_cast<float>(cols - 1);
    y = y < 0.f ? 0.f : (y > ymax ? ymax : y);
    x = x < 0.f ? 0.f : (x > xmax ? xmax : x);
    const int y0 = static_cast<int>(y), x0 = static_cast<int>(x);
    const int y1 = imin(y0 + 1, rows - 1), x1 = imin(x0 + 1, cols - 1);
    const float fy = sub(y, static_cast<float>(y0)), fx = sub(x, static_cast<float>(x0));
    const float* a = g + static_cast<std::size_t>(y0) * cols;
    const float* b = g + static_cast<std::size_t>(y1) * cols;
    const float top = add(mul(sub(1.f, fx), a[x0]), mul(fx, a[x1]));
    const float bot = add(mul(sub(1.f, fx), b[x0]), mul(fx, b[x1]));
    return add(mul(sub(1.f, fy), top), mul(fy, bot));
}

}  // namespace helirec::sld_core
