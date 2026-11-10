#include <algorithm>
#include <cmath>
#include <cstring>

#include "common/output_core.hpp"
#include "helirec/kernels.hpp"

namespace helirec::cpu {

void to_uint8(const float* src, std::size_t src_pitch, std::uint8_t* out, int ny, int nx, const Uint8Window& w) {
    if (!(w.hi > w.lo)) throw Error("to_uint8: the window needs hi > lo");
    if (w.mask && (w.mask_ny < 1 || w.mask_nx < 1 || w.mask_fy < 1 || w.mask_fx < 1)) throw Error("to_uint8: empty mask");
    const float lo = static_cast<float>(w.lo), range = static_cast<float>(w.hi - w.lo);
    const int n = w.disk_n;
#pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y) {
        const float* s = src + static_cast<std::size_t>(y) * src_pitch;
        std::uint8_t* o = out + static_cast<std::size_t>(y) * nx;
        // the row's pixels inside the disk: [a, b)
        int a = 0, b = nx;
        if (n > 0) {
            a = 0;
            while (a < nx && output_core::outside_disk(w.x0 + a, w.y0 + y, n)) ++a;
            b = nx;
            while (b > a && output_core::outside_disk(w.x0 + b - 1, w.y0 + y, n)) --b;
            std::memset(o, 0, static_cast<std::size_t>(a));
            std::memset(o + b, 0, static_cast<std::size_t>(nx - b));
        }
        if (w.mask) {
            // runs of pixels sharing a mask element
            const std::uint8_t* k = w.mask + static_cast<std::size_t>(output_core::mask_index(y, w.mask_oy, w.mask_fy, w.mask_ny)) * w.mask_nx;
            for (int x = a; x < b;) {
                const int j = output_core::mask_index(x, w.mask_ox, w.mask_fx, w.mask_nx);
                const int end = j == w.mask_nx - 1 ? b : std::min(b, (j + 1) * w.mask_fx - w.mask_ox);
                if (k[j]) {
                    for (; x < end; ++x) o[x] = output_core::window_u8(s[x], lo, range);
                } else {
                    std::memset(o + x, 0, static_cast<std::size_t>(end - x));
                    x = end;
                }
            }
        } else {
            for (int x = a; x < b; ++x) o[x] = output_core::window_u8(s[x], lo, range);
        }
    }
}

}  // namespace helirec::cpu
