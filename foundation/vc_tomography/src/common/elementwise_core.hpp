#pragma once
// Elementwise operations on float arrays, written once for the CPU and CUDA kernels: each is a functor of the
// element index, run by cpu::each (cpu/each.hpp) or cuda::each (cuda_utils.cuh). Every operation is rounded on its
// own (hd::), so both backends give the same values.
#include <cstddef>
#include <cstdint>

#include "hd_math.hpp"
#include "sino_core.hpp"

namespace vc_tomography::elementwise {

struct Fill {
    float* x; float v;
    HR_HD void operator()(std::size_t i) const { x[i] = v; }
};
struct MinusLog {   // -log of x clamped to [lo, hi] (sino_core::minus_log)
    float* x; float lo, hi;
    HR_HD void operator()(std::size_t i) const { x[i] = sino_core::minus_log(x[i], lo, hi); }
};
struct Multiply {   // x *= w
    float* x; const float* w;
    HR_HD void operator()(std::size_t i) const { x[i] = hd::mul(x[i], w[i]); }
};
struct WeightedRow {    // a zero detector weight masks even a nonfinite sample
    const float *rows, *weights; float* out;
    HR_HD void operator()(std::size_t i) const { out[i] = weights[i] == 0.f ? 0.f : hd::mul(rows[i], weights[i]); }
};
struct AddScaled {  // x += s y
    float* x; const float* y; float s;
    HR_HD void operator()(std::size_t i) const { x[i] = hd::add(x[i], hd::mul(s, y[i])); }
};
struct WeightDifference {   // t = x (w2 - w)
    const float *x, *w, *w2; float* t;
    HR_HD void operator()(std::size_t i) const { t[i] = hd::mul(x[i], hd::sub(w2[i], w[i])); }
};
struct MultiplyAdd {        // x = x w + t
    float* x; const float *w, *t;
    HR_HD void operator()(std::size_t i) const { x[i] = hd::add(hd::mul(x[i], w[i]), t[i]); }
};
struct AddInto {            // r += a
    float* r; const float* a;
    HR_HD void operator()(std::size_t i) const { r[i] = hd::add(r[i], a[i]); }
};
struct Divide {             // r /= rw where rw > 0, else 0
    float* r; const float* rw;
    HR_HD void operator()(std::size_t i) const { r[i] = rw[i] > 0 ? hd::div(r[i], rw[i]) : 0.f; }
};

// ---- the tiles of a mosaic in one sinogram (vc_tomography/mosaic.hpp)
struct TileRows {           // a tile's rows for the canvas: its weight times a gain per column, and its value times that weight
    const float *s, *w, *gain; int n, stride; float *out_s, *out_w;
    HR_HD void operator()(std::size_t i) const {
        const std::size_t r = i / static_cast<std::size_t>(n), c = i % static_cast<std::size_t>(n), o = r * static_cast<std::size_t>(stride) + c;
        const float g = hd::mul(w[i], gain[c]);
        out_w[o] = g;
        out_s[o] = g > 0.f ? hd::mul(s[i], g) : 0.f;   // (a value without weight need not be finite)
    }
};
struct PlaceRows {          // canvas[r, first[r] + c] += scale x rows[r, c]; what falls outside the canvas is dropped
    const float* rows; const std::int32_t* first; float scale; bool nonneg; int n, stride, width; float* canvas;
    HR_HD void operator()(std::size_t i) const {
        const std::size_t r = i / static_cast<std::size_t>(n);
        const int c = static_cast<int>(i % static_cast<std::size_t>(n)), j = first[r] + c;
        if (j < 0 || j >= width) return;
        float v = rows[r * static_cast<std::size_t>(stride) + c];
        if (nonneg && !(v > 0.f)) return;
        const std::size_t o = r * static_cast<std::size_t>(width) + j;
        canvas[o] = hd::add(canvas[o], hd::mul(scale, v));
    }
};
struct FoldOpposite {       // a ray's two recordings, S[r, j] and S[r + half_rows, width - 1 - j], both become their sum
    float* S; int half_rows, width;
    HR_HD void operator()(std::size_t i) const {
        const std::size_t r = i / static_cast<std::size_t>(width), j = i % static_cast<std::size_t>(width);
        float& a = S[r * static_cast<std::size_t>(width) + j];
        float& b = S[(r + half_rows) * static_cast<std::size_t>(width) + (width - 1 - j)];
        const float t = hd::add(a, b);
        a = t, b = t;
    }
};

}  // namespace vc_tomography::elementwise
