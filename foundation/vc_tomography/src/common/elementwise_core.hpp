#pragma once
// Elementwise operations on float arrays, written once for the CPU and CUDA kernels: each is a functor of the
// element index, run by cpu::each (cpu/each.hpp) or cuda::each (cuda_utils.cuh). Every operation is rounded on its
// own (hd::), so both backends give the same values.
#include <cstddef>

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

}  // namespace vc_tomography::elementwise
