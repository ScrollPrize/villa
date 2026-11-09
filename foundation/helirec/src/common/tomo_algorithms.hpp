#pragma once
// SIRT, MLEM and gridrec written once against a backend B (CPU: src/cpu/tomo.cpp, CUDA:
// src/cuda/tomo.cu); per-element work is in HR_HD functors so both backends run the same arithmetic.
//
// Image: the whole N x N grid b.g.
//
// B provides:  g (grid geometry), n_angles, buf(slot, n) persistent float buffers,
//   ints(host vector) -> persistent index buffer,
//   project(x, sel, n_sel, epi)      for the rays k of the angles in sel: epi(k, A_k x)
//   len2(out)                        out[k] = ray norm of ray k over the disk (cached)
//   backproject(y, x, sel, n_sel, accumulate)   x (+)= sum_{a in sel} A_a^T y[a]
//   each(n, functor), fill, weighted_rows, gridrec(weighted rows, out N x N, args)
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "helirec/tomo.hpp"
#include "tomo_core.hpp"

namespace helirec::tomo_algos {

enum Slot { kMask, kTmp, kUpd, kWrows, kSd, kInitWeights };

// ---------------------------------------------------------------- per-ray functors and projector epilogues
struct MaskF {
    const float *w, *rows; float* m; bool confidence;
    HR_HD void operator()(std::size_t i) const {
        m[i] = w[i] > 0.f && hd::is_finite(w[i]) && hd::is_finite(rows[i]) ? (confidence ? w[i] : 1.f) : 0.f;
    }
};
struct InitialCopyF {
    const float *rows, *w; float *out, *iw;
    HR_HD void operator()(std::size_t i) const {
        const bool valid = w[i] > 0.f && hd::is_finite(w[i]) && hd::is_finite(rows[i]);
        out[i] = valid ? rows[i] : 0.f;
        iw[i] = valid ? w[i] : 0.f;
    }
};
struct SirtEpi {  // relax (b - Ax) / norm_i
    const float *b, *len2, *m; float* out; float relax;
    HR_HD void operator()(std::size_t k, float v) const {
        out[k] = (len2[k] != 0.f && m[k] != 0.f) ? relax * m[k] * (b[k] - v) / len2[k] : 0.f;
    }
};
struct EmEpi {  // b / Ax; EM assumes b >= 0, Ax > 0: rays with b <= 0 or Ax <= 0 give 0
    const float *b, *m; float* out;
    HR_HD void operator()(std::size_t k, float v) const {
        out[k] = (m[k] != 0.f && v > 0.f && b[k] > 0.f) ? b[k] / v : 0.f;
    }
};
struct StoreEpi {
    float* out;
    HR_HD void operator()(std::size_t k, float v) const { out[k] = v; }
};
struct ClampMinF {  // EM start: a positive image
    float* x; float lo;
    HR_HD void operator()(std::size_t i) const { x[i] = x[i] > lo ? x[i] : lo; }
};

// ---------------------------------------------------------------- per-pixel functors
struct SirtApplyF {  // x += upd / sum_dist
    float* x; const float *upd, *sd;
    HR_HD void operator()(std::size_t i) const { if (sd[i] != 0.f) x[i] += upd[i] / sd[i]; }
};
struct EmApplyF {  // x *= upd / sum_dist
    float* x; const float *upd, *sd;
    HR_HD void operator()(std::size_t i) const { if (sd[i] != 0.f) x[i] *= upd[i] / sd[i]; }
};

template <class B>
void run(B& b, const TomoArgs& a, const float* rows, const float* weights, float* x) {
    const int NG = b.n_angles, N = b.g.N;
    const std::size_t NS = static_cast<std::size_t>(NG) * b.g.n_det, NP = static_cast<std::size_t>(N) * N;
    const std::string& alg = a.algorithm;
    if (alg == "sirt" && (a.norm != 2 || !std::isfinite(a.relax) || a.relax <= 0.f || a.relax >= 2.f))
        throw Error("sirt requires norm=rowsum and finite 0 < relax < 2");
    b.pj = a.projector;
    b.norm = a.norm;
    std::vector<int32_t> all(NG);
    for (int i = 0; i < NG; ++i) all[i] = i;
    const int32_t* sel = b.ints(all);

    if (alg == "gridrec" || a.init == "gridrec") {
        float* wr = b.buf(kWrows, NS);
        if (alg == "gridrec") b.weighted_rows(rows, weights, wr, NS);
        else b.initial_rows(rows, weights, wr, b.buf(kInitWeights, NS), NS);
        TomoArgs g = a;
        if (alg != "gridrec") g.filter_name = "ramlak";
        b.gridrec(wr, x, g);
        if (alg == "gridrec") return;
    } else {
        b.fill(x, a.init_value, NP);
    }
    float* m = b.buf(kMask, NS);
    float* tmp = b.buf(kTmp, NS);
    b.each(NS, MaskF{weights, rows, m, a.confidence_weights});
    b.fill(tmp, 0.f, NS);
    const bool sirt = alg == "sirt";
    if (!sirt && alg != "mlem") throw Error("unknown method '" + alg + "'");
    const float* len2 = sirt ? b.len2() : nullptr;
    if (!sirt) b.each(NP, ClampMinF{x, a.init_value > 0.f ? a.init_value : 1e-6f});
    float* upd = b.buf(kUpd, NP);
    float* sd = b.buf(kSd, NP);
    b.backproject(m, sd, sel, NG, false);   // A^T m: column sums of the measured rays, fixed over the iterations
    for (int it = 0; it < a.num_iter; ++it) {
        if (sirt) b.project(x, sel, NG, SirtEpi{rows, len2, m, tmp, a.relax});
        else b.project(x, sel, NG, EmEpi{rows, m, tmp});
        b.backproject(tmp, upd, sel, NG, false);
        if (sirt) b.each(NP, SirtApplyF{x, upd, sd});
        else b.each(NP, EmApplyF{x, upd, sd});
    }
}

}  // namespace helirec::tomo_algos
