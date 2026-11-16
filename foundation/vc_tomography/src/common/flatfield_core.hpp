#pragma once
// One flat-field corrected sample, shared by the CPU and CUDA kernels.
#include <cmath>

#include "hd_math.hpp"

namespace vc_tomography::flatfield_core {

// (raw - dark) / (flat - dark), held at or above `threshold`; NaN (0/0, a pixel without signal) becomes `threshold`.
// With take_log, the line integral -log of that.
HR_HD float value(float raw, float dark, float flat, float threshold, bool take_log) {
    float v = hd::div(hd::sub(raw, dark), hd::sub(flat, dark));
    v = v >= threshold ? v : threshold;  // false for NaN
#if defined(__CUDA_ARCH__)
    return take_log ? -logf(v) : v;
#else
    return take_log ? -std::log(v) : v;
#endif
}

}  // namespace vc_tomography::flatfield_core
