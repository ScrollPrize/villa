#pragma once
// One bin of a spectrum times a Fourier filter's, shared by the CPU (FFTW) and CUDA (cuFFT) filters: real (fi
// absent) or complex, every product rounded alone.
#include "hd_math.hpp"

namespace vc_tomography::fourier_core {

// (re, im) times fr, or times (fr + i fi) when `complex`
HR_HD void filter_bin(float& re, float& im, float fr, float fi, bool complex) {
    using namespace hd;
    if (!complex) {
        re = mul(re, fr);
        im = mul(im, fr);
        return;
    }
    const float r = re, i = im;
    re = sub(mul(r, fr), mul(i, fi));
    im = add(mul(r, fi), mul(i, fr));
}

}  // namespace vc_tomography::fourier_core
