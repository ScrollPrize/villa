#pragma once
// Explicitly rounded float arithmetic shared by CPU and CUDA kernels.
//
// Kernels that must agree across backends (and with the nabuxx reference)
// spell out every fused multiply-add instead of relying on compiler
// contraction: CUDA uses the _rn intrinsics (never re-contracted by nvcc), the
// CPU uses std::fma with -ffp-contract=off (see CMakeLists.txt).
// The contraction pattern of nvcc for  a*b + c*d + e  is  fma(a, b, c*d) + e.
#include <cmath>

#if defined(__CUDACC__)
#define HR_HD __host__ __device__ __forceinline__
#else
#define HR_HD inline
#endif

namespace helirec::hd {

HR_HD float fma(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
HR_HD float mul(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
HR_HD float add(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
HR_HD float sub(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}
HR_HD float div(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
HR_HD float floor(float x) {
#if defined(__CUDA_ARCH__)
    return floorf(x);
#else
    return std::floor(x);
#endif
}
HR_HD bool is_nan(float x) {
#if defined(__CUDA_ARCH__)
    return isnan(x);
#else
    return std::isnan(x);
#endif
}
HR_HD int imin(int a, int b) { return a < b ? a : b; }
HR_HD int imax(int a, int b) { return a > b ? a : b; }

}  // namespace helirec::hd
