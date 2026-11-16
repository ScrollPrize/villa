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
// Before a loop over independent elements with one operation each: lets the CPU's compiler use its vector unit
// (element by element the operations and their order stay the same; nothing where OpenMP is off, or under CUDA).
#if defined(_OPENMP) && !defined(__CUDACC__)
#define HR_SIMD _Pragma("omp simd")
#else
#define HR_SIMD
#endif

namespace helirec::hd {

HR_HD float fma(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
// a * b + c as the compilers evaluate the plain expression: contracted by nvcc (one rounding), two rounded
// operations on the CPU (-ffp-contract=off). For kernels whose results are pinned per backend rather than across them.
HR_HD float muladd(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return a * b + c;
#endif
}
// x / 6, correctly rounded for x = 0 and x >= 2.4e-38 (checked over every float); CUDA: without the division
HR_HD float div6(float x) {
#if defined(__CUDA_ARCH__)
    const float r = 0.16666667163372039795f, q = __fmul_rn(x, r);
    return __fmaf_rn(__fmaf_rn(-6.f, q, x), r, q);
#else
    return x / 6.f;
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
// the same in double precision (positions that single precision cannot hold: detector_map_core.hpp, sino_core.hpp)
HR_HD double dadd(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a, b);
#else
    return a + b;
#endif
}
HR_HD double dsub(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dsub_rn(a, b);
#else
    return a - b;
#endif
}
HR_HD double dmul(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a, b);
#else
    return a * b;
#endif
}
HR_HD double ddiv(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __ddiv_rn(a, b);
#else
    return a / b;
#endif
}
// a * b + c in double precision, as muladd: one rounding under CUDA (whose consumer devices are slow in double
// precision), two on the CPU. The two differ by a rounding of double precision: for sums that end in single precision.
HR_HD double dmuladd(double a, double b, double c) {
#if defined(__CUDA_ARCH__)
    return __fma_rn(a, b, c);
#else
    return a * b + c;
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
HR_HD bool is_finite(float x) {
#if defined(__CUDA_ARCH__)
    return isfinite(x);
#else
    return std::isfinite(x);
#endif
}
HR_HD int imin(int a, int b) { return a < b ? a : b; }
HR_HD int imax(int a, int b) { return a > b ? a : b; }

}  // namespace helirec::hd
