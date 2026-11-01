#pragma once
#include <cmath>
#ifdef __CUDACC__
#define VCR_EIGEN_HD __host__ __device__
#else
#define VCR_EIGEN_HD
#endif
namespace vcr::math {
// Cyclic Jacobi eigen-decomposition of a symmetric d x d matrix (d <= 3).
// Ascending values; vectors are columns, sign-normalized so the largest
// magnitude component is positive.
VCR_EIGEN_HD inline void symmetric_eigen(double* a, int d, double* values, double* vectors) {
    for (int i = 0; i < d * d; ++i) vectors[i] = i % (d + 1) == 0;
    for (int sweep = 0; sweep < 32; ++sweep) {
        double off = 0, total = 0;
        for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) { total += a[i * d + j] * a[i * d + j]; if (i != j) off += a[i * d + j] * a[i * d + j]; }
        if (off <= 1e-30 * total || off == 0) break;
        for (int p = 0; p < d - 1; ++p) for (int q = p + 1; q < d; ++q) {
            const double apq = a[p * d + q];
            if (apq == 0) continue;
            const double theta = (a[q * d + q] - a[p * d + p]) / (2 * apq);
            const double tt = (theta >= 0 ? 1. : -1.) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
            const double c = 1 / std::sqrt(tt * tt + 1), s = tt * c;
            for (int k = 0; k < d; ++k) {
                const double akp = a[k * d + p], akq = a[k * d + q];
                a[k * d + p] = c * akp - s * akq; a[k * d + q] = s * akp + c * akq;
            }
            for (int k = 0; k < d; ++k) {
                const double apk = a[p * d + k], aqk = a[q * d + k];
                a[p * d + k] = c * apk - s * aqk; a[q * d + k] = s * apk + c * aqk;
            }
            for (int k = 0; k < d; ++k) {
                const double vkp = vectors[k * d + p], vkq = vectors[k * d + q];
                vectors[k * d + p] = c * vkp - s * vkq; vectors[k * d + q] = s * vkp + c * vkq;
            }
        }
    }
    for (int i = 0; i < d; ++i) values[i] = a[i * d + i];
    for (int i = 0; i < d; ++i) for (int j = i + 1; j < d; ++j) if (values[j] < values[i]) {
        const double v = values[i]; values[i] = values[j]; values[j] = v;
        for (int k = 0; k < d; ++k) { const double x = vectors[k * d + i]; vectors[k * d + i] = vectors[k * d + j]; vectors[k * d + j] = x; }
    }
    for (int j = 0; j < d; ++j) {
        int largest = 0; for (int k = 1; k < d; ++k) if (std::fabs(vectors[k * d + j]) > std::fabs(vectors[largest * d + j])) largest = k;
        if (vectors[largest * d + j] < 0) for (int k = 0; k < d; ++k) vectors[k * d + j] = -vectors[k * d + j];
    }
}
}
#undef VCR_EIGEN_HD
