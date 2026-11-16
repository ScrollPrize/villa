#pragma once
#include <functional>
#include "helirec/calibration/solve.hpp"

namespace helirec::calibration::fitting {
// The evaluator retains its CPU/CUDA image buffers; only value, gradient and
// curvature cross this interface. The maximum is independently validated using
// the exact final Hessian, including when Ceres reports convergence.
using Correlation = std::function<void(const double*, double&, double*, double (*)[3])>;
struct CorrelationPeak {
    double at[3] = {}, value = 0, hessian[3][3] = {};
    bool maximum = false;
    FitSummary solver;
};
CorrelationPeak correlation_peak(const Correlation& evaluate, const bool active[3], const double start[3], double normalization);
}
