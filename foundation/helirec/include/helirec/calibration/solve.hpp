#pragma once
// Ceres fits in double precision on one thread. Equations use residual y - prediction;
// duplicate matrix entries add. Unused columns are fixed, dependent active columns
// are rejected: callers must choose their model's gauge explicitly.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "helirec/common.hpp"

namespace helirec::calibration {

struct SolveOptions {
    double huber = 0;  // scalar Huber threshold in residual units; 0: least squares
    int max_iterations = 100;
    double function_tolerance = 1e-7;
    double gradient_tolerance = 1e-12;
    double parameter_tolerance = 1e-12;
    double rcond = 1e-12;  // rank threshold on the column-normalised design
};

struct FitSummary {
    double cost = 0;  // Ceres convention: half the sum of robust squared residuals
    int iterations = 0;
    int evaluations = 0;
    bool usable = false;
    std::string termination = "NOT_RUN", message, solver_version;
};

struct LinearProblem {
    std::size_t unknowns = 0;
    std::vector<std::int64_t> row, col;
    std::vector<double> value, y;
    std::vector<std::int64_t> fixed;
    std::vector<double> fixed_value;
};

struct LinearSolution : FitSummary {
    std::vector<double> x, residual;
    std::vector<double> weight;  // sqrt(rho'(residual^2)) at the returned parameters
    double spread = 0;          // 1.4826 MAD, a diagnostic only
    int rank = 0;               // rank among active columns, excluding fixed columns
    int active = 0, unused = 0;
};

LinearSolution solve_linear(const LinearProblem& problem, const SolveOptions& options = {});

// Centred/scaled straight-line regression, also used by acquisition diagnostics.
struct LineFit : FitSummary {
    double slope = 0, intercept = 0, rms = 0, max_abs = 0;
    std::size_t n = 0;
};
LineFit fit_line(const std::vector<double>& x, const std::vector<double>& y, const SolveOptions& options = {});

}  // namespace helirec::calibration
