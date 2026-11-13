#pragma once
// Configuration and result handling shared by all model-specific Ceres fits.
#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/SparseCore>
#include <ceres/ceres.h>
#include <ceres/gradient_problem_solver.h>
#include <ceres/version.h>
#include "helirec/calibration/solve.hpp"

namespace helirec::calibration::fitting {
inline void validate(const SolveOptions& o) {
    if (o.max_iterations < 1 || !std::isfinite(o.huber) || o.huber < 0 ||
        !std::isfinite(o.rcond) || o.rcond <= 0 || o.rcond >= 1 ||
        !std::isfinite(o.function_tolerance) || o.function_tolerance < 0 ||
        !std::isfinite(o.gradient_tolerance) || o.gradient_tolerance < 0 ||
        !std::isfinite(o.parameter_tolerance) || o.parameter_tolerance < 0)
        throw Error("Ceres fit: invalid solver options");
}
// The rank of a design whose columns are normalised, at the relative threshold rcond (solve.cpp): what
// solve_linear and the geometry's solver both refuse dependent unknowns by. dependent: the columns the pivoting
// left out, in rising order (which of a dependent set they are is the pivoting's choice).
int design_rank(const Eigen::SparseMatrix<double>& design, double rcond, std::vector<int>* dependent = nullptr);
inline ceres::Solver::Options options(int active, const SolveOptions& o = {}) {
    validate(o);
    ceres::Solver::Options s;
    s.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    s.linear_solver_type = active <= 64 ? ceres::DENSE_QR : ceres::SPARSE_NORMAL_CHOLESKY;
    s.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    s.num_threads = 1;
    s.max_num_iterations = o.max_iterations;
    s.function_tolerance = o.function_tolerance;
    s.gradient_tolerance = o.gradient_tolerance;
    s.parameter_tolerance = o.parameter_tolerance;
    s.logging_type = ceres::SILENT;
    return s;
}
inline ceres::GradientProblemSolver::Options gradient_options(const SolveOptions& o = {}) {
    validate(o);
    ceres::GradientProblemSolver::Options s;
    s.line_search_direction_type = ceres::BFGS;
    s.line_search_type = ceres::WOLFE;
    s.max_num_iterations = o.max_iterations;
    s.function_tolerance = o.function_tolerance;
    s.gradient_tolerance = o.gradient_tolerance;
    s.parameter_tolerance = o.parameter_tolerance;
    s.logging_type = ceres::SILENT;
    return s;
}
inline FitSummary summary(const ceres::Solver::Summary& s) {
    FitSummary r;
    r.cost = s.final_cost;
    r.iterations = std::max(0, static_cast<int>(s.iterations.size()) - 1);
    r.evaluations = s.num_residual_evaluations + s.num_jacobian_evaluations;
    r.usable = s.IsSolutionUsable();
    r.termination = ceres::TerminationTypeToString(s.termination_type);
    r.message = s.message;
    r.solver_version = CERES_VERSION_STRING;
    return r;
}
inline FitSummary summary(const ceres::GradientProblemSolver::Summary& s) {
    FitSummary r;
    r.cost = s.final_cost;
    r.iterations = std::max(0, static_cast<int>(s.iterations.size()) - 1);
    r.evaluations = s.num_cost_evaluations + s.num_gradient_evaluations;
    r.usable = s.IsSolutionUsable();
    r.termination = ceres::TerminationTypeToString(s.termination_type);
    r.message = s.message;
    r.solver_version = CERES_VERSION_STRING;
    return r;
}
}  // namespace helirec::calibration::fitting
