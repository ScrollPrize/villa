#include "calibration/correlation_peak.hpp"
#include "calibration/ceres_support.hpp"
#include <Eigen/Cholesky>

namespace helirec::calibration::fitting {
namespace {
class Objective final : public ceres::FirstOrderFunction {
  public:
    Correlation evaluate;
    double normalization;
    Eigen::Vector3d origin;
    Eigen::MatrixXd transform;
    mutable Eigen::Vector3d last = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    mutable double value = 0, gradient[3] = {}, hessian[3][3] = {};
    mutable int evaluations = 0;
    int NumParameters() const override { return static_cast<int>(transform.cols()); }
    void at(const Eigen::Vector3d& x) const {
        if (x == last) return;
        evaluate(x.data(), value, gradient, hessian);
        last = x;
        ++evaluations;
    }
    bool Evaluate(const double* q, double* cost, double* g) const override {
        const Eigen::Vector3d x = origin + transform * Eigen::Map<const Eigen::VectorXd>(q, NumParameters());
        at(x);
        *cost = -value / normalization;
        if (!std::isfinite(*cost)) return false;
        if (g) {
            Eigen::Map<Eigen::VectorXd> out(g, NumParameters());
            out = -transform.transpose() * Eigen::Map<const Eigen::Vector3d>(gradient) / normalization;
            if (!out.allFinite()) return false;
        }
        return true;
    }
};
// Stop in sample units, the same accuracy criterion used to validate a peak.
// Ceres still owns all steps and line searches; this callback only checks
// stationarity using derivatives already returned by the backend.
class PeakConvergence final : public ceres::IterationCallback {
    Objective& fn_;
    double* q_;
    std::vector<int> axes_;
  public:
    PeakConvergence(Objective& fn,double* q,const int* axes,int n) : fn_(fn),q_(q),axes_(axes,axes+n) {}
    ceres::CallbackReturnType operator()(const ceres::IterationSummary&) override {
        fn_.at(fn_.origin+fn_.transform*Eigen::Map<Eigen::VectorXd>(q_,axes_.size()));
        Eigen::MatrixXd h(axes_.size(),axes_.size());
        Eigen::VectorXd g(axes_.size());
        for (std::size_t i = 0; i < axes_.size(); ++i) {
            g[i] = fn_.gradient[axes_[i]];
            for (std::size_t j = 0; j < axes_.size(); ++j) h(i,j) = -fn_.hessian[axes_[i]][axes_[j]];
        }
        Eigen::LLT<Eigen::MatrixXd> factor(h);
        if (h.allFinite() && g.allFinite() && factor.info() == Eigen::Success) {
            const Eigen::VectorXd remaining = factor.solve(g);
            if (remaining.allFinite() && remaining.cwiseAbs().maxCoeff() < 1e-6) return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
        }
        return ceres::SOLVER_CONTINUE;
    }
};

}
CorrelationPeak correlation_peak(const Correlation& evaluate, const bool active[3], const double start[3], double normalization) {
    CorrelationPeak out;
    int axes[3], n = 0;
    for (int j = 0; j < 3; ++j) if (active[j]) axes[n++] = j;
    if (!n || !(normalization > 0) || !std::isfinite(normalization)) return out;
    auto* fn = new Objective;
    fn->evaluate = evaluate;
    fn->normalization = normalization;
    fn->origin = Eigen::Map<const Eigen::Vector3d>(start);
    fn->at(fn->origin);
    // A fixed change of coordinates gives BFGS comparable sensitivities on the
    // frame/row/column axes. No fitting step is taken outside Ceres.
    Eigen::MatrixXd curvature(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) curvature(i,j) = -fn->hessian[axes[i]][axes[j]] / normalization;
    Eigen::LLT<Eigen::MatrixXd> initial(curvature);
    Eigen::MatrixXd scale = Eigen::MatrixXd::Identity(n,n);
    if (curvature.allFinite() && initial.info() == Eigen::Success)
        scale = initial.matrixU().solve(scale).eval();
    fn->transform = Eigen::MatrixXd::Zero(3,n);
    for (int i = 0; i < n; ++i) fn->transform.row(axes[i]) = scale.row(i);
    ceres::GradientProblem problem(fn);  // owns fn
    double q[3] = {};
    ceres::GradientProblemSolver::Summary summary_;
    auto options = gradient_options();
    options.function_tolerance = 1e-12;
    options.gradient_tolerance = 0;
    options.update_state_every_iteration = true;
    PeakConvergence convergence(*fn,q,axes,n);
    options.callbacks.push_back(&convergence);
    ceres::Solve(options, problem, q, &summary_);
    out.solver = summary(summary_);
    const Eigen::Vector3d at = fn->origin + fn->transform * Eigen::Map<Eigen::VectorXd>(q,n);
    fn->at(at);
    out.solver.evaluations = fn->evaluations;
    out.value = fn->value;
    Eigen::VectorXd g(n);
    for (int i = 0; i < n; ++i) {
        g[i] = fn->gradient[axes[i]] / normalization;
        for (int j = 0; j < n; ++j) curvature(i,j) = -fn->hessian[axes[i]][axes[j]] / normalization;
    }
    Eigen::LLT<Eigen::MatrixXd> final(curvature);
    // Exact curvature and local stationarity in sample units, not a termination
    // flag alone. A flat or saddle direction cannot establish a measured peak.
    if (out.solver.usable && g.allFinite() && curvature.allFinite() && final.info() == Eigen::Success) {
        const Eigen::VectorXd remaining = final.solve(g);
        out.maximum = remaining.allFinite() && remaining.cwiseAbs().maxCoeff() < 1e-3;
    }
    for (int i = 0; i < 3; ++i) {
        out.at[i] = active[i] ? at[i] : 0;
        for (int j = 0; j < 3; ++j) out.hessian[i][j] = active[i] && active[j] ? fn->hessian[i][j] : 0;
    }
    return out;
}
}
