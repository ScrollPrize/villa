#include "vc/fiber_tracer/FiberStripNormals.hpp"

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

cv::Vec3d unit(const cv::Vec3d& v)
{
    const double n = cv::norm(v);
    return n > 0.0 ? v / n : cv::Vec3d(0, 0, 0);
}

cv::Vec3d perpendicular(const cv::Vec3d& v, const cv::Vec3d& t)
{
    return v - t * v.dot(t);
}

// sqrt(w_t) (theta_b - theta_a)
struct TwistResidual {
    double weight;
    template <typename T>
    bool operator()(const T* a, const T* b, T* residual) const
    {
        residual[0] = T(weight) * (b[0] - a[0]);
        return true;
    }
};

// sqrt(w_l) <l(theta), delta t> with l = -sin(theta) u + cos(theta) v and
// delta t expressed in the vertex frame as (du, dv).
struct LateralResidual {
    double weight, du, dv;
    template <typename T>
    bool operator()(const T* theta, T* residual) const
    {
        residual[0] = T(weight) * (-ceres::sin(theta[0]) * T(du) + ceres::cos(theta[0]) * T(dv));
        return true;
    }
};

// sqrt(w_a w_i) (n(theta) - a) with the oriented axis in the vertex frame
// (au, av, at) and n = (cos, sin, 0). The axis was oriented by the
// geometry-only field beforehand, so the term has one minimum per node.
struct AxisResidual {
    double weight, au, av, at;
    template <typename T>
    bool operator()(const T* theta, T* residual) const
    {
        const T c = ceres::cos(theta[0]), s = ceres::sin(theta[0]);
        residual[0] = T(weight) * (c - T(au));
        residual[1] = T(weight) * (s - T(av));
        residual[2] = T(weight) * (-T(at));
        return true;
    }
};

}  // namespace

namespace
{

std::vector<double> arcLengths(const std::vector<cv::Vec3d>& points)
{
    std::vector<double> arc(points.size(), 0.0);
    for (size_t i = 1; i < points.size(); ++i) arc[i] = arc[i - 1] + cv::norm(points[i] - points[i - 1]);
    return arc;
}

// Gaussian smoothing of the positions along the arc (weights by vertex arc
// spacing so uneven tracing does not bias the average), truncated at 3 sigma.
std::vector<cv::Vec3d> smoothAlongArc(const std::vector<cv::Vec3d>& points, const std::vector<double>& arc, double sigma)
{
    if (!(sigma > 0.0)) return points;
    const size_t n = points.size();
    std::vector<double> span(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const double before = i ? arc[i] - arc[i - 1] : 0.0, after = i + 1 < n ? arc[i + 1] - arc[i] : 0.0;
        span[i] = 0.5 * (before + after);
        if (!(span[i] > 0.0)) span[i] = 1e-6;
    }
    std::vector<cv::Vec3d> smoothed(n);
    const double reach = 3.0 * sigma, inv = -0.5 / (sigma * sigma);
    for (size_t i = 0; i < n; ++i) {
        cv::Vec3d sum(0, 0, 0);
        double weight = 0.0;
        for (size_t j = i; j < n && arc[j] - arc[i] <= reach; ++j) {
            const double d = arc[j] - arc[i], w = span[j] * std::exp(d * d * inv);
            sum += points[j] * w; weight += w;
        }
        for (size_t j = i; j-- > 0 && arc[i] - arc[j] <= reach;) {
            const double d = arc[i] - arc[j], w = span[j] * std::exp(d * d * inv);
            sum += points[j] * w; weight += w;
        }
        smoothed[i] = weight > 0.0 ? sum / weight : points[i];
    }
    return smoothed;
}

cv::Vec3d at(const std::vector<cv::Vec3d>& points, const std::vector<double>& arc, double s)
{
    if (s <= arc.front()) return points.front();
    if (s >= arc.back()) return points.back();
    const auto it = std::upper_bound(arc.begin(), arc.end(), s);
    const size_t j = static_cast<size_t>(it - arc.begin());
    const double span = arc[j] - arc[j - 1], f = span > 0.0 ? (s - arc[j - 1]) / span : 0.0;
    return points[j - 1] + (points[j] - points[j - 1]) * f;
}

FiberStripNormalResult solveChain(
    const std::vector<cv::Vec3d>& points, const FiberStripNormalOptions& options,
    const std::vector<FiberStripAxisTarget>* targets);

void validateOptions(const FiberStripNormalOptions& options)
{
    if (options.twistWeight < 0.0 || options.lateralWeight < 0.0 || options.axisWeight < 0.0 ||
        options.maxIterations < 1 || options.smoothingSigma < 0.0 || options.resampleSpacing < 0.0)
        throw std::invalid_argument("invalid fiber strip normal options");
}

}  // namespace

FiberStripChain fiberStripChain(const std::vector<cv::Vec3d>& points, const FiberStripNormalOptions& options)
{
    validateOptions(options);
    FiberStripChain chain;
    if (points.size() < 3) return chain;
    chain.vertexArc = arcLengths(points);
    const auto smoothed = smoothAlongArc(points, chain.vertexArc, options.smoothingSigma);
    if (options.resampleSpacing > 0.0) {
        for (double s = 0.0; s < chain.vertexArc.back(); s += options.resampleSpacing) {
            chain.points.push_back(at(smoothed, chain.vertexArc, s)); chain.arc.push_back(s);
        }
        chain.points.push_back(smoothed.back()); chain.arc.push_back(chain.vertexArc.back());
    } else { chain.points = smoothed; chain.arc = chain.vertexArc; }
    return chain;
}

FiberStripNormalResult solveFiberStripNormals(
    const std::vector<cv::Vec3d>& points, const FiberStripNormalOptions& options,
    const std::vector<FiberStripAxisTarget>* targets)
{
    FiberStripNormalResult result;
    const auto chainData = fiberStripChain(points, options);
    const auto& chain = chainData.points;
    const auto& chainArc = chainData.arc;
    const auto& arc = chainData.vertexArc;
    if (chain.size() < 3) return result;
    if (targets && targets->size() != chain.size())
        throw std::invalid_argument("fiber strip axis targets must match the chain points");
    auto solved = solveChain(chain, options, targets);
    if (solved.normals.empty()) return result;
    // Normals back on the original vertices: unit interpolation along the arc.
    result = std::move(solved);
    std::vector<cv::Vec3d> normals(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        const auto it = std::upper_bound(chainArc.begin(), chainArc.end(), arc[i]);
        const size_t j = std::clamp<size_t>(static_cast<size_t>(it - chainArc.begin()), 1, chainArc.size() - 1);
        const double span = chainArc[j] - chainArc[j - 1], f = span > 0.0 ? std::clamp((arc[i] - chainArc[j - 1]) / span, 0.0, 1.0) : 0.0;
        cv::Vec3d n = result.normals[j - 1] * (1.0 - f) + result.normals[j] * f;
        normals[i] = cv::norm(n) > 1e-9 ? unit(n) : result.normals[f < 0.5 ? j - 1 : j];
    }
    result.normals = std::move(normals);
    return result;
}

namespace
{

FiberStripNormalResult solveChain(
    const std::vector<cv::Vec3d>& points, const FiberStripNormalOptions& options,
    const std::vector<FiberStripAxisTarget>* targets)
{
    FiberStripNormalResult result;
    const size_t n = points.size();
    if (n < 3) return result;
    // Segment directions; a zero-length segment repeats its predecessor.
    std::vector<cv::Vec3d> segment(n - 1);
    for (size_t s = 0; s + 1 < n; ++s) {
        segment[s] = unit(points[s + 1] - points[s]);
        if (cv::norm(segment[s]) == 0.0 && s > 0) segment[s] = segment[s - 1];
    }
    if (cv::norm(segment[0]) == 0.0) {
        for (size_t s = 1; s + 1 < n; ++s)
            if (cv::norm(segment[s]) > 0.0) { segment[0] = segment[s]; break; }
        if (cv::norm(segment[0]) == 0.0) return result;
    }
    // Vertex tangent: bisector of the adjacent segments; the ends use their segment.
    std::vector<cv::Vec3d> tangent(n);
    for (size_t i = 0; i < n; ++i) {
        if (i == 0) tangent[i] = segment[0];
        else if (i + 1 == n) tangent[i] = segment[n - 2];
        else tangent[i] = unit(segment[i - 1] + segment[i]);
        if (cv::norm(tangent[i]) == 0.0) tangent[i] = i ? tangent[i - 1] : segment[0];
    }
    // Parallel-transported reference frame (projection update).
    std::vector<cv::Vec3d> u(n), v(n);
    cv::Vec3d seed = perpendicular(cv::Vec3d(0, 0, 1), tangent[0]);
    if (cv::norm(seed) < 1e-9) seed = perpendicular(cv::Vec3d(1, 0, 0), tangent[0]);
    if (cv::norm(seed) < 1e-9) seed = perpendicular(cv::Vec3d(0, 1, 0), tangent[0]);
    u[0] = unit(seed);
    v[0] = tangent[0].cross(u[0]);
    for (size_t i = 1; i < n; ++i) {
        cv::Vec3d next = perpendicular(u[i - 1], tangent[i]);
        if (cv::norm(next) < 1e-9) next = perpendicular(v[i - 1], tangent[i]);
        u[i] = unit(next);
        v[i] = tangent[i].cross(u[i]);
    }
    // Tangent turning at interior vertices, in the vertex frame.
    std::vector<double> du(n, 0.0), dv(n, 0.0);
    for (size_t i = 1; i + 1 < n; ++i) {
        const cv::Vec3d delta = segment[i] - segment[i - 1];
        du[i] = delta.dot(u[i]);
        dv[i] = delta.dot(v[i]);
    }
    // Initial angles: the turning direction where the fiber turns, carried along
    // the chain by the representative nearest to the previous vertex (period pi).
    std::vector<double> theta(n, 0.0);
    double previous = 0.0;
    bool started = false;
    for (size_t i = 0; i < n; ++i) {
        const double magnitude = std::hypot(du[i], dv[i]);
        double target = magnitude > 0.0 ? std::atan2(dv[i], du[i]) : previous;
        if (!started && magnitude > 0.0) { started = true; previous = target; }
        while (target - previous > kPi / 2) target -= kPi;
        while (target - previous < -kPi / 2) target += kPi;
        theta[i] = target;
        previous = target;
    }
    const double twist = std::sqrt(options.twistWeight), lateral = std::sqrt(options.lateralWeight);
    ceres::Solver::Options solver;
    solver.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver.max_num_iterations = options.maxIterations;
    solver.num_threads = 1;
    solver.logging_type = ceres::SILENT;
    // Pass 1: geometry only. Pass 2 (with targets): the geometry field orients
    // every axis target, then the signed axis term pulls the normals from there.
    const bool fuse = targets && options.axisWeight > 0.0;
    ceres::Solver::Summary summary;
    for (int pass = 0; pass < (fuse ? 2 : 1); ++pass) {
        ceres::Problem problem;
        for (size_t i = 0; i < n; ++i) problem.AddParameterBlock(&theta[i], 1);
        for (size_t i = 0; i + 1 < n; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<TwistResidual, 1, 1, 1>(new TwistResidual{twist}),
                nullptr, &theta[i], &theta[i + 1]);
        }
        for (size_t i = 1; i + 1 < n; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<LateralResidual, 1, 1>(new LateralResidual{lateral, du[i], dv[i]}),
                nullptr, &theta[i]);
        }
        if (pass == 1) {
            for (size_t i = 0; i < n; ++i) {
                const auto& target = (*targets)[i];
                cv::Vec3d a = unit(target.axis);
                if (!(target.weight > 0.0) || cv::norm(a) == 0.0) continue;
                const cv::Vec3d current = u[i] * std::cos(theta[i]) + v[i] * std::sin(theta[i]);
                if (current.dot(a) < 0.0) a = -a;
                const double weight = std::sqrt(options.axisWeight * target.weight);
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<AxisResidual, 3, 1>(
                        new AxisResidual{weight, a.dot(u[i]), a.dot(v[i]), a.dot(tangent[i])}),
                    nullptr, &theta[i]);
            }
        }
        ceres::Solve(solver, &problem, &summary);
        if (pass == 0) result.initialCost = summary.initial_cost;
    }
    result.finalCost = summary.final_cost;
    result.iterations = static_cast<int>(summary.iterations.size());
    result.converged = summary.termination_type == ceres::CONVERGENCE ||
                       summary.termination_type == ceres::NO_CONVERGENCE;
    result.normals.resize(n);
    for (size_t i = 0; i < n; ++i)
        result.normals[i] = unit(u[i] * std::cos(theta[i]) + v[i] * std::sin(theta[i]));
    return result;
}

}  // namespace

std::vector<FiberStripNormalResult> solveFiberStripNormals(
    const std::vector<std::vector<cv::Vec3d>>& fibers,
    const FiberStripNormalOptions& options, int threads,
    const std::vector<std::vector<FiberStripAxisTarget>>* targets)
{
    if (targets && targets->size() != fibers.size())
        throw std::invalid_argument("fiber strip axis targets must match the fibers");
    std::vector<FiberStripNormalResult> results(fibers.size());
    cv::parallel_for_(cv::Range(0, static_cast<int>(fibers.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const auto f = static_cast<size_t>(i);
            results[f] = solveFiberStripNormals(fibers[f], options, targets ? &(*targets)[f] : nullptr);
        }
    }, std::max(1, threads));
    return results;
}

}  // namespace vc::fiber_tracer
