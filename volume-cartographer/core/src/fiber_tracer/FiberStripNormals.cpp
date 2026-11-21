#include "vc/fiber_tracer/FiberStripNormals.hpp"

#include <atomic>

#include <ceres/ceres.h>

#include <algorithm>
#include <limits>
#include <array>
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
// (au, av, at) and n = (cos, sin, 0). The axis was oriented beforehand (by
// continuity along the chain), so the term has one minimum per node.
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
    // Pass 1: geometry only. Pass 2 (with targets): the axis targets are oriented
    // by continuity along the chain, the geometry field decides the fiber's
    // overall facing, then the signed axis term pulls the normals from there.
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
            // Fixed (signed) targets decide the fiber's gauge: flip the geometry
            // field when their weighted majority points against it.
            double vote = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const auto& target = (*targets)[i];
                if (!target.fixed || !(target.weight > 0.0)) continue;
                const cv::Vec3d a = unit(target.axis);
                if (cv::norm(a) == 0.0) continue;
                vote += target.weight * a.dot(u[i] * std::cos(theta[i]) + v[i] * std::sin(theta[i]));
            }
            if (vote < 0.0) for (size_t i = 0; i < n; ++i) theta[i] += kPi;
            // The axis targets carry no sign. It is not taken from the geometry
            // field at each node - that field is only the bending direction,
            // and where it drifts past a right angle from the sheet it hands the
            // two sides of the fiber opposite signs, a flip the prediction does
            // not have. Instead the same cost is first minimised node by node
            // from one end: each node's angle against the previous node held
            // fixed (twist), its own lateral term and the pull to its axis,
            // signed against the previous node's normal. The twist weight
            // keeps that normal smooth through a noisy axis, so every target
            // gets the sign of a normal that never turned over. The full solve
            // below then refines all nodes together from there.
            std::vector<cv::Vec3d> oriented(n, cv::Vec3d(0, 0, 0));
            std::vector<char> has(n, 0);
            {
                constexpr int kSteps = 720;  // half-degree candidates, then a parabolic refinement
                static const auto table = [] {
                    std::array<std::array<double, 2>, kSteps> out{};
                    for (int k = 0; k < kSteps; ++k) {
                        const double angle = -kPi + 2 * kPi * k / kSteps;
                        out[size_t(k)] = {std::cos(angle), std::sin(angle)};
                    }
                    return out;
                }();
                const double twistWeight = options.twistWeight, lateralWeight = options.lateralWeight;
                std::vector<double> stepped(n, theta[0]);
                const auto anchorAngle = [&](size_t i, double& angle) {
                    const auto& target = (*targets)[i];
                    const cv::Vec3d a = unit(target.axis);
                    if (!target.fixed || !(target.weight > 0.0) || cv::norm(a) == 0.0) return false;
                    const double au = a.dot(u[i]), av = a.dot(v[i]);
                    if (std::hypot(au, av) < 1e-9) return false;
                    angle = std::atan2(av, au);
                    return true;
                };
                double first = 0.0;
                bool anchorSeen = false;
                if (anchorAngle(0, first)) { stepped[0] = first; anchorSeen = true; }
                for (size_t i = 1; i < n; ++i) {
                    const double before = stepped[i - 1];
                    double fixedAngle = 0.0;
                    if (anchorAngle(i, fixedAngle)) {
                        // An anchor is a boundary condition: its own angle, in
                        // the turn nearest the previous node. The first anchor
                        // also decides the facing of the nodes before it.
                        double previous = before;
                        if (!anchorSeen && std::cos(fixedAngle - before) < 0.0) {
                            for (size_t j = 0; j < i; ++j) stepped[j] += kPi;
                            previous += kPi;
                        }
                        anchorSeen = true;
                        while (fixedAngle - previous > kPi) fixedAngle -= 2 * kPi;
                        while (fixedAngle - previous < -kPi) fixedAngle += 2 * kPi;
                        stepped[i] = fixedAngle;
                        continue;
                    }
                    const auto& target = (*targets)[i];
                    cv::Vec3d a = unit(target.axis);
                    const bool pulled = !target.fixed && target.weight > 0.0 && cv::norm(a) > 0.0;
                    double au = 0.0, av = 0.0, axisWeight = 0.0;
                    if (pulled) {
                        const cv::Vec3d previous = u[i] * std::cos(before) + v[i] * std::sin(before);
                        if (previous.dot(a) < 0.0) a = -a;
                        au = a.dot(u[i]); av = a.dot(v[i]);
                        axisWeight = options.axisWeight * target.weight;
                    }
                    const bool interior = i + 1 < n;
                    const double cb = std::cos(before), sb = std::sin(before);
                    const auto cost = [&](double c, double sn, double delta) {
                        // c, sn: cos and sin of the candidate angle before + delta.
                        double value = twistWeight * delta * delta;
                        if (interior) { const double l = -sn * du[i] + c * dv[i]; value += lateralWeight * l * l; }
                        if (pulled) value += axisWeight * ((c - au) * (c - au) + (sn - av) * (sn - av));
                        return value;
                    };
                    int best = 0; double bestCost = std::numeric_limits<double>::infinity();
                    std::array<double, kSteps> costs;
                    for (int k = 0; k < kSteps; ++k) {
                        const double cd = table[size_t(k)][0], sd = table[size_t(k)][1];
                        costs[size_t(k)] = cost(cb * cd - sb * sd, sb * cd + cb * sd, -kPi + 2 * kPi * k / kSteps);
                        if (costs[size_t(k)] < bestCost) { bestCost = costs[size_t(k)]; best = k; }
                    }
                    double delta = -kPi + 2 * kPi * best / kSteps;
                    if (best > 0 && best + 1 < kSteps) {
                        const double left = costs[size_t(best - 1)], right = costs[size_t(best + 1)];
                        const double curve = left - 2 * bestCost + right;
                        if (curve > 0.0) delta += 0.5 * (left - right) / curve * (2 * kPi / kSteps);
                    }
                    stepped[i] = before + delta;
                }
                // Which way the whole fiber faces: the signed anchors when there
                // are any (the gauge flip above already sided the geometry
                // field with them), otherwise the geometry field's majority.
                double fieldVote = 0.0;
                bool anchored = false;
                for (size_t i = 0; i < n; ++i) {
                    const auto& target = (*targets)[i];
                    if (target.fixed && target.weight > 0.0) anchored = true;
                    if (!(target.weight > 0.0)) continue;
                    fieldVote += target.weight * std::cos(stepped[i] - theta[i]);
                }
                if (!anchored && fieldVote < 0.0) for (auto& angle : stepped) angle += kPi;
                for (size_t i = 0; i < n; ++i) {
                    const auto& target = (*targets)[i];
                    cv::Vec3d a = unit(target.axis);
                    if (!target.fixed && target.weight > 0.0 && cv::norm(a) > 0.0) {
                        const cv::Vec3d normal = u[i] * std::cos(stepped[i]) + v[i] * std::sin(stepped[i]);
                        oriented[i] = normal.dot(a) < 0.0 ? -a : a;
                        has[i] = 1;
                    }
                    theta[i] = stepped[i];
                }
            }
            for (size_t i = 0; i < n; ++i) {
                const auto& target = (*targets)[i];
                const cv::Vec3d anchor = unit(target.axis);
                if (!(target.weight > 0.0) || cv::norm(anchor) == 0.0) continue;
                if (target.fixed) {
                    // A boundary condition: the normal is the anchor's direction
                    // (projected into the node's plane) and does not move.
                    if (std::hypot(anchor.dot(u[i]), anchor.dot(v[i])) < 1e-9) continue;  // anchor along the fiber: no direction
                    problem.SetParameterBlockConstant(&theta[i]);
                    continue;
                }
                const cv::Vec3d& a = oriented[i];
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
    const std::vector<std::vector<FiberStripAxisTarget>>* targets,
    const FiberStripNormalProgress& onFiberSolved)
{
    if (targets && targets->size() != fibers.size())
        throw std::invalid_argument("fiber strip axis targets must match the fibers");
    std::vector<FiberStripNormalResult> results(fibers.size());
    std::atomic<std::int64_t> solved{0};
    const auto total = static_cast<std::int64_t>(fibers.size());
    cv::parallel_for_(cv::Range(0, static_cast<int>(fibers.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const auto f = static_cast<size_t>(i);
            results[f] = solveFiberStripNormals(fibers[f], options, targets ? &(*targets)[f] : nullptr);
            if (onFiberSolved) onFiberSolved(solved.fetch_add(1) + 1, total);
        }
    }, std::max(1, threads));
    return results;
}

}  // namespace vc::fiber_tracer
