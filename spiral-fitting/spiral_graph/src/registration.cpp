#include <spiral_graph/registration.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <ceres/ceres.h>

namespace spiral::registration {
namespace {

struct Pose {
    double angle = 0.0;
    double tu = 0.0;
    double tv = 0.0;
    bool reflected = false;
};

std::array<double, 2> apply(const Pose& pose, double u, double v)
{
    if (pose.reflected) v = -v;
    const double cosine = std::cos(pose.angle);
    const double sine = std::sin(pose.angle);
    return {
        cosine * u - sine * v + pose.tu,
        sine * u + cosine * v + pose.tv,
    };
}

double squared_error(const Pose& pose, const Correspondence2d& value)
{
    const auto transformed = apply(pose, value.source_u, value.source_v);
    const double du = transformed[0] - value.target_u;
    const double dv = transformed[1] - value.target_v;
    return du * du + dv * dv;
}

std::optional<Pose> pair_pose(
    const Correspondence2d& first,
    const Correspondence2d& second,
    bool reflected)
{
    double source_du = second.source_u - first.source_u;
    double source_dv = second.source_v - first.source_v;
    if (reflected) source_dv = -source_dv;
    const double target_du = second.target_u - first.target_u;
    const double target_dv = second.target_v - first.target_v;
    const double source_length = std::hypot(source_du, source_dv);
    const double target_length = std::hypot(target_du, target_dv);
    if (source_length < 1e-8 || target_length < 1e-8) return std::nullopt;
    Pose pose;
    pose.reflected = reflected;
    pose.angle = std::atan2(target_dv, target_du)
        - std::atan2(source_dv, source_du);
    const auto origin = apply(pose, first.source_u, first.source_v);
    pose.tu = first.target_u - origin[0];
    pose.tv = first.target_v - origin[1];
    return pose;
}

struct Residual {
    Residual(const Correspondence2d& value, bool reflected)
        : value(value), reflected(reflected) {}
    template <typename T>
    bool operator()(const T* parameters, T* residual) const
    {
        const T u = T(value.source_u);
        const T v = reflected ? -T(value.source_v) : T(value.source_v);
        const T cosine = ceres::cos(parameters[0]);
        const T sine = ceres::sin(parameters[0]);
        residual[0] = cosine * u - sine * v + parameters[1] - T(value.target_u);
        residual[1] = sine * u + cosine * v + parameters[2] - T(value.target_v);
        return true;
    }
    Correspondence2d value;
    bool reflected;
};

Pose refine(
    std::span<const Correspondence2d> values,
    const std::vector<std::size_t>& included,
    Pose pose,
    double huber)
{
    double parameters[3]{pose.angle, pose.tu, pose.tv};
    ceres::Problem problem;
    for (const std::size_t index : included) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<Residual, 2, 3>(
                new Residual(values[index], pose.reflected)),
            new ceres::HuberLoss(huber), parameters);
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) return pose;
    pose.angle = parameters[0];
    pose.tu = parameters[1];
    pose.tv = parameters[2];
    return pose;
}

std::vector<std::pair<std::size_t, std::size_t>> hypothesis_pairs(
    std::size_t count, std::size_t limit)
{
    if (count < 2 || limit == 0) return {};
    std::set<std::pair<std::size_t, std::size_t>> selected;
    if (count <= limit + 1 && count * (count - 1) / 2 <= limit) {
        for (std::size_t first = 0; first < count; ++first) {
            for (std::size_t second = first + 1; second < count; ++second) {
                selected.emplace(first, second);
            }
        }
    } else {
        const auto mix = [](std::uint64_t value) {
            value += 0x9e3779b97f4a7c15ull;
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
            return value ^ (value >> 31);
        };
        for (std::uint64_t attempt = 0;
             selected.size() < limit && attempt < limit * 32; ++attempt) {
            std::size_t first = static_cast<std::size_t>(
                mix(2 * attempt) % count);
            std::size_t second = static_cast<std::size_t>(
                mix(2 * attempt + 1) % count);
            if (first == second) second = (second + 1) % count;
            if (second < first) std::swap(first, second);
            selected.emplace(first, second);
        }
        // This deterministic fallback is only reachable after unusually many
        // hash collisions and stops as soon as the bounded reservoir is full.
        for (std::size_t first = 0;
             first < count && selected.size() < limit; ++first) {
            for (std::size_t second = first + 1;
                 second < count && selected.size() < limit; ++second) {
                selected.emplace(first, second);
            }
        }
    }
    return {selected.begin(), selected.end()};
}

std::vector<std::size_t> inliers(
    std::span<const Correspondence2d> values, const Pose& pose, double tolerance)
{
    const double squared = tolerance * tolerance;
    std::vector<std::size_t> output;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (squared_error(pose, values[index]) <= squared) output.push_back(index);
    }
    return output;
}

std::vector<Correspondence2d> hypothesis_reservoir(
    std::span<const Correspondence2d> values)
{
    constexpr std::size_t maximum = 8192;
    if (values.size() <= maximum) return {values.begin(), values.end()};
    std::vector<Correspondence2d> output;
    output.reserve(maximum);
    for (std::size_t index = 0; index < maximum; ++index) {
        const auto source = static_cast<std::size_t>(
            static_cast<long double>(index)
            * static_cast<long double>(values.size() - 1)
            / static_cast<long double>(maximum - 1));
        output.push_back(values[source]);
    }
    return output;
}

std::size_t spatially_distinct(
    std::span<const Correspondence2d> values,
    const std::vector<std::size_t>& included,
    double tolerance)
{
    std::set<std::pair<std::int64_t, std::int64_t>> cells;
    for (const std::size_t index : included) {
        cells.emplace(
            static_cast<std::int64_t>(std::floor(values[index].source_u / tolerance)),
            static_cast<std::int64_t>(std::floor(values[index].source_v / tolerance)));
    }
    return cells.size();
}

} // namespace

Result fit_rigid_2d(
    std::span<const Correspondence2d> correspondences, const Options& options)
{
    if (options.min_inliers == 0 || !(options.inlier_tolerance > 0.0)
        || !(options.max_refit_rms > 0.0) || options.max_hypotheses == 0) {
        throw std::invalid_argument("invalid registration options");
    }
    Result result;
    if (correspondences.size() < options.min_inliers) {
        result.rejection = "too_few_contacts";
        return result;
    }
    for (const auto& value : correspondences) {
        if (!(std::isfinite(value.source_u) && std::isfinite(value.source_v)
              && std::isfinite(value.target_u) && std::isfinite(value.target_v))) {
            throw std::invalid_argument("registration correspondence is non-finite");
        }
    }
    std::optional<Pose> best;
    std::vector<std::size_t> best_inliers;
    double best_rms = std::numeric_limits<double>::infinity();
    const std::vector<Correspondence2d> reservoir
        = hypothesis_reservoir(correspondences);
    for (const auto [first, second] : hypothesis_pairs(
             reservoir.size(), options.max_hypotheses)) {
        for (int reflected = 0; reflected <= (options.allow_reflection ? 1 : 0); ++reflected) {
            const auto pose = pair_pose(
                reservoir[first], reservoir[second], reflected != 0);
            if (!pose) continue;
            auto included = inliers(
                reservoir, *pose, options.inlier_tolerance);
            if (included.size() < options.min_inliers) continue;
            double sum = 0.0;
            for (const std::size_t index : included) {
                sum += squared_error(*pose, reservoir[index]);
            }
            const double rms = std::sqrt(sum / included.size());
            if (!best || included.size() > best_inliers.size()
                || (included.size() == best_inliers.size() && rms < best_rms)) {
                best = pose;
                best_inliers = std::move(included);
                best_rms = rms;
            }
        }
    }
    if (!best) {
        result.rejection = "ransac_inlier_gate";
        return result;
    }
    best_inliers = inliers(
        correspondences, *best, options.inlier_tolerance);
    if (best_inliers.size() < options.min_inliers) {
        result.rejection = "full_inlier_gate";
        return result;
    }
    for (int pass = 0; pass < 3; ++pass) {
        *best = refine(
            correspondences, best_inliers, *best, options.inlier_tolerance);
        best_inliers = inliers(
            correspondences, *best, options.inlier_tolerance);
        if (best_inliers.size() < options.min_inliers) {
            result.rejection = "refit_inlier_gate";
            return result;
        }
    }
    if (spatially_distinct(
            correspondences, best_inliers, options.inlier_tolerance)
        < options.min_inliers) {
        result.rejection = "spatial_inlier_gate";
        return result;
    }
    double sum = 0.0;
    for (const std::size_t index : best_inliers) {
        sum += squared_error(*best, correspondences[index]);
    }
    result.rms = std::sqrt(sum / best_inliers.size());
    if (!std::isfinite(result.rms) || result.rms > options.max_refit_rms) {
        result.rejection = "refit_rms";
        return result;
    }
    const double cosine = std::cos(best->angle);
    const double sine = std::sin(best->angle);
    result.accepted = true;
    result.reflected = best->reflected;
    result.r00 = cosine;
    result.r01 = best->reflected ? sine : -sine;
    result.r10 = sine;
    result.r11 = best->reflected ? -cosine : cosine;
    result.translation_u = best->tu;
    result.translation_v = best->tv;
    result.inliers = best_inliers.size();
    return result;
}

namespace {

struct AbsoluteGraphResidual {
    AbsoluteGraphResidual(
        double local_u, double local_v, double target_u, double target_v,
        bool reflected)
        : local_u(local_u), local_v(local_v), target_u(target_u),
          target_v(target_v), reflected(reflected) {}

    template <typename T>
    bool operator()(const T* parameters, T* residual) const
    {
        const T u = T(local_u);
        const T v = reflected ? -T(local_v) : T(local_v);
        const T cosine = ceres::cos(parameters[0]);
        const T sine = ceres::sin(parameters[0]);
        residual[0] = cosine * u - sine * v + parameters[1] - T(target_u);
        residual[1] = sine * u + cosine * v + parameters[2] - T(target_v);
        return true;
    }

    double local_u;
    double local_v;
    double target_u;
    double target_v;
    bool reflected;
};

struct RelativeGraphResidual {
    RelativeGraphResidual(
        double first_u, double first_v, double second_u, double second_v,
        bool first_reflected, bool second_reflected)
        : first_u(first_u), first_v(first_v), second_u(second_u),
          second_v(second_v), first_reflected(first_reflected),
          second_reflected(second_reflected) {}

    template <typename T>
    bool operator()(const T* first, const T* second, T* residual) const
    {
        const T au = T(first_u);
        const T av = first_reflected ? -T(first_v) : T(first_v);
        const T bu = T(second_u);
        const T bv = second_reflected ? -T(second_v) : T(second_v);
        const T ac = ceres::cos(first[0]);
        const T as = ceres::sin(first[0]);
        const T bc = ceres::cos(second[0]);
        const T bs = ceres::sin(second[0]);
        residual[0] = ac * au - as * av + first[1]
            - (bc * bu - bs * bv + second[1]);
        residual[1] = as * au + ac * av + first[2]
            - (bs * bu + bc * bv + second[2]);
        return true;
    }

    double first_u;
    double first_v;
    double second_u;
    double second_v;
    bool first_reflected;
    bool second_reflected;
};

} // namespace

PoseGraphResult refine_pose_graph(
    std::span<const Pose2d> initial,
    std::span<const AbsolutePoseConstraint> absolute,
    std::span<const RelativePoseConstraint> relative,
    double huber_transition,
    std::size_t max_iterations,
    std::size_t workers)
{
    if (initial.empty()) throw std::invalid_argument("pose graph is empty");
    if (!(huber_transition > 0.0) || max_iterations == 0 || workers == 0) {
        throw std::invalid_argument("invalid pose graph options");
    }
    std::vector<std::array<double, 3>> parameters(initial.size());
    std::vector<bool> reflected(initial.size(), false);
    for (std::size_t index = 0; index < initial.size(); ++index) {
        const auto& pose = initial[index];
        const double determinant = pose.r00 * pose.r11 - pose.r01 * pose.r10;
        if (!(std::isfinite(pose.r00) && std::isfinite(pose.r01)
              && std::isfinite(pose.r10) && std::isfinite(pose.r11)
              && std::isfinite(pose.translation_u)
              && std::isfinite(pose.translation_v))
            || std::abs(std::abs(determinant) - 1.0) > 1e-4) {
            throw std::invalid_argument("invalid initial pose graph transform");
        }
        reflected[index] = determinant < 0.0;
        parameters[index] = {
            std::atan2(pose.r10, pose.r00),
            pose.translation_u,
            pose.translation_v,
        };
    }
    ceres::Problem problem;
    for (auto& value : parameters) problem.AddParameterBlock(value.data(), 3);
    auto* loss = new ceres::HuberLoss(huber_transition);
    for (const auto& value : absolute) {
        if (value.patch >= initial.size()) {
            throw std::invalid_argument("absolute pose constraint index is invalid");
        }
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<AbsoluteGraphResidual, 2, 3>(
                new AbsoluteGraphResidual(
                    value.local_u, value.local_v,
                    value.target_u, value.target_v,
                    reflected[value.patch])),
            loss, parameters[value.patch].data());
    }
    for (const auto& value : relative) {
        if (value.first_patch >= initial.size()
            || value.second_patch >= initial.size()
            || value.first_patch == value.second_patch) {
            throw std::invalid_argument("relative pose constraint index is invalid");
        }
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RelativeGraphResidual, 2, 3, 3>(
                new RelativeGraphResidual(
                    value.first_u, value.first_v,
                    value.second_u, value.second_v,
                    reflected[value.first_patch], reflected[value.second_patch])),
            loss, parameters[value.first_patch].data(),
            parameters[value.second_patch].data());
    }
    if (absolute.empty()) problem.SetParameterBlockConstant(parameters[0].data());
    ceres::Solver::Options options;
    options.max_num_iterations = static_cast<int>(max_iterations);
    options.num_threads = static_cast<int>(workers);
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    PoseGraphResult output;
    output.usable = summary.IsSolutionUsable();
    output.initial_cost = summary.initial_cost;
    output.final_cost = summary.final_cost;
    output.iterations = summary.iterations.size();
    output.poses.reserve(initial.size());
    for (std::size_t index = 0; index < initial.size(); ++index) {
        const double cosine = std::cos(parameters[index][0]);
        const double sine = std::sin(parameters[index][0]);
        output.poses.push_back({
            cosine,
            reflected[index] ? sine : -sine,
            sine,
            reflected[index] ? -cosine : cosine,
            parameters[index][1],
            parameters[index][2],
        });
    }
    return output;
}

} // namespace spiral::registration
