#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct FiberPrincipalAxis {
    cv::Vec3d axis{0.0, 0.0, 0.0};
    double largestEigenvalue = 0.0;
    double secondEigenvalue = 0.0;
    bool valid = false;
    bool unique = false;
};

struct FiberPrincipalAxisF {
    cv::Vec3f axis{0.0F, 0.0F, 0.0F};
    float largestEigenvalue = 0.0F;
    float secondEigenvalue = 0.0F;
    bool valid = false;
    bool unique = false;
};

[[nodiscard]] cv::Vec3d canonicalFiberAxis(cv::Vec3d axis);
[[nodiscard]] cv::Vec3f canonicalFiberAxisF(cv::Vec3f axis);

[[nodiscard]] cv::Matx33d fiberAxisTensor(const cv::Vec3d& unitAxis, double weight = 1.0);
[[nodiscard]] cv::Matx33f fiberAxisTensorF(const cv::Vec3f& unitAxis, float weight = 1.0F);

[[nodiscard]] FiberPrincipalAxis principalFiberAxis(const cv::Matx33d& tensor);
[[nodiscard]] FiberPrincipalAxisF principalFiberAxisF(const cv::Matx33f& tensor);

// Resolves a symmetric 3x3 tensor in closed form. Ambiguous top eigenvalues
// remain non-unique; the iterative solver is used only when a clear-gap
// eigenvector cannot be reconstructed to a bounded residual.
[[nodiscard]] FiberPrincipalAxis principalFiberAxisClosedForm(
    const cv::Matx33d& tensor,
    bool* usedIterativeFallback = nullptr);
[[nodiscard]] FiberPrincipalAxisF principalFiberAxisClosedFormF(
    const cv::Matx33f& tensor,
    bool* usedIterativeFallback = nullptr);

template <typename Scalar>
struct FiberAxisPairFit {
    std::array<cv::Vec<Scalar, 3>, 2> axes{};
    std::vector<std::uint8_t> assignments;
    Scalar objective = Scalar{-1};
    std::size_t iterations = 0;
    std::size_t bestIteration = 0;
};

// Refines two unoriented axes against weighted direction observations. The
// accessors are inlined so callers can retain their native observation layout.
template <typename Scalar, typename DirectionAt, typename WeightAt>
[[nodiscard]] FiberAxisPairFit<Scalar> refineFiberAxisPair(
    std::size_t observationCount,
    std::array<cv::Vec<Scalar, 3>, 2> axes,
    int maximumIterations,
    Scalar convergenceTolerance,
    DirectionAt&& directionAt,
    WeightAt&& weightAt)
{
    const auto assign = [&](const std::array<cv::Vec<Scalar, 3>, 2>& candidates) {
        std::vector<std::uint8_t> assignments(observationCount, 0);
        for (std::size_t index = 0; index < observationCount; ++index) {
            const auto direction = directionAt(index);
            const Scalar first = direction.dot(candidates[0]);
            const Scalar second = direction.dot(candidates[1]);
            assignments[index] = first * first >= second * second ? 0 : 1;
        }
        return assignments;
    };
    const auto tensor = [&](const std::vector<std::uint8_t>& assignments,
                            std::uint8_t component) {
        cv::Matx<Scalar, 3, 3> result = cv::Matx<Scalar, 3, 3>::zeros();
        for (std::size_t index = 0; index < observationCount; ++index) {
            if (assignments[index] != component)
                continue;
            const auto direction = directionAt(index);
            const Scalar weight = weightAt(index);
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    result(row, column) +=
                        weight * direction[row] * direction[column];
                }
            }
        }
        return result;
    };
    const auto objective = [&](const std::array<cv::Vec<Scalar, 3>, 2>& candidates) {
        Scalar result = Scalar{0};
        for (std::size_t index = 0; index < observationCount; ++index) {
            const auto direction = directionAt(index);
            const Scalar first = direction.dot(candidates[0]);
            const Scalar second = direction.dot(candidates[1]);
            result += weightAt(index) *
                std::max(first * first, second * second);
        }
        return result;
    };
    const auto principal = [](const cv::Matx<Scalar, 3, 3>& value) {
        if constexpr (std::is_same_v<Scalar, float>)
            return principalFiberAxisF(value);
        else
            return principalFiberAxis(value);
    };

    FiberAxisPairFit<Scalar> best;
    std::vector<std::uint8_t> previousAssignments;
    std::vector<std::uint8_t> twoBackAssignments;
    for (int iteration = 0; iteration < maximumIterations; ++iteration) {
        ++best.iterations;
        auto assignments = assign(axes);
        auto updated = axes;
        for (std::uint8_t component = 0; component < 2; ++component) {
            const auto fitted = principal(tensor(assignments, component));
            if (fitted.unique)
                updated[component] = fitted.axis;
        }
        const Scalar candidateObjective = objective(updated);
        if (best.objective < Scalar{0} || candidateObjective > best.objective) {
            best.axes = updated;
            best.assignments = assignments;
            best.objective = candidateObjective;
            best.bestIteration = static_cast<std::size_t>(iteration);
        }

        const Scalar update = std::max(
            Scalar{1} - std::clamp(std::abs(axes[0].dot(updated[0])), Scalar{0}, Scalar{1}),
            Scalar{1} - std::clamp(std::abs(axes[1].dot(updated[1])), Scalar{0}, Scalar{1}));
        const bool unchanged = !previousAssignments.empty() &&
            assignments == previousAssignments;
        const bool twoCycle = !twoBackAssignments.empty() &&
            assignments == twoBackAssignments;
        axes = updated;
        if ((unchanged && update <= convergenceTolerance) || twoCycle)
            break;
        twoBackAssignments = std::move(previousAssignments);
        previousAssignments = std::move(assignments);
    }
    best.assignments = assign(best.axes);
    best.objective = objective(best.axes);
    return best;
}

}  // namespace vc::fiber_tracer
