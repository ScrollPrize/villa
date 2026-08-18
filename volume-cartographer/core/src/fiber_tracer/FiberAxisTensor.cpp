#include "vc/fiber_tracer/FiberAxisTensor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kMatrixEpsilon = 1.0e-15;

bool finiteVector(const cv::Vec3d& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

bool finiteMatrix(const cv::Matx33d& matrix)
{
    for (const double value : matrix.val) {
        if (!std::isfinite(value))
            return false;
    }
    return true;
}

double determinant(const cv::Matx33d& matrix)
{
    return
        matrix(0, 0) * (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)) -
        matrix(0, 1) * (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) +
        matrix(0, 2) * (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));
}

cv::Vec3d cross(const cv::Vec3d& left, const cv::Vec3d& right)
{
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

cv::Vec3d normalized(const cv::Vec3d& value)
{
    const double norm2 = value.dot(value);
    if (!(norm2 > kMatrixEpsilon * kMatrixEpsilon) || !std::isfinite(norm2))
        return {0.0, 0.0, 0.0};
    return value / std::sqrt(norm2);
}

}  // namespace

cv::Vec3d canonicalFiberAxis(cv::Vec3d axis)
{
    axis = normalized(axis);
    size_t signAxis = 0;
    double largestAbsolute = std::abs(axis[0]);
    for (size_t index = 1; index < 3; ++index) {
        const double candidate = std::abs(axis[static_cast<int>(index)]);
        if (candidate > largestAbsolute) {
            largestAbsolute = candidate;
            signAxis = index;
        }
    }
    if (axis[static_cast<int>(signAxis)] < 0.0)
        axis *= -1.0;
    return axis;
}

cv::Matx33d fiberAxisTensor(const cv::Vec3d& unitAxis, double weight)
{
    cv::Matx33d tensor = cv::Matx33d::zeros();
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column)
            tensor(row, column) = weight * unitAxis[row] * unitAxis[column];
    }
    return tensor;
}

FiberPrincipalAxis principalFiberAxis(const cv::Matx33d& input)
{
    cv::Matx33d matrix = input;
    cv::Matx33d eigenvectors = cv::Matx33d::eye();
    constexpr std::array<std::pair<int, int>, 3> rotations = {std::pair{0, 1}, std::pair{0, 2}, std::pair{1, 2}};
    for (int sweep = 0; sweep < 32; ++sweep) {
        bool changed = false;
        for (const auto [p, q] : rotations) {
            const double app = matrix(p, p);
            const double aqq = matrix(q, q);
            const double apq = matrix(p, q);
            const double scale = std::max({1.0, std::abs(app), std::abs(aqq)});
            if (std::abs(apq) <= kMatrixEpsilon * scale)
                continue;
            changed = true;
            const double tau = (aqq - app) / (2.0 * apq);
            const double sign = tau >= 0.0 ? 1.0 : -1.0;
            const double tangent = sign / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
            const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
            const double sine = tangent * cosine;
            for (int index = 0; index < 3; ++index) {
                if (index == p || index == q)
                    continue;
                const double aip = matrix(index, p);
                const double aiq = matrix(index, q);
                matrix(index, p) = matrix(p, index) = cosine * aip - sine * aiq;
                matrix(index, q) = matrix(q, index) = sine * aip + cosine * aiq;
            }
            matrix(p, p) = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
            matrix(q, q) = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
            matrix(p, q) = matrix(q, p) = 0.0;
            for (int row = 0; row < 3; ++row) {
                const double vip = eigenvectors(row, p);
                const double viq = eigenvectors(row, q);
                eigenvectors(row, p) = cosine * vip - sine * viq;
                eigenvectors(row, q) = sine * vip + cosine * viq;
            }
        }
        if (!changed)
            break;
    }

    std::array<int, 3> order{0, 1, 2};
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return matrix(a, a) > matrix(b, b); });
    FiberPrincipalAxis result;
    result.largestEigenvalue = matrix(order[0], order[0]);
    result.secondEigenvalue = matrix(order[1], order[1]);
    result.axis = canonicalFiberAxis({
        eigenvectors(0, order[0]),
        eigenvectors(1, order[0]),
        eigenvectors(2, order[0]),
    });
    result.valid = result.largestEigenvalue > kMatrixEpsilon && finiteVector(result.axis);
    const double gapTolerance = 1.0e-12 * std::max(1.0, std::abs(result.largestEigenvalue));
    result.unique = result.valid && result.largestEigenvalue - result.secondEigenvalue > gapTolerance;
    return result;
}

FiberPrincipalAxis principalFiberAxisClosedForm(
    const cv::Matx33d& input,
    bool* usedIterativeFallback)
{
    if (usedIterativeFallback != nullptr)
        *usedIterativeFallback = false;

    FiberPrincipalAxis result;
    if (!finiteMatrix(input))
        return result;

    double inputScale = 0.0;
    for (const double value : input.val)
        inputScale = std::max(inputScale, std::abs(value));
    if (!(inputScale > kMatrixEpsilon))
        return result;

    const cv::Matx33d matrix = input * (1.0 / inputScale);
    const double offDiagonalSquared =
        matrix(0, 1) * matrix(0, 1) +
        matrix(0, 2) * matrix(0, 2) +
        matrix(1, 2) * matrix(1, 2);

    std::array<double, 3> eigenvalues{};
    if (!(offDiagonalSquared > kMatrixEpsilon * kMatrixEpsilon)) {
        eigenvalues = {matrix(0, 0), matrix(1, 1), matrix(2, 2)};
    } else {
        const double mean =
            (matrix(0, 0) + matrix(1, 1) + matrix(2, 2)) / 3.0;
        const double centeredSquared =
            (matrix(0, 0) - mean) * (matrix(0, 0) - mean) +
            (matrix(1, 1) - mean) * (matrix(1, 1) - mean) +
            (matrix(2, 2) - mean) * (matrix(2, 2) - mean) +
            2.0 * offDiagonalSquared;
        const double radius = std::sqrt(centeredSquared / 6.0);
        if (!(radius > kMatrixEpsilon) || !std::isfinite(radius))
            return result;
        cv::Matx33d normalized = matrix;
        for (int axis = 0; axis < 3; ++axis)
            normalized(axis, axis) -= mean;
        normalized *= 1.0 / radius;
        const double halfDeterminant = std::clamp(
            determinant(normalized) * 0.5, -1.0, 1.0);
        const double angle = std::acos(halfDeterminant) / 3.0;
        eigenvalues[0] = mean + 2.0 * radius * std::cos(angle);
        eigenvalues[2] = mean + 2.0 * radius *
            std::cos(angle + 2.0 * std::numbers::pi_v<double> / 3.0);
        eigenvalues[1] = 3.0 * mean - eigenvalues[0] - eigenvalues[2];
    }

    std::array<int, 3> order{0, 1, 2};
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
        return eigenvalues[left] > eigenvalues[right];
    });
    const double largestScaled = eigenvalues[order[0]];
    const double secondScaled = eigenvalues[order[1]];
    result.largestEigenvalue = largestScaled * inputScale;
    result.secondEigenvalue = secondScaled * inputScale;
    result.valid = result.largestEigenvalue > kMatrixEpsilon;
    const double gapTolerance =
        1.0e-12 * std::max(1.0, std::abs(result.largestEigenvalue));
    result.unique = result.valid &&
        result.largestEigenvalue - result.secondEigenvalue > gapTolerance;
    if (!result.unique)
        return result;

    if (!(offDiagonalSquared > kMatrixEpsilon * kMatrixEpsilon)) {
        result.axis = {0.0, 0.0, 0.0};
        result.axis[order[0]] = 1.0;
        return result;
    }

    cv::Matx33d shifted = matrix;
    for (int axis = 0; axis < 3; ++axis)
        shifted(axis, axis) -= largestScaled;
    const std::array<cv::Vec3d, 3> rows{
        cv::Vec3d{shifted(0, 0), shifted(0, 1), shifted(0, 2)},
        cv::Vec3d{shifted(1, 0), shifted(1, 1), shifted(1, 2)},
        cv::Vec3d{shifted(2, 0), shifted(2, 1), shifted(2, 2)},
    };
    const std::array<cv::Vec3d, 3> candidates{
        cross(rows[0], rows[1]),
        cross(rows[0], rows[2]),
        cross(rows[1], rows[2]),
    };
    size_t best = 0;
    double bestNormSquared = candidates[0].dot(candidates[0]);
    for (size_t index = 1; index < candidates.size(); ++index) {
        const double normSquared = candidates[index].dot(candidates[index]);
        if (normSquared > bestNormSquared) {
            best = index;
            bestNormSquared = normSquared;
        }
    }
    if (bestNormSquared > kMatrixEpsilon * kMatrixEpsilon &&
        std::isfinite(bestNormSquared)) {
        result.axis = canonicalFiberAxis(candidates[best]);
        const cv::Vec3d residual = matrix * result.axis -
            largestScaled * result.axis;
        const double residualTolerance =
            1.0e-10 * std::max(1.0, std::abs(largestScaled));
        if (finiteVector(result.axis) &&
            residual.dot(residual) <= residualTolerance * residualTolerance) {
            return result;
        }
    }

    if (usedIterativeFallback != nullptr)
        *usedIterativeFallback = true;
    return principalFiberAxis(input);
}

}  // namespace vc::fiber_tracer
