#include "vc/fiber_tracer/FiberAxisTensor.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kMatrixEpsilon = 1.0e-15;

bool finiteVector(const cv::Vec3d& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
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

}  // namespace vc::fiber_tracer
