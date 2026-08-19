#include "vc/fiber_tracer/FiberAxisTensor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>

namespace vc::fiber_tracer
{
namespace
{

template <typename Scalar>
constexpr Scalar matrixEpsilon()
{
    if constexpr (std::is_same_v<Scalar, float>)
        return 1.0e-7F;
    return 1.0e-15;
}

template <typename Scalar>
constexpr Scalar eigenvalueGapEpsilon()
{
    if constexpr (std::is_same_v<Scalar, float>)
        return 1.0e-6F;
    return 1.0e-12;
}

template <typename Scalar>
constexpr Scalar eigenvectorResidualEpsilon()
{
    if constexpr (std::is_same_v<Scalar, float>)
        return 1.0e-5F;
    return 1.0e-10;
}

template <typename Scalar>
bool finiteVector(const cv::Vec<Scalar, 3>& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

template <typename Scalar>
bool finiteMatrix(const cv::Matx<Scalar, 3, 3>& matrix)
{
    for (const Scalar value : matrix.val) {
        if (!std::isfinite(value))
            return false;
    }
    return true;
}

template <typename Scalar>
Scalar determinant(const cv::Matx<Scalar, 3, 3>& matrix)
{
    return
        matrix(0, 0) * (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)) -
        matrix(0, 1) * (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) +
        matrix(0, 2) * (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));
}

template <typename Scalar>
cv::Vec<Scalar, 3> cross(const cv::Vec<Scalar, 3>& left, const cv::Vec<Scalar, 3>& right)
{
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

template <typename Scalar>
cv::Vec<Scalar, 3> normalized(const cv::Vec<Scalar, 3>& value)
{
    const Scalar norm2 = value.dot(value);
    const Scalar epsilon = matrixEpsilon<Scalar>();
    if (!(norm2 > epsilon * epsilon) || !std::isfinite(norm2))
        return {Scalar{0}, Scalar{0}, Scalar{0}};
    return value / std::sqrt(norm2);
}

}  // namespace

template <typename Scalar>
cv::Vec<Scalar, 3> canonicalFiberAxisImpl(cv::Vec<Scalar, 3> axis)
{
    axis = normalized(axis);
    size_t signAxis = 0;
    Scalar largestAbsolute = std::abs(axis[0]);
    for (size_t index = 1; index < 3; ++index) {
        const Scalar candidate = std::abs(axis[static_cast<int>(index)]);
        if (candidate > largestAbsolute) {
            largestAbsolute = candidate;
            signAxis = index;
        }
    }
    if (axis[static_cast<int>(signAxis)] < Scalar{0})
        axis *= Scalar{-1};
    return axis;
}

template <typename Scalar>
cv::Matx<Scalar, 3, 3> fiberAxisTensorImpl(const cv::Vec<Scalar, 3>& unitAxis, Scalar weight)
{
    cv::Matx<Scalar, 3, 3> tensor = cv::Matx<Scalar, 3, 3>::zeros();
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column)
            tensor(row, column) = weight * unitAxis[row] * unitAxis[column];
    }
    return tensor;
}

template <typename Scalar>
struct PrincipalAxisResult {
    cv::Vec<Scalar, 3> axis{Scalar{0}, Scalar{0}, Scalar{0}};
    Scalar largestEigenvalue = Scalar{0};
    Scalar secondEigenvalue = Scalar{0};
    bool valid = false;
    bool unique = false;
};

template <typename Scalar>
PrincipalAxisResult<Scalar> principalFiberAxisImpl(const cv::Matx<Scalar, 3, 3>& input)
{
    cv::Matx<Scalar, 3, 3> matrix = input;
    cv::Matx<Scalar, 3, 3> eigenvectors = cv::Matx<Scalar, 3, 3>::eye();
    constexpr std::array<std::pair<int, int>, 3> rotations = {std::pair{0, 1}, std::pair{0, 2}, std::pair{1, 2}};
    for (int sweep = 0; sweep < 32; ++sweep) {
        bool changed = false;
        for (const auto [p, q] : rotations) {
            const Scalar app = matrix(p, p);
            const Scalar aqq = matrix(q, q);
            const Scalar apq = matrix(p, q);
            const Scalar scale = std::max({Scalar{1}, std::abs(app), std::abs(aqq)});
            if (std::abs(apq) <= matrixEpsilon<Scalar>() * scale)
                continue;
            changed = true;
            const Scalar tau = (aqq - app) / (Scalar{2} * apq);
            const Scalar sign = tau >= Scalar{0} ? Scalar{1} : Scalar{-1};
            const Scalar tangent = sign / (std::abs(tau) + std::sqrt(Scalar{1} + tau * tau));
            const Scalar cosine = Scalar{1} / std::sqrt(Scalar{1} + tangent * tangent);
            const Scalar sine = tangent * cosine;
            for (int index = 0; index < 3; ++index) {
                if (index == p || index == q)
                    continue;
                const Scalar aip = matrix(index, p);
                const Scalar aiq = matrix(index, q);
                matrix(index, p) = matrix(p, index) = cosine * aip - sine * aiq;
                matrix(index, q) = matrix(q, index) = sine * aip + cosine * aiq;
            }
            matrix(p, p) = cosine * cosine * app - Scalar{2} * sine * cosine * apq + sine * sine * aqq;
            matrix(q, q) = sine * sine * app + Scalar{2} * sine * cosine * apq + cosine * cosine * aqq;
            matrix(p, q) = matrix(q, p) = Scalar{0};
            for (int row = 0; row < 3; ++row) {
                const Scalar vip = eigenvectors(row, p);
                const Scalar viq = eigenvectors(row, q);
                eigenvectors(row, p) = cosine * vip - sine * viq;
                eigenvectors(row, q) = sine * vip + cosine * viq;
            }
        }
        if (!changed)
            break;
    }

    std::array<int, 3> order{0, 1, 2};
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return matrix(a, a) > matrix(b, b); });
    PrincipalAxisResult<Scalar> result;
    result.largestEigenvalue = matrix(order[0], order[0]);
    result.secondEigenvalue = matrix(order[1], order[1]);
    result.axis = canonicalFiberAxisImpl<Scalar>({
        eigenvectors(0, order[0]),
        eigenvectors(1, order[0]),
        eigenvectors(2, order[0]),
    });
    result.valid = result.largestEigenvalue > matrixEpsilon<Scalar>() && finiteVector(result.axis);
    const Scalar gapTolerance = eigenvalueGapEpsilon<Scalar>() *
        std::max(Scalar{1}, std::abs(result.largestEigenvalue));
    result.unique = result.valid && result.largestEigenvalue - result.secondEigenvalue > gapTolerance;
    return result;
}

template <typename Scalar>
PrincipalAxisResult<Scalar> principalFiberAxisClosedFormImpl(
    const cv::Matx<Scalar, 3, 3>& input,
    bool* usedIterativeFallback)
{
    if (usedIterativeFallback != nullptr)
        *usedIterativeFallback = false;

    PrincipalAxisResult<Scalar> result;
    if (!finiteMatrix(input))
        return result;

    Scalar inputScale = Scalar{0};
    for (const Scalar value : input.val)
        inputScale = std::max(inputScale, std::abs(value));
    if (!(inputScale > matrixEpsilon<Scalar>()))
        return result;

    const cv::Matx<Scalar, 3, 3> matrix = input * (Scalar{1} / inputScale);
    const Scalar offDiagonalSquared =
        matrix(0, 1) * matrix(0, 1) +
        matrix(0, 2) * matrix(0, 2) +
        matrix(1, 2) * matrix(1, 2);

    std::array<Scalar, 3> eigenvalues{};
    if (!(offDiagonalSquared > matrixEpsilon<Scalar>() * matrixEpsilon<Scalar>())) {
        eigenvalues = {matrix(0, 0), matrix(1, 1), matrix(2, 2)};
    } else {
        const Scalar mean =
            (matrix(0, 0) + matrix(1, 1) + matrix(2, 2)) / Scalar{3};
        const Scalar centeredSquared =
            (matrix(0, 0) - mean) * (matrix(0, 0) - mean) +
            (matrix(1, 1) - mean) * (matrix(1, 1) - mean) +
            (matrix(2, 2) - mean) * (matrix(2, 2) - mean) +
            Scalar{2} * offDiagonalSquared;
        const Scalar radius = std::sqrt(centeredSquared / Scalar{6});
        if (!(radius > matrixEpsilon<Scalar>()) || !std::isfinite(radius))
            return result;
        cv::Matx<Scalar, 3, 3> normalized = matrix;
        for (int axis = 0; axis < 3; ++axis)
            normalized(axis, axis) -= mean;
        normalized *= Scalar{1} / radius;
        const Scalar halfDeterminant = std::clamp(
            determinant(normalized) * Scalar{0.5}, Scalar{-1}, Scalar{1});
        const Scalar angle = std::acos(halfDeterminant) / Scalar{3};
        eigenvalues[0] = mean + Scalar{2} * radius * std::cos(angle);
        eigenvalues[2] = mean + Scalar{2} * radius *
            std::cos(angle + Scalar{2} * std::numbers::pi_v<Scalar> / Scalar{3});
        eigenvalues[1] = Scalar{3} * mean - eigenvalues[0] - eigenvalues[2];
    }

    std::array<int, 3> order{0, 1, 2};
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
        return eigenvalues[left] > eigenvalues[right];
    });
    const Scalar largestScaled = eigenvalues[order[0]];
    const Scalar secondScaled = eigenvalues[order[1]];
    result.largestEigenvalue = largestScaled * inputScale;
    result.secondEigenvalue = secondScaled * inputScale;
    result.valid = result.largestEigenvalue > matrixEpsilon<Scalar>();
    const Scalar gapTolerance = eigenvalueGapEpsilon<Scalar>() *
        std::max(Scalar{1}, std::abs(result.largestEigenvalue));
    result.unique = result.valid &&
        result.largestEigenvalue - result.secondEigenvalue > gapTolerance;
    if (!result.unique)
        return result;

    if (!(offDiagonalSquared > matrixEpsilon<Scalar>() * matrixEpsilon<Scalar>())) {
        result.axis = {Scalar{0}, Scalar{0}, Scalar{0}};
        result.axis[order[0]] = Scalar{1};
        return result;
    }

    cv::Matx<Scalar, 3, 3> shifted = matrix;
    for (int axis = 0; axis < 3; ++axis)
        shifted(axis, axis) -= largestScaled;
    const std::array<cv::Vec<Scalar, 3>, 3> rows{
        cv::Vec<Scalar, 3>{shifted(0, 0), shifted(0, 1), shifted(0, 2)},
        cv::Vec<Scalar, 3>{shifted(1, 0), shifted(1, 1), shifted(1, 2)},
        cv::Vec<Scalar, 3>{shifted(2, 0), shifted(2, 1), shifted(2, 2)},
    };
    const std::array<cv::Vec<Scalar, 3>, 3> candidates{
        cross(rows[0], rows[1]),
        cross(rows[0], rows[2]),
        cross(rows[1], rows[2]),
    };
    size_t best = 0;
    Scalar bestNormSquared = candidates[0].dot(candidates[0]);
    for (size_t index = 1; index < candidates.size(); ++index) {
        const Scalar normSquared = candidates[index].dot(candidates[index]);
        if (normSquared > bestNormSquared) {
            best = index;
            bestNormSquared = normSquared;
        }
    }
    if (bestNormSquared > matrixEpsilon<Scalar>() * matrixEpsilon<Scalar>() &&
        std::isfinite(bestNormSquared)) {
        result.axis = canonicalFiberAxisImpl<Scalar>(candidates[best]);
        const cv::Vec<Scalar, 3> residual = matrix * result.axis -
            largestScaled * result.axis;
        const Scalar residualTolerance = eigenvectorResidualEpsilon<Scalar>() *
            std::max(Scalar{1}, std::abs(largestScaled));
        if (finiteVector(result.axis) &&
            residual.dot(residual) <= residualTolerance * residualTolerance) {
            return result;
        }
    }

    if (usedIterativeFallback != nullptr)
        *usedIterativeFallback = true;
    return principalFiberAxisImpl<Scalar>(input);
}

cv::Vec3d canonicalFiberAxis(cv::Vec3d axis)
{
    return canonicalFiberAxisImpl<double>(axis);
}

cv::Vec3f canonicalFiberAxisF(cv::Vec3f axis)
{
    return canonicalFiberAxisImpl<float>(axis);
}

cv::Matx33d fiberAxisTensor(const cv::Vec3d& unitAxis, double weight)
{
    return fiberAxisTensorImpl<double>(unitAxis, weight);
}

cv::Matx33f fiberAxisTensorF(const cv::Vec3f& unitAxis, float weight)
{
    return fiberAxisTensorImpl<float>(unitAxis, weight);
}

FiberPrincipalAxis principalFiberAxis(const cv::Matx33d& tensor)
{
    const auto result = principalFiberAxisImpl<double>(tensor);
    return {result.axis, result.largestEigenvalue, result.secondEigenvalue,
            result.valid, result.unique};
}

FiberPrincipalAxisF principalFiberAxisF(const cv::Matx33f& tensor)
{
    const auto result = principalFiberAxisImpl<float>(tensor);
    return {result.axis, result.largestEigenvalue, result.secondEigenvalue,
            result.valid, result.unique};
}

FiberPrincipalAxis principalFiberAxisClosedForm(
    const cv::Matx33d& tensor,
    bool* usedIterativeFallback)
{
    const auto result = principalFiberAxisClosedFormImpl<double>(
        tensor, usedIterativeFallback);
    return {result.axis, result.largestEigenvalue, result.secondEigenvalue,
            result.valid, result.unique};
}

FiberPrincipalAxisF principalFiberAxisClosedFormF(
    const cv::Matx33f& tensor,
    bool* usedIterativeFallback)
{
    const auto result = principalFiberAxisClosedFormImpl<float>(
        tensor, usedIterativeFallback);
    return {result.axis, result.largestEigenvalue, result.secondEigenvalue,
            result.valid, result.unique};
}

}  // namespace vc::fiber_tracer
