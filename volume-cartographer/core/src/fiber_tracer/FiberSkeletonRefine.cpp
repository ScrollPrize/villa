#include "vc/fiber_tracer/FiberSkeletonRefine.hpp"

#include <ceres/ceres.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vc::fiber_tracer
{
namespace
{

using Point3 = std::array<double, 3>;

// Every norm is evaluated as `sqrt(|v|^2 + kNormEpsilon)` so that Jets stay
// finite at degenerate configurations (zero-length steps, antiparallel unit
// vectors). The resulting bias on a unit vector is ~5e-13 and the smallest
// representable angle becomes ~1e-6 rad.
constexpr double kNormEpsilon = 1.0e-12;
// An interpolated direction/normal shorter than this is treated as invalid.
constexpr double kPredictionNormMin = 1.0e-9;
// `asin` is only evaluated on values clamped this far away from +-1.
constexpr double kUnitClampMargin = 1.0e-6;
// Tangential projections shorter than this fall back to the full 3D angle.
constexpr double kProjectionNormMin = 1.0e-6;
// Floor of the alignment term, so that its square root is differentiable.
constexpr double kAlignmentFloor = 1.0e-3;
// Floor of the spacing denominator, so that the residual stays finite if the
// solver drives the shared spacing parameter to zero.
constexpr double kSpacingFloor = 1.0e-9;

template <typename T>
T dot3(const T* left, const T* right)
{
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

template <typename T>
void cross3(const T* left, const T* right, T* out)
{
    out[0] = left[1] * right[2] - left[2] * right[1];
    out[1] = left[2] * right[0] - left[0] * right[2];
    out[2] = left[0] * right[1] - left[1] * right[0];
}

template <typename T>
T guardedNorm3(const T* value)
{
    using std::sqrt;
    return sqrt(dot3(value, value) + T(kNormEpsilon));
}

template <typename T>
T clampZeroOne(const T& value)
{
    if (value < T(0.0))
        return T(0.0);
    if (value > T(1.0))
        return T(1.0);
    return value;
}

template <typename T>
T clampUnitInterior(const T& value)
{
    if (value < T(-1.0 + kUnitClampMargin))
        return T(-1.0 + kUnitClampMargin);
    if (value > T(1.0 - kUnitClampMargin))
        return T(1.0 - kUnitClampMargin);
    return value;
}

// `atan2(|left x right|, left . right)`: the same angle `acos` returns for unit
// inputs, but with a bounded derivative near 0 and pi.
template <typename T>
T angleBetween(const T* left, const T* right)
{
    using std::atan2;
    std::array<T, 3> perpendicular{T(0.0), T(0.0), T(0.0)};
    cross3(left, right, perpendicular.data());
    return atan2(guardedNorm3(perpendicular.data()), dot3(left, right));
}

template <typename T>
struct InterpolatedPrediction {
    T presence = T(0.0);
    std::array<T, 3> direction{T(0.0), T(0.0), T(0.0)};
    std::array<T, 3> normal{T(0.0), T(0.0), T(0.0)};
    bool valid = false;
    bool normalValid = false;
};

// Trilinear interpolation of one corner block. The block contents are
// constants of the current evaluation; only the cell-local coordinates carry
// derivatives, so Jets flow through the interpolation weights.
template <typename T>
InterpolatedPrediction<T> interpolateBlock(
    const FiberSkeletonCornerBlock& block, const T* pointBaseXYZ)
{
    InterpolatedPrediction<T> result;
    const double spacing =
        block.spacingBase > kNormEpsilon ? block.spacingBase : 1.0;
    const double inverseSpacing = 1.0 / spacing;
    std::array<T, 3> local{T(0.0), T(0.0), T(0.0)};
    for (int axis = 0; axis < 3; ++axis) {
        local[axis] =
            (pointBaseXYZ[axis] - T(block.cellOriginBaseXYZ[axis])) *
            T(inverseSpacing);
    }

    T presence = T(0.0);
    std::array<T, 3> direction{T(0.0), T(0.0), T(0.0)};
    std::array<T, 3> normal{T(0.0), T(0.0), T(0.0)};
    for (int corner = 0; corner < 8; ++corner) {
        const int offsetX = corner & 1;
        const int offsetY = (corner >> 1) & 1;
        const int offsetZ = (corner >> 2) & 1;
        const T weightX = offsetX != 0 ? local[0] : T(1.0) - local[0];
        const T weightY = offsetY != 0 ? local[1] : T(1.0) - local[1];
        const T weightZ = offsetZ != 0 ? local[2] : T(1.0) - local[2];
        const T weight = weightX * weightY * weightZ;
        if (block.valid[corner] != 0) {
            result.valid = true;
            presence += weight * T(block.presence[corner]);
            const cv::Vec3d& cornerDirection = block.direction[corner];
            for (int axis = 0; axis < 3; ++axis)
                direction[axis] += weight * T(cornerDirection[axis]);
        }
        if (block.normalValid[corner] != 0) {
            result.normalValid = true;
            const cv::Vec3d& cornerNormal = block.normal[corner];
            for (int axis = 0; axis < 3; ++axis)
                normal[axis] += weight * T(cornerNormal[axis]);
        }
    }
    result.presence = presence;

    if (result.valid) {
        const T squared = dot3(direction.data(), direction.data());
        if (squared < T(kPredictionNormMin * kPredictionNormMin)) {
            result.valid = false;
        } else {
            const T norm = guardedNorm3(direction.data());
            for (int axis = 0; axis < 3; ++axis)
                result.direction[axis] = direction[axis] / norm;
        }
    }
    if (result.normalValid) {
        const T squared = dot3(normal.data(), normal.data());
        if (squared < T(kPredictionNormMin * kPredictionNormMin)) {
            result.normalValid = false;
        } else {
            const T norm = guardedNorm3(normal.data());
            for (int axis = 0; axis < 3; ++axis)
                result.normal[axis] = normal[axis] / norm;
        }
    }
    return result;
}

// The length-scaled alignment term `L * (1 - score)` of the discrete metric,
// floored so its square root is differentiable.
//
// Deliberate deviation: the discrete metric charges
// `invalidPredictionCostPerVoxel * L` (4 * L by default) when the candidate
// prediction is invalid. Here an invalid prediction only drives the score to
// zero, so the term saturates at `L`. Keeping the larger constant would make
// the residual discontinuous in the geometry exactly where the solver needs a
// usable gradient.
template <typename T>
T alignmentSquaredTerm(
    const FiberSkeletonCornerBlock& currentBlock,
    const FiberSkeletonCornerBlock& nextBlock, const T* currentPoint,
    const T* nextPoint, const T* previousDirection,
    const T* candidateDirection, const T& candidateLength)
{
    const auto current = interpolateBlock(currentBlock, currentPoint);
    const auto next = interpolateBlock(nextBlock, nextPoint);

    T score = T(0.0);
    if (next.valid) {
        // Mirrors `prepareFiberLocalIncomingAlignmentInline`, which substitutes
        // the incoming step direction when the prediction at the current vertex
        // is unavailable.
        const std::array<T, 3> currentDirection = current.valid
            ? current.direction
            : std::array<T, 3>{previousDirection[0], previousDirection[1],
                               previousDirection[2]};
        const T* u = currentDirection.data();
        const T* v = next.direction.data();
        score = next.presence;
        score *= clampZeroOne(dot3(previousDirection, candidateDirection));
        score *= clampZeroOne(dot3(previousDirection, u));
        score *= clampZeroOne(dot3(previousDirection, v));
        score *= clampZeroOne(dot3(u, candidateDirection));
        score *= clampZeroOne(dot3(u, v));
        score *= clampZeroOne(dot3(candidateDirection, v));
    }

    const T value = candidateLength * (T(1.0) - score);
    return value < T(kAlignmentFloor) ? T(kAlignmentFloor) : value;
}

template <typename T>
struct SmoothnessAngles {
    // Tangential angle, or the isotropic fallback angle when `isotropic`.
    T planar = T(0.0);
    T normal = T(0.0);
    bool isotropic = true;
};

// Mirrors `fiberLocalSmoothnessCostCandidatePreparedInline` minus its free
// angle: this objective charges the full angle instead of the excess over
// `freeAngleRadians`, so that the residual is linear in the angle and its
// square reproduces the discrete quadratic exactly when the free angle is 0.
template <typename T>
SmoothnessAngles<T> smoothnessAngles(
    const InterpolatedPrediction<T>& prediction, const T* previousDirection,
    const T* candidateDirection)
{
    using std::abs;
    using std::asin;

    SmoothnessAngles<T> result;
    if (!prediction.normalValid) {
        result.planar = angleBetween(previousDirection, candidateDirection);
        result.isotropic = true;
        return result;
    }
    result.isotropic = false;

    const T* normal = prediction.normal.data();
    const T previousNormal =
        clampUnitInterior(dot3(previousDirection, normal));
    const T candidateNormal =
        clampUnitInterior(dot3(candidateDirection, normal));
    std::array<T, 3> previousTangent{T(0.0), T(0.0), T(0.0)};
    std::array<T, 3> candidateTangent{T(0.0), T(0.0), T(0.0)};
    for (int axis = 0; axis < 3; ++axis) {
        previousTangent[axis] =
            previousDirection[axis] - normal[axis] * previousNormal;
        candidateTangent[axis] =
            candidateDirection[axis] - normal[axis] * candidateNormal;
    }
    const T previousTangentNorm = guardedNorm3(previousTangent.data());
    const T candidateTangentNorm = guardedNorm3(candidateTangent.data());
    if (previousTangentNorm <= T(kProjectionNormMin) ||
        candidateTangentNorm <= T(kProjectionNormMin)) {
        result.planar = angleBetween(previousDirection, candidateDirection);
    } else {
        for (int axis = 0; axis < 3; ++axis) {
            previousTangent[axis] /= previousTangentNorm;
            candidateTangent[axis] /= candidateTangentNorm;
        }
        result.planar =
            angleBetween(previousTangent.data(), candidateTangent.data());
    }
    result.normal = abs(asin(candidateNormal) - asin(previousNormal));
    return result;
}

struct SmoothnessScales {
    double tangent = 0.0;
    double normal = 0.0;
    double isotropic = 0.0;
};

// Two residuals, always. Autodiff needs a fixed residual count, so the
// isotropic fallback puts its single value in `residuals[0]` and zeroes
// `residuals[1]`; `isotropicOut` reports which case was taken so the cost
// breakdown can attribute `residuals[0]` to the right term.
template <typename T>
void smoothnessResidualCore(
    const FiberSkeletonCornerBlock& block, const T* targetPoint,
    const T* previousDirection, const T* candidateDirection,
    const T& meanLength, const SmoothnessScales& scales, T* residuals,
    bool* isotropicOut)
{
    using std::sqrt;

    const auto prediction = interpolateBlock(block, targetPoint);
    const auto angles =
        smoothnessAngles(prediction, previousDirection, candidateDirection);
    // `1 / sqrt(max(1, Lbar))`. The weight square roots are folded in as
    // constants so that a zero weight never differentiates `sqrt` at zero.
    const T effectiveLength = meanLength > T(1.0) ? meanLength : T(1.0);
    const T inverseScale = T(1.0) / sqrt(effectiveLength);
    if (angles.isotropic) {
        residuals[0] = T(scales.isotropic) * inverseScale * angles.planar;
        residuals[1] = T(0.0);
    } else {
        residuals[0] = T(scales.tangent) * inverseScale * angles.planar;
        residuals[1] = T(scales.normal) * inverseScale * angles.normal;
    }
    if (isotropicOut != nullptr)
        *isotropicOut = angles.isotropic;
}

template <typename T>
void stepDirection(
    const T* from, const T* to, std::array<T, 3>& direction, T& length)
{
    std::array<T, 3> step{T(0.0), T(0.0), T(0.0)};
    for (int axis = 0; axis < 3; ++axis)
        step[axis] = to[axis] - from[axis];
    length = guardedNorm3(step.data());
    for (int axis = 0; axis < 3; ++axis)
        direction[axis] = step[axis] / length;
}

template <typename T>
bool alignmentResidualCore(
    const std::vector<FiberSkeletonCornerBlock>& blocks,
    std::size_t currentVertex, const T* previousDirection,
    const T* currentPoint, const T* nextPoint, T* residuals)
{
    using std::sqrt;

    std::array<T, 3> candidateDirection{T(0.0), T(0.0), T(0.0)};
    T candidateLength = T(0.0);
    stepDirection(currentPoint, nextPoint, candidateDirection, candidateLength);
    residuals[0] = sqrt(alignmentSquaredTerm(
        blocks[currentVertex], blocks[currentVertex + 1], currentPoint,
        nextPoint, previousDirection, candidateDirection.data(),
        candidateLength));
    return true;
}

// Alignment for step `currentVertex`, with the previous direction taken from
// the incoming step.
struct AlignmentResidual {
    const std::vector<FiberSkeletonCornerBlock>* blocks = nullptr;
    std::size_t currentVertex = 0;

    template <typename T>
    bool operator()(
        const T* const previousPoint, const T* const currentPoint,
        const T* const nextPoint, T* residuals) const
    {
        std::array<T, 3> previousDirection{T(0.0), T(0.0), T(0.0)};
        T previousLength = T(0.0);
        stepDirection(
            previousPoint, currentPoint, previousDirection, previousLength);
        return alignmentResidualCore(
            *blocks, currentVertex, previousDirection.data(), currentPoint,
            nextPoint, residuals);
    }
};

// Alignment for step 0, with the previous direction fixed by `startTangent`.
struct AlignmentStartResidual {
    const std::vector<FiberSkeletonCornerBlock>* blocks = nullptr;
    Point3 previousDirection{0.0, 0.0, 0.0};

    template <typename T>
    bool operator()(
        const T* const currentPoint, const T* const nextPoint,
        T* residuals) const
    {
        const std::array<T, 3> previous{
            T(previousDirection[0]), T(previousDirection[1]),
            T(previousDirection[2])};
        return alignmentResidualCore(
            *blocks, 0, previous.data(), currentPoint, nextPoint, residuals);
    }
};

struct SmoothnessResidual {
    const std::vector<FiberSkeletonCornerBlock>* blocks = nullptr;
    std::size_t targetVertex = 0;
    SmoothnessScales scales;

    template <typename T>
    bool evaluate(
        const T* previousPoint, const T* currentPoint, const T* nextPoint,
        T* residuals, bool* isotropicOut) const
    {
        std::array<T, 3> previousDirection{T(0.0), T(0.0), T(0.0)};
        std::array<T, 3> candidateDirection{T(0.0), T(0.0), T(0.0)};
        T previousLength = T(0.0);
        T candidateLength = T(0.0);
        stepDirection(
            previousPoint, currentPoint, previousDirection, previousLength);
        stepDirection(
            currentPoint, nextPoint, candidateDirection, candidateLength);
        const T meanLength = (previousLength + candidateLength) * T(0.5);
        smoothnessResidualCore(
            (*blocks)[targetVertex], nextPoint, previousDirection.data(),
            candidateDirection.data(), meanLength, scales, residuals,
            isotropicOut);
        return true;
    }

    template <typename T>
    bool operator()(
        const T* const previousPoint, const T* const currentPoint,
        const T* const nextPoint, T* residuals) const
    {
        return evaluate(
            previousPoint, currentPoint, nextPoint, residuals, nullptr);
    }
};

// Boundary smoothness at step 0: the previous direction is `startTangent` and
// the length scale is `L_0`.
struct SmoothnessStartResidual {
    const std::vector<FiberSkeletonCornerBlock>* blocks = nullptr;
    SmoothnessScales scales;
    Point3 previousDirection{0.0, 0.0, 0.0};

    template <typename T>
    bool evaluate(
        const T* currentPoint, const T* nextPoint, T* residuals,
        bool* isotropicOut) const
    {
        const std::array<T, 3> previous{
            T(previousDirection[0]), T(previousDirection[1]),
            T(previousDirection[2])};
        std::array<T, 3> candidateDirection{T(0.0), T(0.0), T(0.0)};
        T candidateLength = T(0.0);
        stepDirection(
            currentPoint, nextPoint, candidateDirection, candidateLength);
        smoothnessResidualCore(
            (*blocks)[1], nextPoint, previous.data(),
            candidateDirection.data(), candidateLength, scales, residuals,
            isotropicOut);
        return true;
    }

    template <typename T>
    bool operator()(
        const T* const currentPoint, const T* const nextPoint,
        T* residuals) const
    {
        return evaluate(currentPoint, nextPoint, residuals, nullptr);
    }
};

// Boundary smoothness past the last vertex: the candidate direction is
// `endTangent` and the length scale is `L_{N-1}`.
struct SmoothnessEndResidual {
    const std::vector<FiberSkeletonCornerBlock>* blocks = nullptr;
    std::size_t targetVertex = 0;
    SmoothnessScales scales;
    Point3 candidateDirection{0.0, 0.0, 0.0};

    template <typename T>
    bool evaluate(
        const T* previousPoint, const T* currentPoint, T* residuals,
        bool* isotropicOut) const
    {
        const std::array<T, 3> candidate{
            T(candidateDirection[0]), T(candidateDirection[1]),
            T(candidateDirection[2])};
        std::array<T, 3> previousDirection{T(0.0), T(0.0), T(0.0)};
        T previousLength = T(0.0);
        stepDirection(
            previousPoint, currentPoint, previousDirection, previousLength);
        smoothnessResidualCore(
            (*blocks)[targetVertex], currentPoint, previousDirection.data(),
            candidate.data(), previousLength, scales, residuals, isotropicOut);
        return true;
    }

    template <typename T>
    bool operator()(
        const T* const previousPoint, const T* const currentPoint,
        T* residuals) const
    {
        return evaluate(previousPoint, currentPoint, residuals, nullptr);
    }
};

struct SpacingResidual {
    double weight = 0.0;

    template <typename T>
    bool operator()(
        const T* const currentPoint, const T* const nextPoint,
        const T* const spacing, T* residuals) const
    {
        std::array<T, 3> step{T(0.0), T(0.0), T(0.0)};
        for (int axis = 0; axis < 3; ++axis)
            step[axis] = nextPoint[axis] - currentPoint[axis];
        const T length = guardedNorm3(step.data());
        const T denominator = spacing[0] > T(kSpacingFloor)
            ? spacing[0]
            : T(kSpacingFloor);
        residuals[0] = T(weight) * (length - spacing[0]) / denominator;
        return true;
    }
};

double checkedSqrt(double value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(
            std::string{name} + " must be finite and nonnegative");
    }
    return std::sqrt(value);
}

Point3 checkedUnitTangent(const cv::Vec3d& tangent, const char* name)
{
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(tangent[axis])) {
            throw std::invalid_argument(
                std::string{name} + " must be finite");
        }
    }
    const double norm = std::sqrt(tangent.dot(tangent));
    if (!(norm > kPredictionNormMin)) {
        throw std::invalid_argument(
            std::string{name} + " must not be degenerate");
    }
    return {tangent[0] / norm, tangent[1] / norm, tangent[2] / norm};
}

struct PreparedConfig {
    SmoothnessScales scales;
    double spacingWeight = 0.0;
    std::optional<Point3> startTangent;
    std::optional<Point3> endTangent;
    int maxIterations = 1;
    double functionTolerance = 1.0e-6;
    bool freeStart = false;
    bool freeEnd = false;
};

PreparedConfig prepareConfig(const FiberSkeletonRefineConfig& config)
{
    PreparedConfig prepared;
    prepared.scales.normal =
        checkedSqrt(config.weights.normal, "Skeleton normal weight");
    prepared.scales.tangent =
        checkedSqrt(config.weights.tangent, "Skeleton tangent weight");
    prepared.scales.isotropic =
        checkedSqrt(config.weights.isotropic, "Skeleton isotropic weight");
    // Like the smoothness scales, the spacing residual carries sqrt(weight) so
    // its squared contribution is `weight * ((L - s) / s)^2` and all four
    // configured weights are directly comparable.
    prepared.spacingWeight =
        checkedSqrt(config.weights.spacing, "Skeleton spacing weight");
    if (config.solver.maxIterations < 1) {
        throw std::invalid_argument(
            "Skeleton refinement requires at least one solver iteration");
    }
    if (!std::isfinite(config.solver.functionTolerance) ||
        config.solver.functionTolerance <= 0.0) {
        throw std::invalid_argument(
            "Skeleton refinement function tolerance must be finite and "
            "positive");
    }
    prepared.maxIterations = config.solver.maxIterations;
    prepared.functionTolerance = config.solver.functionTolerance;
    if (config.startTangent) {
        prepared.startTangent =
            checkedUnitTangent(*config.startTangent, "Skeleton start tangent");
    }
    if (config.endTangent) {
        prepared.endTangent =
            checkedUnitTangent(*config.endTangent, "Skeleton end tangent");
    }
    prepared.freeStart = config.freeStart;
    prepared.freeEnd = config.freeEnd;
    return prepared;
}

std::vector<Point3> checkedParameters(const std::vector<cv::Vec3d>& points)
{
    if (points.size() < 2) {
        throw std::invalid_argument(
            "Skeleton polyline must have at least two points");
    }
    std::vector<Point3> parameters(points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(points[index][axis])) {
                throw std::invalid_argument(
                    "Skeleton polyline points must be finite");
            }
            parameters[index][axis] = points[index][axis];
        }
    }
    return parameters;
}

double polylineLength(const std::vector<Point3>& parameters)
{
    double length = 0.0;
    for (std::size_t step = 0; step + 1 < parameters.size(); ++step) {
        double squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double delta =
                parameters[step + 1][axis] - parameters[step][axis];
            squared += delta * delta;
        }
        length += std::sqrt(squared);
    }
    return length;
}

double initialSpacing(double length, std::size_t stepCount)
{
    const double spacing = length / static_cast<double>(stepCount);
    return spacing > kSpacingFloor ? spacing : kSpacingFloor;
}

// Reloads one corner block per vertex, with the sign reference taken from the
// centered difference of the current geometry.
class CornerBlockLoader
{
public:
    explicit CornerBlockLoader(FiberSkeletonFieldProvider& fields)
        : fields_{fields}
    {
    }

    void load(
        const std::vector<Point3>& parameters,
        std::vector<FiberSkeletonCornerBlock>& blocks)
    {
        const std::size_t count = parameters.size();
        points_.resize(count);
        directions_.resize(count);
        for (std::size_t index = 0; index < count; ++index) {
            points_[index] = cv::Vec3d{
                parameters[index][0], parameters[index][1],
                parameters[index][2]};
        }

        cv::Vec3d fallback{1.0, 0.0, 0.0};
        for (std::size_t step = 0; step + 1 < count; ++step) {
            const cv::Vec3d difference = points_[step + 1] - points_[step];
            const double norm = std::sqrt(difference.dot(difference));
            if (norm > kNormEpsilon) {
                fallback = cv::Vec3d{
                    difference[0] / norm, difference[1] / norm,
                    difference[2] / norm};
                break;
            }
        }
        for (std::size_t index = 0; index < count; ++index) {
            cv::Vec3d difference{0.0, 0.0, 0.0};
            if (count >= 2) {
                if (index == 0)
                    difference = points_[1] - points_[0];
                else if (index + 1 == count)
                    difference = points_[count - 1] - points_[count - 2];
                else
                    difference = points_[index + 1] - points_[index - 1];
            }
            const double norm = std::sqrt(difference.dot(difference));
            directions_[index] = norm > kNormEpsilon
                ? cv::Vec3d{
                      difference[0] / norm, difference[1] / norm,
                      difference[2] / norm}
                : fallback;
        }
        fields_.loadCorners(points_, directions_, blocks);
    }

private:
    FiberSkeletonFieldProvider& fields_;
    std::vector<cv::Vec3d> points_;
    std::vector<cv::Vec3d> directions_;
};

class CornerBlockCallback final : public ceres::EvaluationCallback
{
public:
    CornerBlockCallback(
        FiberSkeletonFieldProvider& fields,
        const std::vector<Point3>& parameters,
        std::vector<FiberSkeletonCornerBlock>& blocks)
        : loader_{fields}, parameters_{parameters}, blocks_{blocks}
    {
    }

    void PrepareForEvaluation(bool, bool new_evaluation_point) override
    {
        if (!new_evaluation_point)
            return;
        loader_.load(parameters_, blocks_);
    }

private:
    CornerBlockLoader loader_;
    const std::vector<Point3>& parameters_;
    std::vector<FiberSkeletonCornerBlock>& blocks_;
};

void accumulateSmoothness(
    FiberSkeletonCostBreakdown& breakdown,
    const std::array<double, 2>& residuals, bool isotropic)
{
    if (isotropic) {
        breakdown.isotropicSmoothness += residuals[0] * residuals[0];
    } else {
        breakdown.tangentSmoothness += residuals[0] * residuals[0];
        breakdown.normalSmoothness += residuals[1] * residuals[1];
    }
}

// The shared scalar evaluator: the same functors the solver uses, instantiated
// on `double`, so the reported breakdown is exactly the sum of the squared
// residuals of each term. Ceres reports `0.5 * sum of squares`, which is why
// the breakdown is computed here instead of read off the summary.
FiberSkeletonCostBreakdown evaluateBreakdown(
    const std::vector<Point3>& parameters,
    const std::vector<FiberSkeletonCornerBlock>& blocks, double spacing,
    const PreparedConfig& prepared)
{
    FiberSkeletonCostBreakdown breakdown;
    const std::size_t pointCount = parameters.size();
    const std::size_t stepCount = pointCount - 1;
    for (std::size_t step = 0; step < stepCount; ++step) {
        double alignment = 0.0;
        std::array<double, 2> smoothness{0.0, 0.0};
        bool isotropic = false;
        if (step == 0) {
            if (prepared.startTangent) {
                AlignmentStartResidual{&blocks, *prepared.startTangent}(
                    parameters[0].data(), parameters[1].data(), &alignment);
                breakdown.alignment += alignment * alignment;
                SmoothnessStartResidual{
                    &blocks, prepared.scales, *prepared.startTangent}
                    .evaluate(
                        parameters[0].data(), parameters[1].data(),
                        smoothness.data(), &isotropic);
                accumulateSmoothness(breakdown, smoothness, isotropic);
            }
        } else {
            AlignmentResidual{&blocks, step}(
                parameters[step - 1].data(), parameters[step].data(),
                parameters[step + 1].data(), &alignment);
            breakdown.alignment += alignment * alignment;
            SmoothnessResidual{&blocks, step + 1, prepared.scales}.evaluate(
                parameters[step - 1].data(), parameters[step].data(),
                parameters[step + 1].data(), smoothness.data(), &isotropic);
            accumulateSmoothness(breakdown, smoothness, isotropic);
        }

        double spacingResidual = 0.0;
        SpacingResidual{prepared.spacingWeight}(
            parameters[step].data(), parameters[step + 1].data(), &spacing,
            &spacingResidual);
        breakdown.spacing += spacingResidual * spacingResidual;
    }

    if (prepared.endTangent) {
        std::array<double, 2> smoothness{0.0, 0.0};
        bool isotropic = false;
        SmoothnessEndResidual{
            &blocks, pointCount - 1, prepared.scales, *prepared.endTangent}
            .evaluate(
                parameters[pointCount - 2].data(),
                parameters[pointCount - 1].data(), smoothness.data(),
                &isotropic);
        accumulateSmoothness(breakdown, smoothness, isotropic);
    }
    return breakdown;
}

}  // namespace

FiberSkeletonRefineResult refineFixedEndpointPolyline(
    const std::vector<cv::Vec3d>& initialPointsBaseXYZ,
    FiberSkeletonFieldProvider& fields, const FiberSkeletonRefineConfig& config)
{
    const PreparedConfig prepared = prepareConfig(config);
    std::vector<Point3> parameters = checkedParameters(initialPointsBaseXYZ);
    const std::size_t pointCount = parameters.size();
    const std::size_t stepCount = pointCount - 1;

    FiberSkeletonRefineResult result;
    result.initialLengthBase = polylineLength(parameters);
    double spacing = initialSpacing(result.initialLengthBase, stepCount);

    std::vector<FiberSkeletonCornerBlock> blocks;
    CornerBlockLoader loader{fields};
    loader.load(parameters, blocks);
    result.initialCost =
        evaluateBreakdown(parameters, blocks, spacing, prepared);

    CornerBlockCallback callback{fields, parameters, blocks};
    ceres::Problem::Options problemOptions;
    problemOptions.evaluation_callback = &callback;
    ceres::Problem problem{problemOptions};
    for (std::size_t index = 0; index < pointCount; ++index)
        problem.AddParameterBlock(parameters[index].data(), 3);
    problem.AddParameterBlock(&spacing, 1);
    if (!prepared.freeStart)
        problem.SetParameterBlockConstant(parameters.front().data());
    if (!prepared.freeEnd)
        problem.SetParameterBlockConstant(parameters.back().data());

    for (std::size_t step = 0; step < stepCount; ++step) {
        if (step == 0) {
            if (prepared.startTangent) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<
                        AlignmentStartResidual, 1, 3, 3>(
                            new AlignmentStartResidual{
                                &blocks, *prepared.startTangent}),
                    nullptr, parameters[0].data(), parameters[1].data());
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<
                        SmoothnessStartResidual, 2, 3, 3>(
                            new SmoothnessStartResidual{
                                &blocks, prepared.scales,
                                *prepared.startTangent}),
                    nullptr, parameters[0].data(), parameters[1].data());
            }
        } else {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<AlignmentResidual, 1, 3, 3, 3>(
                    new AlignmentResidual{&blocks, step}),
                nullptr, parameters[step - 1].data(), parameters[step].data(),
                parameters[step + 1].data());
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<
                    SmoothnessResidual, 2, 3, 3, 3>(
                        new SmoothnessResidual{
                            &blocks, step + 1, prepared.scales}),
                nullptr, parameters[step - 1].data(), parameters[step].data(),
                parameters[step + 1].data());
        }
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SpacingResidual, 1, 3, 3, 1>(
                new SpacingResidual{prepared.spacingWeight}),
            nullptr, parameters[step].data(), parameters[step + 1].data(),
            &spacing);
    }
    if (prepared.endTangent) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SmoothnessEndResidual, 2, 3, 3>(
                new SmoothnessEndResidual{
                    &blocks, pointCount - 1, prepared.scales,
                    *prepared.endTangent}),
            nullptr, parameters[pointCount - 2].data(),
            parameters[pointCount - 1].data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    options.num_threads = 1;
    options.max_num_iterations = prepared.maxIterations;
    options.function_tolerance = prepared.functionTolerance;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;
    options.update_state_every_iteration = true;

    const auto started = std::chrono::steady_clock::now();
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    result.solveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();

    result.pointsBaseXYZ.resize(pointCount);
    for (std::size_t index = 0; index < pointCount; ++index) {
        result.pointsBaseXYZ[index] = cv::Vec3d{
            parameters[index][0], parameters[index][1], parameters[index][2]};
    }
    loader.load(parameters, blocks);
    result.finalCost = evaluateBreakdown(parameters, blocks, spacing, prepared);
    result.lengthBase = polylineLength(parameters);
    result.spacingBase = spacing;
    result.iterations = std::max(
        0, static_cast<int>(summary.iterations.size()) - 1);
    result.converged = summary.termination_type == ceres::CONVERGENCE;
    result.terminationMessage = summary.BriefReport();
    return result;
}

FiberSkeletonCostBreakdown evaluateFiberSkeletonPolylineCost(
    const std::vector<cv::Vec3d>& pointsBaseXYZ,
    FiberSkeletonFieldProvider& fields, const FiberSkeletonRefineConfig& config)
{
    const PreparedConfig prepared = prepareConfig(config);
    const std::vector<Point3> parameters = checkedParameters(pointsBaseXYZ);
    const double spacing =
        initialSpacing(polylineLength(parameters), parameters.size() - 1);

    std::vector<FiberSkeletonCornerBlock> blocks;
    CornerBlockLoader loader{fields};
    loader.load(parameters, blocks);
    return evaluateBreakdown(parameters, blocks, spacing, prepared);
}

}  // namespace vc::fiber_tracer
