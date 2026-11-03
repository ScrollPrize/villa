#pragma once
// Shared CPU/device scalar math. Extracted from native scorer; no fast math.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <utility>
#ifdef __CUDACC__
#define VC_TRACE_HD __host__ __device__
#else
#define VC_TRACE_HD
#endif
namespace vc::fiber_tracer {
enum class FiberLocalSmoothnessMode {
    None,
    NormalAware,
    IsotropicFallback,
};

struct FiberLocalSmoothnessConfig {
    float isotropicWeight = 2.0f;
    float normalWeight = 0.1f;
    float tangentWeight = 10.0f;
    float freeAngleRadians = 0.0f;
};

template<class Scalar> struct BasicFiberLocalSmoothnessCost {
    Scalar isotropic = 0.0f;
    Scalar normal = 0.0f;
    Scalar tangent = 0.0f;
    FiberLocalSmoothnessMode mode = FiberLocalSmoothnessMode::None;

    VC_TRACE_HD Scalar total() const noexcept { return isotropic + normal + tangent; }
};

using FiberLocalSmoothnessCost = BasicFiberLocalSmoothnessCost<float>;

namespace portable {
template<class V> using ScalarOf = decltype(std::declval<V>().dot(std::declval<V>()));
// ADL permits analytic scalar types while preserving the existing float path.
template<class S> VC_TRACE_HD auto mathSqrt(S x) { using std::sqrt; return sqrt(x); }
template<class S> VC_TRACE_HD auto mathAcos(S x) { using std::acos; return acos(x); }
template<class S> VC_TRACE_HD auto mathAsin(S x) { using std::asin; return asin(x); }
template<class S> VC_TRACE_HD auto mathAbs(S x) { using std::abs; return abs(x); }
template<class S> VC_TRACE_HD bool mathFinite(S x) { using std::isfinite; return isfinite(x); }

template<class V> VC_TRACE_HD V traceUnit(const V& v,const V& fallback) {
    const float len=std::sqrt(v.dot(v));
    return len>1.0e-6f && std::isfinite(len)?v/len:fallback;
}
template<class V> VC_TRACE_HD std::array<V,2> traceBasis(const V& direction) {
    const V axis=traceUnit(direction,V{1,0,0});
    const std::array<V,3> refs={V{1,0,0},V{0,1,0},V{0,0,1}};
    size_t r=0;float best=std::abs(axis.dot(refs[0]));
    for(size_t i=1;i<refs.size();++i) {
        const float value=std::abs(axis.dot(refs[i]));
        if(value<best){best=value;r=i;}
    }
    const V b0=traceUnit(axis.cross(refs[r]),V{0,0,0});
    return {b0,traceUnit(axis.cross(b0),V{0,0,0})};
}
// Caller validates reference/tolerance; invalid candidate width is no evidence.
VC_TRACE_HD inline float widthPresenceScale(double width,double mean,double tolerance,bool cosine) {
    if(tolerance==0)return 1;
    if(!std::isfinite(width)||width<=0)return 0;
    const double distance=std::abs(width/mean-1)/tolerance;
    if(cosine)return distance>=1?0.f:static_cast<float>(std::cos(1.57079632679489661923*distance));
    return static_cast<float>(std::clamp(1-distance,0.,1.));
}
inline constexpr float kFiberLocalEpsilon = 1.0e-6f;
enum class FiberLocalPreparedCandidateSmoothnessMode : std::uint8_t {
    InvalidDirection,
    IsotropicFallback,
    NormalAwareDegenerateTangent,
    NormalAware,
};

template<class V> struct FiberLocalPreparedCandidateSmoothness {
    V normal{0.0f, 0.0f, 0.0f};
    V tangent{0.0f, 0.0f, 0.0f};
    ScalarOf<V> normalAngle = 0.0f;
    FiberLocalPreparedCandidateSmoothnessMode mode =
        FiberLocalPreparedCandidateSmoothnessMode::InvalidDirection;
};


template<class S> VC_TRACE_HD static inline S clampFiberLocalUnit(S value)
{
    if (!mathFinite(value))
        return 0.0f;
    return std::clamp(value, S(-1.0f), S(1.0f));
}

template<class S> VC_TRACE_HD static inline S clampFiberLocalPositiveUnit(S value)
{
    if (!mathFinite(value))
        return 0.0f;
    return std::clamp(value, S(0.0f), S(1.0f));
}

template<class S> VC_TRACE_HD static inline S clampFiberLocalFinitePositiveUnit(S value)
{
    return std::min(std::max(value, S(0.0f)), S(1.0f));
}

template<class V>
VC_TRACE_HD static inline V finiteFiberLocalOrZero(const V& value)
{
    if (!mathFinite(value[0]) || !mathFinite(value[1]) ||
        !mathFinite(value[2])) {
        return {0.0f, 0.0f, 0.0f};
    }
    return value;
}

template<class V>
VC_TRACE_HD static inline V normalizeFiberLocalOrZero(const V& value)
{
    const auto length = mathSqrt(value.dot(value));
    if (!(length > kFiberLocalEpsilon) || !mathFinite(length))
        return {0.0f, 0.0f, 0.0f};
    return value / length;
}

template<class V>
VC_TRACE_HD static inline ScalarOf<V> fiberLocalAngleBetweenUnit(
    const V& left, const V& right)
{
    return mathAcos(clampFiberLocalUnit(left.dot(right)));
}

template<class S> VC_TRACE_HD static inline S fiberLocalExcessAngleSquared(S angle, float freeAngle)
{
    const auto excess = std::max(S(0.0f), angle - freeAngle);
    return excess * excess;
}

template<class V>
VC_TRACE_HD static inline ScalarOf<V> fiberLocalAlignmentLossInline(
    ScalarOf<V> presence,
    const V& previousStepDirection,
    const V& candidateStepDirection,
    const V& currentPredictionDirection,
    const V& candidatePredictionDirection,
    std::array<ScalarOf<V>, 6>* factors = nullptr, bool boundUnknownPrediction = false)
{
    size_t index = 0;
    const auto factor = [&](ScalarOf<V> value) {
        const auto f = clampFiberLocalPositiveUnit(value);
        if (factors) (*factors)[index++] = f;
        return f;
    };
    auto score = clampFiberLocalPositiveUnit(presence);
    score *= factor(
        previousStepDirection.dot(candidateStepDirection));
    score *= factor(
        previousStepDirection.dot(currentPredictionDirection));
    score *= boundUnknownPrediction ? 1.f : factor(
        previousStepDirection.dot(candidatePredictionDirection));
    score *= factor(
        currentPredictionDirection.dot(candidateStepDirection));
    score *= boundUnknownPrediction ? 1.f : factor(
        currentPredictionDirection.dot(candidatePredictionDirection));
    score *= boundUnknownPrediction ? 1.f : factor(
        candidateStepDirection.dot(candidatePredictionDirection));
    return 1.0f - score;
}

template<class V>
VC_TRACE_HD static inline FiberLocalPreparedCandidateSmoothness<V>
prepareFiberLocalCandidateSmoothnessInline(
    const V& candidateStepDirection,
    const V& normal,
    bool normalValid)
{
    FiberLocalPreparedCandidateSmoothness<V> prepared;
    prepared.normal = normal;
    constexpr float epsilon2 = kFiberLocalEpsilon * kFiberLocalEpsilon;
    if (candidateStepDirection.dot(candidateStepDirection) <= epsilon2)
        return prepared;
    if (!normalValid || normal.dot(normal) <= epsilon2) {
        prepared.mode =
            FiberLocalPreparedCandidateSmoothnessMode::IsotropicFallback;
        return prepared;
    }

    const auto candidateNormal = clampFiberLocalUnit(
        candidateStepDirection.dot(normal));
    prepared.tangent = normalizeFiberLocalOrZero(
        candidateStepDirection - normal * candidateNormal);
    prepared.normalAngle = mathAsin(candidateNormal);
    prepared.mode = prepared.tangent.dot(prepared.tangent) > epsilon2
        ? FiberLocalPreparedCandidateSmoothnessMode::NormalAware
        : FiberLocalPreparedCandidateSmoothnessMode::NormalAwareDegenerateTangent;
    return prepared;
}

template<class V>
VC_TRACE_HD static inline BasicFiberLocalSmoothnessCost<ScalarOf<V>>
fiberLocalSmoothnessCostCandidatePreparedInline(
    const V& previousStepDirection,
    const V& candidateStepDirection,
    const FiberLocalPreparedCandidateSmoothness<V>& candidatePrepared,
    const FiberLocalSmoothnessConfig& config)
{
    BasicFiberLocalSmoothnessCost<ScalarOf<V>> cost;
    constexpr float epsilon2 = kFiberLocalEpsilon * kFiberLocalEpsilon;
    if (previousStepDirection.dot(previousStepDirection) <= epsilon2 ||
        candidatePrepared.mode ==
            FiberLocalPreparedCandidateSmoothnessMode::InvalidDirection) {
        return cost;
    }

    if (candidatePrepared.mode ==
        FiberLocalPreparedCandidateSmoothnessMode::IsotropicFallback) {
        const auto isotropicAngle = fiberLocalAngleBetweenUnit(
            previousStepDirection, candidateStepDirection);
        cost.isotropic = config.isotropicWeight *
                         fiberLocalExcessAngleSquared(
                             isotropicAngle, config.freeAngleRadians);
        cost.mode = FiberLocalSmoothnessMode::IsotropicFallback;
        return cost;
    }

    const auto previousNormal = clampFiberLocalUnit(
        previousStepDirection.dot(candidatePrepared.normal));
    const V previousTangent = normalizeFiberLocalOrZero(
        previousStepDirection - candidatePrepared.normal * previousNormal);
    const bool tangentValid =
        previousTangent.dot(previousTangent) > epsilon2 &&
        candidatePrepared.mode ==
            FiberLocalPreparedCandidateSmoothnessMode::NormalAware;
    const auto tangentAngle = tangentValid
        ? fiberLocalAngleBetweenUnit(
              previousTangent, candidatePrepared.tangent)
        : fiberLocalAngleBetweenUnit(
              previousStepDirection, candidateStepDirection);
    const ScalarOf<V> normalAngle = mathAbs(
        candidatePrepared.normalAngle - mathAsin(previousNormal));
    cost.tangent = config.tangentWeight * fiberLocalExcessAngleSquared(
        tangentAngle, config.freeAngleRadians);
    cost.normal = config.normalWeight * fiberLocalExcessAngleSquared(
        normalAngle, config.freeAngleRadians);
    cost.mode = FiberLocalSmoothnessMode::NormalAware;
    return cost;
}

template<class V>
VC_TRACE_HD static inline BasicFiberLocalSmoothnessCost<ScalarOf<V>> fiberLocalSmoothnessCostInline(
    const V& previousStepDirection,
    const V& candidateStepDirection,
    const V& normal,
    bool normalValid,
    const FiberLocalSmoothnessConfig& config)
{
    constexpr float epsilon2 = kFiberLocalEpsilon * kFiberLocalEpsilon;
    if (previousStepDirection.dot(previousStepDirection) <= epsilon2 ||
        candidateStepDirection.dot(candidateStepDirection) <= epsilon2) {
        return {};
    }
    return fiberLocalSmoothnessCostCandidatePreparedInline(
        previousStepDirection, candidateStepDirection,
        prepareFiberLocalCandidateSmoothnessInline(
            candidateStepDirection, normal, normalValid),
        config);
}

} // namespace portable
} // namespace vc::fiber_tracer
