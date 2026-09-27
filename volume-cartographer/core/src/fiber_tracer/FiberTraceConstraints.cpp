#include "vc/fiber_tracer/FiberTraceConstraints.hpp"

#include "vc/fiber_tracer/LasagnaNormalAlignment.hpp"

#include "vc/core/io/PolylineObj.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-12;
using Point3 = bg::model::point<double, 3, bg::cs::cartesian>;
using Box3 = bg::model::box<Point3>;

struct SampleRecord {
    std::size_t piece = 0;
    std::size_t sample = 0;
};

using RTreeValue = std::pair<Point3, std::size_t>;
using PointTree = bgi::rtree<RTreeValue, bgi::quadratic<32>>;

struct PiecePair {
    std::size_t a = 0;
    std::size_t b = 0;

    friend bool operator<(const PiecePair& left, const PiecePair& right)
    {
        return std::tie(left.a, left.b) < std::tie(right.a, right.b);
    }
};

struct PairCandidate {
    PiecePair pieces;
    std::size_t sampleA = 0;
    std::size_t sampleB = 0;
    std::size_t globalSampleA = 0;
    std::size_t globalSampleB = 0;
    double distance = std::numeric_limits<double>::infinity();
    double arcA = -1.0, arcB = -1.0;
};

enum class RejectReason { None, Tangent, Winding, WindingCutoff, Consistency };

struct ScoredCandidate {
    std::optional<FiberTraceConstraint> constraint;
    std::vector<std::pair<cv::Vec3d, cv::Vec3d>> parallelConnectors;
    RejectReason rejection = RejectReason::None;
    PairCandidate candidate;
};

struct SignedWindingSample {
    double value = 0.0;
    double absoluteAlignment = 0.0;
    std::size_t component = 0;
};

double length(const cv::Vec3d& value)
{
    return std::sqrt(std::max(0.0, value.dot(value)));
}

cv::Vec3d normalizedOrZero(const cv::Vec3d& value)
{
    const double magnitude = length(value);
    if (!(magnitude > kEpsilon) || !std::isfinite(magnitude))
        return {0.0, 0.0, 0.0};
    return value * (1.0 / magnitude);
}

double median(std::vector<double> values)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const std::size_t upper = values.size() / 2;
    if (values.size() % 2 != 0)
        return values[upper];
    return 0.5 * (values[upper - 1] + values[upper]);
}

std::optional<SignedWindingSample> signedWindingSample(
    const std::pair<cv::Vec3d, cv::Vec3d>& connector,
    double windingDistance,
    const LasagnaNormalAlignmentField& alignedNormals)
{
    if (!std::isfinite(windingDistance))
        return std::nullopt;
    const cv::Vec3d delta = connector.second - connector.first;
    const double connectorLength = cv::norm(delta);
    if (!(connectorLength > kEpsilon) || !std::isfinite(connectorLength))
        return std::nullopt;
    const cv::Vec3d midpoint = 0.5 * (connector.first + connector.second);
    const auto atA = alignedNormals.nearest(connector.first);
    const auto atMidpoint = alignedNormals.nearest(midpoint);
    const auto atB = alignedNormals.nearest(connector.second);
    if (!atA || !atMidpoint || !atB ||
        atA->component != atMidpoint->component ||
        atB->component != atMidpoint->component) {
        return std::nullopt;
    }
    const double signedAlignment =
        (delta / connectorLength).dot(cv::Vec3d{
            atMidpoint->normal[0],
            atMidpoint->normal[1],
            atMidpoint->normal[2],
        });
    if (!std::isfinite(signedAlignment) ||
        std::abs(signedAlignment) <= kEpsilon) {
        return std::nullopt;
    }
    return SignedWindingSample{
        std::copysign(windingDistance, signedAlignment),
        std::clamp(std::abs(signedAlignment), 0.0, 1.0),
        atMidpoint->component,
    };
}

Point3 point3(const cv::Vec3d& point)
{
    return {point[0], point[1], point[2]};
}

Box3 queryBox(const cv::Vec3d& point, double radius)
{
    const cv::Vec3d delta{radius, radius, radius};
    return {point3(point - delta), point3(point + delta)};
}

void validateConfig(const FiberTraceConstraintConfig& config)
{
    const bool valid =
        std::isfinite(config.resampleSpacingBaseVoxels) &&
        config.resampleSpacingBaseVoxels > 0.0 &&
        std::isfinite(config.targetPieceLengthBaseVoxels) &&
        config.targetPieceLengthBaseVoxels > 0.0 &&
        std::isfinite(config.pieceOverlapBaseVoxels) &&
        config.pieceOverlapBaseVoxels >= 0.0 &&
        config.pieceOverlapBaseVoxels < config.targetPieceLengthBaseVoxels &&
        std::isfinite(config.maximumDistanceBaseVoxels) &&
        config.maximumDistanceBaseVoxels >= 0.0 &&
        std::isfinite(config.tangentWindowBaseVoxels) &&
        config.tangentWindowBaseVoxels > 0.0 &&
        std::isfinite(config.phaseRefinementStepFraction) &&
        config.phaseRefinementStepFraction > 0.0 &&
        std::isfinite(config.phaseRefinementLimitFraction) &&
        config.phaseRefinementLimitFraction >=
            config.phaseRefinementStepFraction &&
        std::isfinite(config.correspondenceGridStepFraction) &&
        config.correspondenceGridStepFraction > 0.0 &&
        std::isfinite(config.correspondenceGridLimitFraction) &&
        config.correspondenceGridLimitFraction >=
            config.correspondenceGridStepFraction &&
        config.correspondenceGridLimitFraction < 1.0 &&
        std::isfinite(config.correspondenceGridStepWeight) &&
        config.correspondenceGridStepWeight >= 0.0 &&
        std::isfinite(config.correspondenceGridPerpendicularWeight) &&
        config.correspondenceGridPerpendicularWeight >= 0.0 &&
        std::isfinite(config.correspondenceGridDirectionWeight) &&
        config.correspondenceGridDirectionWeight >= 0.0 &&
        std::isfinite(config.correspondenceGridLengthWeight) &&
        config.correspondenceGridLengthWeight >= 0.0 &&
        config.correspondenceGridStepWeight +
                config.correspondenceGridPerpendicularWeight +
                config.correspondenceGridDirectionWeight +
                config.correspondenceGridLengthWeight >
            0.0 &&
        std::isfinite(config.windingIntegrationStepBaseVoxels) &&
        config.windingIntegrationStepBaseVoxels > 0.0 &&
        std::isfinite(config.maximumWindingDistance) &&
        config.maximumWindingDistance > 0.0;
    if (!valid)
        throw std::invalid_argument("Fiber trace constraint configuration is invalid");
    if (config.localEvents &&
        (!(config.localSpacingBaseVoxels > 0.) || !std::isfinite(config.localSpacingBaseVoxels) ||
         !(config.localTangentSpanBaseVoxels > 0.) || !std::isfinite(config.localTangentSpanBaseVoxels) ||
         !(config.localParallelSpanBaseVoxels >= 2.*config.localTangentSpanBaseVoxels) || !std::isfinite(config.localParallelSpanBaseVoxels) ||
         !(config.localReciprocalToleranceBaseVoxels > 0.) || !std::isfinite(config.localReciprocalToleranceBaseVoxels) ||
         !(config.localAngularToleranceDegrees > 0. && config.localAngularToleranceDegrees < 45.) ||
         !(config.localDistanceTolerance > 0.) || !std::isfinite(config.localDistanceTolerance) ||
         config.localRefinementRounds > 4 || !config.extraPieceCutsBaseVoxels.empty()))
        throw std::invalid_argument("Invalid local constraint event configuration");
}

std::optional<cv::Vec3d> centeredTangent(
    const PolylineArcGeometry& geometry,
    double arc,
    double window)
{
    const double center = std::clamp(arc, 0.0, geometry.length());
    const double before = std::max(0.0, center - 0.5 * window);
    const double after = std::min(geometry.length(), center + 0.5 * window);
    if (!(after > before + kEpsilon))
        return std::nullopt;
    const auto tangent = normalizedOrZero(
        samplePolylineArc(geometry, after).point -
        samplePolylineArc(geometry, before).point);
    if (length(tangent) <= kEpsilon)
        return std::nullopt;
    return tangent;
}

std::vector<std::pair<double, double>> pieceIntervals(
    double lineLength,
    const FiberTraceConstraintConfig& config)
{
    if (!(lineLength > kEpsilon))
        return {};
    std::size_t count = 1;
    if (lineLength > config.targetPieceLengthBaseVoxels + kEpsilon) {
        const double stride = config.targetPieceLengthBaseVoxels -
            config.pieceOverlapBaseVoxels;
        count = static_cast<std::size_t>(std::ceil(
            (lineLength - config.pieceOverlapBaseVoxels) / stride));
        count = std::max<std::size_t>(1, count);
    }
    const double span = (lineLength +
        static_cast<double>(count - 1) * config.pieceOverlapBaseVoxels) /
        static_cast<double>(count);
    const double stride = span - config.pieceOverlapBaseVoxels;
    std::vector<std::pair<double, double>> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const double begin = static_cast<double>(index) * stride;
        const double end = index + 1 == count ? lineLength : begin + span;
        result.emplace_back(begin, end);
    }
    return result;
}

FiberTraceConstraintPiece makePiece(
    std::size_t traceIndex,
    std::size_t pieceIndex,
    const PolylineArcGeometry& geometry,
    double begin,
    double end,
    double spacing)
{
    FiberTraceConstraintPiece piece;
    piece.traceIndex = traceIndex;
    piece.pieceIndex = pieceIndex;
    piece.beginArcBaseVoxels = begin;
    piece.endArcBaseVoxels = end;
    for (double arc = begin; arc < end - kEpsilon; arc += spacing) {
        piece.sampleArcsBaseVoxels.push_back(arc);
        piece.samplePointsBaseXYZ.push_back(samplePolylineArc(geometry, arc).point);
    }
    if (piece.sampleArcsBaseVoxels.empty() ||
        end - piece.sampleArcsBaseVoxels.back() > kEpsilon) {
        piece.sampleArcsBaseVoxels.push_back(end);
        piece.samplePointsBaseXYZ.push_back(samplePolylineArc(geometry, end).point);
    }
    return piece;
}

bool betterCandidate(const PairCandidate& candidate, const PairCandidate& current)
{
    if (candidate.distance < current.distance - kEpsilon)
        return true;
    if (std::abs(candidate.distance - current.distance) > kEpsilon)
        return false;
    return std::tie(candidate.globalSampleA, candidate.globalSampleB) <
        std::tie(current.globalSampleA, current.globalSampleB);
}

struct RefinedCorrespondence {
    double arcA = 0.0;
    double arcB = 0.0;
    cv::Vec3d pointA{0.0, 0.0, 0.0};
    cv::Vec3d pointB{0.0, 0.0, 0.0};
    cv::Vec3d tangentA{0.0, 0.0, 0.0};
    cv::Vec3d tangentB{0.0, 0.0, 0.0};
    double energy = std::numeric_limits<double>::infinity();
    double stepResidual = std::numeric_limits<double>::infinity();
    double offsetMagnitude = std::numeric_limits<double>::infinity();
    double offsetA = 0.0;
    double offsetB = 0.0;
};

bool betterCorrespondence(
    const RefinedCorrespondence& candidate,
    const RefinedCorrespondence& current)
{
    if (candidate.energy < current.energy - kEpsilon)
        return true;
    if (std::abs(candidate.energy - current.energy) > kEpsilon)
        return false;
    if (candidate.stepResidual < current.stepResidual - kEpsilon)
        return true;
    if (std::abs(candidate.stepResidual - current.stepResidual) > kEpsilon)
        return false;
    if (candidate.offsetMagnitude < current.offsetMagnitude - kEpsilon)
        return true;
    if (std::abs(candidate.offsetMagnitude - current.offsetMagnitude) > kEpsilon)
        return false;
    return std::tie(candidate.offsetA, candidate.offsetB) <
        std::tie(current.offsetA, current.offsetB);
}

std::optional<std::pair<double, double>> refinedDistanceArcs(
    const FiberTraceConstraintPiece& pieceA,
    const FiberTraceConstraintPiece& pieceB,
    const PolylineArcGeometry& geometryA,
    const PolylineArcGeometry& geometryB,
    double nominalA,
    double nominalB,
    int walkDirection,
    int orientation,
    double currentPhase,
    double refinementStep,
    double refinementLimit,
    double& selectedPhase)
{
    std::array<double, 4> phases{
        currentPhase,
        std::clamp(currentPhase + refinementStep, -refinementLimit, refinementLimit),
        std::clamp(currentPhase - refinementStep, -refinementLimit, refinementLimit),
        0.0,
    };
    double bestDistance = std::numeric_limits<double>::infinity();
    std::optional<std::pair<double, double>> best;
    for (std::size_t index = 0; index < phases.size(); ++index) {
        const double phase = phases[index];
        if (std::find(phases.begin(), phases.begin() + static_cast<std::ptrdiff_t>(index), phase) !=
            phases.begin() + static_cast<std::ptrdiff_t>(index)) {
            continue;
        }
        const double arcA = nominalA + static_cast<double>(walkDirection) * phase;
        const double arcB = nominalB -
            static_cast<double>(orientation * walkDirection) * phase;
        if (arcA < pieceA.beginArcBaseVoxels - kEpsilon ||
            arcA > pieceA.endArcBaseVoxels + kEpsilon ||
            arcB < pieceB.beginArcBaseVoxels - kEpsilon ||
            arcB > pieceB.endArcBaseVoxels + kEpsilon) {
            continue;
        }
        const double distance = length(
            samplePolylineArc(geometryA, arcA).point -
            samplePolylineArc(geometryB, arcB).point);
        if (!best.has_value() || distance < bestDistance - kEpsilon) {
            best = std::pair{arcA, arcB};
            bestDistance = distance;
            selectedPhase = phase;
        }
    }
    return best;
}

std::optional<RefinedCorrespondence> refinedCorrespondence(
    const FiberTraceConstraintPiece& pieceA,
    const FiberTraceConstraintPiece& pieceB,
    const PolylineArcGeometry& geometryA,
    const PolylineArcGeometry& geometryB,
    double previousA,
    double previousB,
    int walkDirection,
    int orientation,
    double targetStep,
    double gridStep,
    double gridLimit,
    double tangentWindow,
    double stepWeight,
    double perpendicularWeight,
    double directionWeight,
    double lengthWeight)
{
    const int gridRadius = static_cast<int>(std::floor(
        gridLimit / gridStep + kEpsilon));
    std::optional<RefinedCorrespondence> best;
    const cv::Vec3d previousConnector =
        samplePolylineArc(geometryB, previousB).point -
        samplePolylineArc(geometryA, previousA).point;
    const double previousConnectorLength = length(previousConnector);
    const cv::Vec3d previousConnectorDirection =
        normalizedOrZero(previousConnector);
    for (int indexA = -gridRadius; indexA <= gridRadius; ++indexA) {
        const double offsetA = static_cast<double>(indexA) * gridStep;
        const double advanceA = targetStep + offsetA;
        if (!(advanceA > kEpsilon))
            continue;
        const double candidateArcA = previousA +
            static_cast<double>(walkDirection) * advanceA;
        if (candidateArcA < pieceA.beginArcBaseVoxels - kEpsilon ||
            candidateArcA > pieceA.endArcBaseVoxels + kEpsilon) {
            continue;
        }
        const auto tangentA = centeredTangent(
            geometryA, candidateArcA, tangentWindow);
        if (!tangentA.has_value())
            continue;
        const cv::Vec3d pointA =
            samplePolylineArc(geometryA, candidateArcA).point;

        for (int indexB = -gridRadius; indexB <= gridRadius; ++indexB) {
            const double offsetB = static_cast<double>(indexB) * gridStep;
            const double advanceB = targetStep + offsetB;
            if (!(advanceB > kEpsilon))
                continue;
            const double candidateArcB = previousB +
                static_cast<double>(orientation * walkDirection) * advanceB;
            if (candidateArcB < pieceB.beginArcBaseVoxels - kEpsilon ||
                candidateArcB > pieceB.endArcBaseVoxels + kEpsilon) {
                continue;
            }
            const auto tangentB = centeredTangent(
                geometryB, candidateArcB, tangentWindow);
            if (!tangentB.has_value())
                continue;
            const cv::Vec3d pointB =
                samplePolylineArc(geometryB, candidateArcB).point;
            const cv::Vec3d connector = pointB - pointA;
            const double connectorLength = length(connector);
            if (!(connectorLength > kEpsilon))
                continue;
            const cv::Vec3d connectorDirection = connector / connectorLength;
            const double normalizedOffsetA = offsetA / targetStep;
            const double normalizedOffsetB = offsetB / targetStep;
            const double stepResidual =
                normalizedOffsetA * normalizedOffsetA +
                normalizedOffsetB * normalizedOffsetB;
            const double perpendicularA = connectorDirection.dot(*tangentA);
            const double perpendicularB = connectorDirection.dot(*tangentB);
            const double directionChange =
                length(previousConnectorDirection) > kEpsilon
                ? 1.0 - std::clamp(
                      previousConnectorDirection.dot(connectorDirection),
                      -1.0,
                      1.0)
                : 0.0;
            const double normalizedLengthChange =
                (connectorLength - previousConnectorLength) / targetStep;

            RefinedCorrespondence candidate;
            candidate.arcA = candidateArcA;
            candidate.arcB = candidateArcB;
            candidate.pointA = pointA;
            candidate.pointB = pointB;
            candidate.tangentA = *tangentA;
            candidate.tangentB = *tangentB;
            candidate.stepResidual = stepResidual;
            candidate.energy = stepWeight * stepResidual +
                perpendicularWeight * (
                    perpendicularA * perpendicularA +
                    perpendicularB * perpendicularB) +
                directionWeight * directionChange +
                lengthWeight * normalizedLengthChange * normalizedLengthChange;
            candidate.offsetMagnitude =
                std::abs(offsetA) + std::abs(offsetB);
            candidate.offsetA = offsetA;
            candidate.offsetB = offsetB;
            if (!best.has_value() || betterCorrespondence(candidate, *best))
                best = candidate;
        }
    }
    return best;
}

ScoredCandidate scoreLocalCandidate(
    PairCandidate candidate,
    const std::vector<FiberTraceConstraintPiece>& pieces,
    const std::vector<std::optional<PolylineArcGeometry>>& geometries,
    const FiberTraceConstraintConfig& config)
{
    const auto& pa = pieces[candidate.pieces.a];
    const auto& pb = pieces[candidate.pieces.b];
    const auto& ga = *geometries[pa.traceIndex];
    const auto& gb = *geometries[pb.traceIndex];
    double a = candidate.arcA >= 0. ? candidate.arcA : pa.sampleArcsBaseVoxels[candidate.sampleA];
    double b = candidate.arcB >= 0. ? candidate.arcB : pb.sampleArcsBaseVoxels[candidate.sampleB];
    const double h = config.localTangentSpanBaseVoxels;
    const double reach = config.localParallelSpanBaseVoxels;
    const double cosine = std::cos(config.localAngularToleranceDegrees * std::acos(-1.) / 180.);
    const double sine = std::sin(config.localAngularToleranceDegrees * std::acos(-1.) / 180.);
    auto failure = [&]() {
        ScoredCandidate result;
        result.rejection = RejectReason::Consistency;
        result.candidate = candidate;
        result.candidate.arcA = a;
        result.candidate.arcB = b;
        return result;
    };
    const auto tangent = [&](const auto& g, double arc, int side) -> std::optional<cv::Vec3d> {
        if (arc < 0. || arc > g.length() || (side<0 ? arc-h<0. : arc+h>g.length())) return std::nullopt;
        const auto p = samplePolylineArc(g, arc).point;
        auto t = normalizedOrZero(side < 0 ? p-samplePolylineArc(g, arc-h).point : samplePolylineArc(g, arc+h).point-p);
        if (length(t) <= kEpsilon) return std::nullopt;
        return t;
    };
    const auto relation = [&](double aa, double bb) {
        bool parallel = true, orthogonal = true;
        int valid=0;
        for (int sa : {-1,1}) for (int sb : {-1,1}) {
            const auto ta = tangent(ga, aa, sa), tb = tangent(gb, bb, sb);
            if (!ta || !tb) {
                if(config.preserveInputLinesAsPieces) continue;
                return 0;
            }
            ++valid;
            const double dot = std::abs(ta->dot(*tb));
            parallel = parallel && dot >= cosine;
            orthogonal = orthogonal && dot <= sine;
        }
        return valid==0 ? 0 : parallel ? 1 : orthogonal ? 2 : 0;
    };
    int kind = relation(a,b);
    if (kind == 0) return failure();
    const double seedA = a, seedB = b;
    // Alternating bounded projection refines orthogonal encounters. Parallel
    // events keep their A arc: sliding both points would erase repeated samples.
    for (int i=0; i<4; ++i) {
        b = projectPointToPolylineArc(gb, samplePolylineArc(ga,a).point,
            std::max(0.,seedB-reach), std::min(gb.length(),seedB+reach)).arc;
        if (kind == 2)
            a = projectPointToPolylineArc(ga, samplePolylineArc(gb,b).point,
                std::max(0.,seedA-reach), std::min(ga.length(),seedA+reach)).arc;
    }
    if (relation(a,b) != kind) return failure();
    const auto pointA=samplePolylineArc(ga,a).point, pointB=samplePolylineArc(gb,b).point;
    if (length(pointB-pointA) > config.maximumDistanceBaseVoxels) return failure();
    const auto reciprocal = projectPointToPolylineArc(ga, pointB,
        std::max(0.,a-reach), std::min(ga.length(),a+reach));
    if (std::abs(reciprocal.arc-a) > config.localReciprocalToleranceBaseVoxels) return failure();
    const auto ta=tangent(ga,a,1).has_value()?tangent(ga,a,1):tangent(ga,a,-1);
    const auto tb=tangent(gb,b,1).has_value()?tangent(gb,b,1):tangent(gb,b,-1);
    const int orientation = ta->dot(*tb) < 0. ? -1 : 1;
    FiberTraceConstraint c;
    c.pieceA=candidate.pieces.a; c.pieceB=candidate.pieces.b;
    c.arcABaseVoxels=a; c.arcBBaseVoxels=b;
    c.pointABaseXYZ=pointA; c.pointBBaseXYZ=pointB;
    c.closestDistanceBaseVoxels=length(pointB-pointA);
    c.parallelScore=kind==1 ? 1. : 0.; c.perpendicularScore=1.-c.parallelScore;
    c.localEvent=true; c.localOrientation=orientation;
    c.supportABegin=std::max(0.,a-h); c.supportAEnd=std::min(ga.length(),a+h);
    c.supportBBegin=std::max(0.,b-h); c.supportBEnd=std::min(gb.length(),b+h);
    ScoredCandidate result;
    result.parallelConnectors.emplace_back(pointA,pointB);
    // Both halves independently certify the same base-point relationship.
    // Orthogonal connectors use shorter probes, parallel connectors longer walks.
    const double span = kind==1 ? reach : h;
    int certifiedA=0, certifiedB=0;
    for (int side : {-1,1}) {
        const bool fitsA=a+side*span-h>=0. && a+side*span+h<=ga.length();
        const bool fitsB=b+orientation*side*span-h>=0. && b+orientation*side*span+h<=gb.length();
        if(!fitsA || !fitsB) {
            if(!config.preserveInputLinesAsPieces) return failure();
            if(kind==1 || (!fitsA && !fitsB))continue;
        }
        double previousB=b;
        for (double distance : {span*.5, span}) {
            const double aa=a+side*distance;
            const double nominalB=b+orientation*side*distance;
            double bb=nominalB;
            if (kind==1) {
                bb=projectPointToPolylineArc(gb,samplePolylineArc(ga,aa).point,
                    std::max(0.,nominalB-h),std::min(gb.length(),nominalB+h)).arc;
                if (orientation*side*(bb-previousB)<=0. || relation(aa,bb)!=1) return failure();
                const auto back=projectPointToPolylineArc(ga,samplePolylineArc(gb,bb).point,
                    std::max(0.,aa-h),std::min(ga.length(),aa+h));
                if (std::abs(back.arc-aa)>config.localReciprocalToleranceBaseVoxels) return failure();
                previousB=bb;
            } else {
                // Probe each trace independently around a localized crossing;
                // moving both does not define parallel correspondence here.
                bb=b;
                if ((fitsA && relation(aa,b)!=2) || (fitsB && relation(a,nominalB)!=2)) return failure();
            }
            if(fitsA) {
                const auto qa=samplePolylineArc(ga,aa).point, qb=samplePolylineArc(gb,bb).point;
                if (length(qb-qa)>config.maximumDistanceBaseVoxels) return failure();
                result.parallelConnectors.emplace_back(qa,qb);
                c.supportABegin=std::min(c.supportABegin,aa-h); c.supportAEnd=std::max(c.supportAEnd,aa+h);
                c.supportBBegin=std::min(c.supportBBegin,std::max(0.,bb-h)); c.supportBEnd=std::max(c.supportBEnd,std::min(gb.length(),bb+h));
            }
            if(kind==2 && fitsB) {
                if(length(samplePolylineArc(gb,nominalB).point-pointA)>config.maximumDistanceBaseVoxels) return failure();
                result.parallelConnectors.emplace_back(pointA,samplePolylineArc(gb,nominalB).point);
                c.supportBBegin=std::min(c.supportBBegin,nominalB-h); c.supportBEnd=std::max(c.supportBEnd,nominalB+h);
            }
        }
        certifiedA+=fitsA; certifiedB+=fitsB;
    }
    if(!certifiedA || !certifiedB)return failure();
    c.parallelCorrespondenceSamples=result.parallelConnectors.size();
    result.constraint=c;
    result.candidate=candidate; result.candidate.arcA=a; result.candidate.arcB=b;
    return result;
}

ScoredCandidate scoreCandidate(
    const PairCandidate& candidate,
    const std::vector<FiberTraceConstraintPiece>& pieces,
    const std::vector<std::optional<PolylineArcGeometry>>& geometries,
    const FiberTraceConstraintConfig& config)
{
    if (config.localEvents)
        return scoreLocalCandidate(candidate,pieces,geometries,config);
    const auto& pieceA = pieces[candidate.pieces.a];
    const auto& pieceB = pieces[candidate.pieces.b];
    const auto& geometryA = *geometries[pieceA.traceIndex];
    const auto& geometryB = *geometries[pieceB.traceIndex];
    const double arcA = pieceA.sampleArcsBaseVoxels[candidate.sampleA];
    const double arcB = pieceB.sampleArcsBaseVoxels[candidate.sampleB];
    const auto initialA = centeredTangent(
        geometryA, arcA, config.tangentWindowBaseVoxels);
    const auto initialB = centeredTangent(
        geometryB, arcB, config.tangentWindowBaseVoxels);
    if (!initialA.has_value() || !initialB.has_value())
        return {std::nullopt, {}, RejectReason::Tangent};

    const double initialDot = std::clamp(initialA->dot(*initialB), -1.0, 1.0);
    const int orientation = initialDot < 0.0 ? -1 : 1;
    double parallelSum = std::clamp(
        initialA->dot(*initialB * static_cast<double>(orientation)), -1.0, 1.0);
    std::size_t parallelCount = 1;
    std::vector<std::pair<cv::Vec3d, cv::Vec3d>> parallelConnectors;
    parallelConnectors.emplace_back(
        pieceA.samplePointsBaseXYZ[candidate.sampleA],
        pieceB.samplePointsBaseXYZ[candidate.sampleB]);
    double advanceResidualSum = 0.0;
    double perpendicularResidualSum = 0.0;
    double lengthChangeSum = 0.0;
    double directionChangeSum = 0.0;
    std::size_t diagnosticSteps = 0;
    std::size_t perpendicularSamples = 0;
    std::size_t limitHits = 0;
    const auto recordPerpendicular = [&](
                                         const cv::Vec3d& connector,
                                         const cv::Vec3d& tangentA,
                                         const cv::Vec3d& tangentB) {
        const cv::Vec3d direction = normalizedOrZero(connector);
        if (length(direction) <= kEpsilon)
            return;
        perpendicularResidualSum += 0.5 * (
            std::abs(direction.dot(tangentA)) +
            std::abs(direction.dot(tangentB)));
        ++perpendicularSamples;
    };
    if (config.collectParallelCorrespondenceDiagnostics) {
        recordPerpendicular(
            parallelConnectors.front().second - parallelConnectors.front().first,
            *initialA,
            *initialB);
    }
    const auto recordStep = [&](
                                double previousA,
                                double previousB,
                                double nextA,
                                double nextB,
                                const cv::Vec3d& tangentA,
                                const cv::Vec3d& tangentB,
                                const std::pair<cv::Vec3d, cv::Vec3d>& previous,
                                const std::pair<cv::Vec3d, cv::Vec3d>& next,
                                bool hitLimit) {
        const double target = config.resampleSpacingBaseVoxels;
        advanceResidualSum += 0.5 * (
            std::abs(std::abs(nextA - previousA) - target) +
            std::abs(std::abs(nextB - previousB) - target)) / target;
        const cv::Vec3d previousConnector = previous.second - previous.first;
        const cv::Vec3d nextConnector = next.second - next.first;
        const double previousLength = length(previousConnector);
        const double nextLength = length(nextConnector);
        lengthChangeSum += std::abs(nextLength - previousLength) / target;
        const cv::Vec3d previousDirection = normalizedOrZero(previousConnector);
        const cv::Vec3d nextDirection = normalizedOrZero(nextConnector);
        if (length(previousDirection) > kEpsilon &&
            length(nextDirection) > kEpsilon) {
            directionChangeSum += 1.0 - std::clamp(
                previousDirection.dot(nextDirection), -1.0, 1.0);
        }
        recordPerpendicular(nextConnector, tangentA, tangentB);
        limitHits += hitLimit;
        ++diagnosticSteps;
    };
    const double refinementStep = config.resampleSpacingBaseVoxels *
        config.phaseRefinementStepFraction;
    const double refinementLimit = config.resampleSpacingBaseVoxels *
        config.phaseRefinementLimitFraction;
    const double gridStep = config.resampleSpacingBaseVoxels *
        config.correspondenceGridStepFraction;
    const double gridLimit = config.resampleSpacingBaseVoxels *
        config.correspondenceGridLimitFraction;

    for (const int walkDirection : {-1, 1}) {
        if (config.parallelCorrespondence ==
            FiberTraceParallelCorrespondence::Distance) {
            double phase = 0.0;
            double previousA = arcA;
            double previousB = arcB;
            auto previousConnector = parallelConnectors.front();
            for (std::size_t step = 1;; ++step) {
                const double distance = static_cast<double>(step) *
                    config.resampleSpacingBaseVoxels;
                const double nominalA = arcA +
                    static_cast<double>(walkDirection) * distance;
                const double nominalB = arcB +
                    static_cast<double>(orientation * walkDirection) * distance;
                if (nominalA < pieceA.beginArcBaseVoxels - kEpsilon ||
                    nominalA > pieceA.endArcBaseVoxels + kEpsilon ||
                    nominalB < pieceB.beginArcBaseVoxels - kEpsilon ||
                    nominalB > pieceB.endArcBaseVoxels + kEpsilon) {
                    break;
                }
                double selectedPhase = phase;
                const auto arcs = refinedDistanceArcs(
                    pieceA,
                    pieceB,
                    geometryA,
                    geometryB,
                    nominalA,
                    nominalB,
                    walkDirection,
                    orientation,
                    phase,
                    refinementStep,
                    refinementLimit,
                    selectedPhase);
                if (!arcs.has_value())
                    break;
                phase = selectedPhase;
                const auto tangentA = centeredTangent(
                    geometryA, arcs->first, config.tangentWindowBaseVoxels);
                const auto tangentB = centeredTangent(
                    geometryB, arcs->second, config.tangentWindowBaseVoxels);
                if (!tangentA.has_value() || !tangentB.has_value())
                    continue;
                parallelSum += std::clamp(
                    tangentA->dot(*tangentB * static_cast<double>(orientation)),
                    -1.0,
                    1.0);
                const std::pair connector{
                    samplePolylineArc(geometryA, arcs->first).point,
                    samplePolylineArc(geometryB, arcs->second).point};
                if (config.collectParallelCorrespondenceDiagnostics) {
                    recordStep(
                        previousA,
                        previousB,
                        arcs->first,
                        arcs->second,
                        *tangentA,
                        *tangentB,
                        previousConnector,
                        connector,
                        std::abs(selectedPhase) >= refinementLimit - kEpsilon);
                }
                previousA = arcs->first;
                previousB = arcs->second;
                previousConnector = connector;
                parallelConnectors.push_back(connector);
                ++parallelCount;
            }
        } else {
            double previousA = arcA;
            double previousB = arcB;
            auto previousConnector = parallelConnectors.front();
            for (;;) {
                const auto correspondence = refinedCorrespondence(
                    pieceA,
                    pieceB,
                    geometryA,
                    geometryB,
                    previousA,
                    previousB,
                    walkDirection,
                    orientation,
                    config.resampleSpacingBaseVoxels,
                    gridStep,
                    gridLimit,
                    config.tangentWindowBaseVoxels,
                    config.correspondenceGridStepWeight,
                    config.correspondenceGridPerpendicularWeight,
                    config.correspondenceGridDirectionWeight,
                    config.correspondenceGridLengthWeight);
                if (!correspondence.has_value())
                    break;
                parallelSum += std::clamp(
                    correspondence->tangentA.dot(
                        correspondence->tangentB * static_cast<double>(orientation)),
                    -1.0,
                    1.0);
                const std::pair connector{
                    correspondence->pointA,
                    correspondence->pointB};
                if (config.collectParallelCorrespondenceDiagnostics) {
                    recordStep(
                        previousA,
                        previousB,
                        correspondence->arcA,
                        correspondence->arcB,
                        correspondence->tangentA,
                        correspondence->tangentB,
                        previousConnector,
                        connector,
                        std::abs(correspondence->offsetA) >= gridLimit - kEpsilon ||
                            std::abs(correspondence->offsetB) >= gridLimit - kEpsilon);
                }
                ++parallelCount;
                previousA = correspondence->arcA;
                previousB = correspondence->arcB;
                previousConnector = connector;
                parallelConnectors.push_back(connector);
            }
        }
    }

    const double rawParallel = std::clamp(
        parallelSum / static_cast<double>(parallelCount), 0.0, 1.0);
    const double rawPerpendicular = 1.0 - std::abs(initialDot);
    const double evidence = rawParallel + rawPerpendicular;
    if (!(evidence > kEpsilon) || !std::isfinite(evidence))
        return {std::nullopt, {}, RejectReason::Tangent};

    const cv::Vec3d pointA = pieceA.samplePointsBaseXYZ[candidate.sampleA];
    const cv::Vec3d pointB = pieceB.samplePointsBaseXYZ[candidate.sampleB];
    FiberTraceConstraint constraint;
    constraint.pieceA = candidate.pieces.a;
    constraint.pieceB = candidate.pieces.b;
    constraint.arcABaseVoxels = arcA;
    constraint.arcBBaseVoxels = arcB;
    constraint.pointABaseXYZ = pointA;
    constraint.pointBBaseXYZ = pointB;
    constraint.closestDistanceBaseVoxels = candidate.distance;
    constraint.parallelScore = rawParallel / evidence;
    constraint.perpendicularScore = rawPerpendicular / evidence;
    constraint.windingDistance = 0.0;
    constraint.parallelCorrespondenceSamples =
        config.collectParallelCorrespondenceDiagnostics ? parallelCount : 0;
    if (diagnosticSteps != 0) {
        const double inverse = 1.0 / static_cast<double>(diagnosticSteps);
        constraint.parallelMeanAdvanceResidualFraction =
            advanceResidualSum * inverse;
        constraint.parallelMeanConnectorLengthChangeFraction =
            lengthChangeSum * inverse;
        constraint.parallelMeanConnectorDirectionChange =
            directionChangeSum * inverse;
        constraint.parallelLimitHitFraction =
            static_cast<double>(limitHits) * inverse;
    }
    if (perpendicularSamples != 0) {
        constraint.parallelMeanConnectorTangentAbsDot =
            perpendicularResidualSum /
            static_cast<double>(perpendicularSamples);
    }
    if (constraint.parallelScore <= constraint.perpendicularScore)
        parallelConnectors.resize(1);
    return {
        std::move(constraint),
        std::move(parallelConnectors),
        RejectReason::None,
    };
}

void deduplicateLocalEvents(std::vector<ScoredCandidate>& scored)
{
    // Projection can discover the same encounter from several nearby seeds.
    // Quantization is ONLY an index: test actual arc distances in adjacent bins.
    using Key = std::tuple<std::size_t,std::size_t,long long,long long>;
    std::map<Key,std::vector<std::size_t>> cells;
    constexpr double tolerance=8.;
    std::vector<ScoredCandidate> unique;
    for (auto& s : scored) {
        if (!s.constraint) { unique.push_back(std::move(s)); continue; }
        const auto& c=*s.constraint;
        const auto a=static_cast<long long>(std::floor(c.arcABaseVoxels/tolerance));
        const auto b=static_cast<long long>(std::floor(c.arcBBaseVoxels/tolerance));
        bool duplicate=false;
        for (int da=-1;da<=1;++da) for(int db=-1;db<=1;++db) {
            auto it=cells.find({c.pieceA,c.pieceB,a+da,b+db});
            if(it==cells.end()) continue;
            for(auto i:it->second) {
                const auto& other=*unique[i].constraint;
                if(c.parallelScore==other.parallelScore &&
                   std::abs(c.arcABaseVoxels-other.arcABaseVoxels)<=(c.parallelScore>.5?1e-4:tolerance) &&
                   std::abs(c.arcBBaseVoxels-other.arcBBaseVoxels)<=tolerance) duplicate=true;
            }
        }
        if(duplicate) continue;
        cells[{c.pieceA,c.pieceB,a,b}].push_back(unique.size());
        unique.push_back(std::move(s));
    }
    scored=std::move(unique);
}

void appendContinuity(FiberTraceConstraintReport& report,
    const std::vector<std::vector<std::size_t>>& piecesByTrace,
    const std::vector<std::optional<PolylineArcGeometry>>& geometries)
{
    for (const auto& tracePieces : piecesByTrace) {
        for (std::size_t index = 1; index < tracePieces.size(); ++index) {
            const std::size_t pieceA = tracePieces[index - 1];
            const std::size_t pieceB = tracePieces[index];
            const auto& a = report.pieces[pieceA];
            const auto& b = report.pieces[pieceB];
            const double arc = 0.5 * (a.endArcBaseVoxels + b.beginArcBaseVoxels);
            const auto point = samplePolylineArc(
                *geometries[a.traceIndex], arc).point;
            report.constraints.push_back({
                pieceA,
                pieceB,
                arc,
                arc,
                point,
                point,
                0.0,
                1.0,
                0.0,
                0.0,
                true,
            });
            ++report.hardConstraints;
        }
    }
}

void measureCandidates(
    std::vector<ScoredCandidate>& scored, const FiberTraceConstraintConfig& config,
    const FiberTraceWindingDistance& windingDistance,
    const FiberTraceWindingDistanceBatch& windingDistanceBatch,
    const LasagnaNormalAlignmentField* alignedNormals, int workers)
{
    std::vector<std::size_t> acceptedIndices;
    std::vector<std::size_t> connectorOffsets;
    std::vector<std::pair<cv::Vec3d, cv::Vec3d>> connectors;
    acceptedIndices.reserve(scored.size());
    connectorOffsets.reserve(scored.size() + 1);
    for (std::size_t index = 0; index < scored.size(); ++index) {
        if (!scored[index].constraint.has_value())
            continue;
        if (scored[index].parallelConnectors.empty()) {
            throw std::logic_error(
                "Scored fiber trace constraint has no parallel connectors");
        }
        acceptedIndices.push_back(index);
        connectorOffsets.push_back(connectors.size());
        connectors.insert(
            connectors.end(),
            scored[index].parallelConnectors.begin(),
            scored[index].parallelConnectors.end());
    }
    connectorOffsets.push_back(connectors.size());
    std::vector<double> windings(connectors.size());
    if (!connectors.empty() && windingDistanceBatch) {
        windings = windingDistanceBatch(
            connectors,
            config.windingIntegrationStepBaseVoxels,
            workers);
        if (windings.size() != connectors.size()) {
            throw std::runtime_error(
                "Fiber trace constraint winding batch returned the wrong result count");
        }
    } else if (!connectors.empty()) {
        std::vector<std::exception_ptr> windingFailures(connectors.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(guided, 16) num_threads(workers)
#endif
        for (std::ptrdiff_t index = 0;
             index < static_cast<std::ptrdiff_t>(connectors.size());
             ++index) {
            try {
                const auto& connector = connectors[static_cast<std::size_t>(index)];
                windings[static_cast<std::size_t>(index)] = windingDistance(
                    connector.first,
                    connector.second,
                    config.windingIntegrationStepBaseVoxels);
            } catch (...) {
                windingFailures[static_cast<std::size_t>(index)] =
                    std::current_exception();
            }
        }
        for (const auto& failure : windingFailures) {
            if (failure)
                std::rethrow_exception(failure);
        }
    }
    for (std::size_t index = 0; index < acceptedIndices.size(); ++index) {
        auto& result = scored[acceptedIndices[index]];
        const std::size_t begin = connectorOffsets[index];
        const std::size_t end = connectorOffsets[index + 1];
        const double closestWinding = windings[begin];
        if (!std::isfinite(closestWinding)) {
            result.constraint.reset();
            result.rejection = RejectReason::Winding;
            continue;
        }

        std::vector<double> parallelWindings;
        parallelWindings.reserve(end - begin);
        for (std::size_t sample = begin; sample < end; ++sample) {
            if (std::isfinite(windings[sample]))
                parallelWindings.push_back(windings[sample]);
        }
        if (parallelWindings.empty()) {
            result.constraint.reset();
            result.rejection = RejectReason::Winding;
            continue;
        }

        result.constraint->windingDistance = closestWinding;
        result.constraint->parallelWindingDistance = median(parallelWindings);
        if (alignedNormals != nullptr) {
            const auto closestSigned = signedWindingSample(
                connectors[begin], closestWinding, *alignedNormals);
            if (closestSigned) {
                result.constraint->signedWindingDelta = closestSigned->value;
                result.constraint->perpendicularNormalAlignment =
                    closestSigned->absoluteAlignment;
                result.constraint->windingNormalComponent =
                    closestSigned->component;
                std::vector<double> signedParallelWindings;
                std::vector<double> parallelAlignments;
                signedParallelWindings.reserve(end - begin);
                parallelAlignments.reserve(end - begin);
                for (std::size_t sample = begin; sample < end; ++sample) {
                    const auto signedSample = signedWindingSample(
                        connectors[sample], windings[sample], *alignedNormals);
                    if (signedSample &&
                        signedSample->component == closestSigned->component) {
                        signedParallelWindings.push_back(signedSample->value);
                        parallelAlignments.push_back(
                            signedSample->absoluteAlignment);
                    }
                }
                if (!signedParallelWindings.empty()) {
                    result.constraint->signedParallelWindingDelta =
                        median(signedParallelWindings);
                    result.constraint->parallelWindingDistance = std::abs(
                        *result.constraint->signedParallelWindingDelta);
                    result.constraint->parallelNormalAlignment =
                        median(parallelAlignments);
                }
            }
        }

        if (config.localEvents) {
            const auto& c = *result.constraint;
            const double step = c.parallelScore > c.perpendicularScore ? 1.0 : 0.5;
            bool consistent = true;
            for (std::size_t sample = begin; sample < end; ++sample) {
                if (!std::isfinite(windings[sample]) ||
                    std::abs(windings[sample] - closestWinding) > config.localDistanceTolerance ||
                    std::llround(windings[sample] / step) != std::llround(closestWinding / step)) {
                    consistent = false;
                    break;
                }
                if (alignedNormals != nullptr && closestWinding > config.localDistanceTolerance) {
                    const auto value = signedWindingSample(connectors[sample], windings[sample], *alignedNormals);
                    if (!value || !c.signedWindingDelta ||
                        value->component != c.windingNormalComponent ||
                        std::signbit(value->value) != std::signbit(*c.signedWindingDelta))
                        consistent = false;
                }
            }
            if (!consistent) {
                result.constraint.reset();
                result.rejection = RejectReason::Consistency;
                continue;
            }
            // Off-base samples certify the measurement; they do not move or average it.
            result.constraint->parallelWindingDistance = closestWinding;
            result.constraint->signedParallelWindingDelta = c.signedWindingDelta;
            result.constraint->parallelNormalAlignment = c.perpendicularNormalAlignment;
        }
        if (config.enforceMaximumWindingDistance &&
            (!(closestWinding < config.maximumWindingDistance) ||
             !(result.constraint->parallelWindingDistance <
               config.maximumWindingDistance))) {
            result.constraint.reset();
            result.rejection = RejectReason::WindingCutoff;
        }
    }
}

// Evidence belongs to certified source-trace encounters, not their later
// segmentation. Partition repeated parallel evidence before remapping IDs.
void normalizeLocalParallelEvidence(FiberTraceConstraintReport& report,
    const std::vector<std::optional<PolylineArcGeometry>>& geometries,
    const FiberTraceConstraintConfig& config)
{
    std::map<PiecePair,std::vector<std::size_t>> repeats;
    for(std::size_t i=0;i<report.constraints.size();++i)
        if(report.constraints[i].parallelScore>.5)
            repeats[{report.constraints[i].pieceA,report.constraints[i].pieceB}].push_back(i);
    for(auto& [pair,ids]:repeats) {
        std::stable_sort(ids.begin(),ids.end(),[&](auto a,auto b) {
            return report.constraints[a].arcABaseVoxels<report.constraints[b].arcABaseVoxels;
        });
        for(std::size_t first=0;first<ids.size();) {
            std::size_t endGroup=first+1;
            const double arc=report.constraints[ids[first]].arcABaseVoxels;
            while(endGroup<ids.size() && report.constraints[ids[endGroup]].arcABaseVoxels==arc)
                ++endGroup;
            for(std::size_t k=first;k<endGroup;++k) {
                auto& c=report.constraints[ids[k]];
                double begin=c.supportABegin,end=c.supportAEnd;
                if(!config.preserveInputLinesAsPieces) {
                    const double margin=config.localParallelSpanBaseVoxels+config.localTangentSpanBaseVoxels;
                    begin=std::max(begin,margin);
                    end=std::min(end,geometries[report.pieces[c.pieceA].traceIndex]->length()-margin);
                }
                if(first>0)begin=std::max(begin,.5*(report.constraints[ids[first-1]].arcABaseVoxels+arc));
                if(endGroup<ids.size())end=std::min(end,.5*(report.constraints[ids[endGroup]].arcABaseVoxels+arc));
                if(!(end>begin))
                    throw std::logic_error("Certified parallel event has no represented source support");
                c.evidenceWeight=(end-begin)/(128.*static_cast<double>(endGroup-first));
            }
            first=endGroup;
        }
    }
}

// All event identities still refer to whole source traces here. Derive cuts
// before optional parallel-factor removal so the ablation has identical pieces.
void splitLocalEvents(FiberTraceConstraintReport& report,
    const std::vector<std::optional<PolylineArcGeometry>>& geometries,
    const FiberTraceConstraintConfig& config)
{
    normalizeLocalParallelEvidence(report,geometries,config);
    report.localCertifiedEvents=report.constraints.size();
    const auto originalPieces=report.pieces;
    std::vector<std::vector<double>> orthogonal(geometries.size()), cuts(geometries.size());
    std::map<PiecePair,std::vector<std::size_t>> pairs;
    for(std::size_t i=0;i<report.constraints.size();++i) {
        const auto& c=report.constraints[i];
        pairs[{c.pieceA,c.pieceB}].push_back(i);
        if(c.perpendicularScore>c.parallelScore) {
            orthogonal[originalPieces[c.pieceA].traceIndex].push_back(c.arcABaseVoxels);
            orthogonal[originalPieces[c.pieceB].traceIndex].push_back(c.arcBBaseVoxels);
        }
    }
    const auto sortUnique=[](auto& v) {
        std::sort(v.begin(),v.end());
        v.erase(std::unique(v.begin(),v.end(),[](double a,double b){return std::abs(a-b)<1e-4;}),v.end());
    };
    for(std::size_t t=0;t<orthogonal.size();++t) {
        auto& arcs=orthogonal[t]; sortUnique(arcs);
        for(std::size_t i=1;i<arcs.size();++i) {
            cuts[t].push_back(.5*(arcs[i-1]+arcs[i]));
            ++report.localMidpointCuts;
        }
    }
    for(auto& [pair,indices]:pairs) {
        std::sort(indices.begin(),indices.end(),[&](auto i,auto j) {
            const auto& a=report.constraints[i]; const auto& b=report.constraints[j];
            return std::tie(a.arcABaseVoxels,a.arcBBaseVoxels)<std::tie(b.arcABaseVoxels,b.arcBBaseVoxels);
        });
        for(std::size_t k=1;k<indices.size();++k) {
            const auto& a=report.constraints[indices[k-1]];
            const auto& b=report.constraints[indices[k]];
            // Two-sided certification leaves an ambiguous band around a real
            // transition. Bridge that bounded band, not only adjacent samples.
            const double bracket=2*(config.localSpacingBaseVoxels+
                config.localParallelSpanBaseVoxels+config.localTangentSpanBaseVoxels);
            if(std::abs(a.arcABaseVoxels-b.arcABaseVoxels)>bracket ||
               std::abs(a.arcBBaseVoxels-b.arcBBaseVoxels)>bracket) continue;
            const auto category=[](const auto& c) {
                return std::tuple{c.parallelScore>c.perpendicularScore,
                    std::llround(c.windingDistance/(c.parallelScore>.5?1.:.5)),
                    c.signedWindingDelta ? (*c.signedWindingDelta<0.?-1:1):0};
            };
            if(category(a)!=category(b) && (a.parallelScore>.5 || b.parallelScore>.5)) {
                cuts[originalPieces[pair.a].traceIndex].push_back(.5*(a.arcABaseVoxels+b.arcABaseVoxels));
                cuts[originalPieces[pair.b].traceIndex].push_back(.5*(a.arcBBaseVoxels+b.arcBBaseVoxels));
                report.localTransitionCuts+=2;
            }
        }
    }
    std::vector<std::vector<std::size_t>> piecesByTrace(geometries.size());
    if(!config.preserveInputLinesAsPieces) {
        report.pieces.clear(); report.resampledPoints=0;
        for(std::size_t t=0;t<geometries.size();++t) {
            if(!geometries[t]) continue;
            auto& c=cuts[t]; sortUnique(c);
            c.erase(std::remove_if(c.begin(),c.end(),[&](double x){return x<=1e-4 || x>=geometries[t]->length()-1e-4;}),c.end());
            c.push_back(geometries[t]->length());
            double begin=0.;
            for(double end:c) {
                piecesByTrace[t].push_back(report.pieces.size());
                report.pieces.push_back(makePiece(t,piecesByTrace[t].size()-1,*geometries[t],begin,end,config.resampleSpacingBaseVoxels));
                report.resampledPoints+=report.pieces.back().samplePointsBaseXYZ.size();
                begin=end;
            }
        }
    } else {
        for(std::size_t i=0;i<report.pieces.size();++i)
            piecesByTrace[report.pieces[i].traceIndex].push_back(i);
        report.localMidpointCuts=report.localTransitionCuts=0;
    }
    std::size_t missingOwners=0;
    report.constraints=remapCertifiedFiberTraceEvents(report.constraints,originalPieces,report.pieces,missingOwners);
    if(missingOwners)
        throw std::logic_error("Full subdivision lost certified event owners");
    std::erase_if(report.constraints,[&](const auto& c) {
        if(config.localIncludeParallel || c.parallelScore<=.5)return false;
        ++report.localParallelOmitted; return true;
    });
    report.localRetainedCertifiedEvents=report.constraints.size();
    appendContinuity(report,piecesByTrace,geometries);
}

}  // namespace

std::vector<FiberTraceConstraint> remapCertifiedFiberTraceEvents(
    std::span<const FiberTraceConstraint> events,
    std::span<const FiberTraceConstraintPiece> sourcePieces,
    std::span<const FiberTraceConstraintPiece> targetPieces,
    std::size_t& missingOwners)
{
    std::map<std::size_t,std::vector<std::size_t>> byTrace;
    for(std::size_t i=0;i<targetPieces.size();++i)byTrace[targetPieces[i].traceIndex].push_back(i);
    for(auto& [trace,ids]:byTrace) {
        std::stable_sort(ids.begin(),ids.end(),[&](auto a,auto b) {
            return targetPieces[a].beginArcBaseVoxels<targetPieces[b].beginArcBaseVoxels;
        });
        for(std::size_t i=1;i<ids.size();++i)
            if(targetPieces[ids[i]].endArcBaseVoxels<targetPieces[ids[i-1]].endArcBaseVoxels)
                throw std::invalid_argument("Nested target intervals are not ordered fiber pieces");
    }
    const auto owner=[&](std::size_t source,double arc)->std::optional<std::size_t> {
        if(source>=sourcePieces.size())throw std::out_of_range("Certified event source piece");
        const auto found=byTrace.find(sourcePieces[source].traceIndex);
        if(found==byTrace.end())return std::nullopt;
        const auto& ids=found->second;
        auto it=std::upper_bound(ids.begin(),ids.end(),arc,[&](double x,std::size_t id) {
            return x<targetPieces[id].beginArcBaseVoxels;
        });
        if(it==ids.begin())return std::nullopt;
        const auto id=*std::prev(it);
        if(arc>targetPieces[id].endArcBaseVoxels)return std::nullopt;
        return id;
    };
    missingOwners=0;
    std::vector<FiberTraceConstraint> result;
    result.reserve(events.size());
    for(auto c:events) {
        if(c.hardContinuity)throw std::invalid_argument("Cannot remap a continuation as a certified event");
        const auto a=owner(c.pieceA,c.arcABaseVoxels),b=owner(c.pieceB,c.arcBBaseVoxels);
        if(!a||!b){++missingOwners;continue;}
        c.pieceA=*a; c.pieceB=*b;
        result.push_back(std::move(c));
    }
    return result;
}

FiberTraceConstraintReport extractFiberTraceConstraints(
    const std::vector<FiberletCropTraceLine>& lines,
    const FiberTraceConstraintConfig& config,
    const FiberTraceWindingDistance& windingDistance,
    const FiberTraceWindingDistanceBatch& windingDistanceBatch,
    const FiberTraceConstraintTracePairFilter& tracePairFilter,
    const LasagnaNormalAlignmentField* alignedNormals)
{
    validateConfig(config);
    FiberTraceConstraintReport report;
    report.inputTraces = lines.size();
    std::vector<std::optional<PolylineArcGeometry>> geometries(lines.size());
    std::vector<std::vector<std::size_t>> piecesByTrace(lines.size());

    if (!config.extraPieceCutsBaseVoxels.empty() &&
        (config.extraPieceCutsBaseVoxels.size() != lines.size() || config.preserveInputLinesAsPieces))
        throw std::invalid_argument("Piece cut hints require one entry per trace and ordinary splitting");
    const auto prepareStarted = std::chrono::steady_clock::now();
    for (std::size_t traceIndex = 0; traceIndex < lines.size(); ++traceIndex) {
        try {
            geometries[traceIndex] = makePolylineArcGeometry(
                lines[traceIndex].pointsBaseXYZ);
        } catch (const std::invalid_argument&) {
            ++report.skippedDegenerateTraces;
            continue;
        }
        auto intervals = (config.preserveInputLinesAsPieces || config.localEvents) ? std::vector<std::pair<double, double>>{{0.0, geometries[traceIndex]->length()}}
                                                                 : pieceIntervals(
            geometries[traceIndex]->length(), config);
        if (!config.extraPieceCutsBaseVoxels.empty()) {
            intervals.clear();
            double begin = 0.0;
            auto cuts = config.extraPieceCutsBaseVoxels[traceIndex];
            for (double cut : cuts) {
                if (!std::isfinite(cut) || cut <= begin || cut >= geometries[traceIndex]->length())
                    throw std::invalid_argument("Piece cuts must be ordered interior arcs");
                begin = cut;
            }
            cuts.push_back(geometries[traceIndex]->length());
            begin = 0.0;
            for (double end : cuts) {
                for (const auto& interval : pieceIntervals(end - begin, config))
                    intervals.emplace_back(begin + interval.first, begin + interval.second);
                begin = end;
            }
        }
        piecesByTrace[traceIndex].reserve(intervals.size());
        for (std::size_t pieceIndex = 0; pieceIndex < intervals.size(); ++pieceIndex) {
            const auto [begin, end] = intervals[pieceIndex];
            piecesByTrace[traceIndex].push_back(report.pieces.size());
            report.pieces.push_back(makePiece(
                traceIndex,
                pieceIndex,
                *geometries[traceIndex],
                begin,
                end,
                config.localEvents ? config.localSpacingBaseVoxels : config.resampleSpacingBaseVoxels));
            report.resampledPoints +=
                report.pieces.back().samplePointsBaseXYZ.size();
        }
    }

    appendContinuity(report, piecesByTrace, geometries);
    report.prepareSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prepareStarted).count();

    const auto searchStarted = std::chrono::steady_clock::now();
    std::vector<SampleRecord> samples;
    samples.reserve(report.resampledPoints);
    std::vector<RTreeValue> values;
    values.reserve(report.resampledPoints);
    for (std::size_t pieceIndex = 0; pieceIndex < report.pieces.size(); ++pieceIndex) {
        const auto& piece = report.pieces[pieceIndex];
        for (std::size_t sampleIndex = 0;
             sampleIndex < piece.samplePointsBaseXYZ.size();
             ++sampleIndex) {
            const std::size_t globalIndex = samples.size();
            samples.push_back({pieceIndex, sampleIndex});
            values.emplace_back(
                point3(piece.samplePointsBaseXYZ[sampleIndex]), globalIndex);
        }
    }
    const PointTree tree(values.begin(), values.end());
    std::map<PiecePair, PairCandidate> bestByPair;
    // Arc-pair cells preserve separate encounters of the same two source traces.
    using LocalKey = std::tuple<std::size_t,std::size_t,std::size_t,std::size_t>;
    std::map<LocalKey, PairCandidate> localCandidates;
    std::vector<RTreeValue> hits;
    const double maximumDistanceSquared =
        config.maximumDistanceBaseVoxels * config.maximumDistanceBaseVoxels;
    for (std::size_t globalA = 0; globalA < samples.size(); ++globalA) {
        const auto& sampleA = samples[globalA];
        const auto& pieceA = report.pieces[sampleA.piece];
        const cv::Vec3d pointA = pieceA.samplePointsBaseXYZ[sampleA.sample];
        hits.clear();
        tree.query(
            bgi::intersects(queryBox(
                pointA, config.maximumDistanceBaseVoxels)),
            std::back_inserter(hits));
        for (const auto& hit : hits) {
            const std::size_t globalB = hit.second;
            if (globalB <= globalA)
                continue;
            const auto& sampleB = samples[globalB];
            if (sampleA.piece == sampleB.piece)
                continue;
            const auto& pieceB = report.pieces[sampleB.piece];
            if (pieceA.traceIndex == pieceB.traceIndex)
                continue;
            if (tracePairFilter && !tracePairFilter(pieceA.traceIndex, pieceB.traceIndex)) {
                continue;
            }
            const cv::Vec3d pointB = pieceB.samplePointsBaseXYZ[sampleB.sample];
            const cv::Vec3d delta = pointA - pointB;
            const double distanceSquared = delta.dot(delta);
            if (!std::isfinite(distanceSquared) ||
                distanceSquared > maximumDistanceSquared + kEpsilon) {
                continue;
            }
            ++report.spatialHits;
            PairCandidate candidate;
            if (sampleA.piece < sampleB.piece) {
                candidate.pieces = {sampleA.piece, sampleB.piece};
                candidate.sampleA = sampleA.sample;
                candidate.sampleB = sampleB.sample;
                candidate.globalSampleA = globalA;
                candidate.globalSampleB = globalB;
            } else {
                candidate.pieces = {sampleB.piece, sampleA.piece};
                candidate.sampleA = sampleB.sample;
                candidate.sampleB = sampleA.sample;
                candidate.globalSampleA = globalB;
                candidate.globalSampleB = globalA;
            }
            candidate.distance = std::sqrt(std::max(0.0, distanceSquared));
            if (config.localEvents) {
                const LocalKey key{candidate.pieces.a,candidate.pieces.b,candidate.sampleA,candidate.sampleB};
                localCandidates.emplace(key,candidate);
                continue;
            }
            const auto found = bestByPair.find(candidate.pieces);
            if (found == bestByPair.end())
                bestByPair.emplace(candidate.pieces, candidate);
            else if (betterCandidate(candidate, found->second))
                found->second = candidate;
        }
    }
    std::vector<PairCandidate> candidates;
    for (const auto& [key,candidate] : localCandidates) candidates.push_back(candidate);
    candidates.reserve(bestByPair.size());
    for (const auto& [key, candidate] : bestByPair) {
        (void)key;
        candidates.push_back(candidate);
    }
    report.measuredCandidates = candidates.size();
    report.searchSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - searchStarted).count();

    if (!candidates.empty() && !windingDistance && !windingDistanceBatch)
        throw std::invalid_argument("Fiber trace constraint winding sampler is missing");
    const auto scoreStarted = std::chrono::steady_clock::now();
    const std::size_t requestedThreads = config.parallelThreads == 0
        ? std::max<std::size_t>(1, std::thread::hardware_concurrency())
        : config.parallelThreads;
    const int workers = static_cast<int>(std::min<std::size_t>(
        requestedThreads,
        std::max<std::size_t>(1, candidates.size())));
    std::vector<ScoredCandidate> scored(candidates.size());
    std::vector<std::exception_ptr> failures(candidates.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(guided, 16) num_threads(workers)
#endif
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(candidates.size());
         ++index) {
        try {
            scored[static_cast<std::size_t>(index)] = scoreCandidate(
                candidates[static_cast<std::size_t>(index)],
                report.pieces,
                geometries,
                config);
        } catch (...) {
            failures[static_cast<std::size_t>(index)] = std::current_exception();
        }
    }
    for (const auto& failure : failures) {
        if (failure)
            std::rethrow_exception(failure);
    }
    report.orientationScoreSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scoreStarted).count();

    const auto windingStarted = std::chrono::steady_clock::now();
    if(config.localEvents) deduplicateLocalEvents(scored);
    measureCandidates(scored, config, windingDistance, windingDistanceBatch, alignedNormals, workers);
    if(config.localEvents) {
        for(std::size_t round=0;round<config.localRefinementRounds;++round) {
            std::map<PiecePair,std::vector<std::size_t>> groups;
            for(std::size_t i=0;i<scored.size();++i) groups[scored[i].candidate.pieces].push_back(i);
            std::vector<PairCandidate> extra;
            for(auto& [pair,ids]:groups) {
                std::sort(ids.begin(),ids.end(),[&](auto i,auto j) {
                    return std::tie(scored[i].candidate.arcA,scored[i].candidate.arcB) <
                        std::tie(scored[j].candidate.arcA,scored[j].candidate.arcB);
                });
                const auto state=[](const ScoredCandidate& s) {
                    if(!s.constraint) return std::tuple{0,0LL,0};
                    const auto& c=*s.constraint;
                    return std::tuple{c.parallelScore>.5?1:2,
                        std::llround(c.windingDistance/(c.parallelScore>.5?1.:.5)),
                        c.signedWindingDelta?(*c.signedWindingDelta<0.?-1:1):0};
                };
                for(std::size_t k=1;k<ids.size();++k) {
                    const auto& a=scored[ids[k-1]]; const auto& b=scored[ids[k]];
                    double gap=std::abs(a.candidate.arcA-b.candidate.arcA);
                    if(state(a)==state(b) || gap<config.localSpacingBaseVoxels/std::pow(2.,round+1) ||
                       gap>2*config.localSpacingBaseVoxels ||
                       std::abs(a.candidate.arcB-b.candidate.arcB)>2*config.localSpacingBaseVoxels) continue;
                    auto c=a.candidate;
                    c.arcA=.5*(a.candidate.arcA+b.candidate.arcA);
                    c.arcB=.5*(a.candidate.arcB+b.candidate.arcB);
                    extra.push_back(c);
                }
            }
            if(extra.empty()) break;
            std::vector<ScoredCandidate> refined(extra.size());
            std::vector<std::exception_ptr> errors(extra.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(guided, 16) num_threads(workers)
#endif
            for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(extra.size());++i) {
                try { refined[i]=scoreLocalCandidate(extra[i],report.pieces,geometries,config); }
                catch(...) { errors[i]=std::current_exception(); }
            }
            for(const auto& e:errors)if(e)std::rethrow_exception(e);
            measureCandidates(refined,config,windingDistance,windingDistanceBatch,alignedNormals,workers);
            report.localRefinedCandidates+=refined.size();
            scored.insert(scored.end(),std::make_move_iterator(refined.begin()),std::make_move_iterator(refined.end()));
            deduplicateLocalEvents(scored);
        }
    }
    report.windingScoreSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - windingStarted).count();

    for (auto& result : scored) {
        if (result.constraint.has_value()) {
            if (alignedNormals != nullptr) {
                if (result.constraint->signedWindingDelta &&
                    result.constraint->signedParallelWindingDelta) {
                    ++report.signedWindingConstraints;
                } else {
                    ++report.skippedSignedWindingConstraints;
                }
            }
            report.constraints.push_back(std::move(*result.constraint));
        } else if (result.rejection == RejectReason::Tangent)
            ++report.rejectedTangents;
        else if (result.rejection == RejectReason::Winding)
            ++report.rejectedWinding;
        else if (result.rejection == RejectReason::WindingCutoff)
            ++report.rejectedWindingCutoff;
        else if (result.rejection == RejectReason::Consistency)
            ++report.localRejectedConsistency;
    }
    if(config.localEvents) {
        splitLocalEvents(report,geometries,config);
        report.signedWindingConstraints=report.skippedSignedWindingConstraints=0;
        if(alignedNormals != nullptr) {
            for(const auto& c:report.constraints) {
                if(c.hardContinuity)continue; // Counted by the shared finalization below.
                if(c.signedWindingDelta && c.signedParallelWindingDelta)
                    ++report.signedWindingConstraints;
                else ++report.skippedSignedWindingConstraints;
            }
        }
    }
    std::sort(report.constraints.begin(), report.constraints.end(),
        [&](const auto& left, const auto& right) {
            if(!config.localEvents)
                return std::tie(left.pieceA,left.pieceB,left.hardContinuity) <
                    std::tie(right.pieceA,right.pieceB,right.hardContinuity);
            return std::tie(left.pieceA, left.pieceB, left.hardContinuity,left.arcABaseVoxels,left.arcBBaseVoxels) <
                std::tie(right.pieceA, right.pieceB, right.hardContinuity,right.arcABaseVoxels,right.arcBBaseVoxels);
        });
    report.scoreSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scoreStarted).count();
    if (alignedNormals != nullptr) {
        for (auto& constraint : report.constraints) {
            if (!constraint.hardContinuity)
                continue;
            constraint.signedWindingDelta = 0.0;
            constraint.signedParallelWindingDelta = 0.0;
            ++report.signedWindingConstraints;
        }
    }
    return report;
}

double dominantFiberTraceConstraintWindingDistance(
    const FiberTraceConstraint& constraint) noexcept
{
    return constraint.parallelScore > constraint.perpendicularScore
        ? constraint.parallelWindingDistance
        : constraint.windingDistance;
}

std::vector<FiberTraceOrderedCrossSourceConstraint>
orderMeasuredCrossSourceFiberTraceConstraints(
    const FiberTraceConstraintReport& report,
    const std::vector<std::size_t>& sourceIdsByTrace)
{
    if (sourceIdsByTrace.size() != report.inputTraces) {
        throw std::invalid_argument(
            "Fiber trace constraint source IDs do not match input traces");
    }
    std::vector<FiberTraceOrderedCrossSourceConstraint> ordered;
    ordered.reserve(report.constraints.size());
    for (std::size_t index = 0; index < report.constraints.size(); ++index) {
        const auto& constraint = report.constraints[index];
        if (constraint.hardContinuity)
            continue;
        if (constraint.pieceA >= report.pieces.size() ||
            constraint.pieceB >= report.pieces.size()) {
            throw std::invalid_argument(
                "Fiber trace constraint references an invalid piece");
        }
        const std::size_t traceA = report.pieces[constraint.pieceA].traceIndex;
        const std::size_t traceB = report.pieces[constraint.pieceB].traceIndex;
        if (traceA >= sourceIdsByTrace.size() ||
            traceB >= sourceIdsByTrace.size()) {
            throw std::invalid_argument(
                "Fiber trace constraint piece references an invalid trace");
        }
        const std::size_t sourceA = sourceIdsByTrace[traceA];
        const std::size_t sourceB = sourceIdsByTrace[traceB];
        if (sourceA != sourceB) {
            ordered.push_back({
                index,
                std::min(sourceA, sourceB),
                std::max(sourceA, sourceB),
                constraint.perpendicularScore >= constraint.parallelScore,
            });
        }
    }
    return ordered;
}

void orientFiberTraceConstraintWindings(
    FiberTraceConstraintReport& report,
    const LasagnaNormalAlignmentField& alignedNormals)
{
    report.signedWindingConstraints = 0;
    report.skippedSignedWindingConstraints = 0;
    for (auto& constraint : report.constraints) {
        constraint.signedWindingDelta.reset();
        constraint.signedParallelWindingDelta.reset();
        constraint.perpendicularNormalAlignment.reset();
        constraint.parallelNormalAlignment.reset();
        constraint.windingNormalComponent.reset();
        if (constraint.hardContinuity) {
            constraint.signedWindingDelta = 0.0;
            constraint.signedParallelWindingDelta = 0.0;
            constraint.perpendicularNormalAlignment = 1.0;
            constraint.parallelNormalAlignment = 1.0;
            ++report.signedWindingConstraints;
            continue;
        }
        if (!(constraint.perpendicularScore > 0.0) &&
            !(constraint.parallelScore > 0.0))
            continue;
        const cv::Vec3d connector =
            constraint.pointBBaseXYZ - constraint.pointABaseXYZ;
        const double connectorLength = cv::norm(connector);
        if (!(connectorLength > kEpsilon) || !std::isfinite(connectorLength)) {
            ++report.skippedSignedWindingConstraints;
            continue;
        }
        const cv::Vec3d midpoint =
            0.5 * (constraint.pointABaseXYZ + constraint.pointBBaseXYZ);
        const auto atA = alignedNormals.nearest(constraint.pointABaseXYZ);
        const auto atMidpoint = alignedNormals.nearest(midpoint);
        const auto atB = alignedNormals.nearest(constraint.pointBBaseXYZ);
        if (!atA || !atMidpoint || !atB ||
            atA->component != atMidpoint->component ||
            atB->component != atMidpoint->component) {
            ++report.skippedSignedWindingConstraints;
            continue;
        }
        const double signedAlignment =
            (connector / connectorLength).dot(cv::Vec3d{
                atMidpoint->normal[0],
                atMidpoint->normal[1],
                atMidpoint->normal[2],
            });
        if (!std::isfinite(signedAlignment) ||
            std::abs(signedAlignment) <= kEpsilon) {
            ++report.skippedSignedWindingConstraints;
            continue;
        }
        constraint.signedWindingDelta = std::copysign(
            constraint.windingDistance, signedAlignment);
        constraint.perpendicularNormalAlignment =
            std::clamp(std::abs(signedAlignment), 0.0, 1.0);
        constraint.signedParallelWindingDelta = std::copysign(
            constraint.parallelWindingDistance, signedAlignment);
        constraint.windingNormalComponent = atMidpoint->component;
        ++report.signedWindingConstraints;
    }
}

std::vector<FiberletCropTraceLine> makeFiberTraceConstraintPieceLines(
    const std::vector<FiberletCropTraceLine>& sourceLines,
    const FiberTraceConstraintReport& constraints)
{
    if (constraints.inputTraces != sourceLines.size()) {
        throw std::invalid_argument(
            "Constraint piece source count does not match input traces");
    }
    std::vector<PolylineArcGeometry> geometries;
    geometries.reserve(sourceLines.size());
    for (const auto& line : sourceLines)
        geometries.push_back(makePolylineArcGeometry(line.pointsBaseXYZ));

    std::vector<FiberletCropTraceLine> result;
    result.reserve(constraints.pieces.size());
    for (const auto& piece : constraints.pieces) {
        if (piece.traceIndex >= geometries.size() ||
            !std::isfinite(piece.beginArcBaseVoxels) ||
            !std::isfinite(piece.endArcBaseVoxels) ||
            !(piece.endArcBaseVoxels > piece.beginArcBaseVoxels) ||
            piece.beginArcBaseVoxels < -kEpsilon ||
            piece.endArcBaseVoxels >
                geometries[piece.traceIndex].length() + kEpsilon) {
            throw std::invalid_argument(
                "Constraint piece has invalid source geometry interval");
        }
        FiberletCropTraceLine line;
        line.pointsBaseXYZ = slicePolylineArc(
            geometries[piece.traceIndex],
            piece.beginArcBaseVoxels,
            piece.endArcBaseVoxels);
        result.push_back(std::move(line));
    }
    return result;
}

FiberTraceConstraintSubsetResult subsetFiberTraceConstraintReport(const FiberTraceConstraintReport& constraints, std::span<const std::size_t> retainedPieceIndices)
{
    const std::size_t missing = std::numeric_limits<std::size_t>::max();
    FiberTraceConstraintSubsetResult result;
    result.retainedPieceIndices.assign(retainedPieceIndices.begin(), retainedPieceIndices.end());

    std::vector<std::size_t> oldPieceToNew(constraints.pieces.size(), missing);
    std::vector<std::vector<std::size_t>> retainedPiecesByTrace(constraints.inputTraces);
    std::size_t previous = 0;
    bool first = true;
    for (std::size_t newPiece = 0; newPiece < retainedPieceIndices.size(); ++newPiece) {
        const std::size_t oldPiece = retainedPieceIndices[newPiece];
        if (oldPiece >= constraints.pieces.size() || (!first && oldPiece <= previous)) {
            throw std::invalid_argument("Constraint subset piece indices must be sorted and unique");
        }
        first = false;
        previous = oldPiece;
        const std::size_t oldTrace = constraints.pieces[oldPiece].traceIndex;
        if (oldTrace >= constraints.inputTraces) {
            throw std::invalid_argument("Constraint subset piece references an invalid trace");
        }
        oldPieceToNew[oldPiece] = newPiece;
        retainedPiecesByTrace[oldTrace].push_back(oldPiece);
    }

    std::vector<std::size_t> newTraceByOldPiece(constraints.pieces.size(), missing);
    std::vector<std::size_t> newPieceIndexByOldPiece(constraints.pieces.size(), missing);
    for (std::size_t oldTrace = 0; oldTrace < constraints.inputTraces; ++oldTrace) {
        auto& pieces = retainedPiecesByTrace[oldTrace];
        std::sort(pieces.begin(), pieces.end(), [&](std::size_t a, std::size_t b) {
            return constraints.pieces[a].pieceIndex < constraints.pieces[b].pieceIndex;
        });
        std::size_t previousPieceIndex = missing;
        std::size_t localPieceIndex = 0;
        std::size_t newTrace = missing;
        for (const std::size_t oldPiece : pieces) {
            const std::size_t oldPieceIndex = constraints.pieces[oldPiece].pieceIndex;
            if (previousPieceIndex == missing || oldPieceIndex != previousPieceIndex + 1) {
                newTrace = result.retainedTraceIndices.size();
                result.retainedTraceIndices.push_back(oldTrace);
                localPieceIndex = 0;
            }
            newTraceByOldPiece[oldPiece] = newTrace;
            newPieceIndexByOldPiece[oldPiece] = localPieceIndex++;
            previousPieceIndex = oldPieceIndex;
        }
    }

    result.report = constraints;
    result.report.inputTraces = result.retainedTraceIndices.size();
    result.report.pieces.clear();
    result.report.constraints.clear();
    result.report.hardConstraints = 0;
    result.report.signedWindingConstraints = 0;
    result.report.skippedSignedWindingConstraints = 0;
    result.report.pieces.reserve(retainedPieceIndices.size());
    for (const std::size_t oldPiece : retainedPieceIndices) {
        auto piece = constraints.pieces[oldPiece];
        piece.traceIndex = newTraceByOldPiece.at(oldPiece);
        piece.pieceIndex = newPieceIndexByOldPiece.at(oldPiece);
        result.report.pieces.push_back(std::move(piece));
    }

    result.report.constraints.reserve(constraints.constraints.size());
    for (std::size_t originalIndex = 0;
         originalIndex < constraints.constraints.size(); ++originalIndex) {
        const auto& original = constraints.constraints[originalIndex];
        if (original.pieceA >= constraints.pieces.size() || original.pieceB >= constraints.pieces.size() || original.pieceA == original.pieceB) {
            throw std::invalid_argument("Constraint subset link references an invalid piece pair");
        }
        const std::size_t newA = oldPieceToNew[original.pieceA];
        const std::size_t newB = oldPieceToNew[original.pieceB];
        if (newA == missing || newB == missing)
            continue;
        auto constraint = original;
        constraint.pieceA = newA;
        constraint.pieceB = newB;
        result.report.hardConstraints += constraint.hardContinuity ? 1 : 0;
        if (constraint.signedWindingDelta) {
            ++result.report.signedWindingConstraints;
        } else if (!constraint.hardContinuity && constraint.perpendicularScore > 0.0) {
            ++result.report.skippedSignedWindingConstraints;
        }
        result.report.constraints.push_back(std::move(constraint));
        result.retainedConstraintIndices.push_back(originalIndex);
    }
    return result;
}

FiberTraceConstraintPruningResult pruneFiberTraceConstraintsByStrength(
    const FiberTraceConstraintReport& constraints,
    double maximumDistanceBaseVoxels,
    std::size_t maximumConstraintsPerTrace)
{
    if (!std::isfinite(maximumDistanceBaseVoxels) ||
        !(maximumDistanceBaseVoxels > 0.0)) {
        throw std::invalid_argument(
            "Constraint pruning maximum distance must be finite and positive");
    }
    if (maximumConstraintsPerTrace == 0) {
        throw std::invalid_argument(
            "Constraint pruning limit must be positive");
    }

    struct RankedConstraint {
        std::size_t index = 0;
        std::size_t pieceA = 0;
        std::size_t pieceB = 0;
        std::size_t traceA = 0;
        std::size_t traceB = 0;
        double strength = 0.0;
        double distance = 0.0;
    };

    if (constraints.inputTraces == 0 && !constraints.pieces.empty()) {
        throw std::invalid_argument(
            "Constraint pruning report has pieces but no input traces");
    }
    std::vector<bool> validTrace(constraints.inputTraces, false);
    for (const auto& piece : constraints.pieces) {
        if (piece.traceIndex >= constraints.inputTraces) {
            throw std::invalid_argument(
                "Constraint pruning piece references an invalid trace");
        }
        validTrace[piece.traceIndex] = true;
    }

    std::vector<RankedConstraint> ranked;
    ranked.reserve(constraints.constraints.size());
    std::vector<std::size_t> hardIndices;
    hardIndices.reserve(constraints.hardConstraints);
    std::map<PiecePair, std::size_t> uniquePairs;
    for (std::size_t index = 0; index < constraints.constraints.size(); ++index) {
        const auto& constraint = constraints.constraints[index];
        if (constraint.pieceA >= constraints.pieces.size() ||
            constraint.pieceB >= constraints.pieces.size() ||
            constraint.pieceA == constraint.pieceB) {
            throw std::invalid_argument(
                "Constraint pruning link references an invalid piece pair");
        }
        const PiecePair pair{
            std::min(constraint.pieceA, constraint.pieceB),
            std::max(constraint.pieceA, constraint.pieceB)};
        if (!uniquePairs.emplace(pair, index).second) {
            throw std::invalid_argument(
                "Constraint pruning report contains a duplicate piece pair");
        }
        const std::size_t traceA =
            constraints.pieces[constraint.pieceA].traceIndex;
        const std::size_t traceB =
            constraints.pieces[constraint.pieceB].traceIndex;
        if (constraint.hardContinuity) {
            if (traceA != traceB) {
                throw std::invalid_argument(
                    "Constraint pruning hard link crosses source traces");
            }
            hardIndices.push_back(index);
            continue;
        }
        if (traceA == traceB) {
            throw std::invalid_argument(
                "Constraint pruning soft link stays within one source trace");
        }
        if (!std::isfinite(constraint.closestDistanceBaseVoxels) ||
            constraint.closestDistanceBaseVoxels < 0.0 ||
            constraint.closestDistanceBaseVoxels >
                maximumDistanceBaseVoxels + kEpsilon ||
            !std::isfinite(constraint.parallelScore) ||
            constraint.parallelScore < 0.0 ||
            constraint.parallelScore > 1.0) {
            throw std::invalid_argument(
                "Constraint pruning link has invalid strength evidence");
        }
        const double certainty =
            std::abs(2.0 * constraint.parallelScore - 1.0);
        const double proximity = std::max(
            0.0,
            1.0 - constraint.closestDistanceBaseVoxels /
                maximumDistanceBaseVoxels);
        ranked.push_back({
            index,
            pair.a,
            pair.b,
            traceA,
            traceB,
            certainty * proximity,
            constraint.closestDistanceBaseVoxels,
        });
    }

    auto graphStats = [&](const std::vector<std::size_t>& indices) {
        FiberTraceConstraintGraphStats stats;
        std::vector<std::size_t> activeTraces;
        activeTraces.reserve(validTrace.size());
        std::vector<std::size_t> activeIndex(validTrace.size(),
                                             std::numeric_limits<std::size_t>::max());
        for (std::size_t trace = 0; trace < validTrace.size(); ++trace) {
            if (!validTrace[trace])
                continue;
            activeIndex[trace] = activeTraces.size();
            activeTraces.push_back(trace);
        }
        stats.traces = activeTraces.size();
        std::vector<std::size_t> degree(stats.traces, 0);
        std::vector<std::size_t> parent(stats.traces);
        for (std::size_t index = 0; index < parent.size(); ++index)
            parent[index] = index;
        const auto findRoot = [&](std::size_t node) {
            while (parent[node] != node) {
                parent[node] = parent[parent[node]];
                node = parent[node];
            }
            return node;
        };
        for (const std::size_t index : indices) {
            const auto& constraint = constraints.constraints[index];
            if (constraint.hardContinuity)
                continue;
            const std::size_t traceA =
                constraints.pieces[constraint.pieceA].traceIndex;
            const std::size_t traceB =
                constraints.pieces[constraint.pieceB].traceIndex;
            const std::size_t a = activeIndex[traceA];
            const std::size_t b = activeIndex[traceB];
            ++degree[a];
            ++degree[b];
            ++stats.crossTraceConstraints;
            const std::size_t rootA = findRoot(a);
            const std::size_t rootB = findRoot(b);
            if (rootA != rootB)
                parent[rootB] = rootA;
        }
        if (degree.empty())
            return stats;
        std::sort(degree.begin(), degree.end());
        stats.minimumDegree = degree.front();
        stats.maximumDegree = degree.back();
        stats.meanDegree =
            2.0 * static_cast<double>(stats.crossTraceConstraints) /
            static_cast<double>(stats.traces);
        const std::size_t middle = degree.size() / 2;
        stats.medianDegree = degree.size() % 2 == 0
            ? 0.5 * static_cast<double>(degree[middle - 1] + degree[middle])
            : static_cast<double>(degree[middle]);
        stats.isolatedTraces = static_cast<std::size_t>(
            std::count(degree.begin(), degree.end(), 0));
        for (std::size_t index = 0; index < parent.size(); ++index) {
            if (findRoot(index) == index)
                ++stats.connectedComponents;
        }
        return stats;
    };

    FiberTraceConstraintPruningResult result;
    auto& report = result.report;
    report.maximumConstraintsPerTrace = maximumConstraintsPerTrace;
    report.inputTotalConstraints = constraints.constraints.size();
    report.hardConstraints = hardIndices.size();

    std::vector<std::vector<std::size_t>> incident(constraints.inputTraces);
    std::vector<const RankedConstraint*> byIndex(constraints.constraints.size(), nullptr);
    std::vector<std::size_t> positiveIndices;
    positiveIndices.reserve(ranked.size());
    for (const auto& entry : ranked) {
        byIndex[entry.index] = &entry;
        if (!(entry.strength > 0.0)) {
            ++report.rejectedZeroStrength;
            continue;
        }
        positiveIndices.push_back(entry.index);
        incident[entry.traceA].push_back(entry.index);
        incident[entry.traceB].push_back(entry.index);
    }
    report.before = graphStats(positiveIndices);
    const auto better = [&](std::size_t leftIndex, std::size_t rightIndex) {
        const auto& left = *byIndex[leftIndex];
        const auto& right = *byIndex[rightIndex];
        if (left.strength != right.strength)
            return left.strength > right.strength;
        if (left.distance != right.distance)
            return left.distance < right.distance;
        return std::tie(left.pieceA, left.pieceB) <
            std::tie(right.pieceA, right.pieceB);
    };
    std::vector<unsigned char> nominations(constraints.constraints.size(), 0);
    for (auto& traceIncident : incident) {
        std::sort(traceIncident.begin(), traceIncident.end(), better);
        const std::size_t count =
            std::min(maximumConstraintsPerTrace, traceIncident.size());
        for (std::size_t rank = 0; rank < count; ++rank)
            ++nominations[traceIncident[rank]];
    }

    std::vector<bool> retained(constraints.constraints.size(), false);
    for (const std::size_t index : hardIndices)
        retained[index] = true;
    for (const auto& entry : ranked) {
        if (entry.strength > 0.0 && nominations[entry.index] == 2)
            retained[entry.index] = true;
    }

    std::vector<std::size_t> mutualIndices;
    mutualIndices.reserve(positiveIndices.size());
    std::vector<std::size_t> degree(constraints.inputTraces, 0);
    std::vector<std::size_t> parent(constraints.inputTraces);
    for (std::size_t trace = 0; trace < parent.size(); ++trace)
        parent[trace] = trace;
    const auto findRoot = [&](std::size_t trace) {
        while (parent[trace] != trace) {
            parent[trace] = parent[parent[trace]];
            trace = parent[trace];
        }
        return trace;
    };
    const auto unite = [&](std::size_t traceA, std::size_t traceB) {
        const std::size_t rootA = findRoot(traceA);
        const std::size_t rootB = findRoot(traceB);
        if (rootA == rootB)
            return false;
        parent[std::max(rootA, rootB)] = std::min(rootA, rootB);
        return true;
    };
    for (const std::size_t index : positiveIndices) {
        if (!retained[index])
            continue;
        mutualIndices.push_back(index);
        const auto& entry = *byIndex[index];
        ++degree[entry.traceA];
        ++degree[entry.traceB];
        unite(entry.traceA, entry.traceB);
    }
    report.mutual = graphStats(mutualIndices);
    if (report.mutual.connectedComponents < report.before.connectedComponents) {
        throw std::runtime_error(
            "Constraint pruning mutual graph has invalid connectivity");
    }
    report.expectedRecoveryBridges =
        report.mutual.connectedComponents - report.before.connectedComponents;

    std::vector<std::size_t> recoveryCandidates;
    recoveryCandidates.reserve(positiveIndices.size());
    for (const std::size_t index : positiveIndices) {
        if (!retained[index])
            recoveryCandidates.push_back(index);
    }
    std::sort(recoveryCandidates.begin(), recoveryCandidates.end(), better);
    report.recoveryCandidates = recoveryCandidates.size();
    const auto acceptRecovery = [&](std::size_t index) {
        const auto& entry = *byIndex[index];
        if (!unite(entry.traceA, entry.traceB))
            return false;
        retained[index] = true;
        ++degree[entry.traceA];
        ++degree[entry.traceB];
        ++report.recoveryBridges;
        return true;
    };

    for (const std::size_t index : recoveryCandidates) {
        const auto& entry = *byIndex[index];
        if (findRoot(entry.traceA) == findRoot(entry.traceB) ||
            degree[entry.traceA] >= maximumConstraintsPerTrace ||
            degree[entry.traceB] >= maximumConstraintsPerTrace) {
            continue;
        }
        if (acceptRecovery(index))
            ++report.capRespectingRecoveryBridges;
    }

    while (report.recoveryBridges < report.expectedRecoveryBridges) {
        std::optional<std::size_t> selected;
        std::size_t selectedOverflow = std::numeric_limits<std::size_t>::max();
        for (const std::size_t index : recoveryCandidates) {
            if (retained[index])
                continue;
            const auto& entry = *byIndex[index];
            if (findRoot(entry.traceA) == findRoot(entry.traceB))
                continue;
            const std::size_t overflow =
                (degree[entry.traceA] + 1 > maximumConstraintsPerTrace
                     ? degree[entry.traceA] + 1 - maximumConstraintsPerTrace
                     : 0) +
                (degree[entry.traceB] + 1 > maximumConstraintsPerTrace
                     ? degree[entry.traceB] + 1 - maximumConstraintsPerTrace
                     : 0);
            if (!selected || overflow < selectedOverflow ||
                (overflow == selectedOverflow && better(index, *selected))) {
                selected = index;
                selectedOverflow = overflow;
            }
        }
        if (!selected || !acceptRecovery(*selected)) {
            throw std::runtime_error(
                "Constraint pruning could not recover source graph connectivity");
        }
        if (selectedOverflow > 0)
            ++report.fallbackOverflowBridges;
        else
            ++report.capRespectingRecoveryBridges;
    }

    for (std::size_t trace = 0; trace < degree.size(); ++trace) {
        if (validTrace[trace] && degree[trace] > maximumConstraintsPerTrace)
            ++report.tracesAboveTargetDegree;
    }
    for (const std::size_t index : positiveIndices) {
        if (!retained[index])
            ++report.rejectedNotMutual;
    }
    std::vector<std::size_t> retainedIndices;
    retainedIndices.reserve(constraints.constraints.size());
    result.constraints.reserve(constraints.constraints.size());
    for (std::size_t index = 0; index < constraints.constraints.size(); ++index) {
        if (!retained[index])
            continue;
        retainedIndices.push_back(index);
        result.constraints.push_back(constraints.constraints[index]);
    }
    std::sort(result.constraints.begin(), result.constraints.end(),
        [](const auto& left, const auto& right) {
            const PiecePair leftPair{
                std::min(left.pieceA, left.pieceB),
                std::max(left.pieceA, left.pieceB)};
            const PiecePair rightPair{
                std::min(right.pieceA, right.pieceB),
                std::max(right.pieceA, right.pieceB)};
            return std::tie(leftPair.a, leftPair.b, left.hardContinuity) <
                std::tie(rightPair.a, rightPair.b, right.hardContinuity);
        });
    report.retainedTotalConstraints = result.constraints.size();
    report.after = graphStats(retainedIndices);
    if (report.recoveryBridges != report.expectedRecoveryBridges ||
        report.after.connectedComponents != report.before.connectedComponents) {
        throw std::runtime_error(
            "Constraint pruning connectivity recovery invariant failed");
    }
    return result;
}

FiberTraceConstraintObjPaths fiberTraceConstraintObjPaths(
    const std::filesystem::path& outputBase)
{
    const auto directory = outputBase.parent_path();
    const std::string stem = outputBase.has_extension()
        ? outputBase.stem().string()
        : outputBase.filename().string();
    if (stem.empty())
        throw std::invalid_argument("constraint OBJ output basename is empty");
    return {
        directory / (stem + "_perpendicular_same_winding.obj"),
        directory / (stem + "_perpendicular_separate_winding.obj"),
        directory / (stem + "_parallel_same_winding.obj"),
        directory / (stem + "_parallel_separate_winding.obj"),
    };
}

FiberTraceConstraintObjReport writeFiberTraceConstraintObjs(
    const FiberTraceConstraintReport& report,
    const std::filesystem::path& outputBase)
{
    FiberTraceConstraintObjReport result;
    result.paths = fiberTraceConstraintObjPaths(outputBase);
    std::vector<vc::core::io::NamedPolyline> perpendicularSame;
    std::vector<vc::core::io::NamedPolyline> perpendicularSeparate;
    std::vector<vc::core::io::NamedPolyline> parallelSame;
    std::vector<vc::core::io::NamedPolyline> parallelSeparate;
    for (const auto& constraint : report.constraints) {
        if (constraint.hardContinuity)
            continue;
        vc::core::io::NamedPolyline line{
            "constraint_piece_" + std::to_string(constraint.pieceA) + "_" +
                std::to_string(constraint.pieceB),
            {constraint.pointABaseXYZ, constraint.pointBBaseXYZ},
        };
        if (constraint.perpendicularScore > 0.5) {
            if (constraint.windingDistance < 0.5)
                perpendicularSame.push_back(std::move(line));
            else
                perpendicularSeparate.push_back(std::move(line));
        } else if (constraint.parallelScore > 0.5) {
            if (constraint.parallelWindingDistance < 0.5)
                parallelSame.push_back(std::move(line));
            else
                parallelSeparate.push_back(std::move(line));
        }
    }
    const auto directory = result.paths.perpendicularSameWinding.parent_path();
    if (!directory.empty())
        std::filesystem::create_directories(directory);
    vc::core::io::writePolylinesObj(
        perpendicularSame, result.paths.perpendicularSameWinding,
        "VC3D perpendicular same-winding crop-trace constraints");
    vc::core::io::writePolylinesObj(
        perpendicularSeparate, result.paths.perpendicularSeparateWinding,
        "VC3D perpendicular separate-winding crop-trace constraints");
    vc::core::io::writePolylinesObj(
        parallelSame, result.paths.parallelSameWinding,
        "VC3D parallel same-winding crop-trace constraints");
    vc::core::io::writePolylinesObj(
        parallelSeparate, result.paths.parallelSeparateWinding,
        "VC3D parallel separate-winding crop-trace constraints");
    result.perpendicularSameWinding = perpendicularSame.size();
    result.perpendicularSeparateWinding = perpendicularSeparate.size();
    result.parallelSameWinding = parallelSame.size();
    result.parallelSeparateWinding = parallelSeparate.size();
    return result;
}

}  // namespace vc::fiber_tracer
