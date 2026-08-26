#include "vc/fiber_tracer/FiberTraceConstraints.hpp"

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
};

enum class RejectReason { None, Tangent, Winding };

struct ScoredCandidate {
    std::optional<FiberTraceConstraint> constraint;
    RejectReason rejection = RejectReason::None;
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
        std::isfinite(config.windingIntegrationStepBaseVoxels) &&
        config.windingIntegrationStepBaseVoxels > 0.0;
    if (!valid)
        throw std::invalid_argument("Fiber trace constraint configuration is invalid");
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

std::optional<std::pair<double, double>> refinedArcs(
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

ScoredCandidate scoreCandidate(
    const PairCandidate& candidate,
    const std::vector<FiberTraceConstraintPiece>& pieces,
    const std::vector<std::optional<PolylineArcGeometry>>& geometries,
    const FiberTraceConstraintConfig& config,
    const FiberTraceWindingDistance& windingDistance)
{
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
        return {{}, RejectReason::Tangent};

    const double initialDot = std::clamp(initialA->dot(*initialB), -1.0, 1.0);
    const int orientation = initialDot < 0.0 ? -1 : 1;
    double parallelSum = std::clamp(
        initialA->dot(*initialB * static_cast<double>(orientation)), -1.0, 1.0);
    std::size_t parallelCount = 1;
    const double refinementStep = config.resampleSpacingBaseVoxels *
        config.phaseRefinementStepFraction;
    const double refinementLimit = config.resampleSpacingBaseVoxels *
        config.phaseRefinementLimitFraction;

    for (const int walkDirection : {-1, 1}) {
        double phase = 0.0;
        for (std::size_t step = 1;; ++step) {
            const double distance = static_cast<double>(step) *
                config.resampleSpacingBaseVoxels;
            const double nominalA = arcA + static_cast<double>(walkDirection) * distance;
            const double nominalB = arcB +
                static_cast<double>(orientation * walkDirection) * distance;
            if (nominalA < pieceA.beginArcBaseVoxels - kEpsilon ||
                nominalA > pieceA.endArcBaseVoxels + kEpsilon ||
                nominalB < pieceB.beginArcBaseVoxels - kEpsilon ||
                nominalB > pieceB.endArcBaseVoxels + kEpsilon) {
                break;
            }
            double selectedPhase = phase;
            const auto arcs = refinedArcs(
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
            ++parallelCount;
        }
    }

    const double rawParallel = std::clamp(
        parallelSum / static_cast<double>(parallelCount), 0.0, 1.0);
    const double rawPerpendicular = 1.0 - std::abs(initialDot);
    const double evidence = rawParallel + rawPerpendicular;
    if (!(evidence > kEpsilon) || !std::isfinite(evidence))
        return {{}, RejectReason::Tangent};

    const cv::Vec3d pointA = pieceA.samplePointsBaseXYZ[candidate.sampleA];
    const cv::Vec3d pointB = pieceB.samplePointsBaseXYZ[candidate.sampleB];
    const double winding = windingDistance(
        pointA,
        pointB,
        config.windingIntegrationStepBaseVoxels);
    if (!std::isfinite(winding))
        return {{}, RejectReason::Winding};

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
    constraint.windingDistance = winding;
    return {std::move(constraint), RejectReason::None};
}

}  // namespace

FiberTraceConstraintReport extractFiberTraceConstraints(
    const std::vector<FiberletCropTraceLine>& lines,
    const FiberTraceConstraintConfig& config,
    const FiberTraceWindingDistance& windingDistance)
{
    validateConfig(config);
    FiberTraceConstraintReport report;
    report.inputTraces = lines.size();
    std::vector<std::optional<PolylineArcGeometry>> geometries(lines.size());
    std::vector<std::vector<std::size_t>> piecesByTrace(lines.size());

    const auto prepareStarted = std::chrono::steady_clock::now();
    for (std::size_t traceIndex = 0; traceIndex < lines.size(); ++traceIndex) {
        try {
            geometries[traceIndex] = makePolylineArcGeometry(
                lines[traceIndex].pointsBaseXYZ);
        } catch (const std::invalid_argument&) {
            ++report.skippedDegenerateTraces;
            continue;
        }
        const auto intervals = pieceIntervals(
            geometries[traceIndex]->length(), config);
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
                config.resampleSpacingBaseVoxels));
            report.resampledPoints +=
                report.pieces.back().samplePointsBaseXYZ.size();
        }
    }

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
            const auto found = bestByPair.find(candidate.pieces);
            if (found == bestByPair.end())
                bestByPair.emplace(candidate.pieces, candidate);
            else if (betterCandidate(candidate, found->second))
                found->second = candidate;
        }
    }
    std::vector<PairCandidate> candidates;
    candidates.reserve(bestByPair.size());
    for (const auto& [key, candidate] : bestByPair) {
        (void)key;
        candidates.push_back(candidate);
    }
    report.measuredCandidates = candidates.size();
    report.searchSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - searchStarted).count();

    if (!candidates.empty() && !windingDistance)
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
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers)
#endif
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(candidates.size());
         ++index) {
        try {
            scored[static_cast<std::size_t>(index)] = scoreCandidate(
                candidates[static_cast<std::size_t>(index)],
                report.pieces,
                geometries,
                config,
                windingDistance);
        } catch (...) {
            failures[static_cast<std::size_t>(index)] = std::current_exception();
        }
    }
    for (const auto& failure : failures) {
        if (failure)
            std::rethrow_exception(failure);
    }
    for (auto& result : scored) {
        if (result.constraint.has_value())
            report.constraints.push_back(std::move(*result.constraint));
        else if (result.rejection == RejectReason::Tangent)
            ++report.rejectedTangents;
        else if (result.rejection == RejectReason::Winding)
            ++report.rejectedWinding;
    }
    std::sort(report.constraints.begin(), report.constraints.end(),
        [](const auto& left, const auto& right) {
            return std::tie(left.pieceA, left.pieceB, left.hardContinuity) <
                std::tie(right.pieceA, right.pieceB, right.hardContinuity);
        });
    report.scoreSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scoreStarted).count();
    return report;
}

}  // namespace vc::fiber_tracer
