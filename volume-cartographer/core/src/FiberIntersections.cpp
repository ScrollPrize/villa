#include "vc/atlas/FiberIntersections.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <limits>
#include <set>
#include <unordered_set>
#include <utility>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <ceres/ceres.h>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace vc::atlas {
namespace {

constexpr double kEpsilon = 1.0e-12;

using Point3 = bg::model::point<double, 3, bg::cs::cartesian>;
using Box3 = bg::model::box<Point3>;
using RTreeValue = std::pair<Box3, FiberSegmentEntry>;
using SegmentTree = bgi::rtree<RTreeValue, bgi::quadratic<32>>;

double dot(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double norm(const cv::Vec3d& v)
{
    return std::sqrt(std::max(0.0, dot(v, v)));
}

cv::Vec3d normalizedOrZero(const cv::Vec3d& v)
{
    const double n = norm(v);
    if (n <= kEpsilon ||
        !std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
        return {0.0, 0.0, 0.0};
    }
    return v * (1.0 / n);
}

bool finitePoint(const cv::Vec3d& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

Point3 bgPoint(const cv::Vec3d& p)
{
    return Point3(p[0], p[1], p[2]);
}

Box3 bgBox(const cv::Vec3d& mn, const cv::Vec3d& mx)
{
    return Box3(bgPoint(mn), bgPoint(mx));
}

Box3 expandedBox(const FiberSegmentEntry& entry, double radius)
{
    return bgBox(entry.aabbMin - cv::Vec3d{radius, radius, radius},
                 entry.aabbMax + cv::Vec3d{radius, radius, radius});
}

std::vector<double> cumulativeArclengths(const FiberPolyline& fiber)
{
    std::vector<double> lengths(fiber.points.size(), 0.0);
    for (size_t i = 1; i < fiber.points.size(); ++i) {
        const double step = norm(fiber.points[i].position - fiber.points[i - 1].position);
        lengths[i] = lengths[i - 1] + (std::isfinite(step) ? step : 0.0);
    }
    return lengths;
}

double totalLength(const FiberPolyline& fiber)
{
    const auto lengths = cumulativeArclengths(fiber);
    return lengths.empty() ? 0.0 : lengths.back();
}

struct FiberSample {
    cv::Vec3d position{0.0, 0.0, 0.0};
    cv::Vec3d tangent{0.0, 0.0, 0.0};
    cv::Vec3d normal{0.0, 0.0, 0.0};
    bool hasNormal = false;
};

FiberSample sampleFiber(const FiberPolyline& fiber, double arclength)
{
    FiberSample sample;
    if (fiber.points.empty()) {
        return sample;
    }
    if (fiber.points.size() == 1) {
        sample.position = fiber.points.front().position;
        if (fiber.points.front().normal) {
            sample.normal = normalizedOrZero(*fiber.points.front().normal);
            sample.hasNormal = norm(sample.normal) > kEpsilon;
        }
        return sample;
    }

    const auto lengths = cumulativeArclengths(fiber);
    const double clamped = std::clamp(arclength, 0.0, lengths.back());
    size_t segment = 0;
    while (segment + 1 < lengths.size() && lengths[segment + 1] < clamped) {
        ++segment;
    }
    if (segment + 1 >= fiber.points.size()) {
        segment = fiber.points.size() - 2;
    }

    const double l0 = lengths[segment];
    const double l1 = lengths[segment + 1];
    const double span = std::max(kEpsilon, l1 - l0);
    const double t = std::clamp((clamped - l0) / span, 0.0, 1.0);
    const auto& a = fiber.points[segment];
    const auto& b = fiber.points[segment + 1];
    sample.position = a.position * (1.0 - t) + b.position * t;
    sample.tangent = normalizedOrZero(b.position - a.position);
    if (a.normal && b.normal) {
        sample.normal = normalizedOrZero(*a.normal * (1.0 - t) + *b.normal * t);
        sample.hasNormal = norm(sample.normal) > kEpsilon;
    } else if (a.normal) {
        sample.normal = normalizedOrZero(*a.normal);
        sample.hasNormal = norm(sample.normal) > kEpsilon;
    } else if (b.normal) {
        sample.normal = normalizedOrZero(*b.normal);
        sample.hasNormal = norm(sample.normal) > kEpsilon;
    }
    return sample;
}

struct ClosestSegments {
    double distance = std::numeric_limits<double>::infinity();
    double s = 0.0;
    double t = 0.0;
};

ClosestSegments closestSegmentParameters(const cv::Vec3d& p1,
                                         const cv::Vec3d& q1,
                                         const cv::Vec3d& p2,
                                         const cv::Vec3d& q2)
{
    const cv::Vec3d d1 = q1 - p1;
    const cv::Vec3d d2 = q2 - p2;
    const cv::Vec3d r = p1 - p2;
    const double a = dot(d1, d1);
    const double e = dot(d2, d2);
    const double f = dot(d2, r);

    double s = 0.0;
    double t = 0.0;
    if (a <= kEpsilon && e <= kEpsilon) {
        return {norm(p1 - p2), 0.0, 0.0};
    }
    if (a <= kEpsilon) {
        t = std::clamp(f / e, 0.0, 1.0);
    } else {
        const double c = dot(d1, r);
        if (e <= kEpsilon) {
            s = std::clamp(-c / a, 0.0, 1.0);
        } else {
            const double b = dot(d1, d2);
            const double denom = a * e - b * b;
            if (denom != 0.0) {
                s = std::clamp((b * f - c * e) / denom, 0.0, 1.0);
            }
            t = (b * s + f) / e;
            if (t < 0.0) {
                t = 0.0;
                s = std::clamp(-c / a, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = std::clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }

    const cv::Vec3d c1 = p1 + d1 * s;
    const cv::Vec3d c2 = p2 + d2 * t;
    return {norm(c1 - c2), s, t};
}

std::vector<FiberIntersectionCandidate> clusterCandidates(
    std::vector<FiberIntersectionCandidate> candidates,
    double arclengthTolerance)
{
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.sourceFiberId != b.sourceFiberId) return a.sourceFiberId < b.sourceFiberId;
        if (a.targetFiberId != b.targetFiberId) return a.targetFiberId < b.targetFiberId;
        if (a.straightDistance != b.straightDistance) return a.straightDistance < b.straightDistance;
        if (a.sourceArclength != b.sourceArclength) return a.sourceArclength < b.sourceArclength;
        return a.targetArclength < b.targetArclength;
    });

    std::vector<FiberIntersectionCandidate> clustered;
    for (const auto& candidate : candidates) {
        bool duplicate = false;
        for (const auto& kept : clustered) {
            if (kept.sourceFiberId != candidate.sourceFiberId ||
                kept.targetFiberId != candidate.targetFiberId) {
                continue;
            }
            if (std::abs(kept.sourceArclength - candidate.sourceArclength) <= arclengthTolerance &&
                std::abs(kept.targetArclength - candidate.targetArclength) <= arclengthTolerance) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            clustered.push_back(candidate);
        }
    }
    return clustered;
}

struct JointIntersectionResidual {
    const FiberPolyline& source;
    const FiberPolyline& target;
    FiberIntersectionCeresOptions options;

    bool operator()(const double* const sourceS,
                    const double* const targetS,
                    double* residuals) const
    {
        const FiberSample a = sampleFiber(source, sourceS[0]);
        const FiberSample b = sampleFiber(target, targetS[0]);
        const cv::Vec3d delta = (a.position - b.position) * options.distanceWeight;
        residuals[0] = delta[0];
        residuals[1] = delta[1];
        residuals[2] = delta[2];
        residuals[3] = 0.0;
        residuals[4] = 0.0;
        if (options.normalOrthogonalityWeight > 0.0 && a.hasNormal && norm(b.tangent) > kEpsilon) {
            const double d = dot(a.normal, b.tangent);
            residuals[3] = options.normalOrthogonalityWeight * d * d;
        }
        if (options.normalOrthogonalityWeight > 0.0 && b.hasNormal && norm(a.tangent) > kEpsilon) {
            const double d = dot(b.normal, a.tangent);
            residuals[4] = options.normalOrthogonalityWeight * d * d;
        }
        return true;
    }
};

std::array<uint64_t, 4> orderedPair(uint64_t aId, uint64_t aGen, uint64_t bId, uint64_t bGen)
{
    if (aId < bId || (aId == bId && aGen <= bGen)) {
        return {aId, aGen, bId, bGen};
    }
    return {bId, bGen, aId, aGen};
}

} // namespace

struct FiberSpatialIndex::Impl {
    std::vector<FiberPolyline> committedFibers;
    std::vector<RTreeValue> committedValues;
    SegmentTree committedTree;
    std::array<std::optional<FiberPolyline>, 2> recentFibers;
    std::array<std::vector<RTreeValue>, 2> recentValues;
    std::array<SegmentTree, 2> recentTrees;
    std::unordered_map<uint64_t, uint64_t> generations;

    void rebuildCommitted()
    {
        committedValues.clear();
        for (const auto& fiber : committedFibers) {
            for (const auto& segment : fiberSegments(fiber)) {
                committedValues.emplace_back(bgBox(segment.aabbMin, segment.aabbMax), segment);
            }
        }
        committedTree = SegmentTree(committedValues.begin(), committedValues.end());
    }

    void rebuildRecent(size_t slot)
    {
        recentValues[slot].clear();
        if (recentFibers[slot]) {
            for (const auto& segment : fiberSegments(*recentFibers[slot])) {
                recentValues[slot].emplace_back(bgBox(segment.aabbMin, segment.aabbMax), segment);
            }
        }
        recentTrees[slot] = SegmentTree(recentValues[slot].begin(), recentValues[slot].end());
    }

    bool hasRecentFiber(uint64_t fiberId) const
    {
        return std::any_of(recentFibers.begin(), recentFibers.end(), [fiberId](const auto& fiber) {
            return fiber && fiber->id == fiberId;
        });
    }
};

void FiberSpatialIndex::clear()
{
    impl_ = std::make_shared<Impl>();
}

void FiberSpatialIndex::upsertCommitted(const FiberPolyline& fiber)
{
    if (!impl_) clear();
    impl_->committedFibers.erase(
        std::remove_if(impl_->committedFibers.begin(),
                       impl_->committedFibers.end(),
                       [id = fiber.id](const FiberPolyline& existing) {
                           return existing.id == id;
                       }),
        impl_->committedFibers.end());
    impl_->committedFibers.push_back(fiber);
    impl_->generations[fiber.id] = fiber.generation;
    impl_->rebuildCommitted();
}

void FiberSpatialIndex::upsertRecent(const FiberPolyline& fiber)
{
    if (!impl_) clear();
    if (impl_->recentFibers[0] && impl_->recentFibers[0]->id == fiber.id) {
        impl_->recentFibers[0] = fiber;
        impl_->rebuildRecent(0);
    } else if (impl_->recentFibers[1] && impl_->recentFibers[1]->id == fiber.id) {
        impl_->recentFibers[1] = fiber;
        impl_->rebuildRecent(1);
    } else {
        impl_->recentFibers[1] = impl_->recentFibers[0];
        impl_->recentValues[1] = std::move(impl_->recentValues[0]);
        impl_->recentTrees[1] = SegmentTree(impl_->recentValues[1].begin(), impl_->recentValues[1].end());
        impl_->recentFibers[0] = fiber;
        impl_->rebuildRecent(0);
    }
    impl_->generations[fiber.id] = fiber.generation;
}

void FiberSpatialIndex::removeFiber(uint64_t fiberId)
{
    if (!impl_) clear();
    impl_->committedFibers.erase(
        std::remove_if(impl_->committedFibers.begin(),
                       impl_->committedFibers.end(),
                       [fiberId](const FiberPolyline& fiber) { return fiber.id == fiberId; }),
        impl_->committedFibers.end());
    for (size_t i = 0; i < impl_->recentFibers.size(); ++i) {
        if (impl_->recentFibers[i] && impl_->recentFibers[i]->id == fiberId) {
            impl_->recentFibers[i].reset();
            impl_->rebuildRecent(i);
        }
    }
    impl_->generations.erase(fiberId);
    impl_->rebuildCommitted();
}

uint64_t FiberSpatialIndex::generation(uint64_t fiberId) const
{
    if (!impl_) return 0;
    const auto it = impl_->generations.find(fiberId);
    return it == impl_->generations.end() ? 0 : it->second;
}

std::vector<FiberIntersectionCandidate> FiberSpatialIndex::candidatesForFiber(
    const FiberPolyline& source,
    const FiberIntersectionBroadPhaseOptions& options) const
{
    if (!impl_) {
        return {};
    }

    std::vector<FiberIntersectionCandidate> candidates;
    const auto sourceSegments = fiberSegments(source);
    auto scanTree = [&](const SegmentTree& tree, bool committed) {
        std::vector<RTreeValue> hits;
        for (const auto& sourceSegment : sourceSegments) {
            hits.clear();
            tree.query(bgi::intersects(expandedBox(sourceSegment, options.maxDistance)),
                       std::back_inserter(hits));
            for (const auto& hit : hits) {
                const auto& target = hit.second;
                if (target.fiberId == source.id) {
                    continue;
                }
                const auto genIt = impl_->generations.find(target.fiberId);
                if (genIt == impl_->generations.end() || genIt->second != target.generation) {
                    continue;
                }
                if (committed && impl_->hasRecentFiber(target.fiberId)) {
                    continue;
                }
                const ClosestSegments closest = closestSegmentParameters(
                    sourceSegment.a,
                    sourceSegment.b,
                    target.a,
                    target.b);
                if (closest.distance > options.maxDistance) {
                    continue;
                }
                const double sourceArc = sourceSegment.arclength0 +
                    (sourceSegment.arclength1 - sourceSegment.arclength0) * closest.s;
                const double targetArc = target.arclength0 +
                    (target.arclength1 - target.arclength0) * closest.t;
                candidates.push_back(FiberIntersectionCandidate{
                    source.id,
                    source.generation,
                    sourceSegment.segmentIndex,
                    sourceArc,
                    target.fiberId,
                    target.generation,
                    target.segmentIndex,
                    targetArc,
                    closest.distance,
                });
            }
        }
    };

    scanTree(impl_->committedTree, true);
    scanTree(impl_->recentTrees[0], false);
    scanTree(impl_->recentTrees[1], false);
    return clusterCandidates(std::move(candidates), options.clusterArclength);
}

bool FiberIntersectionCache::lookup(uint64_t fiberA,
                                    uint64_t generationA,
                                    uint64_t fiberB,
                                    uint64_t generationB,
                                    const FiberIntersectionBroadPhaseOptions& broad,
                                    const FiberIntersectionCeresOptions& ceres,
                                    std::vector<FiberIntersectionResult>& results) const
{
    const auto ordered = orderedPair(fiberA, generationA, fiberB, generationB);
    Key key{ordered[0], ordered[1], ordered[2], ordered[3], broad, ceres};
    const auto it = entries_.find(key);
    if (it == entries_.end()) {
        return false;
    }
    results = it->second;
    for (auto& result : results) {
        result.cacheHit = true;
        result.ceresSolves = 0;
    }
    return true;
}

void FiberIntersectionCache::store(uint64_t fiberA,
                                   uint64_t generationA,
                                   uint64_t fiberB,
                                   uint64_t generationB,
                                   const FiberIntersectionBroadPhaseOptions& broad,
                                   const FiberIntersectionCeresOptions& ceres,
                                   std::vector<FiberIntersectionResult> results)
{
    const auto ordered = orderedPair(fiberA, generationA, fiberB, generationB);
    Key key{ordered[0], ordered[1], ordered[2], ordered[3], broad, ceres};
    entries_[std::move(key)] = std::move(results);
}

void FiberIntersectionCache::pruneFiber(uint64_t fiberId)
{
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->first.fiberA == fiberId || it->first.fiberB == fiberId) {
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
}

void FiberIntersectionCache::clear()
{
    entries_.clear();
}

size_t FiberIntersectionCache::size() const
{
    return entries_.size();
}

std::vector<FiberSegmentEntry> fiberSegments(const FiberPolyline& fiber)
{
    std::vector<FiberSegmentEntry> segments;
    if (fiber.points.size() < 2) {
        return segments;
    }
    const auto lengths = cumulativeArclengths(fiber);
    segments.reserve(fiber.points.size() - 1);
    for (size_t i = 1; i < fiber.points.size(); ++i) {
        const cv::Vec3d a = fiber.points[i - 1].position;
        const cv::Vec3d b = fiber.points[i].position;
        if (!finitePoint(a) || !finitePoint(b)) {
            continue;
        }
        FiberSegmentEntry entry;
        entry.fiberId = fiber.id;
        entry.generation = fiber.generation;
        entry.segmentIndex = static_cast<int>(i - 1);
        entry.a = a;
        entry.b = b;
        entry.arclength0 = lengths[i - 1];
        entry.arclength1 = lengths[i];
        entry.aabbMin = {std::min(a[0], b[0]), std::min(a[1], b[1]), std::min(a[2], b[2])};
        entry.aabbMax = {std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2])};
        segments.push_back(entry);
    }
    return segments;
}

FiberIntersectionResult refineFiberIntersectionCandidate(
    const FiberPolyline& source,
    const FiberPolyline& target,
    const FiberIntersectionCandidate& candidate,
    const FiberIntersectionCeresOptions& options)
{
    double sourceS = std::clamp(candidate.sourceArclength, 0.0, totalLength(source));
    double targetS = std::clamp(candidate.targetArclength, 0.0, totalLength(target));

    ceres::Problem problem;
    auto* residual = new ceres::NumericDiffCostFunction<JointIntersectionResidual,
                                                        ceres::CENTRAL,
                                                        5,
                                                        1,
                                                        1>(
        new JointIntersectionResidual{source, target, options});
    problem.AddResidualBlock(residual, nullptr, &sourceS, &targetS);
    problem.SetParameterLowerBound(&sourceS, 0, 0.0);
    problem.SetParameterUpperBound(&sourceS, 0, totalLength(source));
    problem.SetParameterLowerBound(&targetS, 0, 0.0);
    problem.SetParameterUpperBound(&targetS, 0, totalLength(target));

    double initialCost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions{}, &initialCost, nullptr, nullptr, nullptr);

    ceres::Solver::Options solverOptions;
    solverOptions.max_num_iterations = std::max(0, options.maxIterations);
    solverOptions.num_threads = 1;
    solverOptions.linear_solver_type = ceres::DENSE_QR;
    solverOptions.logging_type = ceres::SILENT;
    solverOptions.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(solverOptions, &problem, &summary);

    double finalCost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions{}, &finalCost, nullptr, nullptr, nullptr);
    const FiberSample sourceSample = sampleFiber(source, sourceS);
    const FiberSample targetSample = sampleFiber(target, targetS);

    FiberIntersectionResult result;
    result.sourceFiberId = source.id;
    result.sourceGeneration = source.generation;
    result.targetFiberId = target.id;
    result.targetGeneration = target.generation;
    result.candidateDistance = candidate.straightDistance;
    result.refinedScore = finalCost;
    result.sourceArclength = sourceS;
    result.targetArclength = targetS;
    result.sourcePoint = sourceSample.position;
    result.targetPoint = targetSample.position;
    result.converged = summary.IsSolutionUsable();
    result.ceresSolves = 1;
    result.ceresIterations = static_cast<int>(summary.iterations.size());
    result.usedNormalResiduals =
        options.normalOrthogonalityWeight > 0.0 &&
        ((sourceSample.hasNormal && norm(targetSample.tangent) > kEpsilon) ||
         (targetSample.hasNormal && norm(sourceSample.tangent) > kEpsilon));
    result.message = summary.BriefReport();
    (void)initialCost;
    return result;
}

std::vector<FiberIntersectionResult> deduplicateFiberIntersectionResults(
    std::vector<FiberIntersectionResult> results,
    double arclengthTolerance)
{
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        if (a.sourceFiberId != b.sourceFiberId) return a.sourceFiberId < b.sourceFiberId;
        if (a.targetFiberId != b.targetFiberId) return a.targetFiberId < b.targetFiberId;
        if (a.refinedScore != b.refinedScore) return a.refinedScore < b.refinedScore;
        if (a.sourceArclength != b.sourceArclength) return a.sourceArclength < b.sourceArclength;
        return a.targetArclength < b.targetArclength;
    });

    std::vector<FiberIntersectionResult> deduped;
    for (const auto& result : results) {
        bool duplicate = false;
        for (const auto& kept : deduped) {
            if (kept.sourceFiberId != result.sourceFiberId ||
                kept.targetFiberId != result.targetFiberId) {
                continue;
            }
            if (std::abs(kept.sourceArclength - result.sourceArclength) <= arclengthTolerance &&
                std::abs(kept.targetArclength - result.targetArclength) <= arclengthTolerance) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            deduped.push_back(result);
        }
    }
    return deduped;
}

std::vector<FiberIntersectionResult> searchFiberIntersections(
    const std::vector<FiberPolyline>& fibers,
    const std::vector<uint64_t>& sourceFiberIds,
    const std::vector<uint64_t>& targetFiberIds,
    FiberSpatialIndex& index,
    FiberIntersectionCache* cache,
    const FiberIntersectionBroadPhaseOptions& broad,
    const FiberIntersectionCeresOptions& ceres)
{
    std::unordered_map<uint64_t, const FiberPolyline*> byId;
    for (const auto& fiber : fibers) {
        byId[fiber.id] = &fiber;
        index.upsertCommitted(fiber);
    }

    std::unordered_set<uint64_t> sourceSet(sourceFiberIds.begin(), sourceFiberIds.end());
    std::unordered_set<uint64_t> targetSet(targetFiberIds.begin(), targetFiberIds.end());
    std::set<std::pair<uint64_t, uint64_t>> searchedPairs;
    std::vector<FiberIntersectionResult> allResults;

    for (uint64_t sourceId : sourceFiberIds) {
        auto sourceIt = byId.find(sourceId);
        if (sourceIt == byId.end()) {
            continue;
        }
        const FiberPolyline& source = *sourceIt->second;
        for (const auto& candidate : index.candidatesForFiber(source, broad)) {
            if (!targetSet.count(candidate.targetFiberId)) {
                continue;
            }
            const uint64_t a = std::min(candidate.sourceFiberId, candidate.targetFiberId);
            const uint64_t b = std::max(candidate.sourceFiberId, candidate.targetFiberId);
            if (sourceSet.count(candidate.targetFiberId) &&
                targetSet.count(candidate.sourceFiberId) &&
                searchedPairs.count({a, b})) {
                continue;
            }
            auto targetIt = byId.find(candidate.targetFiberId);
            if (targetIt == byId.end()) {
                continue;
            }
            std::vector<FiberIntersectionResult> pairResults;
            if (cache && cache->lookup(source.id,
                                       source.generation,
                                       targetIt->second->id,
                                       targetIt->second->generation,
                                       broad,
                                       ceres,
                                       pairResults)) {
                allResults.insert(allResults.end(), pairResults.begin(), pairResults.end());
                searchedPairs.insert({a, b});
                continue;
            }

            std::vector<FiberIntersectionCandidate> pairCandidates;
            for (const auto& c : index.candidatesForFiber(source, broad)) {
                if (c.targetFiberId == targetIt->second->id) {
                    pairCandidates.push_back(c);
                }
            }
            pairCandidates = clusterCandidates(std::move(pairCandidates), broad.clusterArclength);
            std::vector<std::future<FiberIntersectionResult>> futures;
            futures.reserve(pairCandidates.size());
            for (const auto& c : pairCandidates) {
                futures.push_back(std::async(std::launch::async,
                                             [&source, target = targetIt->second, c, ceres]() {
                                                 return refineFiberIntersectionCandidate(
                                                     source,
                                                     *target,
                                                     c,
                                                     ceres);
                                             }));
            }
            for (auto& future : futures) {
                pairResults.push_back(future.get());
            }
            pairResults = deduplicateFiberIntersectionResults(std::move(pairResults),
                                                              ceres.deduplicateArclength);
            if (cache) {
                cache->store(source.id,
                             source.generation,
                             targetIt->second->id,
                             targetIt->second->generation,
                             broad,
                             ceres,
                             pairResults);
            }
            allResults.insert(allResults.end(), pairResults.begin(), pairResults.end());
            searchedPairs.insert({a, b});
        }
    }

    return allResults;
}

} // namespace vc::atlas
