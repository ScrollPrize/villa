#include "vc/atlas/FiberIntersections.hpp"

#include "vc/lasagna/LasagnaNormalSampler.hpp"

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
constexpr double kDefaultMaxSampleSpacing = 100.0;
constexpr int kDefaultSeedStride = 100;

using Point3 = bg::model::point<double, 3, bg::cs::cartesian>;
using Box3 = bg::model::box<Point3>;

struct FiberDenseSample {
    int denseSampleIndex = -1;
    int segmentIndex = -1;
    cv::Vec3d position{0.0, 0.0, 0.0};
    double arclength = 0.0;
};

struct FiberPointEntry {
    uint64_t fiberId = 0;
    uint64_t generation = 1;
    int denseSampleIndex = -1;
    int segmentIndex = -1;
    cv::Vec3d position{0.0, 0.0, 0.0};
    double arclength = 0.0;
};

using PointRTreeValue = std::pair<Point3, FiberPointEntry>;
using PointTree = bgi::rtree<PointRTreeValue, bgi::quadratic<32>>;

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

Box3 pointQueryBox(const cv::Vec3d& point, double radius)
{
    return bgBox(point - cv::Vec3d{radius, radius, radius},
                 point + cv::Vec3d{radius, radius, radius});
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

double sanitizedSampleSpacing(double spacing)
{
    return std::isfinite(spacing) && spacing > 0.0 ? spacing : kDefaultMaxSampleSpacing;
}

int sanitizedSeedStride(int stride)
{
    return stride > 0 ? stride : kDefaultSeedStride;
}

std::vector<FiberDenseSample> denseSamplesForFiber(const FiberPolyline& fiber, double maxSampleSpacing)
{
    std::vector<FiberDenseSample> samples;
    if (fiber.points.size() < 2) {
        return samples;
    }

    const double spacing = sanitizedSampleSpacing(maxSampleSpacing);
    const auto lengths = cumulativeArclengths(fiber);
    auto addSample = [&samples](int segmentIndex,
                                const cv::Vec3d& position,
                                double arclength) {
        if (!finitePoint(position) || !std::isfinite(arclength)) {
            return;
        }
        if (!samples.empty() &&
            std::abs(samples.back().arclength - arclength) <= kEpsilon &&
            norm(samples.back().position - position) <= kEpsilon) {
            return;
        }
        FiberDenseSample sample;
        sample.denseSampleIndex = static_cast<int>(samples.size());
        sample.segmentIndex = segmentIndex;
        sample.position = position;
        sample.arclength = arclength;
        samples.push_back(sample);
    };

    for (size_t i = 1; i < fiber.points.size(); ++i) {
        const cv::Vec3d a = fiber.points[i - 1].position;
        const cv::Vec3d b = fiber.points[i].position;
        if (!finitePoint(a) || !finitePoint(b)) {
            continue;
        }
        const double segmentLength = lengths[i] - lengths[i - 1];
        if (!std::isfinite(segmentLength) || segmentLength <= kEpsilon) {
            continue;
        }
        const int segmentIndex = static_cast<int>(i - 1);
        addSample(segmentIndex, a, lengths[i - 1]);
        const int steps = std::max(1, static_cast<int>(std::ceil(segmentLength / spacing)));
        for (int step = 1; step <= steps; ++step) {
            const double t = static_cast<double>(step) / static_cast<double>(steps);
            addSample(segmentIndex, a * (1.0 - t) + b * t, lengths[i - 1] + segmentLength * t);
        }
    }

    return samples;
}

std::vector<PointRTreeValue> pointValuesForFiber(const FiberPolyline& fiber,
                                                 const std::vector<FiberDenseSample>& samples)
{
    std::vector<PointRTreeValue> values;
    values.reserve(samples.size());
    for (const auto& sample : samples) {
        if (!finitePoint(sample.position)) {
            continue;
        }
        FiberPointEntry entry;
        entry.fiberId = fiber.id;
        entry.generation = fiber.generation;
        entry.denseSampleIndex = sample.denseSampleIndex;
        entry.segmentIndex = sample.segmentIndex;
        entry.position = sample.position;
        entry.arclength = sample.arclength;
        values.emplace_back(bgPoint(sample.position), entry);
    }
    return values;
}

double squaredDistance(const cv::Vec3d& a, const cv::Vec3d& b)
{
    const cv::Vec3d delta = a - b;
    return dot(delta, delta);
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
    const vc::lasagna::LasagnaNormalSampler* windingSampler = nullptr;

    bool operator()(const double* const sourceS,
                    const double* const targetS,
                    double* residuals) const
    {
        const FiberSample a = sampleFiber(source, sourceS[0]);
        const FiberSample b = sampleFiber(target, targetS[0]);
        if (windingSampler) {
            double windingDistance = windingSampler->windingDistance(a.position, b.position);
            if (!std::isfinite(windingDistance)) {
                windingDistance = norm(a.position - b.position);
            }
            residuals[0] = options.distanceWeight * windingDistance;
            residuals[1] = 0.0;
            residuals[2] = 0.0;
        } else {
            const cv::Vec3d delta = (a.position - b.position) * options.distanceWeight;
            residuals[0] = delta[0];
            residuals[1] = delta[1];
            residuals[2] = delta[2];
        }
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

FiberIntersectionCandidate normalizedCandidateForPair(const FiberIntersectionCandidate& candidate,
                                                       uint64_t sourceFiberId,
                                                       uint64_t targetFiberId)
{
    if (candidate.sourceFiberId == sourceFiberId &&
        candidate.targetFiberId == targetFiberId) {
        return candidate;
    }
    FiberIntersectionCandidate normalized = candidate;
    std::swap(normalized.sourceFiberId, normalized.targetFiberId);
    std::swap(normalized.sourceGeneration, normalized.targetGeneration);
    std::swap(normalized.sourceSegmentIndex, normalized.targetSegmentIndex);
    std::swap(normalized.sourceArclength, normalized.targetArclength);
    return normalized;
}

FiberIntersectionResult normalizedResultForPair(FiberIntersectionResult result,
                                                uint64_t sourceFiberId,
                                                uint64_t targetFiberId)
{
    if (result.sourceFiberId == sourceFiberId &&
        result.targetFiberId == targetFiberId) {
        return result;
    }
    if (result.sourceFiberId == targetFiberId &&
        result.targetFiberId == sourceFiberId) {
        std::swap(result.sourceFiberId, result.targetFiberId);
        std::swap(result.sourceGeneration, result.targetGeneration);
        std::swap(result.sourceArclength, result.targetArclength);
        std::swap(result.sourcePoint, result.targetPoint);
    }
    return result;
}

} // namespace

struct FiberSpatialIndex::Impl {
    std::vector<FiberPolyline> committedFibers;
    mutable std::unordered_map<uint64_t, std::vector<FiberDenseSample>> committedSamples;
    mutable std::vector<PointRTreeValue> committedValues;
    mutable PointTree committedTree;
    std::array<std::optional<FiberPolyline>, 2> recentFibers;
    mutable std::array<std::vector<FiberDenseSample>, 2> recentSamples;
    mutable std::array<std::vector<PointRTreeValue>, 2> recentValues;
    mutable std::array<PointTree, 2> recentTrees;
    std::unordered_map<uint64_t, uint64_t> generations;
    mutable double indexedMaxSampleSpacing = std::numeric_limits<double>::quiet_NaN();

    void rebuildCommitted(double maxSampleSpacing) const
    {
        committedSamples.clear();
        committedValues.clear();
        for (const auto& fiber : committedFibers) {
            auto samples = denseSamplesForFiber(fiber, maxSampleSpacing);
            auto values = pointValuesForFiber(fiber, samples);
            committedValues.insert(committedValues.end(), values.begin(), values.end());
            committedSamples[fiber.id] = std::move(samples);
        }
        committedTree = PointTree(committedValues.begin(), committedValues.end());
    }

    void rebuildRecent(size_t slot, double maxSampleSpacing) const
    {
        recentSamples[slot].clear();
        recentValues[slot].clear();
        if (recentFibers[slot]) {
            recentSamples[slot] = denseSamplesForFiber(*recentFibers[slot], maxSampleSpacing);
            recentValues[slot] = pointValuesForFiber(*recentFibers[slot], recentSamples[slot]);
        }
        recentTrees[slot] = PointTree(recentValues[slot].begin(), recentValues[slot].end());
    }

    void rebuildAll(double maxSampleSpacing) const
    {
        const double spacing = sanitizedSampleSpacing(maxSampleSpacing);
        rebuildCommitted(spacing);
        rebuildRecent(0, spacing);
        rebuildRecent(1, spacing);
        indexedMaxSampleSpacing = spacing;
    }

    void ensureSpacing(double maxSampleSpacing) const
    {
        const double spacing = sanitizedSampleSpacing(maxSampleSpacing);
        if (!std::isfinite(indexedMaxSampleSpacing) ||
            std::abs(indexedMaxSampleSpacing - spacing) > kEpsilon) {
            rebuildAll(spacing);
        }
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
    impl_->rebuildCommitted(std::isfinite(impl_->indexedMaxSampleSpacing)
                                ? impl_->indexedMaxSampleSpacing
                                : kDefaultMaxSampleSpacing);
}

void FiberSpatialIndex::upsertRecent(const FiberPolyline& fiber)
{
    if (!impl_) clear();
    if (impl_->recentFibers[0] && impl_->recentFibers[0]->id == fiber.id) {
        impl_->recentFibers[0] = fiber;
        impl_->rebuildRecent(0, std::isfinite(impl_->indexedMaxSampleSpacing)
                                ? impl_->indexedMaxSampleSpacing
                                : kDefaultMaxSampleSpacing);
    } else if (impl_->recentFibers[1] && impl_->recentFibers[1]->id == fiber.id) {
        impl_->recentFibers[1] = fiber;
        impl_->rebuildRecent(1, std::isfinite(impl_->indexedMaxSampleSpacing)
                                ? impl_->indexedMaxSampleSpacing
                                : kDefaultMaxSampleSpacing);
    } else {
        impl_->recentFibers[1] = impl_->recentFibers[0];
        impl_->recentValues[1] = std::move(impl_->recentValues[0]);
        impl_->recentSamples[1] = std::move(impl_->recentSamples[0]);
        impl_->recentTrees[1] = PointTree(impl_->recentValues[1].begin(), impl_->recentValues[1].end());
        impl_->recentFibers[0] = fiber;
        impl_->rebuildRecent(0, std::isfinite(impl_->indexedMaxSampleSpacing)
                                ? impl_->indexedMaxSampleSpacing
                                : kDefaultMaxSampleSpacing);
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
            impl_->rebuildRecent(i, std::isfinite(impl_->indexedMaxSampleSpacing)
                                    ? impl_->indexedMaxSampleSpacing
                                    : kDefaultMaxSampleSpacing);
        }
    }
    impl_->generations.erase(fiberId);
    impl_->rebuildCommitted(std::isfinite(impl_->indexedMaxSampleSpacing)
                                ? impl_->indexedMaxSampleSpacing
                                : kDefaultMaxSampleSpacing);
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

    const double maxDistance = std::isfinite(options.maxDistance) && options.maxDistance >= 0.0
        ? options.maxDistance
        : 0.0;
    impl_->ensureSpacing(options.maxSampleSpacing);
    const auto sourceSamples = denseSamplesForFiber(source, options.maxSampleSpacing);
    if (sourceSamples.empty()) {
        return {};
    }

    std::vector<FiberIntersectionCandidate> candidates;
    const double maxDistanceSq = maxDistance * maxDistance;
    std::unordered_map<uint64_t, std::vector<int>> coverageByTarget;

    struct OrderedHit {
        FiberPointEntry entry;
        double distanceSq = std::numeric_limits<double>::infinity();
    };

    auto targetSamplesFor = [&](const FiberPointEntry& entry,
                                bool committed,
                                size_t recentSlot) -> const std::vector<FiberDenseSample>* {
        if (committed) {
            const auto it = impl_->committedSamples.find(entry.fiberId);
            return it == impl_->committedSamples.end() ? nullptr : &it->second;
        }
        if (recentSlot >= impl_->recentSamples.size()) {
            return nullptr;
        }
        return &impl_->recentSamples[recentSlot];
    };

    auto directLocalSearch = [&](int sourceStartIndex,
                                 const std::vector<FiberDenseSample>& targetSamples,
                                 int targetStartIndex,
                                 uint64_t targetFiberId,
                                 uint64_t targetGeneration) {
        struct DirectSearchResult {
            FiberIntersectionCandidate candidate;
            std::vector<int> visitedSourceIndices;
        };

        int sourceIndex = std::clamp(sourceStartIndex, 0, static_cast<int>(sourceSamples.size()) - 1);
        int targetIndex = std::clamp(targetStartIndex, 0, static_cast<int>(targetSamples.size()) - 1);
        std::vector<int> visitedSourceIndices;
        visitedSourceIndices.push_back(sourceIndex);

        double bestDistanceSq = squaredDistance(sourceSamples[sourceIndex].position,
                                                targetSamples[targetIndex].position);
        for (;;) {
            int bestSourceIndex = sourceIndex;
            int bestTargetIndex = targetIndex;
            for (int ds = -1; ds <= 1; ++ds) {
                const int nextSourceIndex = sourceIndex + ds;
                if (nextSourceIndex < 0 ||
                    nextSourceIndex >= static_cast<int>(sourceSamples.size())) {
                    continue;
                }
                for (int dt = -1; dt <= 1; ++dt) {
                    if (ds == 0 && dt == 0) {
                        continue;
                    }
                    const int nextTargetIndex = targetIndex + dt;
                    if (nextTargetIndex < 0 ||
                        nextTargetIndex >= static_cast<int>(targetSamples.size())) {
                        continue;
                    }
                    const double distanceSq = squaredDistance(
                        sourceSamples[nextSourceIndex].position,
                        targetSamples[nextTargetIndex].position);
                    if (distanceSq < bestDistanceSq - kEpsilon) {
                        bestDistanceSq = distanceSq;
                        bestSourceIndex = nextSourceIndex;
                        bestTargetIndex = nextTargetIndex;
                    }
                }
            }
            if (bestSourceIndex == sourceIndex && bestTargetIndex == targetIndex) {
                break;
            }
            sourceIndex = bestSourceIndex;
            targetIndex = bestTargetIndex;
            visitedSourceIndices.push_back(sourceIndex);
        }

        const auto& sourceSample = sourceSamples[sourceIndex];
        const auto& targetSample = targetSamples[targetIndex];
        DirectSearchResult result;
        result.candidate = FiberIntersectionCandidate{
            source.id,
            source.generation,
            sourceSample.segmentIndex,
            sourceSample.arclength,
            targetFiberId,
            targetGeneration,
            targetSample.segmentIndex,
            targetSample.arclength,
            std::sqrt(std::max(0.0, bestDistanceSq)),
        };
        result.visitedSourceIndices = std::move(visitedSourceIndices);
        return result;
    };

    auto scanTree = [&](const PointTree& tree, bool committed, size_t recentSlot) {
        std::vector<PointRTreeValue> pointHits;
        std::vector<OrderedHit> hits;
        std::vector<char> processed(sourceSamples.size(), 0);
        const int stride = sanitizedSeedStride(options.seedStride);

        auto processSourceIndex = [&](int sourceIndex) {
            processed[static_cast<size_t>(sourceIndex)] = 1;
            const auto& sourceSample = sourceSamples[static_cast<size_t>(sourceIndex)];
            pointHits.clear();
            hits.clear();
            tree.query(bgi::intersects(pointQueryBox(sourceSample.position, maxDistance)),
                       std::back_inserter(pointHits));
            for (const auto& pointHit : pointHits) {
                const auto& target = pointHit.second;
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
                const double distanceSq = squaredDistance(sourceSample.position, target.position);
                if (distanceSq > maxDistanceSq) {
                    continue;
                }
                hits.push_back(OrderedHit{target, distanceSq});
            }
            std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) {
                if (a.distanceSq != b.distanceSq) return a.distanceSq < b.distanceSq;
                if (a.entry.fiberId != b.entry.fiberId) return a.entry.fiberId < b.entry.fiberId;
                return a.entry.denseSampleIndex < b.entry.denseSampleIndex;
            });

            for (const auto& hit : hits) {
                auto& coverage = coverageByTarget[hit.entry.fiberId];
                if (coverage.empty()) {
                    coverage.assign(sourceSamples.size(), -1);
                }
                if (coverage[static_cast<size_t>(sourceIndex)] != -1) {
                    continue;
                }
                const auto* targetSamples = targetSamplesFor(hit.entry, committed, recentSlot);
                if (!targetSamples ||
                    hit.entry.denseSampleIndex < 0 ||
                    hit.entry.denseSampleIndex >= static_cast<int>(targetSamples->size()) ||
                    targetSamples->empty()) {
                    continue;
                }

                auto direct = directLocalSearch(sourceIndex,
                                                *targetSamples,
                                                hit.entry.denseSampleIndex,
                                                hit.entry.fiberId,
                                                hit.entry.generation);
                if (direct.candidate.straightDistance > maxDistance) {
                    continue;
                }
                const int resultIndex = static_cast<int>(candidates.size());
                candidates.push_back(std::move(direct.candidate));
                for (const int visitedSourceIndex : direct.visitedSourceIndices) {
                    if (visitedSourceIndex >= 0 &&
                        visitedSourceIndex < static_cast<int>(coverage.size())) {
                        coverage[static_cast<size_t>(visitedSourceIndex)] = resultIndex;
                    }
                }
            }
        };

        for (size_t i = 0; i < sourceSamples.size(); i += static_cast<size_t>(stride)) {
            processSourceIndex(static_cast<int>(i));
        }
        for (size_t i = 0; i < sourceSamples.size(); ++i) {
            if (!processed[i]) {
                processSourceIndex(static_cast<int>(i));
            }
        }
    };

    scanTree(impl_->committedTree, true, 0);
    scanTree(impl_->recentTrees[0], false, 0);
    scanTree(impl_->recentTrees[1], false, 1);
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
    const FiberIntersectionCeresOptions& options,
    const vc::lasagna::LasagnaNormalSampler* windingSampler)
{
    double sourceS = std::clamp(candidate.sourceArclength, 0.0, totalLength(source));
    double targetS = std::clamp(candidate.targetArclength, 0.0, totalLength(target));

    ceres::Problem problem;
    auto* residual = new ceres::NumericDiffCostFunction<JointIntersectionResidual,
                                                        ceres::CENTRAL,
                                                        5,
                                                        1,
                                                        1>(
        new JointIntersectionResidual{source, target, options, windingSampler});
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
    result.windingDistance = windingSampler
        ? windingSampler->windingDistance(sourceSample.position, targetSample.position)
        : norm(sourceSample.position - targetSample.position);
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
    const FiberIntersectionCeresOptions& ceres,
    const vc::lasagna::LasagnaNormalSampler* windingSampler)
{
    if (windingSampler) {
        cache = nullptr;
    }

    std::unordered_map<uint64_t, const FiberPolyline*> byId;
    for (const auto& fiber : fibers) {
        byId[fiber.id] = &fiber;
        index.upsertCommitted(fiber);
    }

    std::unordered_set<uint64_t> sourceSet(sourceFiberIds.begin(), sourceFiberIds.end());
    std::unordered_set<uint64_t> targetSet(targetFiberIds.begin(), targetFiberIds.end());
    std::set<std::pair<uint64_t, uint64_t>> searchedPairs;
    std::vector<FiberIntersectionResult> allResults;
    std::unordered_map<uint64_t, std::vector<FiberIntersectionCandidate>> candidateCache;

    auto candidatesFor = [&](const FiberPolyline& fiber) -> const std::vector<FiberIntersectionCandidate>& {
        auto it = candidateCache.find(fiber.id);
        if (it == candidateCache.end()) {
            it = candidateCache.emplace(fiber.id, index.candidatesForFiber(fiber, broad)).first;
        }
        return it->second;
    };

    for (uint64_t sourceId : sourceFiberIds) {
        auto sourceIt = byId.find(sourceId);
        if (sourceIt == byId.end()) {
            continue;
        }
        const FiberPolyline& source = *sourceIt->second;
        for (uint64_t targetId : targetFiberIds) {
            if (targetId == sourceId) {
                continue;
            }
            auto targetIt = byId.find(targetId);
            if (targetIt == byId.end()) {
                continue;
            }
            const FiberPolyline& target = *targetIt->second;
            const uint64_t a = std::min(source.id, target.id);
            const uint64_t b = std::max(source.id, target.id);
            if (sourceSet.count(target.id) &&
                targetSet.count(source.id) &&
                searchedPairs.count({a, b})) {
                continue;
            }

            std::vector<FiberIntersectionResult> pairResults;
            if (cache && cache->lookup(source.id,
                                       source.generation,
                                       target.id,
                                       target.generation,
                                       broad,
                                       ceres,
                                       pairResults)) {
                for (auto& result : pairResults) {
                    result = normalizedResultForPair(std::move(result), source.id, target.id);
                }
                allResults.insert(allResults.end(), pairResults.begin(), pairResults.end());
                searchedPairs.insert({a, b});
                continue;
            }

            std::vector<FiberIntersectionCandidate> pairCandidates;
            for (const auto& c : candidatesFor(source)) {
                if (c.targetFiberId == target.id) {
                    pairCandidates.push_back(c);
                }
            }
            for (const auto& c : candidatesFor(target)) {
                if (c.targetFiberId == source.id) {
                    pairCandidates.push_back(normalizedCandidateForPair(c, source.id, target.id));
                }
            }
            pairCandidates = clusterCandidates(std::move(pairCandidates), broad.clusterArclength);
            if (pairCandidates.empty()) {
                searchedPairs.insert({a, b});
                continue;
            }

            std::vector<std::future<FiberIntersectionResult>> futures;
            futures.reserve(pairCandidates.size());
            for (const auto& c : pairCandidates) {
                futures.push_back(std::async(std::launch::async,
                                             [&source, &target, c, ceres, windingSampler]() {
                                                 return refineFiberIntersectionCandidate(
                                                     source,
                                                     target,
                                                     c,
                                                     ceres,
                                                     windingSampler);
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
                             target.id,
                             target.generation,
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
