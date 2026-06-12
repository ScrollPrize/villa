#include "vc/atlas/FiberIntersections.hpp"

#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
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

struct ArclengthDomain {
    double start = 0.0;
    double end = 0.0;
};

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

cv::Vec3d pointAtSegmentArclength(const cv::Vec3d& a,
                                  const cv::Vec3d& b,
                                  double segmentStart,
                                  double segmentEnd,
                                  double arclength)
{
    const double span = std::max(kEpsilon, segmentEnd - segmentStart);
    const double t = std::clamp((arclength - segmentStart) / span, 0.0, 1.0);
    return a * (1.0 - t) + b * t;
}

std::optional<double> closestArclengthOnPolyline(const FiberPolyline& fiber,
                                                 const std::vector<double>& lengths,
                                                 const cv::Vec3d& point)
{
    if (!finitePoint(point) || fiber.points.size() < 2 || lengths.size() != fiber.points.size()) {
        return std::nullopt;
    }
    double bestDistanceSq = std::numeric_limits<double>::infinity();
    double bestArclength = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 1; i < fiber.points.size(); ++i) {
        const cv::Vec3d a = fiber.points[i - 1].position;
        const cv::Vec3d b = fiber.points[i].position;
        if (!finitePoint(a) || !finitePoint(b)) {
            continue;
        }
        const cv::Vec3d ab = b - a;
        const double denom = dot(ab, ab);
        if (!std::isfinite(denom) || denom <= kEpsilon) {
            continue;
        }
        const double t = std::clamp(dot(point - a, ab) / denom, 0.0, 1.0);
        const cv::Vec3d projected = a + ab * t;
        const cv::Vec3d delta = point - projected;
        const double distanceSq = dot(delta, delta);
        if (distanceSq < bestDistanceSq) {
            bestDistanceSq = distanceSq;
            bestArclength = lengths[i - 1] + (lengths[i] - lengths[i - 1]) * t;
        }
    }
    if (!std::isfinite(bestArclength)) {
        return std::nullopt;
    }
    return bestArclength;
}

ArclengthDomain activeArclengthDomain(const FiberPolyline& fiber,
                                      const std::vector<double>& lengths)
{
    const double fullEnd = lengths.empty() ? 0.0 : lengths.back();
    ArclengthDomain domain{0.0, fullEnd};
    if (fiber.controlPoints.size() < 2 || fiber.points.size() < 2) {
        return domain;
    }

    double first = std::numeric_limits<double>::infinity();
    double last = -std::numeric_limits<double>::infinity();
    int finiteControls = 0;
    for (const cv::Vec3d& control : fiber.controlPoints) {
        const auto arclength = closestArclengthOnPolyline(fiber, lengths, control);
        if (!arclength || !std::isfinite(*arclength)) {
            continue;
        }
        first = std::min(first, *arclength);
        last = std::max(last, *arclength);
        ++finiteControls;
    }
    if (finiteControls < 2 || !std::isfinite(first) || !std::isfinite(last) ||
        last - first <= kEpsilon) {
        return domain;
    }
    domain.start = std::clamp(first, 0.0, fullEnd);
    domain.end = std::clamp(last, domain.start, fullEnd);
    return domain;
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
    const ArclengthDomain domain = activeArclengthDomain(fiber, lengths);
    if (domain.end - domain.start <= kEpsilon) {
        return samples;
    }
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
        const double clippedStart = std::max(lengths[i - 1], domain.start);
        const double clippedEnd = std::min(lengths[i], domain.end);
        if (clippedEnd - clippedStart <= kEpsilon) {
            continue;
        }
        const int segmentIndex = static_cast<int>(i - 1);
        addSample(segmentIndex,
                  pointAtSegmentArclength(a, b, lengths[i - 1], lengths[i], clippedStart),
                  clippedStart);
        const int steps = std::max(1, static_cast<int>(std::ceil((clippedEnd - clippedStart) / spacing)));
        for (int step = 1; step <= steps; ++step) {
            const double t = static_cast<double>(step) / static_cast<double>(steps);
            const double arclength = clippedStart + (clippedEnd - clippedStart) * t;
            addSample(segmentIndex,
                      pointAtSegmentArclength(a, b, lengths[i - 1], lengths[i], arclength),
                      arclength);
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
    FiberIntersectionCancelCallback cancelCallback;

    bool operator()(const double* const sourceS,
                    const double* const targetS,
                    double* residuals) const
    {
        if (cancelCallback && cancelCallback()) {
            return false;
        }
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

class FiberIntersectionCancelIterationCallback final : public ceres::IterationCallback {
public:
    explicit FiberIntersectionCancelIterationCallback(FiberIntersectionCancelCallback cancelCallback)
        : cancelCallback_(std::move(cancelCallback))
    {
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& /*summary*/) override
    {
        return cancelCallback_ && cancelCallback_()
            ? ceres::SOLVER_ABORT
            : ceres::SOLVER_CONTINUE;
    }

private:
    FiberIntersectionCancelCallback cancelCallback_;
};

class SearchFiberPointIndex {
public:
    explicit SearchFiberPointIndex(double maxSampleSpacing)
        : maxSampleSpacing_(sanitizedSampleSpacing(maxSampleSpacing))
    {
    }

    void insertFiber(const FiberPolyline& fiber)
    {
        auto samples = denseSamplesForFiber(fiber, maxSampleSpacing_);
        auto values = pointValuesForFiber(fiber, samples);
        samplesByFiberId_[fiber.id] = std::move(samples);
        values_.insert(values_.end(),
                       std::make_move_iterator(values.begin()),
                       std::make_move_iterator(values.end()));
    }

    void finalize()
    {
        tree_ = PointTree(values_.begin(), values_.end());
    }

    [[nodiscard]] std::vector<FiberIntersectionCandidate> candidatesForFiber(
        const FiberPolyline& source,
        const FiberIntersectionBroadPhaseOptions& options,
        FiberIntersectionCancelCallback cancelCallback) const
    {
        const double maxDistance = std::isfinite(options.maxDistance) && options.maxDistance >= 0.0
            ? options.maxDistance
            : 0.0;
        const auto sourceSamplesIt = samplesByFiberId_.find(source.id);
        const auto fallbackSourceSamples = sourceSamplesIt == samplesByFiberId_.end()
            ? denseSamplesForFiber(source, maxSampleSpacing_)
            : std::vector<FiberDenseSample>{};
        const auto& sourceSamples = sourceSamplesIt == samplesByFiberId_.end()
            ? fallbackSourceSamples
            : sourceSamplesIt->second;
        if (sourceSamples.empty() || values_.empty()) {
            return {};
        }

        std::vector<FiberIntersectionCandidate> candidates;
        const double maxDistanceSq = maxDistance * maxDistance;
        std::unordered_map<uint64_t, std::vector<int>> coverageByTarget;

        struct OrderedHit {
            FiberPointEntry entry;
            double distanceSq = std::numeric_limits<double>::infinity();
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
                if (cancelCallback && cancelCallback()) {
                    break;
                }
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

        std::vector<PointRTreeValue> pointHits;
        std::vector<OrderedHit> hits;
        std::vector<char> processed(sourceSamples.size(), 0);
        const int stride = sanitizedSeedStride(options.seedStride);

        auto processSourceIndex = [&](int sourceIndex) {
            if (cancelCallback && cancelCallback()) {
                return;
            }
            processed[static_cast<size_t>(sourceIndex)] = 1;
            const auto& sourceSample = sourceSamples[static_cast<size_t>(sourceIndex)];
            pointHits.clear();
            hits.clear();
            tree_.query(bgi::intersects(pointQueryBox(sourceSample.position, maxDistance)),
                        std::back_inserter(pointHits));
            for (const auto& pointHit : pointHits) {
                const auto& target = pointHit.second;
                if (target.fiberId == source.id) {
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
                if (cancelCallback && cancelCallback()) {
                    return;
                }
                auto& coverage = coverageByTarget[hit.entry.fiberId];
                if (coverage.empty()) {
                    coverage.assign(sourceSamples.size(), -1);
                }
                if (coverage[static_cast<size_t>(sourceIndex)] != -1) {
                    continue;
                }
                const auto targetSamplesIt = samplesByFiberId_.find(hit.entry.fiberId);
                if (targetSamplesIt == samplesByFiberId_.end() ||
                    hit.entry.denseSampleIndex < 0 ||
                    hit.entry.denseSampleIndex >= static_cast<int>(targetSamplesIt->second.size()) ||
                    targetSamplesIt->second.empty()) {
                    continue;
                }

                auto direct = directLocalSearch(sourceIndex,
                                                targetSamplesIt->second,
                                                hit.entry.denseSampleIndex,
                                                hit.entry.fiberId,
                                                hit.entry.generation);
                if (cancelCallback && cancelCallback()) {
                    return;
                }
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
            if (cancelCallback && cancelCallback()) {
                return {};
            }
            processSourceIndex(static_cast<int>(i));
        }
        for (size_t i = 0; i < sourceSamples.size(); ++i) {
            if (cancelCallback && cancelCallback()) {
                return {};
            }
            if (!processed[i]) {
                processSourceIndex(static_cast<int>(i));
            }
        }

        return clusterCandidates(std::move(candidates), options.clusterArclength);
    }

private:
    double maxSampleSpacing_ = kDefaultMaxSampleSpacing;
    std::unordered_map<uint64_t, std::vector<FiberDenseSample>> samplesByFiberId_;
    std::vector<PointRTreeValue> values_;
    PointTree tree_;
};

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

int atlasSearchPhaseProgressPercent(AtlasSearchProgressPhase phase, size_t completed, size_t total)
{
    constexpr int kPhaseCount = 4;
    constexpr int kPercentMax = 100;
    const int phaseIndex = std::clamp(static_cast<int>(phase), 0, kPhaseCount - 1);
    const int phaseBase = (phaseIndex * kPercentMax) / kPhaseCount;
    const int phaseEnd = ((phaseIndex + 1) * kPercentMax) / kPhaseCount;
    const int phaseSpan = phaseEnd - phaseBase;
    if (total == 0) {
        return phaseBase + (completed > 0 ? phaseSpan : 0);
    }
    const size_t clampedCompleted = std::min(completed, total);
    const double fraction = static_cast<double>(clampedCompleted) / static_cast<double>(total);
    return phaseBase + static_cast<int>(std::round(fraction * static_cast<double>(phaseSpan)));
}

std::vector<FiberIntersectionCandidate> FiberSpatialIndex::candidatesForFiber(
    const FiberPolyline& source,
    const FiberIntersectionBroadPhaseOptions& options,
    FiberIntersectionCancelCallback cancelCallback) const
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
            if (cancelCallback && cancelCallback()) {
                break;
            }
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
            if (cancelCallback && cancelCallback()) {
                return;
            }
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
                if (cancelCallback && cancelCallback()) {
                    return;
                }
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
                if (cancelCallback && cancelCallback()) {
                    return;
                }
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
            if (cancelCallback && cancelCallback()) {
                return;
            }
            processSourceIndex(static_cast<int>(i));
        }
        for (size_t i = 0; i < sourceSamples.size(); ++i) {
            if (cancelCallback && cancelCallback()) {
                return;
            }
            if (!processed[i]) {
                processSourceIndex(static_cast<int>(i));
            }
        }
    };

    scanTree(impl_->committedTree, true, 0);
    if (cancelCallback && cancelCallback()) {
        return {};
    }
    scanTree(impl_->recentTrees[0], false, 0);
    if (cancelCallback && cancelCallback()) {
        return {};
    }
    scanTree(impl_->recentTrees[1], false, 1);
    if (cancelCallback && cancelCallback()) {
        return {};
    }
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
    const vc::lasagna::LasagnaNormalSampler* windingSampler,
    FiberIntersectionCancelCallback cancelCallback)
{
    if (cancelCallback && cancelCallback()) {
        FiberIntersectionResult result;
        result.sourceFiberId = source.id;
        result.sourceGeneration = source.generation;
        result.targetFiberId = target.id;
        result.targetGeneration = target.generation;
        result.candidateDistance = candidate.straightDistance;
        result.sourceArclength = candidate.sourceArclength;
        result.targetArclength = candidate.targetArclength;
        result.message = "canceled";
        return result;
    }

    const auto sourceLengths = cumulativeArclengths(source);
    const auto targetLengths = cumulativeArclengths(target);
    const ArclengthDomain sourceDomain = activeArclengthDomain(source, sourceLengths);
    const ArclengthDomain targetDomain = activeArclengthDomain(target, targetLengths);
    double sourceS = std::clamp(candidate.sourceArclength, sourceDomain.start, sourceDomain.end);
    double targetS = std::clamp(candidate.targetArclength, targetDomain.start, targetDomain.end);

    ceres::Problem problem;
    auto* residual = new ceres::NumericDiffCostFunction<JointIntersectionResidual,
                                                        ceres::CENTRAL,
                                                        5,
                                                        1,
                                                        1>(
        new JointIntersectionResidual{source, target, options, windingSampler, cancelCallback});
    problem.AddResidualBlock(residual, nullptr, &sourceS, &targetS);
    problem.SetParameterLowerBound(&sourceS, 0, sourceDomain.start);
    problem.SetParameterUpperBound(&sourceS, 0, sourceDomain.end);
    problem.SetParameterLowerBound(&targetS, 0, targetDomain.start);
    problem.SetParameterUpperBound(&targetS, 0, targetDomain.end);

    double initialCost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions{}, &initialCost, nullptr, nullptr, nullptr);

    ceres::Solver::Options solverOptions;
    solverOptions.max_num_iterations = std::max(0, options.maxIterations);
    solverOptions.num_threads = 1;
    solverOptions.linear_solver_type = ceres::DENSE_QR;
    solverOptions.logging_type = ceres::SILENT;
    solverOptions.minimizer_progress_to_stdout = false;
    FiberIntersectionCancelIterationCallback cancelIterationCallback(cancelCallback);
    if (cancelCallback) {
        solverOptions.callbacks.push_back(&cancelIterationCallback);
    }
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

std::optional<size_t> nearestIntersectionResultByArclength(
    const std::vector<FiberIntersectionResult>& results,
    double sourceArclength,
    double targetArclength)
{
    if (!std::isfinite(sourceArclength) || !std::isfinite(targetArclength)) {
        return std::nullopt;
    }

    std::optional<size_t> bestIndex;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        if (!std::isfinite(result.sourceArclength) ||
            !std::isfinite(result.targetArclength)) {
            continue;
        }
        const double sourceDelta = result.sourceArclength - sourceArclength;
        const double targetDelta = result.targetArclength - targetArclength;
        const double distance = std::hypot(sourceDelta, targetDelta);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = i;
        }
    }
    return bestIndex;
}

std::vector<FiberIntersectionResult> searchFiberIntersections(
    const std::vector<FiberPolyline>& fibers,
    const std::vector<uint64_t>& sourceFiberIds,
    const std::vector<uint64_t>& targetFiberIds,
    FiberIntersectionCache* cache,
    const FiberIntersectionBroadPhaseOptions& broad,
    const FiberIntersectionCeresOptions& ceres,
    const vc::lasagna::LasagnaNormalSampler* windingSampler,
    FiberIntersectionProgressCallback progressCallback,
    FiberIntersectionCancelCallback cancelCallback)
{
    if (windingSampler) {
        cache = nullptr;
    }

    std::unordered_map<uint64_t, const FiberPolyline*> byId;
    SearchFiberPointIndex index(broad.maxSampleSpacing);
    if (progressCallback) {
        progressCallback(AtlasSearchProgressPhase::BuildSpatialIndex, 0, fibers.size());
    }
    size_t indexedFibers = 0;
    for (const auto& fiber : fibers) {
        if (cancelCallback && cancelCallback()) {
            return {};
        }
        byId[fiber.id] = &fiber;
        index.insertFiber(fiber);
        ++indexedFibers;
        if (progressCallback && indexedFibers < fibers.size()) {
            progressCallback(AtlasSearchProgressPhase::BuildSpatialIndex,
                             indexedFibers,
                             fibers.size());
        }
    }
    if (cancelCallback && cancelCallback()) {
        return {};
    }
    index.finalize();
    if (progressCallback) {
        progressCallback(AtlasSearchProgressPhase::BuildSpatialIndex,
                         fibers.size(),
                         fibers.size());
    }

    std::unordered_set<uint64_t> sourceSet(sourceFiberIds.begin(), sourceFiberIds.end());
    std::unordered_set<uint64_t> targetSet(targetFiberIds.begin(), targetFiberIds.end());
    const size_t pairTotal = sourceFiberIds.size() * targetFiberIds.size();

    struct PairJob {
        const FiberPolyline* source = nullptr;
        const FiberPolyline* target = nullptr;
        bool skip = true;
    };
    std::vector<PairJob> pairJobs;
    pairJobs.reserve(pairTotal);
    std::set<std::pair<uint64_t, uint64_t>> scheduledPairs;
    for (uint64_t sourceId : sourceFiberIds) {
        const auto sourceIt = byId.find(sourceId);
        for (uint64_t targetId : targetFiberIds) {
            PairJob job;
            if (sourceIt != byId.end() && targetId != sourceId) {
                const auto targetIt = byId.find(targetId);
                if (targetIt != byId.end()) {
                    const FiberPolyline& source = *sourceIt->second;
                    const FiberPolyline& target = *targetIt->second;
                    const uint64_t a = std::min(source.id, target.id);
                    const uint64_t b = std::max(source.id, target.id);
                    if (!(sourceSet.count(target.id) &&
                          targetSet.count(source.id) &&
                          scheduledPairs.count({a, b}))) {
                        job.source = &source;
                        job.target = &target;
                        job.skip = false;
                        scheduledPairs.insert({a, b});
                    }
                }
            }
            pairJobs.push_back(job);
        }
    }

    std::vector<uint64_t> candidateFiberIds;
    candidateFiberIds.reserve(fibers.size());
    std::unordered_map<uint64_t, size_t> candidateIndexByFiberId;
    for (const auto& job : pairJobs) {
        if (job.skip) {
            continue;
        }
        for (const FiberPolyline* fiber : {job.source, job.target}) {
            if (!fiber || candidateIndexByFiberId.count(fiber->id)) {
                continue;
            }
            const size_t index = candidateFiberIds.size();
            candidateIndexByFiberId.emplace(fiber->id, index);
            candidateFiberIds.push_back(fiber->id);
        }
    }

    const size_t searchWorkTotal = pairTotal + candidateFiberIds.size();
    size_t completedSearchWork = 0;
    std::mutex progressMutex;
    auto reportSearchWorkFinished = [&]() {
        std::lock_guard<std::mutex> lock(progressMutex);
        if (completedSearchWork < searchWorkTotal) {
            ++completedSearchWork;
        }
        if (progressCallback) {
            progressCallback(AtlasSearchProgressPhase::SearchPairs,
                             completedSearchWork,
                             searchWorkTotal);
        }
    };
    if (progressCallback) {
        progressCallback(AtlasSearchProgressPhase::SearchPairs, 0, searchWorkTotal);
    }

    std::vector<std::vector<FiberIntersectionCandidate>> candidateResults(candidateFiberIds.size());
    std::atomic_size_t nextCandidate{0};
    std::atomic_bool canceled{false};
    const unsigned hardwareThreads = std::thread::hardware_concurrency();
    const size_t candidateWorkerCount = std::max<size_t>(
        1,
        std::min<size_t>(candidateFiberIds.empty() ? 1 : candidateFiberIds.size(),
                         hardwareThreads == 0 ? 4 : hardwareThreads));

    auto candidateWorker = [&]() {
        for (;;) {
            if (canceled.load(std::memory_order_relaxed)) {
                return;
            }
            const size_t candidateIndex = nextCandidate.fetch_add(1, std::memory_order_relaxed);
            if (candidateIndex >= candidateFiberIds.size()) {
                return;
            }
            if (cancelCallback && cancelCallback()) {
                canceled.store(true, std::memory_order_relaxed);
                return;
            }
            const auto fiberIt = byId.find(candidateFiberIds[candidateIndex]);
            if (fiberIt == byId.end()) {
                continue;
            }
            candidateResults[candidateIndex] =
                index.candidatesForFiber(*fiberIt->second, broad, cancelCallback);
            reportSearchWorkFinished();
            if (cancelCallback && cancelCallback()) {
                canceled.store(true, std::memory_order_relaxed);
                return;
            }
        }
    };

    std::vector<std::thread> candidateWorkers;
    candidateWorkers.reserve(candidateWorkerCount);
    for (size_t i = 0; i < candidateWorkerCount; ++i) {
        candidateWorkers.emplace_back(candidateWorker);
    }
    for (auto& worker : candidateWorkers) {
        worker.join();
    }
    if (canceled.load(std::memory_order_relaxed)) {
        return {};
    }

    auto candidatesFor = [&](uint64_t fiberId) -> const std::vector<FiberIntersectionCandidate>& {
        static const std::vector<FiberIntersectionCandidate> empty;
        const auto it = candidateIndexByFiberId.find(fiberId);
        return it == candidateIndexByFiberId.end() ? empty : candidateResults[it->second];
    };

    std::vector<std::vector<FiberIntersectionResult>> resultsByJob(pairJobs.size());
    std::mutex cacheMutex;
    std::atomic_size_t nextPair{0};
    const size_t pairWorkerCount = std::max<size_t>(
        1,
        std::min<size_t>(pairJobs.empty() ? 1 : pairJobs.size(),
                         hardwareThreads == 0 ? 4 : hardwareThreads));

    auto pairWorker = [&]() {
        for (;;) {
            if (canceled.load(std::memory_order_relaxed)) {
                return;
            }
            const size_t jobIndex = nextPair.fetch_add(1, std::memory_order_relaxed);
            if (jobIndex >= pairJobs.size()) {
                return;
            }
            const PairJob& job = pairJobs[jobIndex];
            if (job.skip) {
                reportSearchWorkFinished();
                continue;
            }
            if (cancelCallback && cancelCallback()) {
                canceled.store(true, std::memory_order_relaxed);
                return;
            }
            const FiberPolyline& source = *job.source;
            const FiberPolyline& target = *job.target;

            std::vector<FiberIntersectionResult> pairResults;
            {
                std::lock_guard<std::mutex> lock(cacheMutex);
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
                    resultsByJob[jobIndex] = std::move(pairResults);
                    reportSearchWorkFinished();
                    continue;
                }
            }

            std::vector<FiberIntersectionCandidate> pairCandidates;
            for (const auto& c : candidatesFor(source.id)) {
                if (cancelCallback && cancelCallback()) {
                    canceled.store(true, std::memory_order_relaxed);
                    return;
                }
                if (c.targetFiberId == target.id) {
                    pairCandidates.push_back(c);
                }
            }
            for (const auto& c : candidatesFor(target.id)) {
                if (cancelCallback && cancelCallback()) {
                    canceled.store(true, std::memory_order_relaxed);
                    return;
                }
                if (c.targetFiberId == source.id) {
                    pairCandidates.push_back(normalizedCandidateForPair(c, source.id, target.id));
                }
            }
            pairCandidates = clusterCandidates(std::move(pairCandidates), broad.clusterArclength);
            if (pairCandidates.empty()) {
                reportSearchWorkFinished();
                continue;
            }

            for (const auto& c : pairCandidates) {
                if (cancelCallback && cancelCallback()) {
                    canceled.store(true, std::memory_order_relaxed);
                    return;
                }
                pairResults.push_back(refineFiberIntersectionCandidate(source,
                                                                       target,
                                                                       c,
                                                                       ceres,
                                                                       windingSampler,
                                                                       cancelCallback));
            }
            if (cancelCallback && cancelCallback()) {
                canceled.store(true, std::memory_order_relaxed);
                return;
            }
            pairResults = deduplicateFiberIntersectionResults(std::move(pairResults),
                                                              ceres.deduplicateArclength);
            {
                std::lock_guard<std::mutex> lock(cacheMutex);
                if (cache) {
                    cache->store(source.id,
                                 source.generation,
                                 target.id,
                                 target.generation,
                                 broad,
                                 ceres,
                                 pairResults);
                }
            }
            resultsByJob[jobIndex] = std::move(pairResults);
            reportSearchWorkFinished();
        }
    };

    std::vector<std::thread> pairWorkers;
    pairWorkers.reserve(pairWorkerCount);
    for (size_t i = 0; i < pairWorkerCount; ++i) {
        pairWorkers.emplace_back(pairWorker);
    }
    for (auto& worker : pairWorkers) {
        worker.join();
    }
    if (canceled.load(std::memory_order_relaxed)) {
        return {};
    }

    std::vector<FiberIntersectionResult> allResults;
    for (const auto& pairResults : resultsByJob) {
        allResults.insert(allResults.end(), pairResults.begin(), pairResults.end());
    }

    return allResults;
}

} // namespace vc::atlas
