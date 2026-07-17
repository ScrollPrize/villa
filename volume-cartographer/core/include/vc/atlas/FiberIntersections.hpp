#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace vc::lasagna {
class LasagnaNormalSampler;
}

namespace vc::atlas {

struct FiberPoint {
    cv::Vec3d position{0.0, 0.0, 0.0};
    std::optional<cv::Vec3d> normal;
};

struct FiberPolyline {
    uint64_t id = 0;
    uint64_t generation = 1;
    std::vector<FiberPoint> points;
    std::vector<cv::Vec3d> controlPoints;
};

struct FiberSegmentEntry {
    uint64_t fiberId = 0;
    uint64_t generation = 1;
    int segmentIndex = -1;
    cv::Vec3d a{0.0, 0.0, 0.0};
    cv::Vec3d b{0.0, 0.0, 0.0};
    double arclength0 = 0.0;
    double arclength1 = 0.0;
    cv::Vec3d aabbMin{0.0, 0.0, 0.0};
    cv::Vec3d aabbMax{0.0, 0.0, 0.0};
};

struct FiberIntersectionBroadPhaseOptions {
    double maxDistance = 500.0;
    double maxSampleSpacing = 100.0;
    int seedStride = 100;
    double clusterArclength = 8.0;

    friend bool operator==(const FiberIntersectionBroadPhaseOptions& a,
                           const FiberIntersectionBroadPhaseOptions& b)
    {
        return a.maxDistance == b.maxDistance &&
               a.maxSampleSpacing == b.maxSampleSpacing &&
               a.seedStride == b.seedStride &&
               a.clusterArclength == b.clusterArclength;
    }
};

struct FiberIntersectionCeresOptions {
    int maxIterations = 50;
    double distanceWeight = 1.0;
    double normalOrthogonalityWeight = 0.25;
    double deduplicateArclength = 4.0;

    friend bool operator==(const FiberIntersectionCeresOptions& a,
                           const FiberIntersectionCeresOptions& b)
    {
        return a.maxIterations == b.maxIterations &&
               a.distanceWeight == b.distanceWeight &&
               a.normalOrthogonalityWeight == b.normalOrthogonalityWeight &&
               a.deduplicateArclength == b.deduplicateArclength;
    }
};

struct FiberIntersectionCandidate {
    uint64_t sourceFiberId = 0;
    uint64_t sourceGeneration = 1;
    int sourceSegmentIndex = -1;
    double sourceArclength = 0.0;
    uint64_t targetFiberId = 0;
    uint64_t targetGeneration = 1;
    int targetSegmentIndex = -1;
    double targetArclength = 0.0;
    double straightDistance = 0.0;
};

struct FiberIntersectionResult {
    uint64_t sourceFiberId = 0;
    uint64_t sourceGeneration = 1;
    uint64_t targetFiberId = 0;
    uint64_t targetGeneration = 1;
    double candidateDistance = 0.0;
    double refinedScore = 0.0;
    double windingDistance = std::numeric_limits<double>::infinity();
    double sourceArclength = 0.0;
    double targetArclength = 0.0;
    cv::Vec3d sourcePoint{0.0, 0.0, 0.0};
    cv::Vec3d targetPoint{0.0, 0.0, 0.0};
    bool converged = false;
    bool cacheHit = false;
    int ceresSolves = 0;
    int ceresIterations = 0;
    bool usedNormalResiduals = false;
    std::string message;
};

enum class FiberSideStripIntersectionSource {
    BranchLink,
    FiberSegment,
};

enum class FiberSideStripProgressPhase {
    BuildStripTriangles,
    BuildTriangleIndex,
    BranchLinks,
    FiberSegments,
    Deduplicate,
};

struct FiberSideStripLineQuery {
    uint64_t fiberId = 0;
    cv::Vec3d point{0.0, 0.0, 0.0};
    cv::Vec3d direction{0.0, 0.0, 1.0};
    cv::Vec3d connectorStart{std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN()};
};

struct FiberSideStripQueryOptions {
    cv::Mat_<cv::Vec3f> stripPoints;
    double deduplicateStripDistance = 1.0e-3;
    double aabbPadding = 1.0e-6;
    size_t maxResults = 64;
    size_t workerThreads = 0;
    std::vector<uint64_t> excludedFiberIds;
    std::vector<const FiberPolyline*> queryFibers;
    std::vector<FiberSideStripLineQuery> branchLinks;
};

struct FiberSideStripIntersection {
    uint64_t fiberId = 0;
    uint64_t generation = 1;
    int segmentIndex = -1;
    double arclength = 0.0;
    cv::Vec3d point{0.0, 0.0, 0.0};
    double stripRow = std::numeric_limits<double>::quiet_NaN();
    double stripCol = std::numeric_limits<double>::quiet_NaN();
    double distance = 0.0;
    FiberSideStripIntersectionSource source =
        FiberSideStripIntersectionSource::FiberSegment;
    cv::Vec3d connectorStart{std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN()};
    cv::Vec3d projectionTarget{std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::quiet_NaN()};
    size_t branchLinkIndex = std::numeric_limits<size_t>::max();
};

enum class AtlasSearchProgressPhase {
    PrepareInputs = 0,
    BuildSpatialIndex = 1,
    SearchPairs = 2,
    PrepareSigningSurface = 3,
    FinishResults = 4,
};

[[nodiscard]] int atlasSearchPhaseProgressPercent(AtlasSearchProgressPhase phase,
                                                  size_t completed,
                                                  size_t total);

using FiberIntersectionProgressCallback =
    std::function<void(AtlasSearchProgressPhase phase, size_t completed, size_t total)>;
using FiberIntersectionCancelCallback = std::function<bool()>;
using FiberSideStripProgressCallback =
    std::function<void(FiberSideStripProgressPhase phase, size_t completed, size_t total)>;

class FiberSpatialIndex {
public:
    void clear();
    void upsertCommitted(const FiberPolyline& fiber);
    void upsertRecent(const FiberPolyline& fiber);
    void removeFiber(uint64_t fiberId);

    [[nodiscard]] uint64_t generation(uint64_t fiberId) const;
    [[nodiscard]] std::vector<FiberIntersectionCandidate> candidatesForFiber(
        const FiberPolyline& source,
        const FiberIntersectionBroadPhaseOptions& options,
        FiberIntersectionCancelCallback cancelCallback = {}) const;
    [[nodiscard]] std::vector<FiberSideStripIntersection> sideStripIntersections(
        const FiberSideStripQueryOptions& options,
        FiberSideStripProgressCallback progressCallback = {},
        FiberIntersectionCancelCallback cancelCallback = {}) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

class FiberIntersectionCache {
public:
    [[nodiscard]] bool lookup(uint64_t fiberA,
                              uint64_t generationA,
                              uint64_t fiberB,
                              uint64_t generationB,
                              const FiberIntersectionBroadPhaseOptions& broad,
                              const FiberIntersectionCeresOptions& ceres,
                              std::vector<FiberIntersectionResult>& results) const;
    void store(uint64_t fiberA,
               uint64_t generationA,
               uint64_t fiberB,
               uint64_t generationB,
               const FiberIntersectionBroadPhaseOptions& broad,
               const FiberIntersectionCeresOptions& ceres,
               std::vector<FiberIntersectionResult> results);
    void pruneFiber(uint64_t fiberId);
    void clear();
    [[nodiscard]] size_t size() const;

private:
    struct Key {
        uint64_t fiberA = 0;
        uint64_t generationA = 0;
        uint64_t fiberB = 0;
        uint64_t generationB = 0;
        FiberIntersectionBroadPhaseOptions broad;
        FiberIntersectionCeresOptions ceres;

        friend bool operator==(const Key& a, const Key& b)
        {
            return a.fiberA == b.fiberA &&
                   a.generationA == b.generationA &&
                   a.fiberB == b.fiberB &&
                   a.generationB == b.generationB &&
                   a.broad == b.broad &&
                   a.ceres == b.ceres;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& key) const
        {
            size_t seed = 0;
            auto combine = [&seed](size_t value) {
                seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            };
            combine(std::hash<uint64_t>{}(key.fiberA));
            combine(std::hash<uint64_t>{}(key.generationA));
            combine(std::hash<uint64_t>{}(key.fiberB));
            combine(std::hash<uint64_t>{}(key.generationB));
            combine(std::hash<double>{}(key.broad.maxDistance));
            combine(std::hash<double>{}(key.broad.maxSampleSpacing));
            combine(std::hash<int>{}(key.broad.seedStride));
            combine(std::hash<double>{}(key.broad.clusterArclength));
            combine(std::hash<int>{}(key.ceres.maxIterations));
            combine(std::hash<double>{}(key.ceres.distanceWeight));
            combine(std::hash<double>{}(key.ceres.normalOrthogonalityWeight));
            combine(std::hash<double>{}(key.ceres.deduplicateArclength));
            return seed;
        }
    };
    std::unordered_map<Key, std::vector<FiberIntersectionResult>, KeyHash> entries_;
};

[[nodiscard]] std::vector<FiberSegmentEntry> fiberSegments(const FiberPolyline& fiber);

[[nodiscard]] FiberIntersectionResult refineFiberIntersectionCandidate(
    const FiberPolyline& source,
    const FiberPolyline& target,
    const FiberIntersectionCandidate& candidate,
    const FiberIntersectionCeresOptions& options,
    const vc::lasagna::LasagnaNormalSampler* windingSampler = nullptr,
    FiberIntersectionCancelCallback cancelCallback = {});

[[nodiscard]] std::vector<FiberIntersectionResult> deduplicateFiberIntersectionResults(
    std::vector<FiberIntersectionResult> results,
    double arclengthTolerance);

[[nodiscard]] std::optional<size_t> nearestIntersectionResultByArclength(
    const std::vector<FiberIntersectionResult>& results,
    double sourceArclength,
    double targetArclength);

[[nodiscard]] std::vector<FiberIntersectionResult> searchFiberIntersections(
    const std::vector<FiberPolyline>& fibers,
    const std::vector<uint64_t>& sourceFiberIds,
    const std::vector<uint64_t>& targetFiberIds,
    FiberIntersectionCache* cache,
    const FiberIntersectionBroadPhaseOptions& broad,
    const FiberIntersectionCeresOptions& ceres,
    const vc::lasagna::LasagnaNormalSampler* windingSampler = nullptr,
    FiberIntersectionProgressCallback progressCallback = {},
    FiberIntersectionCancelCallback cancelCallback = {});

} // namespace vc::atlas
