#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/fiber_tracer/FiberReferenceReplayCore.hpp"
#include <future>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>

namespace vc::fiber_tracer {

struct FiberChunkTraceConfig {
    double coreBase=512, haloBase=64, gridBase=4, seedPresence=.8, nmsBase=8;
    double tubeHalfBase=32, tubeStepBase=16, tubePresence=.4;
    double tubeTurnDegrees=15, tubeTotalDegrees=30, rimDrop=.1;
    double rimNormalBase=16, rimTangentBase=32;
    double joinDistanceBase=12, joinAngleDegrees=25, joinMinimumBase=32;
    bool joinWidthWeight=false;
    bool pinSamplingRegion=true;
    // Reject accidentally enormous dense seed grids before allocating them.
    size_t maxGridSamples=64ULL*1024*1024;
};

struct FiberChunkPaths {
    std::array<int,3> key{}; // XYZ core indices; all geometry below is base XYZ
    cv::Vec3d lo{},hi{}; // expanded core + halo, upper bound exclusive
    std::vector<std::vector<cv::Vec3d>> lines;
    std::vector<std::array<std::string,2>> endReasons;
    size_t maxima=0,seeds=0;
};

// Synchronous: on success invoke each index exactly once; join every admitted
// job before returning or throwing. Failure may cancel not-yet-started jobs.
// Never retain the callback. Empty means serial execution.
using FiberChunkParallelFor=std::function<void(size_t,const std::function<void(size_t)>&)>;
// Optional cooperative scheduler wait. Return only when ready() is true, or
// throw on cancellation. Empty uses the shared future's ordinary blocking wait.
using FiberChunkWait=std::function<void(const std::function<bool()>&)>;

[[nodiscard]] FiberTraceOneWayResult traceFiberChunkHalf(const FiberPredictionSource&,
    const cv::Vec3d& seedTrace,const cv::Vec3d& direction,double distanceTrace,
    const FiberTraceConfig&,const vc::lasagna::NormalSampler*,
    const std::array<cv::Vec3d,2>& boundsTrace);

[[nodiscard]] std::shared_ptr<FiberChunkPaths> buildFiberChunkPaths(
    const std::array<int,3>& key,const FiberPredictionSource&,const FiberTraceConfig&,
    const vc::lasagna::NormalSampler&,const FiberChunkTraceConfig&,double traceToBaseScale,
    const FiberChunkParallelFor& parallelFor={});

[[nodiscard]] std::shared_ptr<FiberChunkPaths> loadFiberChunkPaths(const std::filesystem::path&);

// One cache instance/output directory belongs to one immutable prediction and
// normal dataset identity. Source identity is supplied by the caller; geometry
// and scoring settings are checked as well, both in memory and on disk.
class FiberChunkTraceCache {
public:
    FiberChunkTraceCache(FiberChunkTraceConfig,std::filesystem::path output,std::string sourceIdentity);
    [[nodiscard]] const FiberChunkTraceConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::shared_ptr<const FiberChunkPaths> get(const std::array<int,3>&,
        const FiberPredictionSource&,const FiberTraceConfig&,const vc::lasagna::NormalSampler&,
        double traceToBaseScale,const FiberChunkParallelFor& parallelFor={},const FiberChunkWait& wait={});
private:
    FiberChunkTraceConfig config_;
    std::filesystem::path output_;
    std::string identity_;
    nlohmann::json specification_;
    std::mutex mutex_;
    std::map<std::array<int,3>,std::shared_future<std::shared_ptr<const FiberChunkPaths>>> chunks_;
};

struct FiberChunkMatch {
    size_t line=0,segment=0;
    bool reverse=false,valid=false;
    cv::Vec3d point{};
    double arc=0,overlap=0,error=0,angleError=0;
    double widthCompatibility=1,weightedOverlap=0;
};
struct FiberChunkInitialRegion {
    cv::Vec3d center{};
    cv::Matx33d metric=cv::Matx33d::eye(); // (p-center)' metric (p-center)<=1
};
[[nodiscard]] FiberChunkMatch matchFiberChunkPaths(const FiberChunkPaths&,
    const std::vector<cv::Vec3d>& history,const cv::Vec3d& initialDirection,
    const FiberChunkTraceConfig&,bool initial,const FiberChunkInitialRegion* region=nullptr,
    const std::function<std::vector<float>(const std::vector<cv::Vec3d>&)>& widths={},
    double widthTolerance=.4);

// Immutable chunks in base XYZ. The provider must return the requested key and
// remain synchronous. Use a cache.get adapter for on-demand construction.
using FiberChunkProvider=std::function<std::shared_ptr<const FiberChunkPaths>(const std::array<int,3>&)>;
struct FiberChunkRouteJoin {
    std::array<int,3> key{};
    FiberChunkMatch match;
    bool initial=false;
    double spliceDistanceBase=0;
};
using FiberChunkJoinObserver=std::function<void(const FiberChunkRouteJoin&)>;

// Follows actual cached suffixes, not a resampled reference. The initial snap
// and backward matching context are not emitted. The caller owns the seed.
// This extracted route policy requires 16-base-voxel tracing steps.
[[nodiscard]] std::string traceFiberChunkRoute(
    const FiberChunkProvider&,const FiberChunkTraceConfig&,const FiberPredictionSource&,
    const FiberTraceConfig&,double traceToBaseScale,const FiberChunkInitialRegion&,
    const cv::Vec3d& direction,double maximumDistanceBase,int maximumSteps,
    const ReplayPointObserver&,const FiberChunkJoinObserver& joins={});

struct FiberChunkReplayConfig {
    double errorThresholdBase=20, excursionGraceBase=0, tangentialThresholdFactor=4;
    double matchRefineSteps=1;
};
// Uses the shared reference replay protocol. A supplied nextSeed policy is
// preserved and snapped to a cached path. Otherwise search reference positions
// in 16-base-voxel increments for an initial cached path. Other consistency
// callbacks (including reverse checks) are passed through unchanged.
[[nodiscard]] FiberReplayTraceResult traceFiberChunkReferenceReplay(
    FiberChunkTraceCache&,const FiberPredictionSource&,const FiberTraceConfig&,
    double traceToBaseScale,const vc::lasagna::NormalSampler& tracingNormals,
    const PolylineArcGeometry& reference,double startArcBase,double endArcBase,
    const vc::lasagna::NormalSampler& evaluationNormals,double normalWorkingToBaseScale,
    const FiberChunkReplayConfig& = {},const FiberChunkParallelFor& = {},const FiberChunkWait& = {},
    const FiberTraceProgressCallback& = {},const FiberReplayFailureCallback& = {},
    const ReplayConsistencyPolicy& = {},const FiberChunkJoinObserver& = {});

} // namespace vc::fiber_tracer
