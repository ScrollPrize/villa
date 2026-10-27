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
    // Replay-only options; never alter cached path construction.
    int initialAlignmentPower=0; // 0 nearest, 1 cosine, 2 cosine squared
    double initialNormalRadiusBase=0, initialTangentialRadiusBase=0;
    bool jointJoinScore=false;
    std::array<double,4> joinWeights{1,1,1,1}; // overlap,width,distance,direction
    double conflictNormalBase=0, conflictTangentBase=0; // zero disables
    double conflictAgreementBase=64, conflictSeparationBase=32;
    bool cacheOnly=false; // never construct a missing chunk
    bool presencePeakRestarts=false; // historical aligned-peak seed policy, including no-path stops
    // Construction/backend options are part of construction cache identity.
    bool refineSeed=false;
    double seedMovementBase=0;
    std::string gpuLibrary; // optional CUDA plugin; CPU default
    int gpuDevices=0; // visible device count, caller must hold manager grants
    int onlineRefineIterations=0, onlineRefineWindow=0;
    std::string onlineRefineMode="legacy"; // explicit comparison control
    std::string beamCommitMode="shared_prefix"; // experimental GPU rolling horizon
    int gpuBatchHalves=8192;
    int gpuBatchMemoryMiB=2048; // workspace cap, excludes resident fields
    double onlineRefineEpsilonBase=.25, onlineRefineStepBase=1, onlineRefineCapBase=8;
    std::filesystem::path benchmarkExport; // explicit fixture export, empty disables
    bool pinSamplingRegion=true;
    bool channelHints=true; // generic cache hint; no change to construction identity
    std::size_t workerCacheBytes=0; // 0: shared; otherwise experimental per-worker decoded LRU
    // Reject accidentally enormous dense seed grids before allocating them.
    size_t maxGridSamples=64ULL*1024*1024;
};

[[nodiscard]] nlohmann::json fiberChunkConfigJson(const FiberChunkTraceConfig&);
[[nodiscard]] FiberChunkTraceConfig fiberChunkConfigFromJson(const nlohmann::json&);
[[nodiscard]] nlohmann::json fiberChunkCacheSpecification(const FiberChunkTraceConfig&,
    const FiberTraceConfig&,double traceToBaseScale,const std::string& sourceIdentity);

struct FiberChunkPaths {
    std::array<int,3> key{}; // XYZ core indices; all geometry below is base XYZ
    cv::Vec3d lo{},hi{}; // expanded core + halo, upper bound exclusive
    std::vector<std::vector<cv::Vec3d>> lines;
    std::vector<std::array<std::string,2>> endReasons;
    size_t maxima=0,seeds=0;
    nlohmann::json timings=nlohmann::json::object();
    // Empty annotation means unchecked. Geometry is always retained on disk.
    nlohmann::json conflictAnnotation=nullptr;
    std::vector<bool> conflictFlags;
};

// Synchronous: on success invoke each index exactly once; join every admitted
// job before returning or throwing. Failure may cancel not-yet-started jobs.
// Never retain the callback. Empty means serial execution.
using FiberChunkParallelFor=std::function<void(size_t,const std::function<void(size_t)>&)>;
// Optional cooperative scheduler wait. Return only when ready() is true, or
// throw on cancellation. Empty uses the shared future's ordinary blocking wait.
using FiberChunkWait=std::function<void(const std::function<bool()>&)>;

struct FiberChunkSeedFit {
    cv::Vec3d point{}, direction{};
    double width=0, score=-1;
};
[[nodiscard]] FiberChunkSeedFit refineFiberChunkSeed(const FiberPredictionSource&,
    const cv::Vec3d&,const cv::Vec3d&,const FiberChunkPaths&,
    const FiberChunkTraceConfig&,double traceToBaseScale);

// Clears conflicting paths on the supplied copy; preserves path indices and
// endpoint reasons. Never overwrites construction-cache payloads. Normals use
// working coordinates, all geometry/thresholds are base voxels.
[[nodiscard]] nlohmann::json filterFiberChunkConflicts(FiberChunkPaths&,
    const FiberChunkTraceConfig&,const vc::lasagna::NormalSampler&,double traceToBaseScale,
    const FiberChunkParallelFor& parallelFor={},const FiberPredictionSource* readinessSource=nullptr,
    const std::vector<size_t>* auditQueries=nullptr);

// Non-destructive variant; identical decisions to filterFiberChunkConflicts.
[[nodiscard]] nlohmann::json inspectFiberChunkConflicts(const FiberChunkPaths&,
    const FiberChunkTraceConfig&,const vc::lasagna::NormalSampler&,double,
    const FiberChunkParallelFor& = {},const FiberPredictionSource* = nullptr,
    const std::vector<size_t>* = nullptr);

[[nodiscard]] FiberTraceOneWayResult traceFiberChunkHalf(const FiberPredictionSource&,
    const cv::Vec3d& seedTrace,const cv::Vec3d& direction,double distanceTrace,
    const FiberTraceConfig&,const vc::lasagna::NormalSampler*,
    const std::array<cv::Vec3d,2>& boundsTrace);

[[nodiscard]] std::shared_ptr<FiberChunkPaths> buildFiberChunkPaths(
    const std::array<int,3>& key,const FiberPredictionSource&,const FiberTraceConfig&,
    const vc::lasagna::NormalSampler&,const FiberChunkTraceConfig&,double traceToBaseScale,
    const FiberChunkParallelFor& parallelFor={},const FiberChunkWait& wait={});

[[nodiscard]] std::shared_ptr<FiberChunkPaths> loadFiberChunkPaths(const std::filesystem::path&);

// Load/validate/annotate or build/annotate/publish exactly one chunk. Caller owns
// unique-key scheduling and the output-directory writer lock. No RAM retention.
// outcome is "built", "annotated", or "reused" (no output rewrite).
// readOnly rejects missing geometry or stale annotations without any writes.
[[nodiscard]] std::shared_ptr<FiberChunkPaths> prepareFiberChunkPaths(
    const std::array<int,3>&,const FiberPredictionSource&,const FiberTraceConfig&,
    const vc::lasagna::NormalSampler&,const FiberChunkTraceConfig&,double,
    const std::filesystem::path&,const std::string& sourceIdentity,std::string& outcome,
    const FiberChunkParallelFor& = {},const FiberChunkWait& = {},bool readOnly=false);

// Shared validation for matching and explicit-path route starts.
void validateFiberChunkConflictAnnotation(const FiberChunkPaths&,const FiberChunkTraceConfig&);

// One cache instance/output directory belongs to one immutable prediction and
// normal dataset identity. Source identity is supplied by the caller; geometry
// and scoring settings are checked as well, both in memory and on disk.
class FiberChunkTraceCache {
public:
    FiberChunkTraceCache(FiberChunkTraceConfig,std::filesystem::path output,std::string sourceIdentity);
    [[nodiscard]] const FiberChunkTraceConfig& config() const noexcept { return config_; }
    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return output_; }
    [[nodiscard]] const std::string& sourceIdentity() const noexcept { return identity_; }
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
    double jointScore=0;
    double initialScore=0;
    int initialSamples=0;
    bool ambiguous=false;
    double overlapBeginArc=0; // exact target arc reached by accepted backward correspondence
    std::vector<std::pair<double,double>> overlapArcs; // history distance from end, target arc
};
struct FiberChunkInitialRegion {
    cv::Vec3d center{};
    cv::Matx33d metric=cv::Matx33d::eye(); // (p-center)' metric (p-center)<=1
};
// All qualified orientations, before ranking or ambiguity suppression. Shared by
// runtime matching and the complete-inventory continuation prefilter.
enum class FiberChunkMatchPurpose { Continuation, TerminalRelation, FullEquivalence };
[[nodiscard]] std::vector<FiberChunkMatch> findFiberChunkMatches(const FiberChunkPaths&,
    const std::vector<cv::Vec3d>& history,const cv::Vec3d& initialDirection,
    const FiberChunkTraceConfig&,bool initial,const FiberChunkInitialRegion* region=nullptr,
    const std::function<std::vector<float>(const std::vector<cv::Vec3d>&)>& widths={},
    double widthTolerance=.4,const std::vector<size_t>* candidateLines=nullptr,
    FiberChunkMatchPurpose purpose=FiberChunkMatchPurpose::Continuation);

struct FiberChunkPathReduction {
    // Original path ID -> representative; -1 pre-existing inactive, -2 ambiguity.
    std::vector<int64_t> representative;
    std::vector<std::array<std::string,2>> endReasons;
    size_t eligible=0,terminalPairs=0,equivalentPairs=0,fusedPaths=0,ambiguousPaths=0;
    bool rejectAmbiguousPaths=true;
    size_t retainedAmbiguousPaths=0;
    [[nodiscard]] nlohmann::json json() const;
    void apply(FiberChunkPaths&) const;
};
[[nodiscard]] FiberChunkPathReduction reduceFiberChunkPaths(const FiberChunkPaths&,
    const FiberChunkTraceConfig&,
    const std::function<std::vector<float>(const std::vector<cv::Vec3d>&)>& widths={},
    double widthTolerance=.4,bool rejectAmbiguousPaths=true);
[[nodiscard]] FiberChunkMatch matchFiberChunkPaths(const FiberChunkPaths&,
    const std::vector<cv::Vec3d>& history,const cv::Vec3d& initialDirection,
    const FiberChunkTraceConfig&,bool initial,const FiberChunkInitialRegion* region=nullptr,
    const std::function<std::vector<float>(const std::vector<cv::Vec3d>&)>& widths={},
    double widthTolerance=.4,const vc::lasagna::NormalSampler* normals=nullptr,double normalScale=1);

// Immutable chunks in base XYZ. The provider must return the requested key and
// remain synchronous. Use a cache.get adapter for on-demand construction.
using FiberChunkProvider=std::function<std::shared_ptr<const FiberChunkPaths>(const std::array<int,3>&)>;
struct FiberChunkRouteJoin {
    std::array<int,3> key{};
    FiberChunkMatch match;
    bool initial=false;
    double spliceDistanceBase=0;
    double representedOverlapBeginArc=0;
};
using FiberChunkJoinObserver=std::function<void(const FiberChunkRouteJoin&)>;
using FiberChunkContinuationGate=std::function<std::string(const FiberChunkRouteJoin&,const FiberChunkRouteJoin&)>;
std::vector<float> sampleFiberChunkWidths(const FiberPredictionSource&,const std::vector<cv::Vec3d>&,double scale);
std::string checkFiberChunkSplice(const FiberPredictionSource&,const FiberTraceConfig&,double scale,
    const cv::Vec3d& previous,const cv::Vec3d& point,const cv::Vec3d& direction);
// A distance cap to the next already-owned arc, then commit only emitted arcs.
// Arcs use the matched orientation; callbacks run after successful splice checks.
struct FiberChunkRouteOwnership {
    std::function<double(const FiberChunkRouteJoin&)> available;
    std::function<void(const FiberChunkRouteJoin&,double,double)> consume;
    std::function<void(const FiberChunkRouteJoin&,double,double)> alias;
};

// Explicit seed on an already selected cached path; no nearest-path reselection.
struct FiberChunkRouteStart {
    size_t line=0,vertex=0;
    bool reverse=false;
    double arc=-1; // optional original-orientation arc, allows uncovered subpieces
};

// Follows actual cached suffixes, not a resampled reference. The initial snap
// and backward matching context are not emitted. The caller owns the seed.
// This extracted route policy requires 16-base-voxel tracing steps.
[[nodiscard]] std::string traceFiberChunkRoute(
    const FiberChunkProvider&,const FiberChunkTraceConfig&,const FiberPredictionSource&,
    const FiberTraceConfig&,double traceToBaseScale,const FiberChunkInitialRegion&,
    const cv::Vec3d& direction,double maximumDistanceBase,int maximumSteps,
    const ReplayPointObserver&,const FiberChunkJoinObserver& joins={},
    const FiberChunkRouteStart* start=nullptr,const vc::lasagna::NormalSampler* normals=nullptr,
    const FiberChunkRouteOwnership* ownership=nullptr,const FiberChunkContinuationGate& continuation={});

[[nodiscard]] std::array<int,3> fiberChunkKeyAt(const cv::Vec3d&,double coreBase);
struct FiberChunkNext {std::array<int,3> key{};std::string stop;};
[[nodiscard]] FiberChunkNext nextFiberChunk(const FiberChunkPaths&,const cv::Vec3d& end,
    const cv::Vec3d& previous,const std::string& reason,const FiberChunkTraceConfig&);

struct FiberChunkReplayConfig {
    double errorThresholdBase=20, excursionGraceBase=0, tangentialThresholdFactor=4;
    double matchRefineSteps=1;
};
[[nodiscard]] nlohmann::json fiberChunkRouteMatchingSpecification(const FiberChunkTraceConfig&);
// Read-only, complete export decisions, validated against source/configuration.
// Does not import globally owned intervals: independent reference replays start
// with fresh ownership so results cannot depend on benchmark ordering.
class FiberChunkRoutingRules {
    struct Impl;
    std::shared_ptr<const Impl> impl_;
public:
    FiberChunkRoutingRules(const std::filesystem::path& exportDirectory,
        const std::filesystem::path& chunkDirectory,const FiberChunkTraceConfig&,
        const FiberTraceConfig&,double scale,const std::string& identity);
    [[nodiscard]] std::shared_ptr<const FiberChunkPaths> apply(std::shared_ptr<const FiberChunkPaths>) const;
    [[nodiscard]] std::string continuation(const FiberChunkRouteJoin&,const FiberChunkRouteJoin&) const;
    void validate(const FiberChunkTraceCache&,const FiberTraceConfig&,double scale) const;
};
// Immutable exported routes. Replay follows one stored path to its stored end;
// it never reconstructs joins or consults future reference geometry to select it.
class FiberStoredRouteSet {
    struct Impl;
    std::shared_ptr<const Impl> impl_;
public:
    FiberStoredRouteSet(std::vector<std::vector<cv::Vec3d>>,
        std::vector<std::array<std::string,2>> endReasons);
    [[nodiscard]] FiberReplayTraceResult replay(const FiberPredictionSource&,const FiberTraceConfig&,
        double traceToBaseScale,const PolylineArcGeometry& reference,
        const vc::lasagna::NormalSampler& evaluationNormals,double normalWorkingToBaseScale,
        const FiberChunkTraceConfig&,const FiberChunkReplayConfig&) const;
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
    const ReplayConsistencyPolicy& = {},const FiberChunkJoinObserver& = {},
    const FiberChunkRoutingRules* = nullptr);

} // namespace vc::fiber_tracer
