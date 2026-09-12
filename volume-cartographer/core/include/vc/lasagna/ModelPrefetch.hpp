#pragma once

#include "vc/core/render/ChunkCache.hpp"

#include <atomic>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::lasagna {

// Own only the shared cache and immutable sampling geometry. In particular,
// never retain LasagnaChannelBinding::group (owned by the dataset).
struct ModelPrefetchSource {
    std::shared_ptr<vc::render::ChunkCache> cache;
    double spacing = 1.0; // source voxels -> caller's input coordinates
    bool remote = false;
};

struct ModelPrefetchOptions {
    double radius = 32.0;
    size_t maxRequests = 1024;
    size_t maxPlanningSteps = 131072;
    size_t maxPlannedBytes = 128ULL * 1024ULL * 1024ULL;
};

struct ModelPrefetchReport {
    size_t planned = 0;
    size_t submitted = 0;
    size_t alreadyQueued = 0;
    size_t alreadyResolved = 0;
    size_t rejected = 0;
    size_t skipped = 0; // permanent omissions, not admission pressure
    size_t plannedBytes = 0;
    bool truncated = false;
    double planningMs = 0.0;
    size_t replans = 0;
    size_t turnRefreshes = 0;
    size_t referencePlans = 0;
    size_t referenceFallbacks = 0;
    size_t curvaturePlans = 0;
};

// Startup switch; a disabled run bypasses planning and speculative requests.
[[nodiscard]] bool modelPrefetchEnabled();

enum class ModelPrefetchProjection { Straight, Guided, Curved };
[[nodiscard]] ModelPrefetchProjection modelPrefetchProjection();
[[nodiscard]] const char* modelPrefetchProjectionName();

// Borrowed only during predictor construction. Scale/orientation affect the
// optional copy, never the caller's reference or optimized geometry.
struct ModelPrefetchReference {
    std::span<const cv::Vec3d> points;
    bool reverse = false;
    double scale = 1.0; // reference coordinates -> caller coordinates
};

// A bounded corridor in each source's own grid. Construction does no I/O;
// pump only submits to the existing cache scheduler, never waits for chunks.
// Cancelling discards unsubmitted work; admitted work retains its global cache
// reservation until completion, even after this plan is destroyed.
class ModelPrefetchPlan {
public:
    ModelPrefetchPlan(std::vector<ModelPrefetchSource> sources,
                      const std::vector<cv::Vec3d>& polyline,
                      ModelPrefetchOptions options = {});

    // True once all requests have been submitted/skipped (not downloaded).
    // Call again later after an admission rejection. At most 128 keys are
    // considered per call, so a large resident corridor is also bounded work.
    [[nodiscard]] bool pump(const std::atomic<bool>* cancelled = nullptr);
    [[nodiscard]] const ModelPrefetchReport& report() const { return report_; }

private:
    struct Request {
        std::shared_ptr<vc::render::ChunkCache> cache;
        vc::render::ChunkKey key;
    };
    std::vector<Request> requests_;
    size_t next_ = 0;
    ModelPrefetchReport report_;
};

// All distances, including source spacing, are in the caller's coordinates.
// The native tracer uses trace voxels; the line optimizer uses base voxels.
struct ModelPrefetchWindowOptions {
    double lookahead = 256.0;
    double refreshDistance = 64.0;
    ModelPrefetchOptions corridor{16.0, 128, 4096, 16ULL * 1024ULL * 1024ULL};
    ModelPrefetchProjection projection = ModelPrefetchProjection::Straight;
};

struct ModelPrefetchPrediction {
    std::vector<cv::Vec3d> points;
    bool turnRefresh = false;
    bool reference = false;
    bool referenceFallback = false;
    bool curved = false;
};

// No model reads, cache calls or solver changes. Independent bounds: inspect
// and retain <=2048 reference points, search/emit <=128 segments per refresh,
// retain <=8 spatial observations, and emit 8 segments for curved fallback.
class ModelPrefetchPredictor {
public:
    explicit ModelPrefetchPredictor(ModelPrefetchWindowOptions options,
                                    ModelPrefetchReference reference = {});
    [[nodiscard]] std::optional<ModelPrefetchPrediction> update(
        const cv::Vec3d& origin, const cv::Vec3d& direction, double remainingDistance);
private:
    void observe(const cv::Vec3d& origin, const cv::Vec3d& direction);
    bool followReference(const cv::Vec3d& origin, const cv::Vec3d& direction,
                         double ahead, std::vector<cv::Vec3d>& points);
    bool extrapolateCurve(const cv::Vec3d& origin, const cv::Vec3d& direction,
                          double ahead, std::vector<cv::Vec3d>& points) const;
    ModelPrefetchWindowOptions options_;
    std::vector<cv::Vec3d> reference_;
    std::vector<double> arcs_;
    size_t cursor_ = 0;
    double progress_ = 0.0;
    cv::Vec3d matchedOrigin_{};
    struct Observation { cv::Vec3d origin, direction; };
    std::array<Observation, 8> history_{};
    size_t historySize_ = 0;
    bool planned_ = false;
    bool plannedReference_ = false;
    cv::Vec3d plannedOrigin_{}, plannedDirection_{};
};

// Shared queue-only moving corridor. Required reads and direction calculations stay
// with the caller. An optional planning/admission failure disables this window.
class ModelPrefetchWindow {
public:
    explicit ModelPrefetchWindow(std::vector<ModelPrefetchSource> sources,
                                 ModelPrefetchWindowOptions options = {},
                                 ModelPrefetchReference reference = {});
    // Returns this call's scheduling counters; planningMs includes admission.
    ModelPrefetchReport advance(const cv::Vec3d& origin, const cv::Vec3d& direction,
                                double remainingDistance,
                                const std::atomic<bool>* cancelled = nullptr) noexcept;
private:
    std::vector<ModelPrefetchSource> sources_;
    ModelPrefetchWindowOptions options_;
    std::unique_ptr<ModelPrefetchPlan> plan_;
    ModelPrefetchPredictor predictor_;
};

} // namespace vc::lasagna
