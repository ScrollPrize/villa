#pragma once

#include "vc/core/render/ChunkCache.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
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
};

// Startup switch; a disabled run bypasses planning and speculative requests.
[[nodiscard]] bool modelPrefetchEnabled();

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
};

// Shared queue-only moving ray. Required reads and direction calculations stay
// with the caller. An optional planning/admission failure disables this window.
class ModelPrefetchWindow {
public:
    explicit ModelPrefetchWindow(std::vector<ModelPrefetchSource> sources,
                                 ModelPrefetchWindowOptions options = {});
    // Returns this call's scheduling counters; planningMs includes admission.
    ModelPrefetchReport advance(const cv::Vec3d& origin, const cv::Vec3d& direction,
                                double remainingDistance,
                                const std::atomic<bool>* cancelled = nullptr) noexcept;
private:
    std::vector<ModelPrefetchSource> sources_;
    ModelPrefetchWindowOptions options_;
    std::unique_ptr<ModelPrefetchPlan> plan_;
    cv::Vec3d plannedOrigin_{};
};

} // namespace vc::lasagna
