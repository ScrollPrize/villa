#pragma once

#include "Fiber3DScene.hpp"
#include "LineAnnotationController.hpp"

#include <atomic>

namespace vc3d::fibers3d
{

struct QueryOptions {
    // In active CT volume coordinates. Absent means the source volume bounds.
    std::optional<Bounds> bounds;
    double minLengthMm = 0.0;
    bool horizontal = true, vertical = true, unclassified = true;
    size_t maximumFibers = 50000;
    size_t byteBudget = size_t(512) * 1024 * 1024;
    QString selectedKey;
};

struct AutomatedSource {
    QString path, uuid;
    double nativeToViewer = 1.0;
    double nativeVoxelMm = std::numeric_limits<double>::quiet_NaN();
    Bounds volumeBounds;
};

// Worker-only value adapters. They do not access live controllers, mutate
// annotations, create geometry, or resample curves. Cancelled calls throw and
// their result must not be published. Each AFV call owns its SQLite connection.
Scene loadAutomated(const AutomatedSource& source, const QueryOptions& options, const std::atomic_bool* cancelled = nullptr);
Scene loadAnnotations(
    const LineAnnotationController::FiberMapSnapshot& snapshot,
    const std::vector<LineAnnotationController::FiberSummary>& summaries,
    const QueryOptions& options,
    const std::atomic_bool* cancelled = nullptr);

}  // namespace vc3d::fibers3d
