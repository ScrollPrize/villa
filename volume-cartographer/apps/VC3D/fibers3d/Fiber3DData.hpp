#pragma once

#include "Fiber3DScene.hpp"
#include <vc_object_store/objects/ObjectStoreSession.hpp>

#include <atomic>

namespace vc3d::line_annotation
{
struct GeneratedViews;
}

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

struct StoreSource {
    std::shared_ptr<vc::objects::ObjectStoreSession> session;
    vc::objects::AttachmentIdentity attachment;
    vc::objects::VolumeVoxels volume;
    double voxelSizeMm = std::numeric_limits<double>::quiet_NaN();
    // When set, load this object only (e.g. the complete selected ribbon).
    std::optional<std::string> onlyUuid;
};

// Worker-only value adapters. They do not access live controllers, mutate
// annotations, create geometry, or resample curves. Cancelled calls throw and
// their result must not be published. Store reads pin one immutable transaction.
Scene loadStore(const StoreSource& source, const QueryOptions& options, const std::atomic_bool* cancelled = nullptr);
QString storeFiberKey(const vc::objects::AttachmentIdentity& attachment, const std::string& uuid);

// Snapshot the live Line Annotation geometry in its displayed CT coordinates.
// Preserve original line indices, span styles and control-point markers.
Fiber generatedAnnotationFiber(const line_annotation::GeneratedViews& views);

}  // namespace vc3d::fibers3d
