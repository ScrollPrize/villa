#pragma once

// The persisted fiber record and its branch-link entries, shared by the
// LineAnnotationController, the fiber save/validation helpers
// (LineAnnotationFiberLinks.hpp) and the structural-edit planners
// (LineAnnotationStructuralEdits.hpp), so those can be compiled and tested
// without the controller or Qt Widgets. The controller keeps `using` aliases
// under its former nested names.

#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/VolumeTransforms.hpp>
#include <array>
#include <nlohmann/json.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

#include "vc/fiber_tracer/FiberHvClassification.hpp"
#include "LineAnnotationFiberSegments.hpp"
#include "vc/fiber_tracer/FiberDisplay.hpp"
#include <vc_object_store/Uuid.hpp>

namespace vc3d::line_annotation {

// Persisted branch-link metadata. Live branch refs are coupled to
// LineAnnotationSession::controlPoints, reciprocal refs in linked fibers, and
// saved-fiber control-point ordering. Any live mutation of control points or
// branches must go through the private session paths that call
// syncLinkedBranchMetadataAfterFiberModification().
struct FiberBranchRef {
    std::string edgeUuid;
    int controlPointIndex = -1;
    uint64_t branchFiberId = 0;
    int branchControlPointIndex = -1;
    std::string branchFileName;
    cv::Vec3d controlPointDirection{0.0, 0.0, 0.0};
    cv::Vec3d branchControlPointDirection{0.0, 0.0, 0.0};
    cv::Vec3d controlPointPosition{0.0, 0.0, 0.0};
    cv::Vec3d branchControlPointPosition{0.0, 0.0, 0.0};
    // Link awaits reviewer approval; kept in sync on both reciprocal refs.
    bool pending = false;
    // The two control points sit on ADJACENT windings, not the same one:
    // the V fiber's point one winding inside the H fiber's (horizontals
    // lie on the front of the sheet, verticals on the back, so a V fiber
    // showing through to the next wrap out is one sheet thickness from
    // it). Which side is inside follows from the fibers' effective H/V
    // tags; a pair that is not one H and one V (a tag can change, a new
    // fiber has none yet) is not refused here but flagged as an error by
    // the fiber map, and carries no winding constraint there. Immutable
    // for a link (delete and re-link to change), mirrored on both
    // reciprocal refs.
    bool adjacent = false;
    // Created by the Fiber Map's automatic link ("Link A and B") and not yet
    // reviewed: drawn with a dashed marker in the pending colours. Invariant
    // `autoLink => pending && !adjacent`, enforced wherever a ref is read,
    // merged or written; approving the link clears it with `pending`.
    // Mirrored on both reciprocal refs like `pending` (JSON key "auto").
    bool autoLink = false;
    // The peer was read in the open volume, so its position and direction
    // are in the volume's voxels. Links are only made to such fibers.
    bool branchInVolume = true;
};

// A fiber read in an open volume: positions, vectors and widths are in that
// volume's voxels, mapped from the stored frame by the store's transform.
struct FiberVolumeFrame {
    std::string volume;
    vc::objects::VolumeAffine storedToVolume;
};

struct StoredFiber {
    std::optional<vc::objects::PinnedObject> storePin;
    // Set when read in the open volume (bound to it by scale or through a
    // registration); unset for a fiber in its stored frame, which is unbound.
    std::optional<FiberVolumeFrame> frame;
    nlohmann::json coordinateIdentity = nlohmann::json::object();
    double width = 0.0;
    double widthGapFraction = vc::fiber_tracer::kDefaultFiberWidthGapFraction;
    uint64_t id = 0;
    std::string uuid = vc::objects::util::newUuid();
    std::string username;
    std::string startedAt;
    uint64_t sequence = 0;
    std::string fileName;
    uint64_t generation = 1;
    std::vector<StoredControlPoint> controlPoints;
    mutable std::vector<cv::Vec3d> linePoints;
    mutable bool linePointsLoaded = true;
    std::optional<vc::objects::FiberDetail> catalogDetail;
    // Stored snapshots only. Live-session branch metadata must be converted
    // through storedFiberFromSession()/saveSessionAsFiber() so the central
    // hook can remap linked control-point indices before serialization.
    std::vector<FiberBranchRef> branches;
    vc::fiber_tracer::FiberHvClassification hvClassification;
    std::string manualHvTag;
    std::vector<std::string> tags;
    FiberOptimizationMode optimizationMode = FiberOptimizationMode::Lasagna;
    bool modelOnly = false;
    // Inspection of a read-only object-store attachment.
    bool readOnly = false;
    std::string editorError;
};

}  // namespace vc3d::line_annotation
