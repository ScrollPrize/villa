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
#include <cmath>
#include <nlohmann/json.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

#include "LineAnnotationFiberClassification.hpp"
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
};

// A fiber read in an open volume: positions, vectors and widths are in that
// volume's voxels, mapped from the stored frame by the store's transform.
struct FiberVolumeFrame {
    std::string volume;
    vc::objects::VolumeAffine storedToVolume;
};

// Fiber attributes between frames. Directions follow the affine's linear
// part and normals its inverse transpose, both kept unit length; a uniform
// scale changes neither, so they are kept exactly. Widths scale by the
// uniform factor (the cube root of the determinant's magnitude otherwise).
inline bool uniformScale(const vc::objects::VolumeAffine& m)
{
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            if (r != c && m(r, c) != 0) return false;
    return m(0, 0) > 0 && m(0, 0) == m(1, 1) && m(1, 1) == m(2, 2);
}
inline cv::Vec3d mapFiberDirection(const vc::objects::VolumeAffine& m, const cv::Vec3d& d)
{
    if (uniformScale(m)) return d;
    const auto q = m.apply({d[0], d[1], d[2]}, true);
    const cv::Vec3d v{q[0], q[1], q[2]};
    const double n = std::sqrt(v.dot(v));
    return n > 0 ? v / n : v;
}
inline cv::Vec3d mapFiberNormal(const vc::objects::VolumeAffine& m, const cv::Vec3d& normal)
{
    if (uniformScale(m)) return normal;
    const auto inverse = m.inverse();
    cv::Vec3d v;
    for (int i = 0; i < 3; ++i) v[i] = inverse(0, i) * normal[0] + inverse(1, i) * normal[1] + inverse(2, i) * normal[2];
    const double n = std::sqrt(v.dot(v));
    return n > 0 ? v / n : v;
}
inline double mapFiberWidth(const vc::objects::VolumeAffine& m, double width)
{
    if (uniformScale(m)) return width * m(0, 0);
    const double det = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
                       m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
    return width * std::cbrt(std::abs(det));
}

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
    FiberHvClassification hvClassification;
    std::string manualHvTag;
    std::vector<std::string> tags;
    FiberOptimizationMode optimizationMode = FiberOptimizationMode::Lasagna;
    bool modelOnly = false;
    // Coordinate domain in which control_points and line_points are
    // stored. New Spiral-created fibers record the fiber manifest's L0
    // shape so a downsampled active volume can display them correctly.
    std::optional<std::array<std::size_t, 3>> coordinateBaseShapeZYX;
    // Inspection of a read-only object-store attachment.
    bool readOnly = false;
    std::string editorError;
};

}  // namespace vc3d::line_annotation
