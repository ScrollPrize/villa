#pragma once

// The persisted fiber record and its branch-link entries, shared by the
// LineAnnotationController, the fiber save/validation helpers
// (LineAnnotationFiberLinks.hpp) and the structural-edit planners
// (LineAnnotationStructuralEdits.hpp), so those can be compiled and tested
// without the controller or Qt Widgets. The controller keeps `using` aliases
// under its former nested names.

#include <array>
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

namespace vc3d::line_annotation {

// Persisted branch-link metadata. Live branch refs are coupled to
// LineAnnotationSession::controlPoints, reciprocal refs in linked fibers, and
// saved-fiber control-point ordering. Any live mutation of control points or
// branches must go through the private session paths that call
// syncLinkedBranchMetadataAfterFiberModification().
struct FiberBranchRef {
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
};

struct StoredFiber {
    double width = 0.0;
    double widthGapFraction = vc::fiber_tracer::kDefaultFiberWidthGapFraction;
    uint64_t id = 0;
    std::string username;
    std::string startedAt;
    uint64_t sequence = 0;
    std::string fileName;
    std::filesystem::path sourceRoot;
    uint64_t generation = 1;
    std::vector<StoredControlPoint> controlPoints;
    std::vector<cv::Vec3d> linePoints;
    // Stored snapshots only. Live-session branch metadata must be converted
    // through storedFiberFromSession()/saveSessionAsFiber() so the central
    // hook can remap linked control-point indices before serialization.
    std::vector<FiberBranchRef> branches;
    FiberHvClassification hvClassification;
    std::string manualHvTag;
    std::vector<std::string> tags;
    FiberOptimizationMode optimizationMode = FiberOptimizationMode::Lasagna;
    // Coordinate domain in which control_points and line_points are
    // stored. New Spiral-created fibers record the fiber manifest's L0
    // shape so a downsampled active volume can display them correctly.
    std::optional<std::array<std::size_t, 3>> coordinateBaseShapeZYX;
    bool needsSave = false;
    // The file's write time as of the READ that produced this record
    // (loadFiberFile), so a save decided from that read - the adjacent
    // link heal - can tell a file the sync replaced in the meantime and
    // leave it alone (the next load heals again). Unset for fibers not
    // read from disk.
    std::optional<std::filesystem::file_time_type> loadedWriteTime;
    // Presence at read time, including an explicitly empty array. Only
    // a missing array permits restoring adjacent refs from peers.
    bool adjacentBranchesPresent = true;
    // healOneSidedAdjacentLinks marked this record for saving.
    bool adjacentHealed = false;
    // Load put the gap span tags in step with the break point tags (a
    // version-3 file, or one edited by hand); saved back under the same
    // stale-file guard as the adjacent heal.
    bool gapHealed = false;
    // The load dropped one or more of this record's link entries in memory
    // only (the user kept the files unchanged): the file on disk still
    // holds them. Structural edits consult this before trusting the record
    // as the on-disk truth.
    bool linkEntriesDroppedAtLoad = false;
};

}  // namespace vc3d::line_annotation
