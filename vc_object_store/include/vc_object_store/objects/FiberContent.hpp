#pragma once
// The content of a stored fiber that span fingerprints are taken over: its
// control points with their stable ids and annotations, and its line points.
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <optional>
#include <vector>

namespace vc::objects {
class StoreReader;
struct ObjectRevision;

// A span is the stretch of a fiber between two neighbouring control points.
struct FiberSpanFingerprint {
    int64_t idLow = 0, idHigh = 0; // the span's two control-point ids, ordered
    uint64_t hash = 0;
    bool operator==(const FiberSpanFingerprint&) const = default;
};

// Control point i's line index is the position of its coordinates in the line
// points (the ordered membership scan VC3D uses); -1 when the control is not
// on the line.
struct FiberPieceContent {
    std::vector<int64_t> controlIds;
    std::vector<geometry::Point> controlPoints;
    std::vector<nlohmann::json> annotations; // one object per control point
    std::vector<geometry::Point> linePoints;
    std::vector<int64_t> controlLineIndex;

    std::optional<size_t> indexOf(int64_t id) const;
    // Fill controlLineIndex from the coordinates (exact match, in order).
    void resolveLineIndices();
};
FiberPieceContent fiberPieceContent(const StoreReader&, const ObjectRevision&);
FiberPieceContent fiberPieceContent(const geometry::Fiber&, const nlohmann::json& controlAnnotations);

// FNV-1a 64 over canonical little-endian bytes of the span between control
// index `lower` and `lower + 1`: its line points walked from the control with
// the lower id to the one with the higher id (stable under fiber reversal),
// then the lower control's `segment_to_next` annotation.
uint64_t fiberSpanHash(const FiberPieceContent&, size_t lower);
} // namespace vc::objects
