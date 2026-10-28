#pragma once
#include "internal/Spatial.hpp"
#include "spatial/BoundsIndex.hpp"
#include "spatial/SurfaceTiles.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include <memory>
#include <span>

namespace vc::objects::detail {
// An uncommitted surface (ObjectStoreSession drafts) held as the store's
// spatial tiles, each with the bounds a stored tile is indexed by. Immutable
// once shared: an edit makes a new draft sharing the unchanged tiles, so a
// query keeps the draft it started with.
struct SurfaceDraft {
    struct TileData {
        SurfaceTile tile;              // what queries read: points and combined validity
        std::vector<uint8_t> channel;  // the valid channel (1 where the surface has none)
    };
    struct Tile {
        SurfaceTileRect rect;
        std::optional<Bounds> bounds;
        std::shared_ptr<const TileData> data;
    };
    int64_t rows = 0, columns = 0;
    uint64_t revision = 0;  // session-unique, set when it is installed
    CoordinateJson coordinates;
    std::optional<AttachmentIdentity> attachment;
    std::vector<Tile> tiles;
    // The tiles with bounds (`bounded[i]` is the tile of index box i).
    BoundsIndex index;
    std::vector<size_t> bounded;
    // Tiles whose bounds overlap `box`, in row/column order.
    std::vector<size_t> tilesIn(const Bounds& box) const;
};

std::shared_ptr<const SurfaceDraft> makeSurfaceDraft(const vc::geometry::Surface&, const CoordinateJson& coordinates,
                                                     const std::optional<AttachmentIdentity>&);
std::shared_ptr<const SurfaceDraft> editSurfaceDraft(const SurfaceDraft&, int64_t row, int64_t column, int64_t rows,
                                                     int64_t columns, std::span<const vc::geometry::SurfacePoint>,
                                                     std::span<const uint8_t> validity);

// Where a draft lies for a query's coordinate filters: nothing when they
// exclude it, else the mapping of its coordinates to the query's (none: as stored).
struct DraftPlacement { std::optional<VolumeAffine> toTarget; };
std::optional<DraftPlacement> placeDraft(const SurfaceDraft&, const std::optional<std::string>& coordinates,
                                         const std::optional<std::string>& volume, double targetScale,
                                         const std::optional<VolumeAffine>& toTarget, const VolumeTransforms*);

// A draft's runs in a region query, as geometryRegion cuts a stored surface,
// appended to `result` within its output budget, with an empty revision.
// Returns whether the draft crossed the region.
bool draftRegion(const std::string& uuid, const SurfaceDraft&, const DraftPlacement&, const RegionSlice&,
                 const RegionOptions&, const RegionCancel&, RegionResult& result);
// A draft's nearest point, as nearestSurfaces finds a stored surface's.
std::optional<SurfaceProximityHit> draftNearest(const std::string& uuid, const SurfaceDraft&, const DraftPlacement&,
                                                const std::array<double, 3>& point, double tolerance);
}
