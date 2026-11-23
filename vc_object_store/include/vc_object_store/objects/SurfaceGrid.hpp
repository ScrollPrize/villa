#pragma once
// A stored surface as a whole grid in a volume's voxels, read through any
// attachment's reader (local file or hosted store).
#include "vc_object_store/objects/Geometry.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects {
class TransformedGeometryView;
struct SurfaceGrid {
    int64_t rows = 0, columns = 0;
    // Row-major xyz in the volume's voxels; (-1,-1,-1) at invalid vertices.
    std::vector<float> points;
    std::vector<uint8_t> valid;
    std::array<double, 2> scale{1, 1};  // grid scale in the read volume's voxels
    ObjectRecord record;                // the record of the version read
    ObjectVersion reference() const { return record.reference(); }
    // Payload identity of its geometry (its channels read through it).
    std::string geometryRevision;
    std::vector<std::string> channels;  // per-vertex arrays other than points and valid
};

// The surface's current version, or numbered `version` in `generation` (default
// the current owner's lineage), in voxels of `volume` at `targetScale`, through the store's
// mapping to it (its binding or a registration). Throws std::invalid_argument
// when the surface (version) does not exist or does not map to `volume`.
SurfaceGrid readSurfaceGrid(const StoreReader&, const std::string& surfaceUuid, const std::string& volume,
                            double targetScale = 1.0, std::optional<int64_t> version = std::nullopt,
                            const std::optional<std::string>& generation = std::nullopt);
// Exact selection is independent of the object's current owner or visibility.
SurfaceGrid readSurfaceGrid(const StoreReader&, const ObjectVersion&, const std::string& volume,
                            double targetScale = 1.0);
// Materialize an already selected state, without re-resolving current metadata.
// Without a transformed view, return stored coordinates; masked coordinates can
// be retained for editing with applyValidity=false.
SurfaceGrid readSurfaceGrid(const StoreReader&, const ObjectRevision&,
                            const TransformedGeometryView* transformed = nullptr, bool applyValidity = true);
}
