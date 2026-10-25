#pragma once
// A surface's correction point collections (VC point collections anchored on
// the surface) in the object store: point_collection objects with record meta
// role "corrections", attached to the surface by "attachment" relationships
// with the same role (data on the surface, never graph links).
#include "ObjectStoreCoordinates.hpp"
#include "utils/Json.hpp"
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <cstdint>
#include <array>
#include <map>
#include <vector>
#include <string>

class Volume;

namespace vc3d::corrections {

// The surface's corrections, as collection JSON by VC collection id, with
// points read through `view` (the surface's) when given. Throws when a
// collection has no mapping to the view's volume.
std::map<uint64_t, utils::Json> read(vc::objects::ObjectStoreSession& session,
                                     const vc::objects::AttachmentIdentity& attachment,
                                     const std::string& surfaceUuid,
                                     const std::optional<vc3d::VolumeView>& view = std::nullopt);

// Make the surface's corrections exactly `collections`: new ones become objects
// bound to `volume` at its voxel scale, changed ones get new geometry (mapped
// back through `view` to each collection's stored frame), and dropped ones are
// retired. One checked action.
void write(vc::objects::ObjectStoreSession& session, const vc::objects::AttachmentIdentity& attachment,
           const std::string& surfaceUuid, const std::map<uint64_t, utils::Json>& collections, const Volume& volume,
           const std::optional<vc3d::VolumeView>& view = std::nullopt);

// A growth that used corrections, kept for training: the surface's geometry
// revisions before and after, the grid region and volume bounds the
// corrections covered, and the correction points in placement order.
struct GrowthAnnotation {
    struct Point {
        uint64_t collectionId = 0, pointId = 0;
        std::string collectionName;
        std::array<double, 3> position{0, 0, 0};
        int64_t creationTime = 0;
    };
    std::string surfaceUuid, beforeGeometry, afterGeometry, volumeUsed;
    std::array<int, 4> gridRegion{0, 0, 0, 0}; // x, y, width, height
    std::array<double, 3> boundsMin{0, 0, 0}, boundsMax{0, 0, 0};
    std::vector<Point> points;
};

// Store it as a point_collection attached to the surface (role
// "growth_annotation"). Returns the new object's UUID.
std::string writeGrowthAnnotation(vc::objects::ObjectStoreSession& session, const vc::objects::AttachmentIdentity& attachment,
                                  const GrowthAnnotation& annotation, const Volume& volume);

} // namespace vc3d::corrections
