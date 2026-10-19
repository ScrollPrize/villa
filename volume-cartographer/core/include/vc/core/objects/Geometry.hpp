#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects {

// Contiguous C-order, little-endian arrays. Supported dtypes: |u1, <i4,
// <i8, <u8, <f4, <f8. Shape and byte length are checked before publication.
struct GeometryArray {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

struct GeometryData {
    std::string kind; // patch, fiber, point_collection, track
    int64_t formatVersion = 1;
    std::string coordinateSystem = "source_xyz";
    std::string attributes = "{}"; // immutable JSON object; no geometry coordinates
    std::vector<GeometryArray> arrays;
};

struct GeometryArrayInfo {
    std::string name, dtype;
    std::vector<int64_t> shape;
    int64_t byteLength = 0;
};

// Applies to new revisions only. Unchanged geometry retains its revision and
// layout. Zero selects legacy linear storage; other arrays remain linear.
struct GeometryStorageOptions {
    int64_t patchTileSize = 128; // supported sizes: 0, 32, 64, 128 vertices
};

// Descriptors do not load payloads. Revision UUIDs are independent of object
// UUIDs and content hashes. Parentage records edits, including undo/reverts.
struct GeometryRevision {
    std::string uuid, objectUuid;
    std::optional<std::string> parentUuid;
    std::string contentHash, kind, coordinateSystem, attributes;
    int64_t formatVersion = 1;
    int64_t objectGeneration = 0;
    std::string createdBy, createdAt, createdHost;
    std::vector<GeometryArrayInfo> arrays;
};

} // namespace vc::objects
