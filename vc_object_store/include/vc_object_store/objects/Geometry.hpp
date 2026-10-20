#pragma once

#include <cstdint>
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects {

// Contiguous C-order, little-endian arrays. Supported dtypes: |u1, <u2, <i4,
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
// layout. Zero selects legacy linear storage; all aligned patch channels use the same tile size.
struct GeometryStorageOptions {
    int64_t patchTileSize = 128; // supported sizes: 0, 32, 64, 128 vertices
};

// Replace a contiguous range of scalar elements in an existing array.
// elementOffset indexes the flattened C-order array (XYZ is three elements).
// Bytes use the array's existing little-endian dtype. Shape/dtype/attributes
// remain unchanged. Edits are applied in order; overlapping writes use the last
// value. The store captures original bytes and removes net no-ops.
struct GeometryArrayEdit {
    std::string array;
    int64_t elementOffset = 0;
    std::vector<uint8_t> bytes;
};

// A structural sequence is assembled from immutable slices and/or new rows.
// A reference copies identities and annotations as well as coordinates. New
// point_collection/control_points rows require explicit nonnegative stable IDs.
struct GeometrySequencePart {
    std::optional<std::string> revision;
    int64_t start = 0, count = 0;
    bool reverse = false;
    std::vector<uint8_t> bytes;
    std::vector<int64_t> ids;
    std::string annotations = "[]"; // one JSON object/null per new row, or []
};
struct GeometrySequenceEdit {
    std::string array;
    int64_t start = 0, eraseCount = 0;
    std::vector<GeometrySequencePart> parts;
    // Required when any existing/new annotation has span/direction semantics.
    // Complete resulting positional annotations, transformed by the caller.
    std::optional<std::string> transformedAnnotations;
};

// New logical (r,c) maps to old (r+row,c+column). Negative offsets grow
// top/left. Every array requires one explicitly typed fill pixel, even on crop.
struct GeometrySurfaceResize {
    int64_t row = 0, column = 0, rows = 0, columns = 0;
    std::vector<GeometryArray> fill; // shape {1,1[,components]}, matching dtype
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
    std::array<int64_t, 2> gridOrigin{0, 0}; // stable surface grid coordinates
};

} // namespace vc::objects
