#pragma once
#include "vc/core/objects/ObjectStore.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include <functional>

namespace vc::objects::detail {
// A bounded reader for either a published revision or a proposed payload.
struct DependencyGeometry {
    std::string kind, attributes;
    std::map<std::string, GeometryArrayInfo> arrays;
    std::function<std::vector<uint8_t>(const std::string&, int64_t, int64_t)> read;
};

inline DependencyGeometry dependencyGeometry(const ObjectStore& db, const Txn& txn, const GeometryRevision& revision)
{
    DependencyGeometry result{revision.kind, revision.attributes, {}, {}};
    for (const auto& array : revision.arrays) result.arrays.emplace(array.name, array);
    result.read = [&db, &txn, uuid = revision.uuid](const auto& name, auto offset, auto count) {
        return db.readGeometryArray(txn, uuid, name, offset, count);
    };
    return result;
}

inline DependencyGeometry dependencyGeometry(const GeometryData& data)
{
    DependencyGeometry result{data.kind, data.attributes, {}, {}};
    for (const auto& array : data.arrays)
        result.arrays.emplace(array.name, GeometryArrayInfo{array.name, array.dtype, array.shape, static_cast<int64_t>(array.bytes.size())});
    result.read = [&data](const auto& name, int64_t offset, int64_t count) {
        const auto& array = *std::find_if(data.arrays.begin(), data.arrays.end(), [&](const auto& a) { return a.name == name; });
        if (offset < 0 || count < 0 || offset > static_cast<int64_t>(array.bytes.size()) - count)
            throw std::runtime_error("geometry dependency range out of bounds");
        return std::vector<uint8_t>(array.bytes.begin() + offset, array.bytes.begin() + offset + count);
    };
    return result;
}

// Missing/deleted points return nullopt. Malformed references never resolve.
inline std::optional<std::vector<uint8_t>> pointValue(const DependencyGeometry& geometry, const PointReference& point)
{
    auto index = point.index;
    auto name = point.array;
    if (name == "point_ids") {
        if (geometry.kind != "point_collection" || index.size() != 1 || index[0] < 0) return std::nullopt;
        const auto found = geometry.arrays.find(name);
        if (found == geometry.arrays.end()) return std::nullopt;
        GeometrySnapshot snapshot;
        snapshot.arrays[name] = geometry.read(name, 0, found->second.byteLength);
        const auto ids = snapshot.values<int64_t>(name);
        const auto id = std::find(ids.begin(), ids.end(), index[0]);
        if (id == ids.end()) return std::nullopt;
        if (std::find(id + 1, ids.end(), index[0]) != ids.end())
            throw std::runtime_error("geometry dependency has ambiguous point ID");
        index = {static_cast<int64_t>(id - ids.begin())}; name = "points";
    } else {
        // Collections use stable IDs, never vector offsets.
        if (geometry.kind == "point_collection") return std::nullopt;
        if (geometry.kind == "fiber" ? (name != "control_points" && name != "line_points") : name != "points")
            return std::nullopt;
    }
    const auto found = geometry.arrays.find(name);
    if (found == geometry.arrays.end()) return std::nullopt;
    const auto& array = found->second;
    if (array.shape.size() != index.size() + 1 || array.shape.back() != 3) return std::nullopt;
    int64_t offset = 0;
    for (size_t i = 0; i < index.size(); ++i) {
        if (index[i] < 0 || index[i] >= array.shape[i]) return std::nullopt;
        offset = offset * array.shape[i] + index[i];
    }
    const int64_t stride = array.dtype == "<f8" ? 24 : 12;
    auto bytes = geometry.read(name, offset * stride, stride);
    GeometrySnapshot snapshot; snapshot.arrays["point"] = bytes;
    if (array.dtype == "<f8") (void)snapshot.values<double>("point");
    else if (array.dtype == "<f4") {
        const auto xyz = snapshot.values<float>("point");
        if (geometry.kind == "patch" && xyz[2] <= 0) return std::nullopt;
    }
    return bytes;
}

inline void validateDependencies(const Anchor& anchor, const DependencyGeometry& geometry)
{
    if (anchor.wholeGeometry && anchor.points.empty()) return;
    if (anchor.wholeGeometry || anchor.points.empty())
        throw std::runtime_error("geometry edit requires explicit point dependencies or whole_geometry");
    for (const auto& point : anchor.points)
        if (!pointValue(geometry, point))
            throw std::runtime_error("geometry edit refers to a missing or invalid point; update or remove the relationship");
}

inline bool dependenciesChanged(const Anchor& anchor, const DependencyGeometry& before, const DependencyGeometry& after)
{
    if (anchor.wholeGeometry || anchor.points.empty() || before.kind != after.kind || before.attributes != after.attributes) return true;
    for (const auto& point : anchor.points) {
        const auto old = pointValue(before, point), next = pointValue(after, point);
        if (!old || !next || old != next) return true;
        // Positional identities cannot be assumed to survive insertion/removal.
        if (point.array != "point_ids" && before.arrays.at(point.array).shape != after.arrays.at(point.array).shape) return true;
    }
    return false;
}

inline bool endpointChanged(const ObjectStore& db, const Txn& txn, const Anchor& anchor,
                            const std::optional<std::string>& bound, const std::optional<std::string>& head)
{
    if (bound == head) return false;
    if (!bound || !head) return true;
    const auto before = db.geometryRevision(txn, *bound), after = db.geometryRevision(txn, *head);
    if (!before || !after) return true;
    return dependenciesChanged(anchor, dependencyGeometry(db, txn, *before), dependencyGeometry(db, txn, *after));
}
} // namespace vc::objects::detail
