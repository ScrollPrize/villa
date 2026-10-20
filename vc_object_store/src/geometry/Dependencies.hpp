#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include <functional>
#include "geometry/GeometryReader.hpp"

namespace vc::objects::detail {
// A bounded reader for either a published revision or a proposed payload.
struct DependencyGeometry {
    std::string kind, attributes;
    std::map<std::string, std::string> fingerprints;
    std::map<std::string, GeometryArrayInfo> arrays;
    std::function<std::vector<uint8_t>(const std::string&, int64_t, int64_t)> read;
    struct PointIndex {
        bool loaded = false;
        std::vector<std::pair<int64_t, int64_t>> offsets;
    };
    std::shared_ptr<PointIndex> pointIndex = std::make_shared<PointIndex>();
};

// Nested helpers in a publication/repair share revision readers. The cache is
// confined to this operation, not the connection or a future transaction.
using DependencyCache = std::map<std::pair<const Txn*, std::string>, DependencyGeometry>;
inline thread_local DependencyCache* activeDependencies = nullptr;
class DependencyScope {
    std::unique_ptr<DependencyCache> owned_;
public:
    DependencyScope() {
        if (!activeDependencies) {
            owned_ = std::make_unique<DependencyCache>(); activeDependencies = owned_.get();
        }
    }
    ~DependencyScope() { if (owned_) activeDependencies = nullptr; }
};

inline DependencyGeometry dependencyGeometry(const ObjectStore& db, const Txn& txn, const GeometryRevision& revision)
{
    const auto key = std::make_pair(&txn, revision.uuid);
    if (activeDependencies) {
        const auto cached = activeDependencies->find(key);
        if (cached != activeDependencies->end()) return cached->second;
    }
    DependencyGeometry result; result.kind = revision.kind; result.attributes = revision.attributes;
    for (const auto& array : revision.arrays) result.arrays.emplace(array.name, array);
    auto reader = std::make_shared<GeometryReader>(txn, revision.uuid);
    for (const auto& [name, a] : reader->manifest().at("arrays").items()) {
        auto descriptor = a; descriptor.erase("source_revision");
        result.fingerprints[name] = descriptor.dump();
    }
    result.read = [reader](const auto& name, auto offset, auto count) {
        return reader->read(name, offset, count);
    };
    if (activeDependencies) {
        if (activeDependencies->size() == 8) activeDependencies->erase(activeDependencies->begin());
        activeDependencies->emplace(key, result);
    }
    return result;
}

inline DependencyGeometry dependencyGeometry(const GeometryData& data)
{
    DependencyGeometry result; result.kind = data.kind; result.attributes = data.attributes;
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
        auto& cache = *geometry.pointIndex;
        if (!cache.loaded) {
            GeometrySnapshot snapshot;
            snapshot.arrays[name] = geometry.read(name, 0, found->second.byteLength);
            const auto ids = snapshot.values<int64_t>(name);
            cache.offsets.reserve(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) cache.offsets.emplace_back(ids[i], int64_t(i));
            std::sort(cache.offsets.begin(), cache.offsets.end());
            cache.loaded = true;
        }
        const auto id = std::lower_bound(cache.offsets.begin(), cache.offsets.end(), std::pair{index[0], int64_t(0)});
        if (id == cache.offsets.end() || id->first != index[0]) return std::nullopt;
        if (id + 1 != cache.offsets.end() && (id + 1)->first == index[0])
            throw std::runtime_error("geometry dependency has ambiguous point ID");
        index = {id->second}; name = "points";
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
    if (before.kind != after.kind || before.attributes != after.attributes) return true;
    auto changed = [&](const std::string& name) {
        const auto a = before.arrays.find(name), b = after.arrays.find(name);
        if (a == before.arrays.end() || b == after.arrays.end()) return a != before.arrays.end() || b != after.arrays.end();
        if (a->second.dtype != b->second.dtype || a->second.shape != b->second.shape) return true;
        if (before.fingerprints.contains(name) && after.fingerprints.contains(name))
            return before.fingerprints.at(name) != after.fingerprints.at(name);
        return before.read(name, 0, a->second.byteLength) != after.read(name, 0, b->second.byteLength);
    };
    for (const auto& channel : anchor.channels) if (changed(channel)) return true;
    if (anchor.wholeGeometry || anchor.points.empty()) {
        // Whole geometry includes every array; unknown legacy dependencies
        // remain conservative. Explicit point anchors depend only on their
        // declared points and optional channels.
        std::set<std::string> names;
        for (const auto& [name, a] : before.arrays) names.insert(name);
        for (const auto& [name, a] : after.arrays) names.insert(name);
        for (const auto& name : names) if (changed(name)) return true;
        return false;
    }
    auto identical = [&](const std::string& name) {
        return before.fingerprints.contains(name) && after.fingerprints.contains(name) &&
               before.fingerprints.at(name) == after.fingerprints.at(name);
    };
    for (const auto& point : anchor.points) {
        if (identical(point.array) && (point.array != "point_ids" || identical("points"))) continue;
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
