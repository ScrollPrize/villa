#include "vc/core/graph/GeometryIngest.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Tiff.hpp"
#include "vc/core/util/PackedTrackStore.hpp"
#include "vc/core/util/Uuid.hpp"
#include "GraphFiberInput.hpp"
#include "GraphPatchInput.hpp"
#include "GraphTrackInput.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <span>

namespace vc::graph {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
Json readJson(const fs::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    return Json::parse(input);
}
std::string uuid(const Json& data)
{
    auto value = data.at("uuid").get<std::string>();
    if (!util::isUuid(value)) throw std::runtime_error("source has no valid UUID");
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return value;
}
// Store explicit little-endian scalar bytes on every supported host.
template<class T> GeometryArray array(std::string name, std::string dtype, std::vector<int64_t> shape, std::span<const T> values)
{
    GeometryArray result{std::move(name), std::move(dtype), std::move(shape), {}};
    result.bytes.resize(values.size_bytes());
    if (!values.empty()) std::memcpy(result.bytes.data(), values.data(), values.size_bytes());
    if constexpr (std::endian::native != std::endian::little)
        for (size_t i = 0; i < result.bytes.size(); i += sizeof(T))
            std::reverse(result.bytes.begin() + i, result.bytes.begin() + i + sizeof(T));
    return result;
}
GeometryArray points(const char* name, const std::vector<std::array<double, 3>>& values)
{
    std::vector<double> flat;
    flat.reserve(values.size() * 3);
    for (const auto& point : values) flat.insert(flat.end(), point.begin(), point.end());
    return array<double>(name, "<f8", {static_cast<int64_t>(values.size()), 3}, flat);
}
struct Prepared {
    ObjectRecord object;
    GeometryData geometry;
};
ObjectRecord sourceObject(const Json& root, const std::string& kind, const fs::path& source)
{
    ObjectRecord result;
    result.uuid = uuid(root); result.kind = kind; result.name = source.stem().string();
    result.storageKind = "file"; result.storageRef = fs::absolute(source).lexically_normal().string();
    result.formatVersion = 1;
    return result;
}
Prepared fiber(const fs::path& source)
{
    auto root = readJson(source);
    if (root.value("type", "") != "vc3d_fiber") throw std::runtime_error("expected vc3d_fiber JSON");
    Prepared p;
    p.object = sourceObject(root, "fiber", source); p.geometry.kind = "fiber";
    for (const char* key : {"control_points", "line_points"}) {
        if (!root.contains(key)) continue;
        if (!root.at(key).is_array()) throw std::runtime_error("invalid fiber point array");
        for (const auto& entry : root.at(key)) {
            const auto& xyz = entry.is_object() ? entry.at("position") : entry;
            if (!xyz.is_array() || xyz.size() != 3) throw std::runtime_error("fiber points require exactly three XYZ coordinates");
        }
    }
    const auto control = detail::fiberPoints(root, "control_points"), line = detail::fiberPoints(root, "line_points");
    p.geometry.arrays = {points("control_points", control), points("line_points", line)};
    FiberDetail f;
    f.controlPointCount = control.size(); f.linePointCount = line.size();
    f.length = detail::fiberPolylineLength(line.empty() ? control : line);
    f.width = root.value("width", 0.0); f.widthGapFraction = root.value("width_gap_fraction", 0.2);
    f.optimizationMode = root.value("optimization_mode", "");
    if (root.contains("hv_classification")) {
        const auto& hv = root.at("hv_classification");
        const auto manual = hv.value("manual_tag", "");
        f.manualFamily = !manual.empty(); f.family = f.manualFamily ? manual : hv.value("automatic_tag", "");
    }
    p.object.fiber = f;
    // Retain per-control tracing settings and other source annotations, without
    // duplicating coordinate arrays as JSON. Positional order remains unchanged.
    for (const char* key : {"control_points", "line_points"}) {
        Json extras = Json::array();
        for (const auto& entry : root.value(key, Json::array())) {
            if (entry.is_object()) { auto extra = entry; extra.erase("position"); extras.push_back(std::move(extra)); }
            else extras.push_back(nullptr);
        }
        root.erase(key);
        // Most line points have no extra fields. Avoid millions of JSON nulls.
        if (std::any_of(extras.begin(), extras.end(), [](const auto& e) { return !e.is_null() && !e.empty(); }))
            root[std::string(key) + "_attributes"] = std::move(extras);
    }
    p.geometry.attributes = root.dump();
    return p;
}
Prepared patch(const fs::path& source)
{
    auto fingerprint = [&]() {
        std::map<std::string, std::pair<uintmax_t, fs::file_time_type>> result;
        for (const auto& entry : fs::directory_iterator(source))
            if (entry.is_regular_file() && (entry.path().extension() == ".tif" || entry.path().filename() == "meta.json"))
                result[entry.path().filename().string()] = {entry.file_size(), entry.last_write_time()};
        return result;
    };
    const auto before = fingerprint();
    const auto root = readJson(source / "meta.json");
    Prepared p;
    p.object = sourceObject(root, "patch", source); p.geometry.kind = "patch";
    detail::validatePatchFiles(source);
    QuadSurface surface(source);
    const auto grid = surface.rawPoints();
    std::vector<float> values;
    values.reserve(grid.total() * 3);
    for (int row = 0; row < grid.rows; ++row)
        for (int col = 0; col < grid.cols; ++col)
            for (int axis = 0; axis < 3; ++axis) values.push_back(grid(row, col)[axis]);
    p.geometry.arrays.push_back(array<float>("points", "<f4", {grid.rows, grid.cols, 3}, values));
    PatchDetail d;
    d.gridWidth = grid.cols; d.gridHeight = grid.rows;
    const auto scale = surface.scale(); d.scale = {scale[0], scale[1]};
    if (root.contains("bbox") && root.at("bbox").is_array() && root.at("bbox").size() == 2) {
        const auto low = root.at("bbox")[0].get<std::array<double, 3>>();
        const auto high = root.at("bbox")[1].get<std::array<double, 3>>();
        d.bbox = {low[0], low[1], low[2], high[0], high[1], high[2]};
    }
    p.object.patch = d;
    if (before != fingerprint()) throw std::runtime_error("patch changed while reading; retry ingestion");
    p.geometry.attributes = root.dump();
    return p;
}
std::vector<Prepared> collections(const fs::path& source)
{
    const auto root = readJson(source);
    if (root.value("vc_pointcollections_json_version", "") != "1" || !root.at("collections").is_object())
        throw std::runtime_error("expected vc_pointcollections JSON version 1");
    std::vector<Prepared> result;
    std::set<std::string> seen;
    for (const auto& [collectionId, original] : root.at("collections").items()) {
        Prepared p;
        p.object = sourceObject(original, "point_collection", source);
        if (!seen.insert(p.object.uuid).second) throw std::runtime_error("duplicate collection UUID");
        p.object.name = original.value("name", ""); p.geometry.kind = "point_collection";
        std::map<int64_t, Json> ordered;
        if (!original.at("points").is_object()) throw std::runtime_error("points must be an object");
        for (const auto& [key, value] : original.at("points").items()) {
            int64_t id;
            const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), id);
            if (error != std::errc{} || end != key.data() + key.size() || id < 0 || !ordered.emplace(id, value).second)
                throw std::runtime_error("invalid or duplicate numeric point ID");
        }
        std::vector<int64_t> ids;
        std::vector<std::array<double, 3>> coordinates;
        auto attributes = original; attributes.erase("points");
        attributes["source_collection_id"] = collectionId;
        attributes["point_attributes"] = Json::array();
        for (const auto& [id, entry] : ordered) {
            if (!entry.at("p").is_array() || entry.at("p").size() != 3) throw std::runtime_error("invalid XYZ point");
            auto xyz = entry.at("p").get<std::array<double, 3>>();
            for (double value : xyz) if (!std::isfinite(value)) throw std::runtime_error("non-finite point");
            ids.push_back(id); coordinates.push_back(xyz);
            auto extra = entry; extra.erase("p"); attributes["point_attributes"].push_back(std::move(extra));
        }
        p.geometry.arrays.push_back(points("points", coordinates));
        p.geometry.arrays.push_back(array<int64_t>("point_ids", "<i8", {static_cast<int64_t>(ids.size())}, ids));
        p.geometry.attributes = attributes.dump();
        result.push_back(std::move(p));
    }
    return result;
}
Prepared track(const PackedTrackStore& store, size_t row, ObjectRecord object)
{
    Prepared p;
    p.object = std::move(object); p.geometry.kind = "track";
    const auto start = store.offsets()[row], count = store.offsets()[row + 1] - start;
    std::vector<int32_t> xyz;
    xyz.reserve(static_cast<size_t>(count) * 3);
    const auto raw = store.coordinates();
    for (int64_t i = 0; i < count; ++i) {
        for (int axis = 2; axis >= 0; --axis) xyz.push_back(raw[3 * (start + i) + axis]);
    }

    p.geometry.arrays.push_back(array<int32_t>("points", "<i4", {count, 3}, xyz));
    p.object.track->pointCount = count; p.object.track->length = detail::trackPolylineLength(store, row, nullptr);
    p.object.track->family = detail::trackFamily(store, row);
    p.geometry.attributes = Json{{"family", p.object.track->family}, {"external_key", p.object.track->externalKey}}.dump();
    return p;
}
struct Expected { std::optional<std::string> revision; std::optional<int64_t> generation; };
Expected expectedAt(GraphDb& db, const Txn& read, const std::string& id)
{
    Expected e;
    if (const auto object = db.objectByUuid(read, id)) e.generation = object->generation;
    if (const auto geometry = db.currentGeometry(read, id)) e.revision = geometry->uuid;
    return e;
}
Expected expected(GraphDb& db, const std::string& id)
{
    auto read = db.read();
    return expectedAt(db, read, id);
}
void publish(GraphDb& db, Prepared p, const Expected& e, GeometryIngestReport& report)
{
    auto write = db.write();
    const auto existing = db.objectByUuid(write, p.object.uuid);
    if ((existing ? std::optional(existing->generation) : std::nullopt) != e.generation)
        throw std::runtime_error("object changed while reading source; retry ingestion");
    if (existing && existing->kind != p.object.kind) throw std::runtime_error("source UUID belongs to another object kind");
    if (!existing) db.upsertObject(write, p.object);
    const auto revision = db.putGeometry(write, p.object.uuid, p.geometry, e.revision);
    if (e.revision && revision == *e.revision) { write.commit(); ++report.unchanged; return; }
    // Geometry publication owns its descriptor; keep user metadata/tags intact.
    // Refresh searchable measurements without replacing source-independent fields.
    auto record = *db.objectByUuid(write, p.object.uuid);
    if (p.object.fiber) record.fiber = p.object.fiber;
    if (p.object.patch) record.patch = p.object.patch;
    if (p.object.track) record.track = p.object.track;
    db.upsertObject(write, record);
    write.commit(); ++report.ingested;
}
void problem(GeometryIngestReport& report, const std::string& label, const std::exception& error)
{
    report.problems.push_back(label + ": " + error.what());
}
}

GeometryIngestReport ingestRegisteredGeometry(GraphDb& db, const std::optional<fs::path>& datasetRoot,
    const std::vector<std::string>& objectUuids, const util::ProgressCallback& progress)
{
    GeometryIngestReport report;
    std::vector<ObjectRecord> objects;
    {
        auto read = db.read();
        if (objectUuids.empty()) {
            for (const char* kind : {"patch", "fiber", "point_collection", "track"}) {
                auto batch = db.objectsByKind(read, kind); objects.insert(objects.end(), batch.begin(), batch.end());
            }
        } else {
            std::set<std::string> seen;
            for (const auto& id : objectUuids) {
                const auto object = db.objectByUuid(read, id);
                if (!object) report.problems.push_back(id + ": object not found");
                else if (seen.insert(object->uuid).second) objects.push_back(*object);
            }
        }
    }
    std::map<fs::path, std::vector<Prepared>> pclCache;
    std::map<fs::path, std::unique_ptr<PackedTrackStore>> trackCache;
    for (size_t i = 0; i < objects.size(); ++i) {
        const auto& object = objects[i];
        util::reportProgress(progress, "ingesting geometry", i, objects.size(), object.uuid);
        try {
            const auto e = expected(db, object.uuid);
            Prepared p;
            if (object.kind == "track") {
                if (!object.track) throw std::runtime_error("track detail missing");
                fs::path path(object.track->datasetRef);
                if (path.is_relative()) path = datasetRoot.value_or(db.path().parent_path()) / path;
                const auto packed = resolvePackedTrackStore(path);
                auto& store = trackCache[packed];
                if (!store) store = std::make_unique<PackedTrackStore>(packed);
                const auto id = detail::trackSourceId(object.track->externalKey);
                const auto found = std::lower_bound(store->sourceIds().begin(), store->sourceIds().end(), id);
                if (found == store->sourceIds().end() || *found != id) throw std::runtime_error("track source ID missing");
                p = track(*store, found - store->sourceIds().begin(), object);
            } else {
                std::optional<fs::path> path;
                { auto read = db.read(); path = db.resolveStoragePath(read, object.uuid, datasetRoot); }
                if (!path) throw std::runtime_error("source not found by UUID");
                if (object.kind == "patch") p = patch(*path);
                else if (object.kind == "fiber") p = fiber(*path);
                else if (object.kind == "point_collection") {
                    if (!pclCache.contains(*path)) pclCache[*path] = collections(*path);
                    const auto& collections = pclCache.at(*path);
                    const auto found = std::find_if(collections.begin(), collections.end(), [&](const auto& c) { return c.object.uuid == object.uuid; });
                    if (found == collections.end()) throw std::runtime_error("collection disappeared");
                    p = *found;
                } else throw std::runtime_error("unsupported object kind");
                if (p.object.uuid != object.uuid) throw std::runtime_error("source UUID changed during ingestion");
            }
            publish(db, std::move(p), e, report);
        } catch (const std::exception& error) { problem(report, object.uuid, error); }
    }
    util::reportProgress(progress, "ingesting geometry", objects.size(), objects.size());
    return report;
}

GeometryIngestReport ingestGeometrySource(GraphDb& db, const std::string& kind, const fs::path& source,
    const util::ProgressCallback& progress)
{
    GeometryIngestReport report;
    try {
        if (kind == "track") {
            const auto packed = resolvePackedTrackStore(source);
            PackedTrackStore store(packed);
            std::map<std::string, ObjectRecord> existing;
            std::set<std::string> refs;
            {
                auto read = db.read();
                for (const auto& object : db.objectsByKind(read, "track")) {
                    if (!object.track) continue;
                    fs::path path(object.track->datasetRef);
                    if (path.is_relative()) path = db.path().parent_path() / path;
                    std::error_code error;
                    if (fs::weakly_canonical(path, error) == packed && !error) {
                        refs.insert(object.track->datasetRef); existing.emplace(object.track->externalKey, object);
                    }
                }
            }
            if (refs.size() > 1) throw std::runtime_error("track dataset registered under multiple aliases");
            const auto ref = refs.empty() ? packed.string() : *refs.begin();
            for (size_t row = 0; row < store.sourceIds().size(); ++row) {
                const auto key = std::to_string(store.sourceIds()[row]);
                util::reportProgress(progress, "ingesting tracks", row, store.sourceIds().size(), key);
                try {
                    ObjectRecord object;
                    if (existing.contains(key)) object = existing.at(key);
                    else {
                        object.uuid = util::newUuid(); object.kind = "track"; object.name = key;
                        object.storageKind = "dataset"; object.storageRef = ref;
                        object.track = TrackDetail{}; object.track->datasetRef = ref; object.track->externalKey = key;
                    }
                    const auto e = expected(db, object.uuid);
                    publish(db, track(store, row, object), e, report);
                } catch (const std::exception& error) { problem(report, key, error); }
            }
        } else {
            std::vector<Prepared> batch;
            std::map<std::string, Expected> expectations;
            {
                // Pin the graph snapshot before reading files so a concurrent
                // geometry edit cannot be overwritten by a slow source load.
                auto read = db.read();
                if (kind == "patch") batch.push_back(patch(source));
                else if (kind == "fiber") batch.push_back(fiber(source));
                else if (kind == "point_collection") batch = collections(source);
                else throw std::runtime_error("unsupported geometry kind: " + kind);
                for (const auto& p : batch) expectations[p.object.uuid] = expectedAt(db, read, p.object.uuid);
            }
            for (auto& p : batch) {
                try {
                    const auto e = expectations.at(p.object.uuid);
                    publish(db, std::move(p), e, report);
                } catch (const std::exception& error) { problem(report, source.string(), error); }
            }
        }
    } catch (const std::exception& error) { problem(report, source.string(), error); }
    return report;
}
} // namespace vc::graph
