#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/Uuid.hpp"
#include "formats/FiberInput.hpp"
#include "formats/FiberMetadata.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "geometry/Dependencies.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include <algorithm>
#include <set>

namespace vc::objects {
using Json = nlohmann::json;
size_t fiberBranchControlIndex(const std::vector<int64_t>& controlIds, const Anchor& anchor)
{
    if (anchor.points.size() != 1 || anchor.points.front().index.size() != 1)
        throw std::runtime_error("Fiber branch must identify one control point");
    const auto& point = anchor.points.front();
    const auto id = point.index.front();
    if (point.array == "control_point_ids") {
        const auto found = std::find(controlIds.begin(), controlIds.end(), id);
        if (found != controlIds.end()) return size_t(found - controlIds.begin());
    }
    throw std::runtime_error("Fiber branch control point is absent from selected geometry");
}

GeometrySource decodeFiberJson(const std::string& text)
{
    auto root = Json::parse(text);
    if (root.value("type", "") != "vc3d_fiber") throw std::runtime_error("expected vc3d_fiber JSON");
    if (root.value("version", int64_t(0)) != 4)
        throw std::runtime_error("Fiber transfer requires version 4");
    GeometrySource p;
    p.object.uuid = util::canonicalUuid(root.at("uuid").get<std::string>());
    p.object.kind = "fiber";
    p.object.name = root.value("name", std::string{});
    p.object.formatVersion = 4;
    p.object.tags = root.value("tags", std::vector<std::string>{});
    for (const char* key : {"control_points", "line_points"}) {
        if (!root.contains(key)) continue;
        if (!root.at(key).is_array()) throw std::runtime_error("invalid fiber point array");
        for (const auto& entry : root.at(key)) {
            if (std::string_view(key) == "control_points" && !entry.is_object())
                throw std::runtime_error("Fiber control points require position objects");
            const auto& xyz = entry.is_object() ? entry.at("position") : entry;
            if (!xyz.is_array() || xyz.size() != 3) throw std::runtime_error("fiber points require exactly three XYZ coordinates");
        }
    }
    const auto control = detail::fiberPoints(root, "control_points"), line = detail::fiberPoints(root, "line_points");
    p.geometry = vc::geometry::encode(vc::geometry::Fiber{control, line,
        root.value("control_point_ids", std::vector<int64_t>{})});
    root.erase("control_point_ids");
    FiberDetail f;
    f.controlPointCount = control.size(); f.linePointCount = line.size();
    f.length = vc::geometry::polylineLength(line.empty() ? control : line);
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
    p.source = std::move(root);
    return p;
}
std::string encodeFiberJson(const std::string& uuid, const GeometryData& data,
                           const std::string& source)
{
    const auto geometry = std::get<vc::geometry::Fiber>(vc::geometry::decode(data));
    auto attributes = Json::parse(source);
    attributes["uuid"] = uuid;
    attributes["type"] = "vc3d_fiber";
    attributes["version"] = 4;
    attributes["control_point_ids"] = geometry.controlPointIds;
    for (const auto* name : {"control_points", "line_points"}) {
        const auto& values = std::string_view(name) == "control_points" ? geometry.controlPoints : geometry.linePoints;
        const auto key = std::string(name) + "_attributes";
        const auto extras = attributes.value(key, Json());
        if (!extras.is_null() && (!extras.is_array() || extras.size() != values.size()))
            throw std::runtime_error(key + " must have one entry per point");
        attributes.erase(key);
        auto points = Json::array();
        for (size_t i = 0; i < values.size(); ++i) {
            if (!extras.is_null() && !extras[i].is_null() && !extras[i].is_object())
                throw std::runtime_error("invalid " + key);
            Json xyz = values[i];
            if ((!extras.is_null() && extras[i].is_object()) ||
                std::string_view(name) == "control_points") {
                auto entry = !extras.is_null() && extras[i].is_object() ? extras[i] : Json::object();
                entry["position"] = std::move(xyz); points.push_back(std::move(entry));
            } else points.push_back(std::move(xyz));
        }
        attributes[name] = std::move(points);
    }
    return attributes.dump();
}
std::string fiberRevisionJson(const ObjectStore& db,const Txn& txn,const std::string& objectRevision,uint64_t byteBudget)
{
    const auto revision=db.objectRevision(txn,objectRevision);
    if(!revision || revision->record.kind!="fiber" || !revision->geometryRevision)
        throw std::runtime_error("Fiber object revision has no geometry");
    const auto geometry = db.geometryRevision(txn, *revision->geometryRevision);
    if (!geometry) throw std::runtime_error("Fiber geometry revision is missing");
    return encodeFiberJson(revision->objectUuid, db.readGeometryData(txn, geometry->uuid, byteBudget),
                          detail::fiberMetadata(db, txn, revision->record, *geometry).dump());
}

namespace detail {
Json fiberMetadata(const ObjectStore& db, const Txn& txn, const ObjectRecord& record, const GeometryRevision& geometry)
{
    auto source = geometryTransferMetadata(Json::parse(record.meta));
    for (const auto* name : {"control_points", "line_points"}) {
        const auto attrs = Json::parse(db.geometryPointAnnotations(txn, geometry.uuid, name));
        const auto key = std::string(name) + "_attributes";
        if (std::any_of(attrs.begin(), attrs.end(), [](const auto& a) { return !a.is_null() && !a.empty(); })) source[key] = attrs;
        else source.erase(key);
    }
    source["name"] = record.name;
    source["filename"] = record.uuid + ".json";
    source["version"] = 4;
    source["tags"] = record.tags;
    source["branches"] = Json::array(); source["adjacent_branches"] = Json::array();
    if (record.fiber) {
        const auto& fiber = *record.fiber;
        source["width"] = fiber.width; source["width_gap_fraction"] = fiber.widthGapFraction;
        if (!fiber.optimizationMode.empty()) source["optimization_mode"] = fiber.optimizationMode;
        else source.erase("optimization_mode");
        source["hv_classification"]["manual_tag"] = fiber.manualFamily ? fiber.family : "";
        if (!fiber.manualFamily) source["hv_classification"]["automatic_tag"] = fiber.family;
    }
    return source;
}

Json fiberBranchJson(const ObjectStore& db, const Txn& txn, const EdgeRecord& edge,
    const Anchor& own, const Anchor& other, const GeometryRevision& ownGeometry,
    const GeometryRevision& otherGeometry)
{
    const auto endpoint = [&](const GeometryRevision& geometry, const Anchor& anchor) {
        GeometrySnapshot snapshot;
        std::vector<int64_t> ids;
        for (const auto& array : geometry.arrays) {
            if (array.name == "control_point_ids") {
                snapshot.arrays[array.name] = db.readGeometryArray(txn, geometry.uuid, array.name);
                ids = snapshot.values<int64_t>(array.name);
            }
        }
        const auto index = fiberBranchControlIndex(ids, anchor);
        const auto bytes = pointValue(dependencyGeometry(db, txn, geometry), anchor.points.front());
        if (!bytes) throw std::runtime_error("Fiber branch control point is missing");
        snapshot.arrays["point"] = *bytes;
        return std::pair{index, snapshot.values<double>("point")};
    };
    if (!own.direction || !other.direction) throw std::runtime_error("Fiber branch endpoint direction is missing");
    const auto [i, position] = endpoint(ownGeometry, own);
    const auto [j, targetPosition] = endpoint(otherGeometry, other);
    return Json{{"edge_uuid", edge.uuid}, {"target_uuid", otherGeometry.objectUuid},
        {"branch_file", otherGeometry.objectUuid + ".json"},
        {"control_point_index", i}, {"branch_control_point_index", j},
        {"control_point_position", position}, {"branch_control_point_position", targetPosition},
        {"control_point_direction", *own.direction}, {"branch_control_point_direction", *other.direction},
        {"pending", edge.approvalState != "approved"}};
}
}

std::map<std::string, std::string> fiberNetworkJson(
    const ObjectStore& db, const Txn& txn, const RevisionSelection& selection,
    const std::vector<std::string>& roots, uint64_t byteBudget)
{
    std::map<std::string, ObjectRevision> revisions;
    for (const auto& [uuid, pin] : selection) {
        if (!pin) throw std::invalid_argument("Fiber export requires explicit object revision pins");
        const auto revision = db.objectRevision(txn, *pin);
        if (!revision || revision->objectUuid != uuid || revision->record.kind != "fiber" ||
            revision->deleted)
            throw std::invalid_argument("Invalid fiber export revision for " + uuid);
        revisions.emplace(uuid, *revision);
    }
    std::vector<EdgeRecord> edges;
    std::map<std::string, std::vector<std::string>> neighbours;
    for (const auto& selected : db.relationshipsAt(txn, selection)) {
        const auto& edge = selected.relationship.record;
        if (edge.kind != "fiber_branch") continue;
        // An omitted endpoint must never silently remove a branch constraint.
        if (!revisions.contains(edge.aUuid) || !revisions.contains(edge.bUuid))
            throw std::invalid_argument("Fiber export selection is missing a branch endpoint");
        if (!selected.applicable) continue;
        edges.push_back(edge);
        neighbours[edge.aUuid].push_back(edge.bUuid);
        neighbours[edge.bUuid].push_back(edge.aUuid);
    }
    std::set<std::string> included;
    std::vector<std::string> pending = roots;
    if (pending.empty()) for (const auto& [uuid, revision] : revisions)
        if (!revision.record.retired) pending.push_back(uuid);
    while (!pending.empty()) {
        auto uuid = std::move(pending.back()); pending.pop_back();
        if (!revisions.contains(uuid) || revisions.at(uuid).record.retired)
            throw std::invalid_argument("Unknown active fiber export root " + uuid);
        if (!included.insert(uuid).second) continue;
        for (const auto& peer : neighbours[uuid]) pending.push_back(peer);
    }
    std::map<std::string, Json> fibers;
    for (const auto& uuid : included)
        fibers.emplace(uuid, Json::parse(fiberRevisionJson(db, txn, revisions.at(uuid).uuid, byteBudget)));
    for (const auto& edge : edges) {
        if (!included.contains(edge.aUuid)) continue;
        const auto aGeometry = db.geometryRevision(txn, *revisions.at(edge.aUuid).geometryRevision);
        const auto bGeometry = db.geometryRevision(txn, *revisions.at(edge.bUuid).geometryRevision);
        const auto key = Json::parse(edge.meta).value("adjacent", false) ? "adjacent_branches" : "branches";
        fibers.at(edge.aUuid)[key].push_back(detail::fiberBranchJson(db, txn, edge, edge.aAnchor, edge.bAnchor, *aGeometry, *bGeometry));
        fibers.at(edge.bUuid)[key].push_back(detail::fiberBranchJson(db, txn, edge, edge.bAnchor, edge.aAnchor, *bGeometry, *aGeometry));
    }
    std::map<std::string, std::string> result;
    for (auto& [uuid, fiber] : fibers) result.emplace(uuid, fiber.dump());
    return result;
}

}
