#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/Uuid.hpp"
#include "formats/FiberInput.hpp"
#include <algorithm>
#include <set>

namespace vc::objects {
using Json = nlohmann::json;
FiberDocument decodeFiberJson(const std::string& text)
{
    auto root = Json::parse(text);
    if (root.value("type", "") != "vc3d_fiber") throw std::runtime_error("expected vc3d_fiber JSON");
    FiberDocument p;
    p.object.uuid = util::canonicalUuid(root.at("uuid").get<std::string>());
    p.object.kind = "fiber";
    p.object.name = root.value("name", std::string{});
    p.object.formatVersion = root.value("version", int64_t(1));
    p.object.tags = root.value("tags", std::vector<std::string>{});
    for (const char* key : {"control_points", "line_points"}) {
        if (!root.contains(key)) continue;
        if (!root.at(key).is_array()) throw std::runtime_error("invalid fiber point array");
        for (const auto& entry : root.at(key)) {
            const auto& xyz = entry.is_object() ? entry.at("position") : entry;
            if (!xyz.is_array() || xyz.size() != 3) throw std::runtime_error("fiber points require exactly three XYZ coordinates");
        }
    }
    const auto control = detail::fiberPoints(root, "control_points"), line = detail::fiberPoints(root, "line_points");
    p.geometry = vc::geometry::encode(vc::geometry::Fiber{control, line});
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
    p.source = root.dump();
    return p;
}
std::string encodeFiberJson(const std::string& uuid, const GeometryData& data,
                           const std::string& source)
{
    const auto geometry = std::get<vc::geometry::Fiber>(vc::geometry::decode(data));
    auto attributes = Json::parse(source);
    attributes["uuid"] = uuid;
    attributes["type"] = "vc3d_fiber";
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
                (std::string_view(name) == "control_points" && attributes.value("version", 1) >= 3)) {
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
    const auto metadata = Json::parse(revision->record.meta);
    auto source=metadata.value("source",Json::object());
    for (const auto* key : {"coordinates", "coordinate_binding"})
        if (metadata.contains(key)) source[key] = metadata.at(key);
    for(const auto* name:{"control_points","line_points"}) {
        const auto attrs=Json::parse(db.geometryPointAnnotations(txn,*revision->geometryRevision,name));
        const auto key=std::string(name)+"_attributes";
        if(std::any_of(attrs.begin(),attrs.end(),[](const auto& a) {return !a.is_null() && !a.empty();})) source[key]=attrs;
        else source.erase(key);
    }
    source.erase("branches"); source.erase("adjacent_branches");
    source["version"]=revision->record.formatVersion;
    source["tags"]=revision->record.tags;
    if(revision->record.fiber) {
        source["width"]=revision->record.fiber->width;
        source["width_gap_fraction"]=revision->record.fiber->widthGapFraction;
        if (!revision->record.fiber->optimizationMode.empty()) source["optimization_mode"]=revision->record.fiber->optimizationMode;
        else source.erase("optimization_mode");
        if (!revision->record.fiber->family.empty()) {
            source["hv_classification"]["manual_tag"]=revision->record.fiber->manualFamily ? revision->record.fiber->family : "";
            if(!revision->record.fiber->manualFamily) source["hv_classification"]["automatic_tag"]=revision->record.fiber->family;
        }
    }
    return encodeFiberJson(revision->objectUuid,db.readGeometryData(txn,*revision->geometryRevision,byteBudget),source.dump());
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
            revision->deleted || revision->record.retired || !revision->geometryRevision)
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
    if (pending.empty()) for (const auto& [uuid, revision] : revisions) pending.push_back(uuid);
    while (!pending.empty()) {
        auto uuid = std::move(pending.back()); pending.pop_back();
        if (!revisions.contains(uuid)) throw std::invalid_argument("Unknown fiber export root " + uuid);
        if (!included.insert(uuid).second) continue;
        for (const auto& peer : neighbours[uuid]) pending.push_back(peer);
    }
    struct FiberExport {
        Json document;
        std::vector<int64_t> controlIds;
    };
    std::map<std::string, FiberExport> fibers;
    for (const auto& uuid : included) {
        const auto& revision = revisions.at(uuid);
        FiberExport fiber;
        fiber.document = Json::parse(fiberRevisionJson(db, txn, revision.uuid, byteBudget));
        fiber.document["name"] = revision.record.name;
        fiber.document["filename"] = uuid + ".json";
        fiber.document["branches"] = Json::array();
        fiber.document["adjacent_branches"] = Json::array();
        const auto geometry = db.geometryRevision(txn, *revision.geometryRevision);
        if (std::any_of(geometry->arrays.begin(), geometry->arrays.end(),
                       [](const auto& array) { return array.name == "control_point_ids"; })) {
            const auto data = db.readGeometryArray(txn, *revision.geometryRevision, "control_point_ids");
            for (size_t i = 0; i < data.size(); i += 8) {
                uint64_t id = 0;
                for (size_t byte = 0; byte < 8; ++byte) id |= uint64_t(data[i + byte]) << (8 * byte);
                fiber.controlIds.push_back(static_cast<int64_t>(id));
            }
        }
        fibers.emplace(uuid, std::move(fiber));
    }
    const auto controlIndex = [](const FiberExport& fiber, const Anchor& anchor) -> size_t {
        if (anchor.points.size() != 1 || anchor.points.front().index.size() != 1)
            throw std::runtime_error("Fiber branch must identify one control point");
        const auto& point = anchor.points.front();
        const auto id = point.index.front();
        if (point.array == "control_point_ids") {
            const auto found = std::find(fiber.controlIds.begin(), fiber.controlIds.end(), id);
            if (found != fiber.controlIds.end()) return size_t(found - fiber.controlIds.begin());
        } else if (point.array == "control_points" && id >= 0 &&
                   uint64_t(id) < fiber.document.at("control_points").size()) return size_t(id);
        throw std::runtime_error("Fiber branch control point is absent from selected geometry");
    };
    for (const auto& edge : edges) {
        if (!included.contains(edge.aUuid)) continue;
        auto& a = fibers.at(edge.aUuid); auto& b = fibers.at(edge.bUuid);
        const auto ai = controlIndex(a, edge.aAnchor), bi = controlIndex(b, edge.bAnchor);
        const auto append = [&](FiberExport& local, const FiberExport& target,
                                size_t i, size_t j, const Anchor& own, const Anchor& other) {
            if (!own.direction || !other.direction)
                throw std::runtime_error("Fiber branch endpoint direction is missing");
            const auto position = [](const Json& point) { return point.is_object() ? point.at("position") : point; };
            Json branch{{"edge_uuid", edge.uuid},
                        {"branch_file", target.document.at("filename")},
                        {"control_point_index", i}, {"branch_control_point_index", j},
                        {"control_point_position", position(local.document.at("control_points").at(i))},
                        {"branch_control_point_position", position(target.document.at("control_points").at(j))},
                        {"control_point_direction", *own.direction}, {"branch_control_point_direction", *other.direction},
                        {"pending", edge.approvalState != "approved"}};
            const bool adjacent = Json::parse(edge.meta).value("adjacent", false);
            local.document[adjacent ? "adjacent_branches" : "branches"].push_back(std::move(branch));
        };
        append(a, b, ai, bi, edge.aAnchor, edge.bAnchor);
        append(b, a, bi, ai, edge.bAnchor, edge.aAnchor);
    }
    std::map<std::string, std::string> result;
    for (auto& [uuid, fiber] : fibers) result.emplace(uuid, fiber.document.dump());
    return result;
}

}
