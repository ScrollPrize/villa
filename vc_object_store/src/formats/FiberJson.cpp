#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/Uuid.hpp"
#include "formats/FiberInput.hpp"
#include <algorithm>

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
    auto source=Json::parse(revision->record.meta).value("source",Json::object());
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

}
