#include "internal/Anchors.hpp"
#include "vc_object_store/Uuid.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc::objects::detail {
void validateAnchor(const Anchor& anchor)
{
    for (const auto* vector : {&anchor.position, &anchor.direction})
        if (*vector) for (double value : **vector)
            if (!std::isfinite(value)) throw std::invalid_argument("anchors must be finite");
    if (anchor.wholeGeometry && !anchor.points.empty())
        throw std::invalid_argument("anchor cannot have both whole_geometry and point dependencies");
    for (const auto& name : anchor.channels)
        if (name.empty() || name == "points") throw std::invalid_argument("invalid channel dependency name");
    for (const auto& key : anchor.metadataKeys)
        if (key.empty()) throw std::invalid_argument("metadata dependency keys must be non-empty");
    for (const auto& point : anchor.points)
        if (point.array.empty() || point.index.empty() ||
            std::any_of(point.index.begin(), point.index.end(), [](auto i) { return i < 0; }))
            throw std::invalid_argument("invalid anchor point reference");
}
void validateEdgeAnchors(const EdgeRecord& edge)
{
    validateAnchor(edge.aAnchor); validateAnchor(edge.bAnchor);
    if (util::canonicalUuid(edge.aUuid) == util::canonicalUuid(edge.bUuid) &&
        (edge.aAnchor.points.empty() || edge.bAnchor.points.empty() ||
         edge.aAnchor.points.front() == edge.bAnchor.points.front()))
        throw std::invalid_argument("self-edges require two distinct point references");
}
void swapEndpoints(EdgeRecord& edge)
{
    std::swap(edge.aUuid, edge.bUuid);
    std::swap(edge.aAnchor, edge.bAnchor);
}
void canonicalizeEdge(EdgeRecord& edge)
{
    edge.aUuid = util::canonicalUuid(edge.aUuid);
    edge.bUuid = util::canonicalUuid(edge.bUuid);
    validateEdgeAnchors(edge);
    if (edge.bUuid < edge.aUuid || (edge.aUuid == edge.bUuid &&
        dependenciesToJson(edge.bAnchor).dump() < dependenciesToJson(edge.aAnchor).dump()))
        swapEndpoints(edge);
}
std::string edgeIdentity(const EdgeRecord& input)
{
    // Identity does not need to copy potentially large metadata or tag lists.
    EdgeRecord edge;
    edge.kind = input.kind; edge.aUuid = input.aUuid; edge.bUuid = input.bUuid;
    edge.aAnchor = input.aAnchor; edge.bAnchor = input.bAnchor;
    canonicalizeEdge(edge);
    return nlohmann::json::array({edge.kind, edge.aUuid, edge.bUuid,
        dependenciesToJson(edge.aAnchor), dependenciesToJson(edge.bAnchor)}).dump();
}

nlohmann::json dependenciesToJson(const Anchor& anchor)
{
    validateAnchor(anchor);
    nlohmann::json value = nlohmann::json::object();
    if (anchor.wholeGeometry) value["whole_geometry"] = true;
    if (!anchor.channels.empty()) value["channels"] = anchor.channels;
    if (!anchor.metadataKeys.empty()) value["metadata_keys"] = anchor.metadataKeys;
    if (!anchor.points.empty()) {
        value["points"] = nlohmann::json::array();
        for (const auto& point : anchor.points) {
            value["points"].push_back({{"array", point.array}, {"index", point.index}});
        }
    }
    return value;
}

void dependenciesFromJson(Anchor& anchor, const nlohmann::json& value)
{
    anchor.channels = value.value("channels", std::vector<std::string>{});
    anchor.wholeGeometry = value.value("whole_geometry", false);
    anchor.metadataKeys = value.value("metadata_keys", std::vector<std::string>{});
    for (const auto& point : value.value("points", nlohmann::json::array()))
        anchor.points.push_back({point.at("array").get<std::string>(), point.at("index").get<std::vector<int64_t>>()});
    (void)dependenciesToJson(anchor);
}

nlohmann::json anchorToJson(const Anchor& a)
{
    nlohmann::json j = nlohmann::json::object();
    if (a.position) j["position"] = {(*a.position)[0], (*a.position)[1], (*a.position)[2]};
    if (a.direction) j["direction"] = {(*a.direction)[0], (*a.direction)[1], (*a.direction)[2]};
    j.update(dependenciesToJson(a));
    return j;
}

Anchor anchorFromJson(const nlohmann::json& j)
{
    Anchor a;
    if (!j.is_object()) throw std::invalid_argument("anchor must be a JSON object");
    for (const auto& [key, value] : j.items())
        if (key != "points" && key != "whole_geometry" && key != "position" && key != "direction" && key != "metadata_keys" && key != "channels")
            throw std::invalid_argument("unknown anchor field: " + key);
    dependenciesFromJson(a, j);
    for (const char* key : {"position", "direction"}) {
        if (j.contains(key) && !j.at(key).is_null()) {
            if (!j.at(key).is_array() || j.at(key).size() != 3)
                throw std::invalid_argument(std::string("anchor ") + key + " must have 3 numbers");
            std::array<double, 3> v{};
            for (int i = 0; i < 3; ++i) v[size_t(i)] = j.at(key).at(size_t(i)).get<double>();
            if (std::string(key) == "position") a.position = v; else a.direction = v;
        }
    }
    validateAnchor(a);
    return a;
}

} // namespace vc::objects::detail
