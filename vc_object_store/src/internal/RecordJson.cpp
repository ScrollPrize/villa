#include "internal/RecordJson.hpp"
#include "vc_object_store/Uuid.hpp"

namespace vc::objects::detail {
nlohmann::json bboxToJson(const std::optional<std::array<double, 6>>& box)
{
    if (!box) return nullptr;
    return {(*box)[0], (*box)[1], (*box)[2], (*box)[3], (*box)[4], (*box)[5]};
}

bool includeField(const std::set<std::string>* fields, const char* key)
{
    return !fields || fields->contains(key);
}

nlohmann::json commonToJson(const RecordCommon& r, const std::set<std::string>* fields)
{
    nlohmann::json j{{"uuid", r.uuid}, {"kind", r.kind}};
    auto put = [&](const char* key, const auto& value) { if (includeField(fields, key)) j[key] = value; };
    put("generation", r.generation);
    put("created_by", r.createdBy); put("created_at", r.createdAt); put("created_host", r.createdHost);
    put("changed_by", r.changedBy); put("changed_at", r.changedAt); put("changed_host", r.changedHost);
    put("tags", r.tags); put("approval_state", r.approvalState);
    if (includeField(fields, "meta")) j["meta"] = nlohmann::json::parse(r.meta);
    put("approved_by", r.approvedBy ? nlohmann::json(*r.approvedBy) : nlohmann::json());
    put("approved_at", r.approvedAt ? nlohmann::json(*r.approvedAt) : nlohmann::json());
    put("approved_host", r.approvedHost ? nlohmann::json(*r.approvedHost) : nlohmann::json());
    put("approved_generation", r.approvedGeneration ? nlohmann::json(*r.approvedGeneration) : nlohmann::json());
    return j;
}

void commonFromJson(const nlohmann::json& j, RecordCommon& r)
{
    r.generation = j.at("generation").get<int64_t>();
    if (r.generation < 1) throw std::runtime_error("object store: snapshot generation must be positive");
    r.createdBy = j.at("created_by").get<std::string>();
    r.createdAt = j.at("created_at").get<std::string>();
    r.createdHost = j.at("created_host").get<std::string>();
    r.changedBy = j.at("changed_by").get<std::string>();
    r.changedAt = j.at("changed_at").get<std::string>();
    r.changedHost = j.at("changed_host").get<std::string>();
    auto optionalText = [&](const char* key) -> std::optional<std::string> {
        const auto& value = j.at(key);
        return value.is_null() ? std::nullopt : std::optional(value.get<std::string>());
    };
    r.approvedBy = optionalText("approved_by");
    r.approvedAt = optionalText("approved_at");
    r.approvedHost = optionalText("approved_host");
    if (!j.at("approved_generation").is_null()) r.approvedGeneration = j.at("approved_generation").get<int64_t>();
    if (r.approvalState != "pending" && r.approvalState != "approved" && r.approvalState != "rejected")
        throw std::runtime_error("object store: invalid snapshot approval state");
    if (r.approvedGeneration && *r.approvedGeneration != r.generation)
        throw std::runtime_error("object store: snapshot approval does not cover its generation");
    if (r.approvalState == "pending" && (r.approvedBy || r.approvedAt || r.approvedHost || r.approvedGeneration))
        throw std::runtime_error("object store: pending snapshot has approval stamps");
}

nlohmann::json objectToJson(const ObjectRecord& r, const std::set<std::string>* fields)
{
    nlohmann::json j = commonToJson(r, fields);
    if (includeField(fields, "retired")) j["retired"] = r.retired;
    if (includeField(fields, "lifecycle_epoch")) j["lifecycle_epoch"] = r.lifecycleEpoch;
    if (includeField(fields, "name")) j["name"] = r.name;
    if (includeField(fields, "format_version")) j["format_version"] = r.formatVersion;
    if (includeField(fields, "storage_kind")) j["storage_kind"] = r.storageKind;
    if (includeField(fields, "storage_ref")) j["storage_ref"] = r.storageRef;
    if (includeField(fields, "data_generation")) j["data_generation"] = r.dataGeneration;
    if (r.dataChecksum && includeField(fields, "data_checksum")) j["data_checksum"] = *r.dataChecksum;
    if (r.fiber && includeField(fields, "fiber")) {
        j["fiber"] = {{"family", r.fiber->family}, {"manual_family", r.fiber->manualFamily},
                      {"control_point_count", r.fiber->controlPointCount}, {"line_point_count", r.fiber->linePointCount},
                      {"length", r.fiber->length}, {"width", r.fiber->width},
                      {"width_gap_fraction", r.fiber->widthGapFraction}, {"optimization_mode", r.fiber->optimizationMode},
                      {"bbox", bboxToJson(r.fiber->bbox)}};
    }
    if (r.patch && includeField(fields, "patch")) {
        j["patch"] = {{"grid_width", r.patch->gridWidth}, {"grid_height", r.patch->gridHeight},
                      {"scale", r.patch->scale}, {"bbox", bboxToJson(r.patch->bbox)}};
    }
    if (r.track && includeField(fields, "track")) {
        j["track"] = {{"dataset_ref", r.track->datasetRef}, {"external_key", r.track->externalKey},
                      {"family", r.track->family}, {"point_count", r.track->pointCount},
                      {"length", r.track->length}, {"bbox", bboxToJson(r.track->bbox)}};
    }
    return j;
}

ObjectRecord objectFromJson(const nlohmann::json& j)
{
    ObjectRecord r;
    r.uuid = j.value("uuid", "");
    r.kind = j.value("kind", "");
    r.name = j.value("name", "");
    r.retired = j.value("retired", false);
    r.lifecycleEpoch = j.value("lifecycle_epoch", "");
    r.formatVersion = j.value("format_version", int64_t(0));
    r.storageKind = j.value("storage_kind", "");
    r.storageRef = j.value("storage_ref", "");
    r.dataGeneration = j.value("data_generation", int64_t(0));
    if (j.contains("data_checksum") && j.at("data_checksum").is_string()) r.dataChecksum = j.at("data_checksum").get<std::string>();
    if (j.contains("tags")) r.tags = j.at("tags").get<std::vector<std::string>>();
    if (j.contains("meta")) r.meta = j.at("meta").dump();
    r.approvalState = j.value("approval_state", "pending");
    if (j.contains("fiber")) {
        const auto& f = j.at("fiber");
        FiberDetail d;
        d.family = f.value("family", "");
        d.manualFamily = f.value("manual_family", false);
        d.controlPointCount = f.value("control_point_count", int64_t(0));
        d.linePointCount = f.value("line_point_count", int64_t(0));
        d.length = f.value("length", 0.0);
        d.width = f.value("width", 0.0);
        d.widthGapFraction = f.value("width_gap_fraction", 0.2);
        d.optimizationMode = f.value("optimization_mode", "");
        if (f.contains("bbox")) d.bbox = bboxFromJson(f.at("bbox"));
        r.fiber = d;
    }
    if (j.contains("patch")) {
        const auto& p = j.at("patch");
        PatchDetail d;
        d.gridWidth = p.value("grid_width", int64_t(0));
        d.gridHeight = p.value("grid_height", int64_t(0));
        d.scale = scaleFromJson(p.value("scale", nlohmann::json(0.0)));
        if (p.contains("bbox")) d.bbox = bboxFromJson(p.at("bbox"));
        r.patch = d;
    }
    if (j.contains("track")) {
        const auto& t = j.at("track");
        TrackDetail d;
        d.datasetRef = t.value("dataset_ref", "");
        d.externalKey = t.value("external_key", "");
        d.family = t.value("family", "");
        d.pointCount = t.value("point_count", int64_t(0));
        d.length = t.value("length", 0.0);
        if (t.contains("bbox")) d.bbox = bboxFromJson(t.at("bbox"));
        r.track = d;
    }
    return r;
}

nlohmann::json edgeToJson(EdgeRecord e, const std::set<std::string>* fields, bool canonicalOrder)
{
    if (canonicalOrder) canonicalizeEdge(e);
    auto j = commonToJson(e, fields);
    j["a"] = e.aUuid;
    j["b"] = e.bUuid;
    if (includeField(fields, "a_anchor")) j["a_anchor"] = anchorToJson(e.aAnchor);
    if (includeField(fields, "b_anchor")) j["b_anchor"] = anchorToJson(e.bAnchor);
    return j;
}

EdgeRecord edgeFromJson(const nlohmann::json& j)
{
    EdgeRecord e;
    e.uuid = j.at("uuid").get<std::string>(); e.kind = j.at("kind").get<std::string>();
    e.aUuid = util::canonicalUuid(j.at("a").get<std::string>());
    e.bUuid = util::canonicalUuid(j.at("b").get<std::string>());
    if (j.contains("a_anchor")) e.aAnchor = anchorFromJson(j.at("a_anchor"));
    if (j.contains("b_anchor")) e.bAnchor = anchorFromJson(j.at("b_anchor"));
    if (j.contains("tags")) e.tags = j.at("tags").get<std::vector<std::string>>();
    if (j.contains("meta")) e.meta = j.at("meta").dump();
    e.approvalState = j.value("approval_state", "pending");
    commonFromJson(j, e);
    return e;
}
} // namespace vc::objects::detail
