#include "internal/RecordJson.hpp"
#include "internal/RecordFields.hpp"
#include "vc_object_store/Uuid.hpp"

namespace vc::objects::detail {
namespace {
template<class T> nlohmann::json fieldJson(const T& value) { return value; }
template<class T> nlohmann::json fieldJson(const std::optional<T>& value) {
    return value ? nlohmann::json(*value) : nlohmann::json();
}
template<class T> void readDetailField(const nlohmann::json& value, T& field) { value.get_to(field); }
void readDetailField(const nlohmann::json& value, std::array<double, 2>& field) { field = scaleFromJson(value); }
void readDetailField(const nlohmann::json& value, std::optional<std::array<double, 6>>& field) { field = bboxFromJson(value); }
}

bool includeField(const std::set<std::string>* fields, const char* key)
{
    return !fields || fields->contains(key);
}

nlohmann::json commonToJson(const RecordCommon& r, const std::set<std::string>* fields)
{
    nlohmann::json j{{"uuid", r.uuid}, {"kind", r.kind}};
    commonFields(r, [&](const char* key, const auto& value) {
        if (includeField(fields, key)) j[key] = fieldJson(value);
    });
    if (includeField(fields, "meta")) j["meta"] = nlohmann::json::parse(r.meta);
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
    objectFields(r, [&](const char* key, const auto& value) {
        if (includeField(fields, key)) j[key] = value;
    });
    if (r.dataChecksum && includeField(fields, "data_checksum")) j["data_checksum"] = *r.dataChecksum;
    objectDetails(r, [&](const char* key, const auto& detail) {
        if (!detail || !includeField(fields, key)) return;
        auto& value = j[key];
        detailFields(*detail, [&](const char* name, const auto& field) { value[name] = fieldJson(field); });
    });
    return j;
}

ObjectRecord objectFromJson(const nlohmann::json& j)
{
    ObjectRecord r;
    r.uuid = j.value("uuid", "");
    r.kind = j.value("kind", "");
    r.retired = j.value("retired", false);
    r.lifecycleEpoch = j.value("lifecycle_epoch", "");
    objectFields(r, [&](const char* key, auto& field) { field = j.value(key, field); });
    if (j.contains("data_checksum") && j.at("data_checksum").is_string()) r.dataChecksum = j.at("data_checksum").get<std::string>();
    if (j.contains("tags")) r.tags = j.at("tags").get<std::vector<std::string>>();
    if (j.contains("meta")) r.meta = j.at("meta").dump();
    r.approvalState = j.value("approval_state", "pending");
    objectDetails(r, [&](const char* key, auto& detail) {
        if (!j.contains(key)) return;
        detail.emplace();
        const auto& value = j.at(key);
        if (!value.is_object()) throw std::invalid_argument("object detail must be an object");
        detailFields(*detail, [&](const char* name, auto& field) {
            if (value.contains(name)) readDetailField(value.at(name), field);
        });
    });
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
