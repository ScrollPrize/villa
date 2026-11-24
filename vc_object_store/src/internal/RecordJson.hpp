#pragma once
#include "internal/Anchors.hpp"
#include "internal/JsonFields.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include <set>
namespace vc::objects::detail {
bool includeField(const std::set<std::string>* fields, const char* key);

nlohmann::json commonToJson(const RecordCommon& r, const std::set<std::string>* fields = nullptr);

// Publication generations are local to a database, not part of transferred
// content identity. Keep provenance, approval state and geometry revision UUIDs.
nlohmann::json recordContent(nlohmann::json record);

void commonFromJson(const nlohmann::json& j, RecordCommon& r);
// Projected records (query fields): reads only the common fields present.
void presentCommonFromJson(const nlohmann::json& j, RecordCommon& r);

nlohmann::json objectToJson(const ObjectRecord& r, const std::set<std::string>* fields = nullptr);

ObjectRecord objectFromJson(const nlohmann::json& j);

nlohmann::json edgeToJson(EdgeRecord e, const std::set<std::string>* fields = nullptr, bool canonicalOrder = true);

// projected: common fields may be partial (see presentCommonFromJson).
EdgeRecord edgeFromJson(const nlohmann::json& j, bool projected = false);

} // namespace vc::objects::detail
