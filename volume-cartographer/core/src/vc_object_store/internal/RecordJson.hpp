#pragma once
#include "internal/Anchors.hpp"
#include "internal/JsonFields.hpp"
#include "vc/core/objects/ObjectStore.hpp"
#include <set>
namespace vc::objects::detail {
nlohmann::json bboxToJson(const std::optional<std::array<double, 6>>& box);

bool includeField(const std::set<std::string>* fields, const char* key);

nlohmann::json commonToJson(const RecordCommon& r, const std::set<std::string>* fields = nullptr);

void commonFromJson(const nlohmann::json& j, RecordCommon& r);

nlohmann::json objectToJson(const ObjectRecord& r, const std::set<std::string>* fields = nullptr);

ObjectRecord objectFromJson(const nlohmann::json& j);

nlohmann::json edgeToJson(EdgeRecord e, const std::set<std::string>* fields = nullptr, bool canonicalOrder = true);

EdgeRecord edgeFromJson(const nlohmann::json& j);

} // namespace vc::objects::detail
