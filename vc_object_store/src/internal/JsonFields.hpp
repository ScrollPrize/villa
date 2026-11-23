#pragma once
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace vc::objects::detail {
// Core metadata/dependency fields accept absent/empty JSON as an empty object.
// Nonempty arrays/scalars remain invalid; extension-owned JSON is not normalized.
nlohmann::json objectJson(const nlohmann::json& value);
nlohmann::json parseObjectJson(const std::string& text);
std::string canonicalObjectJson(const std::string& text);
// Admit serialized containers once at the owning decoder. JSON range iteration
// alone also accepts object values and null; it does not enforce an array schema.
const nlohmann::json& jsonArray(const nlohmann::json& value, const char* context);
// Optional means absent -> empty; an explicitly supplied null is still invalid.
const nlohmann::json& jsonArrayField(const nlohmann::json& record, const char* key, bool optional = false);

std::optional<std::array<double, 6>> bboxFromJson(const nlohmann::json& j);

std::array<double, 2> scaleFromJson(const nlohmann::json& value);

} // namespace vc::objects::detail
