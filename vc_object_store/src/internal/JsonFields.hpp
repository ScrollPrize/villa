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
std::optional<std::array<double, 6>> bboxFromJson(const nlohmann::json& j);

std::array<double, 2> scaleFromJson(const nlohmann::json& value);

} // namespace vc::objects::detail
