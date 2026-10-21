#pragma once
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace vc::objects::detail {
std::optional<std::array<double, 6>> bboxFromJson(const nlohmann::json& j);

std::array<double, 2> scaleFromJson(const nlohmann::json& value);

} // namespace vc::objects::detail
