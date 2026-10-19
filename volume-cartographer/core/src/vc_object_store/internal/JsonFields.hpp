#pragma once
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace vc::objects { namespace {
std::optional<std::array<double, 6>> bboxFromJson(const nlohmann::json& j)
{
    if (!j.is_array() || j.size() != 6) return std::nullopt;
    std::array<double, 6> box{};
    for (int i = 0; i < 6; ++i) box[size_t(i)] = j.at(size_t(i)).get<double>();
    return box;
}

std::array<double, 2> scaleFromJson(const nlohmann::json& value)
{
    std::array<double, 2> scale{};
    if (value.is_number()) {
        scale.fill(value.get<double>());  // v1 scalar scales were isotropic
    } else if (value.is_array() && value.size() == 2) {
        scale = {value.at(0).get<double>(), value.at(1).get<double>()};
    } else {
        throw std::runtime_error("object store: patch scale must be a number or a two-element array");
    }
    for (double axis : scale)
        if (!std::isfinite(axis) || axis < 0)
            throw std::runtime_error("object store: patch scale must be finite and nonnegative");
    return scale;
}

} }
