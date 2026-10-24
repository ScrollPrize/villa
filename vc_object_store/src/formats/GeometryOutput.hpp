#pragma once
#include "geometry/GeometrySnapshot.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
namespace vc::objects::formats {
void fiber(const detail::GeometrySnapshot&, nlohmann::json, const std::filesystem::path&);
void collection(const detail::GeometrySnapshot&, nlohmann::json, const std::filesystem::path&);
void patch(const detail::GeometrySnapshot&, nlohmann::json, const std::filesystem::path&);
void track(const detail::GeometrySnapshot&, const nlohmann::json&, const std::filesystem::path&);
}
