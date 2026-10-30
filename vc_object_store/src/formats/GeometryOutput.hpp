#pragma once
#include "geometry/GeometrySnapshot.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
namespace vc::objects::formats {
void collection(const detail::GeometrySnapshot&, nlohmann::json, const std::filesystem::path&);
// validOnly: vertices invalid in `valid` become the sentinel and no mask.tif is written.
void surface(const detail::GeometrySnapshot&, nlohmann::json, const std::filesystem::path&, bool validOnly = false);
void umbilicus(const detail::GeometrySnapshot&, nlohmann::json, const std::filesystem::path&);
void track(const detail::GeometrySnapshot&, const nlohmann::json&, const std::filesystem::path&);
}
