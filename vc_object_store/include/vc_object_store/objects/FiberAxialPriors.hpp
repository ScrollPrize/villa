#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace vc::objects {
std::unique_ptr<float[]> fiberAxialPriorColumns(std::span<const int64_t> offsets,
    std::span<const double> xyz,const std::filesystem::path& umbilicus,
    const std::string& volume,const std::optional<std::array<int64_t,3>>& baseShapeZYX={},
    double spacing=16.,size_t workers=32);
} // namespace vc::objects
