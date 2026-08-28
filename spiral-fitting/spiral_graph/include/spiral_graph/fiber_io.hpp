#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace spiral::fiber {

struct Point {
    float z = 0.0f;
    float y = 0.0f;
    float x = 0.0f;
};

enum class Axis { horizontal, vertical };

struct Branch {
    std::size_t local_control = 0;
    std::string other_file;
    std::size_t other_control = 0;
    bool pending = false;
};

struct Fiber {
    std::filesystem::path path;
    std::string id;
    Axis axis = Axis::horizontal;
    std::vector<Point> points;
    std::vector<std::size_t> control_line;
    std::vector<Branch> branches;
};

struct LoadOptions {
    float coordinate_scale = 1.0f;
    std::unordered_set<std::string> invalid_fibers;
};

Fiber load_vc3d_fiber(
    const std::filesystem::path& path, const LoadOptions& options = {});

std::vector<Fiber> load_vc3d_fiber_directory(
    const std::filesystem::path& directory, const LoadOptions& options = {});

std::size_t retained_control_point(const Fiber& fiber, std::size_t control);
double arclength(const Fiber& fiber);
const char* axis_name(Axis axis) noexcept;

} // namespace spiral::fiber
