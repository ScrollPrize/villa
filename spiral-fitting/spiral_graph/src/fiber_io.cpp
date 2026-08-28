#include <spiral_graph/fiber_io.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace spiral::fiber {
namespace {

using json = nlohmann::json;
constexpr std::size_t invalid_index = std::numeric_limits<std::size_t>::max();

Point point_json(const json& value, float scale)
{
    const json& position = value.is_object() ? value.at("position") : value;
    if (!position.is_array() || position.size() != 3) {
        throw std::runtime_error("fiber point must be an [x, y, z] array");
    }
    Point point{
        position[2].get<float>() * scale,
        position[1].get<float>() * scale,
        position[0].get<float>() * scale,
    };
    if (!std::isfinite(point.z) || !std::isfinite(point.y)
        || !std::isfinite(point.x)) {
        throw std::runtime_error("fiber point must be finite");
    }
    return point;
}

std::size_t nearest_point(const std::vector<Point>& points, const Point& query)
{
    std::size_t best = 0;
    double best_squared = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < points.size(); ++index) {
        const double dz = points[index].z - query.z;
        const double dy = points[index].y - query.y;
        const double dx = points[index].x - query.x;
        const double squared = dz * dz + dy * dy + dx * dx;
        if (squared < best_squared) {
            best = index;
            best_squared = squared;
        }
    }
    return best;
}

Axis classified_axis(const json& document, const std::filesystem::path& path)
{
    const json classification = document.value("hv_classification", json::object());
    std::string tag = classification.value("manual_tag", std::string{});
    if (tag != "H" && tag != "V") {
        tag = classification.value("automatic_tag", std::string{});
    }
    if (tag == "H") return Axis::horizontal;
    if (tag == "V") return Axis::vertical;
    throw std::runtime_error(
        "fiber has no valid H/V classification: " + path.string());
}

} // namespace

Fiber load_vc3d_fiber(const std::filesystem::path& path, const LoadOptions& options)
{
    if (!(options.coordinate_scale > 0.0f)
        || !std::isfinite(options.coordinate_scale)) {
        throw std::invalid_argument("invalid fiber loading options");
    }
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    json document;
    stream >> document;
    if (document.value("type", "vc3d_fiber") != "vc3d_fiber") {
        throw std::runtime_error("not a VC3D fiber: " + path.string());
    }

    Fiber fiber;
    fiber.path = path;
    fiber.id = path.filename().string();
    fiber.axis = classified_axis(document, path);
    std::vector<Point> controls;
    for (const auto& value : document.at("control_points")) {
        controls.push_back(point_json(value, options.coordinate_scale));
    }
    if (controls.empty()) throw std::runtime_error("fiber has no control points");
    std::vector<Point> line;
    if (document.contains("line_points") && document["line_points"].is_array()) {
        for (const auto& value : document["line_points"]) {
            line.push_back(point_json(value, options.coordinate_scale));
        }
    }
    if (line.size() < controls.size()) line = controls;

    fiber.control_line.reserve(controls.size());
    for (const Point& control : controls) {
        fiber.control_line.push_back(nearest_point(line, control));
    }
    const std::size_t first = fiber.control_line.front();
    const std::size_t last = fiber.control_line.back();
    const std::size_t span_begin = std::min(first, last);
    const std::size_t span_end = std::max(first, last);
    const bool reverse = first > last;
    std::vector<Point> trimmed(
        line.begin() + static_cast<std::ptrdiff_t>(span_begin),
        line.begin() + static_cast<std::ptrdiff_t>(span_end + 1));
    if (reverse) std::reverse(trimmed.begin(), trimmed.end());
    for (auto& index : fiber.control_line) {
        if (index < span_begin || index > span_end) index = invalid_index;
        else index = reverse ? first - index : index - first;
    }

    if (document.contains("branches") && document["branches"].is_array()) {
        for (const auto& value : document["branches"]) {
            if (!value.contains("branch_file")
                || !value.contains("control_point_index")
                || !value.contains("branch_control_point_index")) continue;
            fiber.branches.push_back({
                value.at("control_point_index").get<std::size_t>(),
                std::filesystem::path(value.at("branch_file").get<std::string>())
                    .filename().string(),
                value.at("branch_control_point_index").get<std::size_t>(),
                value.value("pending", false),
            });
        }
    }

    fiber.branches.erase(std::remove_if(
        fiber.branches.begin(), fiber.branches.end(), [&](const Branch& branch) {
            return branch.local_control >= fiber.control_line.size()
                || fiber.control_line[branch.local_control] == invalid_index;
        }), fiber.branches.end());
    fiber.points = std::move(trimmed);
    return fiber;
}

std::vector<Fiber> load_vc3d_fiber_directory(
    const std::filesystem::path& directory, const LoadOptions& options)
{
    if (!std::filesystem::is_directory(directory)) {
        throw std::invalid_argument("fiber path is not a directory: " + directory.string());
    }
    std::vector<std::filesystem::path> paths;
    std::unordered_set<std::string> found_invalid;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;
        const std::string id = entry.path().filename().string();
        if (options.invalid_fibers.contains(id)) {
            found_invalid.insert(id);
        } else {
            paths.push_back(entry.path());
        }
    }
    for (const std::string& id : options.invalid_fibers) {
        if (!found_invalid.contains(id)) {
            throw std::invalid_argument("invalid fiber not found: " + id);
        }
    }
    std::sort(paths.begin(), paths.end());
    std::vector<Fiber> fibers;
    fibers.reserve(paths.size());
    for (const auto& path : paths) fibers.push_back(load_vc3d_fiber(path, options));
    return fibers;
}

std::size_t retained_control_point(const Fiber& fiber, std::size_t control)
{
    if (control >= fiber.control_line.size()) {
        throw std::runtime_error("fiber branch control-point index is out of range");
    }
    const std::size_t wanted = fiber.control_line[control];
    if (wanted == invalid_index || wanted >= fiber.points.size()) {
        throw std::runtime_error("fiber branch control point is outside the retained span");
    }
    return wanted;
}

double arclength(const Fiber& fiber)
{
    double result = 0.0;
    for (std::size_t index = 1; index < fiber.points.size(); ++index) {
        const double dz = fiber.points[index].z - fiber.points[index - 1].z;
        const double dy = fiber.points[index].y - fiber.points[index - 1].y;
        const double dx = fiber.points[index].x - fiber.points[index - 1].x;
        result += std::sqrt(dz * dz + dy * dy + dx * dx);
    }
    return result;
}

const char* axis_name(Axis axis) noexcept
{
    return axis == Axis::horizontal ? "H" : "V";
}

} // namespace spiral::fiber
