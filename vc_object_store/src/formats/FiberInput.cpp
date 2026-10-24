#include "formats/GeometryInput.hpp"
#include "geometry/ArrayBytes.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "formats/FiberInput.hpp"
#include <cmath>
namespace vc::objects::formats {
using Json = nlohmann::json;
namespace fs = std::filesystem;
Prepared fiber(const fs::path& source)
{
    auto root = readJson(source);
    if (root.value("type", "") != "vc3d_fiber") throw std::runtime_error("expected vc3d_fiber JSON");
    Prepared p;
    p.object = sourceObject(root, "fiber", source); p.geometry.kind = "fiber";
    for (const char* key : {"control_points", "line_points"}) {
        if (!root.contains(key)) continue;
        if (!root.at(key).is_array()) throw std::runtime_error("invalid fiber point array");
        for (const auto& entry : root.at(key)) {
            const auto& xyz = entry.is_object() ? entry.at("position") : entry;
            if (!xyz.is_array() || xyz.size() != 3) throw std::runtime_error("fiber points require exactly three XYZ coordinates");
        }
    }
    const auto control = detail::fiberPoints(root, "control_points"), line = detail::fiberPoints(root, "line_points");
    p.geometry = vc::geometry::encode(vc::geometry::Fiber{control, line});
    FiberDetail f;
    f.controlPointCount = control.size(); f.linePointCount = line.size();
    f.length = vc::geometry::polylineLength(line.empty() ? control : line);
    f.width = root.value("width", 0.0); f.widthGapFraction = root.value("width_gap_fraction", 0.2);
    f.optimizationMode = root.value("optimization_mode", "");
    if (root.contains("hv_classification")) {
        const auto& hv = root.at("hv_classification");
        const auto manual = hv.value("manual_tag", "");
        f.manualFamily = !manual.empty(); f.family = f.manualFamily ? manual : hv.value("automatic_tag", "");
    }
    p.object.fiber = f;
    // Retain per-control tracing settings and other source annotations, without
    // duplicating coordinate arrays as JSON. Positional order remains unchanged.
    for (const char* key : {"control_points", "line_points"}) {
        Json extras = Json::array();
        for (const auto& entry : root.value(key, Json::array())) {
            if (entry.is_object()) { auto extra = entry; extra.erase("position"); extras.push_back(std::move(extra)); }
            else extras.push_back(nullptr);
        }
        root.erase(key);
        // Most line points have no extra fields. Avoid millions of JSON nulls.
        if (std::any_of(extras.begin(), extras.end(), [](const auto& e) { return !e.is_null() && !e.empty(); }))
            root[std::string(key) + "_attributes"] = std::move(extras);
    }
    p.source = root;
    return p;
}
}
