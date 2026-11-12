#pragma once
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <string_view>

namespace vc::geometry {
// VC point-map boundary: numeric stable ID keys, XYZ in `p`, remaining
// per-point fields as annotations. File/document policies stay with callers.
struct AnnotatedPointMap {
    PointCollection points;
    nlohmann::json annotations = nlohmann::json::array();
};
int64_t pointIdFromString(std::string_view);
AnnotatedPointMap decodePointMap(const nlohmann::json&);
}
