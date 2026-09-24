#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include "vc/core/util/ArcHermite.hpp"

namespace vc::fiber_tracer {

inline std::string displayNormalSourceFromJson(const nlohmann::json& control)
{
    if (!control.contains("display_normal_source")) return "unknown";
    const auto& source = control.at("display_normal_source");
    if (!control.contains("display_normal") || !source.is_string() ||
        (source != "manual" && source != "interpolated" && source != "unknown"))
        throw std::runtime_error("display_normal_source requires a normal and manual/interpolated/unknown");
    return source.get<std::string>();
}

inline constexpr double kDefaultFiberWidthGapFraction = 0.20;

// Width is the full edge-to-edge distance, so the 80%/120% guide pairs
// are at +/-40% and +/-60% of that value relative to the centerline.
inline std::array<double, 4> fiberWidthEdgeOffsets(
    double width, double gapFraction = kDefaultFiberWidthGapFraction)
{
    const double inner = width * (1.0 - gapFraction) / 2.0;
    const double outer = width * (1.0 + gapFraction) / 2.0;
    return {-outer, -inner, inner, outer};
}

inline double fiberWidthGapFromJson(const nlohmann::json& root)
{
    if (!root.contains("width_gap_fraction")) return kDefaultFiberWidthGapFraction;
    const auto& value = root.at("width_gap_fraction");
    if (!value.is_number()) throw std::runtime_error("width_gap_fraction must be numeric");
    const double gap = value.get<double>();
    if (!std::isfinite(gap) || gap < 0 || gap > 1)
        throw std::runtime_error("width_gap_fraction must be finite and between 0 and 1");
    return gap;
}

inline std::optional<cv::Vec3d> displayUnit(cv::Vec3d v)
{
    const double length = cv::norm(v);
    if (!std::isfinite(length) || length < 1e-6) return std::nullopt;
    return v / length;
}

inline std::optional<cv::Vec3d> projectDisplayNormal(cv::Vec3d normal, cv::Vec3d tangent)
{
    const auto n = displayUnit(normal);
    const auto t = displayUnit(tangent);
    if (!n || !t) return std::nullopt;
    return displayUnit(*n - *t * n->dot(*t));
}

inline cv::Vec3d rotateDisplayNormal(cv::Vec3d normal, cv::Vec3d tangent, double radians)
{
    return normal * std::cos(radians) + tangent.cross(normal) * std::sin(radians);
}

struct FiberWidthDragResult {
    cv::Vec3d center;
    cv::Vec3d normal;
    cv::Vec3d edgeAxis;
    double width;
};

// All coordinates/widths use the same units. Handle 0 is the center, -1/+1
// the nominal edges. Delta translates the selected handle. For an edge, the
// original opposite edge defines orientation only; width always stays fixed.
inline std::optional<FiberWidthDragResult> dragFiberWidth(
    cv::Vec3d center, cv::Vec3d normal, cv::Vec3d edgeAxis,
    cv::Vec3d planeNormal, double width, int handle, cv::Vec3d delta)
{
    const auto axis = displayUnit(edgeAxis);
    const auto plane = displayUnit(planeNormal);
    const auto up = displayUnit(normal);
    if (!axis || !plane || !up || !std::isfinite(width) || width < 0 ||
        !std::isfinite(cv::norm(delta))) return std::nullopt;
    delta -= *plane * delta.dot(*plane);
    if (handle == 0) return FiberWidthDragResult{center + delta, *up, *axis, width};
    if (width == 0 || (handle != -1 && handle != 1)) return std::nullopt;
    const cv::Vec3d opposite = center - *axis * (handle * width / 2);
    const cv::Vec3d moved = center + *axis * (handle * width / 2) + delta;
    const cv::Vec3d span = (moved - opposite) * handle;
    const auto newAxis = displayUnit(span);
    if (!newAxis) return std::nullopt;
    const double angle = std::atan2(plane->dot(axis->cross(*newAxis)), axis->dot(*newAxis));
    return FiberWidthDragResult{moved - *newAxis * (handle * width / 2),
        rotateDisplayNormal(*up, *plane, angle), *newAxis, width};
}

inline std::optional<double> displayNormalOffset(cv::Vec3d baseline, cv::Vec3d normal, cv::Vec3d tangent)
{
    const auto a = projectDisplayNormal(baseline, tangent);
    const auto b = projectDisplayNormal(normal, tangent);
    const auto t = displayUnit(tangent);
    if (!a || !b || !t) return std::nullopt;
    return std::atan2(t->dot(a->cross(*b)), a->dot(*b));
}

inline double fiberWidthFromJson(const nlohmann::json& root)
{
    if (!root.contains("width")) return 0.0;
    if (!root.at("width").is_number()) throw std::runtime_error("fiber width must be numeric");
    const double width = root.at("width").get<double>();
    if (!std::isfinite(width) || width < 0) throw std::runtime_error("fiber width must be finite and nonnegative");
    return width;
}

inline std::optional<cv::Vec3d> displayNormalFromJson(const nlohmann::json& control)
{
    if (!control.contains("display_normal")) return std::nullopt;
    const auto& a = control.at("display_normal");
    if (!a.is_array() || a.size() != 3 || !a[0].is_number() ||
        !a[1].is_number() || !a[2].is_number())
        throw std::runtime_error("display_normal must be a three-component direction");
    const auto n = displayUnit({a[0].get<double>(), a[1].get<double>(), a[2].get<double>()});
    if (!n) throw std::runtime_error("display_normal must be finite and nonzero");
    return n;
}

inline cv::Vec3d displayVectorAt(const std::vector<cv::Vec3f>& values, double position)
{
    if (values.empty()) return {};
    position = std::clamp(position, 0.0, double(values.size() - 1));
    const size_t i = size_t(position), j = std::min(i + 1, values.size() - 1);
    const double t = position - double(i);
    return cv::Vec3d(values[i]) * (1 - t) + cv::Vec3d(values[j]) * t;
}

// Windowed tangent stabilizes cross views and normal editing on dense noisy traces.
inline cv::Vec3d displayTangentAt(const std::vector<cv::Vec3f>& points, double position)
{
    if (points.size() < 2) return {};
    const double last = double(points.size() - 1);
    position = std::clamp(position, 0.0, last);
    return displayVectorAt(points, std::min(last, position + 4.0)) -
           displayVectorAt(points, std::max(0.0, position - 4.0));
}

// C1 smooth, bounded interpolation. Zero controls stay exactly zero; shortest
// circular interpolation avoids a full turn at the +/-pi branch cut.
inline double interpolateDisplayOffset(const std::vector<double>& arcs,
                                      const std::vector<double>& angles, double arc)
{
    if (arcs.empty()) return 0;
    if (arc <= arcs.front()) return angles.front();
    if (arc >= arcs.back()) return angles.back();
    const size_t j = size_t(std::upper_bound(arcs.begin(), arcs.end(), arc) - arcs.begin());
    double t = (arc - arcs[j - 1]) / (arcs[j] - arcs[j - 1]);
    t = t * t * (3 - 2 * t);
    return angles[j - 1] + t * std::remainder(angles[j] - angles[j - 1], 2 * std::acos(-1.0));
}
struct FiberDisplayField {
    std::vector<cv::Vec3f> normals;
    std::vector<cv::Vec3d> controlBaselines, controlTangents;
    std::vector<double> controlOffsets;
    std::vector<size_t> resetControls;
};

// New CPs must not introduce a zero-offset knot into a corrected region.
// Keep genuinely uncorrected regions unset, so they continue following Lasagna.
inline std::optional<cv::Vec3d> inheritedFiberDisplayNormal(
    const FiberDisplayField& field, const std::vector<double>& controlArcs,
    double arc, double linePosition)
{
    if (controlArcs.size() != field.controlOffsets.size() ||
        interpolateDisplayOffset(controlArcs, field.controlOffsets, arc) == 0.0)
        return std::nullopt;
    return displayUnit(displayVectorAt(field.normals, linePosition));
}

inline FiberDisplayField fiberDisplayField(
    const std::vector<cv::Vec3f>& points, const std::vector<cv::Vec3f>& normals,
    const std::vector<double>& positions,
    const std::vector<std::optional<cv::Vec3d>>& manualNormals)
{
    FiberDisplayField out;
    out.normals = normals;
    if (points.size() < 2 || points.size() != normals.size() || positions.size() != manualNormals.size())
        return out;
    std::vector<double> arcs(points.size(), 0.0), controlArcs;
    std::vector<cv::Vec3f> tangents(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        if (i) arcs[i] = arcs[i - 1] + cv::norm(points[i] - points[i - 1]);
        tangents[i] = cv::Vec3f(displayTangentAt(points, double(i)));
    }
    for (size_t i = 0; i < positions.size(); ++i) {
        const double p = std::clamp(positions[i], 0.0, double(points.size() - 1));
        const size_t k = size_t(p), j = std::min(k + 1, points.size() - 1);
        controlArcs.push_back(arcs[k] + (p - double(k)) * (arcs[j] - arcs[k]));
        const auto tangent = displayUnit(displayTangentAt(points, p));
        const auto baseline = projectDisplayNormal(displayVectorAt(normals, p), tangent.value_or(cv::Vec3d{}));
        out.controlBaselines.push_back(baseline.value_or(cv::Vec3d{}));
        out.controlTangents.push_back(tangent.value_or(cv::Vec3d{}));
        double angle = 0.0;
        if (manualNormals[i]) {
            const auto projected = projectDisplayNormal(*manualNormals[i], tangent.value_or(cv::Vec3d{}));
            if (!projected) out.resetControls.push_back(i);
            else if (baseline) angle = *displayNormalOffset(*baseline, *projected, *tangent);
        }
        out.controlOffsets.push_back(angle);
    }
    for (size_t i = 0; i < points.size(); ++i) {
        const auto tangent = displayUnit(cv::Vec3d(tangents[i]));
        const auto baseline = projectDisplayNormal(cv::Vec3d(normals[i]), tangent.value_or(cv::Vec3d{}));
        if (tangent && baseline)
            out.normals[i] = cv::Vec3f(rotateDisplayNormal(*baseline, *tangent,
                interpolateDisplayOffset(controlArcs, out.controlOffsets, arcs[i])));
    }
    return out;
}
} // namespace vc::fiber_tracer
