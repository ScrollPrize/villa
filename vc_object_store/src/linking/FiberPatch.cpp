// Stored fiber adapter for the shared line-to-patch linker.
#include "linking/LinePatch.hpp"
#include "linking/StoredGeometry.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace vc::object_links {
namespace {
std::optional<detail::LineData> loadFiber(detail::StoredGeometry& inputs,
    const objects::ObjectRecord& object, double scale, DetectorReport& report)
{
    try {
        const auto geometry = inputs.load(object);
        detail::LineData fiber;
        fiber.uuid = object.uuid;
        fiber.object = object;
        auto points = [&](const char* name, auto& output) {
            const auto xyz = geometry.values<double>(name);
            for (size_t i = 0; i < xyz.size(); i += 3) {
                const cv::Vec3d p = cv::Vec3d(xyz[i], xyz[i+1], xyz[i+2]) * scale;
                for (int a = 0; a < 3; ++a)
                    if (!std::isfinite(p[a])) throw std::runtime_error("scaled fiber coordinate is non-finite");
                output.push_back(p);
            }
        };
        points("line_points", fiber.line);
        points("control_points", fiber.controls);
        if (fiber.line.size() < 2) {
            report.problems.push_back(object.uuid + ": fewer than two line points, skipped");
            return std::nullopt;
        }
        fiber.arclength.assign(fiber.line.size(), 0.0);
        for (size_t i = 1; i < fiber.line.size(); ++i)
            fiber.arclength[i] = fiber.arclength[i - 1] + cv::norm(fiber.line[i] - fiber.line[i - 1]);
        if (!(fiber.arclength.back() > 0) || !std::isfinite(fiber.arclength.back()))
            throw std::runtime_error("fiber has zero or non-finite length");
        fiber.lastLine = fiber.line.size() - 1;
        if (!fiber.controls.empty()) {
            // Project endpoint controls onto the ordered polyline. Testing
            // arclength (rather than XYZ bounds) also works for curved fibers.
            auto projectArc = [&](const cv::Vec3d& control) {
                double bestDistance = std::numeric_limits<double>::infinity(), bestArc = 0;
                for (size_t i = 1; i < fiber.line.size(); ++i) {
                    const auto delta = fiber.line[i] - fiber.line[i-1];
                    const double length2 = delta.dot(delta);
                    const double t = length2 > 0
                        ? std::clamp((control - fiber.line[i-1]).dot(delta) / length2, 0.0, 1.0) : 0;
                    const auto offset = fiber.line[i-1] + t * delta - control;
                    const double distance = offset.dot(offset);
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        bestArc = fiber.arclength[i-1] + t * (fiber.arclength[i] - fiber.arclength[i-1]);
                    }
                }
                return bestArc;
            };
            const auto a = projectArc(fiber.controls.front()), b = projectArc(fiber.controls.back());
            const auto first = std::lower_bound(fiber.arclength.begin(), fiber.arclength.end(), std::min(a, b));
            const auto past = std::upper_bound(fiber.arclength.begin(), fiber.arclength.end(), std::max(a, b));
            if (past - first < 2 || *(past - 1) <= *first)
                throw std::runtime_error("fewer than two distinct line points between endpoint controls");
            fiber.firstLine = static_cast<size_t>(first - fiber.arclength.begin());
            fiber.lastLine = static_cast<size_t>(past - fiber.arclength.begin() - 1);
        }
        return fiber;
    } catch (const std::exception& e) {
        report.problems.push_back(object.uuid + ": invalid stored fiber: " + e.what());
        return std::nullopt;
    }
}

} // namespace

DetectorReport linkFibersToPatches(objects::ObjectStore& db, const std::filesystem::path& ctVolume,
    const FiberPatchParams& params, const std::atomic_bool* cancel)
{
    detail::validateLinePatchParams(params, params.fiberScale);
    detail::StoredGeometry inputs(db);
    const auto fibers = inputs.select("fiber", params.fiberUuids);
    auto load = [&](size_t index, DetectorReport& report) -> std::optional<detail::LineData> {
        auto fiber = loadFiber(inputs, fibers[index], params.fiberScale, report);
        if (fiber && !params.cacheDirectory.empty()) {
            fiber->stripCacheFile = params.cacheDirectory / (fiber->uuid + ".cbor");
            fiber->rebuildStrip = params.rebuildStrips;
        }
        return fiber;
    };
    return detail::linkLinesToPatches(db, inputs, fibers.size(), load, ctVolume,
        params, params.fiberScale, kFiberPatchDetector, 3, cancel);
}
} // namespace vc::object_links
