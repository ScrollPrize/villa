// Stored fiber adapter for the shared line-to-surface linker.
#include "linking/LineSurface.hpp"
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
        fiber.sourceScale = scale;
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
            detail::reportDiagnostic(report, "load", object.uuid, "fewer than two line points",
                object.uuid + ": fewer than two line points, skipped");
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
        detail::reportFailure(report, "load", object.uuid, e.what(), object.uuid + ": invalid stored fiber: " + e.what());
        return std::nullopt;
    }
}

} // namespace

DetectorReport linkFibersToSurfaces(objects::ObjectStore& db, const std::filesystem::path& ctVolume,
    const FiberSurfaceParams& params, const std::atomic_bool* cancel)
{
    detail::validateLineSurfaceParams(params, params.fiberScale);
    detail::StoredGeometry inputs(db);
    const auto selected = inputs.select("fiber", params.fiberUuids);
    DetectorReport validation;
    const auto scales = detail::coordinateScales(selected, params, validation, params.fiberScale);
    std::optional<detail::LineIncremental> incremental;
    auto fibers = selected;
    if (params.newUuids) {
        // New fibers load the selected surfaces within contact proximity of
        // their stored bounds; new surfaces, the fibers within it of theirs.
        const auto surfaces = inputs.select("surface", params.surfaceUuids);
        incremental.emplace();
        incremental->fresh = *detail::newAmong(params.newUuids, {&selected, &surfaces}, kFiberSurfaceDetector);
        std::set<std::string> allSurfaces, allFibers, wantedSurfaces, wantedFibers;
        for (const auto& object : surfaces) allSurfaces.insert(object.uuid);
        for (const auto& object : selected) allFibers.insert(object.uuid);
        auto near = [&](const objects::ObjectRecord& object, double scale, const std::string& kind,
                        double otherScale, const std::set<std::string>& among, std::set<std::string>& into) {
            const auto box = inputs.workingBounds(object, params, scale);
            if (!box) { into.insert(among.begin(), among.end()); return; }  // no stored bounds: all
            for (const auto& uuid : inputs.objectsNear(kind, *box, params.proximity, params, otherScale))
                if (among.contains(uuid)) into.insert(uuid);
        };
        for (const auto& object : selected)
            if (incremental->fresh.contains(object.uuid)) {
                wantedFibers.insert(object.uuid);
                near(object, params.fiberScale, "surface", params.surfaceScale, allSurfaces, wantedSurfaces);
            }
        for (const auto& object : surfaces)
            if (incremental->fresh.contains(object.uuid)) {
                wantedSurfaces.insert(object.uuid);
                near(object, params.surfaceScale, "fiber", params.fiberScale, allFibers, wantedFibers);
            }
        for (const auto& object : surfaces) {
            incremental->selected.push_back(object.uuid);
            if (wantedSurfaces.contains(object.uuid)) incremental->surfaces.push_back(object);
        }
        for (const auto& object : selected) incremental->selected.push_back(object.uuid);
        std::erase_if(fibers, [&](const objects::ObjectRecord& object) { return !wantedFibers.contains(object.uuid); });
    }
    auto load = [&](size_t index, DetectorReport& report) -> std::optional<detail::LineData> {
        auto fiber = loadFiber(inputs, fibers[index], scales.at(fibers[index].uuid), report);
        if (fiber && !params.cacheDirectory.empty()) {
            fiber->stripCacheFile = params.cacheDirectory / (fiber->uuid + ".cbor");
            fiber->rebuildStrip = params.rebuildStrips;
        }
        return fiber;
    };
    return detail::linkLinesToSurfaces(db, inputs, fibers.size(), load, ctVolume,
        params, params.fiberScale, kFiberSurfaceDetector, kFiberSurfaceVersion, cancel, std::move(validation),
        incremental ? &*incremental : nullptr);
}
} // namespace vc::object_links
