// Fiber JSON adapter for the shared line-to-patch linker.
#include "linking/LinePatch.hpp"
#include "formats/FiberInput.hpp"
#include "vc/core/util/Uuid.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <nlohmann/json.hpp>

namespace vc::graph_links {
namespace {
std::vector<cv::Vec3d> parsePoints(const nlohmann::json& root, const char* key)
{
    std::vector<cv::Vec3d> out;
    for (const auto& p : objects::detail::fiberPoints(root, key)) out.emplace_back(p[0], p[1], p[2]);
    return out;
}

std::optional<detail::LineData> loadFiber(const std::filesystem::path& file, double scale, DetectorReport& report)
{
    try {
        std::ifstream in(file);
        if (!in) throw std::runtime_error("cannot read file");
        std::ostringstream ss;
        ss << in.rdbuf();
        const auto root = nlohmann::json::parse(ss.str());
        if (!root.is_object() || !root.contains("uuid") || !root.contains("line_points"))
            return std::nullopt;  // not a vc3d_fiber file
        detail::LineData fiber;
        fiber.path = file;
        fiber.uuid = root.value("uuid", "");
        if (!vc::util::isUuid(fiber.uuid)) {
            report.problems.push_back(file.filename().string() + ": missing/invalid uuid, skipped");
            return std::nullopt;
        }
        const auto sourceLine = objects::detail::fiberPoints(root, "line_points");
        fiber.object.fiber = objects::FiberDetail{};
        fiber.object.fiber->length = objects::detail::fiberPolylineLength(sourceLine);
        for (const auto& p : sourceLine) fiber.line.emplace_back(p[0], p[1], p[2]);
        fiber.controls = parsePoints(root, "control_points");
        for (auto* points : {&fiber.line, &fiber.controls})
            for (auto& p : *points) {
                p *= scale;
                if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
                    throw std::runtime_error("scaled fiber coordinate is non-finite");
            }
        if (fiber.line.size() < 2) {
            report.problems.push_back(file.filename().string() + ": fewer than two line points, skipped");
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
        report.problems.push_back(file.filename().string() + ": unreadable fiber JSON: " + e.what());
        return std::nullopt;
    }
}

} // namespace

DetectorReport linkFibersToPatches(objects::ObjectStore& db, const std::filesystem::path& fibersDir,
    const std::filesystem::path& patchesDir, const std::filesystem::path& ctVolume,
    const FiberPatchParams& params, const std::atomic_bool* cancel)
{
    detail::validateLinePatchParams(params, params.fiberScale);
    if (cancel && cancel->load()) {
        DetectorReport report;
        report.problems.push_back("fiber-patch: cancelled; no objects or edges committed");
        return report;
    }
    util::reportProgress(params.progress, "scanning fiber directory");
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(fibersDir))
        if (entry.is_regular_file() && entry.path().extension() == ".json") files.push_back(entry.path());
    std::sort(files.begin(), files.end());
    auto load = [&](size_t index, DetectorReport& report) -> std::optional<detail::LineData> {
        auto fiber = loadFiber(files[index], params.fiberScale, report);
        if (!fiber) return {};
        fiber->stripCacheFile = fibersDir / ".strip-cache" / (fiber->uuid + ".cbor");
        fiber->rebuildStrip = params.rebuildStrips;
        auto& object = fiber->object;
        object.uuid = fiber->uuid;
        object.kind = "fiber";
        object.name = fiber->path.filename().string();
        object.storageKind = "file";
        object.storageRef = std::filesystem::absolute(fiber->path).lexically_relative(
            std::filesystem::absolute(db.path().parent_path())).string();
        object.dataGeneration = 1;
        auto& detail = *object.fiber;
        detail.controlPointCount = int64_t(fiber->controls.size());
        detail.linePointCount = int64_t(fiber->line.size());
        return fiber;
    };
    return detail::linkLinesToPatches(db, files.size(), load, patchesDir, ctVolume,
        params, params.fiberScale, kFiberPatchDetector, 3, cancel);
}
} // namespace vc::graph_links
