#include "linking/LinePatch.hpp"
#include "linking/StoredGeometry.hpp"
#include <cmath>
#include <stdexcept>

namespace vc::object_links {
DetectorReport linkTracksToPatches(objects::ObjectStore& db, const std::filesystem::path& ctVolume,
    const TrackPatchParams& params, const std::atomic_bool* cancel)
{
    detail::validateLinePatchParams(params, params.trackScale);
    try {
        detail::StoredGeometry inputs(db);
        const auto tracks = inputs.select("track", params.trackUuids);
        auto load = [&](size_t index, DetectorReport& report) -> std::optional<detail::LineData> {
            const auto& object = tracks[index];
            try {
                const auto geometry = inputs.load(object);
                const auto points = geometry.values<int32_t>("points");
                const size_t count = points.size() / 3;
                if (count < 2) throw std::runtime_error("fewer than two points");
                detail::LineData line;
                line.uuid = object.uuid;
                line.object = object;
                line.lastLine = count - 1;
                line.line.reserve(count);
                line.arclength.reserve(count);
                for (size_t i = 0; i < count; ++i) {
                    if ((i & 1023) == 0 && cancel && cancel->load())
                        throw std::runtime_error("track-patch linking cancelled");
                    const cv::Vec3d xyz = cv::Vec3d(points[3*i], points[3*i+1], points[3*i+2]) * params.trackScale;
                    for (int a = 0; a < 3; ++a)
                        if (points[3*i+a] < 0 || points[3*i+a] >= (1 << 20) || !std::isfinite(xyz[a]))
                            throw std::runtime_error("invalid scaled track coordinate");
                    line.arclength.push_back(line.line.empty() ? 0.0
                        : line.arclength.back() + cv::norm(xyz - line.line.back()));
                    line.line.push_back(xyz);
                }
                if (!(line.arclength.back() > 0) || !std::isfinite(line.arclength.back()))
                    throw std::runtime_error("track has zero or non-finite length");
                return line;
            } catch (const std::exception& error) {
                if (cancel && cancel->load()) throw;
                report.problems.push_back("track " + object.uuid + ": " + error.what() + ", skipped");
                return {};
            }
        };
        return detail::linkLinesToPatches(db, inputs, tracks.size(), load, ctVolume,
            params, params.trackScale, kTrackPatchDetector, 1, cancel);
    } catch (const std::exception& error) {
        DetectorReport report;
        report.problems.push_back(std::string("track-patch: no objects or edges committed: ") + error.what());
        return report;
    }
}
} // namespace vc::object_links
