#include "linking/LinePatch.hpp"
#include "linking/TrackInput.hpp"
#include <cmath>
#include <stdexcept>

namespace vc::object_links {
DetectorReport linkTracksToPatches(objects::ObjectStore& db, const std::filesystem::path& tracksPath,
    const std::filesystem::path& patchesDir, const std::filesystem::path& ctVolume,
    const TrackPatchParams& params, const std::atomic_bool* cancel)
{
    detail::validateLinePatchParams(params, params.trackScale);
    try {
        const auto input = detail::loadTrackInput(db, tracksPath, params.sourceIds, params.trackUuids, cancel, params.progress);
        const PackedTrackStore store(input.packedPath);
        auto load = [&](size_t index, DetectorReport& report) -> std::optional<detail::LineData> {
            const auto& object = input.objects[index];
            try {
                const auto row = detail::trackRow(store, object);
                const auto begin = store.offsets()[row], end = store.offsets()[row+1];
                if (end - begin < 2) throw std::runtime_error("fewer than two points");
                detail::LineData line;
                line.path = input.packedPath / object.track->externalKey;
                line.uuid = object.uuid;
                line.object = object;
                line.lastLine = static_cast<size_t>(end - begin - 1);
                line.line.reserve(static_cast<size_t>(end - begin));
                line.arclength.reserve(static_cast<size_t>(end - begin));
                for (auto i = begin; i < end; ++i) {
                    if ((i & 1023) == 0 && cancel && cancel->load())
                        throw std::runtime_error("track-patch linking cancelled");
                    const auto p = store.coordinates().subspan(static_cast<size_t>(i) * 3, 3);
                    for (int32_t coordinate : p)
                        if (coordinate < 0 || coordinate >= (1 << 20))
                            throw std::runtime_error("voxel coordinates must be in [0, 2^20)");
                    const cv::Vec3d xyz = cv::Vec3d(p[2], p[1], p[0]) * params.trackScale;
                    if (!std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2]))
                        throw std::runtime_error("scaled coordinate is non-finite");
                    line.arclength.push_back(line.line.empty() ? 0.0
                        : line.arclength.back() + cv::norm(xyz - line.line.back()));
                    line.line.push_back(xyz);
                }
                if (!(line.arclength.back() > 0) || !std::isfinite(line.arclength.back()))
                    throw std::runtime_error("track has zero or non-finite length");
                line.object.track->length = detail::trackPolylineLength(store, row, cancel);
                line.object.track->family = detail::trackFamily(store, row);
                return line;
            } catch (const std::exception& error) {
                if (cancel && cancel->load()) throw;
                report.problems.push_back("track " + object.track->externalKey + ": " + error.what() + ", skipped");
                return {};
            }
        };
        return detail::linkLinesToPatches(db, input.objects.size(), load, patchesDir, ctVolume,
            params, params.trackScale, kTrackPatchDetector, 1, cancel);
    } catch (const std::exception& error) {
        DetectorReport report;
        report.problems.push_back(std::string("track-patch: no objects or edges committed: ") + error.what());
        return report;
    }
}
} // namespace vc::object_links
