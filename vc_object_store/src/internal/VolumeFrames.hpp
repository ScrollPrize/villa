#pragma once
#include "internal/RecordSql.hpp"
#include "vc_object_store/objects/VolumeTransforms.hpp"
#include "internal/Sqlite.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace vc::objects::detail {
// One coordinate frame declared in a volume: stored * toVolume = volume voxels,
// with toVolume = 1 / scale_factor exactly as geometryScaleToVolume() computes it.
struct VolumeFrame {
    std::string coordinates; // exact declaration, for the coordinates filters
    double toVolume = 1.0;
    VolumeAffine mapping;
    bool registered = false;
};

// How geometry declared in one frame (`coordinates`, a geometryCoordinates
// object) maps to `volume`'s voxels, as volumeFrames maps that frame; nothing
// when the frame is not placed in the volume.
inline std::optional<VolumeAffine> frameMapping(const nlohmann::json& coordinates, const std::string& volume,
                                                const VolumeTransforms* transforms)
{
    const auto& source = coordinates.at("volume_uuid");
    const auto& scale = coordinates.at("scale_factor");
    if (!source.is_string() || !scale.is_number()) return std::nullopt;
    const double toVolume = 1.0 / scale.get<double>();
    if (transforms && !transforms->graph.registrations().empty()) {
        const auto resolution = transforms->resolve(source.get<std::string>(), volume);
        if (resolution.status != vcr::catalog::TransformStatus::Resolved) return std::nullopt;
        return resolution.resolved().forward * VolumeAffine::scale(toVolume);
    }
    if (source.get<std::string>() != volume) return std::nullopt;
    return VolumeAffine::scale(toVolume);
}

inline std::vector<VolumeFrame> volumeFrames(sqlite3* db, const std::string& volume,
                                             const std::optional<std::string>& kind, bool includeRetired,
                                             const VolumeTransforms* transforms = nullptr)
{
    if (transforms && !transforms->graph.registrations().empty()) {
        std::set<std::string> volumes{volume};
        for(const auto& r:transforms->graph.registrations()){volumes.insert(r.source);volumes.insert(r.target);}
        std::vector<VolumeFrame> result;
        for(const auto& source:volumes) {
            const auto resolution=transforms->resolve(source,volume);
            if(resolution.status!=vcr::catalog::TransformStatus::Resolved)continue;
            for(auto frame:volumeFrames(db,source,kind,includeRetired)) {
                frame.mapping=resolution.resolved().forward*VolumeAffine::scale(frame.toVolume);
                frame.registered=true;result.push_back(std::move(frame));
            }
        }
        return result;
    }
    // Seek each distinct scale of the volume through objects_coordinates: one
    // probe per frame, never a scan of the volume's objects. Scales are
    // validated finite positive numbers; the walk stops at any other value.
    std::vector<std::string> kinds;
    if (kind) kinds.push_back(*kind);
    else {
        Statement next(db, "SELECT kind FROM objects INDEXED BY objects_kind WHERE kind>? ORDER BY kind LIMIT 1");
        for (std::string after;;) {
            next.reset(); next.bind(1, after);
            if (!next.row()) break;
            kinds.push_back(after = next.text(0));
        }
    }
    std::set<double> scales;
    Statement next(db, "SELECT o.coordinate_scale FROM objects o INDEXED BY objects_coordinates "
                       "WHERE o.kind=?1 AND o.coordinate_volume=?2 AND o.coordinate_scale>?3" +
                       (includeRetired ? std::string() : " AND " + activeObjectSql("o.id")) +
                       " ORDER BY o.coordinate_scale LIMIT 1");
    for (const auto& k : kinds)
        for (double after = 0;;) {
            next.reset(); next.bind(1, k); next.bind(2, volume); next.bind(3, after);
            if (!next.row()) break;
            const int type = sqlite3_column_type(next.value, 0);
            if (type != SQLITE_INTEGER && type != SQLITE_FLOAT) break;
            const double scale = next.f64(0);
            if (!std::isfinite(scale) || !(scale > after)) break;
            scales.insert(after = scale);
        }
    std::vector<VolumeFrame> frames;
    for (const double scale : scales)
        frames.push_back({nlohmann::json{{"volume_uuid", volume}, {"scale_factor", scale}}.dump(), 1.0 / scale, VolumeAffine::scale(1.0 / scale)});
    return frames;
}

// Exact inverses of the monotonic map stored -> stored * toVolume: the lowest
// stored value mapping to at least v, and the highest mapping to at most v.
inline double lowestStoredAtLeast(double v, double toVolume)
{
    double x = v / toVolume;
    while (x * toVolume < v) x = std::nextafter(x, INFINITY);
    for (double below = std::nextafter(x, -INFINITY); below * toVolume >= v; below = std::nextafter(x, -INFINITY))
        x = below;
    return x;
}
inline double highestStoredAtMost(double v, double toVolume)
{
    double x = v / toVolume;
    while (x * toVolume > v) x = std::nextafter(x, -INFINITY);
    for (double above = std::nextafter(x, INFINITY); above * toVolume <= v; above = std::nextafter(x, INFINITY))
        x = above;
    return x;
}
// A volume-voxel box in stored units: bounds overlap it exactly when their
// mapped bounds overlap the volume box. An axis too thin to contain a stored
// value keeps the one-step interval around it, as conservative overlap allows.
inline std::array<double, 6> storedBox(const std::array<double, 6>& box, double toVolume)
{
    std::array<double, 6> stored{};
    for (size_t axis = 0; axis < 3; ++axis) {
        const double lower = lowestStoredAtLeast(box[axis], toVolume);
        const double upper = highestStoredAtMost(box[axis + 3], toVolume);
        stored[axis] = std::min(lower, upper);
        stored[axis + 3] = std::max(lower, upper);
    }
    return stored;
}
}
