#pragma once
#include "internal/RecordSql.hpp"
#include "internal/Sqlite.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects::detail {
// One coordinate frame declared in a volume: stored * toVolume = volume voxels,
// with toVolume = 1 / scale_factor exactly as geometryScaleToVolume() computes it.
struct VolumeFrame {
    std::string coordinates; // exact declaration, for the coordinates filters
    double toVolume = 1.0;
};

inline std::vector<VolumeFrame> volumeFrames(sqlite3* db, const std::string& volume,
                                             const std::optional<std::string>& kind, bool includeRetired)
{
    Statement rows(db, "SELECT DISTINCT json_extract(o.meta,'$.coordinates.scale_factor') FROM objects o "
                       "WHERE json_extract(o.meta,'$.coordinates.volume_uuid')=?1 AND (?2 IS NULL OR o.kind=?2)" +
                       (includeRetired ? std::string() : " AND " + activeObjectSql("o.id")));
    rows.bind(1, volume);
    rows.bind(2, kind);
    std::vector<VolumeFrame> frames;
    while (rows.row()) {
        if (rows.isNull(0)) continue;
        const double scale = rows.f64(0);
        if (!std::isfinite(scale) || !(scale > 0)) continue;
        frames.push_back({nlohmann::json{{"volume_uuid", volume}, {"scale_factor", scale}}.dump(), 1.0 / scale});
    }
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
