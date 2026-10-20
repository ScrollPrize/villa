#include "formats/GeometryInput.hpp"
#include "geometry/ArrayBytes.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "formats/TrackInput.hpp"
#include <cmath>
namespace vc::objects::formats {
using Json = nlohmann::json;
namespace fs = std::filesystem;
Prepared track(const PackedTrackStore& store, size_t row, ObjectRecord object)
{
    Prepared p;
    p.object = std::move(object); p.geometry.kind = "track";
    const auto start = store.offsets()[row], count = store.offsets()[row + 1] - start;
    std::vector<int32_t> xyz;
    xyz.reserve(static_cast<size_t>(count) * 3);
    const auto raw = store.coordinates();
    for (int64_t i = 0; i < count; ++i) {
        for (int axis = 2; axis >= 0; --axis) xyz.push_back(raw[3 * (start + i) + axis]);
    }

    p.geometry.arrays.push_back(detail::scalarArray<int32_t>("points", "<i4", {count, 3}, xyz));
    p.object.track->pointCount = count; p.object.track->length = detail::trackPolylineLength(store, row, nullptr);
    p.object.track->family = detail::trackFamily(store, row);
    p.source = Json::object();
    return p;
}
}
