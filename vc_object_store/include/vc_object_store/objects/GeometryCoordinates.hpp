#pragma once

#include "vc_object_store/Uuid.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <optional>

namespace vc::objects {
using CoordinateJson = nlohmann::json;
inline bool hasGeometryCoordinates(const std::string& kind)
{
    return kind == "point_collection" || kind == "surface" || kind == "fiber" || kind == "track" ||
           kind == "umbilicus";
}
// volume_uuid stores the authoritative volume ID verbatim (metadata.json id),
// including catalog IDs such as "20260411134726". It is not an object UUID.
// The other persistent coordinate declaration is the factor
// mapping source-volume voxels to the geometry's annotation coordinates.
inline CoordinateJson geometryCoordinates(CoordinateJson value)
{
    if (!value.is_object())
        throw std::invalid_argument("geometry coordinates must be an object");
    for (const auto& [key, field] : value.items())
        if (key != "volume_uuid" && key != "scale_factor")
            throw std::invalid_argument("unknown geometry coordinate field: " + key);
    if (!value.contains("volume_uuid")) value["volume_uuid"] = nullptr;
    if (!value.contains("scale_factor")) value["scale_factor"] = nullptr;
    if (!value["volume_uuid"].is_null()) {
        if (!value["volume_uuid"].is_string() || value["volume_uuid"].get_ref<const std::string&>().empty())
            throw std::invalid_argument("geometry volume_uuid must be a nonempty volume ID string or null");
    }
    const auto& scale = value["scale_factor"];
    if (!scale.is_null() &&
        (!scale.is_number() || !std::isfinite(scale.get<double>()) || scale.get<double>() <= 0))
        throw std::invalid_argument("geometry scale_factor must be finite and positive or null");
    return value;
}
// The coordinates of geometry stored at `storedUnitsPerVoxel` units per voxel
// of `volume` (its meta.coordinates).
inline CoordinateJson volumeCoordinates(const std::string& volume, double storedUnitsPerVoxel)
{
    return geometryCoordinates({{"volume_uuid", volume}, {"scale_factor", storedUnitsPerVoxel}});
}

inline CoordinateJson mergeGeometryCoordinates(const CoordinateJson& a, const CoordinateJson& b)
{
    auto result = geometryCoordinates(a);
    const auto incoming = geometryCoordinates(b);
    for (const auto* key : {"volume_uuid", "scale_factor"}) {
        if (incoming[key].is_null()) continue;
        if (result[key].is_null()) {
            result[key] = incoming[key];
            continue;
        }
        bool equal = result[key] == incoming[key];
        // JSON's ordinary mixed float/integer comparison can round a large
        // integer. Do not accept conflicting exact scale declarations that way.
        const auto& left = result[key];
        const auto& right = incoming[key];
        if (left.is_number() && right.is_number() &&
            left.is_number_float() != right.is_number_float()) {
            const auto& integer = left.is_number_integer() ? left : right;
            const double floating = (left.is_number_float() ? left : right).get<double>();
            equal = floating < std::ldexp(1.0, 64) && std::floor(floating) == floating &&
                    static_cast<uint64_t>(floating) == integer.get<uint64_t>();
        }
        if (!equal) throw std::invalid_argument(std::string("geometry coordinate conflict: ") + key);
    }
    return result;
}

inline CoordinateJson sourceGeometryCoordinates(const CoordinateJson& source)
{
    if (!source.is_object())
        throw std::invalid_argument("geometry coordinate source must be an object");
    auto result = geometryCoordinates(source.value("coordinates", CoordinateJson::object()));
    CoordinateJson flat = CoordinateJson::object();
    for (const auto* key : {"volume_uuid", "scale_factor"})
        if (source.contains(key)) flat[key] = source[key];
    return mergeGeometryCoordinates(result, flat);
}

inline void eraseSourceGeometryCoordinates(CoordinateJson& source)
{
    for (const auto* key : {"coordinates", "volume_uuid", "scale_factor"}) source.erase(key);
}
// Authoritative object metadata wins over any historical source declarations.
// This is shared by every native exporter; absence never resurrects source data.
inline CoordinateJson geometryTransferMetadata(const CoordinateJson& metadata)
{
    auto source = metadata.value("source", CoordinateJson::object());
    eraseSourceGeometryCoordinates(source);
    source.erase("coordinate_binding");
    source["coordinates"] = geometryCoordinates(metadata.value("coordinates", CoordinateJson::object()));
    if (metadata.contains("coordinate_binding")) source["coordinate_binding"] = metadata.at("coordinate_binding");
    return source;
}

// Convert stored XYZ to voxels of exactly the declared volume. Unknown frames
// remain unknown; names, dimensions, and resolution cannot establish identity.
inline std::optional<double> geometryScaleToVolume(const CoordinateJson& metadata, const std::string& volumeId)
{
    const auto coordinates = geometryCoordinates(metadata.value("coordinates", CoordinateJson::object()));
    if (coordinates["volume_uuid"].is_null() || coordinates["scale_factor"].is_null() ||
        coordinates["volume_uuid"] != volumeId) return std::nullopt;
    const double scale = 1.0 / coordinates["scale_factor"].get<double>();
    if (!std::isfinite(scale) || scale <= 0) throw std::invalid_argument("geometry coordinate conversion is not representable");
    return scale;
}
} // namespace vc::objects
