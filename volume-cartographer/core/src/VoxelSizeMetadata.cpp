#include "vc/core/util/VoxelSizeMetadata.hpp"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace vc::metadata
{

namespace
{

// Follow an exact path of object keys. Returns nullopt as soon as a key is
// missing or a level is not an object, so a caller never has to pre-check the
// shape of a foreign document.
//
// The result is a copy, deliberately. utils::Json::operator[] hands out
// references into a bounded thread-local cache that evicts its oldest entries,
// so a reference held across a loop that itself indexes into the document is
// eventually left dangling. These documents are small.
std::optional<utils::Json> walk(const utils::Json& root,
                                std::initializer_list<std::string_view> keys)
{
    const utils::Json* current = &root;
    for (const std::string_view key : keys) {
        const std::string name(key);
        if (!current->is_object() || !current->contains(name))
            return std::nullopt;
        current = &(*current)[name];
    }
    return *current;
}

// Accept a number, or a string holding one. Several of these documents store
// numbers as strings -- `source.resolution` is `"2"`, not `2` -- and rejecting
// those would lose exactly the field that makes a derived store resolvable.
// Booleans are excluded: JSON true is not the number 1 here.
std::optional<double> asNumber(const utils::Json& value)
{
    if (value.is_boolean())
        return std::nullopt;
    if (value.is_number())
        return value.get_double();
    if (!value.is_string())
        return std::nullopt;

    const std::string text = value.get_string();
    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(text, &consumed);
        while (consumed < text.size() &&
               std::isspace(static_cast<unsigned char>(text[consumed])) != 0) {
            ++consumed;
        }
        if (consumed == text.size())
            return parsed;
    } catch (...) {
    }
    return std::nullopt;
}

std::optional<double> positiveNumber(const utils::Json& value)
{
    const auto number = asNumber(value);
    if (!number || !std::isfinite(*number) || *number <= 0.0)
        return std::nullopt;
    return number;
}

// A pyramid level must be an exact non-negative integer. The upper bound
// matches the 0..31 range the zarr opener enforces on dataset paths, and keeps
// the 2^level factor below from overflowing.
std::optional<int> asPyramidLevel(const utils::Json& value)
{
    const auto number = asNumber(value);
    if (!number || !std::isfinite(*number) || *number < 0.0 || *number > 31.0)
        return std::nullopt;
    const auto level = static_cast<int>(*number);
    if (static_cast<double>(level) != *number)
        return std::nullopt;
    return level;
}

// The ESRF/BM18 acquisition record, either under a "scan" wrapper or at the
// root of the document. `samplePixelSize` is in millimeters.
std::optional<double> detectorVoxelSize(const utils::Json& root, int sourceLevel)
{
    for (const bool scanWrapper : {true, false}) {
        const auto value =
            scanWrapper
                ? walk(root, {"scan", "tomo", "acquisition", "detector", "samplePixelSize"})
                : walk(root, {"tomo", "acquisition", "detector", "samplePixelSize"});
        if (!value)
            continue;
        const auto millimeters = positiveNumber(*value);
        if (!millimeters)
            continue;

        const double micrometers =
            *millimeters * 1000.0 * static_cast<double>(std::uint64_t{1} << sourceLevel);
        if (!std::isfinite(micrometers) || micrometers <= 0.0)
            return std::nullopt;
        return micrometers;
    }
    return std::nullopt;
}

// A voxel size the document states outright, at the top level only.
std::optional<double> explicitVoxelSize(const utils::Json& doc)
{
    if (!doc.is_object())
        return std::nullopt;
    for (const char* key : {"voxelsize", "voxel_size_um", "voxelSizeUm", "pixel_size_um",
                            "pixelSizeUm", "resolution_um"}) {
        if (!doc.contains(key))
            continue;
        if (const auto micrometers = positiveNumber(doc[key]))
            return micrometers;
    }
    return std::nullopt;
}

} // namespace

std::optional<double> voxelSizeFromStoreMetadata(const utils::Json& doc)
{
    if (!doc.is_object())
        return std::nullopt;

    if (auto explicitSize = explicitVoxelSize(doc))
        return explicitSize;

    // A source volume carries its own scan record.
    if (auto direct = detectorVoxelSize(doc, 0))
        return direct;

    // A derived store -- a surface or ink prediction -- records the volume it
    // ran on under `source`: that volume's scan record, and the pyramid level
    // of it that the run consumed.
    const auto source = walk(doc, {"source"});
    if (!source)
        return std::nullopt;
    const auto sourceMetadata = walk(*source, {"metadata"});
    if (!sourceMetadata)
        return std::nullopt;

    // The level is what converts the source volume's pixel size into this
    // store's, so it is not optional. Assuming 0 would report a voxel size too
    // small by whatever factor the run really used, and nothing downstream
    // could tell. Every derived store the catalog publishes states it.
    if (!source->is_object() || !source->contains("resolution"))
        return std::nullopt;
    const auto sourceLevel = asPyramidLevel((*source)["resolution"]);
    if (!sourceLevel)
        return std::nullopt;
    return detectorVoxelSize(*sourceMetadata, *sourceLevel);
}

} // namespace vc::metadata
