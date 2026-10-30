// What a volume store says about itself in its sidecar documents.
#include "vc_reader/zarr.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <string>
#include <string_view>

namespace vcr::zarr {

namespace {

// The value at an exact path of object keys. Foreign documents have any shape.
const Json* walk(const Json& root, std::initializer_list<std::string_view> keys)
{
    const Json* current = &root;
    for (const auto key : keys) {
        if (!current->is_object())
            return nullptr;
        const auto found = current->find(std::string(key));
        if (found == current->end())
            return nullptr;
        current = &*found;
    }
    return current;
}

// A number, or a string holding one (`source.resolution` is "2" in several stores).
std::optional<double> asNumber(const Json& value)
{
    if (value.is_number())
        return value.get<double>();
    if (!value.is_string())
        return std::nullopt;
    const auto text = value.get<std::string>();
    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(text, &consumed);
        while (consumed < text.size() && std::isspace(static_cast<unsigned char>(text[consumed])))
            ++consumed;
        if (consumed == text.size())
            return parsed;
    } catch (const std::exception&) {
    }
    return std::nullopt;
}

std::optional<double> positiveNumber(const Json& value)
{
    const auto number = asNumber(value);
    if (!number || !std::isfinite(*number) || *number <= 0.0)
        return std::nullopt;
    return number;
}

// An exact pyramid level; the bound keeps 2^level representable.
std::optional<int> asPyramidLevel(const Json& value)
{
    const auto number = asNumber(value);
    if (!number || !std::isfinite(*number) || *number < 0.0 || *number > 31.0)
        return std::nullopt;
    const auto level = static_cast<int>(*number);
    if (static_cast<double>(level) != *number)
        return std::nullopt;
    return level;
}

// The ESRF/BM18 acquisition record, under a "scan" wrapper or at the root.
// samplePixelSize is in millimeters.
std::optional<double> detectorVoxelSize(const Json& root, int sourceLevel)
{
    for (const auto* value : {walk(root, {"scan", "tomo", "acquisition", "detector", "samplePixelSize"}),
                              walk(root, {"tomo", "acquisition", "detector", "samplePixelSize"})}) {
        if (!value)
            continue;
        const auto millimeters = positiveNumber(*value);
        if (!millimeters)
            continue;
        const double micrometers = *millimeters * 1000.0 * static_cast<double>(std::uint64_t{1} << sourceLevel);
        if (!std::isfinite(micrometers) || micrometers <= 0.0)
            return std::nullopt;
        return micrometers;
    }
    return std::nullopt;
}

}  // namespace

std::optional<double> voxelSizeUm(const Json& document)
{
    if (!document.is_object())
        return std::nullopt;
    for (const auto* key : {"voxelsize", "voxel_size_um", "voxelSizeUm", "pixel_size_um", "pixelSizeUm", "resolution_um"})
        if (const auto* value = walk(document, {key}))
            if (const auto micrometers = positiveNumber(*value))
                return micrometers;
    if (const auto* value = walk(document, {"scan", "voxelsize"}))
        if (const auto micrometers = positiveNumber(*value))
            return micrometers;
    if (const auto direct = detectorVoxelSize(document, 0))
        return direct;
    // A derived store names the volume it ran on and the pyramid level it consumed.
    // The level converts that volume's pixel size into this store's, so it is required.
    const auto* metadata = walk(document, {"source", "metadata"});
    const auto* resolution = walk(document, {"source", "resolution"});
    if (!metadata || !resolution)
        return std::nullopt;
    const auto level = asPyramidLevel(*resolution);
    if (!level)
        return std::nullopt;
    return detectorVoxelSize(*metadata, *level);
}

namespace {

const Json* acquisitionOf(const Json& document)
{
    if (const auto* scanned = walk(document, {"scan", "tomo", "acquisition"}))
        return scanned;
    return walk(document, {"tomo", "acquisition"});
}

void fillFrom(const Json& document, StoreMetadata& out)
{
    if (!out.voxelSizeUm)
        out.voxelSizeUm = voxelSizeUm(document);
    const auto text = [](const Json* value, std::string& field) {
        if (field.empty() && value && value->is_string())
            field = value->get<std::string>();
    };
    if (const auto* acquisition = acquisitionOf(document)) {
        if (const auto* energy = walk(*acquisition, {"energy"}); energy && !out.energyKeV)
            out.energyKeV = positiveNumber(*energy);
        if (const auto* distance = walk(*acquisition, {"sampleDetectorDistance"}); distance && !out.detectorDistanceMm)
            out.detectorDistanceMm = positiveNumber(*distance);
        text(walk(*acquisition, {"beamline"}), out.beamline);
        text(walk(*acquisition, {"scanRadix"}), out.scanName);
        text(walk(*acquisition, {"date"}), out.acquisitionDate);
    }
    text(walk(document, {"scan_name"}), out.scanName);
    const auto* low = walk(document, {"zarr_export", "target_window_f32_min"});
    const auto* high = walk(document, {"zarr_export", "target_window_f32_max"});
    if (low && high && !out.valueWindow) {
        const auto a = asNumber(*low), b = asNumber(*high);
        if (a && b)
            out.valueWindow = std::array<double, 2>{*a, *b};
    }
    const auto* start = walk(document, {"zarr_export", "z_crop_start"});
    const auto* end = walk(document, {"zarr_export", "z_crop_end"});
    if (start && end && !out.zCrop) {
        const auto a = asNumber(*start), b = asNumber(*end);
        if (a && b)
            out.zCrop = std::array<std::int64_t, 2>{std::int64_t(*a), std::int64_t(*b)};
    }
}

}  // namespace

StoreMetadata storeMetadata(const std::string& input, const MetadataReader& reader)
{
    const auto read = reader ? reader : metadataReader(Access{});
    auto location = input;
    while (location.size() > 1 && location.back() == '/')
        location.pop_back();
    std::vector<std::string> nodes{location};
    // The documents sit at the store's root, above a numbered level.
    const auto slash = location.rfind('/');
    if (slash != std::string::npos && slash + 1 < location.size() &&
        std::all_of(location.begin() + std::ptrdiff_t(slash + 1), location.end(),
                    [](unsigned char c) { return std::isdigit(c) != 0; }))
        nodes.push_back(location.substr(0, slash));
    StoreMetadata out;
    for (const auto& node : nodes) {
        for (const auto* name : {"meta.json", "metadata.json"}) {
            // A missing, unreadable or malformed document says nothing about the store.
            try {
                if (const auto text = read(node, name)) {
                    auto document = Json::parse(*text);
                    fillFrom(document, out);
                    out.documents[name] = std::move(document);
                }
            } catch (const std::exception&) {
            }
        }
        if (!out.documents.empty())
            break;
    }
    return out;
}

}  // namespace vcr::zarr
