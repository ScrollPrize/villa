// What a volume store says about itself in its sidecar documents, and what is known
// about it once the Open Data catalog is merged in.
#include "vc_reader/catalog.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/zarr_metadata.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <map>
#include <mutex>
#include <sstream>
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

Json storeMetadataJson(const StoreMetadata& m)
{
    const auto optional = [](const auto& value) { return value ? Json(*value) : Json(); };
    return Json{{"voxel_size_um", optional(m.voxelSizeUm)},
                {"energy_kev", optional(m.energyKeV)},
                {"detector_distance_mm", optional(m.detectorDistanceMm)},
                {"beamline", m.beamline},
                {"scan_name", m.scanName},
                {"acquisition_date", m.acquisitionDate},
                {"value_window", optional(m.valueWindow)},
                {"z_crop", optional(m.zCrop)},
                {"sample_id", m.sampleId},
                {"volume_id", m.volumeId},
                {"scan_id", m.scanId},
                {"name", m.name},
                {"created_at", m.createdAt},
                {"data_format", m.dataFormat},
                {"documents", m.documents}};
}

StoreMetadata storeMetadataFromJson(const Json& json)
{
    StoreMetadata m;
    const auto number = [&](const char* key, std::optional<double>& field) {
        if (const auto it = json.find(key); it != json.end() && !it->is_null())
            field = it->get<double>();
    };
    number("voxel_size_um", m.voxelSizeUm);
    number("energy_kev", m.energyKeV);
    number("detector_distance_mm", m.detectorDistanceMm);
    m.beamline = json.value("beamline", "");
    m.scanName = json.value("scan_name", "");
    m.acquisitionDate = json.value("acquisition_date", "");
    if (const auto it = json.find("value_window"); it != json.end() && !it->is_null())
        m.valueWindow = it->get<std::array<double, 2>>();
    if (const auto it = json.find("z_crop"); it != json.end() && !it->is_null())
        m.zCrop = it->get<std::array<std::int64_t, 2>>();
    m.sampleId = json.value("sample_id", "");
    m.volumeId = json.value("volume_id", "");
    m.scanId = json.value("scan_id", "");
    m.name = json.value("name", "");
    m.createdAt = json.value("created_at", "");
    m.dataFormat = json.value("data_format", "");
    m.documents = json.value("documents", Json::object());
    return m;
}

namespace {
constexpr const char* kKeptMetadata = ".vc_reader_metadata.json";

std::mutex knownMutex;
std::map<std::string, StoreMetadata> known;  // what this process already knows, by store

std::optional<std::string> readFile(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::nullopt;
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

// The catalog (from its cache file alone when offline); nullptr when it is unavailable.
std::shared_ptr<const catalog::OpenDataManifest> catalogManifest(bool offline, bool refresh)
{
    static std::mutex mutex;
    static std::shared_ptr<const catalog::OpenDataManifest> loaded;
    const std::string source(catalog::kDefaultManifestUrl);
    const auto file = catalog::catalogCacheFile(source);
    try {
        if (offline)
            return catalog::cachedManifest(file);
        std::lock_guard lock(mutex);
        if (!loaded || refresh) {
            loaded.reset();
            loaded = std::make_shared<const catalog::OpenDataManifest>(
                catalog::loadCatalog(source, {file, refresh, 3600, 15}).manifest);
        }
        return loaded;
    } catch (const std::exception&) {
        return nullptr;
    }
}
}  // namespace

void forgetVolumeMetadata()
{
    std::lock_guard lock(knownMutex);
    known.clear();
}

StoreMetadata volumeMetadata(const std::string& input, const OpenOptions& options, bool refresh)
{
    auto location = input;
    while (location.size() > 1 && location.back() == '/')
        location.pop_back();
    const bool remote = isRemote(location);
    const bool offline = options.offline && remote;
    if (refresh && offline)
        throw std::invalid_argument("refresh needs the network; it cannot be combined with offline");
    const auto& access = options.access;
    const auto key = remote ? access.cacheUrl(location) : location;
    // The cache folders of a node of this store.
    const auto folders = [&](const std::string& node) {
        auto dirs = cachedNodeDirs(options.cacheDirs, remote ? access.cacheUrl(node) : node);
        if (options.vc3dVolumeDir && node == location)
            dirs.insert(dirs.begin(), *options.vc3dVolumeDir);
        return dirs;
    };
    const auto kept = [&]() -> std::optional<StoreMetadata> {
        {
            std::lock_guard lock(knownMutex);
            if (const auto it = known.find(key); it != known.end())
                return it->second;
        }
        for (const auto& directory : folders(location))
            if (const auto text = readFile(directory / kKeptMetadata)) {
                try {
                    auto metadata = storeMetadataFromJson(Json::parse(*text));
                    std::lock_guard lock(knownMutex);
                    known[key] = metadata;
                    return metadata;
                } catch (const std::exception&) {
                    // A damaged copy: read the sources again.
                }
            }
        return std::nullopt;
    };
    if (remote && !refresh)
        if (auto metadata = kept())
            return *metadata;

    bool failed = false;
    const auto source = options.metadata ? options.metadata : metadataReader(access);
    const MetadataReader read = [&](const std::string& node, const std::string& name) -> std::optional<std::string> {
        if (remote && !refresh)
            for (const auto& directory : folders(node))
                if (auto text = readFile(directory / name))
                    return text;
        if (offline)
            return std::nullopt;
        try {
            return source(node, name);
        } catch (...) {
            failed = true;  // not the same as a store without this document
            throw;
        }
    };
    auto metadata = storeMetadata(location, read);
    try {
        if (const auto attributes = read(location, ".zattrs"))
            metadata.documents[".zattrs"] = Json::parse(*attributes);
        else if (const auto node = read(location, "zarr.json"))  // v3 keeps them in the node's own document
            metadata.documents[".zattrs"] = Json::parse(*node).value("attributes", Json::object());
    } catch (const std::exception&) {
        // Malformed says nothing; a failed read is noted above.
    }
    // Stores outside the public channels are not in the catalog, so it is not loaded for them.
    bool answered = true;
    if (remote && isPublicLocation(location)) {
        const auto manifest = catalogManifest(offline, refresh);
        answered = manifest != nullptr;
        if (const auto entry = manifest ? manifest->volumeMetadataAt(location) : std::nullopt) {
            if (!metadata.voxelSizeUm)
                metadata.voxelSizeUm = entry->voxelSizeUm;
            if (!metadata.energyKeV)
                metadata.energyKeV = entry->energyKeV;
            if (!metadata.detectorDistanceMm)
                metadata.detectorDistanceMm = entry->detectorDistanceMm;
            metadata.sampleId = entry->sampleId;
            metadata.volumeId = entry->volumeId;
            metadata.scanId = entry->scanId;
            metadata.name = entry->name;
            metadata.createdAt = entry->createdAt;
            metadata.dataFormat = entry->dataFormat;
            metadata.documents["catalog"] = entry->raw;
        }
    }
    // Keep it only once both sources have answered, so a later open can fill what is missing.
    if (remote && answered && !failed && !offline) {
        {
            std::lock_guard lock(knownMutex);
            known[key] = metadata;
        }
        if (options.cacheWrites) {
            auto candidates = folders(location);
            for (const auto& root : cacheRoots(options.cacheDirs))
                candidates.push_back(decodedCachePath(root, key));
            const auto data = storeMetadataJson(metadata).dump();
            for (const auto& candidate : candidates) {
                try {
                    atomicWrite(candidate / kKeptMetadata, data);
                    break;
                } catch (const std::exception&) {
                    // Read-only: try the next; this process still remembers it.
                }
            }
        }
    } else if (refresh) {
        if (auto previous = kept())
            return *previous;
    }
    return metadata;
}

}  // namespace vcr::zarr
