#include "OpenDataManifest.hpp"

#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>

namespace vc3d::opendata {
namespace {

std::string lowerCopy(std::string_view value)
{
    std::string out(value);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

bool iequals(std::string_view a, std::string_view b)
{
    return lowerCopy(a) == lowerCopy(b);
}

bool containsInsensitive(std::string_view haystack, std::string_view needle)
{
    return lowerCopy(haystack).find(lowerCopy(needle)) != std::string::npos;
}

std::optional<std::string> stringValue(const nlohmann::json& obj,
                                       std::initializer_list<std::string_view> keys)
{
    if (!obj.is_object()) {
        return std::nullopt;
    }
    for (const auto key : keys) {
        const auto it = obj.find(std::string(key));
        if (it == obj.end() || it->is_null()) {
            continue;
        }
        if (it->is_string()) {
            return it->get<std::string>();
        }
        if (it->is_number_integer() || it->is_number_unsigned()) {
            return std::to_string(it->get<long long>());
        }
        if (it->is_number_float()) {
            return std::to_string(it->get<double>());
        }
    }
    return std::nullopt;
}

std::optional<std::string> nestedStringValue(const nlohmann::json& obj,
                                             std::initializer_list<std::string_view> keys)
{
    if (auto value = stringValue(obj, keys)) {
        return value;
    }
    if (obj.is_object()) {
        if (const auto it = obj.find("properties"); it != obj.end()) {
            if (auto value = stringValue(*it, keys)) {
                return value;
            }
        }
        if (const auto it = obj.find("sample"); it != obj.end()) {
            if (auto value = stringValue(*it, keys)) {
                return value;
            }
        }
    }
    return std::nullopt;
}

std::optional<double> numberValue(const nlohmann::json& obj,
                                  std::initializer_list<std::string_view> keys)
{
    if (!obj.is_object()) {
        return std::nullopt;
    }
    for (const auto key : keys) {
        const auto it = obj.find(std::string(key));
        if (it == obj.end() || it->is_null()) {
            continue;
        }
        if (it->is_number()) {
            return it->get<double>();
        }
        if (it->is_string()) {
            try {
                return std::stod(it->get<std::string>());
            } catch (const std::exception&) {
                continue;
            }
        }
    }
    if (const auto it = obj.find("properties"); it != obj.end()) {
        return numberValue(*it, keys);
    }
    return std::nullopt;
}

std::optional<int> intValue(const nlohmann::json& obj,
                            std::initializer_list<std::string_view> keys)
{
    if (auto value = numberValue(obj, keys)) {
        return static_cast<int>(*value);
    }
    return std::nullopt;
}

nlohmann::json objectOrEmpty(const nlohmann::json& obj, const char* key)
{
    if (!obj.is_object()) {
        return nlohmann::json::object();
    }
    const auto it = obj.find(key);
    if (it != obj.end() && it->is_object()) {
        return *it;
    }
    return nlohmann::json::object();
}

bool startsWith(std::string_view value, std::string_view prefix)
{
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

std::vector<OpenDataAccessRoot> parseAccessRoots(const nlohmann::json& rootsJson)
{
    std::vector<OpenDataAccessRoot> roots;
    if (!rootsJson.is_array()) {
        return roots;
    }
    roots.reserve(rootsJson.size());
    for (const auto& rootJson : rootsJson) {
        if (!rootJson.is_object()) {
            continue;
        }
        OpenDataAccessRoot root;
        root.raw = rootJson;
        root.properties = objectOrEmpty(rootJson, "properties");
        root.type = stringValue(rootJson, {"type"}).value_or("");
        root.url = stringValue(rootJson, {"url"}).value_or("");
        root.usage = stringValue(rootJson, {"usage"}).value_or("");
        roots.push_back(std::move(root));
    }
    return roots;
}

std::vector<OpenDataOrigin> parseOrigins(const nlohmann::json& originsJson)
{
    std::vector<OpenDataOrigin> origins;
    if (!originsJson.is_array()) {
        return origins;
    }
    origins.reserve(originsJson.size());
    for (const auto& originJson : originsJson) {
        if (!originJson.is_object()) {
            continue;
        }
        OpenDataOrigin origin;
        origin.raw = originJson;
        origin.properties = objectOrEmpty(originJson, "properties");
        origin.path = stringValue(originJson, {"path", "url", "uri"}).value_or("");
        if (const auto it = originJson.find("access_roots"); it != originJson.end()) {
            origin.accessRoots = parseAccessRoots(*it);
        }
        origins.push_back(std::move(origin));
    }
    return origins;
}

void selectPreferredOrigin(OpenDataArtifact& artifact)
{
    for (const auto& origin : artifact.origins) {
        for (const auto& root : origin.accessRoots) {
            if (!iequals(root.usage, "public-read") || root.url.empty()) {
                continue;
            }
            artifact.sourcePath = joinOpenDataUrl(root.url, origin.path);
            artifact.resolvedUrl = resolveOpenDataUrl(artifact.sourcePath);
            artifact.accessUsage = root.usage;
            return;
        }
    }
}

std::vector<OpenDataArtifact> parseArtifacts(const nlohmann::json& ownerJson)
{
    const nlohmann::json* dataJson = nullptr;
    if (ownerJson.is_object()) {
        if (const auto it = ownerJson.find("data"); it != ownerJson.end() && it->is_array()) {
            dataJson = &*it;
        } else if (const auto it = ownerJson.find("artifacts"); it != ownerJson.end() && it->is_array()) {
            dataJson = &*it;
        }
    }
    if (!dataJson) {
        return {};
    }

    std::vector<OpenDataArtifact> artifacts;
    artifacts.reserve(dataJson->size());
    for (const auto& itemJson : *dataJson) {
        if (!itemJson.is_object()) {
            continue;
        }
        OpenDataArtifact artifact;
        artifact.raw = itemJson;
        artifact.properties = objectOrEmpty(itemJson, "properties");
        artifact.type = stringValue(itemJson, {"type"}).value_or("");
        if (const auto it = itemJson.find("origins"); it != itemJson.end()) {
            artifact.origins = parseOrigins(*it);
        }
        selectPreferredOrigin(artifact);
        artifacts.push_back(std::move(artifact));
    }
    return artifacts;
}

OpenDataScan parseScan(std::string id, const nlohmann::json& scanJson)
{
    OpenDataScan scan;
    scan.id = nestedStringValue(scanJson, {"id"}).value_or(std::move(id));
    scan.suffix = nestedStringValue(scanJson, {"suffix"}).value_or("");
    scan.createdAt = nestedStringValue(scanJson, {"created_at", "createdAt", "created"}).value_or("");
    scan.properties = objectOrEmpty(scanJson, "properties");
    scan.artifacts = parseArtifacts(scanJson);
    scan.raw = scanJson.is_object() ? scanJson : nlohmann::json::object();
    return scan;
}

OpenDataVolume parseVolume(std::string id, const nlohmann::json& volumeJson)
{
    OpenDataVolume volume;
    volume.id = nestedStringValue(volumeJson, {"id"}).value_or(std::move(id));
    volume.scanId = nestedStringValue(volumeJson, {"scan_id", "scanId", "scan"}).value_or("");
    volume.suffix = nestedStringValue(volumeJson, {"suffix"}).value_or("");
    volume.pixelSizeUm = numberValue(volumeJson, {"pixel_size_um", "pixelSizeUm", "resolution_um"});
    volume.energyKeV = numberValue(volumeJson, {"energy_kev", "energyKeV"});
    volume.detectorDistanceMm = numberValue(volumeJson, {"detector_distance_mm", "detectorDistanceMm"});
    volume.dataFormat = nestedStringValue(volumeJson, {"data_format", "dataFormat", "format"}).value_or("");
    volume.createdAt = nestedStringValue(volumeJson, {"created_at", "createdAt", "created"}).value_or("");
    volume.properties = objectOrEmpty(volumeJson, "properties");
    volume.artifacts = parseArtifacts(volumeJson);
    volume.raw = volumeJson.is_object() ? volumeJson : nlohmann::json::object();
    return volume;
}

OpenDataSegment parseSegment(std::string id, const nlohmann::json& segmentJson)
{
    OpenDataSegment segment;
    segment.id = nestedStringValue(segmentJson, {"id"}).value_or(id);
    segment.longId = nestedStringValue(segmentJson, {"long_id", "longId"}).value_or("");
    segment.suffix = nestedStringValue(segmentJson, {"suffix"}).value_or("");
    segment.originalVolumeId = nestedStringValue(
        segmentJson,
        {"original_volume_id", "originalVolumeId", "volume_id", "volumeId"}).value_or("");
    segment.width = intValue(segmentJson, {"width"});
    segment.height = intValue(segmentJson, {"height"});
    segment.createdAt = nestedStringValue(segmentJson, {"created_at", "createdAt", "created"}).value_or("");
    segment.properties = objectOrEmpty(segmentJson, "properties");
    segment.artifacts = parseArtifacts(segmentJson);
    segment.raw = segmentJson.is_object() ? segmentJson : nlohmann::json::object();
    return segment;
}

template <typename ParseFn>
auto parseObjectMap(const nlohmann::json& ownerJson, const char* key, ParseFn parse)
{
    using Value = decltype(parse(std::string(), nlohmann::json::object()));
    std::vector<Value> values;
    if (!ownerJson.is_object()) {
        return values;
    }
    const auto it = ownerJson.find(key);
    if (it == ownerJson.end() || !it->is_object()) {
        return values;
    }
    values.reserve(it->size());
    for (auto item = it->begin(); item != it->end(); ++item) {
        values.push_back(parse(item.key(), item.value()));
    }
    return values;
}

OpenDataSample parseSample(std::string id, const nlohmann::json& sampleJson)
{
    OpenDataSample sample;
    sample.id = nestedStringValue(sampleJson, {"id", "sample_id", "sampleId"}).value_or(std::move(id));
    sample.type = nestedStringValue(sampleJson, {"type", "sample_type", "sampleType"}).value_or("");
    sample.description = nestedStringValue(sampleJson, {"description"}).value_or("");
    sample.properties = objectOrEmpty(sampleJson, "sample");
    if (sample.properties.empty()) {
        sample.properties = objectOrEmpty(sampleJson, "properties");
    }
    if (sampleJson.is_object()) {
        if (const auto sampleIt = sampleJson.find("sample"); sampleIt != sampleJson.end()) {
            sample.artifacts = parseArtifacts(*sampleIt);
        }
    }
    if (sample.artifacts.empty()) {
        sample.artifacts = parseArtifacts(sampleJson);
    }
    sample.scans = parseObjectMap(sampleJson, "scans", parseScan);
    sample.volumes = parseObjectMap(sampleJson, "volumes", parseVolume);
    sample.segments = parseObjectMap(sampleJson, "segments", parseSegment);
    sample.raw = sampleJson.is_object() ? sampleJson : nlohmann::json::object();
    return sample;
}

std::vector<OpenDataSample> parseSamples(const nlohmann::json& metadataJson)
{
    std::vector<OpenDataSample> samples;
    if (!metadataJson.is_object()) {
        return samples;
    }
    const auto samplesIt = metadataJson.find("samples");
    if (samplesIt == metadataJson.end() || !samplesIt->is_object()) {
        return samples;
    }
    samples.reserve(samplesIt->size());
    for (auto it = samplesIt->begin(); it != samplesIt->end(); ++it) {
        if (it.value().is_object()) {
            samples.push_back(parseSample(it.key(), it.value()));
        }
    }
    return samples;
}

std::vector<OpenDataModel> parseModels(const nlohmann::json& metadataJson)
{
    std::vector<OpenDataModel> models;
    if (!metadataJson.is_object()) {
        return models;
    }
    const auto modelsIt = metadataJson.find("models");
    if (modelsIt == metadataJson.end() || !modelsIt->is_object()) {
        return models;
    }
    models.reserve(modelsIt->size());
    for (auto it = modelsIt->begin(); it != modelsIt->end(); ++it) {
        OpenDataModel model;
        model.id = it.key();
        model.raw = it.value().is_object() ? it.value() : nlohmann::json::object();
        if (auto parsedId = nestedStringValue(model.raw, {"id", "model_id", "modelId"})) {
            model.id = *parsedId;
        }
        models.push_back(std::move(model));
    }
    return models;
}

} // namespace

bool OpenDataArtifact::hasResolvedUrl() const noexcept
{
    return !resolvedUrl.empty();
}

bool OpenDataSegment::hasArtifactType(std::string_view type) const
{
    return findArtifact(artifacts, type) != nullptr;
}

bool OpenDataSegment::hasTifxyz() const
{
    return preferredTifxyzArtifact(*this) != nullptr;
}

bool OpenDataSegment::hasInkDetection() const
{
    return std::any_of(artifacts.begin(), artifacts.end(), [](const OpenDataArtifact& artifact) {
        return containsInsensitive(artifact.type, "ink");
    });
}

bool OpenDataSegment::hasLayersZarr() const
{
    return std::any_of(artifacts.begin(), artifacts.end(), [](const OpenDataArtifact& artifact) {
        const auto type = lowerCopy(artifact.type);
        return type == "layers_zarr" || type == "layers-zarr" ||
               type == "layers zarr" || (type.find("layers") != std::string::npos &&
                                          type.find("zarr") != std::string::npos);
    });
}

std::size_t OpenDataSample::scanCount() const noexcept
{
    return scans.size();
}

std::size_t OpenDataSample::volumeCount() const noexcept
{
    return volumes.size();
}

std::size_t OpenDataSample::segmentCount() const noexcept
{
    return segments.size();
}

std::size_t OpenDataSample::tifxyzSegmentCount() const
{
    return static_cast<std::size_t>(std::count_if(
        segments.begin(), segments.end(), [](const OpenDataSegment& segment) {
            return segment.hasTifxyz();
        }));
}

std::size_t OpenDataSample::inkDetectionSegmentCount() const
{
    return static_cast<std::size_t>(std::count_if(
        segments.begin(), segments.end(), [](const OpenDataSegment& segment) {
            return segment.hasInkDetection();
        }));
}

const OpenDataSample* OpenDataManifest::findSample(std::string_view id) const noexcept
{
    const auto it = std::find_if(samples.begin(), samples.end(), [&](const OpenDataSample& sample) {
        return sample.id == id;
    });
    return it == samples.end() ? nullptr : &*it;
}

const OpenDataModel* OpenDataManifest::findModel(std::string_view id) const noexcept
{
    const auto it = std::find_if(models.begin(), models.end(), [&](const OpenDataModel& model) {
        return model.id == id;
    });
    return it == models.end() ? nullptr : &*it;
}

OpenDataManifest parseOpenDataManifest(std::string_view jsonText, std::string manifestUrl)
{
    auto root = nlohmann::json::parse(jsonText);
    if (!root.is_object()) {
        throw std::runtime_error("Open-data manifest root must be a JSON object");
    }

    const nlohmann::json* metadata = &root;
    if (const auto it = root.find("metadata"); it != root.end() && it->is_object()) {
        metadata = &*it;
    }

    OpenDataManifest manifest;
    manifest.manifestUrl = std::move(manifestUrl);
    manifest.samples = parseSamples(*metadata);
    manifest.models = parseModels(*metadata);
    manifest.raw = std::move(root);
    return manifest;
}

OpenDataManifest loadOpenDataManifestFile(const std::filesystem::path& manifestPath,
                                          std::string manifestUrl)
{
    std::ifstream input(manifestPath);
    if (!input) {
        throw std::runtime_error("Failed to open open-data manifest: " + manifestPath.string());
    }
    nlohmann::json root;
    input >> root;
    if (manifestUrl.empty()) {
        manifestUrl = manifestPath.string();
    }
    return parseOpenDataManifest(root.dump(), std::move(manifestUrl));
}

OpenDataManifest fetchOpenDataManifest(std::string manifestUrl)
{
    const auto body = vc::httpGetString(manifestUrl);
    if (body.empty()) {
        throw std::runtime_error("Failed to fetch open-data manifest: " + manifestUrl);
    }
    return parseOpenDataManifest(body, std::move(manifestUrl));
}

std::string resolveOpenDataUrl(std::string url)
{
    constexpr std::string_view openDataS3 = "s3://vesuvius-challenge-open-data/";
    constexpr std::string_view challengeS3 = "s3://vesuvius-challenge/";
    if (startsWith(url, openDataS3)) {
        return "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/" +
               url.substr(openDataS3.size());
    }
    if (startsWith(url, challengeS3)) {
        return "https://data.aws.ash2txt.org/samples/" + url.substr(challengeS3.size());
    }
    if (startsWith(url, "s3://") || startsWith(url, "s3+")) {
        return vc::resolveRemoteUrl(url).httpsUrl;
    }
    return url;
}

std::string joinOpenDataUrl(std::string root, std::string path)
{
    if (path.empty()) {
        return root;
    }
    if (startsWith(path, "s3://") || startsWith(path, "s3+") ||
        startsWith(path, "http://") || startsWith(path, "https://")) {
        return path;
    }
    while (!root.empty() && root.back() == '/') {
        root.pop_back();
    }
    while (!path.empty() && path.front() == '/') {
        path.erase(path.begin());
    }
    return root + "/" + path;
}

const OpenDataArtifact* findArtifact(const std::vector<OpenDataArtifact>& artifacts,
                                     std::string_view type) noexcept
{
    const auto it = std::find_if(artifacts.begin(), artifacts.end(), [&](const OpenDataArtifact& artifact) {
        return iequals(artifact.type, type);
    });
    return it == artifacts.end() ? nullptr : &*it;
}

const OpenDataArtifact* preferredTifxyzArtifact(const OpenDataSegment& segment) noexcept
{
    for (const auto* type : {"tifxyz-flattened", "tifxyz-transformed", "tifxyz", "tifxyz-normalized"}) {
        if (const auto* artifact = findArtifact(segment.artifacts, type)) {
            return artifact;
        }
    }
    const auto flattened = std::find_if(
        segment.artifacts.begin(), segment.artifacts.end(), [](const OpenDataArtifact& artifact) {
            const auto type = lowerCopy(artifact.type);
            return type.find("tifxyz") != std::string::npos &&
                   type.find("flattened") != std::string::npos;
        });
    if (flattened != segment.artifacts.end()) {
        return &*flattened;
    }
    const auto transformed = std::find_if(
        segment.artifacts.begin(), segment.artifacts.end(), [](const OpenDataArtifact& artifact) {
            const auto type = lowerCopy(artifact.type);
            return type.find("tifxyz") != std::string::npos &&
                   type.find("transformed") != std::string::npos;
        });
    if (transformed != segment.artifacts.end()) {
        return &*transformed;
    }
    const auto anyTifxyz = std::find_if(
        segment.artifacts.begin(), segment.artifacts.end(), [](const OpenDataArtifact& artifact) {
            return containsInsensitive(artifact.type, "tifxyz");
        });
    return anyTifxyz == segment.artifacts.end() ? nullptr : &*anyTifxyz;
}

const OpenDataArtifact* preferredVolumeArtifact(const OpenDataVolume& volume) noexcept
{
    for (const auto* type : {"zarr", "zarr3", "volume_zarr", "omezarr", "ome-zarr"}) {
        if (const auto* artifact = findArtifact(volume.artifacts, type)) {
            return artifact;
        }
    }
    const auto it = std::find_if(
        volume.artifacts.begin(), volume.artifacts.end(), [](const OpenDataArtifact& artifact) {
            return artifact.hasResolvedUrl() && containsInsensitive(artifact.type, "zarr");
        });
    if (it != volume.artifacts.end()) {
        return &*it;
    }
    return volume.artifacts.empty() ? nullptr : &volume.artifacts.front();
}

const OpenDataArtifact* preferredPhotoArtifact(const OpenDataSample& sample) noexcept
{
    for (const auto* type : {"photo", "image", "thumbnail"}) {
        if (const auto* artifact = findArtifact(sample.artifacts, type)) {
            return artifact;
        }
    }
    const auto it = std::find_if(
        sample.artifacts.begin(), sample.artifacts.end(), [](const OpenDataArtifact& artifact) {
            const auto type = lowerCopy(artifact.type);
            return type.find("photo") != std::string::npos ||
                   type.find("image") != std::string::npos ||
                   type.find("thumbnail") != std::string::npos;
        });
    return it == sample.artifacts.end() ? nullptr : &*it;
}

} // namespace vc3d::opendata
